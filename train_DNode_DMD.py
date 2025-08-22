import os
import torch
import torch.optim as optim
from tqdm import trange
from config.config import Deterministic_Node_DMD_Config
from dataset.generate_synth_dataset import load_synth
from models.node_dmd import NODE_DMD
from utils.losses import loss_fn
from utils.utils import ensure_dir
import random
import numpy as np

def set_seed(seed: int):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_train(cfg: Deterministic_Node_DMD_Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    t_list, coords_list, y_list, _, _, _, _, _ = load_synth(device)

    model = NODE_DMD(
        r=cfg.r,
        hidden_dim=cfg.hidden_dim,
        ode_steps=cfg.ode_steps,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    best = float("inf")
    ensure_dir(cfg.save_dir)

    for epoch in trange(cfg.num_epochs, desc="Training"):
        total = 0.0
        u_pred = y_list[0]
        for idx in range(1, len(t_list)):
            coords = coords_list[idx]
            y_prev = y_list[idx - 1]
            y_next = y_list[idx]
            t_prev = float(t_list[idx - 1])
            t_next = float(t_list[idx])

            opt.zero_grad()
            u_pred, mu, logvar, lambda_param  = model(coords, u_pred.detach(), t_prev, t_next)
            loss = loss_fn(u_pred, y_next, mu, logvar, lambda_param,l1_weight=cfg.l1_weight)
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / (len(t_list) - 1)
        if epoch % cfg.print_every == 0:
            print(f"Epoch {epoch:04d} | avg_loss={avg:.6f}")
        if avg < best:
            best = avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_loss': best,
            }, os.path.join(cfg.save_dir, 'best_model.pt'))

    torch.save({
        'epoch': cfg.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'best_loss': best,
    }, os.path.join(cfg.save_dir, 'final_model.pt'))

    print(f"Training complete. Final model saved at {os.path.join(cfg.save_dir, 'final_model.pt')}")
    print(f"Best model saved at {os.path.join(cfg.save_dir, 'best_model.pt')} with loss {best:.6f}")


if __name__ == "__main__":
    run_train(Deterministic_Node_DMD_Config())
