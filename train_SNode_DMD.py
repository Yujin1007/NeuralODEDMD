import os
import torch
import torch.optim as optim
from tqdm import trange
from config.config import Stochastic_Node_DMD_Config
from dataset.generate_synth_dataset import load_synth
from models.node_dmd import Stochastic_NODE_DMD
from utils.losses import stochastic_loss_fn
from utils.utils import ensure_dir, reparameterize_full
import random
import numpy as np

def set_seed(seed: int):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_train(cfg: Stochastic_Node_DMD_Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    t_list, coords_list, y_list, *_ = load_synth(device)

    model = Stochastic_NODE_DMD(
        r=cfg.r,
        hidden_dim=cfg.hidden_dim,
        ode_steps=cfg.ode_steps,
        process_noise=cfg.process_noise,
        cov_eps=cfg.cov_eps,
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
            # mu_u, logvar_u, mu_phi, logvar_phi, lam, W = model(coords, y_prev, t_prev, t_next)
            mu_u, logvar_u, cov_u, mu_phi, logvar_phi, lam, W = model(coords, u_pred, t_prev, t_next)
            # u_pred = mu_u.detach() # run1
            u_pred = reparameterize_full(mu_u.detach(), cov_u.detach())
            loss, parts = stochastic_loss_fn(
                mu_u, logvar_u, y_next, mu_phi, logvar_phi, lam, W,
                l1_weight=cfg.l1_weight,
                mode_sparsity_weight=cfg.mode_sparsity_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
    run_train(Stochastic_Node_DMD_Config())
