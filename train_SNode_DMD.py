import os
import torch
import torch.optim as optim
from tqdm import trange
from config.config import Stochastic_Node_DMD_Config
from dataset.generate_synth_dataset import load_synth, SynthDataset
from models.node_dmd import Stochastic_NODE_DMD
from utils.losses import stochastic_loss_fn
from utils.utils import ensure_dir, reparameterize_full
import random
import numpy as np
from torch.utils.data import DataLoader
from utils.plots import plot_loss

def set_seed(seed: int):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _prepare_model(cfg: Stochastic_Node_DMD_Config, pretrained_path: str, model_name="best_model.pt") -> Stochastic_NODE_DMD:
    device = torch.device(cfg.device)
    model = Stochastic_NODE_DMD(
        cfg.r, cfg.hidden_dim, cfg.ode_steps, cfg.process_noise, cfg.cov_eps
    ).to(device)
    ckpt = torch.load(os.path.join(pretrained_path, model_name), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    loss = ckpt["best_loss"]
    epoch = ckpt["epoch"]
    print(f"best loss of {loss} saved at epoch {epoch}")
    return model


def run_train(cfg: Stochastic_Node_DMD_Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    t_list, coords_list, y_list, *_ = load_synth(device, T=cfg.data_len)
    dataset = SynthDataset(t_list, coords_list, y_list)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,  # Add  batch_size to your config
        shuffle=False,  # Set to True if you want to shuffle time steps
        num_workers=0,  # Set to >0 for parallel loading, but 0 is fine for small datasets
        drop_last=True  # Keep the last incomplete batch
    )   
    model = Stochastic_NODE_DMD(
        r=cfg.r,
        hidden_dim=cfg.hidden_dim,
        ode_steps=cfg.ode_steps,
        process_noise=cfg.process_noise,
        cov_eps=cfg.cov_eps,
    ).to(device)
    # pretrained_path = "results/stochastic/run11"
    # model = _prepare_model(cfg, pretrained_path)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    best = float("inf")
    ensure_dir(cfg.save_dir)
    
    avg_loss_list = []
    for epoch in trange(cfg.num_epochs, desc="Training"):
        total_loss = 0.0
        num_batches = 0
        # u_pred = y_list[0].unsqueeze(0).repeat(cfg.batch_size, *([1] * y_list[0].dim())).to(device)
        t_prev = torch.tensor(t_list[0], dtype=torch.float32, device=device).unsqueeze(0).repeat(cfg.batch_size, )
        u_pred = y_list[0]
        t_prev = t_prev.item()
        for batch in dataloader:
            t_next, coords, y_next, y_prev = [x.to(device) for x in batch]
            # print(f"shape t_next {t_next[0].shape}, coords {coords[0].shape}, y_next {y_next[0].shape}")
            t_next = t_next.item()          # converts 0-dim tensor or [1] tensor to a Python float
            coords   = coords.squeeze(0)       # [1,102,2] -> [102,2]
            y_next   = y_next.squeeze(0)       # [1,102,2] -> [102,2]
            y_prev   = y_prev.squeeze(0)       # [1,102,2] -> [102,2]

            opt.zero_grad()
            # mu_u, logvar_u, cov_u, mu_phi, logvar_phi, lam, W = model(coords, u_pred, t_prev, t_next)
            mu_u, logvar_u, cov_u, mu_phi, logvar_phi, lam, W = model(coords, y_prev, t_prev, t_next)
            # Observed phi_t from y_next (consistency)
            with torch.no_grad():  # Or enable_grad if backprop through it
                mu_phi_hat, logvar_phi_hat, _ = model.phi_net(coords, y_next)  # lambda는 재사용 가능 but ignore

            u_pred = reparameterize_full(mu_u.detach(), cov_u.detach())
            t_prev = t_next
            loss, parts = stochastic_loss_fn(
                mu_u, logvar_u, y_next, mu_phi, logvar_phi, mu_phi_hat, logvar_phi_hat, lam, W,
                l1_weight=cfg.l1_weight, 
                mode_sparsity_weight=cfg.mode_sparsity_weight,
                kl_phi_weight=cfg.kl_phi_weight,
                cons_weight= cfg.cons_weight * (epoch / cfg.num_epochs)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            
            total_loss += loss.item()
            num_batches += 1

        avg = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_loss_list.append(avg)
        if epoch % cfg.print_every == 0:
            print(f"Epoch {epoch:04d} | avg_loss={avg:.6f}")
            # torchㄴsave_dir, f'ckpt_model_{epoch}.pt'))
        if avg < best:
            best = avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_loss': best,
                'loss_list': avg_loss_list
            }, os.path.join(cfg.save_dir, 'best_model.pt'))
            plot_loss(avg_loss_list, cfg.save_dir)
    torch.save({
        'epoch': cfg.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'best_loss': best,
        'loss_list': avg_loss_list
    }, os.path.join(cfg.save_dir, 'final_model.pt'))
    plot_loss(avg_loss_list, cfg.save_dir, "final_loss.png")

    print(f"Training complete. Final model saved at {os.path.join(cfg.save_dir, 'final_model.pt')}")
    print(f"Best model saved at {os.path.join(cfg.save_dir, 'best_model.pt')} with loss {best:.6f}")


if __name__ == "__main__":
    run_train(Stochastic_Node_DMD_Config())
