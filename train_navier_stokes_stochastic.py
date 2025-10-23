import os
import torch
import torch.optim as optim
from tqdm import trange
from config.config import Navier_Stokes_Stochastic_Config as Config
from dataset.navier_stokes_flow_stochastic import load_synth, SynthDataset
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

def _prepare_model(cfg: Config, pretrained_path: str, model_name="best_model.pt") -> Stochastic_NODE_DMD:
    
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


def run_train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    t_list, coords_list, y_list, y_full, coords_full, nR, T = load_synth(
        cfg.data_path, sample_ratio=cfg.sample_ratio, normalize_t=cfg.normalize_t, device=device, data_len=cfg.data_len
    )
    dataset = SynthDataset(t_list, coords_list, y_list, n_realizations=nR, steps_per_realization=T)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )   
    model = Stochastic_NODE_DMD(
        r=cfg.r,
        hidden_dim=cfg.hidden_dim,
        ode_steps=cfg.ode_steps,
        process_noise=cfg.process_noise,
        cov_eps=cfg.cov_eps,
        dt=cfg.dt,
        mode_frequency=cfg.mode_frequency,
        phi_frequency=cfg.phi_frequency
    ).to(device)
    # pretrained_path = "results/stochastic/run23"
    # model = _prepare_model(cfg, pretrained_path)

    # Save config to output directory
    ensure_dir(cfg.save_dir)
    config_path = os.path.join(cfg.save_dir, "Config.txt")

    with open(config_path, "w") as f:
        for k in dir(cfg):
            if not k.startswith('_') and not callable(getattr(cfg, k)):
                v = getattr(cfg, k)
                f.write(f"{k}: {v}\n")

    initial_lr = cfg.lr
    final_lr = initial_lr * 0.02
    opt = optim.Adam(model.parameters(), lr=initial_lr)
    decay_rate = (final_lr / initial_lr) ** (1 / cfg.num_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda epoch: decay_rate ** epoch
    )
    best = float("inf")
    
    avg_loss_list = []
    for epoch in trange(cfg.num_epochs, desc="Training"):
        total_loss = 0.0
        num_batches = 0
        u_pred = None
        if cfg.train_mode == "teacher_forcing":
            teacher_prob = 1
        elif cfg.train_mode == "autoreg":
            teacher_prob = 0
        elif cfg.train_mode == "evolve":
            teacher_prob = min(1, 1 - (2*epoch / cfg.num_epochs)) # min(1, 1.4 - (epoch * 1.5 / cfg.num_epochs)) #run17
        for batch in dataloader:
            t_prev, t_next, coords, y_next, y_prev = [x.to(device) for x in batch]
            print(f"t_prev : {t_prev}, t_next:{t_next}")
            opt.zero_grad()
            
            if random.random() < teacher_prob:
                mu_u, logvar_u, cov_u, mu_phi, logvar_phi, lam, W = model(coords, y_prev, t_prev, t_next)
                u_pred = reparameterize_full(mu_u.detach(), cov_u.detach())
                with torch.no_grad():  # Or enable_grad if backprop through it
                    mu_phi_hat, logvar_phi_hat, _ = model.phi_net(coords, y_next)
            else:
                if u_pred is None:
                    u_pred = y_prev
                mu_u, logvar_u, cov_u, mu_phi, logvar_phi, lam, W = model(coords, u_pred, t_prev, t_next)
                u_pred = reparameterize_full(mu_u.detach(), cov_u.detach())
                with torch.no_grad():  # Or enable_grad if backprop through it
                    mu_phi_hat, logvar_phi_hat, _ = model.phi_net(coords, u_pred)
            
            loss, parts = stochastic_loss_fn(
                mu_u, logvar_u, y_next, mu_phi, logvar_phi, mu_phi_hat, logvar_phi_hat, lam, W,
                recon_weight=cfg.recon_weight,
                l1_weight=cfg.l1_weight, 
                mode_sparsity_weight=cfg.mode_sparsity_weight,
                kl_phi_weight=cfg.kl_phi_weight,
                cons_weight= cfg.cons_weight * min((epoch / cfg.num_epochs), 1) 
            )
            print(parts)
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            # opt.step()
            
            total_loss += loss.item()
            num_batches += 1

        avg = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_loss_list.append(avg)
        scheduler.step()
        if epoch % cfg.print_every == 0:
            print(f"Epoch {epoch:04d} | avg_loss={avg:.6f} | lr={scheduler.get_last_lr()[0]:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_loss': best,
                'loss_list': avg_loss_list
            }, os.path.join(cfg.save_dir, f'model_{epoch}.pt'))
            plot_loss(avg_loss_list, cfg.save_dir)
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
    run_train(Config())
