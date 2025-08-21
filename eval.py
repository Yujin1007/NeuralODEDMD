# ============= eval_teacher_forcing.py ==================
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config.config import TrainConfig
from dataset.generate_synth_dataset import load_synth
from models.node_dmd import NODE_DMD
from utils.utils import ensure_dir


def run_eval_autoreg(cfg: TrainConfig):
    device = torch.device(cfg.device)
    (
        t_list,
        coords_list,
        y_list,
        y_true_list,
        y_true_full_list,
        coords_full_t,
        *_,
    ) = load_synth(device)

    model = NODE_DMD(cfg.r, cfg.hidden_dim, cfg.ode_steps, cfg.process_noise, cfg.cov_eps).to(device)
    ckpt = torch.load(os.path.join(cfg.save_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    out_dir = os.path.join(cfg.save_dir, 'autoreg')
    ensure_dir(out_dir)

    with torch.no_grad():
        y_pred = y_true_full_list[0]
        for i in range(1, len(t_list)):
            coords = coords_full_t
            y_true = y_true_full_list[i]
            t_prev = float(t_list[i - 1])
            t_next = float(t_list[i])
            mu_u, logvar_u, *_ = model(coords, y_pred, t_prev, t_next)
            y_pred = mu_u
            mse = F.mse_loss(mu_u, y_true).item()

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), c=y_true[:, 0].cpu().numpy(), cmap='viridis')
            plt.title(f'True Real Part at t={t_list[i]}')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.scatter(coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), c=mu_u[:, 0].cpu().numpy(), cmap='viridis')
            plt.title(f'Pred Real Part t={t_list[i]}, MSE: {mse:.3f}')
            plt.colorbar()

            plt.savefig(os.path.join(out_dir, f'prediction_t{i}.png'), dpi=150, bbox_inches='tight')
            plt.close()
def run_eval_teacher_forcing(cfg: TrainConfig):
    device = torch.device(cfg.device)
    (
        t_list,
        coords_list,
        y_list,
        y_true_list,
        y_true_full_list,
        coords_full_t,
        *_,
    ) = load_synth(device)

    model = NODE_DMD(cfg.r, cfg.hidden_dim, cfg.ode_steps, cfg.process_noise, cfg.cov_eps).to(device)
    ckpt = torch.load(os.path.join(cfg.save_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    out_dir = os.path.join(cfg.save_dir, 'teacher_forcing')
    ensure_dir(out_dir)

    with torch.no_grad():
        for i in range(1, len(t_list)):
            coords = coords_full_t
            y_true = y_true_full_list[i]
            y_prev = y_true_full_list[i - 1]
            t_prev = float(t_list[i - 1])
            t_next = float(t_list[i])
            mu_u, logvar_u, *_ = model(coords, y_prev, t_prev, t_next)

            mse = F.mse_loss(mu_u, y_true).item()

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), c=y_true[:, 0].cpu().numpy(), cmap='viridis')
            plt.title(f'True Real Part at t={t_list[i]}')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.scatter(coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), c=mu_u[:, 0].cpu().numpy(), cmap='viridis')
            plt.title(f'Pred Real Part t={t_list[i]}, MSE: {mse:.3f}')
            plt.colorbar()

            plt.savefig(os.path.join(out_dir, f'prediction_t{i}.png'), dpi=150, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    run_eval_teacher_forcing(TrainConfig())
    run_eval_autoreg(TrainConfig())
