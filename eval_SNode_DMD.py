# ============= eval_teacher_forcing.py ==================
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config.config import Stochastic_Node_DMD_Config
from dataset.generate_synth_dataset import load_synth
from models.node_dmd import Stochastic_NODE_DMD
from utils.utils import ensure_dir
from utils.plots import plot_reconstruction

def run_eval_autoreg(cfg: Stochastic_Node_DMD_Config):
    device = torch.device(cfg.device)
    (
        t_list,
        coords_list,
        y_list,
        y_true_list,
        y_true_full_list,
        coords_full_t,
        *_,
    ) = load_synth(device, T=cfg.eval_data_len)

    model = Stochastic_NODE_DMD(cfg.r, cfg.hidden_dim, cfg.ode_steps, cfg.process_noise, cfg.cov_eps).to(device)
    ckpt = torch.load(os.path.join(cfg.save_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    out_dir = os.path.join(cfg.save_dir, 'autoreg')
    ensure_dir(out_dir)
    t_exploit = cfg.data_len 
    t_prev_exploit = cfg.data_len - 1
    y_pred_exploit = None
    vmin = min(torch.min(y[:,0]).item() for y in y_true_full_list)
    vmax = max(torch.max(y[:,0]).item() for y in y_true_full_list)
    with torch.no_grad():
        y_pred = y_true_full_list[0]
        for i in range(1, len(t_list)):
            coords = coords_full_t
            y_true = y_true_full_list[i]
            t_next = float(t_list[i])

            t_prev = float(t_list[i - 1])
            mu_u, logvar_u, *_ = model(coords, y_pred, t_prev, t_next)
            y_pred = mu_u
            # if t_next < t_exploit:
            #     t_prev = float(t_list[i - 1])
            #     mu_u, logvar_u, *_ = model(coords, y_pred, t_prev, t_next)
            #     y_pred = mu_u
            #     if t_next == t_prev_exploit:
            #         y_pred_exploit = y_pred
            # else:
            #     mu_u, logvar_u, *_ = model(coords, y_pred_exploit, t_prev_exploit, t_next)
            
            mse = F.mse_loss(mu_u, y_true).item()
            plot_reconstruction(coords, t_next, y_true, mu_u, mse, out_dir, vmin, vmax)
            
def run_eval_teacher_forcing(cfg: Stochastic_Node_DMD_Config):
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

    model = Stochastic_NODE_DMD(cfg.r, cfg.hidden_dim, cfg.ode_steps, cfg.process_noise, cfg.cov_eps).to(device)
    ckpt = torch.load(os.path.join(cfg.save_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    vmin = min(torch.min(y[:,0]).item() for y in y_true_full_list)
    vmax = max(torch.max(y[:,0]).item() for y in y_true_full_list)
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

            plot_reconstruction(coords, t_next, y_true, mu_u, mse, out_dir, vmin, vmax)

if __name__ == "__main__":
    # run_eval_teacher_forcing(Stochastic_Node_DMD_Config())
    run_eval_autoreg(Stochastic_Node_DMD_Config())
