# ============= eval_rollout.py ==================
import os
import json
import enum
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List
from utils.losses import stochastic_loss_fn
from config.config import Navier_Stokes_Stochastic_Config as Config
from dataset.navier_stokes_flow_stochastic import load_synth, SynthDataset
from models.node_dmd import Stochastic_NODE_DMD
from utils.utils import ensure_dir, eval_uncertainty_metrics, find_subset_indices, reparameterize_full
from utils.plots import plot_reconstruction
import imageio
import xarray as xr
import random
class FeedMode(enum.Enum):
    AUTOREG = "autoreg"
    TEACHER = "teacher_forcing"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def _prepare_model(cfg: Config, model_name="best_model.pt") -> Stochastic_NODE_DMD:
    device = torch.device(cfg.device)
    model = Stochastic_NODE_DMD(
        cfg.r, cfg.hidden_dim, cfg.ode_steps, cfg.process_noise, cfg.cov_eps, cfg.dt, mode_frequency=cfg.mode_frequency, phi_frequency=cfg.phi_frequency
    ).to(device)
    ckpt = torch.load(os.path.join(cfg.save_dir, model_name), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    loss = ckpt["best_loss"]
    epoch = ckpt["epoch"]
    print(f"best loss of {loss} saved at epoch {epoch}")
    return model


def _prepare_data(cfg: Config):
    
    return load_synth(
        cfg.data_path, sample_ratio=cfg.sample_ratio, normalize_t=cfg.normalize_t, device=cfg.device
    )

def _compute_vmin_vmax(y_true_full_list: List[torch.Tensor]) -> Tuple[float, float]:
    vmin = min(torch.min(y[:, 0]).item() for y in y_true_full_list)
    vmax = max(torch.max(y[:, 0]).item() for y in y_true_full_list)
    return vmin, vmax


def _summarize_and_dump(calib_all: List[dict], mse_full_all: List[float], out_dir: str, mode: FeedMode):
    def _avg(key): 
        vals = [d[key] for d in calib_all if key in d]
        return float(np.mean(vals)) if len(vals) > 0 else None

    summary = {
        "avg_mse_full": float(np.mean(mse_full_all)) if mse_full_all else None,
        "coverage@68":  _avg("coverage@68"),
        "coverage@95":  _avg("coverage@95"),
        "z_mean":       _avg("z_mean"),
        "z_std":        _avg("z_std"),
        "z_abs_mean":   _avg("z_abs_mean"),
        "NLL_mean":     _avg("NLL_mean"),
        "mean_pred_var":      _avg("mean_pred_var"),
        "mean_residual_sq":   _avg("mean_residual_sq"),
        "z_noise_mean":       _avg("z_noise_mean"),
        "z_noise_std":        _avg("z_noise_std"),
        "coverage_noise@68":  _avg("coverage_noise@68"),
        "coverage_noise@95":  _avg("coverage_noise@95"),
        "mean_emp_noise_var": _avg("mean_emp_noise_var"),
    }

    with open(os.path.join(out_dir, "uncertainty_metrics_per_t.json"), "w") as f:
        json.dump(calib_all, f, indent=2)
    with open(os.path.join(out_dir, "uncertainty_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    title = "Autoregressive" if mode == FeedMode.AUTOREG else "Teacher Forcing"
    print(f"=== {title} Uncertainty Summary ===")
    for k, v in summary.items():
        print(f"{k:>24}: {v:.6f}" if isinstance(v, float) else f"{k:>24}: {v}")


@torch.no_grad()
def run_eval(cfg: Config, mode: str = "teacher_forcing", model_name: str = "best_model.pt"):
    """
    통합 평가 루틴.
    - mode = "teacher_forcing": y_prev(ground-truth at t_{i-1})를 입력으로 사용
    - mode = "autoreg":         모델 예측 y_pred를 연결해 입력으로 사용
    나머지 로직(로드/플롯/캘리브레이션/저장)은 동일.
    """
    # --- 준비
    feed_mode = FeedMode(mode)
    (
        t_list,
        coords_list,
        y_list,
        y_true_full_list,
        coords_full,
        *_,
    ) = _prepare_data(cfg)
    

    model = _prepare_model(cfg, model_name=model_name)
    # import time
    # time.sleep(1e6)
    vmin, vmax = _compute_vmin_vmax(y_true_full_list)

    # out_dir = os.path.join(cfg.save_dir, feed_mode.value)
    # ensure_dir(out_dir)

    coords_idx = find_subset_indices(coords_full, coords_list[0])

    mse_full_all = []
    calib_all    = []
    out_dir = os.path.join(cfg.save_dir, f"{feed_mode.value}_reconstruction")
    ensure_dir(out_dir)
    # --- 초기 y_in 설정 (AUTOREG 전용)
    y_pred_chain = y_true_full_list[0] if feed_mode == FeedMode.AUTOREG else None
    side = 64
    # side =32
    y_pred_list = [y_true_full_list[0][:, 0].reshape(side,side)]
    frames = []
    frame=plot_reconstruction(coords_full, 0,  y_true_full_list[0],  y_true_full_list[0], 0, out_dir, vmin, vmax)
    frames.append(frame)

    # --- 메인 루프
    
    for i in range(1, cfg.eval_data_len):
        
        coords = coords_full
        y_true = y_true_full_list[i]
        t_prev = float(t_list[i - 1])
        t_next = float(t_list[i])

        if feed_mode == FeedMode.TEACHER:
            y_in = y_true_full_list[i - 1]        # ground-truth teacher forcing
        else:
            # autoreg: 첫 스텝은 초기 truth에서 시작, 이후는 직전 예측을 연결
            y_in = y_pred_chain

        mu_u, logvar_u, cov_u, mu_phi, logvar_phi,lam,W = model(coords, y_in, t_prev, t_next)
        mu_phi_hat, logvar_phi_hat, _ = model.phi_net(coords, mu_u)
        loss, parts = stochastic_loss_fn(
                mu_u, logvar_u, y_true, mu_phi, logvar_phi, mu_phi_hat, logvar_phi_hat, lam, W,
                recon_weight=cfg.recon_weight,
                l1_weight=cfg.l1_weight, 
                mode_sparsity_weight=cfg.mode_sparsity_weight,
                kl_phi_weight=cfg.kl_phi_weight,
                cons_weight= cfg.cons_weight
            )
        u_pred = reparameterize_full(mu_u, cov_u)
            
        if feed_mode == FeedMode.AUTOREG:
            y_pred_chain = u_pred
        y_pred_list.append(u_pred[:, 0].reshape(side,side))
        # --- MSE 및 플롯
        mse = F.mse_loss(u_pred, y_true).item()
        mse_full_all.append(mse)
        # plot_reconstruction(coords, t_next, y_true, mu_u, mse, out_dir, vmin, vmax)
        frame = plot_reconstruction(coords, i, y_true, u_pred, mse, out_dir, vmin, vmax)
        frames.append(frame)
        # --- (B) 관측 서브셋(노이즈 포함)에서 불확실성 캘리브레이션
        y_obs      = y_list[i]         # noisy measurement at t_next (subset coords_list[0])
        # y_clean    = y_true_list[i]    # clean target at subset
        mu_sub     = mu_u[coords_idx]
        logvar_sub = logvar_u[coords_idx]

        metrics = eval_uncertainty_metrics(
            y_obs, mu_sub, logvar_sub
        )
        metrics["time_index"] = i
        metrics["mse_full"]   = mse
        calib_all.append(metrics)
        if i == cfg.data_len:
            _summarize_and_dump(calib_all, mse_full_all, out_dir, feed_mode)
            mse_full_all = []
            calib_all    = []
            metrics = {}
            imageio.mimsave(f"{out_dir}/reconstruction.gif", frames, fps=10)  # fps 조정 가능


        if i > cfg.data_len:
            out_dir = os.path.join(cfg.save_dir, f"{feed_mode.value}_exploitation")
            ensure_dir(out_dir)
    # --- 저장 및 요약 출력
    xvals = np.linspace(0, 1.0, side)
    yvals = np.linspace(0, 1.0, side)
    ds_pred = xr.Dataset(
        data_vars=dict(
            vorticity=(("time", "x", "y"), [y.cpu().numpy() for y in y_pred_list]),
        ),
        coords= {"time": t_list[:cfg.data_len], "x": xvals, "y": yvals},
    )
    ds_pred.to_netcdf(f"{out_dir}/predicted_vorticity.nc")

    imageio.mimsave(f"{out_dir}/exploitation.gif", frames, fps=10)  # fps 조정 가능   
    _summarize_and_dump(calib_all, mse_full_all, out_dir, feed_mode)

@torch.no_grad()
def run_multiple_eval(cfg: Config, mode: str = "teacher_forcing", num_iter: int = 10, model_name: str = "best_model.pt"):
    """
    통합 평가 루틴.
    - mode = "teacher_forcing": y_prev(ground-truth at t_{i-1})를 입력으로 사용
    - mode = "autoreg":         모델 예측 y_pred를 연결해 입력으로 사용
    나머지 로직(로드/플롯/캘리브레이션/저장)은 동일.
    """
    # --- 준비
    set_seed(123)
    # set_seed(1)
    feed_mode = FeedMode(mode)
    (
        t_list,
        coords_list,
        y_list,
        y_true_full_list,
        coords_full,
        *_,
    ) = _prepare_data(cfg)
    

    model = _prepare_model(cfg, model_name=model_name)

    out_dir = os.path.join(cfg.save_dir, f"{feed_mode.value}_reconstruction")
    ensure_dir(out_dir)
    # --- 초기 y_in 설정 (AUTOREG 전용)
    # y_pred_chain = y_true_full_list[0] if feed_mode == FeedMode.AUTOREG else None
    side = 64
    # side =32
    
    xvals = np.linspace(0, 1.0, side)
    yvals = np.linspace(0, 1.0, side)
    # --- 메인 루프
    for niter in range(num_iter):
        y_pred_list = [y_true_full_list[0][:, 0].reshape(side,side)]
        y_pred_chain = y_true_full_list[0] if feed_mode == FeedMode.AUTOREG else None
    
        ensure_dir(out_dir)
        for i in range(1, cfg.eval_data_len):
            coords = coords_full
            y_true = y_true_full_list[i]
            t_prev = float(t_list[i - 1])
            t_next = float(t_list[i])

            if feed_mode == FeedMode.TEACHER:
                y_in = y_true_full_list[i - 1]        # ground-truth teacher forcing
            else:
                # autoreg: 첫 스텝은 초기 truth에서 시작, 이후는 직전 예측을 연결
                y_in = y_pred_chain

            mu_u, logvar_u, cov_u, mu_phi, logvar_phi,lam,W = model(coords, y_in, t_prev, t_next)
            u_pred = reparameterize_full(mu_u, cov_u)
            if feed_mode == FeedMode.AUTOREG:
                y_pred_chain = u_pred
            y_pred_list.append(u_pred[:, 0].reshape(side,side))
            
        
        ds_pred = xr.Dataset(
            data_vars=dict(
                vorticity=(("time", "x", "y"), [y.cpu().numpy() for y in y_pred_list]),
            ),
            coords= {"time": t_list[:cfg.data_len], "x": xvals, "y": yvals},
        )
        ds_pred.to_netcdf(f"{out_dir}/predicted_vorticity_{niter}.nc")

  
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate SNode DMD with config from directory.")
    parser.add_argument("--config_dir", type=str, required=True, help="Directory containing Config.txt")
    parser.add_argument("--ckpt_name", type=str, default="best_model.pt", help="Directory containing Config.txt")
    
    args = parser.parse_args()

    # Load config from txt file
    config_path = os.path.join(args.config_dir, "Config.txt")
    import ast
    config_dict = {}
    with open(config_path, "r") as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                k = k.strip()
                v = v.strip()
                # Try to parse tuple/list, int, float, bool, or leave as string
                if v.lower() == "true":
                    v = True
                elif v.lower() == "false":
                    v = False
                else:
                    try:
                        # Try tuple/list parsing
                        if ("," in v) or (v.startswith("(") and v.endswith(")")):
                            v = ast.literal_eval(v)
                        else:
                            v = int(v)
                    except (ValueError, SyntaxError):
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                config_dict[k] = v

    # Create config object
    cfg = Config()
    for k, v in config_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # run_eval(cfg, mode="teacher_forcing", model_name=args.ckpt_name)
    # run_eval(cfg, mode="autoreg", model_name=args.ckpt_name)

    # run_multiple_eval(cfg, mode="teacher_forcing", num_iter=10, model_name=args.ckpt_name)
    run_multiple_eval(cfg, mode="autoreg", num_iter=10, model_name=args.ckpt_name)