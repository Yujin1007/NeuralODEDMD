# ============= eval_rollout.py ==================
import os
import json
import enum
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List

from config.config import Stochastic_Node_DMD_Config
from dataset.generate_synth_dataset import load_synth
from models.node_dmd import Stochastic_NODE_DMD
from utils.utils import ensure_dir, eval_uncertainty_metrics, find_subset_indices
from utils.plots import plot_reconstruction


class FeedMode(enum.Enum):
    AUTOREG = "autoreg"
    TEACHER = "teacher_forcing"


def _prepare_model(cfg: Stochastic_Node_DMD_Config) -> Stochastic_NODE_DMD:
    device = torch.device(cfg.device)
    model = Stochastic_NODE_DMD(
        cfg.r, cfg.hidden_dim, cfg.ode_steps, cfg.process_noise, cfg.cov_eps
    ).to(device)
    ckpt = torch.load(os.path.join(cfg.save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    loss = ckpt["best_loss"]
    epoch = ckpt["epoch"]
    print(f"best loss of {loss} saved at epoch {epoch}")
    return model


def _prepare_data(cfg: Stochastic_Node_DMD_Config):
    device = torch.device(cfg.device)
    return load_synth(device, T=cfg.eval_data_len, resolution=cfg.resolution)


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
def run_eval(cfg: Stochastic_Node_DMD_Config, mode: str = "teacher_forcing"):
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
        y_true_list,
        y_true_full_list,
        coords_full,
        *_,
    ) = _prepare_data(cfg)

    model = _prepare_model(cfg)
    # import time
    # time.sleep(1e6)
    vmin, vmax = _compute_vmin_vmax(y_true_full_list)

    out_dir = os.path.join(cfg.save_dir, feed_mode.value)
    ensure_dir(out_dir)

    coords_idx = find_subset_indices(coords_full, coords_list[0])

    mse_full_all = []
    calib_all    = []

    # --- 초기 y_in 설정 (AUTOREG 전용)
    y_pred_chain = y_true_full_list[0] if feed_mode == FeedMode.AUTOREG else None

    # --- 메인 루프
    for i in range(1, len(t_list)):
        print(f"Step {i}:", end=" ")
        coords = coords_full
        y_true = y_true_full_list[i]
        t_prev = float(t_list[i - 1])
        t_next = float(t_list[i])

        if feed_mode == FeedMode.TEACHER:
            y_in = y_true_full_list[i - 1]        # ground-truth teacher forcing
        else:
            # autoreg: 첫 스텝은 초기 truth에서 시작, 이후는 직전 예측을 연결
            y_in = y_pred_chain

        mu_u, logvar_u, *_ = model(coords, y_in, t_prev, t_next)

        # 다음 스텝 오토레그 입력 업데이트
        if feed_mode == FeedMode.AUTOREG:
            y_pred_chain = mu_u

        # --- MSE 및 플롯
        mse = F.mse_loss(mu_u, y_true).item()
        mse_full_all.append(mse)
        plot_reconstruction(coords, t_next, y_true, mu_u, mse, out_dir, vmin, vmax)

        # --- (B) 관측 서브셋(노이즈 포함)에서 불확실성 캘리브레이션
        y_obs      = y_list[i]         # noisy measurement at t_next (subset coords_list[0])
        y_clean    = y_true_list[i]    # clean target at subset
        mu_sub     = mu_u[coords_idx]
        logvar_sub = logvar_u[coords_idx]

        metrics = eval_uncertainty_metrics(
            y_obs, mu_sub, logvar_sub, y_true=y_clean
        )
        metrics["time_index"] = i
        metrics["mse_full"]   = mse
        calib_all.append(metrics)

    # --- 저장 및 요약 출력
    _summarize_and_dump(calib_all, mse_full_all, out_dir, feed_mode)


if __name__ == "__main__":
    # 기본값: teacher forcing
    run_eval(Stochastic_Node_DMD_Config(), mode="teacher_forcing")
    # 필요 시:
    # run_eval(Stochastic_Node_DMD_Config(), mode="autoreg")