# ============= eval_rollout.py ==================
import os
import json
import enum
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional

from utils.losses import stochastic_loss_fn
from config.config import Stochastic_Node_DMD_Config
from models.node_dmd import Stochastic_NODE_DMD
from utils.utils import ensure_dir, eval_uncertainty_metrics, find_subset_indices, reparameterize_full, prepare_data
from utils.utils import compute_vmin_vmax, FeedMode, summarize_and_dump
from utils.dynamics_analysis import *
from utils.plots import plot_reconstruction
import imageio
import matplotlib.pyplot as plt
def _json_default(o):
    # complex → 문자열 "(a+bj)" 형태로 저장
    if isinstance(o, complex):
        return f"({o.real}{'+' if o.imag >= 0 else ''}{o.imag}j)"
    # NumPy 스칼라 → 파이썬 스칼라
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    # NumPy 배열 → 리스트
    if isinstance(o, np.ndarray):
        return o.tolist()
    # 그 외는 문자열로 fallback
    return str(o)
def prepare_model(cfg, model_name="best_model.pt"):
    device = torch.device(cfg.device)
    model = Stochastic_NODE_DMD(
        cfg.r, cfg.hidden_dim, cfg.ode_steps, cfg.process_noise, cfg.cov_eps, cfg.dt,
        cfg.mode_frequency, cfg.phi_frequency
    ).to(device)
    ckpt = torch.load(os.path.join(cfg.save_dir, model_name), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    loss = ckpt.get("best_loss", None)
    epoch = ckpt.get("epoch", None)
    print(f"[ckpt] best loss: {loss}, epoch: {epoch}")
    return model
@torch.no_grad()
def run_eval(cfg: Stochastic_Node_DMD_Config, mode: str = "teacher_forcing", model_name: str = "best_model.pt"):
    device = torch.device(cfg.device)
    r = cfg.r
    (
        t_list,
        coords_list,
        y_list,
        y_true_list,
        y_true_full_list,
        coords_full,
        gt_params,
        W_full,   # (N, r_true) complex
    ) = prepare_data(cfg)

    model = prepare_model(cfg, model_name=model_name)
    vmin, vmax = compute_vmin_vmax(y_true_full_list)
    coords_idx = find_subset_indices(coords_full, coords_list[0])
    out_dir = os.path.join(cfg.save_dir, f"{FeedMode(mode).value}_reconstruction")
    ensure_dir(out_dir)
    vmin, vmax = compute_vmin_vmax(y_true_full_list)
    
    # Mode from network
    W_hat_full = model.mode_net(coords_full)                           # (N, r, 2)
    W_hat_full_np = W_hat_full.detach().cpu().numpy()
    if W_hat_full_np.ndim == 4: W_hat_full_np = W_hat_full_np[0]       # (N, r, 2)
    W_hat_full_c = to_complex_xy(W_hat_full_np)                       # (N, r) complex


    dt_eff = getattr(cfg, "phi_dt_eff", 0.1)  # ★ 신규: 연속시간 Δt (합성 0.1)
    # Mode from GT
    W_full_np = np.asarray(W_full)
    if W_full_np.shape[1] != r:
        W_full_np = W_full_np[:, :r]  # 필요시 잘라 쓰기(또는 별도 매칭 로직)
    
    # Match modes by correlation
    perm, _, _ = match_modes_by_corr(W_hat_full_c, W_full_np)
    W_full_np_aligned = W_full_np[:, perm]             # (N, r)
    # torch (N, r, 2)로 변환
    W_full_torch = np_complex_to_torch_xy(W_full_np_aligned, device)

    # --- 시퀀스 컨테이너
    lam_seq_list: List[np.ndarray] = []   # encoder λ(t) 수집
    phi_ls_seq:  List[np.ndarray] = []    # from GT and W from model, calculate phi from full-grid LS
    H, Wpx = cfg.resolution                # eval 고정

    modeviz_dir = os.path.join(out_dir, "mode_portrait_overlays")
    os.makedirs(modeviz_dir, exist_ok=True)
# --- 재구성 표시/캘리브레이션(기존 그대로)
    mse_full_all, calib_all = [], []
    frames = [plot_reconstruction(coords_full, 0, y_true_full_list[0], y_true_full_list[0], 0, out_dir, vmin, vmax)]
    y_pred_chain = y_true_full_list[0] if FeedMode(mode) == FeedMode.AUTOREG else None
    gt_seq = [y_true_full_list[i] for i in range(1, len(t_list))]

    # (A) GT 모드로 절대 레벨 생성 (권장)
    abs_levels_gt = compute_abs_levels_over_sequence(
        W_modes=W_full_torch,          # GT 모드 (정렬된 버전)
        fields_seq=gt_seq,             # GT 필드 전 시간대
        perc=(0.3, 0.6, 0.9),          # 원하는 절대 레벨 비율
        phi_seq_override=None          # LS-phi 사용
    )
    for i in range(1, len(t_list)):
        coords = coords_full
        y_true = y_true_full_list[i]
        t_prev = float(t_list[i - 1])
        t_next = float(t_list[i])

        y_in = y_true_full_list[i - 1] if FeedMode(mode) == FeedMode.TEACHER else y_pred_chain

        # 모델 실행 (full-grid)
        mu_u, logvar_u, cov_u, mu_phi, logvar_phi, lam, W = model(coords, y_in, t_prev, t_next)
    
        # ★ 신규: encoder λ(t_prev) 수집
        lam_seq_list.append(to_complex_xy(lam.detach().cpu().numpy()))  # (r,)

        if i == 1:
            phi_prev = ls_phi_from_fullgrid(W_hat_full_c, y_true_full_list[i - 1])  # φ_{t-1}
            phi_ls_seq.append(phi_prev)
        phi_next = ls_phi_from_fullgrid(W_hat_full_c, y_true_full_list[i])          # φ_{t}
        phi_ls_seq.append(phi_next)

        # 오토레그 입력 업데이트
        # u_pred = reparameterize_full(mu_u, cov_u)
        u_pred = mu_u
        if FeedMode(mode) == FeedMode.AUTOREG:
            y_pred_chain = u_pred

        # MSE/플롯 (기존)
        mse = F.mse_loss(u_pred, y_true).item()
        mse_full_all.append(mse)
        frames.append(plot_reconstruction(coords, i, y_true, u_pred, mse, out_dir, vmin, vmax))

        # 서브셋 캘리브레이션 (기존)
        y_obs      = y_list[i]
        mu_sub     = mu_u[coords_idx]
        logvar_sub = logvar_u[coords_idx]
        metrics = eval_uncertainty_metrics(y_obs, mu_sub, logvar_sub)
        metrics["time_index"] = i
        metrics["mse_full"]   = mse
        calib_all.append(metrics)

        if i == cfg.data_len:
            summarize_and_dump(calib_all, mse_full_all, out_dir, FeedMode(mode))
            imageio.mimsave(f"{out_dir}/reconstruction.gif", frames, fps=10)
        
        # GT 분해(공정 비교): 학습 모드 W_hat_full 또는 GT W_full 정렬본 사용 가능
        _ = render_mode_overlays_for_frame(
                W_modes=W_full_torch,      # 또는 W_hat_full
                field_full_xy=y_true,              # GT field
                coords_full=coords_full,
                out_dir=modeviz_dir, t_idx=i,
                title_prefix="Ground Truth", 
                level_policy="absolute",
                abs_levels_per_mode=abs_levels_gt,
                # vmin=vmin, vmax=vmax,
                contour_colors=["white", "lightpink", "darkorange", "red"],  # 모드 0~3 색
                contour_linewidth=1.5,
                contour_alpha=1.0,
            )

        # 예측 분해: 모델이 낸 W/phi 그대로 사용하고 싶으면 아래처럼 두 단계
        # (1) 먼저 모델 forward로 mu_u, mu_phi, W 얻은 뒤
        _ = render_mode_overlays_for_frame(
                W_modes=W,                          # 모델 시점의 모드
                field_full_xy=mu_u,                 # 모델 예측 필드
                coords_full=coords_full,
                out_dir=modeviz_dir, t_idx=i,
                phi_override=mu_phi,                # 모델의 φ 강제 사용
                title_prefix="NODE DMD",
                level_policy="absolute",
                abs_levels_per_mode=abs_levels_gt,
                # vmin=vmin, vmax=vmax,
                contour_colors=["white", "lightpink", "darkorange", "red"],  # 모드 0~3 색
                contour_linewidth=1.5,
                contour_alpha=1.0,
        )

    imageio.mimsave(f"{out_dir}/exploitation.gif", frames, fps=10)
    summarize_and_dump(calib_all, mse_full_all, out_dir, FeedMode(mode))
    # --- Dynamics Identification Report (continuous-time) ---
    try:
        report = {}
        alpha, omega, b = gt_params
        lam_true = np.asarray(alpha) + 1j*np.asarray(omega)
        report["lambda_true"] = [complex(z) for z in lam_true]

        # --- 모드 품질 (기존 루틴 사용)
        if W_full is not None:
            W_full_np = np.asarray(W_full)
            perm, phases, cosines = match_modes_by_corr(W_hat_full_c, W_full_np)
            report.update({
                "mode_cosine_mean": float(cosines.mean()),
                "mode_cosines_each": cosines.tolist(),
                "mode_perm_true_index_for_each_learned": perm.tolist(),
                "mode_phases_rad": phases.tolist(),
            })
        else:
            report["mode_note"] = "GT W_full 미제공"

        # --- Enc λ: 시계열 중앙값 및 시간-분산
        if len(lam_seq_list) > 0:
            lam_seq = np.stack(lam_seq_list, axis=0)      # (T-1, r)
            lam_enc_med = np.median(lam_seq, axis=0)      # (r,)
            lam_enc_std = np.std(lam_seq, axis=0)         # (r,)
            report["lambda_enc_median"] = [complex(z) for z in lam_enc_med]
            report["lambda_enc_time_std"] = lam_enc_std.tolist()
        else:
            lam_enc_med = None

        # --- φ-기반 λ: permutation만 정렬 + 연속시간 log-ratio + branch 보정
        if len(phi_ls_seq) >= 2:
            phi_arr = np.stack(phi_ls_seq, axis=0)           # (T, r)
            phi_aligned = align_phi_over_time_for_lambda(phi_arr)   # (T, r)  # ★ 핵심
            ratio = phi_aligned[1:] / (phi_aligned[:-1] + 1e-12)     # (T-1, r)
            lam_phi = np.median(np.log(ratio), axis=0) / float(dt_eff)  # (r,)
            lam_phi_bc = branch_correct(lam_phi, lam_true, float(dt_eff))
            report["lambda_from_phi"] = [complex(z) for z in lam_phi_bc]
        else:
            lam_phi_bc = None
        

        if lam_enc_med is not None:
            err_enc, t_enc = match_err(lam_enc_med, lam_true)
            err_enc_perm = np.abs(lam_enc_med[perm] - lam_true)
            
            report.update({
                "lambda_error_abs_enc_vs_true_each": err_enc.tolist(),
                "lambda_error_abs_enc_vs_true_mean": float(err_enc.mean()),
                "lambda_match_true_index_for_each_learned": t_enc.tolist(),
                "lambda_error_abs_enc_vs_true_each_mode_perm": err_enc_perm.tolist(),
                "lambda_error_abs_enc_vs_true_each_mode_perm_mean": float(err_enc_perm.mean()),
            })
            # Calculate eigenvalue error using mode correlation matching (perm)
            # if W_full is not None and 'mode_perm_true_index_for_each_learned' in report:
            
        if lam_phi_bc is not None:
            err_phi, t_phi = match_err(lam_phi_bc, lam_true)
            # t_phi[i] gives the index in lam_true that matches lam_phi_bc[i]
            err_phi_perm = np.abs(lam_phi_bc[perm] - lam_true)

            report.update({
                "lambda_error_abs_phi_vs_true_each": err_phi.tolist(),
                "lambda_error_abs_phi_vs_true_mean": float(err_phi.mean()),
                "lambda_error_abs_phi_vs_true_each_mode_perm": err_phi_perm.tolist(),
                "lambda_error_abs_phi_vs_true_each_mode_perm_mean": float(err_phi_perm.mean()),
            })
            
            # Calculate mode error using t_phi matching (instead of Hungarian algorithm)
            if W_full is not None:
                phases, cosines = compute_mode_error_by_perm(W_hat_full_c, W_full_np, t_phi)
                report.update({
                    "mode_cosine_mean_phi": float(cosines.mean()),
                    "mode_cosines_each_phi": cosines.tolist(),
                    "mode_perm_true_index_for_each_learned_phi": t_phi.tolist(),
                    "mode_phases_rad_phi": phases.tolist(),
                })
        # --- Enc vs φ-기반 일치도
        if (lam_enc_med is not None) and (lam_phi_bc is not None):
            report["lambda_enc_vs_phi_absdiff"] = np.abs(lam_enc_med - lam_phi_bc).tolist()

        # --- A 적합/대각 일관성(LS φ 사용)
        if len(phi_ls_seq) >= 3:
            A = fit_linear_A(phi_aligned)                              # (r, r)
            lam_hat_use = lam_enc_med if lam_enc_med is not None else lam_phi_bc
            offdiag_energy, rel_err = diag_consistency(A, lam_hat_use, float(dt_eff))
            report.update({
                "A_offdiag_energy": offdiag_energy,
                "A_rel_err_to_ideal_diag": rel_err,
            })
            plot_A_heatmap(out_dir, A)

            # 잔차 비율 (Enc 기준, 선택)
            if lam_enc_med is not None:
                Rt = []
                for t in range(phi_aligned.shape[0]-1):
                    pred = np.exp(lam_enc_med * float(dt_eff)) * phi_aligned[t]
                    num = np.linalg.norm(phi_aligned[t+1] - pred)
                    den = np.linalg.norm(pred) + 1e-12
                    Rt.append(num/den)
                report["residual_ratio_median"] = float(np.median(Rt))
                report["residual_ratio_p90"]    = float(np.percentile(Rt, 90.0))
                

        # --- 시각화(가능할 때만)
        if (lam_enc_med is not None) and (lam_phi_bc is not None):
            plot_lambda_compare(out_dir, lam_true, lam_enc_med, lam_phi_bc)

        # 저장 (복소수 안전)
        with open(os.path.join(out_dir, "mode_eig_eval.json"), "w") as f:
            json.dump(report, f, indent=2, default=_json_default)

        print("=== Dynamics Identification Report (continuous-time) ===")
        for k, v in report.items():
            if isinstance(v, float):
                print(f"{k:>36}: {v:.6f}")
            else:
                print(f"{k:>36}: {v}")
    except Exception as e:
        print(f"[eval] dynamics analysis skipped due to error: {e}")
  

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Stochastic NODE-DMD with config.")
    parser.add_argument("--config_dir", type=str, required=True, help="Directory containing Stochastic_Node_DMD_Config.txt")
    parser.add_argument("--ckpt_name", type=str, default="best_model.pt", help="Checkpoint filename")
    args = parser.parse_args()

    # Load config from txt file
    config_path = os.path.join(args.config_dir, "Stochastic_Node_DMD_Config.txt")
    import ast
    config_dict = {}
    with open(config_path, "r") as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                k = k.strip()
                v = v.strip()
                if v.lower() == "true":
                    v = True
                elif v.lower() == "false":
                    v = False
                else:
                    try:
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

    cfg = Stochastic_Node_DMD_Config()
    for k, v in config_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # 연속시간 λ 평가 스텝(합성 시 0.1). Config에 없으면 기본 0.1 사용.
    if not hasattr(cfg, "phi_dt_eff"):
        cfg.phi_dt_eff = 0.1

    # 실행
    run_eval(cfg, mode="teacher_forcing", model_name=args.ckpt_name)
    # run_eval(cfg, mode="autoreg", model_name=args.ckpt_name)