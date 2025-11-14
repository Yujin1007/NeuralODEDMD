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
from dataset.generate_synth_dataset import load_synth
from models.node_dmd import Stochastic_NODE_DMD
from utils.utils import ensure_dir, eval_uncertainty_metrics, find_subset_indices, reparameterize_full
from utils.utils import render_mode_overlays_for_frame, to_complex_xy, ls_phi_from_fullgrid

from utils.plots import plot_reconstruction
import imageio
import matplotlib.pyplot as plt
# --- add this near the top of the file (after imports) ---
def _json_default(o):
    # complex → 문자열 "(a+bj)" 형태로 저장
    if isinstance(o, complex):
        return f"({o.real}{'+' if o.imag >= 0 else ''}{o.imag}j)"
    # NumPy 스칼라 → 파이썬 스칼라
    if isinstance(o, (_np.floating, _np.integer, _np.bool_)):
        return o.item()
    # NumPy 배열 → 리스트
    if isinstance(o, _np.ndarray):
        return o.tolist()
    # 그 외는 문자열로 fallback
    return str(o)


# --- [중요] φ 시간정렬: permutation만 맞추고 phase는 보존 ---
def _align_phi_over_time_for_lambda(phi_seq: np.ndarray):
    """
    phi_seq: (T, r) complex  [φ_0, φ_1, ..., φ_{T-1}]
    반환: permutation만 정렬된 φ 시퀀스 (phase 회전 없음 ⇒ 위상차=주파수 보존)
    """
    phi_seq = np.asarray(phi_seq).copy()
    T, r = phi_seq.shape
    for t in range(1, T):
        a = phi_seq[t-1]   # (r,)
        b = phi_seq[t]     # (r,)
        # phase 불변 유사도(크기 기반). a.conj()*b의 크기만 사용.
        C = np.abs(np.outer(a.conj(), b))     # (r, r)
        row, col = _hungarian_or_greedy_for_max(C)  # 기존 함수 재사용
        phi_seq[t] = b[col]                   # permutation만 적용 (phase 회전 금지)
    return phi_seq

# --- full-grid LS로 φ 복원:  φ = pinv(W) y ---
def _ls_phi_from_fullgrid(W_hat_full_c: np.ndarray, y_full_xy: torch.Tensor) -> np.ndarray:
    """
    W_hat_full_c : (N, r) complex   # 학습 모드(Full grid)
    y_full_xy    : (N, 2) torch     # [real, imag] full-grid field (GT 또는 예측)
    return       : (r,) complex     # LS 복원된 φ
    """
    y_c = (y_full_xy[:, 0].detach().cpu().numpy()
           + 1j * y_full_xy[:, 1].detach().cpu().numpy())  # (N,)
    phi_ls = np.linalg.pinv(W_hat_full_c) @ y_c            # (r,)
    return phi_ls

# --- 분기(2π/Δt) 보정: φ-기반 λ의 허수부 branch 정렬 ---
def _branch_correct(lam_hat_phi: np.ndarray, lam_true: np.ndarray, dt_eff: float):
    out = lam_hat_phi.copy()
    for i in range(len(lam_true)):
        w_t = np.imag(lam_true[i])
        w_h = np.imag(out[i])
        k = np.round((w_t - w_h) * dt_eff / (2*np.pi))   # 정수
        out[i] = np.real(out[i]) + 1j*(w_h + 2*np.pi*k/dt_eff)
    return out
class FeedMode(enum.Enum):
    AUTOREG = "autoreg"
    TEACHER = "teacher_forcing"

# -----------------
# 준비 유틸
# -----------------
def _prepare_model(cfg: Stochastic_Node_DMD_Config, model_name="best_model.pt") -> Stochastic_NODE_DMD:
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

def _prepare_data(cfg: Stochastic_Node_DMD_Config):
    device = torch.device(cfg.device)
    # returns: t_list, coords_list, y_list, y_true_list, y_true_full_list, coords_full, gt_params, W_full
    return load_synth(device, T=cfg.data_len, norm_T=cfg.data_len, resolution=cfg.resolution, dt=cfg.dt)

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

# -----------------
# 동역학 분석 유틸
# -----------------
def _to_complex_xy(arr_xy: np.ndarray) -> np.ndarray:
    """[...,2] -> complex (...), 마지막 축 [real, imag]"""
    return arr_xy[..., 0] + 1j * arr_xy[..., 1]

def _l2_norm_cols(Z: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(Z, axis=0, keepdims=True) + 1e-12
    return Z / nrm

def _hungarian_or_greedy_for_max(M: np.ndarray):
    """최대화 할당 (헝가리안; SciPy가 없으면 그리디 fallback)"""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-M)
        return r, c
    except Exception:
        # greedy
        M = M.copy()
        m, n = M.shape
        rs, cs = [], []
        used_r, used_c = set(), set()
        for _ in range(min(m, n)):
            idx = np.argmax(M)
            i, j = np.unravel_index(idx, M.shape)
            if i in used_r or j in used_c:
                M[i, j] = -np.inf
                continue
            rs.append(i); cs.append(j)
            used_r.add(i); used_c.add(j)
            M[i, :] = -np.inf
            M[:, j] = -np.inf
        return np.array(rs), np.array(cs)

def _match_modes_by_corr(W_hat: np.ndarray, W_true: np.ndarray):
    """열-정규화 후 |<w_hat_i, w_true_j>| 최대화 매칭 + 위상 추정"""
    A = _l2_norm_cols(W_hat); B = _l2_norm_cols(W_true)
    C = np.abs(A.conj().T @ B)  # (r_hat, r_true)
    row, col = _hungarian_or_greedy_for_max(C)
    phases, cosines = [], []
    for i, j in zip(row, col):
        inner = (A[:, i].conj() * B[:, j]).sum()
        phases.append(np.angle(inner)); cosines.append(np.abs(inner))
    return col, np.array(phases), np.array(cosines)  # perm: learned->true

def _align_phi_over_time(phi_seq: np.ndarray):
    """시간 정렬: t-1과 t 사이 모드 permute/위상 정렬 (헝가리안+phase)"""
    phi_seq = np.asarray(phi_seq).copy()
    Tm1, r = phi_seq.shape
    perms = []; thetas = []
    for t in range(1, Tm1):
        a = phi_seq[t-1]; b = phi_seq[t]
        aa = np.abs(a) + 1e-12; bb = np.abs(b) + 1e-12
        C = np.abs(np.outer(a.conj()/aa, b/bb))      # (r,r)
        row, col = _hungarian_or_greedy_for_max(C)
        b_perm = b[col]
        th = np.angle(a.conj()*b_perm)
        b_aligned = b_perm * np.exp(-1j*th)
        phi_seq[t] = b_aligned
        perms.append(col); thetas.append(th)
    return phi_seq, perms, thetas

def _branch_correct(lam_hat_phi: np.ndarray, lam_true: np.ndarray, dt_eff: float):
    """허수부 branch: ω_hat → ω_hat + 2πk/Δt 로 GT ω에 가장 가깝게 이동"""
    out = lam_hat_phi.copy()
    for i in range(len(lam_true)):
        w_t = np.imag(lam_true[i])
        w_h = np.imag(out[i])
        k = np.round((w_t - w_h) * dt_eff / (2*np.pi))   # 정수
        out[i] = np.real(out[i]) + 1j*(w_h + 2*np.pi*k/dt_eff)
    return out

def _principal_angles(W_hat: np.ndarray, W_true: np.ndarray, k: Optional[int] = None):
    """부분공간 각(상위 k)"""
    def _orth(X):
        Q, _ = np.linalg.qr(X)
        return Q
    if k is None: k = min(W_hat.shape[1], W_true.shape[1])
    Ah = _orth(_l2_norm_cols(W_hat)[:, :k])
    At = _orth(_l2_norm_cols(W_true)[:, :k])
    s = np.linalg.svd(Ah.conj().T @ At, compute_uv=False)  # cos(theta)
    s = np.clip(s, 0, 1)
    return np.arccos(s)   # (k,)

def _fit_linear_A(phi_seq_aligned: np.ndarray) -> np.ndarray:
    """phi_{t+1} ≈ A phi_t  (최소제곱)"""
    Phi0 = phi_seq_aligned[:-1].T    # (r, T-2)
    Phi1 = phi_seq_aligned[1:].T     # (r, T-2)
    return Phi1 @ np.linalg.pinv(Phi0)

def _diag_consistency(A: np.ndarray, lam_hat: np.ndarray, dt_eff: float):
    """A와 diag(exp(lam*Δt)) 비교"""
    A_ideal = np.diag(np.exp(lam_hat * dt_eff))
    offdiag = A - np.diag(np.diag(A))
    off_E = float(np.linalg.norm(offdiag, 'fro')**2)
    rel_err = float(np.linalg.norm(A - A_ideal, 'fro') / (np.linalg.norm(A, 'fro') + 1e-12))
    return off_E, rel_err

def _plot_lambda_compare(out_dir, lam_true, lam_enc_med, lam_phi_bc):
    """α/ω 막대 비교 + 복소평면 산점도"""
    os.makedirs(out_dir, exist_ok=True)
    def _split(lv):
        return np.real(lv), np.imag(lv)
    labels = [f"mode{i}" for i in range(len(lam_true))]
    x = np.arange(len(labels))

    # (1) alpha/omega bar
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    W = 0.25
    for a, name, off in [
        (_split(lam_true)[0], "GT α", -W),
        (_split(lam_enc_med)[0], "Enc α", 0.0),
        (_split(lam_phi_bc)[0], "φ α", W)
    ]:
        ax[0].bar(x+off, a, width=W, label=name)
    for w, name, off in [
        (_split(lam_true)[1], "GT ω", -W),
        (_split(lam_enc_med)[1], "Enc ω", 0.0),
        (_split(lam_phi_bc)[1], "φ ω", W)
    ]:
        ax[1].bar(x+off, w, width=W, label=name)
    ax[0].set_ylabel("alpha (real)")
    ax[1].set_ylabel("omega (imag)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[0].legend(ncol=3, fontsize=9)
    ax[1].legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lambda_compare_bars.png"), dpi=150)
    plt.close(fig)

    # (2) complex plane
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(np.real(lam_true),    np.imag(lam_true),    marker='x', s=80, label="GT")
    ax.scatter(np.real(lam_enc_med), np.imag(lam_enc_med), marker='o', s=60, label="Enc med")
    ax.scatter(np.real(lam_phi_bc),  np.imag(lam_phi_bc),  marker='^', s=60, label="φ-based")
    for i in range(len(lam_true)):
        ax.annotate(f"{i}", (np.real(lam_true)[i], np.imag(lam_true)[i]))
    ax.axhline(0, color='k', linewidth=0.5); ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlabel("Re(λ)"); ax.set_ylabel("Im(λ)")
    ax.set_title("Eigenvalues in Complex Plane")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lambda_complex_plane.png"), dpi=150)
    plt.close(fig)

def _plot_A_heatmap(out_dir, A):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(np.abs(A), origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("|A| heatmap")
    ax.set_xlabel("from"); ax.set_ylabel("to")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "A_heatmap.png"), dpi=150)
    plt.close(fig)

def _plot_mode_cosines(out_dir, cosines):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(cosines))
    ax.bar(x, cosines)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x); ax.set_xticklabels([f"m{i}" for i in x])
    ax.set_ylabel("cosine similarity")
    ax.set_title("Mode cosine (learned vs GT)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mode_cosines.png"), dpi=150)
    plt.close(fig)

def _plot_residual_curve(out_dir, Rt):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(np.arange(len(Rt)), Rt, marker='o', linewidth=1.5)
    ax.set_xlabel("time index")
    ax.set_ylabel("residual ratio R_t")
    ax.set_title("Step residual ratio: ||φ_{t+1}-e^{λΔt}φ_t|| / ||e^{λΔt}φ_t||")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residual_ratio_curve.png"), dpi=150)
    plt.close(fig)
def _align_phi_over_time_for_lambda(phi_seq: np.ndarray):
    # permutation만 정렬, phase 회전은 하지 않음
    phi_seq = np.asarray(phi_seq).copy()
    Tm1, r = phi_seq.shape
    for t in range(1, Tm1):
        a = phi_seq[t-1]; b = phi_seq[t]
        # 크기 기반 유사도 (phase 불변)
        C = np.abs(np.outer(a.conj(), b))  # (r,r)
        row, col = _hungarian_or_greedy_for_max(C)
        phi_seq[t] = b[col]  # 회전 X
    return phi_seq
def np_complex_to_torch_xy(arr_c, device):
    # arr_c: (N, r) complex (np.ndarray)
    return torch.stack([
        torch.as_tensor(np.real(arr_c), device=device, dtype=torch.float32),
        torch.as_tensor(np.imag(arr_c), device=device, dtype=torch.float32),
    ], dim=-1)  # (N, r, 2)
# -----------------
# 메인 평가
# -----------------
@torch.no_grad()
def run_eval(cfg: Stochastic_Node_DMD_Config, mode: str = "teacher_forcing", model_name: str = "best_model.pt"):

    (
        t_list,
        coords_list,
        y_list,
        y_true_list,
        y_true_full_list,
        coords_full,
        gt_params,
        W_full,   # (N, r_true) complex
    ) = _prepare_data(cfg)

    model = _prepare_model(cfg, model_name=model_name)
    vmin, vmax = _compute_vmin_vmax(y_true_full_list)
    coords_idx = find_subset_indices(coords_full, coords_list[0])
    out_dir = os.path.join(cfg.save_dir, f"{FeedMode(mode).value}_reconstruction")
    ensure_dir(out_dir)
    # --- 학습 모드(Full grid) 1회 추출
    W_hat_full = model.mode_net(coords_full)                           # (N, r, 2)
    W_hat_full_np = W_hat_full.detach().cpu().numpy()
    if W_hat_full_np.ndim == 4: W_hat_full_np = W_hat_full_np[0]       # (N, r, 2)
    W_hat_full_c = _to_complex_xy(W_hat_full_np)                       # (N, r) complex

    dt_eff = getattr(cfg, "phi_dt_eff", 0.1)  # ★ 신규: 연속시간 Δt (합성 0.1)
    device = torch.device(cfg.device)
    W_full_np = np.asarray(W_full)
    r_learned = W_hat_full_c.shape[1]
    if W_full_np.shape[1] != r_learned:
        W_full_np = W_full_np[:, :r_learned]  # 필요시 잘라 쓰기(또는 별도 매칭 로직)
    # 학습 모드와 GT 모드 정렬(코사인 최대가 되도록 퍼뮤테이션)
    perm, _, _ = _match_modes_by_corr(W_hat_full_c, W_full_np)
    W_full_np_aligned = W_full_np[:, perm]             # (N, r)
    # torch (N, r, 2)로 변환
    W_full_torch = np_complex_to_torch_xy(W_full_np_aligned, device)

    # --- 시퀀스 컨테이너
    lam_seq_list: List[np.ndarray] = []   # encoder λ(t) 수집
    phi_ls_seq:  List[np.ndarray] = []    # ★ 신규: full-grid LS φ(t) 수집
    H, Wpx = cfg.resolution                # eval 고정

    modeviz_dir = os.path.join(out_dir, "mode_portrait_overlays")
    os.makedirs(modeviz_dir, exist_ok=True)
# --- 재구성 표시/캘리브레이션(기존 그대로)
    mse_full_all, calib_all = [], []
    frames = [plot_reconstruction(coords_full, 0, y_true_full_list[0], y_true_full_list[0], 0, out_dir, vmin, vmax)]
    y_pred_chain = y_true_full_list[0] if FeedMode(mode) == FeedMode.AUTOREG else None

    for i in range(1, len(t_list)):
        coords = coords_full
        y_true = y_true_full_list[i]
        t_prev = float(t_list[i - 1])
        t_next = float(t_list[i])

        y_in = y_true_full_list[i - 1] if FeedMode(mode) == FeedMode.TEACHER else y_pred_chain

        # 모델 실행 (full-grid)
        mu_u, logvar_u, cov_u, mu_phi, logvar_phi, lam, W = model(coords, y_in, t_prev, t_next)
    
        # ★ 신규: encoder λ(t_prev) 수집
        lam_seq_list.append(_to_complex_xy(lam.detach().cpu().numpy()))  # (r,)

        # ★ 신규: full-grid LS φ(t) 시퀀스(노이즈/부분관측 영향↓)
        #   - 평가 공정함을 위해 GT field 사용을 권장(y_true_full_list)
        if i == 1:
            phi_prev = _ls_phi_from_fullgrid(W_hat_full_c, y_true_full_list[i - 1])  # φ_{t-1}
            phi_ls_seq.append(phi_prev)
        phi_next = _ls_phi_from_fullgrid(W_hat_full_c, y_true_full_list[i])          # φ_{t}
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
            _summarize_and_dump(calib_all, mse_full_all, out_dir, FeedMode(mode))
            imageio.mimsave(f"{out_dir}/reconstruction.gif", frames, fps=10)

        # --- (A) GT 기준 오버레이: phi는 LS로 역추정 (모델 독립적 분해)
        _ = render_mode_overlays_for_frame(
            W_hat_full=W_full_torch,     # ✅ GT 모드로 분해
            field_full_xy=y_true,        # GT 필드
            grid_shape=(H, Wpx),
            out_dir=modeviz_dir, t_idx=i,
            base_cmap="viridis",
            mode="gt"                    # phi_override=None → LS-phi 사용
        )

        # --- (B) 예측 기준 오버레이: 모델이 낸 phi/W를 그대로 사용
        _ = render_mode_overlays_for_frame(
            W_hat_full=W_hat_full,       # 기본(정적) 학습 모드(없어도 되지만 유지)
            field_full_xy=mu_u,          # 모델 예측 필드
            grid_shape=(H, Wpx),
            out_dir=modeviz_dir, t_idx=i,
            base_cmap="viridis",
            mode="pred",
            phi_override=mu_phi,         # ✅ 모델 φ 사용
            W_override=W                 # ✅ 이 타임스텝의 모델 모드로 오버라이드
        )

    imageio.mimsave(f"{out_dir}/exploitation.gif", frames, fps=10)
    _summarize_and_dump(calib_all, mse_full_all, out_dir, FeedMode(mode))
    # ===============================
    # 동역학 식별 리포트 (연속시간)
    # ===============================
    try:
        report = {}
        alpha, omega, b = gt_params
        lam_true = np.asarray(alpha) + 1j*np.asarray(omega)
        report["lambda_true"] = [complex(z) for z in lam_true]

        # --- 모드 품질 (기존 루틴 사용)
        if W_full is not None:
            W_full_np = np.asarray(W_full)
            perm, phases, cosines = _match_modes_by_corr(W_hat_full_c, W_full_np)
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
            phi_aligned = _align_phi_over_time_for_lambda(phi_arr)   # (T, r)  # ★ 핵심
            ratio = phi_aligned[1:] / (phi_aligned[:-1] + 1e-12)     # (T-1, r)
            lam_phi = np.median(np.log(ratio), axis=0) / float(dt_eff)  # (r,)
            lam_phi_bc = _branch_correct(lam_phi, lam_true, float(dt_eff))
            report["lambda_from_phi"] = [complex(z) for z in lam_phi_bc]
        else:
            lam_phi_bc = None

        # --- GT와의 매칭/오차
        def _match_err(lh, lt):
            Lh = lh.reshape(-1, 1); Lt = lt.reshape(1, -1)
            cost = np.abs(Lh - Lt)
            r_idx, t_idx = _hungarian_or_greedy_for_max(-cost)
            err = cost[r_idx, t_idx]
            return err, t_idx

        if lam_enc_med is not None:
            err_enc, t_enc = _match_err(lam_enc_med, lam_true)
            report.update({
                "lambda_error_abs_enc_vs_true_each": err_enc.tolist(),
                "lambda_error_abs_enc_vs_true_mean": float(err_enc.mean()),
                "lambda_match_true_index_for_each_learned": t_enc.tolist(),
            })
        if lam_phi_bc is not None:
            err_phi, _ = _match_err(lam_phi_bc, lam_true)
            report.update({
                "lambda_error_abs_phi_vs_true_each": err_phi.tolist(),
                "lambda_error_abs_phi_vs_true_mean": float(err_phi.mean()),
            })

        # --- Enc vs φ-기반 일치도
        if (lam_enc_med is not None) and (lam_phi_bc is not None):
            report["lambda_enc_vs_phi_absdiff"] = np.abs(lam_enc_med - lam_phi_bc).tolist()

        # --- A 적합/대각 일관성(LS φ 사용)
        if len(phi_ls_seq) >= 3:
            A = _fit_linear_A(phi_aligned)                              # (r, r)
            lam_hat_use = lam_enc_med if lam_enc_med is not None else lam_phi_bc
            offdiag_energy, rel_err = _diag_consistency(A, lam_hat_use, float(dt_eff))
            report.update({
                "A_offdiag_energy": offdiag_energy,
                "A_rel_err_to_ideal_diag": rel_err,
            })
            _plot_A_heatmap(out_dir, A)

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
                _plot_residual_curve(out_dir, Rt)

        # --- 시각화(가능할 때만)
        if (lam_enc_med is not None) and (lam_phi_bc is not None):
            _plot_lambda_compare(out_dir, lam_true, lam_enc_med, lam_phi_bc)

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
# -----------------
# 엔트리포인트
# -----------------
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