
import os, numpy as np, torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.tri as mtri
# ---- helpers: complex <-> split ----

def to_numpy(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x

# ---- least-squares phi on full grid (complex) ----
def ls_phi_from_fullgrid(W_c: np.ndarray, u_c: np.ndarray, reg: float = 1e-6):
    """
    W_c: (N, r) complex, learned modes on full grid
    u_c: (N,)   complex, a full-grid field (GT or model prediction)
    returns: phi (r,) complex  minimizing ||W phi - u||^2 + reg||phi||^2
    """
    r = W_c.shape[1]
    A = W_c.conj().T @ W_c + reg * np.eye(r, dtype=np.complex128)
    b = W_c.conj().T @ u_c
    phi = np.linalg.solve(A, b)
    return phi

# ---- compute per-mode contribution maps ----
def mode_contributions(W_c: np.ndarray, phi: np.ndarray, grid_shape):
    """
    returns:
      contrib_real: list of (H,W) real-valued Re{W_k * phi_k}
      contrib_mag : list of (H,W) |W_k * phi_k|
    """
    H, W = grid_shape
    r = W_c.shape[1]
    contrib_real, contrib_mag = [], []
    for k in range(r):
        u_k = W_c[:, k] * phi[k]          # (N,) complex
        u_k_real = np.real(u_k).reshape(H, W)
        u_k_mag  = np.abs(u_k).reshape(H, W)
        contrib_real.append(u_k_real)
        contrib_mag .append(u_k_mag)
    return contrib_real, contrib_mag

# ---- overlay style 1: contour lines on top of base image ----
def overlay_mode_contours(base_img, contrib_mag, save_path,
                          levels=(90,95,98), mode_colors=None, cmap_base="viridis",
                          title="Mode Portrait: Contours", dpi=160):
    """
    base_img   : (H,W) real (e.g., reconstructed field real-part)
    contrib_mag: list[(H,W)] absolute contribution per mode
    """
    H, W = base_img.shape
    r = len(contrib_mag)
    if mode_colors is None:
        # distinct colors per mode
        mode_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:r]

    plt.figure(figsize=(5.3, 4.4), dpi=dpi)
    plt.imshow(base_img, origin="lower", cmap=cmap_base)
    handles = []
    for k in range(r):
        mag = contrib_mag[k]
        # percentile-based levels for clean contours
        lv = [np.percentile(mag, p) for p in levels]
        try:
            cs = plt.contour(mag, levels=lv, colors=[mode_colors[k]]*len(lv), linewidths=1.0)
            handles.append(Patch(facecolor=mode_colors[k], edgecolor=mode_colors[k], label=f"mode {k}"))
        except Exception:
            # if levels degenerate, skip
            pass
    # plt.colorbar(fraction=0.046, pad=0.04, label="Reconstruction (base)")
    # cbar.remove()  # 컬러바 제거
    if handles:
        plt.legend(handles=handles, loc="upper right", frameon=True)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# ---- convenience: one call per time index to make overlays ----
def render_mode_overlays_for_frame_mosaic(
    W,          # (N,r,2) torch  [default modes]
    field_full_xy,       # (N,2)   torch  (GT or model pred)
    grid_shape, out_dir, t_idx,
    base_cmap="viridis", mode='gt',
    phi_override=None,   # ✅ (r,2) torch or (r,) complex-like → 사용 시 LS-phi 건너뜀
):
    """
    Produces two overlay images for time t_idx:
      - contour overlay
      - alpha-blend overlay
    """
    W_np = to_numpy(W)                 # (N,r,2)
    field_np = to_numpy(field_full_xy)         # (N,2)
    W_c = to_complex_xy(W_np)              # (N,r) complex
    u_c = to_complex_xy(field_np)              # (N,)   complex
    H, W = grid_shape

    # phi 선택: override가 있으면 그대로 사용, 없으면 LS로 추정
    if phi_override is not None:
        phi_np = to_numpy(phi_override)
        # (r,2) split이면 complex로 변환
        if phi_np.ndim == 2 and phi_np.shape[-1] == 2:
            phi_c = to_complex_xy(phi_np)      # (r,)
        else:
            phi_c = np.asarray(phi_np).reshape(-1)  # already complex-like
    else:
        phi_c = ls_phi_from_fullgrid(W_c, field_full_xy) # (r,)

    # per-mode contributions
    contrib_real, contrib_mag = mode_contributions(W_c, phi_c, (H, W))
    base_img = field_np[:, 0].reshape(H, W)    # Re{u} as base

    # 저장
    os.makedirs(out_dir, exist_ok=True)
    overlay_mode_contours(
        base_img, contrib_mag,
        save_path=os.path.join(out_dir, f"mode_overlay_contours_t{t_idx:03d}_{mode}.png"),
        cmap_base=base_cmap,
        title=f"Contours @ t={t_idx} ({mode})"
    )
    return phi_c  # 사용된 phi를 반환(로그/검증 용)
def render_mode_overlays_for_frame(
    W_modes,            # (N,r,2) torch/np
    field_full_xy,      # (N,2)   torch/np
    coords_full,        # (N,2)   torch/np
    out_dir, t_idx,
    phi_override=None,  # (r,2) 또는 (r,)
    base_cmap="viridis",
    # --- NEW ---
    level_policy="quantile",          # "quantile" | "absolute"
    levels_quantile=(90, 95, 98),     # 사용: level_policy="quantile"
    abs_levels_per_mode=None,         # 사용: level_policy="absolute", shape (r, L)
    point_size=75,
    alpha_points=0.95,
    dpi=160,
    title_prefix="",
    vmin=None,
    vmax=None,
    contour_colors=None,      # 예: ["#E69F00","#56B4E9","#009E73","#CC79A7"]
    contour_linewidth=1.2,
    contour_alpha=1.0,
):
    """
    Scatter + tricontour. 등고선 레벨 정책을 선택:
      - quantile : 프레임별 분위수 (기존 동작; 스케일 변화가 사라짐)
      - absolute : 전역 고정 레벨(모드별 절대값), 시간/모델 비교에 권장
    """
    # --- 입력 정리 ---
    W_np  = to_numpy(W_modes)
    u_np  = to_numpy(field_full_xy)
    xy_np = to_numpy(coords_full)
    W_c   = to_complex_xy(W_np)        # (N,r)
    u_c   = to_complex_xy(u_np)        # (N,)
    x, y  = xy_np[:, 0].reshape(-1), xy_np[:, 1].reshape(-1)

    # --- phi 선택 ---
    if phi_override is not None:
        phi_np = to_numpy(phi_override)
        phi_c  = to_complex_xy(phi_np) if (phi_np.ndim==2 and phi_np.shape[-1]==2) \
                 else np.asarray(phi_np).reshape(-1)
    else:
        phi_c  = ls_phi_from_fullgrid(W_c, field_full_xy)  # (r,)

    # --- per-mode contributions: |W_k(x) * phi_k|
    r = W_c.shape[1]
    contrib_mag = [np.abs(W_c[:, k] * phi_c[k]) for k in range(r)]  # list of (N,)
    base_flat   = u_np[:, 0].reshape(-1)                            # Re{u} 배경

    # --- 플로팅 ---
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 3), dpi=dpi)

    sc = plt.scatter(x, y, c=base_flat, s=point_size, alpha=alpha_points,
                     cmap=base_cmap, edgecolors='none', vmin=vmin, vmax=vmax)
    tri = mtri.Triangulation(x, y)

    handles = []
    # --- NEW: 모드별 색상 준비 ---
    if contour_colors is None:
        contour_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(contour_colors) < r:
        raise ValueError(f"contour_colors 길이({len(contour_colors)})가 모드 개수 r({r})보다 작습니다.")

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for k in range(r):
        mag_k = contrib_mag[k]
        if level_policy == "absolute":
            assert abs_levels_per_mode is not None, "abs_levels_per_mode 필요"
            lv = np.asarray(abs_levels_per_mode[k], dtype=float)  # shape (L,)
        else:  # "quantile"
            qs = np.asarray(levels_quantile, dtype=float)
            lv = np.percentile(mag_k, qs)

        # 레벨이 퇴화하면(최소==최대) contour 생략
        if not (np.isfinite(lv).all() and (lv.max() > lv.min())):
            col = contour_colors[k]
            handles.append(Patch(facecolor=col, edgecolor=col, label=f"mode {k} (flat)"))
            continue
            # handles.append(Patch(facecolor=color_cycle[k % len(color_cycle)],
            #                      edgecolor=color_cycle[k % len(color_cycle)],
            #                      label=f"mode {k} (flat)"))
            # continue

        try:
            plt.tricontour(tri, mag_k, levels=lv,
                           colors=[color_cycle[k % len(color_cycle)]] * len(lv),
                           linewidths=1.0)
            handles.append(Patch(facecolor=color_cycle[k % len(color_cycle)],
                                 edgecolor=color_cycle[k % len(color_cycle)],
                                 label=f"mode {k}"))
        except Exception:
            pass
        col = contour_colors[k]
        plt.tricontour(
            tri, mag_k, levels=lv,
            colors=[col] * len(lv),
            linewidths=contour_linewidth, alpha=contour_alpha
        )
        handles.append(Patch(facecolor=col, edgecolor=col, label=f"mode {k}"))
    if handles:
        plt.legend(
            handles=handles,
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=True, borderaxespad=0.0,
        )
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"{title_prefix} t={t_idx}")
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"mode_overlay_scatter_t{t_idx:03d}_{title_prefix}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return phi_c
def compute_abs_levels_over_sequence(
    W_modes,           # (N,r,2) torch/np : 모드 (복소 분리)
    fields_seq,        # list of (N,2) torch/np : 각 t의 full-field (GT 권장)
    perc=(0.3, 0.6, 0.9),
    phi_seq_override=None  # None이면 LS-phi, 아니면 길이 T의 (r,2) / (r,) 시퀀스
):
    """
    전 시간대에 걸쳐 각 모드 k의 |W_k(x)*phi_k(t)| 최대값 M_k를 구하고
    절대 레벨 행렬 levels_abs[k, j] = perc[j] * M_k 를 반환.
    """
    W_np  = to_numpy(W_modes)
    W_c   = to_complex_xy(W_np)        # (N,r) complex
    r     = W_c.shape[1]

    M = np.zeros(r, dtype=np.float64)  # 각 모드의 전역 최대

    for t, field_full_xy in enumerate(fields_seq):
        u_np = to_numpy(field_full_xy)
        u_c  = to_complex_xy(u_np)     # (N,) complex

        if phi_seq_override is None:
            phi_t = ls_phi_from_fullgrid(W_c, field_full_xy)         # (r,)
        else:
            phi_raw = to_numpy(phi_seq_override[t])
            phi_t   = to_complex_xy(phi_raw) if (phi_raw.ndim==2 and phi_raw.shape[-1]==2) else np.asarray(phi_raw).reshape(-1)

        for k in range(r):
            mag_k = np.abs(W_c[:, k] * phi_t[k])          # (N,)
            M[k]  = max(M[k], float(mag_k.max()))

    # 절대(고정) 레벨 테이블 (r, L)
    levels_abs = np.stack([np.asarray(perc) * mk for mk in M], axis=0)
    return levels_abs  # shape: (r, len(perc))
def l2_norm_cols(Z: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(Z, axis=0, keepdims=True) + 1e-12
    return Z / nrm

def hungarian_or_greedy_for_max(M: np.ndarray):
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

def match_modes_by_corr(W_hat: np.ndarray, W_true: np.ndarray):
    """열-정규화 후 |<w_hat_i, w_true_j>| 최대화 매칭 + 위상 추정"""
    A = l2_norm_cols(W_hat); B = l2_norm_cols(W_true)
    C = np.abs(A.conj().T @ B)  # (r_hat, r_true)
    row, col = hungarian_or_greedy_for_max(C)
    phases, cosines = [], []
    for i, j in zip(row, col):
        inner = (A[:, i].conj() * B[:, j]).sum()
        phases.append(np.angle(inner)); cosines.append(np.abs(inner))
    return col, np.array(phases), np.array(cosines)  # perm: learned->true
def to_complex_xy(arr_xy):   # (...,2) -> complex (...)
    return arr_xy[..., 0] + 1j * arr_xy[..., 1]

def np_complex_to_torch_xy(arr_c, device):
    # arr_c: (N, r) complex (np.ndarray)
    return torch.stack([
        torch.as_tensor(np.real(arr_c), device=device, dtype=torch.float32),
        torch.as_tensor(np.imag(arr_c), device=device, dtype=torch.float32),
    ], dim=-1)  # (N, r, 2)
# --- full-grid LS로 φ 복원:  φ = pinv(W) y ---
def ls_phi_from_fullgrid(W_hat_full_c: np.ndarray, y_full_xy: torch.Tensor) -> np.ndarray:
    """
    W_hat_full_c : (N, r) complex   # 학습 모드(Full grid)
    y_full_xy    : (N, 2) torch     # [real, imag] full-grid field (GT 또는 예측)
    return       : (r,) complex     # LS 복원된 φ
    """
    y_c = (y_full_xy[:, 0].detach().cpu().numpy()
           + 1j * y_full_xy[:, 1].detach().cpu().numpy())  # (N,)
    phi_ls = np.linalg.pinv(W_hat_full_c) @ y_c            # (r,)
    return phi_ls

def match_modes_by_corr(W_hat: np.ndarray, W_true: np.ndarray):
    """열-정규화 후 |<w_hat_i, w_true_j>| 최대화 매칭 + 위상 추정"""
    A = l2_norm_cols(W_hat); B = l2_norm_cols(W_true)
    C = np.abs(A.conj().T @ B)  # (r_hat, r_true)
    row, col = hungarian_or_greedy_for_max(C)
    phases, cosines = [], []
    for i, j in zip(row, col):
        inner = (A[:, i].conj() * B[:, j]).sum()
        phases.append(np.angle(inner)); cosines.append(np.abs(inner))
    return col, np.array(phases), np.array(cosines)  # perm: learned->true

def compute_mode_error_by_perm(W_hat: np.ndarray, W_true: np.ndarray, perm: np.ndarray):
    """주어진 permutation을 사용하여 모드 에러(phases, cosines) 계산"""
    A = l2_norm_cols(W_hat); B = l2_norm_cols(W_true)
    phases, cosines = [], []
    for i, j in enumerate(perm):
        inner = (A[:, i].conj() * B[:, j]).sum()
        phases.append(np.angle(inner)); cosines.append(np.abs(inner))
    return np.array(phases), np.array(cosines)

def align_phi_over_time_for_lambda(phi_seq: np.ndarray):
    # permutation만 정렬, phase 회전은 하지 않음
    phi_seq = np.asarray(phi_seq).copy()
    Tm1, r = phi_seq.shape
    for t in range(1, Tm1):
        a = phi_seq[t-1]; b = phi_seq[t]
        # 크기 기반 유사도 (phase 불변)
        C = np.abs(np.outer(a.conj(), b))  # (r,r)
        row, col = hungarian_or_greedy_for_max(C)
        phi_seq[t] = b[col]  # 회전 X
    return phi_seq

def branch_correct(lam_hat_phi: np.ndarray, lam_true: np.ndarray, dt_eff: float):
    """허수부 branch: ω_hat → ω_hat + 2πk/Δt 로 GT ω에 가장 가깝게 이동"""
    out = lam_hat_phi.copy()
    for i in range(len(lam_true)):
        w_t = np.imag(lam_true[i])
        w_h = np.imag(out[i])
        k = np.round((w_t - w_h) * dt_eff / (2*np.pi))   # 정수
        out[i] = np.real(out[i]) + 1j*(w_h + 2*np.pi*k/dt_eff)
    return out

def match_err(lh, lt):
    Lh = lh.reshape(-1, 1); Lt = lt.reshape(1, -1)
    cost = np.abs(Lh - Lt)
    r_idx, t_idx = hungarian_or_greedy_for_max(-cost)
    err = cost[r_idx, t_idx]
    # Construct mapping: perm[i] gives the index in lt that matches lh[i]
    perm = np.zeros(len(lh), dtype=int)
    for i, j in zip(r_idx, t_idx):
        perm[i] = j
    return err, perm

def fit_linear_A(phi_seq_aligned: np.ndarray) -> np.ndarray:
    """phi_{t+1} ≈ A phi_t  (최소제곱)"""
    Phi0 = phi_seq_aligned[:-1].T    # (r, T-2)
    Phi1 = phi_seq_aligned[1:].T     # (r, T-2)
    return Phi1 @ np.linalg.pinv(Phi0)

def diag_consistency(A: np.ndarray, lam_hat: np.ndarray, dt_eff: float):
    """A와 diag(exp(lam*Δt)) 비교"""
    A_ideal = np.diag(np.exp(lam_hat * dt_eff))
    offdiag = A - np.diag(np.diag(A))
    off_E = float(np.linalg.norm(offdiag, 'fro')**2)
    rel_err = float(np.linalg.norm(A - A_ideal, 'fro') / (np.linalg.norm(A, 'fro') + 1e-12))
    return off_E, rel_err
def plot_A_heatmap(out_dir, A):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    im = ax.imshow(np.abs(A), origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("|A| heatmap")
    ax.set_xlabel("from"); ax.set_ylabel("to")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "A_heatmap.png"), dpi=150)
    plt.close(fig)
def plot_lambda_compare(out_dir, lam_true, lam_enc_med, lam_phi_bc):
    """α/ω 막대 비교 + 복소평면 산점도"""
    os.makedirs(out_dir, exist_ok=True)
    def _split(lv):
        return np.real(lv), np.imag(lv)
    labels = [f"mode{i}" for i in range(len(lam_true))]
    x = np.arange(len(labels))

    # (1) alpha/omega bar
    fig, ax = plt.subplots(2, 1, figsize=(4, 3), sharex=True)
    W = 0.25
    for a, name, off in [
        (_split(lam_true)[0], "GT α", -W),
        # (_split(lam_enc_med)[0], "Enc α", 0.0),
        (_split(lam_phi_bc)[0], "φ α", W)
    ]:
        ax[0].bar(x+off, a, width=W, label=name)
    for w, name, off in [
        (_split(lam_true)[1], "GT ω", -W),
        # (_split(lam_enc_med)[1], "Enc ω", 0.0),
        (_split(lam_phi_bc)[1], "φ ω", W)
    ]:
        ax[1].bar(x+off, w, width=W, label=name)
    ax[0].set_ylabel("alpha (real)")
    ax[1].set_ylabel("omega (imag)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[0].legend(ncol=2, fontsize=9)
    ax[1].legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lambda_compare_bars.png"), dpi=150)
    plt.close(fig)

    # (2) complex plane
    fig, ax = plt.subplots(figsize=(4,3))
    ax.scatter(np.real(lam_true),    np.imag(lam_true),    marker='x', s=80, label="GT")
    # ax.scatter(np.real(lam_enc_med), np.imag(lam_enc_med), marker='o', s=60, label="Enc med")
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