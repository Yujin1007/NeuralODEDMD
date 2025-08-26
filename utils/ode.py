import torch
from torch.autograd.functional import jacobian

# PyTorch 2.x: functorch 통합 (torch.func)
try:
    from torch.func import jvp, vmap, jacfwd
    _HAS_TORCH_FUNC = True
except Exception:
    from torch.autograd.functional import jacobian as _jacobian
    _HAS_TORCH_FUNC = False

def ode_euler(func, phi0, lambda_param, t_start: float, t_end: float, steps: int):
    dt = (t_end - t_start) / steps
    phi = phi0
    t = t_start
    for _ in range(steps):
        dphi = func(phi, lambda_param, t)
        phi = phi + dt * dphi
        t += dt
    return phi



def ode_euler_uncertainty(
    func,
    mu0: torch.Tensor,        # (B,*S) or (*S)  — (*S)==(r,2) 등 비배치도 허용
    logvar0: torch.Tensor,    # (B,*S) or (*S)  — diagonal only
    lambda_param,             # (B,...) or (...)  (브로드캐스트 가능)
    t_start,                  # (B,) or scalar
    t_end,                    # (B,) or scalar
    *,
    process_noise: float = 1e-5,
    cov_eps: float = 1e-6,
    backprop_mean: bool = True,
    basic_dt: float = 0.1,
    max_step: int = 200,
    smallN_thresh: int = 64,          # N<=이면 jacrev+vmap 경로
    covariance_update_every: int = 4, # k스텝마다 분산 업데이트(자코비안) 수행
    reuse_J_steps: int = 2,           # 계산한 J를 다음 s스텝 재사용(0이면 매번 새로계산)
    compute_diagJ: bool = True        # 1차항(2 dt var ⊙ diagJ) 포함 여부
):
    """
    Batched Euler + fast diagonal variance propagation.
    • 비배치 입력(*S)도 자동 배치(B=1)로 변환 후 처리, 마지막에 다시 squeeze.
    """
    device = mu0.device
    dtype  = mu0.dtype

    # --- 비배치 입력 자동 래핑 (B=1) ---
    unbatched = (mu0.dim() == 2)  # 보통 (*S)==(r,2)
    if unbatched:
        mu0      = mu0.unsqueeze(0)          # (1,*S)
        logvar0  = logvar0.unsqueeze(0)      # (1,*S)
        # lambda_param이 (r,2)... 처럼 배치축 없으면 붙여줌
        if torch.is_tensor(lambda_param) and (lambda_param.dim() >= 1) and (lambda_param.shape[0] != 1):
            lambda_param = lambda_param.unsqueeze(0)  # (1, ...)
        # t도 (1,)로
        t_start = torch.as_tensor(t_start, device=device, dtype=dtype).reshape(1)
        t_end   = torch.as_tensor(t_end,   device=device, dtype=dtype).reshape(1)

    B = mu0.shape[0]
    state_shape = mu0.shape[1:]
    N = int(torch.tensor(state_shape, device=device).prod().item()) if state_shape else 1

    # ---- states/vars (분산 경로는 그래프 분리) ----
    mu_flat = mu0.reshape(B, -1).contiguous()                               # (B,N)
    var     = torch.exp(logvar0.detach()).reshape(B, -1).contiguous().clone()  # (B,N)

    # ---- time grid ----
    def _to_B_vec(x):
        if torch.is_tensor(x):
            x = x.to(device=device, dtype=dtype)
            return x.reshape(-1) if x.dim() > 0 else x.reshape(1).expand(B)
        else:
            return torch.tensor([x], device=device, dtype=dtype).expand(B)

    t_start = _to_B_vec(t_start)   # (B,)
    t_end   = _to_B_vec(t_end)     # (B,)

    dT = (t_end - t_start)                                                  # (B,)
    dt_cap = torch.as_tensor(basic_dt, device=device, dtype=dtype)
    n  = torch.ceil(torch.clamp(torch.abs(dT) / dt_cap, min=1.0)).to(torch.long)
    n  = torch.clamp(n, max=max_step)                                       # (B,)
    dt = dT / n.clamp(min=1)                                                # (B,)
    t  = t_start.clone()                                                    # (B,)
    n_max = int(n.max().item())

    # ---- torch.func helpers ----
    has_func = hasattr(torch, "func")
    jacrev   = getattr(torch.func, "jacrev", None)
    vmap     = getattr(torch.func, "vmap",   None)

    # ---- batched drift ----
    def f_batch(mu_flat_in, lam, tt):
        # mu_flat_in: (B,N), lam: (B,...) or broadcastable, tt: (B,)
        phi = mu_flat_in.reshape((B,) + state_shape)             # (B,*S)
        return func(phi, lam, tt).reshape(B, -1)                 # (B,N)

    # ---- J 캐시 (재사용/업데이트 간격) ----
    J_prev         = None                    # (B,N,N)
    J_prev_valid_s = -1

    for s in range(n_max):
        alive = (s < n)                                                         # (B,)
        if not torch.any(alive): break
        dt_eff = (alive.to(dtype) * dt).unsqueeze(1)                            # (B,1)

        # ---- mean update ----
        if backprop_mean:
            dphi = f_batch(mu_flat, lambda_param, t)                            # (B,N)
            mu_flat = mu_flat + dt_eff * dphi
            t = t + dt_eff.squeeze(1)
        else:
            with torch.no_grad():
                dphi = f_batch(mu_flat, lambda_param, t)
                mu_flat = mu_flat + dt_eff * dphi
                t = t + dt_eff.squeeze(1)

        # ---- variance update scheduling ----
        need_cov_update = (s % max(1, covariance_update_every) == 0)
        can_reuse_J     = (J_prev is not None) and (s - J_prev_valid_s <= max(0, reuse_J_steps))

        # 항상 process_noise는 누적
        with torch.no_grad():
            var = torch.clamp(var + dt_eff * process_noise, min=cov_eps)

        if not need_cov_update and can_reuse_J:
            # 캐시된 J로 빠른 보정
            with torch.no_grad():
                idx = torch.nonzero(alive, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    dt_a  = dt[idx].unsqueeze(1)                                # (Ba,1)
                    var_a = var[idx]                                           # (Ba,N)
                    J_a   = J_prev[idx]                                        # (Ba,N,N)
                    if compute_diagJ:
                        diagJ = torch.diagonal(J_a, dim1=-2, dim2=-1)          # (Ba,N)
                        var_a = var_a + 2.0 * dt_a * (var_a * diagJ)
                    JSJT_diag = torch.matmul(J_a.pow(2), var_a.unsqueeze(-1)).squeeze(-1)
                    var_new   = var_a + (dt_a ** 2) * JSJT_diag
                    var[idx]  = torch.clamp(var_new, min=cov_eps)
            continue

        # ---- 여기서만 J 계산 (업데이트 시점) ----
        with torch.no_grad():
            idx = torch.nonzero(alive, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue

        if (N <= smallN_thresh) and has_func and (jacrev is not None) and (vmap is not None):
            # vmap(jacrev)로 alive 한 번에
            mu_a  = mu_flat[idx]                                              # (Ba,N)
            var_a = var[idx]                                                  # (Ba,N)
            # lambda_param 브로드캐스트 정리
            if torch.is_tensor(lambda_param) and lambda_param.shape[:1] == (B,):
                lam_a = lambda_param[idx]
            else:
                lam_a = lambda_param
            t_a   = t[idx]

            def f_single(mu_i, lam_i, t_i):
                x = mu_i.reshape((1,) + state_shape)                           # (1,*S)
                out = func(x, lam_i, t_i.reshape(()))                          # (1,*S)
                return out.reshape(-1)                                         # (N,)

            J_a = torch.func.vmap(torch.func.jacrev(f_single))(mu_a, lam_a, t_a)  # (Ba,N,N)

            # 캐시 갱신
            if (J_prev is None) or (J_prev.shape != (B, N, N)):
                J_prev = torch.zeros((B, N, N), device=device, dtype=dtype)
            J_prev.index_copy_(0, idx, J_a)
            J_prev_valid_s = s

            with torch.no_grad():
                dt_a = dt[idx].unsqueeze(1)
                if compute_diagJ:
                    diagJ = torch.diagonal(J_a, dim1=-2, dim2=-1)             # (Ba,N)
                    var_a = var_a + 2.0 * dt_a * (var_a * diagJ)
                JSJT_diag = torch.matmul(J_a.pow(2), var_a.unsqueeze(-1)).squeeze(-1)
                var_new   = var_a + (dt_a ** 2) * JSJT_diag
                var[idx]  = torch.clamp(var_new, min=cov_eps)

        else:
            # per-sample 경로 (alive만 loop; vectorize=True)
            J_list = []
            for i in idx.tolist():
                mu_i  = mu_flat[i].detach()                                    # (N,)
                var_i = var[i]
                # lambda_param 단일/배치 모두 대응
                lam_i = lambda_param[i] if (torch.is_tensor(lambda_param) and lambda_param.shape[:1] == (B,)) else lambda_param
                t_i   = t[i]
                dt_i  = dt[i]

                def f_flat_i(x_flat):
                    x = x_flat.reshape((1,) + state_shape)
                    out = func(x, lam_i, t_i)                                   # (1,*S)
                    return out.reshape(-1)                                      # (N,)

                J_i = torch.autograd.functional.jacobian(
                    f_flat_i, mu_i, create_graph=False, vectorize=True
                )                                                                # (N,N)
                J_list.append(J_i.unsqueeze(0))

                with torch.no_grad():
                    var_i_new = var_i
                    if compute_diagJ:
                        diagJ = torch.diagonal(J_i)                              # (N,)
                        var_i_new = var_i_new + 2.0 * dt_i * (var_i_new * diagJ)
                    JSJT_diag = torch.matmul(J_i.pow(2), var_i_new)              # (N,)
                    var_i_new = var_i_new + (dt_i ** 2) * JSJT_diag
                    var[i]    = torch.clamp(var_i_new, min=cov_eps)

            if len(J_list) > 0:
                J_a = torch.cat(J_list, dim=0)                                   # (Ba,N,N)
                if (J_prev is None) or (J_prev.shape != (B, N, N)):
                    J_prev = torch.zeros((B, N, N), device=device, dtype=dtype)
                J_prev.index_copy_(0, idx, J_a)
                J_prev_valid_s = s

    # ---- 출력 복원 (비배치였다면 squeeze) ----
    mu_next = mu_flat.reshape((B,) + state_shape)
    cov_out = torch.diag_embed(var)                                            # (B,N,N)
    if unbatched:
        mu_next = mu_next.squeeze(0)           # (*S)
        cov_out = cov_out.squeeze(0)           # (N,N)
    return mu_next, cov_out