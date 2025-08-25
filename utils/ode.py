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
    mu0,
    logvar0,
    lambda_param,
    t_start: float,
    t_end: float,
    steps: int,
    process_noise: float = 1e-5,
    cov_eps: float = 1e-6,
):
    # dt = (t_end - t_start) / steps
    dt = 1/20
    steps = int((t_end - t_start) / dt)
    mu_flat = mu0.flatten()
    var_flat = torch.exp(logvar0).flatten()
    cov = torch.diag(var_flat).clone()  # (2r,2r)
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    t = t_start
    for _ in range(steps):
        phi_mu = mu_flat.reshape(mu0.shape).requires_grad_(True)

        def func_flat(p):
            return func(p.reshape(mu0.shape), lambda_param, t).flatten()

        dphi = func(phi_mu, lambda_param, t)
        f_flat = dphi.detach().flatten()
        J = jacobian(func_flat, phi_mu.flatten()).detach()

        mu_flat = mu_flat + dt * f_flat
        cov = cov + dt * (J @ cov + cov @ J.T) + dt * process_noise * eye
        # Symmetrize + PSD nudge
        cov = (cov + cov.T) / 2 + cov_eps * eye
        t += dt

    mu_next = mu_flat.reshape(mu0.shape)
    return mu_next, cov




def ode_euler_uncertainty2(
    func,
    mu0,
    logvar0,
    lambda_param,
    t_start: float,
    t_end: float,
    process_noise: float = 1e-5,
    cov_eps: float = 1e-6,
    # ↓ 추가 옵션 (기본값은 저비용 대각 전파)
    mode: str = "diag",            # "diag" | "full"
    backprop_mean: bool = True,
    backprop_cov: bool = False,    # 공분산 경로는 기본 no-grad로 가볍게
    basic_dt: float = 0.1,
    max_step: int = 200
):
    """
    Drop-in 대체 버전:
      - mode="diag": 공분산의 대각만 전파 (빠름, 기본)
      - mode="full": A Σ A^T + Qd 이산화로 full-cov 전파 (정확, 다소 느림)
    반환값은 (mu_next, cov) 동일. diag 모드도 cov는 (2r,2r)의 대각 행렬로 반환.
    """
    steps = min(max_step, int(abs((t_end - t_start) / basic_dt)))
    dt = (t_end - t_start) / max(1, steps)
    mu_flat = mu0.flatten()
    var = torch.exp(logvar0).flatten()  # (2r,)
    device, dtype = mu_flat.device, mu_flat.dtype
    n = mu_flat.numel()
    eye = torch.eye(n, device=device, dtype=dtype)

    t = t_start
    for _ in range(steps):
        # ---- 평균 업데이트 (역전파 유지/차단 선택) ----
        if backprop_mean:
            phi_mu = mu_flat.reshape(mu0.shape)
            dphi = func(phi_mu, lambda_param, t).flatten()
            mu_flat = mu_flat + dt * dphi
        else:
            with torch.no_grad():
                phi_mu = mu_flat.reshape(mu0.shape)
                dphi = func(phi_mu, lambda_param, t).flatten()
                mu_flat = mu_flat + dt * dphi

        # ---- 공분산(또는 분산) 업데이트 ----
        # f_flat(x) = func(x_reshaped, lam, t).flatten()
        def f_flat(x_flat):
            x = x_flat.reshape(mu0.shape)
            return func(x, lambda_param, t).flatten()

        if mode == "diag":
            # Σ = diag(var)만 유지. A = I + dt*J.
            # diag(A Σ A^T) = rowwise_sum( (A V) ∘ (A V) ),
            #   where V = diag(sqrt(var)) (각 열이 가중 단위벡터)
            with torch.no_grad() if not backprop_cov else torch.enable_grad():
                sqrt_var = torch.sqrt(torch.clamp(var, min=cov_eps))
                # V_cols: (n, n) 각 열이 sqrt(var) 단위기저
                # 메모리: n^2. n=2r이므로 r<=128 규모면 충분히 경량.
                V_cols = torch.diag(sqrt_var)  # (n, n)

                if _HAS_TORCH_FUNC:
                    # vmap으로 각 열에 대해 jvp(f, mu; v) 병렬 평가
                    cols = V_cols.t()  # (n, n) => 각 row가 한 column 벡터
                    def _single(v_col):
                        # jvp returns (f(mu), J v_col)
                        _, Jv = jvp(f_flat, (mu_flat,), (v_col,))
                        return v_col + dt * Jv  # A v_col
                    AV_cols = vmap(_single)(cols)  # (n, n)
                    AV = AV_cols.t()  # (n, n) 열 방면으로 정렬
                else:
                    # torch.func 없으면 보수적으로 전체 J 계산 (fallback)
                    J = _jacobian(f_flat, mu_flat, create_graph=False)  # (n, n)
                    AV = V_cols + dt * (J @ V_cols)

                # diag(A Σ A^T) = sum_j AV[:, j]^2
                var = (AV * AV).sum(dim=1) + dt * process_noise
                var = torch.clamp(var, min=cov_eps)

        elif mode == "full":
            # Σ를 full로 유지: A = I + dt*J, Σ_next = A Σ A^T + dt*Q
            # 초기 Σ = diag(var)
            # (첫 스텝만 full Σ 구성, 이후부터 full 전파)
            if not isinstance(var, torch.Tensor) or var.dim() == 1:
                cov = torch.diag(var).clone()
            else:
                cov = var  # 이미 full-cov일 수 있음 (재호출 시)

            if _HAS_TORCH_FUNC:
                with torch.no_grad() if not backprop_cov else torch.enable_grad():
                    J = jacfwd(f_flat)(mu_flat)  # (n, n)
                    A = eye + dt * J
                    cov = A @ cov @ A.T + dt * process_noise * eye
                    cov = (cov + cov.T) / 2 + cov_eps * eye
            else:
                # fallback: autograd.functional.jacobian
                with torch.no_grad() if not backprop_cov else torch.enable_grad():
                    J = _jacobian(f_flat, mu_flat, create_graph=False)  # (n, n)
                    A = eye + dt * J
                    cov = A @ cov @ A.T + dt * process_noise * eye
                    cov = (cov + cov.T) / 2 + cov_eps * eye

            var = cov  # 다음 루프에서 그대로 사용

        else:
            raise ValueError("mode must be 'diag' or 'full'.")

        t += dt

    mu_next = mu_flat.reshape(mu0.shape)
    if mode == "diag":
        cov_out = torch.diag(var)  # (n, n) 대각행렬 반환 (기존 타입과 동일)
    else:  # "full"
        cov_out = var  # full-cov 텐서

    return mu_next, cov_out