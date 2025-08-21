import os
import torch
from torch.autograd.functional import jacobian

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def complex_block_matrix(W: torch.Tensor) -> torch.Tensor:
    """Builds real-valued (2m x 2r) matrix from W[...,2] (m x r x 2)."""
    m, r, _ = W.shape
    Wr, Wi = W[..., 0], W[..., 1]
    M = torch.zeros(2 * m, 2 * r, device=W.device, dtype=W.dtype)
    M[0::2, 0:r] = Wr
    M[0::2, r:2 * r] = -Wi
    M[1::2, 0:r] = Wi
    M[1::2, r:2 * r] = Wr
    return M

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
    dt = (t_end - t_start) / steps
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