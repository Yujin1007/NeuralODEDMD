import os
import torch


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

import torch

def reparameterize_full(mu_u: torch.Tensor,
                        cov_u: torch.Tensor,
                        jitter_init: float = 1e-6,
                        jitter_max: float = 1e-2,
                        eps_min_eig: float = 1e-12):
    """
    mu_u:  (..., m, 2)
    cov_u: (..., 2m, 2m)
    return: sample with shape (..., m, 2)
    """
    # ---- shapes ----
    *batch, m, two = mu_u.shape
    assert two == 2, "mu_u 마지막 차원은 2(real, imag)여야 합니다."
    D = m * 2

    # ---- flatten batch dims ----
    Btot = int(torch.tensor(batch).prod().item()) if len(batch) > 0 else 1
    mu_flat = mu_u.reshape(Btot, D)                           # (Btot, D)

    # 대칭화 + 안정화
    cov = 0.5 * (cov_u + cov_u.transpose(-1, -2))            # (..., D, D)
    cov = cov.reshape(Btot, D, D)                             # (Btot, D, D)

    I = torch.eye(D, device=cov.device, dtype=cov.dtype).expand(Btot, D, D)
    jitter = jitter_init

    # ---- try batched Cholesky with escalating jitter ----
    while jitter <= jitter_max:
        try:
            L = torch.linalg.cholesky(cov + jitter * I)       # (Btot, D, D)
            eps = torch.randn_like(mu_flat)                   # (Btot, D)
            # (Btot,1,D) @ (Btot,D,D) -> (Btot,1,D) -> (Btot,D)
            inc = torch.bmm(eps.unsqueeze(1), L.transpose(-1, -2)).squeeze(1)
            z_flat = mu_flat + inc                            # (Btot, D)
            return z_flat.reshape(*batch, m, 2)
        except RuntimeError:
            jitter *= 10.0  # increase jitter and retry

    # ---- fallback: PSD via eigen decomposition (clips tiny negatives) ----
    evals, evecs = torch.linalg.eigh(cov)                     # (Btot, D), (Btot, D, D)
    evals = torch.clamp(evals, min=eps_min_eig)
    sqrt_evals = torch.sqrt(evals)                            # (Btot, D)
    # L_psd = Q diag(sqrt(evals)) Q^T
    L_psd = evecs @ (sqrt_evals.unsqueeze(-2) * evecs.transpose(-1, -2))  # (Btot, D, D)

    eps = torch.randn_like(mu_flat)                           # (Btot, D)
    inc = torch.bmm(eps.unsqueeze(1), L_psd.transpose(-1, -2)).squeeze(1) # (Btot, D)
    z_flat = mu_flat + inc                                    # (Btot, D)

    # 디버그가 필요하면 아래 프린트 유지하세요
    # print(f"z_flat : {z_flat.shape}, mu_flat :{mu_flat.shape}, L_psd : {L_psd.shape}")

    return z_flat.reshape(*batch, m, 2)

# def reparameterize_full(mu_u: torch.Tensor,
#                                cov_u: torch.Tensor,
#                                jitter_init: float = 1e-6,
#                                jitter_max: float = 1e-2,
#                                eps_min_eig: float = 1e-12):
#     """
#     mu_u:  (..., m, 2)
#     cov_u: (..., 2m, 2m)
#     return: sample with shape (..., m, 2)
#     """
#     # ---- shapes ----
#     *batch, m, two = mu_u.shape
#     assert two == 2, "mu_u 마지막 차원은 2(real, imag)여야 합니다."
#     D = m * 2

#     mu_flat = mu_u.reshape(*batch, D)  # (..., 2m)

#     # ---- symmetrize + small jitter for numerical stability ----
#     cov = 0.5 * (cov_u + cov_u.transpose(-1, -2))

#     I = torch.eye(D, device=cov.device, dtype=cov.dtype)
#     jitter = jitter_init

#     # ---- try Cholesky with escalating jitter ----
#     while jitter <= jitter_max:
#         try:
#             L = torch.linalg.cholesky(cov + jitter * I)  # (..., D, D)
#             eps = torch.randn_like(mu_flat)              # (..., D)
#             z_flat = mu_flat + eps @ L.transpose(-1, -2)
#             return z_flat.reshape(*batch, m, 2)
#         except RuntimeError:
#             jitter *= 10.0  # increase jitter and retry

#     # ---- fallback: PSD via eigen decomposition (clips tiny negatives) ----
#     evals, evecs = torch.linalg.eigh(cov)               # (..., D), (..., D, D)
#     evals = torch.clamp(evals, min=eps_min_eig)
#     sqrt_evals = torch.sqrt(evals)                      # (..., D)
#     # sqrt(cov) = Q diag(sqrt(evals)) Q^T
#     L_psd = evecs @ (sqrt_evals.unsqueeze(-2) * evecs.transpose(-1, -2))
#     eps = torch.randn_like(mu_flat)                     # (..., D)
#     z_flat = mu_flat + eps @ L_psd.transpose(-1, -2)
#     print(f"z_flat : {z_flat.shape}, mu_flat :{mu_flat.shape}, L_psd : {L_psd.shape}")
#     return z_flat.reshape(*batch, m, 2)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std