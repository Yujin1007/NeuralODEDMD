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


def reparameterize_full(mu_u, cov_u, jitter=1e-6):
    # mu_u: (m, 2), cov_u: (2m, 2m)
    m = mu_u.shape[0]
    D = m * 2  # 2m
    
    # flatten
    mu_flat = mu_u.reshape(D)          # (2m,)
    
    # 대칭화 & jitter로 안정성 확보
    cov_u = 0.5 * (cov_u + cov_u.T) + jitter * torch.eye(D, device=cov_u.device, dtype=cov_u.dtype)
    
    # Cholesky factorization
    L = torch.linalg.cholesky(cov_u)   # (2m, 2m)
    
    # 표준정규 샘플
    eps = torch.randn(D, device=mu_u.device, dtype=mu_u.dtype)  # (2m,)
    
    # 샘플링
    z_flat = mu_flat + L @ eps         # (2m,)
    
    # reshape back to (m,2)
    return z_flat.reshape(m, 2)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std