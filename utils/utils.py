import os
import torch
import math
import torch.fft as fft
import numpy as np
import xarray as xr
from torch.fft import rfft2, irfft2

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

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

@torch.no_grad()
def eval_uncertainty_metrics(y_obs, mu, logvar, y_true=None, eps=1e-12):
    var = torch.exp(logvar)
    std = torch.sqrt(torch.clamp(var, min=eps))
    res = y_obs - mu
    z   = res / std

    cov68 = (z.abs() <= 1.0).float().mean().item()
    cov95 = (z.abs() <= 2.0).float().mean().item()
    nll   = 0.5 * (torch.log(2*torch.pi*var) + (res**2)/torch.clamp(var, min=eps))
    out = {
        "coverage@68": cov68,
        "coverage@95": cov95,
        "z_mean": z.mean().item(),
        "z_std":  z.std(unbiased=True).item(),
        "z_abs_mean": z.abs().mean().item(),
        "NLL_mean": nll.mean().item(),
        "mean_pred_var": var.mean().item(),
        "mean_residual_sq": (res**2).mean().item(),
    }
    if y_true is not None:
        res_noise = y_obs - y_true
        z_noise   = res_noise / std
        out.update({
            "z_noise_mean": z_noise.mean().item(),
            "z_noise_std":  z_noise.std(unbiased=True).item(),
            "coverage_noise@68": (z_noise.abs() <= 1.0).float().mean().item(),
            "coverage_noise@95": (z_noise.abs() <= 2.0).float().mean().item(),
            "mean_emp_noise_var": (res_noise**2).mean().item(),
        })
    return out

def find_subset_indices(coords_full, coords_subset):
    # 좌표가 매 스텝 동일한 subset이면 한 번만 계산해서 재사용
    cf = coords_full.detach().cpu().numpy()
    cs = coords_subset.detach().cpu().numpy()
    mp = {(float(x), float(y)): i for i, (x, y) in enumerate(cf)}
    idx = [mp[(float(x), float(y))] for (x, y) in cs]
    return torch.as_tensor(idx, device=coords_full.device, dtype=torch.long)



def vorticity_to_velocity(pred_vorticity, rfftmesh):
    """
    pred_vorticity: (batch, nx, ny//2+1) in Fourier or real space
    rfftmesh: (kx, ky)
    """
    # Fourier transform if not already
    if isinstance(pred_vorticity, np.ndarray):
        pred_vorticity = torch.from_numpy(pred_vorticity)

    if torch.is_complex(pred_vorticity) == False:
        w_h = fft.rfft2(pred_vorticity)
    else:
        w_h = pred_vorticity

    kx, ky = rfftmesh
    lap = kx**2 + ky**2
    lap[..., 0, 0] = 1.0

    psi_h = -w_h / (4 * math.pi**2 * lap)
    u_hat = 2j * math.pi * ky * psi_h
    v_hat = -2j * math.pi * kx * psi_h

    u = fft.irfft2(u_hat).real
    v = fft.irfft2(v_hat).real
    return u, v


def make_rfftmesh(n=64, diam=1.0, device="cpu", dtype=torch.float64):
    """
    Create Fourier mesh for 2D rfft2 grid.
    diam: domain size (usually 1.0 or 2π)
    returns (kx, ky) of shape (n, n//2 + 1)
    """
    kx = torch.fft.fftfreq(n, d=diam/n, device=device, dtype=dtype)  # (n,)
    ky = torch.fft.fftfreq(n, d=diam/n, device=device, dtype=dtype)  # (n,)
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")

    # truncate for rfft (positive frequencies in y)
    kx = kx[:, : n//2 + 1]
    ky = ky[:, : n//2 + 1]
    return kx, ky



def reconstruct_uv_from_normalized_vorticity(vorticity_norm, fmin=-4, fmax=4.5, device="cuda:0"):
    """
    Given a cropped & normalized vorticity dataset (ds),
    reconstruct physical u,v fields (unnormalized) using Poisson inversion in Fourier space.
    
    Args:
        ds (xarray.Dataset): normalized dataset containing 'vorticity' (time, x, y)
        fmin, fmax (float): normalization bounds used during training
        device (str): 'cuda:0' or 'cpu'
    Returns:
        xr.Dataset: reconstructed u,v fields (same shape as ds)
    """
    # vorticity_norm = torch.tensor(ds["vorticity"].values, dtype=torch.float64, device=device)  # (T, nx, ny)

    # 1️⃣ Unnormalize vorticity
        # ✅ ensure tensor type
    if isinstance(vorticity_norm, np.ndarray):
        vorticity_norm = torch.tensor(vorticity_norm, dtype=torch.float64, device=device)
    elif not torch.is_tensor(vorticity_norm):
        raise TypeError(f"Unsupported type for vorticity: {type(vorticity_norm)}")
    vorticity = 0.5 * (vorticity_norm + 1) * (fmax - fmin) + fmin  # (T, nx, ny)
    
    # 2️⃣ FFT meshes
    nx, ny = vorticity.shape[1:]
    dx = dy = 2 * np.pi / nx  # domain scaling, consistent with original sim
    kx = torch.fft.fftfreq(nx, d=dx).to(device) * 2 * np.pi
    ky = torch.fft.rfftfreq(ny, d=dy).to(device) * 2 * np.pi
    kx, ky = torch.meshgrid(kx, ky, indexing='ij')
    k2 = kx**2 + ky**2
    k2[0, 0] = 1e-9  # avoid division by zero
    
    # 3️⃣ Spectral transform
    vort_hat = torch.fft.rfft2(torch.tensor(vorticity, dtype=torch.float64, device=device), dim=(-2, -1))  # (T, nx, ny//2+1)
    
    # 4️⃣ Compute streamfunction in Fourier space
    psi_hat = vort_hat / k2[None, :, :]
    
    # 5️⃣ Compute velocity components (u_hat, v_hat)
    u_hat = 1j * ky[None, :, :] * psi_hat
    v_hat = -1j * kx[None, :, :] * psi_hat
    
    # 6️⃣ Inverse FFT to get real velocity fields
    u = torch.fft.irfft2(u_hat, s=(nx, ny)).real
    v = torch.fft.irfft2(v_hat, s=(nx, ny)).real
    
    # # 7️⃣ Build xarray dataset
    # coords = dict(time=ds.time.values, x=ds.x.values, y=ds.y.values)
    # ds_reconstructed = xr.Dataset(
    #     data_vars=dict(
    #         u=(("time", "x", "y"), u.cpu().numpy()),
    #         v=(("time", "x", "y"), v.cpu().numpy()),
    #         vorticity=(("time", "x", "y"), vorticity.cpu().numpy()),
    #     ),
    #     coords=coords
    # )
    # return ds_reconstructed
    return u, v
