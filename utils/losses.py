import math
import torch
import torch.nn as nn
from torch.nn import functional as F
def gaussian_nll_loss(y, mu, logvar):
    return 0.5 * (logvar + (y - mu).pow(2) / logvar.exp() + math.log(2 * math.pi)).mean()

_nll = nn.GaussianNLLLoss(eps=1e-6, reduction='mean')

def stochastic_loss_fn(mu_u, logvar_u, y_next, mu_phi, logvar_phi, mu_phi_hat, logvar_phi_hat, lambda_param, W, *,
              recon_weight: float, l1_weight: float, mode_sparsity_weight: float, kl_phi_weight: float, cons_weight: float):
    var_u = torch.exp(logvar_u)
    recon = _nll(mu_u, y_next, var_u)
    kl_phi = -0.5 * torch.sum(1 + logvar_phi - mu_phi.pow(2) - logvar_phi.exp())
    l1_lambda = l1_weight * torch.mean(torch.abs(lambda_param))
    mode_mags = torch.sqrt(W[..., 0].pow(2) + W[..., 1].pow(2))  # (m,r)
    sparsity = mode_sparsity_weight * mode_mags.mean(dim=0).sum()
    cons_loss = consistency_loss(mu_phi_hat, logvar_phi_hat, mu_phi, logvar_phi, weight=cons_weight)
    loss = recon_weight * recon + kl_phi_weight * kl_phi + l1_lambda + sparsity + cons_loss
    # print(f"Loss components: recon={recon}, kl={kl_phi}, l1={l1_lambda}, sp={sparsity}, cons={cons_loss}")
    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite loss detected: recon={recon}, kl={kl_phi}, l1={l1_lambda}, sp={sparsity}, cons={cons_loss}\nlambda={lambda_param}")
    return loss, {"recon": recon.item(), "kl": (kl_phi_weight * kl_phi).item(), "l1": l1_lambda.item(), "sp": sparsity.item(), "cons": cons_loss.item()}


def consistency_loss(obs_mu_phi, obs_logvar_phi, pred_mu_phi, pred_logvar_phi, weight=1.0, kl_scale=0.001):
    # Mean consistency (MSE)
    mse = torch.mean((obs_mu_phi - pred_mu_phi)**2)
    
    
    # Variance consistency (KL divergence: observed ~ N(obs_mu, obs_var) || predicted ~ N(pred_mu, pred_var))
    obs_var = torch.exp(obs_logvar_phi)  # (r, 2) or (B, r, 2)
    pred_var = torch.exp(pred_logvar_phi)  # (r, 2) or (B, r, 2)
    
    # KL divergence
    kl = 0.5 * torch.mean(
        torch.log(pred_var / obs_var) + (obs_var + (obs_mu_phi - pred_mu_phi)**2) / pred_var - 1
    )
    if not torch.isfinite(weight * (mse + kl_scale * kl)):
        raise ValueError(f"Non-finite loss detected: obs_mu_phi={obs_mu_phi}, obs_logvar_phi={obs_logvar_phi}, pred_mu_phi={pred_mu_phi}, pred_logvar_phi={pred_logvar_phi}")
    return weight * (mse + kl_scale * kl)

def loss_fn(u_pred, y_next, mu, logvar, lambda_param, l1_weight: float):
    # u_pred, mu, logvar, lambda_param = model(coords, y_prev, t_prev, t_next)
    recon_loss = F.mse_loss(u_pred, y_next)  # Predict next step
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    l1_loss = l1_weight * torch.mean(torch.abs(lambda_param))
    return recon_loss + 0.001 * kl_loss + l1_loss

# from Neural DMD code 
def pixel_loss_fn(model, xy, frames, times, *, beta_tv=1e-2, beta_neg=1e-2, beta_sparse=1e-2,
                  frame_max=1.0, frame_min=0.0):
    """
    Returns (total_loss, aux_dict) for Weather / Pixel tutorial.
    """
    # Replace jax.vmap with a loop or torch.vmap if needed
    W0, W_half = model.spatial_forward(xy)  # W0: (N, 1), W_half: (N, r_half)
    al, th = model.temporal_omega()
    Omega = al + 1j * th  # Complex tensor: (r_half,)

    Lambda = torch.exp(Omega[:, None] * times[None, :] * 160.0)  # (r_half, T)
    # Compute reconstruction
    recon = 2 * torch.view_as_real(
        torch.einsum('br,rt,r->bt', W_half, Lambda, model.temporal_b()[1])
    )[..., 0] + W0[:, 0:1] * model.temporal_b()[0]  # (N, T)
    recon = recon * (frame_max - frame_min) + frame_min

    rec_loss = torch.mean((frames - recon) ** 2)
    loss = rec_loss
    # loss += beta_tv * tv_loss(recon.T[0])  # TV on first frame (commented out as in original)
    loss += beta_neg * negative_penalty(recon)
    sparsity_value = sparsity_loss(W0, W_half, *model.temporal_b())
    loss += beta_sparse * sparsity_value
    return loss, {"recon": rec_loss, "sparse": sparsity_value}

# ── generic helpers ────────────────────────────────────────────────────
def sparsity_loss(*arrays):
    """L1 sparsity on any number of tensors."""
    return sum(torch.mean(torch.abs(a)) for a in arrays)

def tv_loss(img):
    """Total-variation (pixel experiment)."""
    return torch.sum(torch.abs(img[:-1] - img[1:])) + \
           torch.sum(torch.abs(img[:, :-1] - img[:, 1:]))

def negative_penalty(x):
    """Penalize negative intensities."""
    return torch.sum(torch.relu(-x) ** 2)