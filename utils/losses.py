import math
import torch
import torch.nn as nn

def gaussian_nll_loss(y, mu, logvar):
    return 0.5 * (logvar + (y - mu).pow(2) / logvar.exp() + math.log(2 * math.pi)).mean()

_nll = nn.GaussianNLLLoss(eps=1e-6, reduction='mean')

def cdmd_loss(mu_u, logvar_u, y_next, mu_phi, logvar_phi, lambda_param, W, *,
              l1_weight: float, mode_sparsity_weight: float):
    var_u = torch.exp(logvar_u)
    recon = _nll(mu_u, y_next, var_u)
    kl_phi = -0.5 * torch.sum(1 + logvar_phi - mu_phi.pow(2) - logvar_phi.exp())
    l1_lambda = l1_weight * torch.mean(torch.abs(lambda_param))
    mode_mags = torch.sqrt(W[..., 0].pow(2) + W[..., 1].pow(2))  # (m,r)
    sparsity = mode_sparsity_weight * mode_mags.mean(dim=0).sum()
    loss = recon + 0.001 * kl_phi + l1_lambda + sparsity
    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite loss detected: recon={recon}, kl={kl_phi}, l1={l1_lambda}, sp={sparsity}")
    return loss, {"recon": recon.item(), "kl": (0.001*kl_phi).item(), "l1": l1_lambda.item(), "sp": sparsity.item()}
