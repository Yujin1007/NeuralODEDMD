import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import complex_block_matrix, reparameterize
from utils.ode import ode_euler, ode_euler_uncertainty

# This module extracts mode information from the input coordinates.
class ModeExtractor(nn.Module):
    def __init__(self, r: int, hidden_dim: int):
        super().__init__()
        self.r = r
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, r * 2),
        )

    def forward(self, coords):  # (m,2) -> (m,r,2)
        out = self.net(coords)
        return out.view(-1, self.r, 2)

# This module outputs latent distribution phi and lambda parameter. 
class PhiEncoder(nn.Module):
    def __init__(self, r: int, hidden_dim: int):
        super().__init__()
        self.r = r
        self.embed = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pool = nn.Linear(hidden_dim, hidden_dim)
        self.phi_mu = nn.Linear(hidden_dim, r * 2)
        self.phi_logvar = nn.Linear(hidden_dim, r * 2)
        self.lambda_out = nn.Linear(hidden_dim, r * 2)

    def forward(self, coords, y):
        x = torch.cat([coords, y], dim=-1)
        h = self.embed(x)
        pooled = F.relu(self.pool(h.mean(dim=0)))
        mu = self.phi_mu(pooled).view(self.r, 2)
        logvar = self.phi_logvar(pooled).view(self.r, 2)
        lambda_param = self.lambda_out(pooled).view(self.r, 2)
        return mu, logvar, lambda_param

# This module implements the ODE function.
class ODENet(nn.Module):
    def __init__(self, r: int, hidden_dim: int):
        super().__init__()
        self.r = r
        self.net = nn.Sequential(
            nn.Linear(r * 2 + r * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, r * 2),
        )

    def forward(self, phi, lambda_param, t: float):
        phi_f = phi.view(-1)
        lam_f = lambda_param.view(-1)
        t_tensor = torch.tensor([t], dtype=phi.dtype, device=phi.device)
        x = torch.cat([phi_f, lam_f, t_tensor], dim=0)
        return self.net(x).view(self.r, 2)

# This module gets latent phi to reconstruct u
class ReconstructionDecoder(nn.Module):
    def __init__(self, r: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + r*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, coords, phi):
        phi_flat = phi.view(1, -1)  # (1, r*2)
        phi_exp = phi_flat.repeat(coords.shape[0], 1)  # (m, r*2)
        input_feat = torch.cat([coords, phi_exp], dim=-1)  # (m, 2 + r*2)
        return self.net(input_feat)

class Stochastic_NODE_DMD(nn.Module):
    def __init__(self, r: int, hidden_dim: int, ode_steps: int, process_noise: float, cov_eps: float):
        super().__init__()
        self.r = r
        self.phi_net = PhiEncoder(r, hidden_dim)
        self.ode_func = ODENet(r, hidden_dim)
        self.mode_net = ModeExtractor(r, hidden_dim)
        self.ode_steps = ode_steps
        self.process_noise = process_noise
        self.cov_eps = cov_eps

    def forward(self, coords, y_prev, t_prev: float, t_next: float):
        mu_phi, logvar_phi, lambda_param = self.phi_net(coords, y_prev)
        mu_phi_next, cov_phi_next = ode_euler_uncertainty(
            self.ode_func,
            mu_phi,
            logvar_phi,
            lambda_param,
            t_prev,
            t_next,
            steps=self.ode_steps,
            process_noise=self.process_noise,
            cov_eps=self.cov_eps,
        )
        W = self.mode_net(coords)
        M = complex_block_matrix(W)  # (2m,2r)
        mu_flat = mu_phi_next.flatten()
        mu_u = (M @ mu_flat).view(coords.shape[0], 2)
        cov_u = M @ cov_phi_next @ M.T
        var_u = torch.clamp(cov_u.diagonal(), min=self.cov_eps).view(coords.shape[0], 2)
        logvar_u = torch.log(var_u)
        return mu_u, logvar_u, cov_u, mu_phi, logvar_phi, lambda_param, W
    

class NODE_DMD(nn.Module):
    def __init__(self, r: int, hidden_dim: int, ode_steps: int):
        super().__init__()
        self.encoder = PhiEncoder(r, hidden_dim)
        self.ode_func = ODENet(r, hidden_dim)
        self.decoder = ReconstructionDecoder(r, hidden_dim)
        self.ode_steps = ode_steps

    def forward(self, coords, y_prev, t_prev, t_next):
        mu, logvar, lambda_param = self.encoder(coords, y_prev)  # Use previous time step
        phi_prev = reparameterize(mu, logvar)
        # phi_next = ode_euler(self.ode_func, phi_prev, lambda_param, t_prev, t_next)
        phi_next = ode_euler(self.ode_func, phi_prev, lambda_param, 0, t_next, self.ode_steps)
        u_pred = self.decoder(coords, phi_next)
        return u_pred, mu, logvar, lambda_param

