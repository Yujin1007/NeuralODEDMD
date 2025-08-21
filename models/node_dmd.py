import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import complex_block_matrix
from utils.ode import ode_euler_uncertainty

class ModeNet(nn.Module):
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


class Encoder(nn.Module):
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


class ODEFunc(nn.Module):
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


class NODE_DMD(nn.Module):
    def __init__(self, r: int, hidden_dim: int, ode_steps: int, process_noise: float, cov_eps: float):
        super().__init__()
        self.r = r
        self.encoder = Encoder(r, hidden_dim)
        self.ode_func = ODEFunc(r, hidden_dim)
        self.mode_net = ModeNet(r, hidden_dim)
        self.ode_steps = ode_steps
        self.process_noise = process_noise
        self.cov_eps = cov_eps

    def forward(self, coords, y_prev, t_prev: float, t_next: float):
        mu, logvar, lambda_param = self.encoder(coords, y_prev)
        mu_phi_next, cov_phi_next = ode_euler_uncertainty(
            self.ode_func,
            mu,
            logvar,
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
        return mu_u, logvar_u, mu, logvar, lambda_param, W

