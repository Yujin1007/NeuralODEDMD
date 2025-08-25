import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import complex_block_matrix, reparameterize
from utils.ode import ode_euler, ode_euler_uncertainty,ode_euler_uncertainty2
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
        mu_phi_next, cov_phi_next = ode_euler_uncertainty2(
            self.ode_func,
            mu_phi,
            logvar_phi,
            lambda_param,
            t_prev,
            t_next,
            # steps=self.ode_steps,
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
        # phi_next = ode_euler(self.ode_func, phi_prev, lambda_param, 0, t_next, self.ode_steps)
        phi_next = ode_euler(self.ode_func, phi_prev, lambda_param, t_prev, t_next, self.ode_steps)
        u_pred = self.decoder(coords, phi_next)
        return u_pred, mu, logvar, lambda_param
'''
'''
# # This module extracts mode information from the input coordinates.
# class ModeExtractor(nn.Module):
#     def __init__(self, r: int, hidden_dim: int):
#         super().__init__()
#         self.r = r
#         self.net = nn.Sequential(
#             nn.Linear(2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, r * 2),
#         )

#     def forward(self, coords):  # (m,2) or (batch_size,m,2) -> (m,r,2) or (batch_size,m,r,2)
#         is_batched = coords.dim() == 3
#         if is_batched:
#             batch_size, m, _ = coords.shape
#             coords_flat = coords.view(batch_size * m, 2)
#             out = self.net(coords_flat)
#             return out.view(batch_size, m, self.r, 2)
#         else:
#             out = self.net(coords)
#             return out.view(-1, self.r, 2)

# # This module outputs latent distribution phi and lambda parameter.
# class PhiEncoder(nn.Module):
#     def __init__(self, r: int, hidden_dim: int):
#         super().__init__()
#         self.r = r
#         self.embed = nn.Sequential(
#             nn.Linear(4, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#         )
#         self.pool = nn.Linear(hidden_dim, hidden_dim)
#         self.phi_mu = nn.Linear(hidden_dim, r * 2)
#         self.phi_logvar = nn.Linear(hidden_dim, r * 2)
#         self.lambda_out = nn.Linear(hidden_dim, r * 2)

#     def forward(self, coords, y):
#         is_batched = coords.dim() == 3
#         if is_batched:
#             batch_size, m, _ = coords.shape
#             if y.dim() == 2:  # If y is [m, 2], repeat to match batch
#                 y = y.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, m, 2]
#             elif y.dim() != 3 or y.shape[:2] != (batch_size, m):
#                 raise ValueError(f"Expected y to have shape [m, 2] or [{batch_size}, {m}, 2], got {y.shape}")
#             x = torch.cat([coords, y], dim=-1)  # [batch_size, m, 4]
#             x_flat = x.view(batch_size * m, 4)
#             h = self.embed(x_flat)  # [batch_size * m, hidden_dim]
#             h = h.view(batch_size, m, -1)  # [batch_size, m, hidden_dim]
#             pooled = F.relu(self.pool(h.mean(dim=1)))  # [batch_size, hidden_dim]
#             mu = self.phi_mu(pooled).view(batch_size, self.r, 2)  # [batch_size, r, 2]
#             logvar = self.phi_logvar(pooled).view(batch_size, self.r, 2)  # [batch_size, r, 2]
#             lambda_param = self.lambda_out(pooled).view(batch_size, self.r, 2)  # [batch_size, r, 2]
#             return mu, logvar, lambda_param
#         else:
#             if y.dim() != 2 or y.shape[0] != coords.shape[0]:
#                 raise ValueError(f"Expected y to have shape [{coords.shape[0]}, 2], got {y.shape}")
#             x = torch.cat([coords, y], dim=-1)  # [m, 4]
#             h = self.embed(x)  # [m, hidden_dim]
#             pooled = F.relu(self.pool(h.mean(dim=0)))  # [hidden_dim]
#             mu = self.phi_mu(pooled).view(self.r, 2)  # [r, 2]
#             logvar = self.phi_logvar(pooled).view(self.r, 2)  # [r, 2]
#             lambda_param = self.lambda_out(pooled).view(self.r, 2)  # [r, 2]
#             return mu, logvar, lambda_param

# # This module implements the ODE function.
# class ODENet(nn.Module):
#     def __init__(self, r: int, hidden_dim: int):
#         super().__init__()
#         self.r = r
#         self.net = nn.Sequential(
#             nn.Linear(r * 2 + r * 2 + 1, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, r * 2),
#         )

#     def forward(self, phi, lambda_param, t):
#         is_batched = phi.dim() == 3
#         if is_batched:
#             batch_size = phi.shape[0]
#             phi_f = phi.view(batch_size, -1)  # [batch_size, r*2]
#             lam_f = lambda_param.view(batch_size, -1)  # [batch_size, r*2]
#             if isinstance(t, torch.Tensor) and t.dim() == 1:
#                 t_tensor = t  # [batch_size]
#             else:
#                 t_tensor = torch.tensor([t] * batch_size, dtype=phi.dtype, device=phi.device)  # [batch_size]
#             x = torch.cat([phi_f, lam_f, t_tensor.unsqueeze(-1)], dim=-1)  # [batch_size, r*2 + r*2 + 1]
#             return self.net(x).view(batch_size, self.r, 2)  # [batch_size, r, 2]
#         else:
#             phi_f = phi.view(-1)  # [r*2]
#             lam_f = lambda_param.view(-1)  # [r*2]
#             t_tensor = torch.tensor([t], dtype=phi.dtype, device=phi.device)  # [1]
#             x = torch.cat([phi_f, lam_f, t_tensor], dim=0)  # [r*2 + r*2 + 1]
#             return self.net(x).view(self.r, 2)  # [r, 2]

# # This module gets latent phi to reconstruct u
# class ReconstructionDecoder(nn.Module):
#     def __init__(self, r: int, hidden_dim: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(2 + r*2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2)
#         )

#     def forward(self, coords, phi):
#         is_batched = coords.dim() == 3
#         if is_batched:
#             batch_size, m, _ = coords.shape
#             phi_flat = phi.view(batch_size, 1, -1)  # [batch_size, 1, r*2]
#             phi_exp = phi_flat.repeat(1, m, 1)  # [batch_size, m, r*2]
#             input_feat = torch.cat([coords, phi_exp], dim=-1)  # [batch_size, m, 2 + r*2]
#             return self.net(input_feat)  # [batch_size, m, 2]
#         else:
#             phi_flat = phi.view(1, -1)  # [1, r*2]
#             phi_exp = phi_flat.repeat(coords.shape[0], 1)  # [m, r*2]
#             input_feat = torch.cat([coords, phi_exp], dim=-1)  # [m, 2 + r*2]
#             return self.net(input_feat)  # [m, 2]

# class Stochastic_NODE_DMD(nn.Module):
#     def __init__(self, r: int, hidden_dim: int, ode_steps: int, process_noise: float, cov_eps: float):
#         super().__init__()
#         self.r = r
#         self.phi_net = PhiEncoder(r, hidden_dim)
#         self.ode_func = ODENet(r, hidden_dim)
#         self.mode_net = ModeExtractor(r, hidden_dim)
#         self.ode_steps = ode_steps
#         self.process_noise = process_noise
#         self.cov_eps = cov_eps

#     def forward(self, coords, y_prev, t_prev, t_next):
#         is_batched = coords.dim() == 3
#         mu_phi, logvar_phi, lambda_param = self.phi_net(coords, y_prev)
#         mu_phi_next, cov_phi_next = ode_euler_uncertainty2(
#             self.ode_func,
#             mu_phi,
#             logvar_phi,
#             lambda_param,
#             t_prev,
#             t_next,
#             process_noise=self.process_noise,
#             cov_eps=self.cov_eps,
#         )
#         W = self.mode_net(coords)
#         if is_batched:
#             batch_size = coords.shape[0]
#             M = complex_block_matrix(W.view(-1, self.r, 2))  # [batch_size*m, 2r]
#             mu_flat = mu_phi_next.view(batch_size, -1)  # [batch_size, r*2]
#             mu_u = torch.bmm(M.view(batch_size, -1, 2 * self.r), mu_flat.unsqueeze(-1)).view(batch_size, -1, 2)  # [batch_size, m, 2]
#             cov_u = torch.bmm(torch.bmm(M, cov_phi_next), M.transpose(-1, -2))  # [batch_size, 2m, 2m]
#             var_u = torch.clamp(cov_u.diagonal(dim1=-2, dim2=-1), min=self.cov_eps).view(batch_size, -1, 2)  # [batch_size, m, 2]
#             logvar_u = torch.log(var_u)
#             return mu_u, logvar_u, cov_u, mu_phi, logvar_phi, lambda_param, W
#         else:
#             M = complex_block_matrix(W)  # [2m, 2r]
#             mu_flat = mu_phi_next.flatten()  # [r*2]
#             mu_u = (M @ mu_flat).view(coords.shape[0], 2)  # [m, 2]
#             cov_u = M @ cov_phi_next @ M.T  # [2m, 2m]
#             var_u = torch.clamp(cov_u.diagonal(), min=self.cov_eps).view(coords.shape[0], 2)  # [m, 2]
#             logvar_u = torch.log(var_u)
#             return mu_u, logvar_u, cov_u, mu_phi, logvar_phi, lambda_param, W

# class NODE_DMD(nn.Module):
#     def __init__(self, r: int, hidden_dim: int, ode_steps: int):
#         super().__init__()
#         self.encoder = PhiEncoder(r, hidden_dim)
#         self.ode_func = ODENet(r, hidden_dim)
#         self.decoder = ReconstructionDecoder(r, hidden_dim)
#         self.ode_steps = ode_steps

#     def forward(self, coords, y_prev, t_prev, t_next):
#         mu, logvar, lambda_param = self.encoder(coords, y_prev)
#         phi_prev = reparameterize(mu, logvar)
#         phi_next = ode_euler(self.ode_func, phi_prev, lambda_param, t_prev, t_next, self.ode_steps)
#         u_pred = self.decoder(coords, phi_next)
#         return u_pred, mu, logvar, lambda_param