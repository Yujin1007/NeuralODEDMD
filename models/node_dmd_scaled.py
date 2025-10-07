import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import complex_block_matrix, reparameterize
from utils.ode import ode_euler_uncertainty_batch  
from utils.ode import ode_euler_uncertainty
# ---------------------------
# 유틸: 배치/비배치 통일 헬퍼
# ---------------------------
def _ensure_batch(x, batch_dim=0):
    """입력이 비배치라면 batch 차원(=0)을 추가하고, 배치 여부(bool)도 함께 반환."""
    if x is None:
        return None, False
    if isinstance(x, (float, int)):
        x = torch.tensor([x], dtype=torch.float32)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.dim() == 0:
        x = x[None]  # (1,)
        return x, False
    # 좌표/필드 텐서들: 보통 (m,2) 또는 (B,m,2)
    # 시각 텐서들: 보통 ()/(B,) 지원 → 여기서는 (B,) 로 통일
    return (x if x.size(0) > 1 or x.dim() >= 2 else x), (x.dim() >= 2 and x.size(0) > 1)

def _is_batched_coords(coords):
    # coords: (m,2) or (B,m,2)
    return coords.dim() == 3

# ---------------------------
# Mode extractor
# ---------------------------
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

    def forward(self, coords):
        """
        coords: (m,2) or (B,m,2)
        returns: (m,r,2) or (B,m,r,2)
        """
        if coords.dim() == 2:
            m = coords.size(0)
            out = self.net(coords)               # (m, r*2)
            return out.view(m, self.r, 2)        # (m, r, 2)
        elif coords.dim() == 3:
            B, m, _ = coords.shape
            x = coords.reshape(B * m, 2)         # (B*m, 2)
            out = self.net(x).view(B, m, self.r, 2)
            return out                           # (B, m, r, 2)
        else:
            raise ValueError("coords must be (m,2) or (B,m,2)")

# ---------------------------
# PhiEncoder
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.lambda_out = nn.Linear(hidden_dim, r * 2)  # 2 * r for raw_alphas and raw_thetas

    def forward(self, coords, y):
        """
        coords, y: (m,2) / (B,m,2)
        returns:
          mu:           (r,2)   or (B,r,2)
          logvar:       (r,2)   or (B,r,2)
          lambda_param: (r,2)   or (B,r,2), where each row is [alpha, theta]
        """
        if coords.dim() != y.dim():
            raise ValueError("coords and y must have the same rank")
        
        if coords.dim() == 2:
            # (m,2)
            x = torch.cat([coords, y], dim=-1)       # (m,4)
            h = self.embed(x)                        # (m,H)
            pooled = h.mean(dim=0)                   # (H,)
            pooled = F.relu(self.pool(pooled))       # (H,)
            mu = self.phi_mu(pooled).view(self.r, 2)
            logvar = self.phi_logvar(pooled).view(self.r, 2)
            # Lambda parameters
            raw_lambda = self.lambda_out(pooled)     # (2 * r,)
            raw_alphas = raw_lambda[:self.r]         # (r,)
            raw_thetas = raw_lambda[self.r:]         # (r,)
            alphas = -2 * torch.sigmoid(raw_alphas)  # (r,), [-2, 0]
            thetas = torch.sigmoid(raw_thetas)       # (r,), (0, 1)
            lambda_param = torch.stack([alphas, thetas], dim=-1)  # (r, 2)
            
            return mu, logvar, lambda_param
        elif coords.dim() == 3:
            # (B,m,2)
            B, m, _ = coords.shape
            x = torch.cat([coords, y], dim=-1)       # (B,m,4)
            h = self.embed(x.view(B * m, 4)).view(B, m, -1)  # (B,m,H)
            pooled = h.mean(dim=1)                   # (B,H)
            pooled = F.relu(self.pool(pooled))       # (B,H)
            mu = self.phi_mu(pooled).view(B, self.r, 2)
            logvar = self.phi_logvar(pooled).view(B, self.r, 2)
            # Lambda parameters
            raw_lambda = self.lambda_out(pooled)     # (B, 2 * r)
            raw_alphas = raw_lambda[:, :self.r]      # (B, r)
            raw_thetas = raw_lambda[:, self.r:]      # (B, r)
            alphas = -2 * torch.sigmoid(raw_alphas)  # (B, r), [-2, 0]
            thetas = torch.sigmoid(raw_thetas) * 5.       # (B, r), (0, 5)
            lambda_param = torch.stack([alphas, thetas], dim=-1)  # (B, r, 2)
            
            return mu, logvar, lambda_param
        else:
            raise ValueError("coords/y must be (m,2) or (B,m,2)")
        
class ODE():
    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    def forward(self, phi, lambda_param, t):

        lam_complex = lambda_param[..., 0] + 1j * lambda_param[..., 1]  # (B,r)
        phi_complex = phi[..., 0] + 1j * phi[..., 1]  # (B,r)
        drift_complex = lam_complex * phi_complex  # (B,r)
        drift = torch.stack([drift_complex.real, drift_complex.imag], dim=-1)  # (B,r,2)
        return drift

class ODENet(nn.Module):
    def __init__(self, r: int, hidden_dim: int):
        super().__init__()
        self.r = r
        self.net = nn.Sequential(
            nn.Linear(r * 2 + r * 2 + r * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, r * 2),
            nn.Tanh(), # for small correction
        )
    
    def forward(self, phi, lambda_param, dt):
        """
        phi:           (r,2) or (B,r,2)
        lambda_param:  (r,2) or (B,r,2)   (broadcastable to phi)
        t:             scalar or (B,)
        returns:       (r,2) or (B,r,2)
        """
        lam_complex = lambda_param[..., 0] + 1j * lambda_param[..., 1]  # (B,r)
        phi_complex = phi[..., 0] + 1j * phi[..., 1]  # (B,r)
        drift_complex = lam_complex * phi_complex * dt # (B,r)
        drift = torch.stack([drift_complex.real, drift_complex.imag], dim=-1)  # (B,r,2)
        # print(f"drift : ", drift.flatten().detach())
        if phi.dim() == 2:
            # 비배치
            r, two = phi.shape
            assert r == self.r and two == 2
            lam = lambda_param
            if lam.dim() == 2:
                lam = lam
            elif lam.dim() == 1:
                lam = lam.view(self.r, 2)
            else:
                lam = lam.expand_as(phi)

            x = torch.cat([phi.reshape(-1), lam.reshape(-1), drift.reshape(-1)], dim=0)  # (6r,)
            out = self.net(x)   
            correction = out.view(self.r, 2)
            dphi = drift + correction  # Residual structure
                                # (r*2,)
            return dphi/dt #  이거 작동 잘 되면 ODE 에서 dt 곱하는거 빼기

        elif phi.dim() == 3:
            # 배치
            B, r, two = phi.shape
            assert r == self.r and two == 2
            lam = lambda_param
            if lam.dim() == 2:
                lam = lam.unsqueeze(0).expand(B, -1, -1)        # (B,r,2)
            elif lam.dim() == 3:
                pass
            else:
                raise ValueError("lambda_param must be (r,2) or (B,r,2)")
            
            x = torch.cat([phi.reshape(B, -1), lam.reshape(B, -1), drift.reshape(B, -1)], dim=1)  # (B, 6r)
            correction = self.net(x).view(B, self.r, 2)
            dphi = drift + correction  # Residual structure
            return dphi / dt #  이거 작동 잘 되면 ODE 에서 dt 곱하는거 빼기
        else:
            raise ValueError("phi must be (r,2) or (B,r,2)")

# ---------------------------
# 메인 모듈
# ---------------------------
class Stochastic_NODE_DMD(nn.Module):
    def __init__(self, r: int, hidden_dim: int, ode_steps: int, process_noise: float, cov_eps: float, dt: float):
        super().__init__()
        self.r = r
        self.phi_net = PhiEncoder(r, hidden_dim)
        self.ode_func = ODENet(r, hidden_dim)
        # self.ode_func = ODE()
        self.mode_net = ModeExtractor(r, hidden_dim)
        self.ode_steps = ode_steps
        self.process_noise = process_noise
        self.cov_eps = cov_eps
        self.ode_dt = dt * 0.1 

    def _complex_block_matrix_batched(self, W):
        """
        W: (B,m,r,2) or (m,r,2)
        returns M: (B, 2m, 2r) or (2m, 2r)
        """
        if W.dim() == 3:
            # 비배치
            return complex_block_matrix(W)   # (2m,2r)
        elif W.dim() == 4:
            B, m, r, _ = W.shape
            # complex_block_matrix가 배치 미지원이면 per-sample 처리
            M_list = [complex_block_matrix(W[i]) for i in range(B)]  # each: (2m,2r)
            return torch.stack(M_list, dim=0)  # (B, 2m, 2r)
        else:
            raise ValueError("W must be (m,r,2) or (B,m,r,2)")

    def forward(model, coords, y_prev, t_prev, t_next):
        mu_phi, logvar_phi, lambda_param = model.phi_net(coords, y_prev)
        
        mu_phi_next, cov_phi_next = ode_euler_uncertainty(
                model.ode_func, mu_phi, logvar_phi, lambda_param, t_prev, t_next,
                process_noise=model.process_noise, cov_eps=model.cov_eps, basic_dt=model.ode_dt,)
        W = model.mode_net(coords)

        # print(f"mu_phi: {mu_phi.flatten()}, cov_phi: {logvar_phi.flatten()}, mu_phi_next: {mu_phi_next.flatten()}, cov_phi_next: {cov_phi_next.flatten()}, lambda_param: {lambda_param.flatten()}, W: {W.flatten()}")
        M = model._complex_block_matrix_batched(W) if coords.dim()==3 else complex_block_matrix(W)
        # mean
        # print(f"b :{mu_phi.flatten().detach()}, \nlambda : {lambda_param.flatten().detach()},\n W: {W.flatten().detach()}")
        if coords.dim() == 3:
            B, m, _ = coords.shape
            mu_phi_flat = mu_phi_next.reshape(B, -1)
            mu_u_flat = torch.matmul(M, mu_phi_flat.unsqueeze(-1)).squeeze(-1)
            mu_u = mu_u_flat.view(B, m, 2)
        else:
            m = coords.size(0)
            mu_phi_flat = mu_phi_next.reshape(-1)
            mu_u_flat = M @ mu_phi_flat
            mu_u = mu_u_flat.view(m, 2)
        # cov
        cov_u = M @ cov_phi_next @ M.transpose(-1, -2)
        if coords.dim() == 3:
            var_u = torch.clamp(torch.diagonal(cov_u, dim1=-2, dim2=-1), min=model.cov_eps)
            B, m, _ = coords.shape
            logvar_u = torch.log(var_u.view(B, m, 2))
        else:
            var_u = torch.clamp(torch.diagonal(cov_u), min=model.cov_eps).view(m, 2)
            logvar_u = torch.log(var_u)
        

        return mu_u, logvar_u, cov_u, mu_phi, logvar_phi, lambda_param, W
    
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
        # normalize to batched
        unbatched = coords.dim() == 2  # coords: (m,2), phi: (r,2)
        if unbatched:
            coords = coords.unsqueeze(0)    # (1,m,2)
            phi    = phi.unsqueeze(0)       # (1,r,2)

        B, m, _ = coords.shape
        # (B, r*2)
        phi_flat = phi.reshape(B, -1)
        # broadcast without allocation (view): (B, m, r*2)
        phi_exp = phi_flat.unsqueeze(1).expand(B, m, phi_flat.size(-1))
        # concat -> (B, m, 2 + r*2)
        x = torch.cat([coords, phi_exp], dim=-1)
        out = self.net(x)  # (B, m, 2)

        return out.squeeze(0) if unbatched else out

import time
def _tick():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

