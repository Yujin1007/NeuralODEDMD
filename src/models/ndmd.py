import torch
import torch.nn as nn
import numpy as np

class TemporalOmegaMLP(nn.Module):
    def __init__(self, *, r_half: int, latent_dim: int = 16,
                 hidden_size: int = 64, layers: int = 2):
        super().__init__()
        self.r_half = r_half
        self.latent = nn.Parameter(torch.randn(latent_dim))
        # Build MLP
        seq = []
        in_dim = latent_dim
        for _ in range(layers):
            seq += [nn.Linear(in_dim, hidden_size), nn.ReLU()]
            in_dim = hidden_size
        seq += [nn.Linear(in_dim, 2 * r_half)]
        self.mlp = nn.Sequential(*seq)

    def forward(self):
        out = self.mlp(self.latent)  # Shape: (2 * r_half,)
        raw_alphas = out[:self.r_half]
        raw_thetas = out[self.r_half:2 * self.r_half]
        alphas = -2 * torch.sigmoid(raw_alphas)  # Ensures alphas in [-2, 0]
        thetas = torch.sigmoid(raw_thetas)       # Ensures thetas in (0, 1)
        return alphas, thetas

class TemporalBMLP(nn.Module):
    def __init__(self, *, r_half: int, latent_dim: int = 16,
                 hidden_size: int = 64, layers: int = 2):
        super().__init__()
        self.r_half = r_half
        self.latent = nn.Parameter(torch.randn(latent_dim))
        # Build MLP
        seq = []
        in_dim = latent_dim
        for _ in range(layers):
            seq += [nn.Linear(in_dim, hidden_size), nn.ReLU()]
            in_dim = hidden_size
        seq += [nn.Linear(in_dim, 1 + 2 * r_half)]
        self.mlp = nn.Sequential(*seq)

    def forward(self):
        out = self.mlp(self.latent)  # (1 + 2 * r_half,)
        b0 = out[0:1]
        raw = out[1:].view(self.r_half, 2)
        b_half = raw[:, 0] + 1j * raw[:, 1]  # Complex tensor
        return b0, b_half

class SinusoidalEncoding(nn.Module):
    def __init__(self, num_frequencies: int):
        super().__init__()
        self.num_frequencies = num_frequencies

    def forward(self, xy):
        # Simple sinusoidal encoding for 2D input
        freqs = torch.linspace(1.0, 2.0 ** self.num_frequencies, self.num_frequencies, device=xy.device)
        x = xy[:, 0:1] * freqs
        y = xy[:, 1:2] * freqs
        encoding = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=-1)
        return encoding

class LearnableFourierEncoding(nn.Module):
    def __init__(self, input_dim: int, num_frequencies: int):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.weights = nn.Parameter(torch.randn(input_dim, num_frequencies))

    def forward(self, xy):
        proj = xy @ self.weights
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class NeuralDMD(nn.Module):
    """Full Neural-DMD model (spatial network + temporal networks)."""
    def __init__(self, *, r: int, hidden_size: int = 256,
                 layers: int = 4, num_frequencies: int = 10,
                 use_learnable_encoding: bool = False,
                 temporal_latent_dim=32, temporal_hidden=64, temporal_layers=2):
        super().__init__()
        assert r % 2 == 0, "`r` must be even so modes pair with conjugates"
        self.r_half = r // 2
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(1e-3))

        # Positional encoding
        self.encoding = LearnableFourierEncoding(input_dim=2, num_frequencies=num_frequencies) \
            if use_learnable_encoding else SinusoidalEncoding(num_frequencies=num_frequencies)

        # Spatial MLP
        in_dim = 2 * (2 * num_frequencies + 1)
        seq = []
        for _ in range(layers):
            seq += [nn.Linear(in_dim, hidden_size), nn.SiLU()]
            in_dim = hidden_size
        seq += [nn.Linear(in_dim, 2 * self.r_half + 1)]
        self.mlp = nn.Sequential(*seq)

        # Temporal networks
        self.temporal_omega = TemporalOmegaMLP(r_half=self.r_half,
                                              latent_dim=temporal_latent_dim,
                                              hidden_size=temporal_hidden,
                                              layers=temporal_layers)
        self.temporal_b = TemporalBMLP(r_half=self.r_half,
                                       latent_dim=temporal_latent_dim,
                                       hidden_size=temporal_hidden,
                                       layers=temporal_layers)

    def spatial_forward(self, xy):
        enc = self.encoding(xy)
        out = self.mlp(enc)
        W0 = out[:, 0:1]
        real, imag = torch.split(out[:, 1:], 2, dim=-1)
        W_half = real + 1j * imag
        return W0, W_half

    def forward(self, xy: torch.Tensor):
        # Apply spatial_forward to each xy coordinate
        W0, W_half = self.spatial_forward(xy)
        alpha, theta = self.temporal_omega()
        Omega = alpha + 1j * theta
        b0, b_half = self.temporal_b()
        b = torch.cat([b_half, b0, b_half.conj()], dim=0)
        W = torch.cat([W_half, W0, W_half.conj()], dim=1)
        return W0, W_half, W, Omega, b