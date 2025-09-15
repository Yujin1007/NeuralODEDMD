from dataclasses import dataclass
import torch
@dataclass
class Stochastic_Node_DMD_Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    r: int = 4
    hidden_dim: int = 64
    num_epochs: int =30_00#100_00
    batch_size: int = 1
    lr: float = 5e-4
    l1_weight: float = 1e-3 #0.1#run18
    mode_sparsity_weight: float = 1e-4# 1e-3
    kl_phi_weight: float = 0.05 #0.001
    cons_weight: float = 0.1
    save_dir: str = "results/stochastic/run26"
    # save_dir: str = "results/ndmd/run1"
    print_every: int = 500
    ode_steps: int = 10
    process_noise: float = 1e-5
    cov_eps: float = 1e-6
    seed: int = 42
    data_len: int = 50
    eval_data_len: int = 100
    resolution: tuple = (32,32)  # Added resolution parameter


@dataclass
class Deterministic_Node_DMD_Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    r: int = 4
    hidden_dim: int = 64
    num_epochs: int = 100_00
    lr: float = 5e-4
    l1_weight: float = 1e-3
    save_dir: str = "results/deterministic/run2"
    print_every: int = 100
    ode_steps: int = 10
    seed: int = 42