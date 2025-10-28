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
    mode_sparsity_weight: float = 1e-3# 1e-3
    kl_phi_weight: float = 1e-4 #0.001
    cons_weight: float = 0.15#0.1 
    recon_weight: float = 0.5
    save_dir: str = "results/stochastic/test_batch"
    print_every: int = 500
    ode_steps: int = 5
    process_noise: float = 1e-5
    cov_eps: float = 1e-6
    seed: int = 42
    data_len: int = 50
    eval_data_len: int = 100
    dt: float = 0.1  # dt for data generation 
    resolution: tuple = (32,32)  # Added resolution parameter
    train_mode: str = "teacher_forcing"  # "teacher_forcing" or "autoreg" or "evolve"
    normalize_t: bool = False


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

@dataclass
class Gray_Scott_Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    r: int = 8
    hidden_dim: int = 64
    num_epochs: int =30_00
    batch_size: int = 1
    lr: float = 5e-4
    l1_weight: float =0#1e-3 
    mode_sparsity_weight: float =0# 1e-3
    kl_phi_weight: float =1e-3
    cons_weight: float = 0.15
    recon_weight: float = 3.
    save_dir: str = "results/gray_scott/test"
    print_every: int = 500
    ode_steps: int = 5
    process_noise: float = 1e-5
    cov_eps: float = 1e-6
    seed: int = 42
    sample_ratio: float = 0.1
    sigma: float = 0.001
    data_len: int = 100
    eval_data_len: int = 200
    mode_frequency: int = 2
    phi_frequency: int = 2
    dt: float = 0.1  # dt for data generation 
    resolution: tuple = (100,100)  # Added resolution parameter
    train_mode: str = "teacher_forcing"  # "teacher_forcing" or "autoreg" or "evolve"
    normalize_t: bool = False

@dataclass
class Navier_Stokes_Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    r: int = 8
    hidden_dim: int = 128
    num_epochs: int =30_00
    batch_size: int = 1
    lr: float = 5e-4
    l1_weight: float =0#1e-3 
    mode_sparsity_weight: float =0# 1e-3
    kl_phi_weight: float =1e-3
    cons_weight: float = 0.15
    recon_weight: float = 3.
    save_dir: str = "results/navier_stokes/s10"
    print_every: int = 500
    ode_steps: int = 5
    process_noise: float = 1e-5
    cov_eps: float = 1e-6
    seed: int = 42
    sample_ratio: float = 0.1
    sigma: float = 0.001
    data_len: int = 50#150 #50
    eval_data_len: int = 99#199 #99
    mode_frequency: int = 2
    phi_frequency: int = 2
    dt: float = 0.1  # dt for data generation 
    train_mode: str = "teacher_forcing"  # "teacher_forcing" or "autoreg" or "evolve"
    normalize_t: bool = False
    # data_path: str = "/share/portal/yk826/physics_dataset/torch-cfd/dataset/navier_stokes_obstacle_flow/obstacle_flow.nc"
    data_path: str = "/share/portal/yk826/physics_dataset/torch-cfd/dataset/navier_stokes_flow/spectral_vorticity.nc"
    # data_path: str = "/share/portal/yk826/physics_dataset/torch-cfd/dataset/navier_stokes_flow/raw_dataset_slow.nc"
@dataclass
class Navier_Stokes_Stochastic_Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    r: int = 8
    hidden_dim: int = 128
    num_epochs: int =30_00
    batch_size: int = 1
    lr: float = 5e-4
    l1_weight: float =0#1e-3 
    mode_sparsity_weight: float =0# 1e-3
    kl_phi_weight: float =1e-3
    cons_weight: float = 0.15
    recon_weight: float = 3.
    save_dir: str = "results/navier_stokes_stochastic/run3"
    print_every: int = 500
    ode_steps: int = 5
    process_noise: float = 1e-5
    cov_eps: float = 1e-6
    seed: int = 42
    sample_ratio: float = 1
    sigma: float = 0.001
    data_len: int = 20
    eval_data_len: int = 20
    mode_frequency: int = 2
    phi_frequency: int = 2
    dt: float = 0.1  # dt for data generation 
    train_mode: str = "evolve"  # "teacher_forcing" or "autoreg" or "evolve"
    normalize_t: bool = False
    data_path: str = "/home/yk826/projects/torch-cfd/dataset/navier_stokes_flow/multiple_traj2/dataset_merged.nc"
    