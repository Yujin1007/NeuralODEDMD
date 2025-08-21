from dataclasses import dataclass

@dataclass
class TrainConfig:
    device: str = "cuda"
    r: int = 4
    hidden_dim: int = 64
    num_epochs: int = 100_00
    lr: float = 5e-4
    l1_weight: float = 1e-3
    mode_sparsity_weight: float = 1e-3
    save_dir: str = "results/run1"
    print_every: int = 100
    ode_steps: int = 10
    process_noise: float = 1e-5
    cov_eps: float = 1e-6
    seed: int = 42