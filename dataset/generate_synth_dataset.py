import torch
from typing import List
import numpy as np
from torch.utils.data import Dataset

# Dataset generation (modified for NumPy)
def make_grid(nx=32, ny=32):
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)
    return coords, (nx, ny)

def true_modes(coords):
    x = coords[:,0]; y = coords[:,1]
    m1 = np.sin(np.pi * (x+1)/2.0) * np.cos(np.pi * (y+1)/2.0)
    m2 = np.cos(np.pi * (x+1)) * np.sin(np.pi * (y+1))
    m3 = np.sin(2*np.pi * x) * np.sin(2*np.pi * y)
    m4 = np.ones_like(x)*0.5
    W = np.stack([m1, m2, m3, m4], axis=1).astype(np.complex64)
    return W

def synth_sequence(T=20, nx=32, ny=32, sample_ratio=0.1, sigma=0.02, seed=0):  # T reduced for speed
# def synth_sequence(T=20, nx=64, ny=64, sample_ratio=0.1, sigma=0.02, seed=0):  # T reduced for speed
    np.random.seed(seed)
    coords_full, shape = make_grid(nx, ny)
    n = coords_full.shape[0]
    r = 4
    W_full = true_modes(coords_full)

    # alpha = np.array([-0.1, -0.05, -0.2, 0.0])
    # omega = np.array([2.0, 4.0, 1.0, 0.0])
    # b = np.array([1.0+0.5j, 0.8-0.3j, 0.5+0.2j, 1.0+0.0j], dtype=np.complex64)
    alpha = np.array([-0.01, -0.05, -0.20, -0.01])  # m4에 약한 감쇠
    omega = np.array([ 2.00,  4.00,  1.00,  0.30])  # m4에 아주 약한 진동
    b = np.array([1.0+0.5j, 0.8-0.3j, 0.7+0.2j, 0.2+0.0j], dtype=np.complex64)  # 상수모드 초기계수 축소

    dt = 0.1 # to control the evolving speed. 
    t_list = list(range(T))
    coords_list = []
    y_list = []
    y_true_list = []
    y_true_full_list = []
    k = max(1, int(n * sample_ratio))
    idx = np.random.choice(n, size=k, replace=False) #Assumes observation is accessible only at fixed locations.
    for t in t_list:
        phi = np.exp((alpha + 1j*omega) * t * dt) * b
        I = W_full @ phi

        coords_t = coords_full[idx]
        y_t = I[idx]

        noise = sigma * (np.random.normal(size=y_t.shape) + 1j*np.random.normal(size=y_t.shape))
        y_t_noisy = (y_t + noise).astype(np.complex64)

        coords_list.append(coords_t)
        y_list.append(y_t_noisy)
        y_true_list.append(y_t)
        y_true_full_list.append(I)

    return t_list, coords_list, y_list, y_true_list, y_true_full_list, coords_full, (alpha, omega, b), W_full


def load_synth(device: torch.device, T=20):
    """Loads synthetic sequence and converts to torch (real/imag split).
    Returns:
        t_list: list[float]
        coords_list: list[Tensor[m,2]]
        y_list: list[Tensor[m,2]] (complex split)
        y_true_list, y_true_full_list, coords_full_t: analogous full-res
        gt_params, W_full: passthroughs from synth_sequence
    """
    (
        t_list,
        coords_list,
        y_list,
        y_true_list,
        y_true_full_list,
        coords_full,
        gt_params,
        W_full,
    ) = synth_sequence(T=T)

    def to_torch_split(lst: List[np.ndarray]):
        yr = [torch.from_numpy(np.real(y)).float().to(device) for y in lst]
        yi = [torch.from_numpy(np.imag(y)).float().to(device) for y in lst]
        return [torch.stack([r, i], dim=-1) for r, i in zip(yr, yi)]

    coords_torch = [torch.from_numpy(c).float().to(device) for c in coords_list]
    y_torch = to_torch_split(y_list)
    y_true_torch = to_torch_split(y_true_list)
    y_true_full_torch = to_torch_split(y_true_full_list)
    coords_full_torch = torch.from_numpy(coords_full).float().to(device)

    return (
        t_list,
        coords_torch,
        y_torch,
        y_true_torch,
        y_true_full_torch,
        coords_full_torch,
        gt_params,
        W_full,
    )



class SynthDataset(Dataset):
    def __init__(self, t_list, coords_list, y_list):
        """
        Custom dataset for synthetic sequence data.
        
        Args:
            t_list: List of time steps (float or int)
            coords_list: List of coordinate tensors [m, 2]
            y_list: List of observation tensors [m, 2] (real, imag)
        """
        self.t_list = t_list
        self.coords_list = coords_list
        self.y_list = y_list
        self.length = len(t_list) - 1  # Number of (t_prev, t_next) pairs

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns data for one time step transition.
        
        Returns:
            t_prev: Float, previous time step
            t_next: Float, next time step
            coords: Tensor[m, 2], coordinates at t_next
            y_prev: Tensor[m, 2], observation at t_prev
            y_next: Tensor[m, 2], observation at t_next
        """
        # t_prev = float(self.t_list[idx])
        t_next = float(self.t_list[idx + 1])
        coords = self.coords_list[idx + 1]
        # y_prev = self.y_list[idx]
        y_next = self.y_list[idx + 1]
        return t_next, coords, y_next