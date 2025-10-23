import os
import math
import torch
import torch.fft as fft
import numpy as np
import xarray as xr
import imageio.v2 as imageio
import matplotlib.pyplot as plt
# --- (파일 상단 유틸 아래에 추가) ---

# -------------------------------------------------------
# 1️⃣ Utility Functions
# -------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def make_coords_full_from_linspace(x_vals, y_vals):
    X, Y = np.meshgrid(x_vals, y_vals, indexing="xy")
    coords_full = np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)
    return coords_full

def to_torch_split_real_only(lst, device):
    yr = [torch.from_numpy(np.array(y, dtype=np.float32)).to(device) for y in lst]
    yi = [torch.zeros_like(r) for r in yr]
    return [torch.stack([r, i], dim=-1) for r, i in zip(yr, yi)]

# -------------------------------------------------------
# 2️⃣ Simulation + GIF + NetCDF
# -------------------------------------------------------
def load_synth(
    nc_path: str,
    sample_ratio: float = 0.2,
    normalize_t: bool = False,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    data_len: int = None
):
    np.random.seed(seed)
    ds = xr.open_dataset(nc_path)
    print(ds.keys())
    vort = ds["vorticity"].transpose("realization", "time", "x", "y").values  # (R, T, n, n)
    n_realizations, T, nx, ny = vort.shape
    tvals = ds["time"].values
    xvals = ds["x"].values
    yvals = ds["y"].values

    coords_full = make_coords_full_from_linspace(xvals, yvals)
    n = coords_full.shape[0]
    m = int(n * sample_ratio)
    idx = np.random.choice(n, size=m, replace=False)

    y_list, y_list_full, coords_list = [], [], []
    for r in range(n_realizations):
        for k in range(T):
            flat = vort[r, k].reshape(-1)
            y_list_full.append(flat.copy())
            y_list.append(flat[idx])
            coords_list.append(coords_full[idx])

    # Normalize time
    if normalize_t:
        tvals_norm = tvals / tvals[-1]
    else:
        tvals_norm = tvals

    # realization 마다 같은 시간값 반복
    t_list = np.tile(tvals_norm, n_realizations).astype(np.float32)

    coords_torch = [torch.from_numpy(c).float().to(device) for c in coords_list]
    coords_full_torch = torch.from_numpy(coords_full).float().to(device)
    y_torch = to_torch_split_real_only(y_list, device)
    y_full_torch = to_torch_split_real_only(y_list_full, device)

    return t_list, coords_torch, y_torch, y_full_torch, coords_full_torch, n_realizations, T
# -------------------------------------------------------
# 4️⃣ Dataset Class
# -------------------------------------------------------
class SynthDataset(torch.utils.data.Dataset):
    def __init__(self, t_list, coords_list, y_list, n_realizations=1, steps_per_realization=None):
        """
        t_list: 전체 시점 리스트 (길이 R*T)
        coords_list: 좌표 리스트 (길이 R*T)
        y_list: 값 리스트 (길이 R*T)
        n_realizations: realization 개수
        steps_per_realization: realization 당 timestep 수
        """
        self.t_list = t_list
        self.coords_list = coords_list
        self.y_list = y_list
        self.n_realizations = n_realizations
        self.steps_per_realization = steps_per_realization or (len(t_list) // n_realizations)

        # 각 realization 내에서 마지막 step은 제외
        self.length = n_realizations * (self.steps_per_realization - 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # realization index 및 local timestep 계산
        r = idx // (self.steps_per_realization - 1)
        local_t = idx % (self.steps_per_realization - 1)

        # 전체 인덱스로 변환
        base_idx = r * self.steps_per_realization
        i0 = base_idx + local_t
        i1 = base_idx + local_t + 1

        t_prev = self.t_list[i0]
        t_next = self.t_list[i1]
        coords = self.coords_list[i1]
        y_prev = self.y_list[i0]
        y_next = self.y_list[i1]
        return t_prev, t_next, coords, y_next, y_prev
# -------------------------------------------------------
# 5️⃣ Example Usage
# -------------------------------------------------------
if __name__ == "__main__":
    
    data_path = "/home/yk826/projects/torch-cfd/dataset/navier_stokes_flow/multiple_traj/dataset_merged.nc"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_list, coords_list, y_list, y_full, coords_full, nR, T = load_synth(
        data_path, sample_ratio=0.1, normalize_t=True, device=device
    )

    dataset = SynthDataset(t_list, coords_list, y_list, n_realizations=nR, steps_per_realization=T)
    print(f"Dataset length: {len(dataset)}")
    print(f"t_list: {t_list}")
    tp, tn, c, yn, yp = dataset[0]
    print(f"Example shapes -> coords:{c.shape}, y_next:{yn.shape}, y_prev:{yp.shape}")