import os
import math
import torch
import torch.fft as fft
import numpy as np
import xarray as xr
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.interpolate as interp
import matplotlib.cm as cm
# from torch_cfd.grids import Grid
# from torch_cfd.initial_conditions import filtered_vorticity_field
# from torch_cfd.spectral import *
from utils.utils import make_rfftmesh, vorticity_to_velocity, reconstruct_uv_from_normalized_vorticity
# --- (파일 상단 유틸 아래에 추가) ---
def choose_netcdf_engine_and_encoding():
    """
    netCDF 엔진 및 인코딩을 자동 선택:
    - netCDF4가 설치되어 있으면: engine='netcdf4' + zlib 압축 사용
    - 아니면: engine=None (SciPy backend) + 압축 인코딩 제거
    """
    try:
        import netCDF4  # noqa: F401
        engine = "netcdf4"
        encoding = {name: {"zlib": True, "complevel": 4}
                    for name in ["u", "v", "vorticity"]}
    except Exception:
        engine = None   # SciPy backend (압축 인코딩 미지원)
        encoding = None
    return engine, encoding
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


def compute_particle_trajectory(u_b, v_b, x_vals, y_vals, t_vals, dt=0.01, start_pos=(-0.5, 0)):
    """
    u_b, v_b : (T, nx, ny)
    x_vals, y_vals : 1D arrays
    t_vals : time array
    dt : integration step
    start_pos : initial particle position
    """
    traj_segments = []
    segment = [start_pos]
    pos = np.array(start_pos, dtype=float)

    nx, ny = len(x_vals), len(y_vals)
    xmin, xmax = x_vals.min(), x_vals.max()
    ymin, ymax = y_vals.min(), y_vals.max()
    bound = (xmin, xmax, ymin, ymax)
    u_b = np.array(u_b.cpu(), copy=True)  # ✅ 강제 복사 + numpy
    v_b = np.array(v_b.cpu(), copy=True)
    # 시간에 따라 velocity interpolation 함수 생성
    interp_u = [interp.RegularGridInterpolator((x_vals, y_vals), u_b[i], bounds_error=False, fill_value=None)
                for i in range(len(t_vals))]
    interp_v = [interp.RegularGridInterpolator((x_vals, y_vals), v_b[i], bounds_error=False, fill_value=None)
                for i in range(len(t_vals))]
    for ti in range(len(t_vals)-1):
        n_steps = 10 # int((t_vals[ti+1] - t_vals[ti]) / dt)
        for _ in range(n_steps):
        # for _ in t_vals[0]:
            u = interp_u[ti](pos)
            v = interp_v[ti](pos)
            pos[0] += dt * u
            pos[1] += dt * v
            segment.append(pos.copy())

            # 범위 벗어나면 trajectory 저장 후 초기화
            if pos[0] < xmin or pos[0] > xmax or pos[1] < ymin or pos[1] > ymax:
                traj_segments.append(np.array(segment))
                pos = np.array(start_pos, dtype=float)
                segment = [pos.copy()]
                break

    if len(segment) > 1:
        traj_segments.append(np.array(segment))

    return traj_segments, bound

def save_trajectory_plot(traj_segments, out_dir="./dataset/navier_stokes_flow", fname="trajectory_plot.png", bound=None):
    plt.figure(figsize=(6,6))
    for seg in traj_segments:
        plt.plot(seg[:,0], seg[:,1], '-', lw=1)
        plt.plot(seg[0,0], seg[0,1], 'go', markersize=3)
        plt.plot(seg[-1,0], seg[-1,1], 'ro', markersize=3)
    if bound is not None:
        plt.xlim(bound[0], bound[1])
        plt.ylim(bound[2], bound[3])
    
    # plt.ylim(-1,1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Particle Trajectories')
    plt.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Trajectory plot saved: {os.path.join(out_dir, fname)}")

def save_trajectories_plot(trajectories, out_dir="./dataset/navier_stokes_flow",
                         fname="trajectory_plot.png", bound=None, pred_label=None):
    """
    trajectories: list of list
        예: [ [traj_0_seg_0, traj_0_seg_1], [traj_1_seg_0], [traj_2_seg_0, traj_2_seg_1, ...] ]
    bound: [xmin, xmax] 또는 [xmin, xmax, ymin, ymax]
    """
    plt.figure(figsize=(7, 7))
    # colors = cm.get_cmap('tab10', len(trajectories))  # dataset별 고유 색상
    
    label_added = set()  # legend 중복 방지
    # idx = 0  # trajectory index (전체)
    pred_idx = len(trajectories)
    if pred_label is not None:
        pred_idx = len(trajectories) - len(pred_label) 
    for exp_i, traj_list in enumerate(trajectories):
        # color = colors(exp_i)
        
        label = f"Dataset {exp_i+1}"
        for seg in traj_list:
            x, y = seg[:, 0], seg[:, 1]
            if exp_i < pred_idx:
                if label not in label_added:
                    plt.plot(x, y, '-', color='gray', lw=1.5, label=label)
                    label_added.add(label)
                else:
                    plt.plot(x, y, '-', color='gray', lw=1.0)
            else:
                plt.plot(x, y, '-', color='red', lw=1.0, label=pred_label[exp_i-pred_idx])
                label_added.add(pred_label[exp_i-pred_idx])
            # trajectory 번호 표시 (중앙 혹은 마지막 위치)
            # cx, cy = x[len(x)//2], y[len(y)//2]
            # plt.text(cx, cy, f"{idx}", fontsize=7, color=color, ha='center', va='center')
            # idx += 1
    # if pred_label is not None:
    #     plt.plot(x, y, '-', color='red', lw=1.0, label=pred_label)
    #     label_added.add(pred_label)
    # 범위 설정
    if len(bound) == 2:
        xmin, xmax = bound
        ymin, ymax = bound
    elif len(bound) == 4:
        xmin, xmax, ymin, ymax = bound
    # else:
    #     xmin, xmax, ymin, ymax = -1, 1, -1, 1

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Particle Trajectories from 3 Simulations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"✅ Combined trajectory plot saved: {path}")
# -------------------------------------------------------
# 2️⃣ Simulation + GIF + NetCDF
# -------------------------------------------------------

# -------------------------------------------------------
# 3️⃣ Gray–Scott Compatible Loader
# -------------------------------------------------------
def load_spectral_nc_as_grayscott_compatible(
    nc_path: str,
    sample_ratio: float = 0.2,
    normalize_t: bool = False,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
):
    np.random.seed(seed)
    ds = xr.open_dataset(nc_path)

    vort = ds["vorticity"].transpose("time", "x", "y").values  # (T, n, n)
    tvals = ds["time"].values
    xvals = ds["x"].values
    yvals = ds["y"].values

    coords_full = make_coords_full_from_linspace(xvals, yvals)
    n = coords_full.shape[0]
    m = int(n * sample_ratio)
    idx = np.random.choice(n, size=m, replace=False)

    y_list, y_list_full, coords_list = [], [], []
    for k in range(vort.shape[0]):
        flat = vort[k].reshape(-1)
        y_list_full.append(flat.copy())
        y_list.append(flat[idx])
        coords_list.append(coords_full[idx])

    # Normalize time
    if normalize_t:
        t_list = [np.float32(t / tvals[-1]) for t in tvals]
    else:
        t_list = [np.float32(t) for t in tvals]

    coords_torch = [torch.from_numpy(c).float().to(device) for c in coords_list]
    coords_full_torch = torch.from_numpy(coords_full).float().to(device)
    y_torch = to_torch_split_real_only(y_list, device)
    y_full_torch = to_torch_split_real_only(y_list_full, device)

    return t_list, coords_torch, y_torch, y_full_torch, coords_full_torch

# -------------------------------------------------------
# 4️⃣ Dataset Class
# -------------------------------------------------------
class VorticityDataset(torch.utils.data.Dataset):
    def __init__(self, t_list, coords_list, y_list):
        self.t_list = t_list
        self.coords_list = coords_list
        self.y_list = y_list
        self.length = len(t_list) - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        t_prev = self.t_list[idx]
        t_next = self.t_list[idx + 1]
        coords = self.coords_list[idx + 1]
        y_prev = self.y_list[idx]
        y_next = self.y_list[idx + 1]
        return t_prev, t_next, coords, y_next, y_prev
def merge_datasets(out_dir: str, num_traj: int, save_name: str = "dataset_merged.nc"):
    datasets = []

    for i in range(num_traj):
        dataset_name = f"dataset_{i}.nc"
        data_path = os.path.join(out_dir, dataset_name)

        if not os.path.exists(data_path):
            print(f"⚠️  Warning: {data_path} not found, skipping.")
            continue

        ds = xr.open_dataset(data_path)
        datasets.append(ds)

    if not datasets:
        raise ValueError("❌ No datasets were loaded. Check your paths or num_traj value.")

    # realization 차원으로 합치기
    ds_merged = xr.concat(datasets, dim="realization")

    # realization 좌표 추가 (0, 1, 2, ...)
    ds_merged = ds_merged.assign_coords(realization=("realization", list(range(len(datasets)))))

    # 저장
    merged_path = os.path.join(out_dir, save_name)
    ds_merged.to_netcdf(merged_path)

    print(f"✅ Merged {len(datasets)} datasets into {merged_path}")
    print(f"   → Dimensions: {ds_merged.dims}")

    return ds_merged
# -------------------------------------------------------
# 5️⃣ Example Usage
# -------------------------------------------------------
if __name__ == "__main__":
    out_dir = "./results/navier_stokes_stochastic/run2/teacher_forcing_reconstruction"
    # data_path = "./results/navier_stokes_stochastic/run2/teacher_forcing_reconstruction/predicted_vorticity.nc"
    data_path_raw = "/home/yk826/projects/torch-cfd/dataset/navier_stokes_flow/multiple_traj2/raw_dataset_0.nc"
        
    trajectories = []
    diam = 1
    ndim = 64
    ds_raw = xr.open_dataset(data_path_raw)
    x_vals = ds_raw["x"].values
    y_vals = ds_raw["y"].values
    t_vals = ds_raw["time"].values
    init_pos = (sum(x_vals)/len(x_vals), sum(y_vals)/len(y_vals))
    dt = 0.02
    
    # rfftmesh = make_rfftmesh(n=ndim, diam=diam)
    # (3) call your function
    # u, v = vorticity_to_velocity(vorticity, rfftmesh)
    for i in range(10):
        data_path = f"/home/yk826/projects/torch-cfd/dataset/navier_stokes_flow/multiple_traj2/dataset_{i}.nc"
        ds = xr.open_dataset(data_path)
    
        vorticity = ds["vorticity"].values  # (T, nx, ny)
        u,v = reconstruct_uv_from_normalized_vorticity(vorticity, fmin=-4, fmax=4.5)
        # define particle dt ( make it  faster to see evident divergence )
        
        trajectory, bound = compute_particle_trajectory(u, v, x_vals, y_vals, t_vals, dt=dt, start_pos=init_pos)
        trajectories.append(trajectory)
    # np.save(os.path.join(out_dir, "particle_trajectory.npy"), trajectories)
        # print("💾 Saved trajectory data (.npy)")

    # merge_datasets(out_dir, num_traj=1)
    pred_data_path = "./results/navier_stokes_stochastic/run2/teacher_forcing_reconstruction/predicted_vorticity.nc"
    ds_pred = xr.open_dataset(pred_data_path)
    vorticity_pred = ds_pred["vorticity"].values
    u_pred, v_pred = reconstruct_uv_from_normalized_vorticity(vorticity_pred, fmin=-4, fmax=4.5)
    trajectory_pred, _ = compute_particle_trajectory(u_pred, v_pred, x_vals, y_vals, t_vals, dt=dt, start_pos=init_pos)
    trajectories.append(trajectory_pred)

    pred_data_path = "./results/navier_stokes_stochastic/run2/autoreg_reconstruction/predicted_vorticity.nc"
    ds_pred = xr.open_dataset(pred_data_path)
    vorticity_pred = ds_pred["vorticity"].values
    u_pred, v_pred = reconstruct_uv_from_normalized_vorticity(vorticity_pred, fmin=-4, fmax=4.5)
    trajectory_pred, _ = compute_particle_trajectory(u_pred, v_pred, x_vals, y_vals, t_vals, dt=dt, start_pos=init_pos)
    trajectories.append(trajectory_pred)
    save_trajectories_plot(trajectories, out_dir=out_dir, fname=f"trajectory_plot_original.png", bound=bound, pred_label=["teacher_forcing", "autoregressive"])
    