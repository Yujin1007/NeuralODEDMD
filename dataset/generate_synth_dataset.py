
from typing import List
import numpy as np
from torch.utils.data import Dataset
import torch
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

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def soft_box_mask(coords, center, half_sizes, edge_soft):
    """
    소프트한 직사각형 마스크.
    coords: (N,2), center=(cx,cy), half_sizes=(hx,hy), edge_soft=부드러움 스케일(좌표 단위)
    반환: (N,) in [0,1]
    """
    x, y = coords[:,0], coords[:,1]
    cx, cy = center
    hx, hy = half_sizes
    eps = max(1e-6, edge_soft)

    # inside-ness 를 시그모이드로 근사: (반폭 - 절대거리) / eps
    sx = _sigmoid((hx - np.abs(x - cx)) / eps)
    sy = _sigmoid((hy - np.abs(y - cy)) / eps)
    return (sx * sy).astype(np.float32)

def path_circle(t, dt, *, center=(0.0, 0.0), radius=0.4, ang_speed=1.0):
    """
    원형 경로: angle = ang_speed * t * dt
    """
    theta = ang_speed * t * dt
    return (center[0] + radius*np.cos(theta), center[1] + radius*np.sin(theta))

def path_polyline(t, dt, *, waypoints, period=None):
    """
    웨이포인트를 직선으로 잇는 폐곡선/개곡선 경로.
    - waypoints: [(x0,y0), (x1,y1), ...]
    - period: 하나의 라운드 트립에 대응하는 시간(초 단위 가정). None이면 전체 길이에 비례해 자동 1.0 사용.
    """
    pts = np.asarray(waypoints, dtype=np.float32)
    if pts.shape[0] < 2:
        return tuple(pts[0]) if pts.shape[0] == 1 else (0.0, 0.0)
    # 세그먼트 길이와 누적 길이
    segs = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(segs, axis=1)
    total = float(seg_len.sum())
    if total <= 1e-8:
        return tuple(pts[0])

    # 시간 → [0,1) 진행도 → 누적길이로 매핑
    T = period if (period is not None and period > 0) else 1.0
    s = ((t * dt) / T) % 1.0
    target_len = s * total

    # 어떤 세그먼트에 있는지 찾기
    acc = 0.0
    for i, L in enumerate(seg_len):
        if target_len <= acc + L or i == len(seg_len)-1:
            # 세그먼트 i에서의 비율 u
            u = 0.0 if L < 1e-8 else (target_len - acc) / L
            p = pts[i]   + u * segs[i]
            return (float(p[0]), float(p[1]))
        acc += L

    return tuple(pts[-1])

def _cells_to_halfsizes_and_edge(nx, ny, size_cells, edge_soft_cells):
    """
    격자 셀 개수 → 좌표계 반폭(hx,hy)와 가장자리 부드러움 eps 로 변환.
    좌표계는 [-1,1]이므로 셀 간격은 dx=2/(nx-1), dy=2/(ny-1).
    """
    dx = 2.0 / max(1, nx-1)
    dy = 2.0 / max(1, ny-1)
    w_cells, h_cells = size_cells
    w = w_cells * dx
    h = h_cells * dy
    hx, hy = 0.5 * w, 0.5 * h
    eps = 0.5 * edge_soft_cells * max(dx, dy)
    return (hx, hy), eps
# def synth_sequence(
#     T,
#     norm_T,
#     resolution,
#     sample_ratio=0.1,
#     sigma=0.1,
#     seed=0,
#     *,
#     # ▼▼ 추가된 옵션들 ▼▼
#     add_object=False,
#     obj_size_cells=(9, 9),         # (가로 셀 수, 세로 셀 수) 기본 2x3
#     edge_soft_cells=1.0,           # 가장자리 소프트닝(셀 단위)
#     obj_amp=0.8+0.0j,              # 물체의 복소 진폭(상수)
#     obj_phase_omega=0.0,           # 물체의 추가 위상진동 omega (rad/sec). 0이면 상수 위상.
#     path_fn="circle",              # "circle" 또는 "polyline" 또는 콜러블
#     path_kwargs=None               # 경로 함수에 전달할 kwargs
# ):
#     """
#     원래 모드 기반 신호 I 에, 선택적으로 '이동하는 소프트 박스' 물체를 가산합니다.
#     - 물체 크기는 격자 셀 개수로 지정(obj_size_cells).
#     - 경로는 원형/폴리라인 예시 또는 사용자 정의 함수(path_fn)로 지정.
#     """
#     np.random.seed(seed)
#     coords_full, shape = make_grid(resolution[0], resolution[1])
#     nx, ny = shape
#     n = coords_full.shape[0]

#     W_full = true_modes(coords_full)

#     # 원래 계수/진동
#     alpha = np.array([-0.01, -0.05, -0.20, -0.01])
#     omega = np.array([ 2.00,  4.00,  1.00,  0.30])
#     b = np.array([1.0+0.5j, 0.8-0.3j, 0.7+0.2j, 0.2+0.0j], dtype=np.complex64)

#     dt = 0.1
#     t_list = list(range(T))
    
#     coords_list = []
#     y_list = []
#     y_true_list = []
#     y_true_full_list = []

#     # 고정 관측 위치 (샘플링 마스크)
#     k = max(1, int(n * sample_ratio))
#     idx = np.random.choice(n, size=k, replace=False)

#     # 경로 설정
#     if path_kwargs is None:
#         path_kwargs = {}
#     if callable(path_fn):
#         get_center = lambda t: path_fn(t, dt, **path_kwargs)
#     elif path_fn == "circle":
#         # 기본 원형 예시: 중심 (0,0), 반지름 0.4, 각속도 1.0 rad/s
#         get_center = lambda t: path_circle(t, dt, **({"center": (0.0, 0.0), "radius": 0.4, "ang_speed": 1.0} | path_kwargs))
#     elif path_fn == "polyline":
#         # 기본 폴리라인 예시: 네 점을 잇는 경로, period=2.0초
#         default_wp = [(-0.6, -0.6), (0.6, -0.4), (0.4, 0.6), (-0.5, 0.5)]
#         defaults = {"waypoints": default_wp, "period": 2.0}
#         get_center = lambda t: path_polyline(t, dt, **(defaults | path_kwargs))
#     else:
#         # 인식 못하면 원점 고정
#         get_center = lambda t: (0.0, 0.0)

#     # 물체 크기 변환(좌표계 반폭 & 가장자리)
#     (hx, hy), edge_eps = _cells_to_halfsizes_and_edge(nx, ny, obj_size_cells, edge_soft_cells)

#     for t in t_list:
#         # 원래 모드 합성
#         phi = np.exp((alpha + 1j*omega) * t * dt) * b
#         I = W_full @ phi  # (N,) complex

#         # 이동 물체 가산 (override)
#         if add_object:
#             cx, cy = get_center(t)
#             mask = soft_box_mask(
#                 coords_full,
#                 center=(cx, cy),
#                 half_sizes=(hx, hy),
#                 edge_soft=edge_eps
#             )  # (N,)

#             # 물체 위치는 항상 override_val 로 설정
#             override_val = 2.0
#             I = I*(1.0 - mask) + override_val*mask

#         # 관측(부분 샘플 + 잡음)
#         coords_t = coords_full[idx]
#         y_t = I[idx]
#         noise = sigma * (np.random.normal(size=y_t.shape) + 1j*np.random.normal(size=y_t.shape))
#         y_t_noisy = (y_t + noise).astype(np.complex64)

#         coords_list.append(coords_t)
#         y_list.append(y_t_noisy)
#         y_true_list.append(y_t)
#         y_true_full_list.append(I.astype(np.complex64))
    
#     t_list = [float(t) * dt for t in t_list]
    
#     return t_list, coords_list, y_list, y_true_list, y_true_full_list, coords_full, (alpha, omega, b), W_full


def synth_sequence(T, norm_T, resolution, dt, sample_ratio=0.1, sigma=0.1, seed=0, normalize_t=False):  # T reduced for speed
    if T is None:
        T = norm_T
    np.random.seed(seed)
    coords_full, shape = make_grid(resolution[0], resolution[1])
    n = coords_full.shape[0]

    W_full = true_modes(coords_full)

    # alpha = np.array([-0.1, -0.05, -0.2, 0.0])
    # omega = np.array([2.0, 4.0, 1.0, 0.0])
    # b = np.array([1.0+0.5j, 0.8-0.3j, 0.5+0.2j, 1.0+0.0j], dtype=np.complex64)
    alpha = np.array([-0.01, -0.05, -0.20, -0.01])  # m4에 약한 감쇠
    omega = np.array([ 2.00,  4.00,  1.00,  0.30])  # m4에 아주 약한 진동
    b = np.array([1.0+0.5j, 0.8-0.3j, 0.7+0.2j, 0.2+0.0j], dtype=np.complex64)  # 상수모드 초기계수 축소

    compensate_frame_time = 0.1/dt # make every experiments to have same sequence. 
    coords_list = []
    y_list = []
    y_true_list = []
    y_true_full_list = []
    k = max(1, int(n * sample_ratio))
    idx = np.random.choice(n, size=k, replace=False) #Assumes observation is accessible only at fixed locations.
    for t in range(T):
        phi = np.exp((alpha + 1j*omega) * t*dt*compensate_frame_time) * b
        I = W_full @ phi

        coords_t = coords_full[idx]
        y_t = I[idx]

        noise = sigma * (np.random.normal(size=y_t.shape) + 1j*np.random.normal(size=y_t.shape))
        y_t_noisy = (y_t + noise).astype(np.complex64)

        coords_list.append(coords_t)
        y_list.append(y_t_noisy)
        y_true_list.append(y_t)
        y_true_full_list.append(I)
    t_list = list(range(T))
    if normalize_t:
        t_list = [np.float32(t) / np.float32(norm_T) for t in t_list]
    else:
        t_list = [np.float32(t) * np.float32(dt) for t in t_list]
    # print(f"t_list dtype: {type(t_list[0])}, coords dtype: {coords_list[0].dtype}, y dtype: {y_list[0].dtype}")
    return t_list, coords_list, y_list, y_true_list, y_true_full_list, coords_full, (alpha, omega, b), W_full



# for NDMD execution, comment out below 
def load_synth(device: torch.device, T=None, norm_T=50, resolution=(32,32), dt=0.1, normalize_t=False):
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
    ) = synth_sequence(T=T, norm_T=norm_T, resolution=resolution, dt=dt, normalize_t=normalize_t)

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
        t_prev = (self.t_list[idx])
        t_next = (self.t_list[idx + 1])
        coords = self.coords_list[idx + 1]
        y_prev = self.y_list[idx]
        y_next = self.y_list[idx + 1]
        return t_prev, t_next, coords, y_next, y_prev

class SynthDataset_seq(Dataset):
    def __init__(self, t_list, coords_list, y_list, K):
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
        self.length = len(t_list) - 1 - K  # Number of (t_prev, t_next) pairs
        self.K = K # sequential data that I'm loading in single step 
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
        t_prev = (self.t_list[idx])
        t_next = self.t_list[idx + 1:idx + 1+self.K]
        coords = self.coords_list[idx + 1:idx + 1+self.K]
        y_prev = self.y_list[idx: idx+self.K]
        y_next = self.y_list[idx + 1: idx + 1+self.K]
        return t_prev,t_next, coords, y_next, y_prev 