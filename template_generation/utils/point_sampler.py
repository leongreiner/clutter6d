import torch
import numpy as np
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points

def sample_surface_points(mesh, num_points: int = 10_000, device=None) -> np.ndarray:
    return _sample_fps_cuda_from_mesh(mesh, k=num_points, oversample=4, device=device)

@torch.no_grad()
def _sample_fps_cuda_from_mesh(mesh, k: int = 10_000, oversample: int = 1, device=None) -> np.ndarray:
    if device is None:
        device = mesh.device if hasattr(mesh, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    assert oversample >= 1
    total_sample = int(k * oversample)
    pts, _ = sample_points_from_meshes(mesh.to(device), total_sample, return_normals=True)  # [1, N, 3]
    assert pts.shape[0] == 1 and pts.shape[1] >= k, "Need at least k candidates"
    c = pts.mean(dim=1, keepdim=True)                         # [1,1,3]
    d2 = ((pts - c) ** 2).sum(-1)                             # [1,N]
    start = int(torch.argmax(d2, dim=1).item()) 
    if start != 0:
        perm = torch.arange(pts.shape[1], device=device)
        perm[0], perm[start] = perm[start], perm[0]
        pts = pts[:, perm, :]

    fps_pts, _ = sample_farthest_points(pts, K=k, random_start_point=False)  # [1,k,3]

    return fps_pts[0].contiguous().cpu().numpy()              # [k,3]
