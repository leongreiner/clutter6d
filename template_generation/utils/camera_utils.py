import torch
import numpy as np
from pytorch3d.renderer import PointLights

def setup_camera_intrinsics(fov_deg: float, image_size: int) -> tuple:
    fov_rad = np.deg2rad(fov_deg)
    focal_length = image_size / (2.0 * np.tan(fov_rad / 2.0))
    principal_point = ((image_size - 1) / 2.0, (image_size - 1) / 2.0)
    return focal_length, principal_point

def calculate_optimal_distance(mesh, fov_rad: float, margin: float = 1.0) -> float:
    verts = mesh.verts_list()[0]
    vmin = verts.min(0)[0]
    vmax = verts.max(0)[0]
    bbox_size = vmax - vmin
    bbox_diagonal = torch.sqrt((bbox_size ** 2).sum()).item()
    circumscribing_radius = bbox_diagonal / 2.0
    min_distance = circumscribing_radius / np.tan(fov_rad / 2.0)
    optimal_distance = min_distance * margin
    min_allowed_distance = bbox_size.max().item() * 1.3
    optimal_distance = max(optimal_distance, min_allowed_distance)
    return optimal_distance

def get_camera_settings(mesh, fov_deg: float, device):
    fov_rad = np.deg2rad(fov_deg)
    distance = calculate_optimal_distance(mesh, fov_rad)
    lights = PointLights(
        device=device,
        ambient_color=[(1, 1, 1)],
        diffuse_color=[(0.8, 0.8, 0.8)],
        specular_color=[(0.5, 0.5, 0.5)],
        location=[[0.0, 0.0, distance]],
    )
    return distance, lights

def generate_camera_positions(subdivisions: int, distance: float, device: torch.device) -> torch.Tensor:
    num_views = 10 * (4 ** subdivisions) + 2
    idx = torch.arange(num_views, dtype=torch.float32, device=device)
    phi = torch.acos(1.0 - 2.0 * (idx + 0.5) / num_views)
    golden = (1.0 + torch.sqrt(torch.tensor(5.0, device=device))) / 2.0
    theta = 2.0 * torch.pi * idx / golden
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    directions = torch.stack([x, y, z], dim=1)
    return directions * distance