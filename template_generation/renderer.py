#!/usr/bin/env python3
import os
import sys
from io import BytesIO
import torch
import numpy as np
from PIL import Image
import h5py
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    BlendParams,
)
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat

from utils import (
    sample_surface_points,
    save_to_hdf5,
    extract_obj_id_from_filename,
    setup_camera_intrinsics,
    get_camera_settings,
    generate_camera_positions
)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)

class ObjRenderer:
    def __init__(
        self,
        device: torch.device = None,
        subdivisions: int = 1,
        image_size: int = 512,
        fov_deg: float = 60.0,
        background_color: tuple = (0.0, 0.0, 0.0),
        center_tolerance: float = 0.02,
        scale_tolerance: float = 0.02,
    ):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.subdivisions = subdivisions
        self.image_size = image_size
        self.fov_deg = fov_deg
        self.center_tolerance = center_tolerance
        self.scale_tolerance = scale_tolerance
        self.blend_params = BlendParams(background_color=background_color)

        self.focal_length, self.principal_point = setup_camera_intrinsics(self.fov_deg, self.image_size)

    def load_glb_mesh(self, glb_path: str):
        io = IO()
        io.register_meshes_format(MeshGlbFormat())
        
        with open(glb_path, "rb") as f:
            mesh = io.load_mesh(f, include_textures=True, device=self.device)
            return mesh

    def validate_mesh_normalization(self, mesh) -> None:
        verts = mesh.verts_list()[0]
        vmin = verts.min(0)[0]; vmax = verts.max(0)[0]
        bbox_center = (vmin + vmax) / 2.0
        if bbox_center.norm().item() > self.center_tolerance:
            raise ValueError(f"Mesh is not centered (bbox center > {self.center_tolerance})")
        max_dim = (vmax - vmin).max().item()
        if abs(max_dim - 1.0) > self.scale_tolerance:
            raise ValueError(f"Mesh is not normalized (max bbox dim {max_dim:.4f})")

    def render(self, glb_path: str, output_dir: str, model_name: str, num_surface_points: int = 10000) -> None:
        if not glb_path.lower().endswith('.glb'):
            raise ValueError("Only GLB files are supported.")
        os.makedirs(output_dir, exist_ok=True)
        torch.cuda.empty_cache()
        filename = os.path.basename(glb_path)
        obj_id = extract_obj_id_from_filename(filename)
        mesh = self.load_glb_mesh(glb_path)
        self.validate_mesh_normalization(mesh)
        
        surface_points = sample_surface_points(mesh, num_points=num_surface_points, device=self.device)
        
        distance, lights = get_camera_settings(mesh, self.fov_deg, self.device)
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=3,
            bin_size=0,
            max_faces_per_bin=int(1e9)
        )
        cam_positions = generate_camera_positions(self.subdivisions, distance, self.device)
        num_views = cam_positions.shape[0]
        R_list, T_list = [], []
        for pos in cam_positions:
            R, T = look_at_view_transform(eye=[pos.tolist()], at=[[0,0,0]], up=[[0,1,0]], device=self.device)
            R_list.append(R); T_list.append(T)
        R_all = torch.cat(R_list, 0); T_all = torch.cat(T_list, 0)
        mesh_batch = mesh.extend(num_views)
        cameras = PerspectiveCameras(
            device=self.device,
            R=R_all,
            T=T_all,
            focal_length=[[self.focal_length, self.focal_length]] * num_views,
            principal_point=[self.principal_point] * num_views,
            image_size=[[self.image_size, self.image_size]] * num_views,
            in_ndc=False
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        shader = SoftPhongShader(device=self.device, cameras=cameras, lights=lights, blend_params=self.blend_params)
        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        torch.cuda.empty_cache()
        render_output = renderer(mesh_batch)
        templates = (render_output[..., :3].cpu().numpy() * 255).clip(0,255).astype(np.uint8)
        K = torch.tensor([
            [self.focal_length, 0.0, self.principal_point[0]],
            [0.0, self.focal_length, self.principal_point[1]],
            [0.0, 0.0, 1.0],
        ], device=self.device)
        camk_batch = K.unsqueeze(0).repeat(num_views,1,1)
        extrinsics = torch.eye(4, device=self.device).unsqueeze(0).repeat(num_views,1,1)
        extrinsics[:, :3, :3] = R_all; extrinsics[:, :3, 3] = T_all
        camk_np = camk_batch.cpu().numpy(); extrinsics_np = extrinsics.cpu().numpy()
        del render_output, mesh_batch, renderer, rasterizer, shader
        torch.cuda.empty_cache()
        save_to_hdf5(templates, camk_np, extrinsics_np, surface_points, obj_id, output_dir, model_name, self.image_size)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Render GLB model and save RGB views as compressed JPGs inside HDF5.")
    parser.add_argument('--glb_path', required=True, help='Path to a single GLB file (must follow obj_id_XXXXXX__*.glb naming)')
    parser.add_argument('--output_root', default='./renders', help='Directory to store output HDF5')
    parser.add_argument('--subdivisions', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=700)
    parser.add_argument('--num_surface_points', type=int, default=10000, help='Number of surface points to sample from the CAD model')
    args = parser.parse_args()
    if not os.path.isfile(args.glb_path):
        raise FileNotFoundError(f"GLB file not found: {args.glb_path}")
    renderer = ObjRenderer(
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        subdivisions=args.subdivisions,
        image_size=args.image_size
    )
    model_name = os.path.splitext(os.path.basename(args.glb_path))[0]
    output_dir = args.output_root
    renderer.render(args.glb_path, output_dir, model_name, args.num_surface_points)
    print("Render complete.")

if __name__ == '__main__':
    main()
