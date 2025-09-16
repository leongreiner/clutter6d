import os
import gc
import numpy as np
import h5py
from io import BytesIO
from PIL import Image

def save_to_hdf5(templates: np.ndarray, camk: np.ndarray, 
                 extrinsics: np.ndarray, surface_points: np.ndarray, obj_id: str,
                 output_dir: str, model_name: str, image_size: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, f"{model_name}.h5")
    
    if os.path.exists(h5_path):
        gc.collect()
        os.remove(h5_path)
    
    with h5py.File(h5_path, 'w') as f:
        templates_group = f.create_group('templates')
        for i, template in enumerate(templates):
            template_buffer = BytesIO()
            Image.fromarray(template, mode='RGB').save(template_buffer, format='JPEG', quality=95)
            templates_group.create_dataset(f'template_{i:03}', data=np.void(template_buffer.getvalue()))
        
        f.create_dataset('camk', data=camk)
        f.create_dataset('extrinsics', data=extrinsics)
        f.create_dataset('surface_points', data=surface_points.astype(np.float32))
        f.attrs['num_views'] = len(templates)
        f.attrs['image_size'] = image_size
        f.attrs['obj_id'] = obj_id
        f.attrs['num_surface_points'] = surface_points.shape[0]

def extract_obj_id_from_filename(filename: str) -> str:
    if not filename.lower().endswith('.glb'):
        raise ValueError(f"Only .glb files supported, got: {filename}")
    base_name = filename[:-4]
    if not base_name.startswith('obj_id_'):
        raise ValueError(f"GLB filename must start with 'obj_id_', got: {filename}")
    parts = base_name.split('__')
    if len(parts) < 2:
        raise ValueError(f"Filename must contain '__' after obj_id_XXXXXX, got: {filename}")
    obj_id_part = parts[0]
    if len(obj_id_part) != 13 or not obj_id_part[7:].isdigit():
        raise ValueError(f"Object ID must be 6 digits after 'obj_id_', got: {obj_id_part}")
    return obj_id_part[7:]