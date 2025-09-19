import os
import sys
import yaml
import h5py
import torch
import numpy as np

def load_config(config_path: str = None) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    if config_path is None:
        config_path = os.path.join(script_dir, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if config['processing']['max_workers'] is None:
        config['processing']['max_workers'] = torch.cuda.device_count()
    return config

def check_render_content(h5_path):
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'templates' not in f:
                print(f"No 'templates' dataset/group in {h5_path}")
                return False
            if not isinstance(f['templates'], h5py.Group):
                print(f"Old format detected (templates as dataset, not group): {h5_path}")
                return False
            template_keys = sorted([k for k in f['templates'].keys() if k.startswith('template_')])
            if not template_keys:
                print(f"No template files found in templates group in {h5_path}")
                return False
            if 'camk' not in f:
                print(f"No 'camk' dataset in {h5_path}")
                return False
            if 'extrinsics' not in f:
                print(f"No 'extrinsics' dataset in {h5_path}")
                return False
            required_attrs = ['num_views', 'image_size', 'obj_id']
            for attr in required_attrs:
                if attr not in f.attrs:
                    print(f"Missing required attribute '{attr}' in {h5_path}")
                    return False
            num_views = f.attrs['num_views']
            if len(template_keys) != num_views:
                print(f"Mismatch between num_views ({num_views}) and templates ({len(template_keys)}) in {h5_path}")
                return False
            templates_list = []
            for key in template_keys[:3]:
                jpg_data = f['templates'][key][()]
                if hasattr(jpg_data, 'tobytes'):
                    jpg_bytes = jpg_data.tobytes()
                else:
                    jpg_bytes = bytes(jpg_data)
                from PIL import Image
                from io import BytesIO
                img = np.array(Image.open(BytesIO(jpg_bytes)))
                templates_list.append(img)
            templates = np.array(templates_list)
            has_content = np.any(templates > 0)
            if not has_content:
                print(f"WARNING: All template pixels are black (0) in {h5_path}")
            return has_content
    except Exception as e:
        print(f"Error checking render content for {h5_path}: {e}")
        return False

def extract_obj_id_from_glb_filename(file_path: str) -> str:
    filename = os.path.basename(file_path)
    if not filename.lower().endswith('.glb'):
        raise ValueError(f"Expected GLB file, got: {filename}")
    base_name = filename[:-4]
    if not base_name.startswith('obj_id_'):
        raise ValueError(f"Filename must start with 'obj_id_', got: {filename}")
    parts = base_name.split('__')
    if len(parts) < 2:
        raise ValueError(f"Filename must contain '__' separator after obj_id_XXXXXX, got: {filename}")
    obj_id_part = parts[0]
    if len(obj_id_part) != 13:
        raise ValueError(f"Object ID must be 6 digits after 'obj_id_', got: {obj_id_part}")
    obj_id = obj_id_part[7:]
    if not obj_id.isdigit():
        raise ValueError(f"Object ID must be numeric, got: {obj_id}")
    return obj_id
