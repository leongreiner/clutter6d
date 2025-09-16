#!/usr/bin/env python3
import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np
import torch
import yaml
from renderer import ObjRenderer

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(script_dir, 'config.yml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set computed values
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



def check_texture_requirements(glb_path: str) -> bool:
    return True



def render_on_gpu(args):
    dataset_key, model_path, config, gpu_id, skip_existing = args
    
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.empty_cache()

    try:
        torch.cuda.empty_cache()
        
        try:
            obj_id = extract_obj_id_from_glb_filename(model_path)
        except ValueError as e:
            return f"CRITICAL ERROR: {e} for file {model_path}"

        h5_path = os.path.join(config['paths']['output_root'], f"{obj_id}.h5")
        if os.path.exists(h5_path):
            if skip_existing and check_render_content(h5_path):
                return f"Skipping {dataset_key}/{obj_id}: already rendered"
            else:
                try:
                    import gc
                    gc.collect()
                    os.remove(h5_path)
                except Exception as e:
                    print(f"Warning: Could not remove {h5_path}: {e}")
        
        if not check_texture_requirements(model_path):
            return f"CRITICAL ERROR: No valid texture found for GLB file {dataset_key}/{obj_id}"
        
        try:
            renderer = ObjRenderer(
                device=device,
                subdivisions=config['rendering']['subdivisions'], 
                image_size=config['rendering']['image_size'],
                fov_deg=config['rendering']['fov_deg'],
                background_color=tuple(config['background_color']),
                center_tolerance=config['validation']['center_tolerance'],
                scale_tolerance=config['validation']['scale_tolerance']
            )
            output_dir = config['paths']['output_root']
            renderer.render(model_path, output_dir, obj_id, 
                          num_surface_points=config['rendering']['num_surface_points'])
            if not check_render_content(h5_path):
                return f"WARNING: No object visible in render for {dataset_key}/{obj_id}"
            else:
                return f"Successfully rendered {dataset_key}/{obj_id} on GPU {gpu_id}"
                
        except ValueError as e:
            if "not centered" in str(e) or "not normalized" in str(e):
                return f"CRITICAL ERROR: Mesh validation failed for {dataset_key}/{obj_id}: {e}"
            else:
                raise
                
    except Exception as e:
        torch.cuda.empty_cache()
        return f"Error rendering {dataset_key}/{os.path.basename(model_path)} on GPU {gpu_id}: {e}"



def main():
    parser = argparse.ArgumentParser(description='Batch render GLB models across multiple GPUs')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip models that have already been rendered')
    
    # Add config override arguments
    parser.add_argument('--base_models_dir', type=str, help='Override base models directory')
    parser.add_argument('--output_root', type=str, help='Override output root directory')
    parser.add_argument('--subdivisions', type=int, help='Override subdivisions')
    parser.add_argument('--image_size', type=int, help='Override image size')
    parser.add_argument('--num_surface_points', type=int, help='Override number of surface points')
    
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.skip_existing:
        config['processing']['skip_existing'] = True
    if args.base_models_dir:
        config['paths']['base_models_dir'] = args.base_models_dir
    if args.output_root:
        config['paths']['output_root'] = args.output_root
    if args.subdivisions:
        config['rendering']['subdivisions'] = args.subdivisions
    if args.image_size:
        config['rendering']['image_size'] = args.image_size
    if args.num_surface_points:
        config['rendering']['num_surface_points'] = args.num_surface_points
    
    BASE_MODELS_DIR = config['paths']['base_models_dir']
    OUTPUT_ROOT = config['paths']['output_root']

    # Get available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available, falling back to CPU")
        return
    print(f"Found {num_gpus} GPUs")

    def batch_render_multi_gpu(dataset_key, models_root):
        model_files = []
        for entry in sorted(os.listdir(models_root)):
            file_path = os.path.join(models_root, entry)
            if (os.path.isfile(file_path) and 
                entry.lower().endswith(config['file_patterns']['glb_extension'])
                and entry.startswith(config['file_patterns']['glb_prefix'])):
                model_files.append(file_path)

        if not model_files:
            print(f"No GLB files with {config['file_patterns']['glb_prefix']} prefix found in {models_root}")
            return

        print(f"Found {len(model_files)} GLB files to render in {dataset_key}")

        render_args = []
        for i, model_file in enumerate(model_files):
            gpu_id = i % num_gpus
            render_args.append((dataset_key, model_file, config, gpu_id, 
                              config['processing']['skip_existing']))

        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            for result in executor.map(render_on_gpu, render_args):
                print(result)
        
    print("Scanning for GLB files in dataset directories...")
    
    dataset_paths = {}
    for dataset_key, dataset_subdir in config['datasets'].items():
        dataset_paths[dataset_key] = os.path.join(BASE_MODELS_DIR, dataset_subdir)
    
    for dataset_key, dataset_path in dataset_paths.items():
        if os.path.exists(dataset_path):
            print(f"Processing {dataset_key} from {dataset_path}")
            batch_render_multi_gpu(dataset_key, dataset_path)
        else:
            print(f"WARNING: Dataset path not found: {dataset_path}")
    
    print("Multi-GPU batch render complete!")

if __name__ == '__main__':
    main()
