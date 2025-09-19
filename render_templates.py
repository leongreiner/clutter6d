#!/usr/bin/env python3

'''
Template renderer for clutter6d using PyTorch3D.
Generates multi-viewpoint object templates from GLB models for template matching.
Features: multi-GPU batch processing, icosphere viewpoint sampling, HDF5 storage with
JPEG compression, camera intrinsics/extrinsics, and surface point sampling.
'''

import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
import torch
from template_generation.renderer import ObjRenderer
from template_generation.utils import load_config, check_render_content, extract_obj_id_from_glb_filename

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

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
            chunk_size = config['rendering'].get('chunk_size', 14)
            renderer.render(model_path, output_dir, obj_id, 
                          num_surface_points=config['rendering']['num_surface_points'],
                          chunk_size=chunk_size)
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


def batch_render_multi_gpu(dataset_key, models_root, config, num_gpus):
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


def main():
    parser = argparse.ArgumentParser(description='Batch render GLB models across multiple GPUs')
    parser.add_argument('--config', default='template_generation/config.yml', type=str, help='Path to config YAML file')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip models that have already been rendered')
    
    # Config override arguments
    parser.add_argument('--base_models_dir', type=str, help='Override base models directory')
    parser.add_argument('--output_root', type=str, help='Override output root directory')
    parser.add_argument('--subdivisions', type=int, help='Override subdivisions')
    parser.add_argument('--image_size', type=int, help='Override image size')
    parser.add_argument('--num_surface_points', type=int, help='Override number of surface points')
    parser.add_argument('--chunk_size', type=int, help='Override chunk size for memory management')
    args = parser.parse_args()
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
    if args.chunk_size:
        config['rendering']['chunk_size'] = args.chunk_size

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    dataset_paths = {}
    for dataset_key, dataset_subdir in config['datasets'].items():
        dataset_paths[dataset_key] = os.path.join(config['paths']['base_models_dir'], dataset_subdir)
    
    for dataset_key, dataset_path in dataset_paths.items():
        if os.path.exists(dataset_path):
            print(f"Processing {dataset_key} from {dataset_path}")
            batch_render_multi_gpu(dataset_key, dataset_path, config, num_gpus)
        else:
            print(f"WARNING: Dataset path not found: {dataset_path}")
    
    print("Multi-GPU batch render complete!")

if __name__ == '__main__':
    main()
