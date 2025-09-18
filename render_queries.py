import blenderproc as bproc

'''
Synthetic dataset generator using BlenderProc for 6D object pose estimation and segmentation.
Generates BOP-format datasets with pose annotations and segmentation masks.
Features: randomized object classes/instances/sizes/materials, randomized cctexture backgrounds,
randomized object sampling and physics simulation for realistic cluttered scenes.
'''

import argparse
import os
import sys
import numpy as np
import yaml

# Add current directory to Python path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from query_generation import (
    create_trunc_poisson_pmf,
    load_random_models, load_model_paths,
    sample_poses,
    initialize_report, create_scene_report, collect_object_data, write_scene_report,
    setup_room_and_lighting, load_texture, randomize_scene_lighting, clean_up,
    generate_camera_poses, set_random_camera_intrinsics, get_camera_radius_and_room_size,
    apply_random_sizes, apply_material_randomization, setup_physics,
    write_scene_gt_obj
)

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="config.yml", help="Path to configuration file")
parser.add_argument('--output_dir', help="Output directory (overrides config value)")
args = parser.parse_args()

# Load configuration file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Override output_dir if provided via command line
if args.output_dir:
    config['dataset']['output_dir'] = args.output_dir

# Create truncated Poisson PMF for instance sampling
instance_values, instance_probs = create_trunc_poisson_pmf(
    lambda_param=config['object_parameters']['poisson_lambda'],
    shift=config['object_parameters']['poisson_shift'],
    min_val=1,
    max_val=config['object_parameters']['max_instances_per_object']
)

# Initialize BlenderProc
bproc.init()
bproc.camera.set_resolution(config['scene_parameters']['resolution'][0], config['scene_parameters']['resolution'][1])

all_model_paths = load_model_paths(config)

# Initialize report
report_filename, run_report = initialize_report(config)

# Setup renderer
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

for i in range(config['scene_parameters']['num_scenes']):
    
    # Randomly select number of object classes and images for this scene
    num_object_classes = np.random.randint(config['object_parameters']['min_objects_per_scene'], 
                                         config['object_parameters']['max_objects_per_scene'] + 1)
    num_images = np.random.randint(config['scene_parameters']['min_images_per_scene'], 
                                 config['scene_parameters']['max_images_per_scene'] + 1)

    # Initialize scene report data
    scene_report = create_scene_report(i + 1, num_object_classes, num_images)

    # Set random camera intrinsics for this scene
    cam_K = set_random_camera_intrinsics(
        config['scene_parameters']['resolution'][0], 
        config['scene_parameters']['resolution'][1]
    )

    # Get camera radius and room size, dependent on camera intrinsics
    radius_min, radius_max, room_size = get_camera_radius_and_room_size(config, cam_K)

    # Setup room, lighting for this scene
    room_planes, light_plane, light_plane_material, light_point = setup_room_and_lighting(room_size)

    # Load random models for this scene
    scene_objects = load_random_models(all_model_paths, num_object_classes, config, instance_values, instance_probs)

    # Apply random sizes to objects
    object_sizes = apply_random_sizes(scene_objects, config)
    scene_report['object_sizes'] = object_sizes.copy() # Store object sizes in scene report

    # Apply material randomization and setup physics
    apply_material_randomization(scene_objects, config)
    setup_physics(scene_objects)
    
    # Load one random texture for this scene
    cc_texture, texture_name = load_texture(config)
    scene_report['cc_texture'] = texture_name
    
    # Randomize scene lighting and background texture
    randomize_scene_lighting(light_plane, light_plane_material, light_point, room_planes, [cc_texture])

    # Collect object data for reporting
    object_instance_counts = collect_object_data(scene_objects, object_sizes)
    scene_report['objects'] = object_instance_counts

    # Sample poses for objects and simulate physics
    sample_poses(scene_objects, max_tries=1000, config=config)
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                    max_simulation_time=10,
                                                    check_object_interval=1,
                                                    substeps_per_frame = 20,
                                                    solver_iters=25)

    # Generate camera poses
    cam_poses = generate_camera_poses(config, scene_objects, num_images, radius_min, radius_max)
    scene_report['num_images_generated'] = cam_poses # Store number of images generated in scene report

    # Rendering and writing data
    data = bproc.renderer.render()
    bproc.writer.write_bop(os.path.join(config['dataset']['output_dir']),
                           target_objects = scene_objects,
                           dataset = config['dataset']['name'],
                           depth_scale = 0.1,
                           depths = data["depth"],
                           colors = data["colors"], 
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10,
                           annotation_unit="mm",
                           delta=0.015)

    # Write scene_gt_obj.json with object sizes
    write_scene_gt_obj(
        output_dir=config['dataset']['output_dir'],
        dataset_name=config['dataset']['name'],
        object_sizes=object_sizes,             
        num_new_frames=len(data["colors"]),     
        split="train_pbr",
        frames_per_chunk=1000,                  
        mode="auto"
    )

    # Write scene report
    write_scene_report(report_filename, scene_report, run_report)
    
    # Memory cleanup
    scene_report = None
    object_instance_counts = None
    clean_up(scene_objects)