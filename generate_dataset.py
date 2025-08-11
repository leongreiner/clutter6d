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
from tqdm import tqdm

# Add current directory to Python path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils import (
    create_trunc_poisson_pmf,
    load_random_models, load_model_paths,
    sample_poses,
    initialize_report, create_scene_report, collect_object_data, write_scene_report,
    setup_room_and_lighting, load_textures, randomize_scene_lighting,
    generate_camera_poses,
    apply_random_sizes, apply_material_randomization, setup_physics, cleanup_scene_objects
)

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="config.yaml", help="Path to configuration file")
args = parser.parse_args()

# Load configuration file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

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

# Setup room, lighting, and load textures
room_planes, light_plane, light_plane_material, light_point = setup_room_and_lighting()
cc_textures = load_textures(config)

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

    # Clear camera poses from previous scene
    bproc.utility.reset_keyframes()

    # Load random models for this scene
    scene_objects = load_random_models(all_model_paths, num_object_classes, config, instance_values, instance_probs)

    # Apply random sizes to objects
    object_sizes = apply_random_sizes(scene_objects, config)
    scene_report['object_sizes'] = object_sizes.copy() # Store object sizes in scene report

    # Apply material randomization and setup physics
    apply_material_randomization(scene_objects, config)
    setup_physics(scene_objects)
    
    # Randomize scene lighting and background texture
    randomize_scene_lighting(light_plane_material, light_point, room_planes, cc_textures)

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
    cam_poses = generate_camera_poses(scene_objects, num_images, config)
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
                           ignore_dist_thres = 10)
    
    # Cleanup scene objects
    cleanup_scene_objects(scene_objects)
    
    # Write scene report
    write_scene_report(report_filename, scene_report, run_report)
    
    # Memory cleanup
    scene_report = None
    object_instance_counts = None