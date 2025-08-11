import blenderproc as bproc
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

from utils.helpers import create_trunc_poisson_pmf
from utils.model_loader import load_random_models, load_model_paths
from utils.pose_sampler import sample_poses
from utils.report_generator import initialize_report, create_scene_report, collect_object_data, write_scene_report
from utils.scene_setup import setup_room_and_lighting, load_textures, randomize_scene_lighting
from utils.camera_utils import generate_camera_poses

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

bproc.init()
bproc.camera.set_resolution(config['scene_parameters']['resolution'][0], config['scene_parameters']['resolution'][1])

print("Scanning for available models...")
all_model_paths = load_model_paths(config)

# Setup room, lighting, and load textures
room_planes, light_plane, light_plane_material, light_point = setup_room_and_lighting()
cc_textures = load_textures(config)

import time
start = time.time()

# Initialize report
report_filename, run_report = initialize_report(config)

# Setup renderer
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

print("Starting scene generation...")

for i in range(config['scene_parameters']['num_scenes']):
    
    # Randomly select number of object classes and images for this scene
    num_object_classes = np.random.randint(config['object_parameters']['min_objects_per_scene'], 
                                         config['object_parameters']['max_objects_per_scene'] + 1)
    num_images = np.random.randint(config['scene_parameters']['min_images_per_scene'], 
                                 config['scene_parameters']['max_images_per_scene'] + 1)
    
    print(f"Scene {i+1}: {num_object_classes} object classes, {num_images} images")

    # Initialize scene report data
    scene_report = create_scene_report(i + 1, num_object_classes, num_images)

    # Clear camera poses from previous scene
    bproc.utility.reset_keyframes()

    scene_objects = load_random_models(all_model_paths, num_object_classes, config, instance_values, instance_probs)

    if not scene_objects:
        print(f"No objects loaded for scene {i+1}, skipping...")
        continue

    # Generate random sizes for each unique object in this scene
    # All instances of the same object have the same size
    object_sizes = {}
    for obj in scene_objects:
        obj_id = obj.get_cp("obj_id")
        if obj_id not in object_sizes:
            # Generate random size for this object type in this scene
            random_size = np.random.uniform(config['object_parameters']['min_size'], config['object_parameters']['max_size'])
            object_sizes[obj_id] = random_size
    # Apply the consistent size to all instances of each object
    for obj in scene_objects:
        obj_id = obj.get_cp("obj_id")
        target_size = object_sizes.get(obj_id)
        obj.set_scale([target_size, target_size, target_size])
    
    # Store object sizes in scene report
    scene_report['object_sizes'] = object_sizes.copy()
    
    print(f"Random object sizes for scene {i+1}:")
    for obj_id, size in object_sizes.items():
        print(f"  Object {obj_id}: {size:.3f}m")

    # Keep original GLB textures, but apply material randomization
    for obj in scene_objects:
        mat = obj.get_materials()[0]        
        mat.set_principled_shader_value("Roughness", 
            np.random.uniform(config['material_randomization']['roughness_min'], 
                            config['material_randomization']['roughness_max']))
        mat.set_principled_shader_value("Metallic", 
            np.random.uniform(config['material_randomization']['metallic_min'], 
                            config['material_randomization']['metallic_max']))
        obj.set_shading_mode('auto')
        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99, collision_shape='CONVEX_HULL')
        obj.hide(False)
    
    # Randomize scene lighting and textures
    randomize_scene_lighting(light_plane_material, light_point, room_planes, cc_textures)

    # Collect object data for reporting
    object_instance_counts = collect_object_data(scene_objects, object_sizes)
    
    # Store object data in scene report
    scene_report['objects'] = object_instance_counts

    sample_poses(scene_objects, max_tries=1000, config=config)
        
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                    max_simulation_time=10,
                                                    check_object_interval=1,
                                                    substeps_per_frame = 20,
                                                    solver_iters=25)

    # Generate camera poses
    cam_poses = generate_camera_poses(scene_objects, num_images, config)

    # Record actual number of images generated
    scene_report['num_images_generated'] = cam_poses
    
    data = bproc.renderer.render()

    bproc.writer.write_bop(os.path.join(config['dataset']['output_dir']),
                           target_objects = scene_objects,
                           dataset = config['dataset']['name'],
                           depth_scale = 0.1,
                           depths = data["depth"],
                           colors = data["colors"], 
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10)
    
    for obj in scene_objects:      
        obj.disable_rigidbody()
        obj.hide(True)
    
    # Write scene report
    write_scene_report(report_filename, scene_report, run_report)
    
    # Memory cleanup
    scene_report = None
    object_instance_counts = None

# Generate final summary
end = time.time()
total_runtime = end - start