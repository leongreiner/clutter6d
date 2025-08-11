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
    
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)

light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

light_point = bproc.types.Light()
light_point.set_energy(200)

folder = config['dataset']['cc_textures_path']
asset_names = os.listdir(folder)

cc_textures = []
for asset in tqdm(asset_names, desc="Loading CC textures", unit="texture", total=len(asset_names)):
    mats = bproc.loader.load_ccmaterials(
        folder_path=folder,
        used_assets=[asset]
    )
    cc_textures.extend(mats)

import time
import csv
from datetime import datetime
start = time.time()

# Initialize report data structure
run_report = []

# Create timestamp for report filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = os.path.join(config['dataset']['output_dir'], config['dataset']['name'], f'generation_report_{timestamp}.csv')

# Initialize CSV file with headers
os.makedirs(os.path.dirname(report_filename), exist_ok=True)
with open(report_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = [
        'scene_id', 'num_object_classes', 'num_images_planned', 'num_images_generated',
        'total_object_instances', 'object_data'  # Will contain detailed object information as JSON string
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
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
    scene_report = {
        'scene_id': i + 1,
        'num_object_classes': num_object_classes,
        'num_images_planned': num_images,
        'num_images_generated': 0,
        'objects': [],
        'object_sizes': {}
    }

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
    
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    light_point.set_location(location)

    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)

    # Debug Printing
    print(f"Placing {len(scene_objects)} object instances in scene {i+1}:")
    
    # Track object instances for report
    object_instance_counts = {}
    
    for j, obj in enumerate(scene_objects):
        obj_path = obj.get_cp("model_path") if obj.has_cp("model_path") else "unknown"
        obj_id = obj.get_cp("obj_id") if obj.has_cp("obj_id") else "unknown"
        instance_id = obj.get_cp("instance_id") if obj.has_cp("instance_id") else "unknown"
        category = obj.get_cp("category") if obj.has_cp("category") else "unknown"
        name = obj.get_cp("model_name") if obj.has_cp("model_name") else "unknown"
        dataset_source = obj.get_cp("dataset_source") if obj.has_cp("dataset_source") else "unknown"
        assigned_size = object_sizes.get(obj_id, "unknown")
        
        # Count instances per object
        if obj_id not in object_instance_counts:
            object_instance_counts[obj_id] = {
                'count': 0,
                'category': category,
                'name': name,
                'dataset_source': dataset_source,
                'model_path': obj_path,
                'size': assigned_size
            }
        object_instance_counts[obj_id]['count'] += 1
        
        print(f"  {j+1:2d}. {obj.get_name()} (obj_id: {obj_id}, cat: {category}, name: {name}, instance: {instance_id}, size: {assigned_size:.3f}m) -> {os.path.basename(obj_path)}")
    
    # Store object data in scene report
    scene_report['objects'] = object_instance_counts

    sample_poses(scene_objects, max_tries=1000, config=config)
        
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                    max_simulation_time=10,
                                                    check_object_interval=1,
                                                    substeps_per_frame = 20,
                                                    solver_iters=25)

    # BVH tree for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(scene_objects)

    cam_poses = 0
    while cam_poses < num_images:

        # Camera positioning adjusted for smaller objects
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = config['camera']['radius_min'],
                                radius_max = config['camera']['radius_max'],
                                elevation_min = config['camera']['elevation_min'],
                                elevation_max = config['camera']['elevation_max'])
        
        # When there are at least 15 objects, compute POI from a random selection of 15 objects to get more varied camera viewpoints
        if len(scene_objects) >= 15:
            poi_objects = np.random.choice(scene_objects, size=15, replace=False)
            poi = bproc.object.compute_poi(poi_objects)
        else:
            poi = bproc.object.compute_poi(scene_objects)
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check 0.3m minimum distance to objects
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

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
    
    # Write scene report immediately to CSV
    import json
    total_instances = sum(obj_data['count'] for obj_data in scene_report['objects'].values())
    object_data_json = json.dumps(scene_report['objects'], indent=2)
    
    row = {
        'scene_id': scene_report['scene_id'],
        'num_object_classes': scene_report['num_object_classes'],
        'num_images_planned': scene_report['num_images_planned'],
        'num_images_generated': scene_report['num_images_generated'],
        'total_object_instances': total_instances,
        'object_data': object_data_json
    }
    
    # Append to CSV file
    with open(report_filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'scene_id', 'num_object_classes', 'num_images_planned', 'num_images_generated',
            'total_object_instances', 'object_data'
        ])
        writer.writerow(row)
    
    # Add scene report to run report for final summary
    run_report.append(scene_report)
    print(f"Scene {i+1} completed: {scene_report['num_images_generated']}/{scene_report['num_images_planned']} images generated")
    print(f"Report updated: {report_filename}")
    
    # Memory cleanup - clear scene data
    scene_report = None
    object_instance_counts = None

# Generate final summary
end = time.time()
total_runtime = end - start