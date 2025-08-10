import blenderproc as bproc
import argparse
import os
import sys
import numpy as np
import glob
import random
import yaml
from tqdm import tqdm

# Add current directory to Python path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils import create_trunc_poisson_pmf, get_obj_id, get_category, get_name

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="config.yaml", help="Path to configuration file")
args = parser.parse_args()

# Load configuration file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Create truncated Poisson PMF for instance sampling
instance_values, instance_probs = create_trunc_poisson_pmf(
    lambda_param=config['poisson_lambda'],
    shift=config['poisson_shift'],
    min_val=1,
    max_val=config['max_instances_per_object']
)

bproc.init()
bproc.camera.set_resolution(config['resolution'][0], config['resolution'][1])

def load_model_paths():
    models_path = config['models_path']
    all_models = {}
    
    gso_path = os.path.join(models_path, 'gso_simplified', '**', '*.glb')
    gso_models = glob.glob(gso_path, recursive=True)
    all_models['gso'] = gso_models
    print(f"Found {len(gso_models)} GSO models")
    
    objaverse_path = os.path.join(models_path, 'objaverse_simplified', '**', '*.glb')
    objaverse_models = glob.glob(objaverse_path, recursive=True)
    all_models['objaverse'] = objaverse_models
    print(f"Found {len(objaverse_models)} Objaverse models")
    
    oo3d_path = os.path.join(models_path, 'oo3d_simplified', '**', '*.glb')
    oo3d_models = glob.glob(oo3d_path, recursive=True)
    all_models['oo3d'] = oo3d_models
    print(f"Found {len(oo3d_models)} OmniObject3D models")
    
    total_models = len(gso_models) + len(objaverse_models) + len(oo3d_models)
    print(f"Total models available: {total_models}")
    
    return all_models

def load_random_models(all_models, num_models):
    all_model_paths = []
    for source, paths in all_models.items():
        all_model_paths.extend([(path, source) for path in paths])
    
    if len(all_model_paths) == 0:
        print("Error: No models found!")
        return []
    
    num_to_sample = min(num_models, len(all_model_paths))
    selected_models = random.sample(all_model_paths, num_to_sample)
    
    all_instances = []
    max_instances = config['max_instances_per_object']
    max_total_instances = config['max_total_instances']
    total_instances_created = 0
    print(f"Loading {num_to_sample} random models with up to {max_instances} instances each (max total: {max_total_instances})...")
    
    loaded_count = 0
    for i, (model_path, source) in enumerate(selected_models):
        # Calculate how many base objects are still coming (including current one)
        remaining_base_objects = num_to_sample - i
        
        try:
            objects = bproc.loader.load_obj(model_path)
        except Exception as e:
            print(f"  Warning: Failed to load {source} model {model_path}: {e}")
            continue
        
        # Keep only mesh objects, delete empty objects
        valid_objects = []
        for obj in objects:
            if obj.get_mesh() is not None:
                valid_objects.append(obj)
            else:
                obj.delete()
        
        if valid_objects:
            base_obj = valid_objects[0]
            
            # Models are normalized to 1x1x1 unit box, scale to real-world sizes
            bbox = base_obj.get_bound_box()
            dimensions = [
                max(corner[0] for corner in bbox) - min(corner[0] for corner in bbox),
                max(corner[1] for corner in bbox) - min(corner[1] for corner in bbox),
                max(corner[2] for corner in bbox) - min(corner[2] for corner in bbox)
            ]
            max_dim = max(dimensions)
            if abs(max_dim - 1.0) > 0.0001:
                print(f"Model is not normalized, scaling to 0.15m size! max_dim = {max_dim}")
                # Normalize to unit size
                normalize_factor = 1.0 / max_dim
                base_obj.set_scale([normalize_factor, normalize_factor, normalize_factor])
                print(f"  Loaded {source} model with original size {max_dim:.3f}m, normalized to unit size")

            # Set properties
            base_obj.set_cp("dataset_source", source)
            base_obj.set_cp("model_path", model_path)
            
            obj_id = get_obj_id(model_path)
            base_obj.set_cp("obj_id", obj_id)
            base_obj.set_cp("category_id", obj_id) # Obj_id in scene_gt.json is created from category_id

            # Custom properties
            category = get_category(model_path)
            name = get_name(model_path)
            base_obj.set_cp("category", category)
            base_obj.set_cp("model_name", name)

            # Choose number of instances using precomputed PMF
            # But limit based on remaining total instance budget and remaining base objects
            remaining_instances = max_total_instances - total_instances_created
            
            # Reserve at least 1 instance for each remaining base object (including this one)
            instances_available_for_extras = remaining_instances - remaining_base_objects
            
            if instances_available_for_extras <= 0:
                # Only room for base objects, no extra instances
                num_instances = 1
                print(f"  Creating 1 instance of {source} model (base only - reserving space for {remaining_base_objects-1} more base objects)")
            else:
                # Normal instance selection, but cap at available budget for extras + 1 base
                desired_instances = np.random.choice(instance_values, p=instance_probs)
                max_instances_for_this_object = min(desired_instances, instances_available_for_extras + 1)
                num_instances = max_instances_for_this_object
                print(f"  Creating {num_instances} instances of {source} model (total so far: {total_instances_created + num_instances}/{max_total_instances})")
            
            # Create instances
            for instance_id in range(1, num_instances + 1):
                if instance_id == 1:
                    instance_obj = base_obj
                else:
                    instance_obj = base_obj.duplicate()
                
                # Set instance-specific properties
                instance_obj.set_cp("instance_id", instance_id)
                
                # Initially hide all objects
                instance_obj.hide(True)
                
                all_instances.append(instance_obj)
                total_instances_created += 1
            
            # Clean up extra objects
            for extra_obj in valid_objects[1:]:
                extra_obj.delete()
            
            loaded_count += 1
    
    print(f"Successfully loaded {loaded_count}/{num_to_sample} models ({len(all_instances)} total instances)")
    return all_instances

print("Scanning for available models...")
all_model_paths = load_model_paths()
    
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

light_point = bproc.types.Light()
light_point.set_energy(200)

folder = config['cc_textures_path']
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

# Function samples 6-DoF poses - adjusted for properly sized objects
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-config['position_sampling']['max_radius'], -config['position_sampling']['max_radius'], 0.0],
                            [-config['position_sampling']['min_radius'], -config['position_sampling']['min_radius'], 0.0])
    max = np.random.uniform([config['position_sampling']['min_radius'], config['position_sampling']['min_radius'], config['position_sampling']['min_height']], 
                            [config['position_sampling']['max_radius'], config['position_sampling']['max_radius'], config['position_sampling']['max_height']])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

from blenderproc.python.utility.CollisionUtility import CollisionUtility
from blenderproc.python.types.MeshObjectUtility import get_all_mesh_objects

def sample_poses_skip_inside(objects_to_sample, sample_pose_func,
                             objects_to_check_collisions=None, max_tries=1000):
    if objects_to_check_collisions is None:
        objects_to_check_collisions = get_all_mesh_objects()

    # Only check collisions against already placed objects
    objects_already = list(set(objects_to_check_collisions) - set(objects_to_sample))
    bvh_cache = {}

    for obj in objects_to_sample:
        for _ in range(max_tries):
            sample_pose_func(obj)
            bvh_cache.pop(obj.get_name(), None)

            # Skip the “inside” part of the test for all obstacles
            no_collision = CollisionUtility.check_intersections(
                obj,
                bvh_cache,
                objects_already,
                list_of_objects_with_no_inside_check=objects_already
            )
            if no_collision:
                break
        objects_already.append(obj)
    
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

print("Starting scene generation...")

for i in range(config['num_scenes']):
    
    # Randomly select number of object classes and images for this scene
    num_object_classes = np.random.randint(config['min_object_classes_per_scene'], 
                                         config['max_object_classes_per_scene'] + 1)
    num_images = np.random.randint(config['min_images_per_scene'], 
                                 config['max_images_per_scene'] + 1)
    
    print(f"Scene {i+1}: {num_object_classes} object classes, {num_images} images")

    # Initialize scene report data
    scene_report = {
        'scene_id': i + 1,
        'num_object_classes': num_object_classes,
        'num_images_planned': num_images,
        'num_images_generated': 0,
        'objects': [],
        'object_sizes': {},
        'cc_texture': '',
        'light_emission_strength': 0,
        'light_emission_color': [],
        'light_point_color': [],
        'light_point_location': []
    }

    # Clear camera poses from previous scene
    bproc.utility.reset_keyframes()

    scene_objects = load_random_models(all_model_paths, num_object_classes)

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
            random_size = np.random.uniform(config['min_object_size'], config['max_object_size'])
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
        # mat = obj.get_materials()[0]        
        # mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        # mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0))
        obj.set_shading_mode('auto')
        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99, collision_shape='CONVEX_HULL')
        obj.hide(False)
    
    # Record lighting parameters
    emission_strength = np.random.uniform(3,6)
    emission_color = np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0])
    scene_report['light_emission_strength'] = emission_strength
    scene_report['light_emission_color'] = emission_color.tolist()
    
    light_plane_material.make_emissive(emission_strength=emission_strength, 
                                    emission_color=emission_color)  
    light_plane.replace_materials(light_plane_material)
    
    point_light_color = np.random.uniform([0.5,0.5,0.5],[1,1,1])
    scene_report['light_point_color'] = point_light_color.tolist()
    light_point.set_color(point_light_color)
    
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    scene_report['light_point_location'] = location.tolist()
    light_point.set_location(location)

    random_cc_texture = np.random.choice(cc_textures)
    # Store texture name in report
    scene_report['cc_texture'] = random_cc_texture.get_name() if hasattr(random_cc_texture, 'get_name') else str(random_cc_texture)
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
    
    sample_poses_skip_inside(scene_objects, sample_pose_func, max_tries=1000)
        
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

    bproc.writer.write_bop(os.path.join(config['output_dir'], 'bop_data'),
                           target_objects = scene_objects,
                           dataset = 'clutter6d',
                           depth_scale = 0.1,
                           depths = data["depth"],
                           colors = data["colors"], 
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10)
    
    for obj in scene_objects:      
        obj.disable_rigidbody()
        obj.hide(True)
    
    # Add scene report to run report
    run_report.append(scene_report)
    print(f"Scene {i+1} completed: {scene_report['num_images_generated']}/{scene_report['num_images_planned']} images generated")

# Generate comprehensive report
end = time.time()
total_runtime = end - start

print(f"Scene generation completed in {total_runtime:.2f} seconds")

# Create timestamp for report filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Generate detailed CSV report
report_filename = os.path.join(config['output_dir'], f'generation_report_{timestamp}.csv')
with open(report_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = [
        'scene_id', 'num_object_classes', 'num_images_planned', 'num_images_generated',
        'total_object_instances', 'cc_texture', 'light_emission_strength',
        'light_emission_color_r', 'light_emission_color_g', 'light_emission_color_b', 'light_emission_color_a',
        'light_point_color_r', 'light_point_color_g', 'light_point_color_b',
        'light_point_location_x', 'light_point_location_y', 'light_point_location_z',
        'object_data'  # Will contain detailed object information as JSON string
    ]
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for scene in run_report:
        # Calculate total instances
        total_instances = sum(obj_data['count'] for obj_data in scene['objects'].values())
        
        # Prepare object data as JSON string for CSV
        import json
        object_data_json = json.dumps(scene['objects'], indent=2)
        
        row = {
            'scene_id': scene['scene_id'],
            'num_object_classes': scene['num_object_classes'],
            'num_images_planned': scene['num_images_planned'],
            'num_images_generated': scene['num_images_generated'],
            'total_object_instances': total_instances,
            'cc_texture': scene['cc_texture'],
            'light_emission_strength': scene['light_emission_strength'],
            'light_emission_color_r': scene['light_emission_color'][0],
            'light_emission_color_g': scene['light_emission_color'][1],
            'light_emission_color_b': scene['light_emission_color'][2],
            'light_emission_color_a': scene['light_emission_color'][3],
            'light_point_color_r': scene['light_point_color'][0],
            'light_point_color_g': scene['light_point_color'][1],
            'light_point_color_b': scene['light_point_color'][2],
            'light_point_location_x': scene['light_point_location'][0],
            'light_point_location_y': scene['light_point_location'][1],
            'light_point_location_z': scene['light_point_location'][2],
            'object_data': object_data_json
        }
        writer.writerow(row)

# Generate detailed text report
text_report_filename = os.path.join(config['output_dir'], f'generation_report_{timestamp}.txt')
with open(text_report_filename, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("CLUTTER6D SCENE GENERATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Runtime: {total_runtime:.2f} seconds\n")
    f.write(f"Configuration File: {args.config}\n")
    f.write(f"Total Scenes Generated: {len(run_report)}\n\n")
    
    # Summary statistics
    total_images_planned = sum(scene['num_images_planned'] for scene in run_report)
    total_images_generated = sum(scene['num_images_generated'] for scene in run_report)
    total_instances = sum(sum(obj_data['count'] for obj_data in scene['objects'].values()) for scene in run_report)
    total_unique_objects = len(set(obj_id for scene in run_report for obj_id in scene['objects'].keys()))
    
    f.write("SUMMARY STATISTICS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total Images Planned: {total_images_planned}\n")
    f.write(f"Total Images Generated: {total_images_generated}\n")
    f.write(f"Image Generation Success Rate: {(total_images_generated/total_images_planned*100):.1f}%\n")
    f.write(f"Total Object Instances: {total_instances}\n")
    f.write(f"Total Unique Objects Used: {total_unique_objects}\n\n")
    
    # Per-scene details
    f.write("SCENE DETAILS:\n")
    f.write("=" * 80 + "\n")
    
    for scene in run_report:
        f.write(f"\nSCENE {scene['scene_id']}:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Object Classes: {scene['num_object_classes']}\n")
        f.write(f"Images Planned: {scene['num_images_planned']}\n")
        f.write(f"Images Generated: {scene['num_images_generated']}\n")
        f.write(f"CC Texture: {scene['cc_texture']}\n")
        f.write(f"Light Emission Strength: {scene['light_emission_strength']:.3f}\n")
        f.write(f"Light Emission Color (RGBA): {scene['light_emission_color']}\n")
        f.write(f"Point Light Color (RGB): {scene['light_point_color']}\n")
        f.write(f"Point Light Location (XYZ): {scene['light_point_location']}\n\n")
        
        f.write("Objects in Scene:\n")
        for obj_id, obj_data in scene['objects'].items():
            f.write(f"  - Object ID: {obj_id}\n")
            f.write(f"    Name: {obj_data['name']}\n")
            f.write(f"    Category: {obj_data['category']}\n")
            f.write(f"    Dataset Source: {obj_data['dataset_source']}\n")
            f.write(f"    Instances: {obj_data['count']}\n")
            f.write(f"    Size: {obj_data['size']:.3f}m\n")
            f.write(f"    Model Path: {obj_data['model_path']}\n\n")

print(f"\nGeneration reports saved:")
print(f"  CSV Report: {report_filename}")
print(f"  Text Report: {text_report_filename}")
print(f"\nTotal scenes: {len(run_report)}")
print(f"Total images planned: {sum(scene['num_images_planned'] for scene in run_report)}")
print(f"Total images generated: {sum(scene['num_images_generated'] for scene in run_report)}")