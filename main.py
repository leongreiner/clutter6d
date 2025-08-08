import blenderproc as bproc
import argparse
import os
import sys
import shutil
import numpy as np
import glob
import random
import hashlib
import logging
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
    
    obj3d_path = os.path.join(models_path, 'oo3d_simplified', '**', '*.glb')
    obj3d_models = glob.glob(obj3d_path, recursive=True)
    all_models['obj3d'] = obj3d_models
    print(f"Found {len(obj3d_models)} OmniObject3D models")
    
    total_models = len(gso_models) + len(objaverse_models) + len(obj3d_models)
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
    print(f"Loading {num_to_sample} random models with up to {max_instances} instances each...")
    
    loaded_count = 0
    for i, (model_path, source) in enumerate(selected_models):
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

            target_size = 0.15  # Fixed size of 15cm for all objects; mm -> m
            scale_factor = target_size / max_dim if max_dim > 0 else target_size
            
            base_obj.set_scale([scale_factor, scale_factor, scale_factor])

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
            base_obj.set_cp("name", name)

            # Choose number of instances using precomputed PMF
            num_instances = np.random.choice(instance_values, p=instance_probs)
            print(f"  Creating {num_instances} instances of {source} model: {max_dim:.3f} -> {target_size:.3f}m (scale={scale_factor:.3f})")
            
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
                instance_obj.disable_rigidbody()
                
                all_instances.append(instance_obj)
            
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

# Function samples 6-DoF poses - adjusted for properly sized objects
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.4, -0.4, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.3], [0.4, 0.4, 0.5])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

print("Starting scene generation...")

for i in range(config['num_scenes']):

    scene_objects = load_random_models(all_model_paths, config['models_per_scene'])

    if not scene_objects:
        print(f"No objects loaded for scene {i+1}, skipping...")
        continue

    # Keep original GLB textures, no material randomization
    for obj in scene_objects:        
        obj.set_shading_mode('auto')
        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99, collision_shape='COMPOUND')
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
    for j, obj in enumerate(scene_objects):
        obj_path = obj.get_cp("model_path") if obj.has_cp("model_path") else "unknown"
        obj_id = obj.get_cp("obj_id") if obj.has_cp("obj_id") else "unknown"
        instance_id = obj.get_cp("instance_id") if obj.has_cp("instance_id") else "unknown"
        category = obj.get_cp("cat") if obj.has_cp("cat") else "unknown"
        name = obj.get_cp("name") if obj.has_cp("name") else "unknown"
        print(f"  {j+1:2d}. {obj.get_name()} (obj_id: {obj_id}, cat: {category}, name: {name}, instance: {instance_id}) -> {os.path.basename(obj_path)}")
    
    bproc.object.sample_poses(objects_to_sample = scene_objects,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)
            
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                    max_simulation_time=10,
                                                    check_object_interval=1,
                                                    substeps_per_frame = 20,
                                                    solver_iters=25)

    # BVH tree for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(scene_objects)

    cam_poses = 0
    while cam_poses < 25:
        # Camera positioning adjusted for smaller objects
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 0.8,   # Increased minimum distance
                                radius_max = 1.5,   # Adjusted maximum distance
                                elevation_min = 5,
                                elevation_max = 89)
        if len(scene_objects) >= 3:
            poi_objects = np.random.choice(scene_objects, size=min(15, len(scene_objects)), replace=False)
            poi = bproc.object.compute_poi(poi_objects)
        else:
            poi = bproc.object.compute_poi(scene_objects)
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check 0.3m minimum distance to objects
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    data = bproc.renderer.render()

    bproc.writer.write_bop(os.path.join(config['output_dir'], 'bop_data'),
                           target_objects = scene_objects,
                           dataset = 'clutter6d',
                           depth_scale = 1,
                           depths = data["depth"],
                           colors = data["colors"], 
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10)
    
    for obj in scene_objects:      
        obj.disable_rigidbody()
        obj.hide(True)