import blenderproc as bproc
import glob
import numpy as np
import random
import os
from .helpers import get_obj_id, get_category, get_name

def load_model_paths(config):
    models_path = config['dataset']['models_path']
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

def load_random_models(all_models, num_models, config, instance_values, instance_probs):
    all_model_paths = []
    for source, paths in all_models.items():
        all_model_paths.extend([(path, source) for path in paths])
    
    if len(all_model_paths) == 0:
        print("Error: No models found!")
        return []
    
    num_to_sample = min(num_models, len(all_model_paths))
    selected_models = random.sample(all_model_paths, num_to_sample)
    
    all_instances = []
    max_instances = config['object_parameters']['max_instances_per_object']
    max_total_instances = config['object_parameters']['max_total_instances']
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
