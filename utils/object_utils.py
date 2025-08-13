import numpy as np

def apply_random_sizes(scene_objects, config):
    '''Generate random sizes for each unique object class in the scene, but
    all instances of the same object class have the same size'''

    object_sizes = {}
    for obj in scene_objects:
        obj_id = obj.get_cp("obj_id")
        if obj_id not in object_sizes:
            # Generate random size for this object class in this scene
            random_size = np.random.uniform(config['object_parameters']['min_size'], config['object_parameters']['max_size'])
            object_sizes[obj_id] = random_size

    # Apply the consistent size to all instances of each object class
    for obj in scene_objects:
        obj_id = obj.get_cp("obj_id")
        target_size = object_sizes.get(obj_id)
        obj.set_scale([target_size, target_size, target_size])
    
    return object_sizes

def apply_material_randomization(scene_objects, config):
    for obj in scene_objects:
        mat = obj.get_materials()[0]        
        mat.set_principled_shader_value("Roughness", 
            np.random.uniform(config['material_randomization']['roughness_min'], 
                            config['material_randomization']['roughness_max']))
        mat.set_principled_shader_value("Metallic", 
            np.random.uniform(config['material_randomization']['metallic_min'], 
                            config['material_randomization']['metallic_max']))

def setup_physics(scene_objects):
    for obj in scene_objects:
        obj.set_shading_mode('auto')
        obj.enable_rigidbody(True, mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99, collision_shape='CONVEX_HULL')
        obj.hide(False)

def cleanup_scene_objects(scene_objects):
    for obj in scene_objects:      
        obj.delete()
