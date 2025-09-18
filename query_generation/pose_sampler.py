import blenderproc as bproc
import numpy as np
from blenderproc.python.utility.CollisionUtility import CollisionUtility
from blenderproc.python.types.MeshObjectUtility import get_all_mesh_objects

def sample_pose_func(obj: bproc.types.MeshObject, config):
    min = np.random.uniform([-config['position_sampling']['max_radius'], -config['position_sampling']['max_radius'], 0.0],
                            [-config['position_sampling']['min_radius'], -config['position_sampling']['min_radius'], 0.0])
    max = np.random.uniform([config['position_sampling']['min_radius'], config['position_sampling']['min_radius'], config['position_sampling']['min_height']], 
                            [config['position_sampling']['max_radius'], config['position_sampling']['max_radius'], config['position_sampling']['max_height']])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

def sample_poses(objects_to_sample, objects_to_check_collisions=None, max_tries=1000, config=None):
    '''Similar to bproc.object.sample_poses, but no "inside" check,
    for faster sampling with non-watertight objects'''

    if objects_to_check_collisions is None:
        objects_to_check_collisions = get_all_mesh_objects()

    objects_already = list(set(objects_to_check_collisions) - set(objects_to_sample))
    bvh_cache = {}

    for obj in objects_to_sample:
        for _ in range(max_tries):
            sample_pose_func(obj, config)
            bvh_cache.pop(obj.get_name(), None)

            # Skip the "inside" part of the test for all obstacles
            no_collision = CollisionUtility.check_intersections(
                obj,
                bvh_cache,
                objects_already,
                list_of_objects_with_no_inside_check=objects_already
            )
            if no_collision:
                break
        objects_already.append(obj)
