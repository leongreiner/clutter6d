import blenderproc as bproc
import numpy as np
from random import random

def get_camera_radius_and_room_size(config, cam_K=None):
    # Calculate radius adjustment based on camera intrinsics
    if cam_K is not None:
        fx = cam_K[0, 0]
        
        # BlenderProc default fx for the given resolution (roughly focal length in pixels)
        width = config['scene_parameters']['resolution'][0]
        fx_baseline = width * 0.5 / np.tan(np.radians(49.134) / 2)  # BlenderProc default FOV ~49.134 degrees
        scale = fx /fx_baseline
        
        # Adjust radius based on focal length
        radius_min = max(0.01, config['camera']['radius_min'] * scale)
        radius_max = max(radius_min + 0.1, config['camera']['radius_max'] * scale)
    else:
        radius_min = config['camera']['radius_min']
        radius_max = config['camera']['radius_max']

    room_size = max(2, radius_max + 0.1)
    return radius_min, radius_max, room_size

def set_random_camera_intrinsics(width, height):
    # Random camera intrinsic per scene
    fx = 400 + random() * 1600  # Random value between [400, 2000)
    fx_fy_offset = 3 * (1 - 2 * random())  # between -3 and 3
    cam_K = np.array(
        [
            [fx, 0.0, width / 2],
            [0.0, fx + fx_fy_offset, height / 2],
            [0.0, 0.0, 1.0],
        ]
    )
    bproc.camera.set_intrinsics_from_K_matrix(cam_K, width, height)
    return cam_K

def generate_camera_poses(config, scene_objects, num_images, radius_min, radius_max):
    # BVH tree for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(scene_objects)

    cam_poses = 0
    while cam_poses < num_images:
        location = bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=radius_min,
            radius_max=radius_max,
            elevation_min=config['camera']['elevation_min'],
            elevation_max=config['camera']['elevation_max']
        )
        
        # When there are at least 15 objects, compute POI from a random selection of 15 objects to get more varied camera viewpoints
        if len(scene_objects) >= 15:
            poi_objects = np.random.choice(scene_objects, size=15, replace=False)
            poi = bproc.object.compute_poi(poi_objects)
        else:
            poi = bproc.object.compute_poi(scene_objects)
        
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            poi - location, 
            inplane_rot=np.random.uniform(-3.14159, 3.14159)
        )
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check 0.3m minimum distance to objects
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    return cam_poses
