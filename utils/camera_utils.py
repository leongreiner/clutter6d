import blenderproc as bproc
import numpy as np

def generate_camera_poses(scene_objects, num_images, config):
    # BVH tree for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(scene_objects)

    cam_poses = 0
    while cam_poses < num_images:
        # Camera positioning adjusted for smaller objects
        location = bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=config['camera']['radius_min'],
            radius_max=config['camera']['radius_max'],
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
