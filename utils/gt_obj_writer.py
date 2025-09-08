import os
import json
import numpy as np
import blenderproc as bproc

def write_scene_gt_obj(output_dir, dataset_name, scene_objects, object_sizes, total_images_so_far):
    """
    Write scene_gt_obj.json file that contains the same information as scene_gt.json
    but with additional obj_size information for each object instance.
    
    Args:
        output_dir (str): Base output directory
        dataset_name (str): Name of the dataset
        scene_objects (list): List of BlenderProc objects in the scene
        object_sizes (dict): Dictionary mapping obj_id to size values
        total_images_so_far (int): Total number of images generated so far (before this scene)
    """
    
    # Get the number of camera poses from the current scene
    cam_poses = bproc.camera.get_camera_poses()
    current_scene_images = len(cam_poses)
    
    # Calculate which folders this scene's images will be written to
    # BOP format uses 1000 images per folder (000000, 000001, etc.)
    start_image_idx = total_images_so_far
    end_image_idx = total_images_so_far + current_scene_images
    
    start_folder = start_image_idx // 1000
    end_folder = (end_image_idx - 1) // 1000
    
    # Group images by folder
    folder_images = {}
    for cam_idx in range(current_scene_images):
        global_image_idx = start_image_idx + cam_idx
        folder_idx = global_image_idx // 1000
        local_image_idx = global_image_idx % 1000
        
        if folder_idx not in folder_images:
            folder_images[folder_idx] = []
        folder_images[folder_idx].append((cam_idx, local_image_idx))
    
    # Write scene_gt_obj.json for each folder that contains images from this scene
    for folder_idx, image_list in folder_images.items():
        scene_folder = os.path.join(output_dir, dataset_name, "train_pbr", f"{folder_idx:06d}")
        os.makedirs(scene_folder, exist_ok=True)
        
        # Check if scene_gt_obj.json already exists, if so load it
        output_file = os.path.join(scene_folder, "scene_gt_obj.json")
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                scene_gt_obj = json.load(f)
        else:
            scene_gt_obj = {}
        
        # Add entries for this scene's images in this folder
        for cam_idx, local_image_idx in image_list:
            scene_gt_obj[str(local_image_idx)] = []
            
            # For each object in the scene
            for obj in scene_objects:
                obj_id = obj.get_cp("obj_id")
                
                # Get the object's pose relative to the camera
                cam2world_matrix = cam_poses[cam_idx]
                world2cam_matrix = np.linalg.inv(cam2world_matrix)
                
                # Get object's world transformation matrix
                obj_world_matrix = obj.get_local2world_mat()
                
                # Calculate object pose in camera coordinates
                obj_cam_matrix = world2cam_matrix @ obj_world_matrix
                
                # Extract rotation (flatten 3x3 rotation matrix to list) and translation
                cam_R_m2c = obj_cam_matrix[:3, :3].flatten().tolist()
                # Convert translation from meters to millimeters (BOP format)
                cam_t_m2c = (obj_cam_matrix[:3, 3] * 1000).tolist()
                
                # Get object size
                obj_size = object_sizes.get(obj_id, 1.0)
                
                # Create entry for this object instance (matching BOP scene_gt.json format + obj_size)
                obj_entry = {
                    "cam_R_m2c": cam_R_m2c,
                    "cam_t_m2c": cam_t_m2c,
                    "obj_id": int(obj_id),
                    "obj_size": float(obj_size)
                }
                
                scene_gt_obj[str(local_image_idx)].append(obj_entry)
        
        # Write the updated JSON file
        with open(output_file, 'w') as f:
            json.dump(scene_gt_obj, f, indent=2)
        
        print(f"Updated scene_gt_obj.json in folder {folder_idx:06d}")
