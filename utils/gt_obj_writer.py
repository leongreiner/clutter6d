import os
import json
import numpy as np

def write_scene_gt_obj(output_dir, dataset_name, scene_objects, object_sizes, total_images_so_far, cam_poses):
    """
    Write scene_gt_obj.json file that contains the same information as scene_gt.json
    but with additional obj_size information for each object instance.
    
    Args:
        output_dir (str): Base output directory
        dataset_name (str): Name of the dataset
        scene_objects (list): List of BlenderProc objects in the scene
        object_sizes (dict): Dictionary mapping obj_id to size values
        total_images_so_far (int): Total number of images generated so far (before this scene)
        cam_poses (list): List of camera poses for this scene
    """
    
    # Get the number of camera poses from the current scene
    current_scene_images = len(cam_poses)
    
    # Calculate which folders this scene's images will be written to
    # BOP format uses 1000 images per folder (000000, 000001, etc.)
    start_image_idx = total_images_so_far
    end_image_idx = total_images_so_far + current_scene_images
    
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
        
        # Read the existing scene_gt.json file created by the BOP writer
        scene_gt_file = os.path.join(scene_folder, "scene_gt.json")
        if not os.path.exists(scene_gt_file):
            print(f"Warning: scene_gt.json not found at {scene_gt_file}")
            continue
            
        with open(scene_gt_file, 'r') as f:
            scene_gt = json.load(f)
        
        # Check if scene_gt_obj.json already exists, if so load it
        output_file = os.path.join(scene_folder, "scene_gt_obj.json")
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                scene_gt_obj = json.load(f)
        else:
            scene_gt_obj = {}
        
        # Add entries for this scene's images in this folder
        for cam_idx, local_image_idx in image_list:
            local_image_str = str(local_image_idx)
            
            # Check if this image exists in scene_gt.json
            if local_image_str in scene_gt:
                # Copy the pose data from scene_gt.json and add obj_size
                scene_gt_obj[local_image_str] = []
                
                for obj_entry in scene_gt[local_image_str]:
                    # Copy existing data
                    obj_entry_with_size = obj_entry.copy()
                    
                    # Add object size
                    obj_id = obj_entry["obj_id"]
                    obj_size = object_sizes.get(obj_id, 1.0)
                    obj_entry_with_size["obj_size"] = float(obj_size)
                    
                    scene_gt_obj[local_image_str].append(obj_entry_with_size)
        
        # Write the updated JSON file
        with open(output_file, 'w') as f:
            json.dump(scene_gt_obj, f, indent=2)
        
        print(f"Updated scene_gt_obj.json in folder {folder_idx:06d}")
