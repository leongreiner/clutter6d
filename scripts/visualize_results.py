#!/usr/bin/env python3
"""
Visualization script for generated synthetic dataset.
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize generated synthetic dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to generated dataset directory"
    )
    
    parser.add_argument(
        "--scene-id",
        type=int,
        default=0,
        help="Scene ID to visualize"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for visualization images"
    )
    
    parser.add_argument(
        "--show-poses",
        action="store_true",
        help="Overlay 6DoF poses on images"
    )
    
    parser.add_argument(
        "--show-bboxes",
        action="store_true",
        help="Show 2D bounding boxes"
    )
    
    parser.add_argument(
        "--create-montage",
        action="store_true",
        help="Create montage of multiple scenes"
    )
    
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=9,
        help="Number of scenes for montage"
    )
    
    return parser.parse_args()


def load_scene_data(data_dir: Path, scene_id: int):
    """Load all data for a specific scene."""
    scene_data = {}
    
    # Load RGB image
    rgb_path = data_dir / 'images' / 'rgb' / f'scene_{scene_id:06d}.png'
    if rgb_path.exists():
        scene_data['rgb'] = np.array(Image.open(rgb_path))
    
    # Load depth image
    depth_path = data_dir / 'images' / 'depth' / f'scene_{scene_id:06d}_depth.png'
    if depth_path.exists():
        depth_img = np.array(Image.open(depth_path))
        # Convert back from uint16 to meters
        scene_data['depth'] = depth_img.astype(np.float32) / 1000.0
    
    # Load segmentation masks
    seg_path = data_dir / 'images' / 'segmentation' / f'scene_{scene_id:06d}_instance_seg.png'
    if seg_path.exists():
        scene_data['segmentation'] = np.array(Image.open(seg_path))
    
    # Load annotations
    annotations_path = data_dir / 'annotations' / f'scene_{scene_id:06d}_annotations.json'
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            scene_data['annotations'] = json.load(f)
    
    # Try COCO format as fallback
    coco_path = data_dir / 'annotations' / f'scene_{scene_id:06d}_coco.json'
    if coco_path.exists() and 'annotations' not in scene_data:
        with open(coco_path, 'r') as f:
            scene_data['coco_annotations'] = json.load(f)
    
    # Load camera info
    camera_path = data_dir / 'annotations' / f'scene_{scene_id:06d}_camera.json'
    if camera_path.exists():
        with open(camera_path, 'r') as f:
            scene_data['camera'] = json.load(f)
    
    return scene_data


def visualize_depth(depth_img: np.ndarray, title: str = "Depth"):
    """Visualize depth image."""
    plt.figure(figsize=(8, 6))
    
    # Clip extreme values for better visualization
    depth_vis = np.clip(depth_img, 0, np.percentile(depth_img[depth_img > 0], 95))
    
    plt.imshow(depth_vis, cmap='plasma')
    plt.colorbar(label='Depth (m)')
    plt.title(title)
    plt.axis('off')
    
    return plt.gcf()


def visualize_segmentation(seg_img: np.ndarray, title: str = "Instance Segmentation"):
    """Visualize segmentation mask with different colors per instance."""
    plt.figure(figsize=(8, 6))
    
    # Create colormap for instances
    unique_instances = np.unique(seg_img)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_instances)))
    
    # Create colored segmentation
    colored_seg = np.zeros((*seg_img.shape, 3))
    for i, instance_id in enumerate(unique_instances):
        if instance_id == 0:  # Background
            continue
        mask = seg_img == instance_id
        colored_seg[mask] = colors[i][:3]
    
    plt.imshow(colored_seg)
    plt.title(f"{title} ({len(unique_instances)-1} instances)")
    plt.axis('off')
    
    return plt.gcf()


def draw_bounding_boxes(img: np.ndarray, annotations: dict, show_labels: bool = True):
    """Draw 2D bounding boxes on image."""
    img_with_boxes = img.copy()
    
    if 'objects' in annotations:
        objects = annotations['objects']
    elif 'annotations' in annotations:  # COCO format
        objects = annotations['annotations']
    else:
        return img_with_boxes
    
    for obj in objects:
        if 'bbox_2d' in obj and obj['bbox_2d']:
            bbox = obj['bbox_2d']['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # Draw rectangle
            cv2.rectangle(img_with_boxes, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            
            if show_labels:
                # Add label
                label = obj.get('category', obj.get('object_id', 'unknown'))
                label_text = f"{label}"
                
                # Add visibility info if available
                if 'visibility' in obj:
                    vis_ratio = obj['visibility'].get('visibility_ratio', 0)
                    label_text += f" ({vis_ratio:.2f})"
                
                cv2.putText(img_with_boxes, label_text, (int(x), int(y-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return img_with_boxes


def project_pose_axes(img: np.ndarray, pose_6d: dict, camera_info: dict, axis_length: float = 0.1):
    """Project and draw 6DoF pose axes on image."""
    if 'intrinsics' not in camera_info:
        return img
        
    K = np.array(camera_info['intrinsics']['K'])
    
    # Get pose information
    translation = np.array(pose_6d['translation'])
    rotation_matrix = np.array(pose_6d['rotation_matrix'])
    
    # Define axis endpoints in object coordinate system
    origin = np.array([0, 0, 0])
    x_axis = np.array([axis_length, 0, 0])
    y_axis = np.array([0, axis_length, 0])
    z_axis = np.array([0, 0, axis_length])
    
    # Transform to world coordinates
    points_3d = np.array([origin, x_axis, y_axis, z_axis])
    points_world = (rotation_matrix @ points_3d.T).T + translation
    
    # Project to image coordinates
    points_2d = []
    for point in points_world:
        if point[2] > 0:  # Point is in front of camera
            projected = K @ point
            if projected[2] != 0:
                pixel = projected[:2] / projected[2]
                points_2d.append(pixel.astype(int))
            else:
                points_2d.append(None)
        else:
            points_2d.append(None)
    
    # Draw axes if all points are valid
    if all(p is not None for p in points_2d):
        img_with_poses = img.copy()
        origin_2d, x_2d, y_2d, z_2d = points_2d
        
        # Draw axes (X=red, Y=green, Z=blue)
        cv2.line(img_with_poses, tuple(origin_2d), tuple(x_2d), (0, 0, 255), 3)  # X - Red
        cv2.line(img_with_poses, tuple(origin_2d), tuple(y_2d), (0, 255, 0), 3)  # Y - Green
        cv2.line(img_with_poses, tuple(origin_2d), tuple(z_2d), (255, 0, 0), 3)  # Z - Blue
        
        # Draw origin point
        cv2.circle(img_with_poses, tuple(origin_2d), 5, (255, 255, 255), -1)
        
        return img_with_poses
    
    return img


def visualize_scene(scene_data: dict, show_poses: bool = False, show_bboxes: bool = False):
    """Create comprehensive visualization of a scene."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # RGB image
    if 'rgb' in scene_data:
        rgb_img = scene_data['rgb']
        
        # Add bounding boxes if requested
        if show_bboxes and 'annotations' in scene_data:
            rgb_img = draw_bounding_boxes(rgb_img, scene_data['annotations'])
        
        # Add pose axes if requested
        if show_poses and 'annotations' in scene_data and 'camera' in scene_data:
            annotations = scene_data['annotations']
            camera_info = scene_data['camera']
            
            if 'objects' in annotations:
                for obj in annotations['objects']:
                    if 'pose_6d' in obj:
                        rgb_img = project_pose_axes(rgb_img, obj['pose_6d'], camera_info)
        
        axes[0, 0].imshow(rgb_img)
        axes[0, 0].set_title('RGB Image')
        axes[0, 0].axis('off')
    
    # Depth image
    if 'depth' in scene_data:
        depth_img = scene_data['depth']
        depth_vis = np.clip(depth_img, 0, np.percentile(depth_img[depth_img > 0], 95))
        im = axes[0, 1].imshow(depth_vis, cmap='plasma')
        axes[0, 1].set_title('Depth Image')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], label='Depth (m)')
    
    # Segmentation
    if 'segmentation' in scene_data:
        seg_img = scene_data['segmentation']
        unique_instances = np.unique(seg_img)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_instances)))
        
        colored_seg = np.zeros((*seg_img.shape, 3))
        for i, instance_id in enumerate(unique_instances):
            if instance_id == 0:  # Background
                continue
            mask = seg_img == instance_id
            colored_seg[mask] = colors[i][:3]
        
        axes[1, 0].imshow(colored_seg)
        axes[1, 0].set_title(f'Instance Segmentation ({len(unique_instances)-1} objects)')
        axes[1, 0].axis('off')
    
    # Object statistics
    if 'annotations' in scene_data:
        annotations = scene_data['annotations']
        
        # Extract object information
        objects = annotations.get('objects', [])
        if not objects and 'annotations' in annotations:  # COCO format
            objects = annotations['annotations']
        
        if objects:
            # Visibility histogram
            visibilities = [obj.get('visibility', {}).get('visibility_ratio', 0) for obj in objects]
            axes[1, 1].hist(visibilities, bins=10, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Visibility Ratio')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title(f'Object Visibility Distribution\n({len(objects)} objects)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No object data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Object Statistics')
    
    plt.tight_layout()
    return fig


def create_scene_montage(data_dir: Path, scene_ids: list, output_path: str = None):
    """Create a montage of multiple scenes."""
    n_scenes = len(scene_ids)
    n_cols = int(np.ceil(np.sqrt(n_scenes)))
    n_rows = int(np.ceil(n_scenes / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_scenes == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, scene_id in enumerate(scene_ids):
        scene_data = load_scene_data(data_dir, scene_id)
        
        if 'rgb' in scene_data:
            axes[i].imshow(scene_data['rgb'])
        
        # Add scene info
        title = f'Scene {scene_id}'
        if 'annotations' in scene_data:
            objects = scene_data['annotations'].get('objects', [])
            if not objects and 'annotations' in scene_data['annotations']:
                objects = scene_data['annotations']['annotations']
            title += f'\n({len(objects)} objects)'
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_scenes, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Montage saved to {output_path}")
    
    return fig


def main():
    """Main function."""
    args = parse_arguments()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)
    
    print(f"Visualizing dataset from: {data_dir}")
    
    try:
        if args.create_montage:
            # Create montage of multiple scenes
            available_scenes = []
            rgb_dir = data_dir / 'images' / 'rgb'
            if rgb_dir.exists():
                for img_file in rgb_dir.glob('scene_*.png'):
                    scene_id = int(img_file.stem.split('_')[1])
                    available_scenes.append(scene_id)
            
            available_scenes.sort()
            selected_scenes = available_scenes[:args.num_scenes]
            
            print(f"Creating montage of {len(selected_scenes)} scenes...")
            
            output_path = None
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / 'scene_montage.png'
            
            fig = create_scene_montage(data_dir, selected_scenes, str(output_path) if output_path else None)
            plt.show()
            
        else:
            # Visualize single scene
            print(f"Loading scene {args.scene_id}...")
            scene_data = load_scene_data(data_dir, args.scene_id)
            
            if not scene_data:
                print(f"Error: No data found for scene {args.scene_id}")
                sys.exit(1)
            
            print(f"Visualizing scene {args.scene_id}...")
            fig = visualize_scene(scene_data, args.show_poses, args.show_bboxes)
            
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f'scene_{args.scene_id:06d}_visualization.png'
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Visualization saved to {output_path}")
            
            plt.show()
            
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
