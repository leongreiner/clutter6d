"""
Annotation Generator for creating ground truth 6DoF poses and segmentation masks.
Generates COCO-style annotations with 6DoF pose information.
"""

import logging
import json
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from pyquaternion import Quaternion
import blenderproc as bproc
import bpy

logger = logging.getLogger(__name__)


class AnnotationGenerator:
    """Generates ground truth annotations for 6DoF pose estimation."""
    
    def __init__(self, config: Dict):
        """
        Initialize the AnnotationGenerator.
        
        Args:
            config: Configuration dictionary containing annotation settings
        """
        self.config = config
        self.output_config = config['output']
        self.camera_config = config['camera']
        
        # Annotation format
        self.annotation_format = self.output_config['annotation_format']
        self.image_size = self.camera_config['image_size']
        
        # Internal state
        self.current_scene_id = 0
        self.annotation_id_counter = 0
        
    def generate_annotations(self, scene_data: Dict, output_dir: Path) -> Dict:
        """
        Generate complete annotations for a scene.
        
        Args:
            scene_data: Scene information including objects and camera
            output_dir: Output directory for annotations
            
        Returns:
            Annotation data dictionary
        """
        logger.info(f"Generating annotations for scene {self.current_scene_id}")
        
        # Render required data
        render_data = self._render_annotation_data()
        
        # Generate different annotation components
        annotations = {
            'scene_id': self.current_scene_id,
            'image_info': self._generate_image_info(scene_data),
            'camera_info': self._generate_camera_info(scene_data),
            'objects': self._generate_object_annotations(scene_data, render_data),
            'segmentation': self._generate_segmentation_annotations(render_data),
            'metadata': self._generate_scene_metadata(scene_data)
        }
        
        # Save annotations in specified format
        self._save_annotations(annotations, output_dir)
        
        self.current_scene_id += 1
        return annotations
        
    def _render_annotation_data(self) -> Dict:
        """Render necessary data for annotation generation."""
        logger.debug("Rendering annotation data...")
        
        # Enable segmentation rendering
        bproc.renderer.enable_segmentation_output(map_by=['instance', 'class'])
        
        # Render RGB, depth, and segmentation
        data = bproc.renderer.render()
        
        render_data = {
            'colors': data['colors'],
            'depth': data['depth'],
            'instance_segmaps': data['instance_segmaps'],
            'class_segmaps': data.get('class_segmaps', None)
        }
        
        return render_data
        
    def _generate_image_info(self, scene_data: Dict) -> Dict:
        """Generate image metadata information."""
        return {
            'width': self.image_size[0],
            'height': self.image_size[1],
            'file_name': f"scene_{self.current_scene_id:06d}.png",
            'depth_file_name': f"scene_{self.current_scene_id:06d}_depth.png",
            'segmentation_file_name': f"scene_{self.current_scene_id:06d}_seg.png"
        }
        
    def _generate_camera_info(self, scene_data: Dict) -> Dict:
        """Generate camera intrinsic and extrinsic parameters."""
        camera = scene_data.get('camera')
        if not camera:
            logger.warning("No camera data found in scene")
            return {}
            
        # Get camera intrinsics
        K = bproc.camera.get_intrinsics_as_K_matrix()
        
        # Get camera pose (extrinsics)
        cam_pose = bproc.camera.get_camera_pose()
        
        camera_info = {
            'intrinsics': {
                'K': K.tolist(),
                'fx': K[0, 0],
                'fy': K[1, 1],
                'cx': K[0, 2],
                'cy': K[1, 2]
            },
            'extrinsics': {
                'camera_pose': cam_pose.tolist(),
                'rotation_matrix': cam_pose[:3, :3].tolist(),
                'translation': cam_pose[:3, 3].tolist()
            },
            'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]  # Assuming no distortion
        }
        
        return camera_info
        
    def _generate_object_annotations(self, scene_data: Dict, render_data: Dict) -> List[Dict]:
        """Generate 6DoF pose annotations for all objects."""
        objects_annotations = []
        
        placement_results = scene_data.get('placement_results', [])
        instance_segmaps = render_data['instance_segmaps'][0]  # First camera view
        
        for i, result in enumerate(placement_results):
            obj_data = result['object_data']
            blender_obj = result['blender_object']
            final_pose = result['final_pose']
            
            # Calculate visibility
            visibility_info = self._calculate_object_visibility(
                blender_obj, instance_segmaps
            )
            
            if visibility_info['visibility_ratio'] < self.config['scene']['min_visibility_threshold']:
                logger.debug(f"Skipping object {i} due to low visibility: {visibility_info['visibility_ratio']:.3f}")
                continue
                
            # Generate 6DoF pose annotation
            pose_annotation = self._generate_6dof_pose(blender_obj, final_pose)
            
            # Generate bounding box
            bbox = self._generate_2d_bbox(blender_obj, visibility_info)
            
            object_annotation = {
                'id': self.annotation_id_counter,
                'object_id': obj_data['id'],
                'instance_id': obj_data['instance_id'],
                'category': obj_data['class'],
                'collection': obj_data['collection'],
                'pose_6d': pose_annotation,
                'bbox_2d': bbox,
                'visibility': visibility_info,
                'area': bbox['area'] if bbox else 0,
                'iscrowd': 0
            }
            
            objects_annotations.append(object_annotation)
            self.annotation_id_counter += 1
            
        return objects_annotations
        
    def _generate_6dof_pose(self, blender_obj: bpy.types.Object, final_pose: Dict) -> Dict:
        """Generate 6DoF pose annotation in multiple formats."""
        # Get world transformation matrix
        transform_matrix = np.array(blender_obj.matrix_world)
        
        # Extract translation
        translation = transform_matrix[:3, 3]
        
        # Extract rotation matrix
        rotation_matrix = transform_matrix[:3, :3]
        
        # Convert to quaternion
        quaternion = Quaternion(matrix=rotation_matrix)
        
        # Convert to axis-angle representation
        axis_angle = quaternion.axis * quaternion.angle
        
        # Convert to Euler angles
        euler_angles = blender_obj.rotation_euler
        
        pose_6d = {
            'translation': translation.tolist(),
            'rotation_matrix': rotation_matrix.tolist(),
            'quaternion': [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
            'axis_angle': axis_angle.tolist(),
            'euler_angles': list(euler_angles),
            'transform_matrix': transform_matrix.tolist(),
            'scale': list(blender_obj.scale)
        }
        
        return pose_6d
        
    def _calculate_object_visibility(self, blender_obj: bpy.types.Object, 
                                   instance_segmap: np.ndarray) -> Dict:
        """Calculate object visibility metrics."""
        # Get object's instance ID for segmentation
        instance_id = blender_obj.get('instance_id', 0)
        
        # Count pixels belonging to this object
        object_pixels = np.sum(instance_segmap == instance_id)
        total_pixels = instance_segmap.shape[0] * instance_segmap.shape[1]
        
        # Calculate bounding box of visible pixels
        if object_pixels > 0:
            y_coords, x_coords = np.where(instance_segmap == instance_id)
            bbox_min = [int(np.min(x_coords)), int(np.min(y_coords))]
            bbox_max = [int(np.max(x_coords)), int(np.max(y_coords))]
            bbox_size = [bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1]]
        else:
            bbox_min = [0, 0]
            bbox_max = [0, 0]
            bbox_size = [0, 0]
            
        # Estimate total object area (projected)
        # This is an approximation - in practice you might want to render
        # the object separately to get the true projected area
        estimated_total_area = max(object_pixels, 100)  # Minimum assumption
        
        visibility_info = {
            'visible_pixels': int(object_pixels),
            'total_pixels': int(total_pixels),
            'visibility_ratio': object_pixels / estimated_total_area,
            'screen_coverage': object_pixels / total_pixels,
            'bbox_visible': {
                'min': bbox_min,
                'max': bbox_max,
                'size': bbox_size
            }
        }
        
        return visibility_info
        
    def _generate_2d_bbox(self, blender_obj: bpy.types.Object, 
                         visibility_info: Dict) -> Optional[Dict]:
        """Generate 2D bounding box from visibility information."""
        if visibility_info['visible_pixels'] == 0:
            return None
            
        bbox_visible = visibility_info['bbox_visible']
        
        # COCO format: [x, y, width, height]
        x = bbox_visible['min'][0]
        y = bbox_visible['min'][1]
        width = bbox_visible['size'][0]
        height = bbox_visible['size'][1]
        
        bbox = {
            'bbox': [x, y, width, height],
            'area': width * height,
            'format': 'COCO'  # [x, y, width, height]
        }
        
        return bbox
        
    def _generate_segmentation_annotations(self, render_data: Dict) -> Dict:
        """Generate segmentation mask information."""
        instance_segmap = render_data['instance_segmaps'][0]
        class_segmap = render_data.get('class_segmaps', [None])[0]
        
        segmentation_info = {
            'instance_segmentation': {
                'shape': instance_segmap.shape,
                'unique_instances': len(np.unique(instance_segmap)),
                'file_name': f"scene_{self.current_scene_id:06d}_instance_seg.png"
            }
        }
        
        if class_segmap is not None:
            segmentation_info['class_segmentation'] = {
                'shape': class_segmap.shape,
                'unique_classes': len(np.unique(class_segmap)),
                'file_name': f"scene_{self.current_scene_id:06d}_class_seg.png"
            }
            
        return segmentation_info
        
    def _generate_scene_metadata(self, scene_data: Dict) -> Dict:
        """Generate scene-level metadata."""
        placement_results = scene_data.get('placement_results', [])
        
        metadata = {
            'total_objects': len(placement_results),
            'object_classes': list(set(r['object_data']['class'] for r in placement_results)),
            'collections_used': list(set(r['object_data']['collection'] for r in placement_results)),
            'scene_bounds': self.config['scene']['scene_bounds'],
            'physics_enabled': self.config['physics']['enable_physics'],
            'random_seed': scene_data.get('random_seed'),
            'generation_timestamp': scene_data.get('timestamp'),
            'lighting_config': scene_data.get('lighting_config', {}),
            'camera_config': scene_data.get('camera_config', {})
        }
        
        return metadata
        
    def _save_annotations(self, annotations: Dict, output_dir: Path):
        """Save annotations in the specified format."""
        scene_id = annotations['scene_id']
        
        if self.annotation_format.lower() == 'coco':
            self._save_coco_annotations(annotations, output_dir, scene_id)
        elif self.annotation_format.lower() == 'yolo':
            self._save_yolo_annotations(annotations, output_dir, scene_id)
        else:
            self._save_custom_annotations(annotations, output_dir, scene_id)
            
    def _save_coco_annotations(self, annotations: Dict, output_dir: Path, scene_id: int):
        """Save annotations in COCO format."""
        # COCO format structure
        coco_data = {
            'info': {
                'description': 'Clutter6D Synthetic Dataset',
                'version': '1.0',
                'year': 2025,
                'contributor': 'Clutter6D Pipeline'
            },
            'images': [
                {
                    'id': scene_id,
                    'width': annotations['image_info']['width'],
                    'height': annotations['image_info']['height'],
                    'file_name': annotations['image_info']['file_name']
                }
            ],
            'annotations': [],
            'categories': []
        }
        
        # Convert object annotations to COCO format
        category_map = {}
        category_id = 1
        
        for obj_ann in annotations['objects']:
            category = obj_ann['category']
            if category not in category_map:
                category_map[category] = category_id
                coco_data['categories'].append({
                    'id': category_id,
                    'name': category,
                    'supercategory': 'object'
                })
                category_id += 1
                
            if obj_ann['bbox_2d']:
                coco_annotation = {
                    'id': obj_ann['id'],
                    'image_id': scene_id,
                    'category_id': category_map[category],
                    'bbox': obj_ann['bbox_2d']['bbox'],
                    'area': obj_ann['bbox_2d']['area'],
                    'iscrowd': 0,
                    'pose_6d': obj_ann['pose_6d'],
                    'visibility': obj_ann['visibility']
                }
                coco_data['annotations'].append(coco_annotation)
                
        # Save COCO annotations
        coco_file = output_dir / 'annotations' / f'scene_{scene_id:06d}_coco.json'
        coco_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(coco_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
            
    def _save_custom_annotations(self, annotations: Dict, output_dir: Path, scene_id: int):
        """Save annotations in custom format with full 6DoF information."""
        # Save complete annotation data
        annotation_file = output_dir / 'annotations' / f'scene_{scene_id:06d}_annotations.json'
        annotation_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f, indent=2)
            
        # Save camera info separately
        camera_file = output_dir / 'annotations' / f'scene_{scene_id:06d}_camera.json'
        with open(camera_file, 'w') as f:
            json.dump(annotations['camera_info'], f, indent=2)
            
        # Save metadata separately
        metadata_file = output_dir / 'annotations' / f'scene_{scene_id:06d}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(annotations['metadata'], f, indent=2)
            
    def reset_counters(self):
        """Reset annotation counters for new dataset generation."""
        self.current_scene_id = 0
        self.annotation_id_counter = 0
