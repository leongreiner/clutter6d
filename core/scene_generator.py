"""
Main Scene Generator that orchestrates the entire synthetic dataset generation pipeline.
Integrates object management, physics simulation, rendering, and annotation generation.
"""

import logging
import random
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import blenderproc as bproc

from .object_manager import ObjectManager
from .physics_simulator import PhysicsSimulator
from .annotation_generator import AnnotationGenerator
from ..utils.camera_utils import CameraManager
from ..utils.lighting_utils import LightingManager
from ..utils.quality_control import QualityController

logger = logging.getLogger(__name__)


class SceneGenerator:
    """Main class for generating synthetic cluttered scenes with 6DoF annotations."""
    
    def __init__(self, config: Dict):
        """
        Initialize the SceneGenerator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scene_config = config['scene']
        self.dataset_config = config['dataset']
        
        # Initialize components
        self.object_manager = ObjectManager(config)
        self.physics_simulator = PhysicsSimulator(config)
        self.annotation_generator = AnnotationGenerator(config)
        self.camera_manager = CameraManager(config)
        self.lighting_manager = LightingManager(config)
        self.quality_controller = QualityController(config)
        
        # Generation state
        self.current_scene = 0
        self.generation_stats = {
            'total_scenes': 0,
            'successful_scenes': 0,
            'failed_scenes': 0,
            'total_objects': 0,
            'generation_time': 0.0
        }
        
        # Initialize BlenderProc
        self._initialize_blenderproc()
        
    def _initialize_blenderproc(self):
        """Initialize BlenderProc environment."""
        logger.info("Initializing BlenderProc...")
        
        # Initialize BlenderProc
        bproc.init()
        
        # Set random seed for reproducibility
        if 'random_seed' in self.dataset_config:
            random.seed(self.dataset_config['random_seed'])
            np.random.seed(self.dataset_config['random_seed'])
            
    def generate_scenes(self, num_scenes: int, output_dir: str) -> Dict:
        """
        Generate multiple synthetic scenes.
        
        Args:
            num_scenes: Number of scenes to generate
            output_dir: Output directory for generated data
            
        Returns:
            Generation statistics and results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting generation of {num_scenes} scenes...")
        logger.info(f"Output directory: {output_path}")
        
        start_time = time.time()
        
        # Reset counters
        self.annotation_generator.reset_counters()
        self.generation_stats = {
            'total_scenes': 0,
            'successful_scenes': 0,
            'failed_scenes': 0,
            'total_objects': 0,
            'generation_time': 0.0,
            'failed_scene_ids': [],
            'quality_stats': {
                'avg_visibility': 0.0,
                'avg_objects_per_scene': 0.0,
                'class_distribution': {}
            }
        }
        
        # Generate scenes
        for scene_id in range(num_scenes):
            try:
                logger.info(f"Generating scene {scene_id + 1}/{num_scenes}")
                
                scene_result = self.generate_single_scene(scene_id, output_path)
                
                if scene_result['success']:
                    self.generation_stats['successful_scenes'] += 1
                    self.generation_stats['total_objects'] += scene_result['num_objects']
                    self._update_quality_stats(scene_result)
                else:
                    self.generation_stats['failed_scenes'] += 1
                    self.generation_stats['failed_scene_ids'].append(scene_id)
                    logger.warning(f"Scene {scene_id} generation failed: {scene_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Exception during scene {scene_id} generation: {e}")
                self.generation_stats['failed_scenes'] += 1
                self.generation_stats['failed_scene_ids'].append(scene_id)
                
            self.generation_stats['total_scenes'] += 1
            
            # Clean up scene for next iteration
            self._cleanup_scene()
            
        # Finalize generation
        end_time = time.time()
        self.generation_stats['generation_time'] = end_time - start_time
        
        # Save generation report
        self._save_generation_report(output_path)
        
        logger.info(f"Generation complete! Generated {self.generation_stats['successful_scenes']}/{num_scenes} scenes successfully")
        logger.info(f"Total generation time: {self.generation_stats['generation_time']:.2f} seconds")
        
        return self.generation_stats
        
    def generate_single_scene(self, scene_id: int, output_dir: Path) -> Dict:
        """
        Generate a single synthetic scene.
        
        Args:
            scene_id: Scene identifier
            output_dir: Output directory
            
        Returns:
            Scene generation result dictionary
        """
        scene_start_time = time.time()
        
        try:
            # Step 1: Sample objects for the scene
            num_objects = random.randint(*self.scene_config['object_count_range'])
            sampled_objects = self.object_manager.sample_objects(
                num_objects=num_objects,
                allow_multiple_instances=self.scene_config['allow_multiple_instances']
            )
            
            if not sampled_objects:
                return {'success': False, 'error': 'No objects could be sampled'}
                
            logger.debug(f"Sampled {len(sampled_objects)} objects for scene {scene_id}")
            
            # Step 2: Load objects into Blender
            blender_objects = self._load_objects_into_blender(sampled_objects)
            
            if not blender_objects:
                return {'success': False, 'error': 'Failed to load objects into Blender'}
                
            # Step 3: Setup lighting
            lighting_config = self.lighting_manager.setup_scene_lighting()
            
            # Step 4: Setup camera
            camera_config = self.camera_manager.setup_camera_pose()
            
            # Step 5: Setup physics and place objects
            if self.config['physics']['enable_physics']:
                self.physics_simulator.setup_physics_world()
                placement_results = self.physics_simulator.place_objects_with_physics(
                    sampled_objects, blender_objects
                )
            else:
                # Simple random placement without physics
                placement_results = self._place_objects_randomly(sampled_objects, blender_objects)
                
            # Step 6: Validate placement quality
            quality_result = self.quality_controller.validate_scene_quality(
                placement_results, blender_objects
            )
            
            if not quality_result['valid']:
                return {
                    'success': False, 
                    'error': f"Scene quality validation failed: {quality_result['issues']}"
                }
                
            # Step 7: Render and generate annotations
            scene_data = {
                'scene_id': scene_id,
                'sampled_objects': sampled_objects,
                'blender_objects': blender_objects,
                'placement_results': placement_results,
                'lighting_config': lighting_config,
                'camera_config': camera_config,
                'quality_result': quality_result,
                'timestamp': datetime.now().isoformat(),
                'random_seed': random.getstate()[1][0] if hasattr(random.getstate()[1], '__getitem__') else None
            }
            
            # Generate annotations
            annotations = self.annotation_generator.generate_annotations(scene_data, output_dir)
            
            # Save rendered images
            self._save_rendered_images(output_dir, scene_id)
            
            # Calculate generation time
            generation_time = time.time() - scene_start_time
            
            result = {
                'success': True,
                'scene_id': scene_id,
                'num_objects': len(sampled_objects),
                'num_visible_objects': len(annotations['objects']),
                'generation_time': generation_time,
                'quality_metrics': quality_result,
                'object_classes': [obj['class'] for obj in sampled_objects],
                'annotations': annotations
            }
            
            logger.debug(f"Scene {scene_id} generated successfully in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error generating scene {scene_id}: {e}")
            return {'success': False, 'error': str(e)}
            
    def _load_objects_into_blender(self, sampled_objects: List[Dict]) -> List:
        """Load sampled objects into Blender scene."""
        blender_objects = []
        
        for i, obj_data in enumerate(sampled_objects):
            try:
                # Load mesh
                mesh = self.object_manager.load_mesh(obj_data)
                if mesh is None:
                    logger.warning(f"Failed to load mesh for object {obj_data['id']}")
                    continue
                    
                # Import into Blender
                blender_obj = bproc.loader.load_obj(obj_data['path'])[0]
                
                # Set instance ID for segmentation
                blender_obj['instance_id'] = i + 1  # Start from 1, 0 is background
                blender_obj.set_attribute('instance_id', i + 1)
                
                # Set category for class segmentation
                blender_obj.set_attribute('category_id', hash(obj_data['class']) % 1000)
                
                # Store object data reference
                blender_obj['object_data'] = obj_data
                
                blender_objects.append(blender_obj)
                
            except Exception as e:
                logger.warning(f"Failed to load object {obj_data['id']} into Blender: {e}")
                continue
                
        return blender_objects
        
    def _place_objects_randomly(self, sampled_objects: List[Dict], 
                               blender_objects: List) -> List[Dict]:
        """Place objects randomly without physics simulation."""
        placement_results = []
        bounds = self.scene_config['scene_bounds']
        
        for obj_data, blender_obj in zip(sampled_objects, blender_objects):
            # Random position
            position = [
                random.uniform(bounds[0], bounds[1]),  # x
                random.uniform(bounds[2], bounds[3]),  # y
                random.uniform(bounds[4] + 0.1, bounds[5])  # z (slightly above ground)
            ]
            
            # Random rotation
            rotation = [
                random.uniform(0, 2 * np.pi),
                random.uniform(0, 2 * np.pi),
                random.uniform(0, 2 * np.pi)
            ]
            
            # Apply transformation
            blender_obj.set_location(position)
            blender_obj.set_rotation_euler(rotation)
            
            result = {
                'object_data': obj_data,
                'blender_object': blender_obj,
                'initial_pose': {'position': position, 'rotation': rotation},
                'final_pose': {'position': position, 'rotation': rotation},
                'settled': True
            }
            
            placement_results.append(result)
            
        return placement_results
        
    def _save_rendered_images(self, output_dir: Path, scene_id: int):
        """Save rendered RGB, depth, and segmentation images."""
        # Create output directories
        rgb_dir = output_dir / 'images' / 'rgb'
        depth_dir = output_dir / 'images' / 'depth'
        seg_dir = output_dir / 'images' / 'segmentation'
        
        for dir_path in [rgb_dir, depth_dir, seg_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Render data
        data = bproc.renderer.render()
        
        # Save RGB image
        rgb_path = rgb_dir / f"scene_{scene_id:06d}.png"
        if 'colors' in data and len(data['colors']) > 0:
            bproc.writer.write_png(str(rgb_path), data['colors'][0])
            
        # Save depth image
        depth_path = depth_dir / f"scene_{scene_id:06d}_depth.png"
        if 'depth' in data and len(data['depth']) > 0:
            # Convert depth to uint16 for PNG storage
            depth_mm = (data['depth'][0] * 1000).astype(np.uint16)
            bproc.writer.write_png(str(depth_path), depth_mm)
            
        # Save segmentation masks
        if 'instance_segmaps' in data and len(data['instance_segmaps']) > 0:
            seg_path = seg_dir / f"scene_{scene_id:06d}_instance_seg.png"
            bproc.writer.write_png(str(seg_path), data['instance_segmaps'][0])
            
        if 'class_segmaps' in data and len(data['class_segmaps']) > 0:
            class_seg_path = seg_dir / f"scene_{scene_id:06d}_class_seg.png"
            bproc.writer.write_png(str(class_seg_path), data['class_segmaps'][0])
            
    def _update_quality_stats(self, scene_result: Dict):
        """Update running quality statistics."""
        quality_stats = self.generation_stats['quality_stats']
        
        # Update visibility statistics
        if 'quality_metrics' in scene_result:
            visibility = scene_result['quality_metrics'].get('avg_visibility', 0.0)
            current_avg = quality_stats['avg_visibility']
            n = self.generation_stats['successful_scenes']
            quality_stats['avg_visibility'] = (current_avg * (n - 1) + visibility) / n
            
        # Update objects per scene
        num_objects = scene_result['num_objects']
        current_avg = quality_stats['avg_objects_per_scene']
        n = self.generation_stats['successful_scenes']
        quality_stats['avg_objects_per_scene'] = (current_avg * (n - 1) + num_objects) / n
        
        # Update class distribution
        for class_name in scene_result['object_classes']:
            if class_name not in quality_stats['class_distribution']:
                quality_stats['class_distribution'][class_name] = 0
            quality_stats['class_distribution'][class_name] += 1
            
    def _cleanup_scene(self):
        """Clean up current scene for next generation."""
        # Clear all objects
        bproc.utility.reset_keyframes()
        
        # Clean up physics
        if self.config['physics']['enable_physics']:
            self.physics_simulator.cleanup_physics()
            
        # Clear materials and textures
        bproc.material.clear_all_materials()
        
    def _save_generation_report(self, output_dir: Path):
        """Save detailed generation report."""
        report_dir = output_dir / 'logs'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f'generation_report_{timestamp}.json'
        
        report = {
            'generation_config': self.config,
            'statistics': self.generation_stats,
            'object_manager_stats': self.object_manager.get_collection_stats(),
            'class_distribution': self.object_manager.get_class_distribution(),
            'generation_timestamp': datetime.now().isoformat()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Generation report saved to {report_file}")
        
    def cleanup(self):
        """Clean up resources and finalize generation."""
        # Final cleanup
        self._cleanup_scene()
        
        # Close BlenderProc
        bproc.utility.reset_keyframes()
        
        logger.info("Scene generator cleanup complete")
