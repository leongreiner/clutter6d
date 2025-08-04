"""
Quality control utilities for validating scene generation quality.
Ensures minimum visibility thresholds, balanced class distribution, and scene validity.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import cv2
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class QualityController:
    """Controls and validates the quality of generated scenes."""
    
    def __init__(self, config: Dict):
        """
        Initialize the QualityController.
        
        Args:
            config: Configuration dictionary containing quality control settings
        """
        self.config = config
        self.quality_config = config['quality_control']
        self.scene_config = config['scene']
        
        # Quality thresholds
        self.min_objects_visible = self.quality_config['min_objects_visible']
        self.max_occlusion_ratio = self.quality_config['max_occlusion_ratio']
        self.min_scene_coverage = self.quality_config['min_scene_coverage']
        self.balance_classes = self.quality_config['balance_classes']
        self.min_visibility_threshold = self.scene_config['min_visibility_threshold']
        
        # Statistics tracking
        self.scene_stats = {
            'total_scenes_validated': 0,
            'valid_scenes': 0,
            'invalid_scenes': 0,
            'class_distribution': defaultdict(int),
            'visibility_stats': [],
            'occlusion_stats': [],
            'coverage_stats': []
        }
        
    def validate_scene_quality(self, placement_results: List[Dict], 
                             blender_objects: List, rendered_data: Optional[Dict] = None) -> Dict:
        """
        Validate the quality of a generated scene.
        
        Args:
            placement_results: Results from physics simulation
            blender_objects: List of Blender objects in scene
            rendered_data: Optional rendered data for visibility analysis
            
        Returns:
            Validation result dictionary
        """
        logger.debug(f"Validating scene quality for {len(placement_results)} objects")
        
        validation_result = {
            'valid': True,
            'issues': [],
            'metrics': {},
            'object_visibility': {},
            'class_balance': {},
            'scene_coverage': 0.0
        }
        
        # Check object settlement and placement
        settlement_result = self._check_object_settlement(placement_results)
        validation_result['metrics']['settlement'] = settlement_result
        
        if not settlement_result['valid']:
            validation_result['valid'] = False
            validation_result['issues'].extend(settlement_result['issues'])
            
        # Check object visibility (if rendered data available)
        if rendered_data:
            visibility_result = self._check_object_visibility(placement_results, rendered_data)
            validation_result['metrics']['visibility'] = visibility_result
            validation_result['object_visibility'] = visibility_result['object_visibility']
            
            if not visibility_result['valid']:
                validation_result['valid'] = False
                validation_result['issues'].extend(visibility_result['issues'])
                
        # Check class balance
        if self.balance_classes:
            balance_result = self._check_class_balance(placement_results)
            validation_result['metrics']['class_balance'] = balance_result
            validation_result['class_balance'] = balance_result['distribution']
            
            if not balance_result['valid']:
                validation_result['valid'] = False
                validation_result['issues'].extend(balance_result['issues'])
                
        # Check scene coverage
        coverage_result = self._check_scene_coverage(placement_results)
        validation_result['metrics']['scene_coverage'] = coverage_result
        validation_result['scene_coverage'] = coverage_result['coverage']
        
        if not coverage_result['valid']:
            validation_result['valid'] = False
            validation_result['issues'].extend(coverage_result['issues'])
            
        # Update statistics
        self._update_validation_stats(validation_result)
        
        return validation_result
        
    def _check_object_settlement(self, placement_results: List[Dict]) -> Dict:
        """Check if objects have settled properly in physics simulation."""
        settled_count = 0
        unstable_objects = []
        out_of_bounds_objects = []
        
        scene_bounds = self.scene_config['scene_bounds']
        
        for i, result in enumerate(placement_results):
            # Check if object settled
            if not result.get('settled', True):
                unstable_objects.append(i)
            else:
                settled_count += 1
                
            # Check if object is within scene bounds
            final_pose = result.get('final_pose', {})
            position = final_pose.get('position', [0, 0, 0])
            
            if not (scene_bounds[0] <= position[0] <= scene_bounds[1] and
                   scene_bounds[2] <= position[1] <= scene_bounds[3] and
                   scene_bounds[4] <= position[2] <= scene_bounds[5]):
                out_of_bounds_objects.append(i)
                
        settled_ratio = settled_count / len(placement_results) if placement_results else 0
        in_bounds_ratio = (len(placement_results) - len(out_of_bounds_objects)) / len(placement_results) if placement_results else 0
        
        # Validation criteria
        valid = (settled_ratio >= 0.8 and  # At least 80% settled
                in_bounds_ratio >= 0.9)      # At least 90% in bounds
        
        issues = []
        if settled_ratio < 0.8:
            issues.append(f"Low settlement ratio: {settled_ratio:.2f} < 0.8")
        if in_bounds_ratio < 0.9:
            issues.append(f"Too many objects out of bounds: {len(out_of_bounds_objects)}")
            
        return {
            'valid': valid,
            'settled_ratio': settled_ratio,
            'in_bounds_ratio': in_bounds_ratio,
            'unstable_objects': unstable_objects,
            'out_of_bounds_objects': out_of_bounds_objects,
            'issues': issues
        }
        
    def _check_object_visibility(self, placement_results: List[Dict], 
                               rendered_data: Dict) -> Dict:
        """Check object visibility in rendered images."""
        if 'instance_segmaps' not in rendered_data:
            return {'valid': True, 'issues': [], 'object_visibility': {}}
            
        instance_segmap = rendered_data['instance_segmaps'][0]  # First camera view
        total_pixels = instance_segmap.shape[0] * instance_segmap.shape[1]
        
        visible_objects = 0
        object_visibility = {}
        visibility_ratios = []
        
        for i, result in enumerate(placement_results):
            instance_id = i + 1  # Instance IDs start from 1
            
            # Count pixels for this object
            object_pixels = np.sum(instance_segmap == instance_id)
            visibility_ratio = object_pixels / total_pixels
            
            object_visibility[instance_id] = {
                'pixels': int(object_pixels),
                'visibility_ratio': visibility_ratio,
                'visible': visibility_ratio >= self.min_visibility_threshold
            }
            
            if visibility_ratio >= self.min_visibility_threshold:
                visible_objects += 1
                visibility_ratios.append(visibility_ratio)
                
        # Calculate average visibility of visible objects
        avg_visibility = np.mean(visibility_ratios) if visibility_ratios else 0.0
        
        # Validation criteria
        valid = visible_objects >= self.min_objects_visible
        
        issues = []
        if visible_objects < self.min_objects_visible:
            issues.append(f"Too few visible objects: {visible_objects} < {self.min_objects_visible}")
            
        return {
            'valid': valid,
            'visible_objects': visible_objects,
            'total_objects': len(placement_results),
            'avg_visibility': avg_visibility,
            'object_visibility': object_visibility,
            'issues': issues
        }
        
    def _check_class_balance(self, placement_results: List[Dict]) -> Dict:
        """Check class distribution balance."""
        class_counts = Counter()
        
        for result in placement_results:
            obj_data = result.get('object_data', {})
            class_name = obj_data.get('class', 'unknown')
            class_counts[class_name] += 1
            
        # Calculate balance metrics
        total_objects = len(placement_results)
        class_distribution = dict(class_counts)
        
        if not class_counts:
            return {
                'valid': False,
                'distribution': {},
                'balance_score': 0.0,
                'issues': ['No objects found for class balance check']
            }
            
        # Calculate entropy as a measure of balance
        probabilities = np.array(list(class_counts.values())) / total_objects
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        max_entropy = np.log2(len(class_counts))
        balance_score = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Check for extreme imbalances
        max_class_ratio = max(class_counts.values()) / total_objects
        min_class_ratio = min(class_counts.values()) / total_objects
        
        # Validation criteria (somewhat lenient for realistic scenes)
        valid = (balance_score >= 0.5 and  # Reasonable diversity
                max_class_ratio <= 0.7)      # No single class dominates too much
        
        issues = []
        if balance_score < 0.5:
            issues.append(f"Low class diversity: balance_score={balance_score:.2f}")
        if max_class_ratio > 0.7:
            issues.append(f"Class dominance too high: {max_class_ratio:.2f}")
            
        return {
            'valid': valid,
            'distribution': class_distribution,
            'balance_score': balance_score,
            'max_class_ratio': max_class_ratio,
            'min_class_ratio': min_class_ratio,
            'issues': issues
        }
        
    def _check_scene_coverage(self, placement_results: List[Dict]) -> Dict:
        """Check how well objects cover the scene space."""
        if not placement_results:
            return {
                'valid': False,
                'coverage': 0.0,
                'issues': ['No objects in scene']
            }
            
        scene_bounds = self.scene_config['scene_bounds']
        
        # Calculate bounding boxes of all objects
        object_positions = []
        for result in placement_results:
            final_pose = result.get('final_pose', {})
            position = final_pose.get('position', [0, 0, 0])
            object_positions.append(position)
            
        object_positions = np.array(object_positions)
        
        # Calculate coverage in each dimension
        scene_dims = [
            scene_bounds[1] - scene_bounds[0],  # X dimension
            scene_bounds[3] - scene_bounds[2],  # Y dimension
            scene_bounds[5] - scene_bounds[4]   # Z dimension
        ]
        
        if object_positions.size > 0:
            object_ranges = [
                np.max(object_positions[:, 0]) - np.min(object_positions[:, 0]),
                np.max(object_positions[:, 1]) - np.min(object_positions[:, 1]),
                np.max(object_positions[:, 2]) - np.min(object_positions[:, 2])
            ]
            
            # Coverage ratio in each dimension
            coverage_ratios = [
                obj_range / scene_dim if scene_dim > 0 else 0
                for obj_range, scene_dim in zip(object_ranges, scene_dims)
            ]
            
            # Overall coverage (geometric mean to avoid dominance by one dimension)
            coverage = np.power(np.prod([max(0.01, ratio) for ratio in coverage_ratios]), 1/3)
        else:
            coverage = 0.0
            
        # Validation criteria
        valid = coverage >= self.min_scene_coverage
        
        issues = []
        if coverage < self.min_scene_coverage:
            issues.append(f"Low scene coverage: {coverage:.2f} < {self.min_scene_coverage}")
            
        return {
            'valid': valid,
            'coverage': coverage,
            'coverage_ratios': coverage_ratios if object_positions.size > 0 else [0, 0, 0],
            'issues': issues
        }
        
    def _update_validation_stats(self, validation_result: Dict):
        """Update running validation statistics."""
        self.scene_stats['total_scenes_validated'] += 1
        
        if validation_result['valid']:
            self.scene_stats['valid_scenes'] += 1
        else:
            self.scene_stats['invalid_scenes'] += 1
            
        # Update visibility stats
        if 'visibility' in validation_result['metrics']:
            visibility_metrics = validation_result['metrics']['visibility']
            if 'avg_visibility' in visibility_metrics:
                self.scene_stats['visibility_stats'].append(visibility_metrics['avg_visibility'])
                
        # Update coverage stats
        if 'scene_coverage' in validation_result['metrics']:
            coverage_metrics = validation_result['metrics']['scene_coverage']
            if 'coverage' in coverage_metrics:
                self.scene_stats['coverage_stats'].append(coverage_metrics['coverage'])
                
        # Update class distribution
        if 'class_balance' in validation_result['metrics']:
            class_metrics = validation_result['metrics']['class_balance']
            if 'distribution' in class_metrics:
                for class_name, count in class_metrics['distribution'].items():
                    self.scene_stats['class_distribution'][class_name] += count
                    
    def get_validation_statistics(self) -> Dict:
        """Get comprehensive validation statistics."""
        stats = self.scene_stats.copy()
        
        # Calculate derived statistics
        if stats['total_scenes_validated'] > 0:
            stats['validation_success_rate'] = stats['valid_scenes'] / stats['total_scenes_validated']
            
            if stats['visibility_stats']:
                stats['avg_scene_visibility'] = np.mean(stats['visibility_stats'])
                stats['std_scene_visibility'] = np.std(stats['visibility_stats'])
                
            if stats['coverage_stats']:
                stats['avg_scene_coverage'] = np.mean(stats['coverage_stats'])
                stats['std_scene_coverage'] = np.std(stats['coverage_stats'])
                
        return stats
        
    def suggest_quality_improvements(self, validation_result: Dict) -> List[str]:
        """Suggest improvements based on validation results."""
        suggestions = []
        
        if not validation_result['valid']:
            issues = validation_result['issues']
            
            for issue in issues:
                if 'visibility' in issue.lower():
                    suggestions.append("Increase lighting or reduce occlusion between objects")
                    suggestions.append("Adjust camera position for better viewpoint")
                    
                elif 'settlement' in issue.lower():
                    suggestions.append("Increase physics simulation time")
                    suggestions.append("Adjust object drop heights")
                    suggestions.append("Check for object interpenetration")
                    
                elif 'class' in issue.lower() and 'balance' in issue.lower():
                    suggestions.append("Adjust object sampling to improve class diversity")
                    suggestions.append("Ensure sufficient objects from each class are available")
                    
                elif 'coverage' in issue.lower():
                    suggestions.append("Increase the number of objects in the scene")
                    suggestions.append("Adjust scene bounds or object placement strategy")
                    
                elif 'bounds' in issue.lower():
                    suggestions.append("Adjust physics simulation parameters")
                    suggestions.append("Check scene bounds configuration")
                    
        return list(set(suggestions))  # Remove duplicates
        
    def reset_statistics(self):
        """Reset validation statistics."""
        self.scene_stats = {
            'total_scenes_validated': 0,
            'valid_scenes': 0,
            'invalid_scenes': 0,
            'class_distribution': defaultdict(int),
            'visibility_stats': [],
            'occlusion_stats': [],
            'coverage_stats': []
        }
