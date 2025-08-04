"""
Configuration utilities for loading and validating pipeline configurations.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Loaded configuration from {config_path}")
        
        # Validate configuration
        validate_config(config)
        
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration: {e}")


def validate_config(config: Dict[str, Any]):
    """
    Validate configuration dictionary for required fields and reasonable values.
    
    Args:
        config: Configuration dictionary to validate
    """
    required_sections = [
        'models', 'scene', 'physics', 'camera', 'lighting', 
        'rendering', 'materials', 'output', 'quality_control', 'dataset'
    ]
    
    # Check required sections
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {missing_sections}")
        
    # Validate models section
    models_config = config['models']
    required_model_fields = ['data_dir', 'collections', 'supported_formats']
    for field in required_model_fields:
        if field not in models_config:
            raise ValueError(f"Missing required field in models section: {field}")
            
    # Validate scene section
    scene_config = config['scene']
    if 'object_count_range' not in scene_config:
        raise ValueError("Missing object_count_range in scene configuration")
    if len(scene_config['object_count_range']) != 2:
        raise ValueError("object_count_range must have exactly 2 values [min, max]")
    if scene_config['object_count_range'][0] > scene_config['object_count_range'][1]:
        raise ValueError("object_count_range min must be <= max")
        
    # Validate camera section
    camera_config = config['camera']
    required_camera_fields = ['image_size', 'fov_range', 'distance_range']
    for field in required_camera_fields:
        if field not in camera_config:
            raise ValueError(f"Missing required field in camera section: {field}")
            
    # Validate image size
    if len(camera_config['image_size']) != 2:
        raise ValueError("image_size must have exactly 2 values [width, height]")
    if any(dim <= 0 for dim in camera_config['image_size']):
        raise ValueError("image_size dimensions must be positive")
        
    logger.info("Configuration validation passed")


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            
        logger.info(f"Configuration saved to {output_path}")
        
    except Exception as e:
        raise RuntimeError(f"Error saving configuration: {e}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration dictionary.
    
    Returns:
        Default configuration
    """
    return {
        'models': {
            'data_dir': '/path/to/3d_models',
            'collections': ['GSO', 'Objaverse', 'OmniObject3D'],
            'max_models': 10000,
            'supported_formats': ['.glb', '.obj', '.ply']
        },
        'scene': {
            'object_count_range': [5, 20],
            'allow_multiple_instances': True,
            'multiple_instance_prob': 0.3,
            'max_instances_per_class': 3,
            'min_visibility_threshold': 0.2,
            'scene_bounds': [-2.0, 2.0, -2.0, 2.0, 0.0, 3.0]
        },
        'physics': {
            'gravity': [0, 0, -9.81],
            'simulation_steps': 100,
            'max_simulation_time': 10.0,
            'collision_margin': 0.01,
            'enable_physics': True,
            'drop_height_range': [0.5, 2.0]
        },
        'camera': {
            'image_size': [640, 480],
            'fov_range': [40, 80],
            'distance_range': [1.0, 4.0],
            'elevation_range': [-30, 60],
            'azimuth_range': [0, 360],
            'look_at_jitter': 0.1
        },
        'lighting': {
            'hdri_dir': 'hdris/',
            'use_hdri': True,
            'hdri_strength_range': [0.5, 2.0],
            'additional_lights': True,
            'num_additional_lights_range': [1, 3],
            'light_energy_range': [10, 100]
        },
        'rendering': {
            'engine': 'CYCLES',
            'samples': 256,
            'denoising': True,
            'motion_blur': False,
            'depth_of_field': False
        },
        'materials': {
            'enable_material_randomization': True,
            'pbr_material_prob': 0.7,
            'metallic_range': [0.0, 1.0],
            'roughness_range': [0.1, 1.0]
        },
        'output': {
            'save_blend_files': False,
            'annotation_format': 'coco',
            'depth_format': 'png',
            'depth_scale': 1000.0,
            'save_intermediate_steps': False
        },
        'quality_control': {
            'min_objects_visible': 3,
            'max_occlusion_ratio': 0.8,
            'min_scene_coverage': 0.3,
            'balance_classes': True
        },
        'dataset': {
            'num_scenes': 1000,
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            'random_seed': 42
        },
        'logging': {
            'log_level': 'INFO',
            'save_generation_log': True,
            'save_scene_metadata': True,
            'log_performance_metrics': True
        }
    }
