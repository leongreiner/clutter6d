#!/usr/bin/env python3
"""
Example usage script demonstrating the Clutter6D pipeline capabilities.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.scene_generator import SceneGenerator
from utils.config_loader import load_config, create_default_config


def example_basic_generation():
    """Example: Basic dataset generation with default settings."""
    print("=== Example 1: Basic Dataset Generation ===")
    
    # Create default configuration
    config = create_default_config()
    
    # Modify for quick demo
    config['dataset']['num_scenes'] = 5
    config['scene']['object_count_range'] = [3, 8]
    config['camera']['image_size'] = [320, 240]  # Smaller for faster rendering
    config['rendering']['samples'] = 64  # Fewer samples for speed
    
    # Initialize generator
    generator = SceneGenerator(config)
    
    # Generate small dataset
    output_dir = "examples/basic_demo"
    stats = generator.generate_scenes(num_scenes=5, output_dir=output_dir)
    
    print(f"Generated {stats['successful_scenes']} scenes in {stats['generation_time']:.1f}s")
    
    # Cleanup
    generator.cleanup()


def example_custom_configuration():
    """Example: Custom configuration for specific use case."""
    print("\n=== Example 2: Custom Configuration ===")
    
    # Create custom configuration
    config = create_default_config()
    
    # High-quality rendering setup
    config['camera']['image_size'] = [1024, 768]
    config['rendering']['samples'] = 512
    config['rendering']['denoising'] = True
    
    # Dense scenes with many objects
    config['scene']['object_count_range'] = [15, 30]
    config['scene']['min_visibility_threshold'] = 0.15  # Lower threshold for dense scenes
    
    # Enable all quality controls
    config['quality_control']['min_objects_visible'] = 5
    config['quality_control']['balance_classes'] = True
    
    print("Custom configuration created:")
    print(f"  Image size: {config['camera']['image_size']}")
    print(f"  Object count: {config['scene']['object_count_range']}")
    print(f"  Rendering samples: {config['rendering']['samples']}")


def example_multi_viewpoint():
    """Example: Generate multiple viewpoints for the same scene."""
    print("\n=== Example 3: Multi-viewpoint Generation ===")
    
    from utils.camera_utils import CameraManager
    
    config = create_default_config()
    camera_manager = CameraManager(config)
    
    # Generate multiple viewpoints
    viewpoints = camera_manager.sample_multiple_viewpoints(num_viewpoints=4)
    
    print(f"Generated {len(viewpoints)} viewpoints:")
    for i, vp in enumerate(viewpoints):
        print(f"  Viewpoint {i}: distance={vp['distance']:.2f}, elevation={vp['elevation']:.1f}°")


def example_quality_analysis():
    """Example: Analyze generation quality metrics."""
    print("\n=== Example 4: Quality Analysis ===")
    
    from utils.quality_control import QualityController
    
    config = create_default_config()
    quality_controller = QualityController(config)
    
    # Simulate some validation results
    mock_placement_results = [
        {
            'object_data': {'id': 'obj1', 'class': 'chair'},
            'final_pose': {'position': [0.5, 0.5, 0.1]},
            'settled': True
        },
        {
            'object_data': {'id': 'obj2', 'class': 'table'},
            'final_pose': {'position': [-0.5, 0.2, 0.2]},
            'settled': True
        }
    ]
    
    # Mock rendered data
    import numpy as np
    mock_rendered_data = {
        'instance_segmaps': [np.random.randint(0, 3, (240, 320))]
    }
    
    # Validate scene quality
    validation_result = quality_controller.validate_scene_quality(
        mock_placement_results, [], mock_rendered_data
    )
    
    print(f"Scene validation result: {'VALID' if validation_result['valid'] else 'INVALID'}")
    if validation_result['issues']:
        print(f"Issues found: {validation_result['issues']}")


def example_material_randomization():
    """Example: Material and lighting randomization."""
    print("\n=== Example 5: Material Randomization ===")
    
    from utils.lighting_utils import LightingManager
    
    config = create_default_config()
    config['materials']['enable_material_randomization'] = True
    config['materials']['pbr_material_prob'] = 0.8
    
    lighting_manager = LightingManager(config)
    
    print("Material randomization enabled:")
    print(f"  PBR material probability: {config['materials']['pbr_material_prob']}")
    print(f"  Metallic range: {config['materials']['metallic_range']}")
    print(f"  Roughness range: {config['materials']['roughness_range']}")


def example_annotation_formats():
    """Example: Different annotation formats."""
    print("\n=== Example 6: Annotation Formats ===")
    
    formats = ['coco', 'yolo', 'custom']
    
    for fmt in formats:
        config = create_default_config()
        config['output']['annotation_format'] = fmt
        print(f"  {fmt.upper()} format: Includes 6DoF poses, bounding boxes, segmentation")


def main():
    """Run all examples."""
    print("Clutter6D Pipeline Examples")
    print("=" * 50)
    
    try:
        # Note: Most examples don't actually run generation to avoid dependencies
        # In a real environment with BlenderProc installed, you could run:
        # example_basic_generation()
        
        example_custom_configuration()
        example_multi_viewpoint()
        example_quality_analysis()
        example_material_randomization()
        example_annotation_formats()
        
        print("\n" + "=" * 50)
        print("Examples completed successfully!")
        print("\nTo run actual generation:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run: python scripts/generate_dataset.py --config configs/default_config.yaml")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
