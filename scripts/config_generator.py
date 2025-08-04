#!/usr/bin/env python3
"""
Configuration generator script for creating customized pipeline configurations.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import create_default_config, save_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate configuration file for dataset generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="configs/custom_config.yaml",
        help="Output path for configuration file"
    )
    
    parser.add_argument(
        "--models-dir",
        type=str,
        default="/dss/dsstbyfs02/pn52ru/pn52ru-dss-0000/common/datasets/clutter6d/3d_models/",
        help="Path to 3D models directory"
    )
    
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=1000,
        help="Number of scenes to generate"
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[640, 480],
        metavar=("WIDTH", "HEIGHT"),
        help="Image resolution"
    )
    
    parser.add_argument(
        "--object-count-range",
        type=int,
        nargs=2,
        default=[5, 20],
        metavar=("MIN", "MAX"),
        help="Range of objects per scene"
    )
    
    parser.add_argument(
        "--disable-physics",
        action="store_true",
        help="Disable physics simulation"
    )
    
    parser.add_argument(
        "--enable-hdri",
        action="store_true",
        default=True,
        help="Enable HDRI lighting"
    )
    
    parser.add_argument(
        "--hdri-dir",
        type=str,
        default="hdris/",
        help="Path to HDRI files directory"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Number of rendering samples for Cycles"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--collections",
        type=str,
        nargs="+",
        default=["GSO", "Objaverse", "OmniObject3D"],
        help="3D model collections to use"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["coco", "yolo", "custom"],
        default="coco",
        help="Annotation format"
    )
    
    return parser.parse_args()


def create_custom_config(args):
    """Create custom configuration based on arguments."""
    config = create_default_config()
    
    # Update with command line arguments
    config['models']['data_dir'] = args.models_dir
    config['models']['collections'] = args.collections
    
    config['dataset']['num_scenes'] = args.num_scenes
    config['dataset']['random_seed'] = args.seed
    
    config['camera']['image_size'] = args.image_size
    
    config['scene']['object_count_range'] = args.object_count_range
    
    config['physics']['enable_physics'] = not args.disable_physics
    
    config['lighting']['use_hdri'] = args.enable_hdri
    config['lighting']['hdri_dir'] = args.hdri_dir
    
    config['rendering']['samples'] = args.samples
    
    config['output']['annotation_format'] = args.format
    
    return config


def main():
    """Main function."""
    args = parse_arguments()
    
    print("Generating configuration file...")
    print(f"Models directory: {args.models_dir}")
    print(f"Number of scenes: {args.num_scenes}")
    print(f"Image size: {args.image_size}")
    print(f"Object count range: {args.object_count_range}")
    print(f"Physics enabled: {not args.disable_physics}")
    print(f"HDRI enabled: {args.enable_hdri}")
    
    # Create configuration
    config = create_custom_config(args)
    
    # Save configuration
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_config(config, str(output_path))
    
    print(f"\nConfiguration saved to: {output_path.absolute()}")
    print("\nYou can now generate the dataset with:")
    print(f"python scripts/generate_dataset.py --config {args.output}")


if __name__ == "__main__":
    main()
