#!/usr/bin/env python3
"""
Main script for generating synthetic cluttered scenes with 6DoF pose annotations.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.scene_generator import SceneGenerator
from utils.config_loader import load_config


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset with 6DoF pose annotations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory for generated dataset"
    )
    
    parser.add_argument(
        "--num-scenes",
        type=int,
        help="Number of scenes to generate (overrides config)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (optional)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (overrides config)"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without generating"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume generation from checkpoint directory"
    )
    
    return parser.parse_args()


def validate_environment():
    """Validate that required dependencies are available."""
    try:
        import blenderproc
        import bpy
        import numpy
        import trimesh
        logging.info("All required dependencies are available")
        return True
    except ImportError as e:
        logging.error(f"Missing required dependency: {e}")
        logging.error("Please install requirements with: pip install -r requirements.txt")
        return False


def print_generation_summary(stats: dict, config: dict):
    """Print a summary of the generation process."""
    print("\n" + "="*60)
    print("DATASET GENERATION SUMMARY")
    print("="*60)
    
    print(f"Total scenes processed: {stats['total_scenes']}")
    print(f"Successful scenes: {stats['successful_scenes']}")
    print(f"Failed scenes: {stats['failed_scenes']}")
    print(f"Success rate: {stats['successful_scenes']/stats['total_scenes']*100:.1f}%")
    print(f"Total objects generated: {stats['total_objects']}")
    print(f"Generation time: {stats['generation_time']:.1f} seconds")
    
    if stats['successful_scenes'] > 0:
        avg_time_per_scene = stats['generation_time'] / stats['successful_scenes']
        print(f"Average time per scene: {avg_time_per_scene:.1f} seconds")
        
        quality_stats = stats.get('quality_stats', {})
        if quality_stats:
            print(f"\nQuality Metrics:")
            print(f"  Average visibility: {quality_stats.get('avg_visibility', 0):.3f}")
            print(f"  Average objects per scene: {quality_stats.get('avg_objects_per_scene', 0):.1f}")
            
            class_dist = quality_stats.get('class_distribution', {})
            if class_dist:
                print(f"  Classes generated: {len(class_dist)}")
                top_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Top 5 classes: {', '.join([f'{k}({v})' for k, v in top_classes])}")
    
    if stats['failed_scene_ids']:
        print(f"\nFailed scene IDs: {stats['failed_scene_ids'][:10]}")  # Show first 10
        if len(stats['failed_scene_ids']) > 10:
            print(f"... and {len(stats['failed_scene_ids']) - 10} more")
    
    print("="*60)


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Clutter6D dataset generation...")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Validate environment
        if not validate_environment():
            sys.exit(1)
            
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.num_scenes:
            config['dataset']['num_scenes'] = args.num_scenes
            logger.info(f"Overriding num_scenes with {args.num_scenes}")
            
        if args.seed:
            config['dataset']['random_seed'] = args.seed
            logger.info(f"Overriding random_seed with {args.seed}")
            
        # Validate configuration only
        if args.validate_only:
            logger.info("Configuration validation passed!")
            return
            
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir.absolute()}")
        
        # Initialize scene generator
        logger.info("Initializing scene generator...")
        generator = SceneGenerator(config)
        
        # Handle resume
        if args.resume:
            logger.info(f"Resuming generation from {args.resume}")
            # TODO: Implement resume functionality
            logger.warning("Resume functionality not yet implemented")
            
        # Generate dataset
        start_time = time.time()
        num_scenes = config['dataset']['num_scenes']
        
        logger.info(f"Starting generation of {num_scenes} scenes...")
        
        stats = generator.generate_scenes(
            num_scenes=num_scenes,
            output_dir=str(output_dir)
        )
        
        # Print summary
        print_generation_summary(stats, config)
        
        # Cleanup
        generator.cleanup()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"Dataset generation completed in {total_time:.1f} seconds")
        
        if stats['successful_scenes'] == 0:
            logger.error("No scenes were generated successfully!")
            sys.exit(1)
        elif stats['failed_scenes'] > stats['successful_scenes'] * 0.1:  # More than 10% failure
            logger.warning(f"High failure rate: {stats['failed_scenes']}/{stats['total_scenes']} scenes failed")
            
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
