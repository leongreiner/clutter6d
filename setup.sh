#!/bin/bash

# Setup script for Clutter6D synthetic dataset generation pipeline

echo "Setting up Clutter6D pipeline..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is required but not installed."
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment 'clutter6d'..."
conda env create -f environment.yml

echo "Activating conda environment..."
echo "To activate the environment, run: conda activate clutter6d"

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.py

# Create default directories
echo "Creating default directories..."
mkdir -p data/synthetic
mkdir -p logs
mkdir -p hdris

# Create example HDRI directory structure
echo "Setting up HDRI directory structure..."
echo "Please place your HDRI files (.hdr, .exr) in the hdris/ directory"

# Create example configuration
echo "Creating example configuration..."
python3 scripts/config_generator.py --output configs/example_config.yaml

echo "Setup complete!"
echo ""
echo "Quick start:"
echo "1. Activate environment: conda activate clutter6d"
echo "2. Place your 3D models in the configured directory"
echo "3. Add HDRI files to hdris/ directory (optional)"
echo "4. Generate dataset: python scripts/generate_dataset.py --config configs/example_config.yaml"
echo "5. Visualize results: python scripts/visualize_results.py --data-dir data/synthetic --scene-id 0"
