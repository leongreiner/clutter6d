# Clutter6D: Synthetic Dataset Generation Pipeline

A comprehensive pipeline for generating synthetic cluttered scenes with 6DoF pose annotations using BlenderProc and physics-based object placement.

## Features

- **CAD Model Collection**: Support for GSO, Objaverse, and OmniObject3D models
- **Physics-based Rendering**: Realistic object placement using BlenderProc physics simulation
- **Scene Randomization**: Random object counts, class combinations, lighting, and camera poses
- **Quality Controls**: Minimum visibility thresholds and class balancing
- **Comprehensive Annotations**: Per-instance segmentation masks and 6DoF poses
- **Reproducible**: Metadata logging with random seeds for reproducibility

## Installation

### Prerequisites

- Python 3.9+ (via Conda)
- BlenderProc 2.6.1+
- CUDA-capable GPU (recommended for faster rendering)

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd clutter6d

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate clutter6d

# Run setup script
chmod +x setup.sh
./setup.sh

# Or manual installation:
conda env create -f environment.yml
conda activate clutter6d
```

### Dependencies

Core packages managed by conda:
- `python=3.9` - Python runtime
- `numpy>=1.21.0` - Numerical computations
- `opencv` - Image processing
- `trimesh` - 3D mesh handling
- `matplotlib>=3.3.0` - Visualization
- `pillow>=8.0.0` - Image handling
- `pyyaml>=5.4.0` - Configuration files
- `scipy>=1.7.0` - Scientific computing
- `scikit-image>=0.18.0` - Image processing
- `tqdm>=4.62.0` - Progress bars
- `imageio>=2.9.0` - Image I/O

Pip-only packages:
- `blenderproc==2.6.1` - Main rendering engine
- `pyquaternion>=0.9.9` - Quaternion operations

## Quick Start

### 1. Configure the Pipeline

```bash
# Generate a default configuration
python scripts/config_generator.py --output configs/my_config.yaml \
    --models-dir /path/to/3d/models \
    --num-scenes 1000 \
    --image-size 640 480

# Or use the provided default
cp configs/default_config.yaml configs/my_config.yaml
```

### 2. Generate Dataset

```bash
# Basic generation
python scripts/generate_dataset.py \
    --config configs/my_config.yaml \
    --output-dir data/synthetic

# With custom parameters
python scripts/generate_dataset.py \
    --config configs/my_config.yaml \
    --output-dir data/synthetic \
    --num-scenes 500 \
    --seed 42
```

### 3. Visualize Results

```bash
# Visualize single scene
python scripts/visualize_results.py \
    --data-dir data/synthetic \
    --scene-id 0 \
    --show-poses \
    --show-bboxes

# Create scene montage
python scripts/visualize_results.py \
    --data-dir data/synthetic \
    --create-montage \
    --num-scenes 9
```

## Configuration

The pipeline uses YAML configuration files. Key sections:

### Models Configuration
```yaml
models:
  data_dir: "/path/to/3d_models"
  collections: ["GSO", "Objaverse", "OmniObject3D"]
  max_models: 10000
  supported_formats: [".glb", ".obj", ".ply"]
```

### Scene Generation
```yaml
scene:
  object_count_range: [5, 20]
  allow_multiple_instances: true
  min_visibility_threshold: 0.2
  scene_bounds: [-2.0, 2.0, -2.0, 2.0, 0.0, 3.0]
```

### Camera Settings
```yaml
camera:
  image_size: [640, 480]
  fov_range: [40, 80]
  distance_range: [1.0, 4.0]
  elevation_range: [-30, 60]
  azimuth_range: [0, 360]
```

### Physics Simulation
```yaml
physics:
  gravity: [0, 0, -9.81]
  simulation_steps: 100
  enable_physics: true
  drop_height_range: [0.5, 2.0]
```

### Quality Control
```yaml
quality_control:
  min_objects_visible: 3
  max_occlusion_ratio: 0.8
  min_scene_coverage: 0.3
  balance_classes: true
```

## Output Format

### Directory Structure
```
output_dir/
├── images/
│   ├── rgb/               # RGB images
│   ├── depth/             # Depth maps (PNG, uint16)
│   └── segmentation/      # Instance segmentation masks
├── annotations/
│   ├── scene_XXXXXX_annotations.json    # Complete annotations
│   ├── scene_XXXXXX_coco.json          # COCO format (optional)
│   ├── scene_XXXXXX_camera.json        # Camera parameters
│   └── scene_XXXXXX_metadata.json      # Scene metadata
└── logs/
    └── generation_report_TIMESTAMP.json
```

### Annotation Format

Each object includes:
- **6DoF Pose**: Translation, rotation (multiple formats)
- **2D Bounding Box**: COCO format [x, y, width, height]
- **Visibility Metrics**: Pixel count, visibility ratio
- **Object Metadata**: Class, collection, instance ID

Example object annotation:
```json
{
  "id": 1,
  "object_id": "GSO_chair_001",
  "category": "chair",
  "pose_6d": {
    "translation": [0.5, -0.2, 0.1],
    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "quaternion": [1.0, 0.0, 0.0, 0.0],
    "transform_matrix": "4x4 homogeneous matrix"
  },
  "bbox_2d": {
    "bbox": [100, 150, 80, 120],
    "area": 9600
  },
  "visibility": {
    "visible_pixels": 2400,
    "visibility_ratio": 0.85
  }
}
```

## Advanced Usage

### Custom Object Sampling
```python
from core.object_manager import ObjectManager

config = load_config("configs/my_config.yaml")
object_manager = ObjectManager(config)

# Sample specific classes
objects = object_manager.sample_objects(
    num_objects=10,
    class_filter=["chair", "table", "bottle"]
)
```

### Multi-viewpoint Generation
```python
from utils.camera_utils import CameraManager

camera_manager = CameraManager(config)
viewpoints = camera_manager.sample_multiple_viewpoints(4)

for viewpoint in viewpoints:
    camera_manager.apply_viewpoint(viewpoint)
    # Render scene...
```

### Quality Analysis
```python
from utils.quality_control import QualityController

quality_controller = QualityController(config)
validation_result = quality_controller.validate_scene_quality(
    placement_results, blender_objects, rendered_data
)

if not validation_result['valid']:
    suggestions = quality_controller.suggest_quality_improvements(validation_result)
```

## Performance Optimization

### GPU Acceleration
- Install CUDA-enabled BlenderProc
- Use GPU devices for Cycles rendering
- Enable OptiX denoising (NVIDIA GPUs)

### Memory Management
- Process scenes in batches
- Clear Blender scene between generations
- Use smaller image resolutions for faster iteration

### Parallel Generation
```bash
# Split generation across multiple processes
python scripts/generate_dataset.py --config config.yaml --output-dir batch1 --num-scenes 250 &
python scripts/generate_dataset.py --config config.yaml --output-dir batch2 --num-scenes 250 &
```

## Troubleshooting

### Common Issues

1. **BlenderProc Installation**
   ```bash
   pip install blenderproc
   # If issues, try:
   pip install blenderproc --no-cache-dir
   ```

2. **GPU Memory Issues**
   - Reduce `rendering.samples`
   - Use smaller `camera.image_size`
   - Reduce `scene.object_count_range`

3. **Physics Simulation Problems**
   - Increase `physics.simulation_steps`
   - Adjust `physics.drop_height_range`
   - Check `scene.scene_bounds`

4. **Low Quality Scenes**
   - Adjust `quality_control` thresholds
   - Improve lighting setup
   - Check object model quality

### Debug Mode
```bash
python scripts/generate_dataset.py \
    --config configs/debug_config.yaml \
    --log-level DEBUG \
    --num-scenes 1
```

## Extending the Pipeline

### Adding New Object Sources
1. Implement loader in `core/object_manager.py`
2. Add collection configuration
3. Update supported formats if needed

### Custom Annotation Formats
1. Extend `core/annotation_generator.py`
2. Add format-specific export methods
3. Update configuration options

### Additional Quality Metrics
1. Extend `utils/quality_control.py`
2. Add new validation criteria
3. Update suggestion system

## Examples

See `examples/example_usage.py` for comprehensive usage examples:

```bash
python examples/example_usage.py
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{clutter6d2025,
  title={Clutter6D: Synthetic Dataset Generation for 6DoF Object Pose Estimation},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/clutter6d}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Add tests for new functionality
4. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
python -m pytest tests/

# Code formatting
black src/
isort src/
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- BlenderProc team for the excellent rendering framework
- GSO, Objaverse, and OmniObject3D for 3D model datasets
- Open3D and trimesh communities for 3D processing tools
