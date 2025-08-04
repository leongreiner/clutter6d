"""
Object Manager for loading and managing 3D CAD models from various collections.
Supports GSO, Objaverse, and OmniObject3D datasets.
"""

import os
import glob
import json
import random
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import trimesh
import numpy as np

logger = logging.getLogger(__name__)


class ObjectManager:
    """Manages 3D object models and provides sampling functionality."""
    
    def __init__(self, config: Dict):
        """
        Initialize the ObjectManager.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.models_dir = Path(config['models']['data_dir'])
        self.collections = config['models']['collections']
        self.max_models = config['models']['max_models']
        self.supported_formats = config['models']['supported_formats']
        
        # Object database
        self.object_database = {}
        self.class_to_objects = {}
        self.loaded_meshes = {}
        
        # Load object database
        self._load_object_database()
        
    def _load_object_database(self):
        """Load and index all available 3D models."""
        logger.info("Loading object database...")
        
        total_loaded = 0
        for collection in self.collections:
            collection_path = self.models_dir / collection
            if not collection_path.exists():
                logger.warning(f"Collection path does not exist: {collection_path}")
                continue
                
            collection_models = self._scan_collection(collection_path, collection)
            self.object_database[collection] = collection_models
            total_loaded += len(collection_models)
            
            # Group by class
            for model_info in collection_models:
                class_name = model_info['class']
                if class_name not in self.class_to_objects:
                    self.class_to_objects[class_name] = []
                self.class_to_objects[class_name].append(model_info)
                
            if total_loaded >= self.max_models:
                break
                
        logger.info(f"Loaded {total_loaded} models from {len(self.object_database)} collections")
        logger.info(f"Found {len(self.class_to_objects)} unique object classes")
        
    def _scan_collection(self, collection_path: Path, collection_name: str) -> List[Dict]:
        """
        Scan a collection directory for 3D models.
        
        Args:
            collection_path: Path to collection directory
            collection_name: Name of the collection
            
        Returns:
            List of model information dictionaries
        """
        models = []
        
        # Find all supported model files
        for format_ext in self.supported_formats:
            pattern = f"**/*{format_ext}"
            model_files = list(collection_path.glob(pattern))
            
            for model_file in model_files:
                try:
                    # Extract class/category from path structure
                    relative_path = model_file.relative_to(collection_path)
                    class_name = self._extract_class_name(relative_path, collection_name)
                    
                    # Validate model file
                    if self._validate_model_file(model_file):
                        model_info = {
                            'id': f"{collection_name}_{model_file.stem}",
                            'path': str(model_file),
                            'collection': collection_name,
                            'class': class_name,
                            'format': format_ext,
                            'size': model_file.stat().st_size
                        }
                        models.append(model_info)
                        
                except Exception as e:
                    logger.debug(f"Skipping file {model_file}: {e}")
                    continue
                    
        return models[:self.max_models // len(self.collections)]
        
    def _extract_class_name(self, relative_path: Path, collection_name: str) -> str:
        """
        Extract object class name from file path.
        
        Args:
            relative_path: Relative path from collection root
            collection_name: Name of the collection
            
        Returns:
            Extracted class name
        """
        # Different collections have different directory structures
        if collection_name.lower() == 'gso':
            # GSO typically has structure: category/object_name/model.glb
            return relative_path.parts[0] if len(relative_path.parts) > 1 else 'unknown'
        elif collection_name.lower() == 'objaverse':
            # Objaverse might have categories in path or filename
            if len(relative_path.parts) > 1:
                return relative_path.parts[0]
            else:
                # Try to extract from filename
                filename = relative_path.stem
                return filename.split('_')[0] if '_' in filename else 'object'
        elif collection_name.lower() == 'omniobject3d':
            # OmniObject3D structure varies
            return relative_path.parts[0] if len(relative_path.parts) > 1 else 'object'
        else:
            return 'unknown'
            
    def _validate_model_file(self, model_path: Path) -> bool:
        """
        Validate that a model file is loadable and has reasonable properties.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if model is valid, False otherwise
        """
        try:
            # Check file size (not too small, not too large)
            file_size = model_path.stat().st_size
            if file_size < 1024 or file_size > 100 * 1024 * 1024:  # 1KB to 100MB
                return False
                
            # Try to load mesh for basic validation
            mesh = trimesh.load(str(model_path))
            if isinstance(mesh, trimesh.Scene):
                # Handle scenes
                if len(mesh.geometry) == 0:
                    return False
                # Use the largest geometry in the scene
                mesh = max(mesh.geometry.values(), key=lambda x: x.volume if hasattr(x, 'volume') else 0)
                
            if not hasattr(mesh, 'vertices') or len(mesh.vertices) < 3:
                return False
                
            return True
            
        except Exception as e:
            logger.debug(f"Model validation failed for {model_path}: {e}")
            return False
            
    def load_mesh(self, model_info: Dict) -> Optional[trimesh.Trimesh]:
        """
        Load a 3D mesh from model information.
        
        Args:
            model_info: Model information dictionary
            
        Returns:
            Loaded trimesh object or None if loading fails
        """
        model_id = model_info['id']
        
        # Check cache first
        if model_id in self.loaded_meshes:
            return self.loaded_meshes[model_id]
            
        try:
            mesh = trimesh.load(model_info['path'])
            
            # Handle scenes
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) == 0:
                    return None
                # Use the largest geometry
                mesh = max(mesh.geometry.values(), key=lambda x: x.volume if hasattr(x, 'volume') else 0)
                
            # Normalize mesh
            mesh = self._normalize_mesh(mesh)
            
            # Cache the mesh
            self.loaded_meshes[model_id] = mesh
            
            return mesh
            
        except Exception as e:
            logger.error(f"Failed to load mesh {model_info['path']}: {e}")
            return None
            
    def _normalize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Normalize mesh to standard size and position.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Normalized mesh
        """
        # Center the mesh
        mesh.vertices -= mesh.center_mass
        
        # Scale to unit size (largest dimension = 1)
        scale = 1.0 / mesh.extents.max()
        mesh.vertices *= scale
        
        return mesh
        
    def sample_objects(self, num_objects: int, allow_multiple_instances: bool = True,
                      class_filter: Optional[List[str]] = None) -> List[Dict]:
        """
        Sample objects for scene generation.
        
        Args:
            num_objects: Number of objects to sample
            allow_multiple_instances: Whether to allow multiple instances of same class
            class_filter: Optional list of classes to filter by
            
        Returns:
            List of sampled object information
        """
        available_classes = list(self.class_to_objects.keys())
        
        if class_filter:
            available_classes = [c for c in available_classes if c in class_filter]
            
        if not available_classes:
            logger.error("No available object classes found")
            return []
            
        sampled_objects = []
        used_classes = set()
        
        for _ in range(num_objects):
            if allow_multiple_instances or len(used_classes) < len(available_classes):
                # Select class
                if allow_multiple_instances:
                    selected_class = random.choice(available_classes)
                else:
                    available_unused = [c for c in available_classes if c not in used_classes]
                    if available_unused:
                        selected_class = random.choice(available_unused)
                    else:
                        selected_class = random.choice(available_classes)
                        
                # Select object from class
                class_objects = self.class_to_objects[selected_class]
                selected_object = random.choice(class_objects).copy()
                
                # Add instance-specific information
                selected_object['instance_id'] = len(sampled_objects)
                
                sampled_objects.append(selected_object)
                used_classes.add(selected_class)
            else:
                break
                
        return sampled_objects
        
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of objects per class."""
        return {class_name: len(objects) for class_name, objects in self.class_to_objects.items()}
        
    def get_collection_stats(self) -> Dict[str, Dict]:
        """Get statistics for each collection."""
        stats = {}
        for collection, models in self.object_database.items():
            stats[collection] = {
                'total_models': len(models),
                'classes': len(set(m['class'] for m in models)),
                'avg_file_size': np.mean([m['size'] for m in models]) if models else 0
            }
        return stats
