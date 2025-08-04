"""
Core components for the Clutter6D synthetic dataset generation pipeline.
"""

from .scene_generator import SceneGenerator
from .object_manager import ObjectManager
from .physics_simulator import PhysicsSimulator
from .annotation_generator import AnnotationGenerator

__all__ = [
    'SceneGenerator',
    'ObjectManager', 
    'PhysicsSimulator',
    'AnnotationGenerator'
]
