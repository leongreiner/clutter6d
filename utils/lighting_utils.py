"""
Lighting utilities for managing HDRI environments, light sources, and material randomization.
"""

import logging
import random
import glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import blenderproc as bproc
import bpy

logger = logging.getLogger(__name__)


class LightingManager:
    """Manages lighting setup including HDRI environments and additional light sources."""
    
    def __init__(self, config: Dict):
        """
        Initialize the LightingManager.
        
        Args:
            config: Configuration dictionary containing lighting settings
        """
        self.config = config
        self.lighting_config = config['lighting']
        self.materials_config = config['materials']
        
        # Lighting parameters
        self.hdri_dir = Path(self.lighting_config['hdri_dir'])
        self.use_hdri = self.lighting_config['use_hdri']
        self.hdri_strength_range = self.lighting_config['hdri_strength_range']
        self.additional_lights = self.lighting_config['additional_lights']
        self.num_additional_lights_range = self.lighting_config['num_additional_lights_range']
        self.light_energy_range = self.lighting_config['light_energy_range']
        
        # Material parameters
        self.enable_material_randomization = self.materials_config['enable_material_randomization']
        self.pbr_material_prob = self.materials_config['pbr_material_prob']
        self.metallic_range = self.materials_config['metallic_range']
        self.roughness_range = self.materials_config['roughness_range']
        
        # Available HDRI files
        self.hdri_files = self._scan_hdri_files()
        
    def _scan_hdri_files(self) -> List[Path]:
        """Scan for available HDRI files."""
        hdri_files = []
        
        if self.hdri_dir.exists():
            patterns = ['*.hdr', '*.exr', '*.hdri']
            for pattern in patterns:
                hdri_files.extend(self.hdri_dir.glob(pattern))
                hdri_files.extend(self.hdri_dir.glob(f"**/{pattern}"))
                
        logger.info(f"Found {len(hdri_files)} HDRI files")
        return hdri_files
        
    def setup_scene_lighting(self) -> Dict:
        """
        Setup complete lighting for the scene.
        
        Returns:
            Lighting configuration dictionary
        """
        logger.debug("Setting up scene lighting...")
        
        lighting_config = {
            'hdri_used': None,
            'hdri_strength': None,
            'additional_lights': [],
            'material_randomization': self.enable_material_randomization
        }
        
        # Setup HDRI environment
        if self.use_hdri and self.hdri_files:
            hdri_config = self._setup_hdri_environment()
            lighting_config.update(hdri_config)
        else:
            # Setup basic environment lighting
            self._setup_basic_environment()
            
        # Add additional light sources
        if self.additional_lights:
            additional_lights_config = self._add_additional_lights()
            lighting_config['additional_lights'] = additional_lights_config
            
        return lighting_config
        
    def _setup_hdri_environment(self) -> Dict:
        """Setup HDRI environment lighting."""
        # Select random HDRI file
        hdri_file = random.choice(self.hdri_files)
        
        # Random strength
        hdri_strength = random.uniform(*self.hdri_strength_range)
        
        try:
            # Load HDRI environment
            bproc.world.set_world_background_hdr_img(str(hdri_file), strength=hdri_strength)
            
            logger.debug(f"Loaded HDRI: {hdri_file.name} with strength {hdri_strength:.2f}")
            
            return {
                'hdri_used': str(hdri_file),
                'hdri_strength': hdri_strength
            }
            
        except Exception as e:
            logger.warning(f"Failed to load HDRI {hdri_file}: {e}")
            self._setup_basic_environment()
            return {'hdri_used': None, 'hdri_strength': None}
            
    def _setup_basic_environment(self):
        """Setup basic environment lighting without HDRI."""
        # Set a random colored environment
        env_color = [
            random.uniform(0.1, 0.3),  # R
            random.uniform(0.1, 0.3),  # G
            random.uniform(0.1, 0.3)   # B
        ]
        
        env_strength = random.uniform(0.5, 1.5)
        
        bproc.world.set_world_background(env_color, strength=env_strength)
        
    def _add_additional_lights(self) -> List[Dict]:
        """Add additional light sources to the scene."""
        num_lights = random.randint(*self.num_additional_lights_range)
        lights_config = []
        
        for i in range(num_lights):
            light_config = self._create_random_light(i)
            lights_config.append(light_config)
            
        return lights_config
        
    def _create_random_light(self, light_id: int) -> Dict:
        """Create a random light source."""
        # Light types
        light_types = ['SUN', 'POINT', 'SPOT', 'AREA']
        light_type = random.choice(light_types)
        
        # Light energy
        energy = random.uniform(*self.light_energy_range)
        
        # Light color (slightly warm to cool)
        color_temp = random.uniform(3000, 6500)  # Kelvin
        color = self._kelvin_to_rgb(color_temp)
        
        # Light position (around the scene)
        scene_bounds = self.config['scene']['scene_bounds']
        margin = 2.0
        
        if light_type == 'SUN':
            # Sun light - position doesn't matter much, but set direction
            position = [0, 0, 5]
            direction = [
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                random.uniform(-1, -0.3)  # Generally downward
            ]
        else:
            # Point, spot, area lights - position around scene
            position = [
                random.uniform(scene_bounds[0] - margin, scene_bounds[1] + margin),
                random.uniform(scene_bounds[2] - margin, scene_bounds[3] + margin),
                random.uniform(scene_bounds[5], scene_bounds[5] + margin)
            ]
            direction = None
            
        # Create light in Blender
        light = self._create_blender_light(light_type, position, energy, color, direction)
        
        light_config = {
            'light_id': light_id,
            'type': light_type,
            'position': position,
            'direction': direction,
            'energy': energy,
            'color': color,
            'color_temperature': color_temp,
            'blender_object': light
        }
        
        return light_config
        
    def _create_blender_light(self, light_type: str, position: List[float], 
                            energy: float, color: List[float], 
                            direction: Optional[List[float]] = None):
        """Create actual light object in Blender."""
        # Create light
        bpy.ops.object.light_add(type=light_type, location=position)
        light = bpy.context.active_object
        
        # Set light properties
        light.data.energy = energy
        light.data.color = color
        
        # Set direction for directional lights
        if direction and light_type in ['SUN', 'SPOT']:
            # Calculate rotation to point in direction
            import mathutils
            direction_vec = mathutils.Vector(direction).normalized()
            # Point the light in the specified direction
            light.rotation_euler = direction_vec.to_track_quat('-Z', 'Y').to_euler()
            
        # Light-specific settings
        if light_type == 'SPOT':
            light.data.spot_size = random.uniform(0.5, 2.0)  # Spot size in radians
            light.data.spot_blend = random.uniform(0.1, 0.5)  # Spot blend
            
        elif light_type == 'AREA':
            light.data.shape = random.choice(['SQUARE', 'RECTANGLE'])
            light.data.size = random.uniform(0.5, 2.0)
            if light.data.shape == 'RECTANGLE':
                light.data.size_y = random.uniform(0.5, 2.0)
                
        return light
        
    def _kelvin_to_rgb(self, kelvin: float) -> List[float]:
        """
        Convert color temperature in Kelvin to RGB.
        Simplified approximation for lighting.
        """
        # Clamp kelvin to reasonable range
        kelvin = max(1000, min(12000, kelvin))
        
        temp = kelvin / 100.0
        
        # Red component
        if temp <= 66:
            red = 255
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
            red = max(0, min(255, red))
            
        # Green component
        if temp <= 66:
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
        else:
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)
        green = max(0, min(255, green))
        
        # Blue component
        if temp >= 66:
            blue = 255
        elif temp <= 19:
            blue = 0
        else:
            blue = temp - 10
            blue = 138.5177312231 * np.log(blue) - 305.0447927307
            blue = max(0, min(255, blue))
            
        return [red/255.0, green/255.0, blue/255.0]
        
    def randomize_materials(self, objects: List) -> Dict:
        """
        Randomize materials for objects in the scene.
        
        Args:
            objects: List of Blender objects
            
        Returns:
            Material randomization configuration
        """
        if not self.enable_material_randomization:
            return {'material_randomization': False}
            
        logger.debug(f"Randomizing materials for {len(objects)} objects")
        
        material_configs = []
        
        for i, obj in enumerate(objects):
            if random.random() < self.pbr_material_prob:
                # Create PBR material
                material_config = self._create_pbr_material(obj, i)
            else:
                # Use simple material
                material_config = self._create_simple_material(obj, i)
                
            material_configs.append(material_config)
            
        return {
            'material_randomization': True,
            'materials': material_configs
        }
        
    def _create_pbr_material(self, obj, material_id: int) -> Dict:
        """Create a PBR material for an object."""
        # Create new material
        material = bproc.material.create(f"pbr_material_{material_id}")
        
        # Random base color
        base_color = [
            random.uniform(0.1, 0.9),
            random.uniform(0.1, 0.9),
            random.uniform(0.1, 0.9),
            1.0  # Alpha
        ]
        
        # Random metallic and roughness
        metallic = random.uniform(*self.metallic_range)
        roughness = random.uniform(*self.roughness_range)
        
        # Set material properties
        material.set_principled_shader_value("Base Color", base_color)
        material.set_principled_shader_value("Metallic", metallic)
        material.set_principled_shader_value("Roughness", roughness)
        
        # Optional: Add some variation
        if random.random() < 0.3:  # 30% chance of emission
            emission_strength = random.uniform(0.1, 1.0)
            material.set_principled_shader_value("Emission Strength", emission_strength)
            
        if random.random() < 0.2:  # 20% chance of subsurface
            subsurface = random.uniform(0.1, 0.5)
            material.set_principled_shader_value("Subsurface", subsurface)
            
        # Apply material to object
        obj.set_material(0, material)
        
        config = {
            'material_id': material_id,
            'type': 'PBR',
            'base_color': base_color,
            'metallic': metallic,
            'roughness': roughness,
            'object_id': obj.get('object_data', {}).get('id', 'unknown')
        }
        
        return config
        
    def _create_simple_material(self, obj, material_id: int) -> Dict:
        """Create a simple material for an object."""
        # Create new material
        material = bproc.material.create(f"simple_material_{material_id}")
        
        # Random color
        color = [
            random.uniform(0.2, 0.8),
            random.uniform(0.2, 0.8),
            random.uniform(0.2, 0.8),
            1.0
        ]
        
        # Simple roughness
        roughness = random.uniform(0.3, 0.8)
        
        # Set basic properties
        material.set_principled_shader_value("Base Color", color)
        material.set_principled_shader_value("Roughness", roughness)
        material.set_principled_shader_value("Metallic", 0.0)
        
        # Apply material to object
        obj.set_material(0, material)
        
        config = {
            'material_id': material_id,
            'type': 'Simple',
            'color': color,
            'roughness': roughness,
            'object_id': obj.get('object_data', {}).get('id', 'unknown')
        }
        
        return config
        
    def add_random_noise_to_lighting(self, noise_strength: float = 0.1):
        """Add random noise/variation to current lighting setup."""
        # Add subtle variations to existing lights
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                # Vary energy slightly
                current_energy = obj.data.energy
                noise = random.uniform(-noise_strength, noise_strength)
                obj.data.energy = max(0.1, current_energy * (1 + noise))
                
                # Vary position slightly for non-sun lights
                if obj.data.type != 'SUN':
                    pos_noise = [
                        random.uniform(-0.2, 0.2),
                        random.uniform(-0.2, 0.2),
                        random.uniform(-0.1, 0.1)
                    ]
                    obj.location = [
                        obj.location[i] + pos_noise[i] for i in range(3)
                    ]
                    
    def cleanup_lights(self):
        """Clean up all created lights."""
        # Remove all lights except default
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT' and obj.name != 'Light':
                bpy.data.objects.remove(obj)
