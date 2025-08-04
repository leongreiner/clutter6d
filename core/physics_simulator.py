"""
Physics Simulator for realistic object placement in cluttered scenes.
Uses BlenderProc's physics simulation capabilities.
"""

import logging
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
import blenderproc as bproc
import bmesh
import bpy
from mathutils import Vector, Matrix

logger = logging.getLogger(__name__)


class PhysicsSimulator:
    """Handles physics-based object placement and simulation."""
    
    def __init__(self, config: Dict):
        """
        Initialize the PhysicsSimulator.
        
        Args:
            config: Configuration dictionary containing physics settings
        """
        self.config = config
        self.physics_config = config['physics']
        self.scene_config = config['scene']
        
        # Physics parameters
        self.gravity = self.physics_config['gravity']
        self.simulation_steps = self.physics_config['simulation_steps']
        self.max_simulation_time = self.physics_config['max_simulation_time']
        self.collision_margin = self.physics_config['collision_margin']
        self.drop_height_range = self.physics_config['drop_height_range']
        
        # Scene bounds
        self.scene_bounds = self.scene_config['scene_bounds']  # [x_min, x_max, y_min, y_max, z_min, z_max]
        
    def setup_physics_world(self):
        """Setup the physics world with gravity and collision detection."""
        logger.info("Setting up physics world...")
        
        # Set gravity
        bpy.context.scene.gravity = self.gravity
        
        # Enable physics simulation
        bpy.context.scene.frame_set(1)
        
        # Create ground plane
        self._create_ground_plane()
        
    def _create_ground_plane(self):
        """Create an invisible ground plane for physics collision."""
        # Create ground plane
        bpy.ops.mesh.primitive_plane_add(
            size=max(abs(self.scene_bounds[1] - self.scene_bounds[0]),
                    abs(self.scene_bounds[3] - self.scene_bounds[2])) * 2,
            location=(0, 0, self.scene_bounds[4])
        )
        
        ground = bpy.context.active_object
        ground.name = "GroundPlane"
        
        # Add rigid body physics
        bpy.ops.rigidbody.object_add(type='PASSIVE')
        ground.rigid_body.type = 'PASSIVE'
        ground.rigid_body.collision_shape = 'BOX'
        
        # Make invisible
        ground.hide_render = True
        ground.hide_viewport = True
        
    def place_objects_with_physics(self, objects_data: List[Dict], 
                                 blender_objects: List[bpy.types.Object]) -> List[Dict]:
        """
        Place objects using physics simulation for realistic positioning.
        
        Args:
            objects_data: List of object information dictionaries
            blender_objects: List of corresponding Blender objects
            
        Returns:
            List of placement results with final poses
        """
        logger.info(f"Placing {len(objects_data)} objects with physics simulation...")
        
        placement_results = []
        
        # Set initial positions for objects
        for obj_data, blender_obj in zip(objects_data, blender_objects):
            initial_pose = self._generate_initial_pose(obj_data, blender_obj)
            self._apply_pose_to_object(blender_obj, initial_pose)
            
            # Setup rigid body physics
            self._setup_object_physics(blender_obj)
            
            placement_results.append({
                'object_data': obj_data,
                'blender_object': blender_obj,
                'initial_pose': initial_pose,
                'final_pose': None,
                'settled': False
            })
        
        # Run physics simulation
        final_poses = self._run_physics_simulation(placement_results)
        
        # Update placement results with final poses
        for i, result in enumerate(placement_results):
            result['final_pose'] = final_poses[i]
            result['settled'] = self._check_object_settled(result['blender_object'])
            
        return placement_results
        
    def _generate_initial_pose(self, obj_data: Dict, blender_obj: bpy.types.Object) -> Dict:
        """
        Generate initial pose for object placement.
        
        Args:
            obj_data: Object information dictionary
            blender_obj: Blender object
            
        Returns:
            Initial pose dictionary with position, rotation, and scale
        """
        # Random position within scene bounds (elevated for dropping)
        x = random.uniform(self.scene_bounds[0], self.scene_bounds[1])
        y = random.uniform(self.scene_bounds[2], self.scene_bounds[3])
        z = random.uniform(
            self.scene_bounds[5] + self.drop_height_range[0],
            self.scene_bounds[5] + self.drop_height_range[1]
        )
        position = [x, y, z]
        
        # Random rotation
        rotation = [
            random.uniform(0, 2 * np.pi),  # X rotation
            random.uniform(0, 2 * np.pi),  # Y rotation
            random.uniform(0, 2 * np.pi)   # Z rotation
        ]
        
        # Random scale variation (within reasonable bounds)
        scale_factor = random.uniform(0.8, 1.2)
        scale = [scale_factor, scale_factor, scale_factor]
        
        return {
            'position': position,
            'rotation': rotation,
            'scale': scale
        }
        
    def _apply_pose_to_object(self, blender_obj: bpy.types.Object, pose: Dict):
        """Apply pose to Blender object."""
        blender_obj.location = pose['position']
        blender_obj.rotation_euler = pose['rotation']
        blender_obj.scale = pose['scale']
        
    def _setup_object_physics(self, blender_obj: bpy.types.Object):
        """Setup rigid body physics for an object."""
        # Select the object
        bpy.context.view_layer.objects.active = blender_obj
        blender_obj.select_set(True)
        
        # Add rigid body
        bpy.ops.rigidbody.object_add(type='ACTIVE')
        
        # Configure rigid body properties
        rb = blender_obj.rigid_body
        rb.type = 'ACTIVE'
        rb.collision_shape = 'CONVEX_HULL'  # Good balance of accuracy and performance
        rb.mass = self._estimate_object_mass(blender_obj)
        rb.friction = random.uniform(0.3, 0.8)
        rb.restitution = random.uniform(0.1, 0.4)  # Bounce
        rb.collision_margin = self.collision_margin
        
        # Enable collision detection
        rb.use_margin = True
        
        # Deselect
        blender_obj.select_set(False)
        
    def _estimate_object_mass(self, blender_obj: bpy.types.Object) -> float:
        """Estimate realistic mass for an object based on its volume."""
        # Calculate object volume
        bpy.context.view_layer.objects.active = blender_obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        bm = bmesh.from_mesh(blender_obj.data)
        bm.faces.ensure_lookup_table()
        volume = bm.calc_volume()
        bm.free()
        
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Estimate density based on object type/material (simplified)
        # Assume average density of common household objects
        density = random.uniform(0.5, 2.0)  # kg/m³ equivalent
        
        mass = max(0.1, volume * density)  # Minimum mass of 0.1
        return mass
        
    def _run_physics_simulation(self, placement_results: List[Dict]) -> List[Dict]:
        """
        Run physics simulation and return final poses.
        
        Args:
            placement_results: List of placement result dictionaries
            
        Returns:
            List of final pose dictionaries
        """
        logger.info("Running physics simulation...")
        
        # Set simulation parameters
        scene = bpy.context.scene
        scene.rigidbody_world.enabled = True
        scene.rigidbody_world.steps_per_second = 60
        scene.rigidbody_world.solver_iterations = 10
        
        # Run simulation
        start_frame = scene.frame_current
        end_frame = start_frame + self.simulation_steps
        
        for frame in range(start_frame, end_frame + 1):
            scene.frame_set(frame)
            
            # Check if simulation has settled early
            if frame > start_frame + 20 and self._check_simulation_settled(placement_results):
                logger.info(f"Simulation settled early at frame {frame}")
                break
                
        # Collect final poses
        final_poses = []
        for result in placement_results:
            blender_obj = result['blender_object']
            final_pose = {
                'position': list(blender_obj.location),
                'rotation': list(blender_obj.rotation_euler),
                'scale': list(blender_obj.scale),
                'matrix_world': np.array(blender_obj.matrix_world)
            }
            final_poses.append(final_pose)
            
        return final_poses
        
    def _check_simulation_settled(self, placement_results: List[Dict]) -> bool:
        """Check if all objects have settled (low velocity)."""
        settled_count = 0
        
        for result in placement_results:
            if self._check_object_settled(result['blender_object']):
                settled_count += 1
                
        # Consider settled if 80% of objects are stable
        return settled_count >= len(placement_results) * 0.8
        
    def _check_object_settled(self, blender_obj: bpy.types.Object) -> bool:
        """Check if a single object has settled."""
        if not blender_obj.rigid_body:
            return True
            
        # Check linear and angular velocity
        rb = blender_obj.rigid_body
        if hasattr(rb, 'angular_velocity') and hasattr(rb, 'linear_velocity'):
            linear_vel = Vector(rb.linear_velocity).length
            angular_vel = Vector(rb.angular_velocity).length
            
            # Thresholds for considering object settled
            return linear_vel < 0.01 and angular_vel < 0.1
            
        return True
        
    def validate_placement(self, placement_results: List[Dict]) -> Dict:
        """
        Validate the quality of object placement.
        
        Args:
            placement_results: List of placement result dictionaries
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'total_objects': len(placement_results),
            'settled_objects': 0,
            'objects_in_bounds': 0,
            'collision_count': 0,
            'objects_out_of_bounds': [],
            'unstable_objects': [],
            'valid_placement': True
        }
        
        for i, result in enumerate(placement_results):
            blender_obj = result['blender_object']
            final_pose = result['final_pose']
            
            # Check if object settled
            if result['settled']:
                validation_results['settled_objects'] += 1
            else:
                validation_results['unstable_objects'].append(i)
                
            # Check if object is within scene bounds
            pos = final_pose['position']
            if (self.scene_bounds[0] <= pos[0] <= self.scene_bounds[1] and
                self.scene_bounds[2] <= pos[1] <= self.scene_bounds[3] and
                self.scene_bounds[4] <= pos[2] <= self.scene_bounds[5]):
                validation_results['objects_in_bounds'] += 1
            else:
                validation_results['objects_out_of_bounds'].append(i)
                
        # Check overall placement validity
        settled_ratio = validation_results['settled_objects'] / validation_results['total_objects']
        in_bounds_ratio = validation_results['objects_in_bounds'] / validation_results['total_objects']
        
        validation_results['valid_placement'] = (
            settled_ratio >= 0.8 and  # At least 80% settled
            in_bounds_ratio >= 0.9    # At least 90% in bounds
        )
        
        return validation_results
        
    def cleanup_physics(self):
        """Clean up physics simulation objects and settings."""
        # Remove rigid body world
        if bpy.context.scene.rigidbody_world:
            bpy.ops.rigidbody.world_remove()
            
        # Remove ground plane
        ground_plane = bpy.data.objects.get("GroundPlane")
        if ground_plane:
            bpy.data.objects.remove(ground_plane)
