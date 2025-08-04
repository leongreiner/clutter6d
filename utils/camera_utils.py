"""
Camera utilities for managing camera poses, intrinsics, and viewpoint sampling.
"""

import logging
import random
import numpy as np
import blenderproc as bproc
from mathutils import Vector, Matrix
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class CameraManager:
    """Manages camera positioning, intrinsics, and pose sampling for scene generation."""
    
    def __init__(self, config: Dict):
        """
        Initialize the CameraManager.
        
        Args:
            config: Configuration dictionary containing camera settings
        """
        self.config = config
        self.camera_config = config['camera']
        self.scene_config = config['scene']
        
        # Camera parameters
        self.image_size = self.camera_config['image_size']
        self.fov_range = self.camera_config['fov_range']
        self.distance_range = self.camera_config['distance_range']
        self.elevation_range = self.camera_config['elevation_range']
        self.azimuth_range = self.camera_config['azimuth_range']
        self.look_at_jitter = self.camera_config['look_at_jitter']
        
        # Scene bounds for camera positioning
        self.scene_bounds = self.scene_config['scene_bounds']
        
    def setup_camera_pose(self) -> Dict:
        """
        Setup camera with random pose looking at the scene.
        
        Returns:
            Camera configuration dictionary
        """
        logger.debug("Setting up camera pose...")
        
        # Calculate scene center
        scene_center = self._get_scene_center()
        
        # Sample camera parameters
        camera_params = self._sample_camera_parameters()
        
        # Calculate camera position based on spherical coordinates
        camera_position = self._calculate_camera_position(
            scene_center, 
            camera_params['distance'],
            camera_params['elevation'],
            camera_params['azimuth']
        )
        
        # Add jitter to look-at point
        look_at_point = self._add_look_at_jitter(scene_center)
        
        # Set camera pose
        self._set_camera_pose(camera_position, look_at_point)
        
        # Set camera intrinsics
        self._set_camera_intrinsics(camera_params['fov'])
        
        camera_config = {
            'position': camera_position.tolist(),
            'look_at': look_at_point.tolist(),
            'fov': camera_params['fov'],
            'distance': camera_params['distance'],
            'elevation': camera_params['elevation'],
            'azimuth': camera_params['azimuth'],
            'image_size': self.image_size
        }
        
        return camera_config
        
    def _get_scene_center(self) -> np.ndarray:
        """Calculate the center point of the scene."""
        x_center = (self.scene_bounds[0] + self.scene_bounds[1]) / 2
        y_center = (self.scene_bounds[2] + self.scene_bounds[3]) / 2
        z_center = (self.scene_bounds[4] + self.scene_bounds[5]) / 2
        
        return np.array([x_center, y_center, z_center])
        
    def _sample_camera_parameters(self) -> Dict:
        """Sample random camera parameters within specified ranges."""
        params = {
            'fov': random.uniform(*self.fov_range),
            'distance': random.uniform(*self.distance_range),
            'elevation': random.uniform(*self.elevation_range),
            'azimuth': random.uniform(*self.azimuth_range)
        }
        
        return params
        
    def _calculate_camera_position(self, scene_center: np.ndarray, distance: float,
                                 elevation: float, azimuth: float) -> np.ndarray:
        """
        Calculate camera position using spherical coordinates.
        
        Args:
            scene_center: Center point of the scene
            distance: Distance from scene center
            elevation: Elevation angle in degrees
            azimuth: Azimuth angle in degrees
            
        Returns:
            Camera position as numpy array
        """
        # Convert angles to radians
        elevation_rad = np.radians(elevation)
        azimuth_rad = np.radians(azimuth)
        
        # Spherical to Cartesian conversion
        x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = distance * np.sin(elevation_rad)
        
        # Add to scene center
        camera_position = scene_center + np.array([x, y, z])
        
        return camera_position
        
    def _add_look_at_jitter(self, scene_center: np.ndarray) -> np.ndarray:
        """Add small random jitter to the look-at point."""
        jitter = np.random.uniform(-self.look_at_jitter, self.look_at_jitter, 3)
        return scene_center + jitter
        
    def _set_camera_pose(self, camera_position: np.ndarray, look_at_point: np.ndarray):
        """Set the camera pose in BlenderProc."""
        # Convert to Vector objects for BlenderProc
        cam_pos = Vector(camera_position)
        look_at = Vector(look_at_point)
        
        # Set camera pose
        bproc.camera.add_camera_pose(
            cam_pose=bproc.math.build_transformation_mat(cam_pos, look_at)
        )
        
    def _set_camera_intrinsics(self, fov: float):
        """Set camera intrinsic parameters."""
        # Set resolution
        bproc.camera.set_resolution(*self.image_size)
        
        # Set field of view
        bproc.camera.set_fov(fov)
        
    def sample_multiple_viewpoints(self, num_viewpoints: int, 
                                 scene_center: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Sample multiple camera viewpoints for the same scene.
        
        Args:
            num_viewpoints: Number of viewpoints to sample
            scene_center: Optional scene center override
            
        Returns:
            List of camera configuration dictionaries
        """
        if scene_center is None:
            scene_center = self._get_scene_center()
            
        viewpoints = []
        
        for i in range(num_viewpoints):
            # Sample camera parameters
            camera_params = self._sample_camera_parameters()
            
            # Calculate camera position
            camera_position = self._calculate_camera_position(
                scene_center,
                camera_params['distance'],
                camera_params['elevation'],
                camera_params['azimuth']
            )
            
            # Add jitter to look-at point
            look_at_point = self._add_look_at_jitter(scene_center)
            
            viewpoint_config = {
                'viewpoint_id': i,
                'position': camera_position.tolist(),
                'look_at': look_at_point.tolist(),
                'fov': camera_params['fov'],
                'distance': camera_params['distance'],
                'elevation': camera_params['elevation'],
                'azimuth': camera_params['azimuth'],
                'image_size': self.image_size
            }
            
            viewpoints.append(viewpoint_config)
            
        return viewpoints
        
    def apply_viewpoint(self, viewpoint_config: Dict):
        """Apply a specific viewpoint configuration."""
        camera_position = np.array(viewpoint_config['position'])
        look_at_point = np.array(viewpoint_config['look_at'])
        fov = viewpoint_config['fov']
        
        # Clear existing camera poses
        bproc.camera.clear_poses()
        
        # Set new pose
        self._set_camera_pose(camera_position, look_at_point)
        self._set_camera_intrinsics(fov)
        
    def get_camera_frustum_bounds(self, camera_position: np.ndarray, 
                                look_at_point: np.ndarray, fov: float,
                                near_plane: float = 0.1, far_plane: float = 10.0) -> Dict:
        """
        Calculate camera frustum bounds for visibility calculations.
        
        Args:
            camera_position: Camera position
            look_at_point: Point camera is looking at
            fov: Field of view in degrees
            near_plane: Near clipping plane distance
            far_plane: Far clipping plane distance
            
        Returns:
            Dictionary with frustum information
        """
        # Camera direction vector
        direction = look_at_point - camera_position
        direction = direction / np.linalg.norm(direction)
        
        # Calculate frustum dimensions
        fov_rad = np.radians(fov)
        aspect_ratio = self.image_size[0] / self.image_size[1]
        
        # Near plane dimensions
        near_height = 2 * np.tan(fov_rad / 2) * near_plane
        near_width = near_height * aspect_ratio
        
        # Far plane dimensions
        far_height = 2 * np.tan(fov_rad / 2) * far_plane
        far_width = far_height * aspect_ratio
        
        frustum_info = {
            'camera_position': camera_position.tolist(),
            'direction': direction.tolist(),
            'fov': fov,
            'aspect_ratio': aspect_ratio,
            'near_plane': near_plane,
            'far_plane': far_plane,
            'near_dimensions': [near_width, near_height],
            'far_dimensions': [far_width, far_height]
        }
        
        return frustum_info
        
    def validate_camera_pose(self, camera_position: np.ndarray, 
                           look_at_point: np.ndarray) -> bool:
        """
        Validate if a camera pose is reasonable.
        
        Args:
            camera_position: Camera position
            look_at_point: Look-at point
            
        Returns:
            True if pose is valid, False otherwise
        """
        # Check if camera is not too close to scene bounds
        margin = 0.5
        extended_bounds = [
            self.scene_bounds[0] - margin, self.scene_bounds[1] + margin,
            self.scene_bounds[2] - margin, self.scene_bounds[3] + margin,
            self.scene_bounds[4] - margin, self.scene_bounds[5] + margin
        ]
        
        # Camera should be outside the extended scene bounds
        if (extended_bounds[0] < camera_position[0] < extended_bounds[1] and
            extended_bounds[2] < camera_position[1] < extended_bounds[3] and
            extended_bounds[4] < camera_position[2] < extended_bounds[5]):
            return False
            
        # Check minimum distance to look-at point
        distance = np.linalg.norm(camera_position - look_at_point)
        if distance < self.distance_range[0] or distance > self.distance_range[1]:
            return False
            
        return True
        
    def create_camera_trajectory(self, num_frames: int, trajectory_type: str = 'circular') -> List[Dict]:
        """
        Create a camera trajectory for video generation.
        
        Args:
            num_frames: Number of frames in the trajectory
            trajectory_type: Type of trajectory ('circular', 'linear', 'random')
            
        Returns:
            List of camera configurations for each frame
        """
        scene_center = self._get_scene_center()
        trajectory = []
        
        if trajectory_type == 'circular':
            # Circular trajectory around scene center
            base_distance = np.mean(self.distance_range)
            base_elevation = np.mean(self.elevation_range)
            
            for i in range(num_frames):
                azimuth = (i / num_frames) * 360
                
                camera_position = self._calculate_camera_position(
                    scene_center, base_distance, base_elevation, azimuth
                )
                
                frame_config = {
                    'frame_id': i,
                    'position': camera_position.tolist(),
                    'look_at': scene_center.tolist(),
                    'fov': np.mean(self.fov_range),
                    'distance': base_distance,
                    'elevation': base_elevation,
                    'azimuth': azimuth
                }
                
                trajectory.append(frame_config)
                
        elif trajectory_type == 'random':
            # Random trajectory with smooth interpolation
            keyframes = min(num_frames // 4, 10)  # Sample keyframes
            keyframe_configs = self.sample_multiple_viewpoints(keyframes, scene_center)
            
            # Interpolate between keyframes
            for i in range(num_frames):
                t = (i / num_frames) * (keyframes - 1)
                keyframe_idx = int(t)
                alpha = t - keyframe_idx
                
                if keyframe_idx < keyframes - 1:
                    # Interpolate between keyframes
                    config1 = keyframe_configs[keyframe_idx]
                    config2 = keyframe_configs[keyframe_idx + 1]
                    
                    position = np.array(config1['position']) * (1 - alpha) + np.array(config2['position']) * alpha
                    look_at = np.array(config1['look_at']) * (1 - alpha) + np.array(config2['look_at']) * alpha
                    fov = config1['fov'] * (1 - alpha) + config2['fov'] * alpha
                    
                    frame_config = {
                        'frame_id': i,
                        'position': position.tolist(),
                        'look_at': look_at.tolist(),
                        'fov': fov
                    }
                else:
                    frame_config = keyframe_configs[-1].copy()
                    frame_config['frame_id'] = i
                    
                trajectory.append(frame_config)
                
        return trajectory
