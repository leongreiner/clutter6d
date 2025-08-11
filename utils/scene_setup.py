import blenderproc as bproc
import numpy as np
import os
from tqdm import tqdm

def setup_room_and_lighting():
    # Create room planes
    room_planes = [
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])
    ]
    
    # Enable physics for room planes
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)

    # Create light plane and material
    light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')

    # Create point light
    light_point = bproc.types.Light()
    light_point.set_energy(200)

    return room_planes, light_plane, light_plane_material, light_point

def load_textures(config):
    folder = config['dataset']['cc_textures_path']
    asset_names = os.listdir(folder)

    cc_textures = []
    for asset in tqdm(asset_names, desc="Loading CC textures", unit="texture", total=len(asset_names)):
        mats = bproc.loader.load_ccmaterials(
            folder_path=folder,
            used_assets=[asset]
        )
        cc_textures.extend(mats)
    
    return cc_textures

def randomize_scene_lighting(light_plane_material, light_point, room_planes, cc_textures):
    # Randomize light plane emission
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3, 6),
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0])
    )
    
    # Randomize point light
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    location = bproc.sampler.shell(
        center=[0, 0, 0], 
        radius_min=1, 
        radius_max=1.5,
        elevation_min=5, 
        elevation_max=89
    )
    light_point.set_location(location)

    # Apply random texture to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)
