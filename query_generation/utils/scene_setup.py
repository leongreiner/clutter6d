import blenderproc as bproc
import numpy as np
import os
import random
import gc
import bpy

def setup_room_and_lighting(room_size: float = 2):
    # Create room planes
    room_planes = [
        bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1]),
        bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1], location=[0, -room_size, room_size], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1], location=[0, room_size, room_size], rotation=[1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1], location=[room_size, 0, room_size], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1], location=[-room_size, 0, room_size], rotation=[0, 1.570796, 0])
    ]
    
    # Enable physics for room planes
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)

    # Create light plane and material
    light_plane = bproc.object.create_primitive('PLANE', scale=[room_size+1, room_size+1, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')

    # Create point light
    light_point = bproc.types.Light()
    light_point.set_energy(200)

    return room_planes, light_plane, light_plane_material, light_point

def load_texture(config):
    folder = config['dataset']['cc_textures_path']
    texture_names = os.listdir(folder)
    while True:
        selected_texture = random.choice(texture_names)
        mats = bproc.loader.load_ccmaterials(
            folder_path=folder,
            used_assets=[selected_texture]
        )
        if mats and len(mats) > 0:
            return mats[0], selected_texture
        else:
            print(f"Warning: No materials loaded from asset {selected_texture}, trying another...")

def randomize_scene_lighting(light_plane, light_plane_material, light_point, room_planes, cc_textures):
    # Randomize light plane emission
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(0.5, 6),
        emission_color=np.random.uniform([0.1, 0.1, 0.1, 1.0], [1.0, 1.0, 1.0, 1.0])
    )
    light_plane.replace_materials(light_plane_material)
    
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

def clean_up(scene_objects): 
    # Delete scene objects
    for obj in scene_objects:
        obj.delete()
    scene_objects.clear()
    
    # Clear up bproc
    bproc.utility.reset_keyframes()
    bproc.clean_up(clean_up_camera=True)
    
    # Clear materials and textures that are no longer referenced
    def count_ids():
        pools = (
            bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.textures,
            bpy.data.node_groups, bpy.data.lights, bpy.data.cameras, bpy.data.curves,
            bpy.data.armatures, bpy.data.collections, bpy.data.worlds
        )
        return sum(len(p) for p in pools)

    prev = count_ids()
    for _ in range(20):
        try:
            bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        except Exception:
            break
        gc.collect()
        curr = count_ids()
        if curr == prev:
            break
        prev = curr