"""Create the confirmed horizontal stepped-bore white model.

Blender units are millimetres.

Assembly:
  - base plate: 200 x 150 x 5 mm
  - tube axis: horizontal, parallel to the 160 mm base direction
  - tube total length: 160 mm
  - closed cap: 5 mm
  - main cavity: ID 80 mm x 150 mm
  - mouth: ID 70 mm x 5 mm
  - main tube OD: 90 mm
  - every specified thickness: 5 mm
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(
    os.environ.get("BLENDER_CODEX_OUTPUT_DIR", ROOT / "outputs" / "blender_codex_white_model")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


BASE_LENGTH = 200.0
BASE_WIDTH = 160.0
BASE_THICKNESS = 5.0

TUBE_LENGTH = 160.0
CAP_THICKNESS = 5.0
MAIN_CAVITY_LENGTH = 150.0
MOUTH_LENGTH = 5.0
THICKNESS = 5.0

MAIN_OUTER_DIAMETER = 90.0
MAIN_INNER_DIAMETER = 80.0
MOUTH_DIAMETER = 70.0
MOUTH_OUTER_DIAMETER = MOUTH_DIAMETER + THICKNESS * 2.0

MAIN_OUTER_RADIUS = MAIN_OUTER_DIAMETER / 2.0
MAIN_INNER_RADIUS = MAIN_INNER_DIAMETER / 2.0
MOUTH_RADIUS = MOUTH_DIAMETER / 2.0
MOUTH_OUTER_RADIUS = MOUTH_OUTER_DIAMETER / 2.0

SEGMENTS = 128
BASE_OVERLAP = 0.5

# The tube is centered on and aligned to the 160 mm base direction.
TUBE_FRONT = -TUBE_LENGTH / 2.0
TUBE_BACK = TUBE_LENGTH / 2.0
CAP_INNER_AXIS = TUBE_FRONT + CAP_THICKNESS
MOUTH_START_AXIS = TUBE_BACK - MOUTH_LENGTH
TUBE_CENTER_Z = BASE_THICKNESS + MAIN_OUTER_RADIUS - BASE_OVERLAP


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(collection):
            if datablock.users == 0:
                collection.remove(datablock)


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def make_material(name, color, roughness):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)
    material.roughness = roughness
    material.metallic = 0.0
    return material


def add_base_plate():
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, BASE_THICKNESS / 2.0))
    base = bpy.context.object
    base.name = "Base plate 200x150x5mm"
    base.dimensions = (BASE_LENGTH, BASE_WIDTH, BASE_THICKNESS)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return base


def add_ring(vertices, radius, axis_pos):
    ring = []
    for index in range(SEGMENTS):
        angle = 2.0 * math.pi * index / SEGMENTS
        ring.append(
            vertices.new(
                (
                    radius * math.cos(angle),
                    axis_pos,
                    TUBE_CENTER_Z + radius * math.sin(angle),
                )
            )
        )
    return ring


def add_quad_strip(faces, lower, upper, reverse=False):
    for index in range(SEGMENTS):
        next_index = (index + 1) % SEGMENTS
        if reverse:
            faces.new(
                (
                    lower[index],
                    upper[index],
                    upper[next_index],
                    lower[next_index],
                )
            )
        else:
            faces.new(
                (
                    lower[index],
                    lower[next_index],
                    upper[next_index],
                    upper[index],
                )
            )


def add_disk(faces, bmesh_module, ring, axis_pos, positive_axis):
    center = bmesh_module.verts.new((0.0, axis_pos, TUBE_CENTER_Z))
    for index in range(SEGMENTS):
        next_index = (index + 1) % SEGMENTS
        if positive_axis:
            faces.new((center, ring[index], ring[next_index]))
        else:
            faces.new((center, ring[next_index], ring[index]))


def add_annulus(faces, outer_ring, inner_ring, positive_axis):
    for index in range(SEGMENTS):
        next_index = (index + 1) % SEGMENTS
        if positive_axis:
            faces.new(
                (
                    outer_ring[index],
                    outer_ring[next_index],
                    inner_ring[next_index],
                    inner_ring[index],
                )
            )
        else:
            faces.new(
                (
                    outer_ring[index],
                    inner_ring[index],
                    inner_ring[next_index],
                    outer_ring[next_index],
                )
            )


def add_stepped_tube():
    bm = bmesh.new()
    vertices = bm.verts
    faces = bm.faces

    outer_front = add_ring(vertices, MAIN_OUTER_RADIUS, TUBE_FRONT)
    outer_shoulder = add_ring(vertices, MAIN_OUTER_RADIUS, MOUTH_START_AXIS)
    outer_back = add_ring(vertices, MOUTH_OUTER_RADIUS, TUBE_BACK)

    inner_front = add_ring(vertices, MAIN_INNER_RADIUS, CAP_INNER_AXIS)
    inner_shoulder = add_ring(vertices, MAIN_INNER_RADIUS, MOUTH_START_AXIS)
    inner_back = add_ring(vertices, MOUTH_RADIUS, TUBE_BACK)

    # Outer wall, stepped down to the smaller mouth section.
    add_quad_strip(faces, outer_front, outer_shoulder)
    add_quad_strip(faces, outer_shoulder, outer_back)

    # Inner wall: ID80 main cavity, then ID70 mouth.
    add_quad_strip(faces, inner_front, inner_shoulder, reverse=True)
    add_quad_strip(faces, inner_shoulder, inner_back, reverse=True)

    # Closed end, 5 mm cap, shoulder, and open end rim.
    add_disk(faces, bm, outer_front, TUBE_FRONT, positive_axis=False)
    add_disk(faces, bm, inner_front, CAP_INNER_AXIS, positive_axis=True)
    add_annulus(faces, outer_shoulder, inner_shoulder, positive_axis=True)
    add_annulus(faces, outer_back, inner_back, positive_axis=True)

    mesh = bpy.data.meshes.new("Horizontal stepped hollow tube mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    tube = bpy.data.objects.new("Horizontal tube OD90 ID80 mouthID70", mesh)
    bpy.context.collection.objects.link(tube)
    return tube


def join_components(base, tube):
    # Keep the explicit tube topology. The tube overlaps the plate by 0.5 mm
    # and both parts become one Blender mesh object without a destructive
    # boolean operation on the stepped opening.
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = base
    base.select_set(True)
    tube.select_set(True)
    bpy.ops.object.join()


def add_model_properties(model):
    model["base_length_mm"] = BASE_LENGTH
    model["base_width_mm"] = BASE_WIDTH
    model["base_thickness_mm"] = BASE_THICKNESS
    model["tube_total_length_mm"] = TUBE_LENGTH
    model["main_outer_diameter_mm"] = MAIN_OUTER_DIAMETER
    model["main_inner_diameter_mm"] = MAIN_INNER_DIAMETER
    model["mouth_diameter_mm"] = MOUTH_DIAMETER
    model["mouth_outer_diameter_mm"] = MOUTH_OUTER_DIAMETER
    model["main_cavity_length_mm"] = MAIN_CAVITY_LENGTH
    model["mouth_length_mm"] = MOUTH_LENGTH
    model["cap_thickness_mm"] = CAP_THICKNESS
    model["all_wall_thickness_mm"] = THICKNESS
    model["tube_axis"] = "Y, horizontal and parallel to the 160 mm base direction"


def verify_opening(model):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = model.evaluated_get(depsgraph)
    bvh = BVHTree.FromObject(evaluated, depsgraph)
    hit, _normal, _index, _distance = bvh.ray_cast(
        Vector((0.0, 300.0, TUBE_CENTER_Z)),
        Vector((0.0, -1.0, 0.0)),
    )
    if hit is None:
        raise RuntimeError("Opening verification ray did not hit the model")
    world_hit = model.matrix_world @ hit
    first_hit_y = float(world_hit.y)
    print(f"Opening center ray first hit Y={first_hit_y:.2f} mm")
    if first_hit_y > MOUTH_START_AXIS - 2.0:
        raise RuntimeError(
            "Opening verification failed: the center ray hit the mouth wall "
            f"at Y={first_hit_y:.2f} mm"
        )


def add_preview_setup(model):
    model.data.materials.append(make_material("White clay", (0.9, 0.9, 0.9), 0.7))
    interior = make_material("Interior shadow", (0.28, 0.28, 0.28), 0.85)
    model.data.materials.append(interior)
    for polygon in model.data.polygons:
        center = polygon.center
        radial_distance = math.hypot(center.x, center.z - TUBE_CENTER_Z)
        if (
            CAP_INNER_AXIS <= center.y <= TUBE_BACK + 1.0
            and radial_distance < MAIN_INNER_RADIUS + 0.2
        ):
            polygon.material_index = 1

    bpy.ops.mesh.primitive_plane_add(size=900.0, location=(0.0, 0.0, -0.05))
    floor = bpy.context.object
    floor.name = "Preview floor"
    floor.data.materials.append(make_material("Preview floor", (0.12, 0.12, 0.12), 0.9))

    lights = (
        ("Key light", (140.0, 220.0, 260.0), 2600.0, 200.0),
        ("Fill light", (-120.0, -100.0, 180.0), 1500.0, 180.0),
        ("Mouth light", (100.0, 260.0, 90.0), 1800.0, 120.0),
    )
    for name, location, energy, size in lights:
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.name = name
        light.data.energy = energy
        light.data.size = size
        look_at(light, (0.0, 0.0, 35.0))

    bpy.ops.object.camera_add(location=(260.0, 300.0, 190.0))
    preview_camera = bpy.context.object
    preview_camera.name = "White model preview camera"
    preview_camera.data.lens = 58.0
    look_at(preview_camera, (0.0, 0.0, 38.0))

    bpy.ops.object.camera_add(location=(130.0, 330.0, 90.0))
    mouth_camera = bpy.context.object
    mouth_camera.name = "Mouth inspection camera"
    mouth_camera.data.lens = 68.0
    look_at(mouth_camera, (0.0, 45.0, TUBE_CENTER_Z))

    scene = bpy.context.scene
    scene.camera = preview_camera
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 1100
    scene.render.resolution_y = 850
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.07, 0.07, 0.07)
    scene.view_settings.exposure = 1.0
    scene["preview_camera"] = preview_camera.name
    scene["mouth_camera"] = mouth_camera.name


def main():
    clear_scene()
    base = add_base_plate()
    tube = add_stepped_tube()
    join_components(base, tube)

    base.name = "White model - one-piece horizontal stepped hollow tube"
    add_model_properties(base)
    verify_opening(base)
    add_preview_setup(base)

    scene = bpy.context.scene
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "white_model.blend"))

    scene.camera = bpy.data.objects["White model preview camera"]
    scene.render.filepath = str(OUTPUT_DIR / "white_model_preview.png")
    bpy.ops.render.render(write_still=True)

    scene.camera = bpy.data.objects["Mouth inspection camera"]
    scene.render.filepath = str(OUTPUT_DIR / "white_model_mouth.png")
    bpy.ops.render.render(write_still=True)

    scene.camera = bpy.data.objects["White model preview camera"]
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "white_model.blend"))
    print("White model written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
