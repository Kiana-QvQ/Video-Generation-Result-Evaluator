"""Create a small product-style robot scene for testing the Codex bridge."""

from __future__ import annotations

import math
import os
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(os.environ.get("BLENDER_CODEX_OUTPUT_DIR", ROOT / "outputs" / "blender_codex_demo"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def material(name: str, color: tuple[float, float, float, float], metallic=0.0, roughness=0.45):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.metallic = metallic
    mat.roughness = roughness
    return mat


def assign_material(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def bevel(obj, width=0.08, segments=3):
    modifier = obj.modifiers.new(name="Soft edges", type="BEVEL")
    modifier.width = width
    modifier.segments = segments


def cube(name, location, scale, mat, bevel_width=0.08):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign_material(obj, mat)
    if bevel_width:
        bevel(obj, bevel_width)
    return obj


def cylinder(name, location, radius, depth, mat, vertices=32, rotation=None):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=location,
        rotation=rotation or (0.0, 0.0, 0.0),
    )
    obj = bpy.context.object
    obj.name = name
    assign_material(obj, mat)
    bevel(obj, min(radius * 0.15, 0.06), 2)
    return obj


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def build_scene():
    clear_scene()

    dark = material("Graphite", (0.025, 0.04, 0.06, 1.0), metallic=0.8, roughness=0.25)
    blue = material("Neon Blue", (0.02, 0.28, 0.95, 1.0), metallic=0.25, roughness=0.22)
    orange = material("Signal Orange", (1.0, 0.18, 0.025, 1.0), metallic=0.2, roughness=0.3)
    white = material("Ceramic White", (0.75, 0.85, 0.95, 1.0), metallic=0.1, roughness=0.28)
    floor_mat = material("Floor", (0.012, 0.018, 0.028, 1.0), metallic=0.15, roughness=0.32)

    cube("Floor", (0.0, 0.0, -0.2), (6.0, 6.0, 0.2), floor_mat, 0.03)
    cube("Display plinth", (0.0, 0.0, 0.25), (2.35, 2.0, 0.25), dark, 0.12)
    cube("Display top", (0.0, 0.0, 0.58), (2.1, 1.75, 0.08), blue, 0.04)

    for x in (-2.0, 2.0):
        for y in (-1.65, 1.65):
            cylinder("Plinth corner", (x, y, 0.72), 0.12, 0.28, orange, vertices=24)

    cube("Robot torso", (0.0, 0.0, 1.7), (0.82, 0.58, 0.75), white, 0.14)
    cube("Robot chest", (0.0, -0.61, 1.75), (0.46, 0.05, 0.32), blue, 0.035)
    cube("Robot head", (0.0, 0.0, 2.85), (0.62, 0.52, 0.48), dark, 0.16)
    cube("Robot face", (0.0, -0.53, 2.82), (0.43, 0.04, 0.23), blue, 0.035)
    for x in (-0.22, 0.22):
        cylinder("Eye", (x, -0.59, 2.86), 0.075, 0.035, white, vertices=20, rotation=(math.pi / 2, 0.0, 0.0))

    cylinder("Antenna", (0.0, 0.0, 3.55), 0.035, 0.45, orange, vertices=16)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.11, location=(0.0, 0.0, 3.79))
    assign_material(bpy.context.object, orange)

    for side in (-1, 1):
        x = 1.05 * side
        arm = cube("Robot arm", (x, 0.0, 1.62), (0.18, 0.34, 0.65), dark, 0.08)
        arm.rotation_euler[1] = math.radians(-12 * side)
        cylinder("Robot hand", (1.08 * side, -0.02, 0.9), 0.22, 0.24, orange, vertices=24, rotation=(0.0, math.pi / 2, 0.0))

    for side in (-1, 1):
        leg_x = 0.42 * side
        cube("Robot leg", (leg_x, 0.0, 0.72), (0.27, 0.35, 0.42), dark, 0.08)
        cube("Robot foot", (leg_x, -0.18, 0.2), (0.34, 0.5, 0.16), white, 0.06)

    bpy.ops.object.light_add(type="AREA", location=(3.8, -4.0, 6.5))
    key = bpy.context.object
    key.name = "Key light"
    key.data.energy = 1100
    key.data.shape = "DISK"
    key.data.size = 4.0
    look_at(key, (0.0, 0.0, 1.4))

    bpy.ops.object.light_add(type="AREA", location=(-4.0, -1.0, 3.0))
    fill = bpy.context.object
    fill.name = "Blue fill"
    fill.data.energy = 800
    fill.data.color = (0.04, 0.2, 1.0)
    fill.data.size = 3.0
    look_at(fill, (0.0, 0.0, 1.4))

    bpy.ops.object.camera_add(location=(6.1, -7.6, 4.8))
    camera = bpy.context.object
    camera.name = "Presentation camera"
    camera.data.lens = 52
    look_at(camera, (0.0, 0.0, 1.65))
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 900
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.004, 0.006, 0.012)

    scene.render.filepath = str(OUTPUT_DIR / "demo_robot.png")
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "demo_robot.blend"))
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    build_scene()
