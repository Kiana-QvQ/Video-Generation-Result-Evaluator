"""Create a lightweight, externally unchanged version with hidden W ribs.

The outer dimensions and visible tube dimensions stay unchanged:
  base 200 x 160 x 5 mm
  tube total length 160 mm, OD 90 mm, main ID 80 mm, mouth ID 70 mm

The base becomes a closed thin-wall cavity with several internal W ribs.
The tube wall keeps the same nominal 5 mm envelope but is represented by
outer/inner skins plus internal radial webs, reducing printable solid volume.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import bpy
import bmesh


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(
    os.environ.get(
        "BLENDER_CODEX_OUTPUT_DIR",
        ROOT / "outputs" / "blender_codex_w_support",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_SCRIPT = Path(__file__).with_name("create_white_model.py")
SPEC = importlib.util.spec_from_file_location("horizontal_white_model", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load base model script: {BASE_SCRIPT}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
BASE.OUTPUT_DIR = OUTPUT_DIR


BASE_TOP_SKIN = 1.0
BASE_BOTTOM_SKIN = 1.0
BASE_RIM = 5.0
BASE_CAVITY_HEIGHT = BASE.BASE_THICKNESS - BASE_TOP_SKIN - BASE_BOTTOM_SKIN
W_RIB_HEIGHT = BASE_CAVITY_HEIGHT
W_RIB_THICKNESS = 2.0
W_RIB_DEPTH = 3.0
W_RIB_OFFSETS = (-45.0, 0.0, 45.0)

TUBE_SKIN = 1.0
TUBE_WEB_COUNT = 8
TUBE_WEB_THICKNESS = 1.2
TUBE_WEB_LENGTH = BASE.MAIN_CAVITY_LENGTH


def add_box(name, location, dimensions):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def add_hollow_base_shell():
    parts = []
    z_bottom = BASE_BOTTOM_SKIN / 2.0
    z_top = BASE.BASE_THICKNESS - BASE_TOP_SKIN / 2.0
    z_mid = BASE_BOTTOM_SKIN + BASE_CAVITY_HEIGHT / 2.0

    parts.append(
        add_box(
            "Base bottom skin",
            (0.0, 0.0, z_bottom),
            (BASE.BASE_LENGTH, BASE.BASE_WIDTH, BASE_BOTTOM_SKIN),
        )
    )
    parts.append(
        add_box(
            "Base top skin",
            (0.0, 0.0, z_top),
            (BASE.BASE_LENGTH, BASE.BASE_WIDTH, BASE_TOP_SKIN),
        )
    )

    inner_length = BASE.BASE_LENGTH - BASE_RIM * 2.0
    inner_width = BASE.BASE_WIDTH - BASE_RIM * 2.0
    parts.extend(
        [
            add_box(
                "Base left rim",
                (-(BASE.BASE_LENGTH - BASE_RIM) / 2.0, 0.0, z_mid),
                (BASE_RIM, inner_width, BASE_CAVITY_HEIGHT),
            ),
            add_box(
                "Base right rim",
                ((BASE.BASE_LENGTH - BASE_RIM) / 2.0, 0.0, z_mid),
                (BASE_RIM, inner_width, BASE_CAVITY_HEIGHT),
            ),
            add_box(
                "Base front rim",
                (0.0, -(BASE.BASE_WIDTH - BASE_RIM) / 2.0, z_mid),
                (inner_length, BASE_RIM, BASE_CAVITY_HEIGHT),
            ),
            add_box(
                "Base back rim",
                (0.0, (BASE.BASE_WIDTH - BASE_RIM) / 2.0, z_mid),
                (inner_length, BASE_RIM, BASE_CAVITY_HEIGHT),
            ),
        ]
    )
    return parts


def add_w_segment(y_offset, start, end, index):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0 + y_offset)
    obj = add_box(
        f"Internal W rib {index}",
        (center[0], center[1], BASE_BOTTOM_SKIN + W_RIB_HEIGHT / 2.0),
        (length, W_RIB_THICKNESS, W_RIB_HEIGHT),
    )
    obj.rotation_euler[2] = angle
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    return obj


def add_internal_w_ribs():
    parts = []
    points = (
        (-BASE.BASE_LENGTH / 2.0 + BASE_RIM + 3.0, -18.0),
        (-45.0, 18.0),
        (0.0, -18.0),
        (45.0, 18.0),
        (BASE.BASE_LENGTH / 2.0 - BASE_RIM - 3.0, -18.0),
    )
    index = 1
    for offset in W_RIB_OFFSETS:
        for start, end in zip(points, points[1:]):
            parts.append(add_w_segment(offset, start, end, index))
            index += 1
    return parts


def ring(vertices, radius, axis_position, center_z):
    result = []
    for index in range(BASE.SEGMENTS):
        angle = 2.0 * math.pi * index / BASE.SEGMENTS
        result.append(
            vertices.new(
                (
                    radius * math.cos(angle),
                    axis_position,
                    center_z + radius * math.sin(angle),
                )
            )
        )
    return result


def quad_strip(faces, lower, upper, reverse=False):
    for index in range(BASE.SEGMENTS):
        next_index = (index + 1) % BASE.SEGMENTS
        values = (
            (lower[index], upper[index], upper[next_index], lower[next_index])
            if reverse
            else (lower[index], lower[next_index], upper[next_index], upper[index])
        )
        faces.new(values)


def annulus(faces, outer, inner):
    for index in range(BASE.SEGMENTS):
        next_index = (index + 1) % BASE.SEGMENTS
        faces.new(
            (
                outer[index],
                outer[next_index],
                inner[next_index],
                inner[index],
            )
        )


def disk(faces, bmesh_module, ring_vertices, axis_position, positive_axis):
    center = bmesh_module.verts.new((0.0, axis_position, BASE.TUBE_CENTER_Z))
    for index in range(BASE.SEGMENTS):
        next_index = (index + 1) % BASE.SEGMENTS
        values = (
            (center, ring_vertices[index], ring_vertices[next_index])
            if positive_axis
            else (center, ring_vertices[next_index], ring_vertices[index])
        )
        faces.new(values)


def add_lightweight_stepped_tube():
    center_z = BASE.TUBE_CENTER_Z
    outer_main = BASE.MAIN_OUTER_RADIUS
    main_inner = BASE.MAIN_INNER_RADIUS
    mouth_outer = BASE.MOUTH_OUTER_RADIUS
    mouth_inner = BASE.MOUTH_RADIUS

    outer_main_inner = outer_main - TUBE_SKIN
    main_inner_outer = main_inner + TUBE_SKIN
    mouth_outer_inner = mouth_outer - TUBE_SKIN
    mouth_inner_outer = mouth_inner + TUBE_SKIN

    bm = bmesh.new()
    vertices = bm.verts
    faces = bm.faces

    # Main outer skin, main inner skin, and reduced mouth skins.
    oo_front = ring(vertices, outer_main, BASE.TUBE_FRONT, center_z)
    oo_shoulder = ring(vertices, outer_main, BASE.MOUTH_START_AXIS, center_z)
    oi_front = ring(vertices, outer_main_inner, BASE.TUBE_FRONT, center_z)
    oi_shoulder = ring(vertices, outer_main_inner, BASE.MOUTH_START_AXIS, center_z)

    io_front = ring(vertices, main_inner_outer, BASE.CAP_INNER_AXIS, center_z)
    io_shoulder = ring(vertices, main_inner_outer, BASE.MOUTH_START_AXIS, center_z)
    ii_front = ring(vertices, main_inner, BASE.CAP_INNER_AXIS, center_z)
    ii_shoulder = ring(vertices, main_inner, BASE.MOUTH_START_AXIS, center_z)

    mo_shoulder = ring(vertices, mouth_outer, BASE.MOUTH_START_AXIS, center_z)
    mo_back = ring(vertices, mouth_outer, BASE.TUBE_BACK, center_z)
    mi_shoulder = ring(vertices, mouth_outer_inner, BASE.MOUTH_START_AXIS, center_z)
    mi_back = ring(vertices, mouth_outer_inner, BASE.TUBE_BACK, center_z)

    mii_shoulder = ring(vertices, mouth_inner_outer, BASE.MOUTH_START_AXIS, center_z)
    mii_back = ring(vertices, mouth_inner_outer, BASE.TUBE_BACK, center_z)
    miii_shoulder = ring(vertices, mouth_inner, BASE.MOUTH_START_AXIS, center_z)
    miii_back = ring(vertices, mouth_inner, BASE.TUBE_BACK, center_z)

    quad_strip(faces, oo_front, oo_shoulder)
    quad_strip(faces, oi_front, oi_shoulder, reverse=True)
    annulus(faces, oo_front, oi_front)
    annulus(faces, oo_shoulder, oi_shoulder)

    quad_strip(faces, io_front, io_shoulder)
    quad_strip(faces, ii_front, ii_shoulder, reverse=True)
    annulus(faces, io_front, ii_front)
    annulus(faces, io_shoulder, ii_shoulder)

    quad_strip(faces, mo_shoulder, mo_back)
    quad_strip(faces, mi_shoulder, mi_back, reverse=True)
    annulus(faces, mo_shoulder, mi_shoulder)
    quad_strip(faces, mii_shoulder, mii_back)
    quad_strip(faces, miii_shoulder, miii_back, reverse=True)
    annulus(faces, mii_shoulder, miii_shoulder)

    # Keep a real 5 mm closed cap and the visible stepped shoulder/rim.
    disk(faces, bm, oo_front, BASE.TUBE_FRONT, positive_axis=False)
    disk(faces, bm, ii_front, BASE.CAP_INNER_AXIS, positive_axis=True)
    annulus(faces, oo_shoulder, mo_shoulder)
    annulus(faces, ii_shoulder, miii_shoulder)
    annulus(faces, mo_back, miii_back)

    mesh = bpy.data.meshes.new("Lightweight double-skin tube mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    tube = bpy.data.objects.new("Lightweight horizontal tube", mesh)
    bpy.context.collection.objects.link(tube)
    return tube


def add_tube_webs():
    parts = []
    outer_web_radius = BASE.MAIN_OUTER_RADIUS - TUBE_SKIN
    inner_web_radius = BASE.MAIN_INNER_RADIUS + TUBE_SKIN
    radius = (inner_web_radius + outer_web_radius) / 2.0
    radial_length = outer_web_radius - inner_web_radius + 0.2
    web_length = BASE.MAIN_CAVITY_LENGTH
    for index in range(TUBE_WEB_COUNT):
        angle = 2.0 * math.pi * index / TUBE_WEB_COUNT
        x = radius * math.cos(angle)
        z = BASE.TUBE_CENTER_Z + radius * math.sin(angle)
        bpy.ops.mesh.primitive_cube_add(
            location=(x, 0.0, z),
        )
        web = bpy.context.object
        web.name = f"Tube internal web {index + 1}"
        web.dimensions = (
            radial_length,
            web_length,
            TUBE_WEB_THICKNESS,
        )
        web.rotation_euler[1] = -angle
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        parts.append(web)
    return parts


def join_all(parts):
    bpy.ops.object.select_all(action="DESELECT")
    active = parts[0]
    bpy.context.view_layer.objects.active = active
    for part in parts:
        part.select_set(True)
    bpy.ops.object.join()
    return active


def estimate_volume_cm3():
    base_shell = (
        BASE.BASE_LENGTH * BASE.BASE_WIDTH * (BASE_TOP_SKIN + BASE_BOTTOM_SKIN)
        + 2.0 * BASE_RIM * (BASE.BASE_WIDTH - 2.0 * BASE_RIM) * BASE_CAVITY_HEIGHT
        + 2.0 * (BASE.BASE_LENGTH - 2.0 * BASE_RIM) * BASE_RIM * BASE_CAVITY_HEIGHT
    )
    tube_skin = (
        math.pi
        * (
            (BASE.MAIN_OUTER_RADIUS**2 - (BASE.MAIN_OUTER_RADIUS - TUBE_SKIN) ** 2)
            * (BASE.TUBE_LENGTH - BASE.MOUTH_LENGTH)
            + ((BASE.MAIN_INNER_RADIUS + TUBE_SKIN) ** 2 - BASE.MAIN_INNER_RADIUS**2)
            * BASE.MAIN_CAVITY_LENGTH
            + (BASE.MOUTH_OUTER_RADIUS**2 - (BASE.MOUTH_OUTER_RADIUS - TUBE_SKIN) ** 2)
            * BASE.MOUTH_LENGTH
            + ((BASE.MOUTH_RADIUS + TUBE_SKIN) ** 2 - BASE.MOUTH_RADIUS**2)
            * BASE.MOUTH_LENGTH
        )
    )
    cap = math.pi * BASE.MAIN_OUTER_RADIUS**2 * BASE.CAP_THICKNESS
    w_ribs = len(W_RIB_OFFSETS) * 4 * 230.0 * W_RIB_THICKNESS * W_RIB_HEIGHT / 1000.0
    tube_webs = (
        TUBE_WEB_COUNT
        * TUBE_WEB_THICKNESS
        * TUBE_WEB_THICKNESS
        * TUBE_WEB_LENGTH
        / 1000.0
    )
    return (base_shell + tube_skin + cap + w_ribs + tube_webs) / 1000.0


def add_print_edge_bevel(model):
    bevel = model.modifiers.new("Print edge chamfers", type="BEVEL")
    bevel.width = 0.8
    bevel.segments = 2
    bevel.limit_method = "ANGLE"
    bevel.angle_limit = math.radians(38.0)
    bevel.harden_normals = True
    model["edge_bevel_mm"] = 0.8


def main():
    BASE.clear_scene()
    parts = add_hollow_base_shell()
    parts.extend(add_internal_w_ribs())
    parts.append(add_lightweight_stepped_tube())
    parts.extend(add_tube_webs())

    model = join_all(parts)
    model.name = "White model - hidden W internal support"
    BASE.add_model_properties(model)
    model["lightweight_design"] = True
    model["base_internal_structure"] = "3 rows of W ribs"
    model["tube_internal_structure"] = f"{TUBE_WEB_COUNT} radial webs"
    model["estimated_dense_volume_cm3"] = round(estimate_volume_cm3(), 1)
    model["target_print_mass_g"] = 200

    add_print_edge_bevel(model)
    BASE.verify_opening(model)
    BASE.add_preview_setup(model)

    scene = bpy.context.scene
    scene.camera = bpy.data.objects["White model preview camera"]
    scene.render.filepath = str(OUTPUT_DIR / "w_support_preview.png")
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "w_support_model.blend"))
    bpy.ops.render.render(write_still=True)

    scene.camera = bpy.data.objects["Mouth inspection camera"]
    scene.render.filepath = str(OUTPUT_DIR / "w_support_mouth.png")
    bpy.ops.render.render(write_still=True)
    scene.camera = bpy.data.objects["White model preview camera"]
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "w_support_model.blend"))

    print("Estimated dense model volume (cm3):", round(estimate_volume_cm3(), 1))
    print("W-support model written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
