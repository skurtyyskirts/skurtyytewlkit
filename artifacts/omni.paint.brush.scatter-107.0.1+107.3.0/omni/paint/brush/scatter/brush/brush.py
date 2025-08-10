import omni.kit.app

BRUSH_TYPE = "Scatter"
BRUSH_FORMAT_VERSION = "104.1.0"


class INSTANCING:
    POINT = "Point Instancing"
    ASSET = "Instanced Asset"
    NONE = "None"


INSTANCING_TYPES = [INSTANCING.POINT, INSTANCING.ASSET, INSTANCING.NONE]


def get_library_path(file):
    path = omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__)
    return f"{path}/assets/{file}"


def fill_default_brush(brush):
    brush["file_version"] = BRUSH_FORMAT_VERSION
    brush["type"] = BRUSH_TYPE
    brush["size"] = 250
    brush["size_ui"] = {
        "key": "size",
        "type": "float",
        "label": "Size",
        "suffix": "cm",
        "tooltip": "Brush Size (Radius)",
        "step": 0.1,
        "min": 1.0,
        "read_only": False,
    }
    brush["falloff"] = 1
    brush["falloff_ui"] = {"key": "falloff", "type": "falloff", "label": "Falloff", "read_only": False}
    brush["density"] = 30
    brush["density_ui"] = {
        "key": "density",
        "type": "float",
        "label": "Density (per Stamp)",
        "tooltip": "Max painted assets in a stamp",
        "read_only": False,
        "step": 0.1,
        "min": 0.0,
    }
    brush["stamp_spacing"] = 0.5
    brush["stamp_spacing_ui"] = {
        "key": "stamp_spacing",
        "type": "float",
        "label": "Stamp Interval",
        "tooltip": "Interval to stamp by in brush distance travelled (cm)",
        "step": 0.01,
        "min": 0.01,
        "max": 1.0,
        "read_only": False,
    }
    brush["object_padding_enabled"] = True
    brush["padding_enabled_ui"] = {
        "key": "object_padding_enabled",
        "type": "bool",
        "label": "Object Padding",
        "read_only": False,
    }
    brush["object_padding"] = -5
    brush["padding_ui"] = {
        "key": "object_padding",
        "type": "float",
        "label": "Padding Size",
        "suffix": "cm",
        "tooltip": "Spacing between assets",
        "step": 0.1,
        "read_only": False,
    }
    brush["vertical_offset"] = 0
    brush["vertical_ui"] = {
        "key": "vertical_offset",
        "type": "float",
        "label": "Vertical Offset",
        "suffix": "cm",
        "tooltip": "Offsets placed assets on the mesh surface",
        "step": 0.1,
        "read_only": False,
    }
    brush["conform_to_surface"] = True
    brush["conform_ui"] = {
        "key": "conform_to_surface",
        "type": "bool",
        "label": "Conform to Surface",
        "tooltip": "Painted assets will conform to mesh surface",
        "read_only": False,
    }
    brush["physics"] = False
    brush["physics_ui"] = {
        "key": "physics",
        "type": "bool",
        "label": "Physics",
        "tooltip": "Painted assets will have Physics Colliders",
        "read_only": False,
    }
    brush["gravity"] = False
    brush["gravity_ui"] = {
        "key": "gravity",
        "type": "bool",
        "label": "Gravity",
        "tooltip": "Painted assets will drop as if gravity is being applied to them",
        "read_only": False,
    }
    brush["instancing"] = INSTANCING.POINT
    brush["instancing_ui"] = {
        "key": "instancing",
        "type": "combo",
        "label": "Instancing",
        "tooltip": "Places assets with selected type",
        "options": ["Point Instancing", "Instanced Asset", "None"],
        "read_only": False,
    }
    brush["align_to_stroke"] = 0.0
    brush["stroke_ui"] = {
        "key": "align_to_stroke",
        "type": "float",
        "label": "Align To Stroke (Ratio)",
        "tooltip": "Aligns the rotation of the placed assets to the stroke",
        "step": 0.01,
        "min": 0.0,
        "max": 1.0,
        "read_only": False,
    }
    brush["rotation"] = {"min": -175, "max": 175}
    brush["rotation_ui"] = {
        "key": "rotation",
        "type": "range",
        "label": "Rotation",
        "tooltip": "Random rotation range of painted assets",
        "step": 0.01,
        "read_only": False,
    }
    brush["scale"] = {"enabled": True, "bias": 0.5, "min": 0.9, "max": 1.1, "weight": 0.2}
    brush["assets"] = [{"path": get_library_path("cube/cube.usd"), "thumbnail": "", "weight": 1.0, "enabled": True}]
    brush["floodable"] = True
    brush["lock_selection"] = True
    brush["erase_enabled"] = True


def update_brush_version(brush):
    if "file_version" not in brush:
        fill_default_brush(brush)

    origin_version = brush["file_version"]
    if origin_version >= BRUSH_FORMAT_VERSION:
        return

    brush["file_version"] = BRUSH_FORMAT_VERSION

    # backward compat for renamed attributes across versions
    if "random_rotation" in brush:
        brush["rotation"] = brush["random_rotation"]
        brush.pop("random_rotation", None)
    if "scale_distribution" in brush:
        brush["scale"] = brush["scale_distribution"]
        brush.pop("scale_distribution", None)
    if "align_to_stroke_direction" in brush:
        brush["align_to_stroke"] = brush["align_to_stroke_direction"]
        brush.pop("align_to_stroke_direction", None)

    if origin_version < "103.0.1" and "bias" in brush["scale"]:
        len = brush["scale"]["max"] - brush["scale"]["min"]
        brush["scale"]["bias"] = (brush["scale"]["bias"] - brush["scale"]["min"]) / len

    # set default value for old brush file
    default_brush = {}
    fill_default_brush(default_brush)
    for k, v in default_brush.items():
        if k not in brush:
            brush[k] = v
