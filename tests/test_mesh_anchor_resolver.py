import re
from pxr import Sdf, Usd, UsdGeom

from tools.custom.mesh_anchor_resolver import (
    resolve_or_create_anchor_for_hit,
    get_active_point_instancer,
)


def _make_stage_with_layers():
    stage = Usd.Stage.CreateInMemory()

    # Root World
    UsdGeom.Xform.Define(stage, Sdf.Path("/World"))

    # Create a capture layer and sublayer it
    capture = Sdf.Layer.CreateAnonymous("capture.usda")
    # Mark it as a capture by folder-like name in identifier path. We cannot move anon layers on FS,
    # but tests don't invoke is_layer_from_capture on anon capture.
    # We'll instead use it only as a read-only sublayer, not as edit target.

    # Create a mod layer and make it the edit target
    mod = Sdf.Layer.CreateAnonymous("mod.usda")
    root = stage.GetRootLayer()
    root.subLayerPaths.insert(0, capture.identifier)
    root.subLayerPaths.insert(0, mod.identifier)

    stage.SetEditTarget(Usd.EditTarget(mod))
    return stage, capture, mod


def test_resolve_or_create_anchor_for_hit_creates_scope_and_anchor_in_mod_layer():
    stage, _capture, mod = _make_stage_with_layers()

    # Simulate a hit on a captured mesh path
    hit = "/RootNode/meshes/mesh_0123456789ABCDEF/subpart"

    # First call creates anchors root and anchor
    anchor_path = resolve_or_create_anchor_for_hit(hit, stage=stage)
    assert str(anchor_path).startswith("/World/ScatterAnchors/")

    prim = stage.GetPrimAtPath(anchor_path)
    assert prim.IsValid()
    # Has tag
    assert prim.GetAttribute("remix_anchor").Get() is True
    assert prim.GetAttribute("remix_anchor:source_hit_path").Get() == hit

    # Ensure authored specs are in mod layer (edit target), not in capture
    assert mod.GetPrimAtPath(anchor_path) is not None

    # Root scope exists
    root_scope = stage.GetPrimAtPath("/World/ScatterAnchors")
    assert root_scope.IsValid()
    assert mod.GetPrimAtPath("/World/ScatterAnchors") is not None

    # Second call returns same anchor path and does not duplicate
    anchor_path_2 = resolve_or_create_anchor_for_hit(hit, stage=stage)
    assert anchor_path_2 == anchor_path


def test_compute_anchor_name_from_hash_patterns():
    stage, _capture, _mod = _make_stage_with_layers()

    # instance path style
    hit = "/RootNode/instances/inst_89ABCDEF01234567/any/child"
    anchor_path = resolve_or_create_anchor_for_hit(hit, stage=stage)
    assert re.match(r"^/World/ScatterAnchors/mesh_[A-Z0-9]{16}$", str(anchor_path))

    # mesh path style
    hit2 = "/RootNode/meshes/mesh_0011223344556677"
    anchor2 = resolve_or_create_anchor_for_hit(hit2, stage=stage)
    assert str(anchor2) == "/World/ScatterAnchors/mesh_0011223344556677"

    # fallback sanitization
    hit3 = "/Some/Unknown/Prim.Path@!"
    anchor3 = resolve_or_create_anchor_for_hit(hit3, stage=stage)
    assert str(anchor3).startswith("/World/ScatterAnchors/")


def test_get_active_point_instancer_creates_under_anchor():
    stage, _capture, mod = _make_stage_with_layers()

    hit = "/RootNode/meshes/mesh_AAAAAAAA00000000"
    anchor_path = resolve_or_create_anchor_for_hit(hit, stage=stage)

    pi_path = get_active_point_instancer(anchor_path, asset_set_key="rocks", stage=stage)
    prim = stage.GetPrimAtPath(pi_path)
    assert prim.IsValid()
    assert prim.IsA(UsdGeom.PointInstancer)

    # Creating again with same key returns same path
    pi_path2 = get_active_point_instancer(anchor_path, asset_set_key="rocks", stage=stage)
    assert pi_path2 == pi_path

    # Different key creates sibling PI
    pi_path3 = get_active_point_instancer(anchor_path, asset_set_key="plants", stage=stage)
    assert str(pi_path3) != str(pi_path)

    # Authored in mod layer
    assert mod.GetPrimAtPath(pi_path) is not None


def test_refuse_edit_on_capture_edit_target():
    stage, capture, _mod = _make_stage_with_layers()
    # Set capture as edit target to trigger refusal
    stage.SetEditTarget(Usd.EditTarget(capture))

    hit = "/RootNode/meshes/mesh_76543210ABCDEF12"
    try:
        resolve_or_create_anchor_for_hit(hit, stage=stage)
        assert False, "Expected RuntimeError for capture layer authoring"
    except RuntimeError as e:
        assert "Refusing to author into the capture layer" in str(e)