"""
PointInstancer authoring utilities for Scatter Brush.

APIs:
- ensure_point_instancer(anchor_path, asset_set_key, prototypes) -> Usd.Prim
- append_instances(pi_prim, instances: list[InstanceSpec]) -> None
- remove_instances_in_radius(pi_prim, center: Gf.Vec3d, radius: float) -> int

Notes:
- Maintains one instancer per (anchor, asset set)
- Manages prototypes list (dedup, append if missing)
- Batches updates under omni.kit.undo scopes
- Preserves optional ids for stable erase workflows
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import omni.kit.undo
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


@dataclass
class InstanceSpec:
    position: Gf.Vec3f | Gf.Vec3d
    # Optional: if provided and orientations are authored on the PI, we will append it;
    # otherwise identity is used for that slot
    orientation: Optional[Gf.Quath] = None
    # Non-uniform scale per instance; if None and scales are authored, use ones
    scale: Optional[Gf.Vec3f] = None
    # Prototype path (root prim of the prototype subgraph). If None, index 0 is used.
    prototype_path: Optional[Sdf.Path] = None
    # Optional client-provided id; if None we auto-assign monotonically increasing ids
    id: Optional[int] = None


def _to_sdf_path(p: Union[str, Sdf.Path, Usd.Prim]) -> Sdf.Path:
    if isinstance(p, Sdf.Path):
        return p
    if isinstance(p, Usd.Prim):
        return p.GetPath()
    if isinstance(p, str):
        return Sdf.Path(p)
    raise TypeError(f"Unsupported path type: {type(p)}")


def _get_stage() -> Usd.Stage:
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage in current context")
    return stage


def _ensure_edit_target_unmuted(stage: Usd.Stage) -> None:
    """If the current edit target layer is muted, unmute it so authoring can proceed."""
    edit_layer = stage.GetEditTarget().GetLayer()
    if edit_layer is None:
        return
    try:
        if stage.IsLayerMuted(edit_layer.identifier):
            stage.UnmuteLayer(edit_layer.identifier)
    except Exception:
        # Be tolerant if stage API not available in headless tests
        pass


def _ensure_default_prototype(stage: Usd.Stage, instancer_prim_path: Sdf.Path) -> Sdf.Path:
    """Ensure a built-in default prototype (a Cube) exists under the instancer's Prototypes scope.
    Returns the path to the prototype root prim.
    """
    protos_scope = instancer_prim_path.AppendPath("Prototypes")
    proto_root_path = protos_scope.AppendPath("DefaultCube")
    prim = stage.GetPrimAtPath(proto_root_path)
    if not prim.IsValid():
        # Create scope and cube prototype
        UsdGeom.Xform.Define(stage, protos_scope)
        UsdGeom.Cube.Define(stage, proto_root_path)
    return proto_root_path


def ensure_point_instancer(
    anchor_path: Union[str, Sdf.Path, Usd.Prim],
    asset_set_key: str,
    prototypes: Optional[Iterable[Union[str, Sdf.Path, Usd.Prim]]] = None,
) -> Usd.Prim:
    """Find or create a UsdGeomPointInstancer under anchor_path keyed by asset_set_key.

    - Creates the prim at f"{anchor}/PI_{asset_set_key}" if missing
    - Ensures the prototypes rel contains each provided prototype exactly once
    - If no prototypes provided, ensures a default cube prototype exists and is set
    """
    stage = _get_stage()
    anchor_sdf = _to_sdf_path(anchor_path)
    pi_path = anchor_sdf.AppendPath(f"PI_{asset_set_key}")

    _ensure_edit_target_unmuted(stage)

    pi = UsdGeom.PointInstancer.Get(stage, pi_path)
    if not pi:
        pi = UsdGeom.PointInstancer.Define(stage, pi_path)

    proto_rel = pi.GetPrototypesRel()
    current_targets: List[Sdf.Path] = list(proto_rel.GetTargets())

    targets: List[Sdf.Path] = []
    if prototypes:
        for p in prototypes:
            sp = _to_sdf_path(p)
            if sp not in targets:
                targets.append(sp)
    else:
        # Ensure a default prototype exists
        targets.append(_ensure_default_prototype(stage, pi.GetPath()))

    # Merge with existing without duplicates, preserving existing order first
    merged: List[Sdf.Path] = []
    for sp in current_targets + targets:
        if sp not in merged:
            merged.append(sp)

    if merged != current_targets:
        with omni.kit.undo.group():
            proto_rel.SetTargets(merged)

    return pi.GetPrim()


def _get_array(attr) -> List:
    v = attr.Get()
    if v is None:
        return []
    # Some USD arrays come back as tuple-like; cast to list for mutation
    return list(v)


def _get_or_create_attr_values(
    pi: UsdGeom.PointInstancer,
) -> Tuple[List[Gf.Vec3f], List[int], Optional[List[Gf.Quath]], Optional[List[Gf.Vec3f]], Optional[List[int]]]:
    positions_attr = pi.GetPositionsAttr()
    if not positions_attr:
        positions_attr = pi.CreatePositionsAttr()
    proto_indices_attr = pi.GetProtoIndicesAttr()
    if not proto_indices_attr:
        proto_indices_attr = pi.CreateProtoIndicesAttr()

    orientations_attr = pi.GetOrientationsAttr()
    scales_attr = pi.GetScalesAttr()
    ids_attr = pi.GetIdsAttr()
    if not ids_attr:
        ids_attr = pi.CreateIdsAttr()

    positions: List[Gf.Vec3f] = _get_array(positions_attr)
    proto_indices: List[int] = _get_array(proto_indices_attr)

    orientations: Optional[List[Gf.Quath]] = None
    scales: Optional[List[Gf.Vec3f]] = None
    ids: Optional[List[int]] = []

    if orientations_attr and orientations_attr.HasAuthoredValueOpinion():
        ov = orientations_attr.Get()
        orientations = list(ov) if ov is not None else []
    if scales_attr and scales_attr.HasAuthoredValueOpinion():
        sv = scales_attr.Get()
        scales = list(sv) if sv is not None else []
    iv = ids_attr.Get()
    ids = list(iv) if iv is not None else []

    return positions, proto_indices, orientations, scales, ids


def _set_arrays(
    pi: UsdGeom.PointInstancer,
    positions: Sequence[Gf.Vec3f],
    proto_indices: Sequence[int],
    orientations: Optional[Sequence[Gf.Quath]],
    scales: Optional[Sequence[Gf.Vec3f]],
    ids: Optional[Sequence[int]],
) -> None:
    pi.GetPositionsAttr().Set(list(positions))
    pi.GetProtoIndicesAttr().Set(list(proto_indices))
    if orientations is not None:
        pi.GetOrientationsAttr().Set(list(orientations))
    if scales is not None:
        pi.GetScalesAttr().Set(list(scales))
    if ids is not None:
        pi.GetIdsAttr().Set(list(ids))


def append_instances(pi_prim: Usd.Prim, instances: List[InstanceSpec]) -> None:
    """Append instances to a PointInstancer prim.

    - Deduplicates/updates prototypes list if an instance references a new prototype
    - Appends to positions, protoIndices, and optional orientations/scales/ids arrays
    - All updates are grouped under a single undo step
    """
    if not pi_prim or not pi_prim.IsValid() or not pi_prim.IsA(UsdGeom.PointInstancer):
        raise ValueError("pi_prim must be a valid UsdGeomPointInstancer prim")

    pi = UsdGeom.PointInstancer(pi_prim)

    # Ensure prototypes cover all referenced prototype_path values
    proto_rel = pi.GetPrototypesRel()
    existing_targets: List[Sdf.Path] = list(proto_rel.GetTargets())
    needed: List[Sdf.Path] = []
    for inst in instances:
        if inst.prototype_path is not None:
            sp = _to_sdf_path(inst.prototype_path)
            if sp not in existing_targets and sp not in needed:
                needed.append(sp)

    new_targets = existing_targets + needed if needed else existing_targets

    # Build mapping to indices
    proto_index_map = {p: i for i, p in enumerate(new_targets)}

    stage = pi_prim.GetStage()
    _ensure_edit_target_unmuted(stage)

    with omni.kit.undo.group():
        if needed:
            proto_rel.SetTargets(new_targets)

        positions, proto_indices, orientations, scales, ids = _get_or_create_attr_values(pi)

        # Decide whether to author orientations/scales arrays based on incoming instances or existing authored arrays
        author_orientations = orientations is not None or any(inst.orientation is not None for inst in instances)
        author_scales = scales is not None or any(inst.scale is not None for inst in instances)
        author_ids = ids is not None

        if author_orientations and orientations is None:
            orientations = []
        if author_scales and scales is None:
            scales = []
        if author_ids and ids is None:
            ids = []

        # Next id base
        next_id = (max(ids) + 1) if (ids and len(ids) > 0) else 1

        for inst in instances:
            # positions/protoIndices
            p = inst.position
            if isinstance(p, Gf.Vec3d):
                p = Gf.Vec3f(p[0], p[1], p[2])
            positions.append(p)

            if inst.prototype_path is None:
                proto_index = 0
            else:
                proto_index = proto_index_map[_to_sdf_path(inst.prototype_path)]
            proto_indices.append(int(proto_index))

            # orientations if authored/desired
            if author_orientations:
                if inst.orientation is not None:
                    orientations.append(inst.orientation)
                else:
                    # identity quath
                    orientations.append(Gf.Quath(1.0, Gf.Vec3h(0.0, 0.0, 0.0)))

            # scales if authored/desired
            if author_scales:
                if inst.scale is not None:
                    scales.append(inst.scale)
                else:
                    scales.append(Gf.Vec3f(1.0, 1.0, 1.0))

            # ids if authored or we want to author ids
            if ids is not None:
                ids.append(int(inst.id if inst.id is not None else next_id))
                next_id += 1

        # Write back
        _set_arrays(pi, positions, proto_indices, orientations, scales, ids)


def remove_instances_in_radius(
    pi_prim: Usd.Prim, center: Union[Gf.Vec3f, Gf.Vec3d], radius: float
) -> int:
    """Remove instances whose position is within radius of center.

    Returns number of removed instances.
    """
    if not pi_prim or not pi_prim.IsValid() or not pi_prim.IsA(UsdGeom.PointInstancer):
        raise ValueError("pi_prim must be a valid UsdGeomPointInstancer prim")
    pi = UsdGeom.PointInstancer(pi_prim)

    stage = pi_prim.GetStage()
    _ensure_edit_target_unmuted(stage)

    positions, proto_indices, orientations, scales, ids = _get_or_create_attr_values(pi)

    if not positions:
        return 0

    # Normalize center to Vec3f for comparison
    if isinstance(center, Gf.Vec3d):
        c = Gf.Vec3f(center[0], center[1], center[2])
    else:
        c = center
    r2 = float(radius) * float(radius)

    keep_mask: List[bool] = []
    removed = 0
    for p in positions:
        dx = p[0] - c[0]
        dy = p[1] - c[1]
        dz = p[2] - c[2]
        keep = (dx * dx + dy * dy + dz * dz) > r2
        keep_mask.append(keep)
        if not keep:
            removed += 1

    if removed == 0:
        return 0

    def _filter(seq: Optional[List], mask: List[bool]) -> Optional[List]:
        if seq is None:
            return None
        return [v for v, k in zip(seq, mask) if k]

    new_positions = _filter(positions, keep_mask)
    new_proto_indices = _filter(proto_indices, keep_mask)
    new_orientations = _filter(orientations, keep_mask)
    new_scales = _filter(scales, keep_mask)
    new_ids = _filter(ids, keep_mask)

    with omni.kit.undo.group():
        _set_arrays(
            pi,
            new_positions or [],
            new_proto_indices or [],
            new_orientations,
            new_scales,
            new_ids,
        )

    return removed