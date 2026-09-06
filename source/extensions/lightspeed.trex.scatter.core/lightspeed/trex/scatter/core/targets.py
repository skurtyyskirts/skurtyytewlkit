"""
* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

from __future__ import annotations

__all__ = [
    "ScatterTarget",
    "canonical_instance",
    "find_capture_mesh",
    "get_instance_root",
    "get_prototype_root",
    "has_scatter_ancestor",
    "instance_count",
    "is_scatter_prim",
    "resolve_target",
    "selected_prototypes",
    "validated_anchor_prototype",
]

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import omni.usd
from lightspeed.common import constants as common_constants
from pxr import Gf, Sdf, Usd, UsdGeom

from .constants import IS_REMIX_SCATTER_ATTR
from .settings import TargetMode

if TYPE_CHECKING:
    from .picking import SurfaceHit

_HASH_PATTERN = r"[A-Z0-9]{16}"
# <inst_|mesh_ path><HASH>[_N]... optionally followed by descendants; group 1 is the hash.
_PROTOTYPE_ROOT_REGEX = re.compile(
    rf"^(?:{re.escape(common_constants.INSTANCE_PATH)}|{re.escape(common_constants.MESH_PATH)})"
    rf"({_HASH_PATTERN})(?:_[0-9]+)*(?:/.*)?$"
)
# group 1 is the inst_<HASH>_<N> prim path.
_INSTANCE_ROOT_REGEX = re.compile(
    rf"^({re.escape(common_constants.INSTANCE_PATH)}{_HASH_PATTERN}(?:_[0-9]+)*)(?:/.*)?$"
)
# Prim NAME of an instance: group 1 is the hash, group 2 the instance index.
_INSTANCE_NAME_REGEX = re.compile(rf"^{re.escape(common_constants.INSTANCE_NAME_PREFIX)}({_HASH_PATTERN})_([0-9]+)$")
_PROTOTYPE_HASH_REGEX = re.compile(rf"^{re.escape(common_constants.MESH_PATH)}({_HASH_PATTERN})$")

# Picked positions come back as float32 and lie exactly on the surface, so a flat mesh's zero-thickness bbox would
# reject its own hit; pad the containment test by a distance far below any mesh size.
_HINT_BBOX_PADDING = 0.05


@dataclass
class ScatterTarget:
    """Where a brush stamp authors placements and which geometry it samples.

    Attributes:
        prototype_root: ``/RootNode/meshes/mesh_<HASH>``, the prim that parents the scatter container.
        parent_instance_root: ``/RootNode/instances/inst_<HASH>_<N>`` whose world transform placements are
            expressed in, or None when the hash has no instance on the stage.
        mesh_path: Instance-side capture mesh used for closest-point, normal and ray queries.
        parent_world: Local-to-world transform of ``parent_instance_root`` (identity when None).
        instance_count: Number of instances of the hash; everything authored under the prototype appears under
            every one of them.
    """

    prototype_root: Sdf.Path
    parent_instance_root: Sdf.Path | None
    mesh_path: Sdf.Path
    parent_world: Gf.Matrix4d
    instance_count: int


def get_prototype_root(path: Sdf.Path | str) -> Sdf.Path | None:
    """Return the ``mesh_<HASH>`` prototype for any path under an instance or a prototype of that hash.

    Args:
        path: Prim path under ``/RootNode/instances/inst_<HASH>_<N>`` or ``/RootNode/meshes/mesh_<HASH>``.

    Returns:
        The prototype path, or None when ``path`` is outside the capture topology (lights included).
    """
    match = _PROTOTYPE_ROOT_REGEX.match(str(path))
    if match is None:
        return None
    return Sdf.Path(f"{common_constants.MESH_PATH}{match.group(1)}")


def get_instance_root(path: Sdf.Path | str) -> Sdf.Path | None:
    """Return the ``inst_<HASH>_<N>`` ancestor-or-self of ``path``, or None when not under an instance."""
    match = _INSTANCE_ROOT_REGEX.match(str(path))
    return Sdf.Path(match.group(1)) if match else None


def is_scatter_prim(prim: Usd.Prim) -> bool:
    """Return whether ``prim`` carries the scatter marker attribute (container or placement)."""
    return bool(prim and prim.HasAttribute(IS_REMIX_SCATTER_ATTR))


def has_scatter_ancestor(prim: Usd.Prim, stop_at: Sdf.Path | None = None) -> bool:
    """Return whether an ancestor of ``prim`` below ``stop_at`` is a scatter prim.

    Args:
        prim: Prim whose ancestors are inspected; the prim itself is not tested.
        stop_at: Exclusive boundary. The walk ends without a match when it reaches this path; None walks up to the
            pseudo-root.
    """
    if not prim:
        return False
    boundary = Sdf.Path(stop_at) if stop_at is not None else None
    parent = prim.GetParent()
    while parent and not parent.IsPseudoRoot():
        if boundary is not None and parent.GetPath() == boundary:
            return False
        if is_scatter_prim(parent):
            return True
        parent = parent.GetParent()
    return False


def find_capture_mesh(stage: Usd.Stage, root_path: Sdf.Path, hint_point: Gf.Vec3d | None = None) -> Sdf.Path | None:
    """Find the mesh under ``root_path`` that the brush should sample.

    Scatter containers (and everything under them) are skipped so the brush never treats a previously scattered
    asset as the surface. With a ``hint_point`` the first mesh whose world bounding box (default purpose) contains
    the point wins; otherwise, or when no box contains it, the first mesh in traversal order is returned.

    Args:
        stage: Stage to traverse.
        root_path: Instance or prototype root; instance proxies are traversed.
        hint_point: World-space point, typically the picked position.

    Returns:
        The mesh prim path, or None when ``root_path`` does not exist or holds no eligible mesh.
    """
    root = stage.GetPrimAtPath(root_path)
    if not root:
        return None
    candidates: list[Usd.Prim] = []
    iterator = iter(Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)))
    for prim in iterator:
        if is_scatter_prim(prim):
            iterator.PruneChildren()
            continue
        if prim.IsA(UsdGeom.Mesh):
            candidates.append(prim)
    if not candidates:
        return None
    if hint_point is not None:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        point = Gf.Vec3d(hint_point)
        for prim in candidates:
            if _padded_world_range(bbox_cache, prim).Contains(point):
                return prim.GetPath()
    return candidates[0].GetPath()


def canonical_instance(stage: Usd.Stage, prototype_root: Sdf.Path) -> Sdf.Path | None:
    """Return the instance of ``prototype_root`` with the lowest ``_N`` suffix, or None when it has none."""
    instances = _instances_of_prototype(stage, prototype_root)
    return instances[0] if instances else None


def instance_count(stage: Usd.Stage, prototype_root: Sdf.Path) -> int:
    """Return how many ``inst_<HASH>_<N>`` prims exist under ``/RootNode/instances`` for ``prototype_root``."""
    return len(_instances_of_prototype(stage, prototype_root))


def resolve_target(
    stage: Usd.Stage, hit: SurfaceHit, mode: TargetMode, anchor_prototype: Sdf.Path | str | None
) -> ScatterTarget | None:
    """Turn a picked surface point into the prototype, parent instance and capture mesh a stamp works with.

    ``TargetMode.HIT_SURFACE`` authors under the prototype of the hit and parents placements to the hit instance
    (or the canonical instance when the hit was on the prototype side). ``TargetMode.ANCHOR`` always authors under
    ``anchor_prototype`` and parents placements to its canonical instance, while the geometry still comes from the
    surface under the cursor.

    Args:
        stage: Stage the hit belongs to.
        hit: Picked prim path and world position.
        mode: Target selection mode.
        anchor_prototype: ``mesh_<HASH>`` path used by ``TargetMode.ANCHOR``; ignored otherwise.

    Returns:
        The resolved target, or None when the hit is outside the capture topology, the anchor is invalid or has no
        instance, or no capture mesh can be found.
    """
    hit_prototype = get_prototype_root(hit.path)
    if hit_prototype is None:
        return None
    hit_instance = get_instance_root(hit.path)
    if mode == TargetMode.ANCHOR:
        prototype = validated_anchor_prototype(stage, anchor_prototype)
        if prototype is None:
            return None
        parent_instance = canonical_instance(stage, prototype)
        if parent_instance is None:
            return None
        geometry_root = hit_instance or hit_prototype
    else:
        prototype = hit_prototype
        parent_instance = hit_instance or canonical_instance(stage, prototype)
        geometry_root = parent_instance or prototype
    mesh_path = find_capture_mesh(stage, geometry_root, hit.world_position)
    if mesh_path is None:
        return None
    return ScatterTarget(
        prototype_root=prototype,
        parent_instance_root=parent_instance,
        mesh_path=mesh_path,
        parent_world=_local_to_world(stage, parent_instance),
        instance_count=instance_count(stage, prototype),
    )


def selected_prototypes(usd_context: omni.usd.UsdContext) -> set[Sdf.Path]:
    """Return the prototype of every selected prim path that lies inside the capture topology."""
    prototypes: set[Sdf.Path] = set()
    for path in usd_context.get_selection().get_selected_prim_paths():
        prototype = get_prototype_root(path)
        if prototype is not None:
            prototypes.add(prototype)
    return prototypes


def validated_anchor_prototype(stage: Usd.Stage, anchor_prototype: Sdf.Path | str | None) -> Sdf.Path | None:
    """Return the anchor as a prototype path when it is exactly a ``mesh_<HASH>`` root that exists on the stage.

    The anchor setting is free-form text, so this is the only way it may reach USD.

    Args:
        stage: Stage the anchor must exist on.
        anchor_prototype: Anchor value, typically ``ScatterBrushSettings.anchor_prototype_path``.

    Returns:
        The prototype path, or None when the anchor is empty, malformed, not a prototype root or absent from the
        stage.
    """
    if anchor_prototype is None:
        return None
    anchor = str(anchor_prototype)
    prototype = get_prototype_root(anchor)
    if prototype is None or str(prototype) != anchor:
        return None
    return prototype if stage.GetPrimAtPath(prototype) else None


def _padded_world_range(bbox_cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> Gf.Range3d:
    """Return the world-space aligned bounds of ``prim`` grown by ``_HINT_BBOX_PADDING``; empty bounds stay empty."""
    world_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if world_range.IsEmpty():
        return world_range
    padding = Gf.Vec3d(_HINT_BBOX_PADDING)
    return Gf.Range3d(world_range.GetMin() - padding, world_range.GetMax() + padding)


def _instances_of_prototype(stage: Usd.Stage, prototype_root: Sdf.Path | str) -> list[Sdf.Path]:
    """Instances of the prototype's hash sorted by their ``_N`` index."""
    match = _PROTOTYPE_HASH_REGEX.match(str(prototype_root))
    if match is None:
        return []
    instances_root = stage.GetPrimAtPath(common_constants.ROOTNODE_INSTANCES)
    if not instances_root:
        return []
    mesh_hash = match.group(1)
    indexed: list[tuple[int, Sdf.Path]] = []
    for child in instances_root.GetChildren():
        name_match = _INSTANCE_NAME_REGEX.match(child.GetName())
        if name_match and name_match.group(1) == mesh_hash:
            indexed.append((int(name_match.group(2)), child.GetPath()))
    indexed.sort(key=lambda item: item[0])
    return [path for _, path in indexed]


def _local_to_world(stage: Usd.Stage, path: Sdf.Path | None) -> Gf.Matrix4d:
    """Return the local-to-world transform of the prim at ``path``; identity when None or not transformable."""
    if path is None:
        return Gf.Matrix4d(1.0)
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    if not xformable:
        return Gf.Matrix4d(1.0)
    return xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
