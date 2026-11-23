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
    "build_prototype_target",
    "canonical_instance",
    "find_capture_mesh",
    "get_instance_root",
    "get_prototype_root",
    "has_scatter_ancestor",
    "instance_count",
    "is_scatter_prim",
    "prototype_side_path",
    "resolve_target",
    "selected_capture_meshes",
    "selected_prototypes",
    "validated_anchor_prototype",
]

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import carb
import omni.usd
from lightspeed.common import constants as common_constants
from omni.flux.utils.common.interactive_usd_notices import register_objects_changed_listener
from pxr import Gf, Sdf, Usd, UsdGeom

from .constants import IS_REMIX_SCATTER_ATTR
from .geometry import _stage_key
from .settings import TargetMode

if TYPE_CHECKING:
    from .picking import SurfaceHit

_LOG_PREFIX = "[lightspeed.trex.scatter.core]"
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
    mesh_path = _picked_capture_mesh(stage, hit.path, geometry_root) or find_capture_mesh(
        stage, geometry_root, hit.world_position
    )
    if mesh_path is None:
        return None
    return ScatterTarget(
        prototype_root=prototype,
        parent_instance_root=parent_instance,
        mesh_path=mesh_path,
        parent_world=_local_to_world(stage, parent_instance),
        instance_count=instance_count(stage, prototype),
    )


def prototype_side_path(path: Sdf.Path | str) -> Sdf.Path | None:
    """Rewrite a path that goes through an instance into the equivalent path under the prototype.

    Scatter placements are authored under the prototype and seen through instance proxies, so a path picked up from
    the selection has to be brought back to the side it was authored on before anything can be read or written.

    Args:
        path: Any prim path inside the capture topology.

    Returns:
        The prototype-side path, or None when ``path`` lies outside the capture topology.
    """
    prototype = get_prototype_root(path)
    if prototype is None:
        return None
    instance_root = get_instance_root(path)
    if instance_root is None:
        return Sdf.Path(path)
    relative = str(path)[len(str(instance_root)) :].lstrip("/")
    return prototype.AppendPath(relative) if relative else prototype


def build_prototype_target(stage: Usd.Stage, prototype_root: Sdf.Path) -> ScatterTarget | None:
    """Build a target for a whole prototype, using its canonical instance for geometry and transform.

    Args:
        stage: Stage the prototype belongs to.
        prototype_root: ``/RootNode/meshes/mesh_<HASH>``.

    Returns:
        The target, or None (logged) when the prototype holds no capture mesh.
    """
    instance_root = canonical_instance(stage, prototype_root)
    mesh_path = find_capture_mesh(stage, prototype_root if instance_root is None else instance_root)
    if mesh_path is None:
        carb.log_warn(f"{_LOG_PREFIX} No capture mesh found under {prototype_root}; skipping scatter target")
        return None
    return ScatterTarget(
        prototype_root=prototype_root,
        parent_instance_root=instance_root,
        mesh_path=mesh_path,
        parent_world=_local_to_world(stage, instance_root),
        instance_count=instance_count(stage, prototype_root),
    )


def selected_prototypes(usd_context: omni.usd.UsdContext) -> set[Sdf.Path]:
    """Return the prototype of every selected prim path that lies inside the capture topology."""
    prototypes: set[Sdf.Path] = set()
    for path in usd_context.get_selection().get_selected_prim_paths():
        prototype = get_prototype_root(path)
        if prototype is not None:
            prototypes.add(prototype)
    return prototypes


def selected_capture_meshes(stage: Usd.Stage, selected_paths: Iterable[Sdf.Path | str]) -> list[Sdf.Path]:
    """Return the capture meshes a selection names, as the instance-side paths the mesh cache works in.

    A selected instance or prototype names its capture mesh, a selected mesh names itself and a selected placement
    names the surface it was painted on. Prototype-side selections are read through the canonical instance, which is
    where the geometry sits in the world.

    Args:
        stage: Stage the selection belongs to.
        selected_paths: Selected prim paths; anything outside the capture topology is skipped.

    Returns:
        Each mesh once, in selection order.
    """
    meshes: list[Sdf.Path] = []
    for selected in selected_paths:
        prototype = get_prototype_root(selected)
        if prototype is None:
            continue
        instance = get_instance_root(selected) or canonical_instance(stage, prototype)
        root = instance or prototype
        path = Sdf.Path(str(selected))
        if instance is not None and get_instance_root(selected) is None:
            path = path.ReplacePrefix(prototype, instance)
        mesh = _picked_capture_mesh(stage, path, root) or find_capture_mesh(stage, root)
        if mesh is not None and mesh not in meshes:
            meshes.append(mesh)
    return meshes


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


def _picked_capture_mesh(stage: Usd.Stage, picked_path: Sdf.Path | str, root_path: Sdf.Path) -> Sdf.Path | None:
    """Return the picked prim itself when it is a capture mesh under ``root_path``, which needs no search at all.

    The picker reports the prim it hit, so a hit on the capture mesh names the surface exactly; guessing it from
    bounding boxes instead can pick a neighbouring mesh whose box also contains the point. A hit on a placement, or on
    anything that is not a mesh, still goes through :func:`find_capture_mesh`.
    """
    path = Sdf.Path(str(picked_path))
    if not path.HasPrefix(root_path):
        return None
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsA(UsdGeom.Mesh) or is_scatter_prim(prim) or has_scatter_ancestor(prim, root_path):
        return None
    return path


class _InstanceIndex:
    """The instances of every prototype hash on one stage, read once and kept current with change notices.

    The brush resolves a target on every hover pick, and reading the instances means walking every child of
    ``/RootNode/instances``: a capture with ten thousand instances spent a whole frame on it each time. Only a resync
    that can add, remove or rename an instance drops what was read. Use from the main thread only.
    """

    def __init__(self):
        self._stage_key: tuple[str, str] | None = None
        self._subscription = None
        self._by_hash: dict[str, list[Sdf.Path]] | None = None

    def instances(self, stage: Usd.Stage, mesh_hash: str) -> list[Sdf.Path]:
        """Return the instances of ``mesh_hash`` sorted by their ``_N`` index; the list must not be modified."""
        key = _stage_key(stage)
        if key is None:
            return _read_instances(stage).get(mesh_hash, [])
        if key != self._stage_key:
            self.reset()
            self._stage_key = key
            self._subscription = register_objects_changed_listener(stage, self._on_objects_changed)
        if self._by_hash is None:
            self._by_hash = _read_instances(stage)
        return self._by_hash.get(mesh_hash, [])

    def reset(self) -> None:
        """Forget the stage and what was read from it, and stop listening to it."""
        if self._subscription is not None:
            self._subscription.revoke()
            self._subscription = None
        self._stage_key = None
        self._by_hash = None

    def _on_objects_changed(self, notice, _stage: Usd.Stage) -> None:
        """Drop what was read when a resync could have changed which instances exist."""
        if self._by_hash is None:
            return
        instances_root = Sdf.Path(common_constants.ROOTNODE_INSTANCES)
        for path in notice.GetResyncedPaths():
            if path.IsPropertyPath():
                continue
            # An ancestor of the instances root (the pseudo-root included) recomposes all of them; a direct child of
            # it is an instance coming, going or being renamed. Anything deeper changes no instance's existence.
            if instances_root.HasPrefix(path) or path.GetParentPath() == instances_root:
                self._by_hash = None
                return


_INSTANCE_INDEX = _InstanceIndex()


def reset_instance_index() -> None:
    """Forget the instances read from the current stage and release its change listener.

    Called on extension shutdown; nothing needs it otherwise, because the index follows the stage on its own.
    """
    _INSTANCE_INDEX.reset()


def _instances_of_prototype(stage: Usd.Stage, prototype_root: Sdf.Path | str) -> list[Sdf.Path]:
    """Instances of the prototype's hash sorted by their ``_N`` index."""
    match = _PROTOTYPE_HASH_REGEX.match(str(prototype_root))
    if match is None:
        return []
    return list(_INSTANCE_INDEX.instances(stage, match.group(1)))


def _read_instances(stage: Usd.Stage) -> dict[str, list[Sdf.Path]]:
    """Read every instance under ``/RootNode/instances``, grouped by hash and sorted by their ``_N`` index."""
    instances_root = stage.GetPrimAtPath(common_constants.ROOTNODE_INSTANCES)
    if not instances_root:
        return {}
    indexed: dict[str, list[tuple[int, Sdf.Path]]] = {}
    for child in instances_root.GetChildren():
        name_match = _INSTANCE_NAME_REGEX.match(child.GetName())
        if name_match:
            indexed.setdefault(name_match.group(1), []).append((int(name_match.group(2)), child.GetPath()))
    return {
        mesh_hash: [path for _, path in sorted(entries, key=lambda item: item[0])]
        for mesh_hash, entries in indexed.items()
    }


def _local_to_world(stage: Usd.Stage, path: Sdf.Path | None) -> Gf.Matrix4d:
    """Return the local-to-world transform of the prim at ``path``; identity when None or not transformable."""
    if path is None:
        return Gf.Matrix4d(1.0)
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    if not xformable:
        return Gf.Matrix4d(1.0)
    return xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
