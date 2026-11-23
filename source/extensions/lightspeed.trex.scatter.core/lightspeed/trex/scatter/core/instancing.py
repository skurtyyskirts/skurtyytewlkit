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
    "InstanceBlock",
    "InstancedEraser",
    "InstancedPool",
    "InstancingEdit",
    "brush_spec_paths",
    "euler_xyz_to_quaternions",
    "instanced_entry_count",
    "instanced_points",
    "instanced_pool",
    "instancers_of_selection",
    "is_scatter_instancer",
    "merge_loose_placements",
    "quaternions_to_euler_xyz",
    "read_block",
    "split_instancers",
    "subscribe_instancer_changes",
]

import hashlib
import itertools
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, fields

import carb
import numpy as np
import omni.client
from lightspeed.common.constants import ROOTNODE_MESHES
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from .constants import (
    CONTAINER_PREFIX,
    INSTANCED_KEYS_ATTR,
    INSTANCED_ORIGINS_ATTR,
    INSTANCED_ROTATIONS_ATTR,
    INSTANCED_TRANSFORM_KEYS_ATTR,
    INSTANCED_TRANSLATES_ATTR,
    INSTANCED_TURNS_ATTR,
    INSTANCED_YAWS_ATTR,
    INSTANCER_PREFIX,
    IS_REMIX_SCATTER_ATTR,
    PLACEMENT_PREFIX,
    PREVIEW_SUFFIX,
    PROTOTYPE_PREFIX,
    PROTOTYPES_SCOPE_NAME,
    SCATTER_ASSET_ATTR,
    SCATTER_BRUSH_ID_ATTR,
    SCATTER_CONTAINER_ATTR,
    SCATTER_KEY_ATTR,
    SCATTER_ORIGIN_ATTR,
    SCATTER_TRANSFORM_KEY_ATTR,
    SCATTER_TURN_ATTR,
    SCATTER_YAW_ATTR,
)
from .placement import (
    PlacementRecord,
    author_placements,
    prune_inert_overs,
    remove_placements,
    restore_prims,
    snapshot_prims,
    within_radius,
)
from .settings import EraseScope
from .targets import ScatterTarget, get_prototype_root, prototype_side_path

_INSTANCER_TYPE_NAME = "PointInstancer"
_SCOPE_TYPE_NAME = "Scope"
_XFORM_TYPE_NAME = "Xform"
_XFORM_OP_ORDER = ("xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale")
_PROTOTYPES_REL = "prototypes"
# A recorded surface is 6 doubles, or 9 with the stroke heading; the instancer keeps 9 per entry, padded with NaN.
_ORIGIN_STRIDE = 9
# Placement names are s_ plus 12 hex digits. An entry id holds those digits as a number, so splitting the entry gives
# the placement its name back; any other name gets an id above this limit, which no such number reaches.
_NAME_PATTERN = re.compile(rf"^{PLACEMENT_PREFIX}([0-9a-f]{{12}})$")
_NAME_ID_LIMIT = 16**12
_ID_LIMIT = 2**63
# What a placement may carry and still be held exactly by an instancer entry. Anything more, a visibility the user
# set or an extra transform op, would be lost in the instancer, so such a placement stays a prim of its own.
_PLACEMENT_ATTRIBUTES = frozenset(
    {
        "IsRemixRef",
        IS_REMIX_SCATTER_ATTR,
        SCATTER_ASSET_ATTR,
        "xformOp:translate",
        "xformOp:rotateXYZ",
        "xformOp:scale",
        "xformOpOrder",
        SCATTER_ORIGIN_ATTR,
        SCATTER_KEY_ATTR,
        SCATTER_TRANSFORM_KEY_ATTR,
        SCATTER_YAW_ATTR,
        SCATTER_TURN_ATTR,
    }
)
_PLACEMENT_INFO_KEYS = frozenset({"specifier", "typeName", "references"})
_OVERRIDE_INFO_KEYS = frozenset({"specifier"})
# A plain authored value: no connection, no time samples, no metadata a copy of the value would not carry.
_PLAIN_ATTRIBUTE_INFO_KEYS = frozenset({"custom", "variability", "default", "typeName"})
_SHADER_INPUT_PREFIX = "inputs:"
_LOG_PREFIX = "[lightspeed.trex.scatter.core]"
# Told (layer, instancer path) whenever the brush rewrites or deletes an instancer. The toolkit viewport's stand-in
# follows instancers through this rather than through every attribute change on the stage, which a live edit of a
# large stroke would otherwise flood it with.
_CHANGE_LISTENERS: list[Callable[[Sdf.Layer, Sdf.Path], None]] = []


@dataclass
class InstanceBlock:
    """Instancer entries in columns: what each one renders with, and what its placement recorded.

    Entries name their prototype instead of indexing it, so a block stays valid however the instancer it was read
    from, or is written to, orders its prototypes. Missing recorded values are NaN, or 0 for the two keys.
    """

    ids: np.ndarray
    """(N,) int64 identities; an edit finds its entries again by these."""
    prototypes: np.ndarray
    """(N,) object array of prototype prim names."""
    translates: np.ndarray
    """(N, 3) float64 ``xformOp:translate`` of each placement."""
    rotations: np.ndarray
    """(N, 3) float32 ``xformOp:rotateXYZ`` of each placement, in degrees."""
    scales: np.ndarray
    """(N, 3) float32 ``xformOp:scale`` of each placement."""
    origins: np.ndarray
    """(N, 9) float64 recorded surfaces, NaN padded."""
    keys: np.ndarray
    """(N,) uint64 random keys."""
    transform_keys: np.ndarray
    """(N,) uint64 transform checksums."""
    yaws: np.ndarray
    """(N,) float64 recorded gizmo yaws."""
    turns: np.ndarray
    """(N, 3) float64 recorded gizmo turns."""

    @classmethod
    def empty(cls, count: int = 0) -> InstanceBlock:
        """Return a block of ``count`` entries holding nothing recorded."""
        return cls(
            ids=np.zeros(count, dtype=np.int64),
            prototypes=np.full(count, "", dtype=object),
            translates=np.zeros((count, 3), dtype=np.float64),
            rotations=np.zeros((count, 3), dtype=np.float32),
            scales=np.ones((count, 3), dtype=np.float32),
            origins=np.full((count, _ORIGIN_STRIDE), np.nan, dtype=np.float64),
            keys=np.zeros(count, dtype=np.uint64),
            transform_keys=np.zeros(count, dtype=np.uint64),
            yaws=np.full(count, np.nan, dtype=np.float64),
            turns=np.full((count, 3), np.nan, dtype=np.float64),
        )

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, selector) -> InstanceBlock:
        """Return the entries a boolean mask or an index array selects, as a new block."""
        return InstanceBlock(**{name: getattr(self, name)[selector] for name in _BLOCK_FIELDS})

    @classmethod
    def concatenate(cls, blocks: Sequence[InstanceBlock]) -> InstanceBlock:
        """Return the entries of every block, in order, as one block."""
        filled = [block for block in blocks if len(block)]
        if not filled:
            return cls.empty()
        if len(filled) == 1:
            return filled[0]
        return cls(**{name: np.concatenate([getattr(block, name) for block in filled]) for name in _BLOCK_FIELDS})


_BLOCK_FIELDS = tuple(field.name for field in fields(InstanceBlock))


@dataclass
class InstancingEdit:
    """Placements moved between their loose prims and the instancer of their patch, as one reversible edit.

    :meth:`instance` puts the block into the instancer and removes the loose prims; :meth:`uninstance` takes the block
    out of the instancer and restores the loose prims. A merge applies the first, a split or an erase the second, and
    undo runs the other one. Entries are found again by id, so either direction is right whatever else the instancer
    gained or lost in between, as long as the edits made after this one were undone first.
    """

    instancer_path: Sdf.Path
    container_path: Sdf.Path
    brush_id: str
    block: InstanceBlock
    snapshot: Sdf.Layer
    """Anonymous layer holding the loose placements and every prototype the block needs, at their own paths."""
    loose_paths: tuple[str, ...] = ()

    def instance(self, layer: Sdf.Layer) -> None:
        """Put the block into the instancer, creating it and its prototypes as needed, and remove the loose prims."""
        with Sdf.ChangeBlock():
            present = [path for path in self.loose_paths if layer.GetPrimAtPath(path) is not None]
            if present:
                remove_placements(layer, present)
            if not len(self.block):
                return
            spec = _ensure_instancer(layer, self.instancer_path, self.container_path.name, self.brush_id)
            for name in set(self.block.prototypes):
                path = _prototype_path(self.instancer_path, name)
                if layer.GetPrimAtPath(path) is None and self.snapshot.GetPrimAtPath(path) is not None:
                    _copy_spec(self.snapshot, layer, path)
            current = read_block(spec)
            fresh = self.block.take(~np.isin(self.block.ids, current.ids))
            _write_entries(layer, spec, InstanceBlock.concatenate([current, fresh]))

    def uninstance(self, layer: Sdf.Layer) -> None:
        """Take the block out of the instancer, dropping what is left empty, and restore the loose prims."""
        with Sdf.ChangeBlock():
            spec = layer.GetPrimAtPath(self.instancer_path)
            if spec is not None and len(self.block):
                current = read_block(spec)
                _write_entries(layer, spec, current.take(~np.isin(current.ids, self.block.ids)))
            missing = [path for path in self.loose_paths if layer.GetPrimAtPath(path) is None]
            if missing:
                restore_prims(layer, self.snapshot, missing)


class InstancedPool:
    """Instancer entries of one prototype with their world positions, read once and queried by distance.

    The instanced counterpart of ``PlacementPool``: an erase stroke takes the entries under the brush out of it stamp by
    stamp instead of reading every instancer again at every stamp.
    """

    def __init__(
        self,
        groups: Sequence[tuple[Sdf.Path, np.ndarray, np.ndarray]],
        surface_positions: Sequence[np.ndarray] | None = None,
    ):
        """
        Args:
            groups: ``(instancer path, (N,) ids, (N, 3) world positions)`` per instancer.
            surface_positions: ``(N, 3)`` world positions of the surface point each entry stands on, per instancer
                in the order of ``groups``; None when they are the positions themselves.
        """
        self._paths = [path for path, _, _ in groups]
        ids = [np.asarray(group_ids, dtype=np.int64) for _, group_ids, _ in groups]
        self._group = np.concatenate([np.full(len(group), index) for index, group in enumerate(ids)] or [[]]).astype(
            np.int64
        )
        self._ids = np.concatenate(ids or [np.empty(0, dtype=np.int64)])
        self._positions = np.vstack(
            [np.reshape(positions, (-1, 3)) for _, _, positions in groups] or [np.empty((0, 3))]
        )
        self._surface_positions = (
            None
            if surface_positions is None
            else np.vstack([np.reshape(positions, (-1, 3)) for positions in surface_positions] or [np.empty((0, 3))])
        )
        self._alive = np.ones(len(self._ids), dtype=bool)

    def __len__(self) -> int:
        return int(np.count_nonzero(self._alive))

    def take_within(self, center_world: Gf.Vec3d | Sequence[float], radius: float) -> dict[Sdf.Path, np.ndarray]:
        """Return the entries still in the pool within ``radius`` of a world point, and take them out of it.

        As for loose placements, an entry is in reach when either it or the surface point it stands on is.

        Args:
            center_world: World point to measure from.
            radius: Largest accepted distance.

        Returns:
            The ids taken, per instancer path; empty when none is in reach.
        """
        if not len(self._ids):
            return {}
        center = np.array([float(center_world[0]), float(center_world[1]), float(center_world[2])])
        within = self._alive & within_radius(center, radius, self._positions, self._surface_positions)
        indices = np.flatnonzero(within)
        self._alive[indices] = False
        taken: dict[Sdf.Path, np.ndarray] = {}
        for group in np.unique(self._group[indices]):
            taken[self._paths[int(group)]] = self._ids[indices[self._group[indices] == group]]
        return taken


class InstancedEraser:
    """Erases instancer entries and keeps what it erased, per instancer, for one undo step.

    :meth:`erase` only notes what to remove and :meth:`flush` removes it. Removing entries rewrites every array of
    their instancer, which costs the size of the whole patch however few entries go, so a stroke that rewrote it at
    every stamp paid that several times a frame; noting the stamps of a frame and writing once pays it once.
    """

    def __init__(self, layer: Sdf.Layer):
        """
        Args:
            layer: Layer the erased instancers are authored in.
        """
        self._layer = layer
        self._edits: dict[Sdf.Path, InstancingEdit] = {}
        self._queued: dict[Sdf.Path, list[np.ndarray]] = {}

    @property
    def edits(self) -> list[InstancingEdit]:
        """One edit per instancer the eraser touched; ``uninstance`` repeats an erase and ``instance`` undoes it.

        Whatever is still noted is removed first, so the edits always cover every erase asked for.
        """
        self.flush()
        return [edit for edit in self._edits.values() if len(edit.block)]

    @property
    def erased_count(self) -> int:
        """Entries erased so far, the ones still noted included once they are found in their instancer."""
        self.flush()
        return sum(len(edit.block) for edit in self._edits.values())

    def erase(self, instancer_path: Sdf.Path, ids: np.ndarray) -> None:
        """Note entries to remove from an instancer of the layer; :meth:`flush` removes them.

        Args:
            instancer_path: Instancer to erase from; nothing happens when the layer does not author it.
            ids: Ids of the entries to remove.
        """
        noted = np.asarray(ids, dtype=np.int64).reshape(-1)
        if len(noted):
            self._queued.setdefault(Sdf.Path(instancer_path), []).append(noted)

    def flush(self) -> int:
        """Remove every entry noted since the last flush, one rewrite per instancer.

        Returns:
            The number of entries removed.
        """
        queued, self._queued = self._queued, {}
        return sum(self._erase_now(instancer_path, np.concatenate(noted)) for instancer_path, noted in queued.items())

    def _erase_now(self, instancer_path: Sdf.Path, ids: np.ndarray) -> int:
        """Remove the entries with ``ids`` from an instancer of the layer and return how many were there."""
        spec = self._layer.GetPrimAtPath(instancer_path)
        if spec is None or not _is_instancer_spec(spec):
            return 0
        current = read_block(spec)
        erased = np.isin(current.ids, ids)
        if not erased.any():
            return 0
        edit = self._edits.get(instancer_path)
        if edit is None:
            # The prototypes are copied before anything is erased: erasing the last entry of one deletes it.
            snapshot = Sdf.Layer.CreateAnonymous("scatter_instancing")
            for name in set(current.prototypes):
                _copy_spec(self._layer, snapshot, _prototype_path(instancer_path, name))
            edit = InstancingEdit(
                instancer_path=instancer_path,
                container_path=_container_path_of(spec),
                brush_id=_string_attribute(spec, SCATTER_BRUSH_ID_ATTR),
                block=InstanceBlock.empty(),
                snapshot=snapshot,
            )
            self._edits[instancer_path] = edit
        edit.block = InstanceBlock.concatenate([edit.block, current.take(erased)])
        _write_entries(self._layer, spec, current.take(~erased))
        return int(np.count_nonzero(erased))


def is_scatter_instancer(prim: Usd.Prim) -> bool:
    """Return whether ``prim`` is the instancer the brush merged a patch into."""
    return bool(prim) and prim.IsA(UsdGeom.PointInstancer) and prim.HasAttribute(IS_REMIX_SCATTER_ATTR)


def subscribe_instancer_changes(fn: Callable[[Sdf.Layer, Sdf.Path], None]) -> Callable[[], None]:
    """Call ``fn(layer, instancer_path)`` whenever the brush rewrites or deletes an instancer.

    Args:
        fn: Listener; it runs inside the edit, so it should only take note and do its work later.

    Returns:
        A function that ends the subscription.
    """
    _CHANGE_LISTENERS.append(fn)

    def unsubscribe() -> None:
        if fn in _CHANGE_LISTENERS:
            _CHANGE_LISTENERS.remove(fn)

    return unsubscribe


def _notify_changed(layer: Sdf.Layer, instancer_path: Sdf.Path) -> None:
    """Tell every listener an instancer changed; a failing listener is logged and never fails the edit."""
    # A copy, so a listener may unsubscribe while it is being told.
    for fn in _CHANGE_LISTENERS.copy():
        try:
            fn(layer, instancer_path)
        except Exception as exc:  # noqa: BLE001 - a viewport stand-in must never break an undoable edit.
            carb.log_error(f"{_LOG_PREFIX} Instancer change listener failed: {exc}")


def euler_xyz_to_quaternions(rotations: np.ndarray) -> np.ndarray:
    """Return the quaternions of ``xformOp:rotateXYZ`` rotations, laid out ``(i, j, k, real)`` as Vt stores them.

    rotateXYZ turns about X first, then Y, then Z, which is the quaternion product ``qz * qy * qx``.

    Args:
        rotations: (N, 3) Euler angles in degrees.

    Returns:
        (N, 4) float64 unit quaternions.
    """
    half = np.radians(np.asarray(rotations, dtype=np.float64).reshape(-1, 3)) * 0.5
    cos, sin = np.cos(half), np.sin(half)
    zero = np.zeros(len(half))
    turn_x = np.stack([cos[:, 0], sin[:, 0], zero, zero], axis=1)
    turn_y = np.stack([cos[:, 1], zero, sin[:, 1], zero], axis=1)
    turn_z = np.stack([cos[:, 2], zero, zero, sin[:, 2]], axis=1)
    real_first = _quaternion_product(turn_z, _quaternion_product(turn_y, turn_x))
    return np.concatenate([real_first[:, 1:], real_first[:, :1]], axis=1)


def quaternions_to_euler_xyz(quaternions: np.ndarray) -> np.ndarray:
    """Return the ``xformOp:rotateXYZ`` angles of quaternions laid out ``(i, j, k, real)``.

    Args:
        quaternions: (N, 4) unit quaternions.

    Returns:
        (N, 3) float64 Euler angles in degrees.
    """
    values = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4)
    x, y, z, w = values[:, 0], values[:, 1], values[:, 2], values[:, 3]
    # Entries of the rotation matrix Rz * Ry * Rx (column vectors) that pin its three angles down.
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + w * z)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(
        np.stack([np.arctan2(r21, r22), np.arcsin(np.clip(-r20, -1.0, 1.0)), np.arctan2(r10, r00)], axis=1)
    )


def read_block(spec: Sdf.PrimSpec) -> InstanceBlock:
    """Read the entries an instancer spec authors.

    The positions decide how many entries there are, because they are what renders. A column that is missing or has
    the wrong length reads as nothing recorded, and missing ids are given fresh ones, so an instancer edited by hand
    can still be split rather than refused.

    Args:
        spec: Instancer prim spec.

    Returns:
        The entries, possibly none.
    """
    positions = _column(spec, "positions", np.float64, 3)
    count = 0 if positions is None else len(positions)
    block = InstanceBlock.empty(count)
    if not count:
        return block
    block.translates = _column(spec, INSTANCED_TRANSLATES_ATTR, np.float64, 3, count)
    if block.translates is None:
        block.translates = positions
    block.rotations = _column(spec, INSTANCED_ROTATIONS_ATTR, np.float32, 3, count)
    if block.rotations is None:
        orientations = _column(spec, "orientations", np.float64, 4, count)
        block.rotations = (
            np.zeros((count, 3), dtype=np.float32)
            if orientations is None
            else quaternions_to_euler_xyz(orientations).astype(np.float32)
        )
    for name, attr_name, dtype, width in (
        ("scales", "scales", np.float32, 3),
        ("origins", INSTANCED_ORIGINS_ATTR, np.float64, _ORIGIN_STRIDE),
        ("keys", INSTANCED_KEYS_ATTR, np.uint64, 0),
        ("transform_keys", INSTANCED_TRANSFORM_KEYS_ATTR, np.uint64, 0),
        ("yaws", INSTANCED_YAWS_ATTR, np.float64, 0),
        ("turns", INSTANCED_TURNS_ATTR, np.float64, 3),
    ):
        values = _column(spec, attr_name, dtype, width, count)
        if values is not None:
            setattr(block, name, values)
    ids = _column(spec, "ids", np.int64, 0, count)
    block.ids = _unique_ids(ids if ids is not None else np.zeros(count, dtype=np.int64))
    names = [path.name for path in _prototype_targets(spec)]
    indices = _column(spec, "protoIndices", np.int64, 0, count)
    if indices is not None and names:
        table = np.array([*names, ""], dtype=object)
        block.prototypes = table[np.where((indices >= 0) & (indices < len(names)), indices, len(names))]
    return block


def merge_loose_placements(
    layer: Sdf.Layer,
    stage: Usd.Stage | None = None,
    exclude: Iterable[str | Sdf.Path] = (),
    prototype_roots: Iterable[Sdf.Path | str] | None = None,
) -> list[InstancingEdit]:
    """Merge every loose placement the layer authors into the instancer of its patch.

    A patch is one scatter container; its instancer is a sibling of it. Placements carrying anything an instancer
    entry could not hold, such as a visibility or a texture set by hand on one of them, are left as they are.

    Args:
        layer: Layer to merge in; only placements it authors are merged, into instancers it authors.
        stage: Stage the layer belongs to, so a new instancer does not take a name another layer already uses.
        exclude: Placement paths to leave loose, such as the stroke still being previewed.
        prototype_roots: Prototypes to merge under; None merges under every prototype.

    Returns:
        The edits made, one per patch; ``uninstance`` each in reverse order to undo them.
    """
    meshes = layer.GetPrimAtPath(ROOTNODE_MESHES)
    if meshes is None:
        return []
    excluded = {Sdf.Path(str(path)) for path in exclude}
    wanted = None if prototype_roots is None else {Sdf.Path(str(path)) for path in prototype_roots}
    edits: list[InstancingEdit] = []
    with Sdf.ChangeBlock():
        for prototype in list(meshes.nameChildren.values()):
            if wanted is not None and prototype.path not in wanted:
                continue
            for container in list(prototype.nameChildren.values()):
                if not container.name.startswith(CONTAINER_PREFIX) or IS_REMIX_SCATTER_ATTR not in container.attributes:
                    continue
                edit = _merge_edit(layer, stage, container, excluded)
                if edit is None:
                    continue
                edit.instance(layer)
                edits.append(edit)
    return edits


def brush_spec_paths(layer: Sdf.Layer, brush_slug: str) -> list[Sdf.Path]:
    """Return every container and instancer spec one preset authored under the prototypes of ``layer``.

    A preset's scatter is its ``scatter_<slug>`` container of loose placements and the instancers its placements were
    merged into. Instancers are matched by the container they record rather than by name, because a merge takes
    ``instanced_<slug>_1`` while the stage still shows a prim at ``instanced_<slug>``, such as the viewport's hidden
    stand-in of an instancer that was just removed. An instancer recording no container predates that attribute and
    is matched by its name.

    Args:
        layer: Layer to read.
        brush_slug: Prim-name-safe preset identifier (see ``ScatterBrushSettings.slug``).

    Returns:
        The spec paths in prototype order, empty when the layer authors nothing under ``/RootNode/meshes``.
    """
    meshes = layer.GetPrimAtPath(ROOTNODE_MESHES)
    if meshes is None:
        return []
    container_name = f"{CONTAINER_PREFIX}{brush_slug}"
    instancer_name = f"{INSTANCER_PREFIX}{brush_slug}"
    paths = []
    for prototype in meshes.nameChildren.values():
        for child in prototype.nameChildren.values():
            if child.name == container_name:
                paths.append(child.path)
            elif _is_instancer_spec(child):
                recorded = _string_attribute(child, SCATTER_CONTAINER_ATTR)
                if recorded == container_name or (not recorded and child.name == instancer_name):
                    paths.append(child.path)
    return paths


def split_instancers(layer: Sdf.Layer, instancer_paths: Iterable[Sdf.Path | str]) -> list[InstancingEdit]:
    """Turn every entry of the given instancers back into a placement prim the brush can edit.

    Each placement gets back the name, transform and recorded values it was merged with, and the texture overrides of
    its prototype.

    Args:
        layer: Layer the instancers are authored in; instancers it does not author are skipped.
        instancer_paths: Instancers to split.

    Returns:
        The edits made, one per instancer; ``instance`` each in reverse order to undo them.
    """
    edits: list[InstancingEdit] = []
    with Sdf.ChangeBlock():
        for instancer_path in dict.fromkeys(Sdf.Path(str(path)) for path in instancer_paths):
            spec = layer.GetPrimAtPath(instancer_path)
            if spec is None or not _is_instancer_spec(spec):
                continue
            block = read_block(spec)
            if not len(block):
                continue
            edit = _split_edit(layer, spec, block)
            edit.uninstance(layer)
            edits.append(edit)
    return edits


def instanced_points(stage: Usd.Stage, target: ScatterTarget) -> np.ndarray:
    """World surface points of every instancer entry painted under ``target.prototype_root``, in any layer.

    The instanced counterpart of ``existing_placement_points``: the point each entry was painted on, which is what
    padding is measured between.

    Args:
        stage: Stage to inspect.
        target: Target whose prototype is scanned and whose ``parent_world`` places the entries.

    Returns:
        A ``(K, 3)`` float64 array, empty when nothing is instanced.
    """
    points = [surface for _, _, _, surface in _instancer_entries(stage, target)]
    return np.vstack(points) if points else np.empty((0, 3), dtype=np.float64)


def instanced_pool(
    stage: Usd.Stage,
    layer: Sdf.Layer,
    target: ScatterTarget,
    scope: EraseScope = EraseScope.ALL_SCATTERED,
    brush_asset_paths: Sequence[str] = (),
) -> InstancedPool:
    """Read the instancer entries of ``target.prototype_root`` an erase in ``layer`` may remove into a pool.

    Args:
        stage: Stage to inspect.
        layer: Erase layer; instancers it does not author are left out, as placements in another layer are.
        target: Target whose prototype is scanned and whose ``parent_world`` places the entries.
        scope: ``BRUSH_ASSETS`` keeps only entries whose prototype references one of ``brush_asset_paths``.
        brush_asset_paths: Absolute asset paths of the brush, used with ``EraseScope.BRUSH_ASSETS``.

    Returns:
        The pool.
    """
    wanted = {_normalized_url(path) for path in brush_asset_paths} if scope == EraseScope.BRUSH_ASSETS else None
    groups = []
    surface_positions = []
    for instancer, ids, positions, surface in _instancer_entries(stage, target):
        spec = layer.GetPrimAtPath(instancer.GetPath())
        if spec is None or not _is_instancer_spec(spec):
            continue
        keep = np.ones(len(ids), dtype=bool) if wanted is None else _entries_with_assets(instancer, wanted, len(ids))
        groups.append((instancer.GetPath(), ids[keep], positions[keep]))
        surface_positions.append(surface[keep])
    return InstancedPool(groups, surface_positions)


def instanced_entry_count(stage: Usd.Stage, prototype_root: Sdf.Path | str) -> int:
    """Return how many instancer entries are painted under a prototype, in any layer."""
    root = stage.GetPrimAtPath(Sdf.Path(str(prototype_root)))
    if not root:
        return 0
    return sum(
        len(UsdGeom.PointInstancer(child).GetPositionsAttr().Get() or ())
        for child in root.GetChildren()
        if is_scatter_instancer(child)
    )


def instancers_of_selection(stage: Usd.Stage, selected_paths: Iterable[Sdf.Path | str]) -> list[Sdf.Path]:
    """Return the instancers a selection names.

    Selecting an instancer or anything inside it names that instancer; a scatter container or one of its placements
    names the instancer the container merges into; a captured mesh or an instance names every instancer of its hash.

    Args:
        stage: Stage the selection belongs to.
        selected_paths: Selected prim paths; instance-side paths are brought back to the prototype side.

    Returns:
        Prototype-side instancer paths in a stable order, without duplicates.
    """
    found: dict[Sdf.Path, None] = {}
    for selected in selected_paths:
        path = prototype_side_path(str(selected))
        if path is None:
            continue
        prototype_root = get_prototype_root(path)
        root = stage.GetPrimAtPath(prototype_root)
        if not root:
            continue
        instancers = [child for child in root.GetChildren() if is_scatter_instancer(child)]
        # The prim directly under the prototype that the selection lies in: an instancer, a container or the capture.
        top = path
        while top.GetParentPath() != prototype_root and top != prototype_root:
            top = top.GetParentPath()
        if top == prototype_root:
            named = instancers
        elif top.name.startswith(CONTAINER_PREFIX):
            named = [prim for prim in instancers if _container_name_of(prim) == top.name]
        elif top.name.endswith(PREVIEW_SUFFIX):
            # A blade clicked in the toolkit viewport is a prim of the instancer's viewport stand-in.
            shown = top.GetParentPath().AppendChild(top.name.removesuffix(PREVIEW_SUFFIX))
            named = [prim for prim in instancers if prim.GetPath() == shown]
        else:
            named = [prim for prim in instancers if prim.GetPath() == top] or instancers
        for prim in named:
            found[prim.GetPath()] = None
    return list(found)


def _container_name_of(instancer: Usd.Prim) -> str:
    """Return the name of the container an instancer prim's entries split back into."""
    attribute = instancer.GetAttribute(SCATTER_CONTAINER_ATTR)
    value = attribute.Get() if attribute else None
    return str(value) if value else f"{CONTAINER_PREFIX}{instancer.GetName().removeprefix(INSTANCER_PREFIX)}"


def _merge_edit(
    layer: Sdf.Layer, stage: Usd.Stage | None, container: Sdf.PrimSpec, excluded: set[Sdf.Path]
) -> InstancingEdit | None:
    """Build the edit merging a container's loose placements into its instancer, without applying it."""
    candidates = []
    for child in container.nameChildren.values():
        if not child.name.startswith(PLACEMENT_PREFIX) or child.path in excluded:
            continue
        signature = _instanceable_signature(child)
        if signature is not None:
            candidates.append((child, signature))
    if not candidates:
        return None
    instancer_path = _instancer_path_for(layer, stage, container.path)
    loose_paths = [str(child.path) for child, _ in candidates]
    snapshot = snapshot_prims(layer, loose_paths)
    instancer_spec = layer.GetPrimAtPath(instancer_path)
    taken = set() if instancer_spec is None else set(read_block(instancer_spec).ids.tolist())
    count = len(candidates)
    block = InstanceBlock.empty(count)
    prototypes = []
    for index, (child, signature) in enumerate(candidates):
        name = _prototype_name(signature)
        prototypes.append(name)
        prototype_path = _prototype_path(instancer_path, name)
        if snapshot.GetPrimAtPath(prototype_path) is None:
            if layer.GetPrimAtPath(prototype_path) is not None:
                _copy_spec(layer, snapshot, prototype_path)
            else:
                _author_prototype(snapshot, prototype_path, signature[0], layer, child.path)
        entry_id = _id_for_name(child.name)
        while entry_id in taken:
            entry_id = _fresh_id(entry_id)
        taken.add(entry_id)
        block.ids[index] = entry_id
        _read_placement_values(child, block, index)
    block.prototypes = np.array(prototypes, dtype=object)
    return InstancingEdit(
        instancer_path=instancer_path,
        container_path=container.path,
        brush_id=_string_attribute(container, SCATTER_BRUSH_ID_ATTR),
        block=block,
        snapshot=snapshot,
        loose_paths=tuple(loose_paths),
    )


def _split_edit(layer: Sdf.Layer, spec: Sdf.PrimSpec, block: InstanceBlock) -> InstancingEdit:
    """Build the edit splitting every entry of an instancer into a placement, without applying it."""
    instancer_path = spec.path
    container_path = _container_path_of(spec)
    brush_id = _string_attribute(spec, SCATTER_BRUSH_ID_ATTR)
    snapshot = Sdf.Layer.CreateAnonymous("scatter_instancing")
    prototype_assets: dict[str, str] = {}
    for name in set(block.prototypes):
        prototype_path = _prototype_path(instancer_path, name)
        if layer.GetPrimAtPath(prototype_path) is None:
            continue
        _copy_spec(layer, snapshot, prototype_path)
        prototype_assets[name] = _prototype_asset(snapshot.GetPrimAtPath(prototype_path))
    existing = layer.GetPrimAtPath(container_path)
    taken = set() if existing is None else set(existing.nameChildren.keys())
    records = []
    sources = []
    for index in range(len(block)):
        asset = prototype_assets.get(str(block.prototypes[index]))
        if not asset:
            # An entry whose prototype is gone renders nothing, and has nothing to reference as a placement.
            continue
        name = _name_for_id(int(block.ids[index]))
        while name in taken:
            name = _name_for_id(_fresh_id(int(block.ids[index])))
        taken.add(name)
        origin = block.origins[index]
        records.append(
            PlacementRecord(
                container_path=str(container_path),
                prim_name=name,
                asset_rel_path=asset,
                asset_abs_path="",
                translate=_triple(block.translates[index]),
                rotate_xyz=_triple(block.rotations[index]),
                scale=_triple(block.scales[index]),
                brush_id=brush_id,
                origin=tuple(float(value) for value in origin[~np.isnan(origin)]),
                key=int(block.keys[index]),
            )
        )
        sources.append(index)
    loose_paths = author_placements(snapshot, records)
    for path, index in zip(loose_paths, sources):
        placement = snapshot.GetPrimAtPath(path)
        _restore_recorded_values(placement, block, index)
        prototype = snapshot.GetPrimAtPath(_prototype_path(instancer_path, str(block.prototypes[index])))
        for child in prototype.nameChildren.values():
            Sdf.CopySpec(snapshot, child.path, snapshot, placement.path.AppendChild(child.name))
    # Entries without a prototype stay out of the block: splitting them would delete them for nothing.
    return InstancingEdit(
        instancer_path=instancer_path,
        container_path=container_path,
        brush_id=brush_id,
        block=block.take(np.array(sources, dtype=np.int64)),
        snapshot=snapshot,
        loose_paths=tuple(loose_paths),
    )


def _instanceable_signature(spec: Sdf.PrimSpec) -> tuple[str, tuple] | None:
    """Return the asset and texture overrides a placement's prototype is made of, or None when it cannot be one.

    Two placements with the same signature share a prototype. None marks a placement carrying anything an instancer
    entry could not hold, which keeps it a prim of its own instead of losing that in the merge.
    """
    if spec.specifier != Sdf.SpecifierDef or spec.typeName != _XFORM_TYPE_NAME:
        return None
    if not set(spec.ListInfoKeys()) <= _PLACEMENT_INFO_KEYS or len(spec.relationships):
        return None
    attributes = spec.attributes
    if not set(attributes.keys()) <= _PLACEMENT_ATTRIBUTES:
        return None
    if any(not set(attribute.ListInfoKeys()) <= _PLAIN_ATTRIBUTE_INFO_KEYS for attribute in attributes):
        return None
    order = attributes.get("xformOpOrder")
    if order is None or tuple(str(token) for token in order.default or ()) != _XFORM_OP_ORDER:
        return None
    if any(attributes.get(name) is None or attributes[name].default is None for name in _XFORM_OP_ORDER):
        return None
    references = spec.referenceList
    items = list(references.GetAddedOrExplicitItems())
    if len(items) != 1 or list(references.deletedItems):
        return None
    reference = items[0]
    if not reference.assetPath or not reference.primPath.isEmpty or reference.layerOffset != Sdf.LayerOffset():
        return None
    recorded_asset = attributes.get(SCATTER_ASSET_ATTR)
    if recorded_asset is not None and recorded_asset.default and str(recorded_asset.default) != reference.assetPath:
        return None
    overrides: list[tuple[str, str, str]] = []
    for child in spec.nameChildren.values():
        if not _collect_overrides(child, spec.path, overrides):
            return None
    return reference.assetPath, tuple(sorted(overrides))


def _collect_overrides(spec: Sdf.PrimSpec, placement_path: Sdf.Path, into: list[tuple[str, str, str]]) -> bool:
    """Add the shader inputs an override chain beneath a placement authors; False when it authors anything else."""
    if spec.specifier != Sdf.SpecifierOver or spec.typeName or not set(spec.ListInfoKeys()) <= _OVERRIDE_INFO_KEYS:
        return False
    if len(spec.relationships):
        return False
    for attribute in spec.attributes:
        if not attribute.name.startswith(_SHADER_INPUT_PREFIX):
            return False
        if not set(attribute.ListInfoKeys()) <= _PLAIN_ATTRIBUTE_INFO_KEYS:
            return False
        value = attribute.default
        frozen = value.path if isinstance(value, Sdf.AssetPath) else repr(value)
        into.append((str(attribute.path.MakeRelativePath(placement_path)), str(attribute.typeName), frozen))
    return all(_collect_overrides(child, placement_path, into) for child in spec.nameChildren.values())


def _read_placement_values(spec: Sdf.PrimSpec, block: InstanceBlock, index: int) -> None:
    """Copy what one placement authors and records into row ``index`` of a block."""
    attributes = spec.attributes
    block.translates[index] = _triple(attributes["xformOp:translate"].default)
    block.rotations[index] = _triple(attributes["xformOp:rotateXYZ"].default)
    block.scales[index] = _triple(attributes["xformOp:scale"].default)
    origin = _default(attributes, SCATTER_ORIGIN_ATTR)
    if origin is not None:
        values = [float(value) for value in origin][:_ORIGIN_STRIDE]
        block.origins[index, : len(values)] = values
    key = _default(attributes, SCATTER_KEY_ATTR)
    if key is not None:
        block.keys[index] = int(key)
    transform_key = _default(attributes, SCATTER_TRANSFORM_KEY_ATTR)
    if transform_key is not None:
        block.transform_keys[index] = int(transform_key)
    yaw = _default(attributes, SCATTER_YAW_ATTR)
    if yaw is not None:
        block.yaws[index] = float(yaw)
    turn = _default(attributes, SCATTER_TURN_ATTR)
    if turn is not None and len(turn) >= 3:
        block.turns[index] = [float(turn[0]), float(turn[1]), float(turn[2])]


def _restore_recorded_values(spec: Sdf.PrimSpec, block: InstanceBlock, index: int) -> None:
    """Give a placement authored from row ``index`` back the recorded values ``author_placements`` cannot write.

    ``author_placements`` stamps the transform as the brush's and always records a key. The entry knows better: a
    placement moved by hand kept a stale checksum, and one painted before keys existed recorded none.
    """
    transform_key = int(block.transform_keys[index])
    if transform_key:
        _set(spec, SCATTER_TRANSFORM_KEY_ATTR, Sdf.ValueTypeNames.UInt64, transform_key, custom=True)
    else:
        _remove_attribute(spec, SCATTER_TRANSFORM_KEY_ATTR)
    if not int(block.keys[index]):
        _remove_attribute(spec, SCATTER_KEY_ATTR)
    yaw = float(block.yaws[index])
    if not np.isnan(yaw):
        _set(spec, SCATTER_YAW_ATTR, Sdf.ValueTypeNames.Double, yaw, custom=True)
    turn = block.turns[index]
    if not np.isnan(turn).any():
        _set(
            spec,
            SCATTER_TURN_ATTR,
            Sdf.ValueTypeNames.DoubleArray,
            Vt.DoubleArray([float(v) for v in turn]),
            custom=True,
        )


def _write_entries(layer: Sdf.Layer, spec: Sdf.PrimSpec, block: InstanceBlock) -> None:
    """Author a block as every entry of an instancer, deleting the instancer when the block is empty.

    Prototypes no entry uses any more are deleted with it, so an instancer only ever lists what it renders. An
    instancer deleted for being empty takes with it every ancestor it leaves as an ``over`` holding nothing.

    A patch holds a handful of prototypes and tens of thousands of entries, and this runs at every merge and every
    frame of an erase, so the entries are only ever walked by numpy: once per prototype, never once each in Python.
    """
    instancer_path = spec.path
    used = set(block.prototypes.tolist())
    known = {name for name in used if name and layer.GetPrimAtPath(_prototype_path(spec.path, name)) is not None}
    if known != used:
        block = block.take(np.isin(block.prototypes, sorted(known)))
    if not len(block):
        _delete_spec(layer, spec)
        prune_inert_overs(layer, instancer_path.GetParentPath())
        _notify_changed(layer, instancer_path)
        return
    names = sorted(known)
    indices = np.zeros(len(block), dtype=np.int32)
    for index, name in enumerate(names):
        indices[block.prototypes == name] = index
    relationship = spec.relationships.get(_PROTOTYPES_REL) or Sdf.RelationshipSpec(spec, _PROTOTYPES_REL, custom=False)
    relationship.targetPathList.explicitItems = [_prototype_path(spec.path, name) for name in names]
    types = Sdf.ValueTypeNames
    # The runtime reads these five and nothing else: half-precision orientations, not the float ones USD also knows.
    orientations = euler_xyz_to_quaternions(block.rotations)
    for attr_name, type_name, array_type, values, dtype, custom in (
        ("positions", types.Point3fArray, Vt.Vec3fArray, block.translates, np.float32, False),
        ("orientations", types.QuathArray, Vt.QuathArray, orientations, np.float16, False),
        ("scales", types.Float3Array, Vt.Vec3fArray, block.scales, np.float32, False),
        ("protoIndices", types.IntArray, Vt.IntArray, indices, np.int32, False),
        ("ids", types.Int64Array, Vt.Int64Array, block.ids, np.int64, False),
        (INSTANCED_TRANSLATES_ATTR, types.Double3Array, Vt.Vec3dArray, block.translates, np.float64, True),
        (INSTANCED_ROTATIONS_ATTR, types.Float3Array, Vt.Vec3fArray, block.rotations, np.float32, True),
        (INSTANCED_ORIGINS_ATTR, types.DoubleArray, Vt.DoubleArray, block.origins.reshape(-1), np.float64, True),
        (INSTANCED_KEYS_ATTR, types.UInt64Array, Vt.UInt64Array, block.keys, np.uint64, True),
        (INSTANCED_TRANSFORM_KEYS_ATTR, types.UInt64Array, Vt.UInt64Array, block.transform_keys, np.uint64, True),
        (INSTANCED_YAWS_ATTR, types.DoubleArray, Vt.DoubleArray, block.yaws, np.float64, True),
        (INSTANCED_TURNS_ATTR, types.Double3Array, Vt.Vec3dArray, block.turns, np.float64, True),
    ):
        _set(spec, attr_name, type_name, array_type.FromNumpy(_contiguous(values, dtype)), custom=custom)
    scope = layer.GetPrimAtPath(spec.path.AppendChild(PROTOTYPES_SCOPE_NAME))
    if scope is not None:
        # keys(): iterating an Sdf children view yields the child specs, not their names.
        for name in [name for name in scope.nameChildren.keys() if name not in known]:  # noqa: SIM118
            del scope.nameChildren[name]
    _notify_changed(layer, instancer_path)


def _ensure_instancer(layer: Sdf.Layer, path: Sdf.Path, container_name: str, brush_id: str) -> Sdf.PrimSpec:
    """Return the instancer spec at ``path``, creating it and its prototypes scope as needed."""
    spec = layer.GetPrimAtPath(path)
    if spec is None:
        spec = Sdf.CreatePrimInLayer(layer, path)
    spec.specifier = Sdf.SpecifierDef
    spec.typeName = _INSTANCER_TYPE_NAME
    if IS_REMIX_SCATTER_ATTR not in spec.attributes:
        _set(spec, IS_REMIX_SCATTER_ATTR, Sdf.ValueTypeNames.Bool, True, custom=True)
    if SCATTER_CONTAINER_ATTR not in spec.attributes:
        _set(spec, SCATTER_CONTAINER_ATTR, Sdf.ValueTypeNames.String, container_name, custom=True)
    if brush_id and SCATTER_BRUSH_ID_ATTR not in spec.attributes:
        _set(spec, SCATTER_BRUSH_ID_ATTR, Sdf.ValueTypeNames.String, brush_id, custom=True)
    scope_path = path.AppendChild(PROTOTYPES_SCOPE_NAME)
    scope = layer.GetPrimAtPath(scope_path) or Sdf.CreatePrimInLayer(layer, scope_path)
    scope.specifier = Sdf.SpecifierDef
    scope.typeName = _SCOPE_TYPE_NAME
    return spec


def _author_prototype(
    layer: Sdf.Layer, path: Sdf.Path, asset_rel_path: str, source_layer: Sdf.Layer, placement_path: Sdf.Path
) -> None:
    """Author a prototype referencing an asset, with the texture overrides the placement at ``placement_path`` carries.

    The prototype authors an empty op order. A placement's own three ops replace the ones the asset authors on its
    root, so the asset root's transform never showed; an empty order keeps it hidden. The runtime ignores a prototype
    root's transform whatever it is, so this is also what makes the toolkit viewport and the game agree.
    """
    spec = Sdf.CreatePrimInLayer(layer, path)
    spec.specifier = Sdf.SpecifierDef
    spec.typeName = _XFORM_TYPE_NAME
    spec.referenceList.prependedItems = [Sdf.Reference(assetPath=asset_rel_path)]
    _set(spec, SCATTER_ASSET_ATTR, Sdf.ValueTypeNames.String, asset_rel_path, custom=True)
    _set(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, Vt.TokenArray([]), variability=Sdf.VariabilityUniform)
    source = source_layer.GetPrimAtPath(placement_path)
    for child in source.nameChildren.values():
        Sdf.CopySpec(source_layer, child.path, layer, path.AppendChild(child.name))


def _instancer_path_for(layer: Sdf.Layer, stage: Usd.Stage | None, container_path: Sdf.Path) -> Sdf.Path:
    """Return the instancer a container's placements merge into in ``layer``.

    That is the instancer the layer already authors for the container, or else the first free name: free in the layer,
    and on the stage, because an instancer defined by another layer would have its entries replaced wholesale by any
    array this layer authored over it.
    """
    parent = container_path.GetParentPath()
    base = f"{INSTANCER_PREFIX}{container_path.name.removeprefix(CONTAINER_PREFIX)}"
    for index in itertools.count():
        path = parent.AppendChild(base if index == 0 else f"{base}_{index}")
        spec = layer.GetPrimAtPath(path)
        if spec is not None:
            if _is_instancer_spec(spec) and _string_attribute(spec, SCATTER_CONTAINER_ATTR) in (
                "",
                container_path.name,
            ):
                return path
            continue
        if stage is not None and stage.GetPrimAtPath(path):
            continue
        return path
    raise AssertionError("unreachable")


def _instancer_entries(
    stage: Usd.Stage, target: ScatterTarget
) -> list[tuple[Usd.Prim, np.ndarray, np.ndarray, np.ndarray]]:
    """Return what every scatter instancer under the target's prototype holds, in world space.

    Returns:
        ``(instancer, ids, world positions, world surface points)`` per instancer. The surface point of an entry is
        the one its placement recorded when it was painted; an entry that records none stands on its own position.
        Ids are made unique exactly as ``read_block`` makes them, so an entry found here is the one an edit finds.
    """
    root = stage.GetPrimAtPath(target.prototype_root)
    if not root:
        return []
    parent_world = _as_numpy_matrix(target.parent_world)
    xform_cache = UsdGeom.XformCache()
    found = []
    for child in root.GetChildren():
        if not is_scatter_instancer(child):
            continue
        instancer = UsdGeom.PointInstancer(child)
        positions = instancer.GetPositionsAttr().Get()
        if not positions:
            continue
        local = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        exact = (
            child.GetAttribute(INSTANCED_TRANSLATES_ATTR).Get()
            if child.HasAttribute(INSTANCED_TRANSLATES_ATTR)
            else None
        )
        if exact is not None and len(exact) == len(local):
            local = np.asarray(exact, dtype=np.float64).reshape(-1, 3)
        surface = local
        origins = (
            child.GetAttribute(INSTANCED_ORIGINS_ATTR).Get() if child.HasAttribute(INSTANCED_ORIGINS_ATTR) else None
        )
        if origins is not None and len(origins) == len(local) * _ORIGIN_STRIDE:
            recorded = np.asarray(origins, dtype=np.float64).reshape(-1, _ORIGIN_STRIDE)[:, :3]
            surface = np.where(np.isnan(recorded).any(axis=1)[:, None], local, recorded)
        ids_value = instancer.GetIdsAttr().Get()
        ids = _unique_ids(
            np.asarray(ids_value, dtype=np.int64)
            if ids_value is not None and len(ids_value) == len(local)
            else np.zeros(len(local), dtype=np.int64)
        )
        relative, _ = xform_cache.ComputeRelativeTransform(child, root)
        matrix = _as_numpy_matrix(relative) @ parent_world
        found.append((child, ids, local @ matrix[:3, :3] + matrix[3, :3], surface @ matrix[:3, :3] + matrix[3, :3]))
    return found


def _entries_with_assets(instancer: Usd.Prim, wanted: set[str], count: int) -> np.ndarray:
    """Return which entries of an instancer render a prototype referencing one of the ``wanted`` assets."""
    targets = UsdGeom.PointInstancer(instancer).GetPrototypesRel().GetForwardedTargets()
    matches = []
    for target in targets:
        prototype = instancer.GetStage().GetPrimAtPath(target)
        asset = _resolved_prototype_asset(prototype) if prototype else None
        matches.append(asset is not None and _normalized_url(asset) in wanted)
    indices = UsdGeom.PointInstancer(instancer).GetProtoIndicesAttr().Get()
    if indices is None or len(indices) != count or not matches:
        return np.zeros(count, dtype=bool)
    table = np.array([*matches, False], dtype=bool)
    values = np.asarray(indices, dtype=np.int64)
    return table[np.where((values >= 0) & (values < len(matches)), values, len(matches))]


def _resolved_prototype_asset(prim: Usd.Prim) -> str | None:
    """Return the absolute path of the asset a prototype references, resolved against the layer that records it."""
    for spec in prim.GetPrimStack():
        attribute = spec.attributes.get(SCATTER_ASSET_ATTR)
        if attribute is not None and attribute.default:
            return spec.layer.ComputeAbsolutePath(str(attribute.default))
    return None


def _prototype_asset(spec: Sdf.PrimSpec) -> str:
    """Return the asset path a prototype spec references, as authored."""
    recorded = spec.attributes.get(SCATTER_ASSET_ATTR)
    if recorded is not None and recorded.default:
        return str(recorded.default)
    items = list(spec.referenceList.GetAddedOrExplicitItems())
    return items[0].assetPath if items else ""


def _prototype_targets(spec: Sdf.PrimSpec) -> list[Sdf.Path]:
    """Return the prototype paths an instancer spec targets."""
    relationship = spec.relationships.get(_PROTOTYPES_REL)
    return [] if relationship is None else list(relationship.targetPathList.GetAddedOrExplicitItems())


def _prototype_name(signature: tuple[str, tuple]) -> str:
    """Return the prototype name of a placement signature: equal signatures share one prototype."""
    return f"{PROTOTYPE_PREFIX}{hashlib.blake2b(repr(signature).encode('utf-8'), digest_size=6).hexdigest()}"


def _prototype_path(instancer_path: Sdf.Path, name: str) -> Sdf.Path:
    """Return the path of a prototype of an instancer."""
    return instancer_path.AppendChild(PROTOTYPES_SCOPE_NAME).AppendChild(name)


def _container_path_of(spec: Sdf.PrimSpec) -> Sdf.Path:
    """Return the container an instancer's entries split back into."""
    name = _string_attribute(spec, SCATTER_CONTAINER_ATTR)
    if not name:
        # An instancer without the attribute was authored by hand; its own name says which patch it holds.
        name = f"{CONTAINER_PREFIX}{spec.name.removeprefix(INSTANCER_PREFIX)}"
    return spec.path.GetParentPath().AppendChild(name)


def _is_instancer_spec(spec: Sdf.PrimSpec) -> bool:
    """Whether a spec defines a scatter instancer."""
    return (
        spec.specifier == Sdf.SpecifierDef
        and spec.typeName == _INSTANCER_TYPE_NAME
        and IS_REMIX_SCATTER_ATTR in spec.attributes
    )


def _id_for_name(name: str) -> int:
    """Return the entry id of a placement name: its 12 hex digits, or a hash above them for any other name."""
    match = _NAME_PATTERN.match(name)
    if match is not None:
        return int(match.group(1), 16)
    digest = int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "big")
    return _NAME_ID_LIMIT + digest % (_ID_LIMIT - _NAME_ID_LIMIT)


def _name_for_id(entry_id: int) -> str:
    """Return the placement name an entry splits back into."""
    if 0 <= entry_id < _NAME_ID_LIMIT:
        return f"{PLACEMENT_PREFIX}{entry_id:012x}"
    return f"{PLACEMENT_PREFIX}{hashlib.blake2b(str(entry_id).encode('utf-8'), digest_size=6).hexdigest()}"


def _fresh_id(seed: int) -> int:
    """Return another id derived from ``seed``, for the rare id two entries would otherwise share."""
    digest = int.from_bytes(hashlib.blake2b(str(seed).encode("utf-8"), digest_size=6).digest(), "big")
    return digest % _NAME_ID_LIMIT


def _unique_ids(ids: np.ndarray) -> np.ndarray:
    """Return the ids with every repeat replaced by a fresh one, so each entry can be found on its own."""
    values = np.asarray(ids, dtype=np.int64).copy()
    _, first = np.unique(values, return_index=True)
    if len(first) == len(values):
        return values
    taken = set(values[first].tolist())
    repeated = np.setdiff1d(np.arange(len(values)), first)
    for index in repeated:
        candidate = int(values[index])
        while candidate in taken:
            candidate = _fresh_id(candidate)
        taken.add(candidate)
        values[index] = candidate
    return values


def _column(spec: Sdf.PrimSpec, name: str, dtype, width: int, count: int | None = None) -> np.ndarray | None:
    """Return an authored array as numpy, or None when it is missing or does not hold ``count`` entries."""
    attribute = spec.attributes.get(name)
    if attribute is None or attribute.default is None:
        return None
    values = np.asarray(attribute.default, dtype=dtype)
    if width:
        if values.size % width:
            return None
        values = values.reshape(-1, width)
    if count is not None and len(values) != count:
        return None
    return values.copy()


def _contiguous(values: np.ndarray, dtype) -> np.ndarray:
    """Return ``values`` as a C-contiguous array of ``dtype``, which ``FromNumpy`` requires."""
    return np.ascontiguousarray(values, dtype=dtype)


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the Hamilton products of two (N, 4) quaternion arrays laid out ``(real, i, j, k)``."""
    lw, lx, ly, lz = left[:, 0], left[:, 1], left[:, 2], left[:, 3]
    rw, rx, ry, rz = right[:, 0], right[:, 1], right[:, 2], right[:, 3]
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=1,
    )


def _as_numpy_matrix(matrix: Gf.Matrix4d) -> np.ndarray:
    """Return a Gf matrix as a (4, 4) numpy array, keeping its row-vector layout."""
    return np.array([[matrix[row][column] for column in range(4)] for row in range(4)], dtype=np.float64)


def _triple(values) -> tuple[float, float, float]:
    """Return any three-component value as a tuple of three Python floats."""
    return (float(values[0]), float(values[1]), float(values[2]))


def _default(attributes, name: str):
    """Return the default value an attribute spec authors, or None when it is missing or authors none."""
    attribute = attributes.get(name)
    return None if attribute is None else attribute.default


def _string_attribute(spec: Sdf.PrimSpec, name: str) -> str:
    """Return the string an attribute spec authors, or an empty string."""
    value = _default(spec.attributes, name)
    return str(value) if value else ""


def _normalized_url(path: str) -> str:
    """Return the asset path normalized for comparison, with forward slashes."""
    return omni.client.normalize_url(path).replace("\\", "/")


def _set(
    spec: Sdf.PrimSpec,
    name: str,
    type_name: Sdf.ValueTypeName,
    value,
    custom: bool = False,
    variability: Sdf.Variability = Sdf.VariabilityVarying,
) -> None:
    """Author ``value`` as the default of attribute ``name`` on ``spec``, creating the attribute spec when missing."""
    attribute = spec.attributes.get(name)
    if attribute is None:
        attribute = Sdf.AttributeSpec(spec, name, type_name, variability)
    if custom:
        attribute.custom = True
    attribute.default = value


def _remove_attribute(spec: Sdf.PrimSpec, name: str) -> None:
    """Remove an attribute spec when ``spec`` has one."""
    attribute = spec.attributes.get(name)
    if attribute is not None:
        spec.RemoveProperty(attribute)


def _delete_spec(layer: Sdf.Layer, spec: Sdf.PrimSpec) -> None:
    """Delete one prim spec, with everything beneath it, from its parent."""
    parent = spec.nameParent or layer.pseudoRoot
    del parent.nameChildren[spec.name]


def _copy_spec(source: Sdf.Layer, destination: Sdf.Layer, path: Sdf.Path) -> None:
    """Copy the spec at ``path``, with everything beneath it, between two layers at the same path."""
    if source.GetPrimAtPath(path) is None:
        return
    Sdf.CreatePrimInLayer(destination, path)
    Sdf.CopySpec(source, path, destination, path)
