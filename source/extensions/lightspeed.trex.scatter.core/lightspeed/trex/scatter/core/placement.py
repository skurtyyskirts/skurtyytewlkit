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
    "PlacementPool",
    "PlacementRecord",
    "PlacementTransform",
    "asset_scale_factor",
    "author_placement_transforms",
    "author_placements",
    "container_path_for",
    "erase_candidates",
    "existing_placement_points",
    "generate_flood",
    "generate_stamp",
    "make_relative_asset_path",
    "new_placement_name",
    "placement_pool",
    "remove_placements",
    "restore_prims",
    "snapshot_prims",
    "transform_key",
]

import hashlib
import math
import struct
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import carb
import numpy as np
import omni.client
from lightspeed.common.constants import IS_REMIX_REF_ATTR
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from .constants import (
    CONTAINER_PREFIX,
    IS_REMIX_SCATTER_ATTR,
    PLACEMENT_PREFIX,
    SCATTER_ASSET_ATTR,
    SCATTER_BRUSH_ID_ATTR,
    SCATTER_KEY_ATTR,
    SCATTER_ORIGIN_ATTR,
    SCATTER_TRANSFORM_KEY_ATTR,
    SCATTER_TURN_ATTR,
    SCATTER_YAW_ATTR,
)
from .geometry import MeshSurfaceCache, SurfaceSample, area_weighted_triangle_samples
from .sampling import (
    PaddingIndex,
    choose_asset,
    compose_parent_space_transform,
    falloff_weight,
    placement_rng,
    sample_disk,
    tangent_basis,
    to_parent_space_origin,
    up_axis_vector,
)
from .settings import EraseScope, ScaleMode, ScatterAssetEntry, ScatterBrushSettings, UpAxis
from .targets import ScatterTarget

_LOG_PREFIX = "[lightspeed.trex.scatter.core]"
_XFORM_TYPE_NAME = "Xform"
_XFORM_OP_ORDER = ("xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale")
_ENCODING_INPUT = "inputs:encoding"
# Below this the asset is a sliver or was never measured, and dividing a target height by it would explode the scale.
_MIN_NATURAL_HEIGHT = 1e-6
# Exclusive upper bound of a placement key. Stays inside a signed 64-bit integer so numpy can draw it and Sdf can
# author it as a uint64 without either end having to care about the sign bit.
_PLACEMENT_KEY_LIMIT = 2**63
# Authored transform layout: translate keeps its doubles, rotation and scale are float3 on the prim and are packed as
# 32-bit floats so a checksum taken here matches one taken after reading the values back from a layer.
_TRANSFORM_PACK_FORMAT = "<3d6f"
# Tolerance on the cosine a surface angle is compared against, for the rounding of cos() at the limit itself.
_ANGLE_COSINE_SLACK = 1e-12


@dataclass
class PlacementRecord:
    """One scattered prim: where it is authored, which asset it references and its parent-space transform.

    Records are the JSON-friendly payload of the scatter commands, so paths are strings and vectors are tuples.
    """

    container_path: str
    prim_name: str
    asset_rel_path: str
    asset_abs_path: str
    translate: tuple[float, float, float]
    rotate_xyz: tuple[float, float, float]
    scale: tuple[float, float, float]
    brush_id: str
    origin: tuple[float, ...] = ()
    """Parent-space surface the placement was composed from, as ``ParentSpaceOrigin.to_values()``.

    Authored on the prim so that changing a setting later can recompose the placement instead of only affecting the
    next stroke.
    """

    key: int = 0
    """The placement's own random key, so its rotation and scale draws can be replayed without its stamp."""

    @property
    def prim_path(self) -> str:
        """Full prim path of the placement: ``<container_path>/<prim_name>``."""
        return f"{self.container_path}/{self.prim_name}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record into a JSON-friendly dictionary (vectors become lists)."""
        return {
            "container_path": self.container_path,
            "prim_name": self.prim_name,
            "asset_rel_path": self.asset_rel_path,
            "asset_abs_path": self.asset_abs_path,
            "translate": list(self.translate),
            "rotate_xyz": list(self.rotate_xyz),
            "scale": list(self.scale),
            "brush_id": self.brush_id,
            "origin": list(self.origin),
            "key": self.key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlacementRecord:
        """Build a record from a dictionary produced by :meth:`to_dict` (or an equivalent command payload).

        Args:
            data: Mapping with the record fields; vectors may be lists or tuples.

        Returns:
            The reconstructed record with tuple vectors.
        """
        return cls(
            container_path=str(data["container_path"]),
            prim_name=str(data["prim_name"]),
            asset_rel_path=str(data["asset_rel_path"]),
            asset_abs_path=str(data.get("asset_abs_path", "")),
            translate=_as_float_triple(data["translate"]),
            rotate_xyz=_as_float_triple(data["rotate_xyz"]),
            scale=_as_float_triple(data["scale"]),
            brush_id=str(data.get("brush_id", "")),
            origin=tuple(float(value) for value in data.get("origin", ())),
            key=int(data.get("key", 0)),
        )


@dataclass
class PlacementTransform:
    """The three transform ops of one placement that already exists, as they are authored on its prim.

    Both directions of a re-apply are expressed with these: the values to write, and the values to write back when
    the edit is undone.
    """

    prim_path: str
    translate: tuple[float, float, float]
    rotate_xyz: tuple[float, float, float]
    scale: tuple[float, float, float]
    origin: tuple[float, ...] = ()
    """Surface to record alongside the ops, or empty to leave the recorded one alone.

    A move changes where a placement sits, so the surface it would be composed from has to move with it; carrying it
    here is what lets one command undo the ops and the surface together.
    """

    yaw: float | None = None
    """Degrees about the stage up axis to record, or None to leave the recorded turn alone."""

    turn: tuple[float, float, float] | None = None
    """Whole turn to record as X, Y, Z degrees in parent space, or None to leave the recorded one alone.

    The yaw above only ever held a turn about the up axis, so a placement tipped over by the rings stood back up the
    next time a settings change recomposed it. This carries the tilt as well, so an aim survives a recompose.
    """

    brush_owned: bool = True
    """Whether writing this marks the placement as the brush's again.

    False leaves the recorded checksum stale, which is how a placement the user moved by hand travels with a patch
    that is dragged and still keeps a later settings change from snapping it back into the brush's arrangement.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize into a JSON-friendly dictionary (vectors become lists)."""
        return {
            "prim_path": self.prim_path,
            "translate": list(self.translate),
            "rotate_xyz": list(self.rotate_xyz),
            "scale": list(self.scale),
            "origin": list(self.origin),
            "yaw": self.yaw,
            "turn": None if self.turn is None else list(self.turn),
            "brush_owned": self.brush_owned,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlacementTransform:
        """Build a transform from a dictionary produced by :meth:`to_dict`."""
        yaw = data.get("yaw")
        turn = data.get("turn")
        return cls(
            prim_path=str(data["prim_path"]),
            translate=_as_float_triple(data["translate"]),
            rotate_xyz=_as_float_triple(data["rotate_xyz"]),
            scale=_as_float_triple(data["scale"]),
            origin=tuple(float(value) for value in data.get("origin", ())),
            yaw=None if yaw is None else float(yaw),
            turn=None if turn is None else _as_float_triple(turn),
            brush_owned=bool(data.get("brush_owned", True)),
        )


def transform_key(translate: Sequence[float], rotate_xyz: Sequence[float], scale: Sequence[float]) -> int:
    """Return the checksum of an authored placement transform.

    A placement records the checksum of the transform the brush gave it, so a re-apply can tell a placement it still
    owns from one the user has moved by hand and must leave alone.

    Args:
        translate: Authored ``xformOp:translate``.
        rotate_xyz: Authored ``xformOp:rotateXYZ``.
        scale: Authored ``xformOp:scale``.

    Returns:
        A 64-bit checksum.
    """
    packed = struct.pack(
        _TRANSFORM_PACK_FORMAT,
        *(float(value) for value in translate),
        *(float(value) for value in rotate_xyz),
        *(float(value) for value in scale),
    )
    return int.from_bytes(hashlib.blake2b(packed, digest_size=8).digest(), "big")


def container_path_for(prototype_root: Sdf.Path | str, brush_slug: str) -> Sdf.Path:
    """Return the scatter container path ``<prototype_root>/scatter_<brush_slug>``.

    Args:
        prototype_root: The ``/RootNode/meshes/mesh_<HASH>`` prototype that parents the placements.
        brush_slug: Prim-name-safe preset identifier (see ``ScatterBrushSettings.slug``).

    Returns:
        The container prim path.
    """
    return Sdf.Path(prototype_root).AppendChild(f"{CONTAINER_PREFIX}{brush_slug}")


def make_relative_asset_path(layer: Sdf.Layer, abs_path: str) -> str:
    """Express an absolute asset path relative to ``layer`` with forward slashes.

    Anonymous layers have no location on disk, so the absolute path is returned (forward slashes only).

    Args:
        layer: The layer the reference will be authored in.
        abs_path: Absolute path of the referenced asset.

    Returns:
        The asset path to author in the reference.
    """
    if layer.anonymous:
        return abs_path.replace("\\", "/")
    relative = omni.client.normalize_url(omni.client.make_relative_url(layer.identifier, abs_path))
    return relative.replace("\\", "/")


def new_placement_name() -> str:
    """Return a fresh placement prim name ``s_<12 hex chars>``."""
    return f"{PLACEMENT_PREFIX}{uuid.uuid4().hex[:12]}"


class PlacementPool:
    """The placements of one prototype with their world positions, read once and queried by distance.

    An erase stroke asks for the placements under the brush at every stamp; reading every placement of the
    prototype back from the stage each time costs the whole prototype per stamp, so a stroke reads them once into
    this and takes them out as it erases them.
    """

    def __init__(self, paths: list[Sdf.Path], positions: np.ndarray):
        """
        Args:
            paths: Placement paths.
            positions: ``(N, 3)`` world positions in the order of ``paths``.
        """
        self._paths = paths
        self._positions = positions.reshape(-1, 3)
        self._alive = np.ones(len(paths), dtype=bool)

    def __len__(self) -> int:
        return int(np.count_nonzero(self._alive))

    def take_within(self, center_world: Gf.Vec3d | Sequence[float], radius: float) -> list[Sdf.Path]:
        """Return the placements still in the pool within ``radius`` of a world point, and take them out of it.

        Args:
            center_world: World point to measure from.
            radius: Largest accepted distance.

        Returns:
            The placement paths, in the order the pool was read in.
        """
        if not self._paths:
            return []
        center = np.array([float(center_world[0]), float(center_world[1]), float(center_world[2])])
        deltas = self._positions - center
        within = self._alive & (np.einsum("ij,ij->i", deltas, deltas) <= radius * radius)
        indices = np.flatnonzero(within)
        self._alive[indices] = False
        return [self._paths[index] for index in indices]


def placement_pool(
    stage: Usd.Stage,
    target: ScatterTarget,
    scope: EraseScope = EraseScope.ALL_SCATTERED,
    brush_asset_paths: Sequence[str] = (),
) -> PlacementPool:
    """Read the placements of ``target.prototype_root`` an erase may remove into a pool.

    Args:
        stage: Stage to inspect.
        target: Target whose prototype is scanned and whose ``parent_world`` composes the local translations.
        scope: ``BRUSH_ASSETS`` keeps only placements whose referenced asset is one of ``brush_asset_paths``.
        brush_asset_paths: Absolute asset paths of the brush, used with ``EraseScope.BRUSH_ASSETS``.

    Returns:
        The pool.
    """
    wanted = {_normalized_url(path) for path in brush_asset_paths} if scope == EraseScope.BRUSH_ASSETS else None
    paths: list[Sdf.Path] = []
    positions: list[np.ndarray] = []
    for prim in _iter_placement_prims(stage, target.prototype_root):
        if wanted is not None:
            asset_path = _resolve_placement_asset(prim)
            if asset_path is None or _normalized_url(asset_path) not in wanted:
                continue
        paths.append(prim.GetPath())
        positions.append(_placement_world_position(prim, target.parent_world))
    stacked = np.vstack(positions) if positions else np.empty((0, 3), dtype=np.float64)
    return PlacementPool(paths, stacked)


def existing_placement_points(stage: Usd.Stage, target: ScatterTarget) -> np.ndarray:
    """World positions of every placement authored under the scatter containers of ``target.prototype_root``.

    Args:
        stage: Stage to inspect.
        target: Target whose prototype is scanned and whose ``parent_world`` composes the local translations.

    Returns:
        A ``(K, 3)`` float64 array, empty when no placement exists.
    """
    points = [
        _placement_world_position(prim, target.parent_world)
        for prim in _iter_placement_prims(stage, target.prototype_root)
    ]
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(points)


def generate_stamp(
    cache: MeshSurfaceCache,
    target: ScatterTarget,
    center: SurfaceSample,
    settings: ScatterBrushSettings,
    assets: Sequence[ScatterAssetEntry],
    rng: np.random.Generator,
    padding_index: PaddingIndex,
    heading_world: Gf.Vec3d | None,
    stage_up: UpAxis,
    stage_meters_per_unit: float,
    layer: Sdf.Layer,
) -> list[PlacementRecord]:
    """Generate the placements of one brush stamp centered on ``center``.

    ``ceil(density)`` candidates are drawn in the brush disk, accepted against the falloff curve scaled by the brush
    strength, projected back onto the capture mesh, filtered by surface angle and padding and finally turned into
    parent-space records.

    Args:
        cache: Mesh cache used to project candidates onto ``target.mesh_path``.
        target: Resolved scatter target.
        center: Surface sample under the cursor (world position and normal).
        settings: Brush settings.
        assets: Brush assets to choose from.
        rng: Random generator driving every random decision of the stamp.
        padding_index: Spatial index of already placed points; accepted points are added to it.
        heading_world: Normalized stroke direction in world space, or None for the first stamp.
        stage_up: Up axis of the stage.
        stage_meters_per_unit: Meters per stage unit, used to convert each asset's own size into stage units.
        layer: Layer the placements will be authored in (used to relativize asset paths).

    Returns:
        The placement records of the stamp, possibly empty.
    """
    count = math.ceil(settings.density)
    if count <= 0:
        return []
    disk = sample_disk(rng, count)
    radial = np.linalg.norm(disk, axis=1)
    weights = np.fromiter((falloff_weight(settings.falloff, float(t)) for t in radial), dtype=np.float64, count=count)
    accepted = rng.random(count) < weights * settings.strength
    if not np.any(accepted):
        return []
    tangent, bitangent = tangent_basis(np.asarray(center.normal, dtype=np.float64), _as_array(heading_world))
    offsets = disk[accepted] * settings.radius
    candidates = (
        np.asarray(center.position, dtype=np.float64)[None, :]
        + offsets[:, :1] * tangent[None, :]
        + offsets[:, 1:] * bitangent[None, :]
    )
    positions, normals, valid = cache.closest_points(target.mesh_path, candidates, settings.radius)
    # Measured from the stage up axis rather than from the surface under the cursor: the cursor itself often lands on
    # a bevel or a slope beside a floor, and measured from there, blades on that slope would all count as in range.
    valid &= _within_surface_angle(normals, up_axis_vector(stage_up), settings.max_surface_angle)
    return _build_records(
        positions,
        normals,
        valid,
        target,
        settings,
        assets,
        rng,
        padding_index,
        heading_world,
        stage_up,
        stage_meters_per_unit,
        layer,
    )


def generate_flood(
    cache: MeshSurfaceCache,
    target: ScatterTarget,
    settings: ScatterBrushSettings,
    assets: Sequence[ScatterAssetEntry],
    rng: np.random.Generator,
    padding_index: PaddingIndex,
    stage_up: UpAxis,
    stage_meters_per_unit: float,
    layer: Sdf.Layer,
    max_count: int,
) -> list[PlacementRecord]:
    """Generate placements covering the whole capture mesh of ``target``.

    The count is ``density`` per brush-disk area, scaled to the mesh area and capped by ``max_count``. Samples on a
    surface steeper than ``max_surface_angle`` from the stage up axis are dropped, as they are for a stamp.

    Args:
        cache: Mesh cache providing the geometry of ``target.mesh_path``.
        target: Resolved scatter target.
        settings: Brush settings.
        assets: Brush assets to choose from.
        rng: Random generator driving every random decision of the flood.
        padding_index: Spatial index of already placed points; accepted points are added to it.
        stage_up: Up axis of the stage.
        stage_meters_per_unit: Meters per stage unit, used to convert each asset's own size into stage units.
        layer: Layer the placements will be authored in (used to relativize asset paths).
        max_count: Hard cap on the number of generated placements.

    Returns:
        The placement records, empty when the mesh has no surface or ``max_count`` is not positive.
    """
    if max_count <= 0:
        return []
    geometry = cache.get(target.mesh_path)
    if geometry is None or geometry.triangle_count == 0 or geometry.area <= 0.0:
        return []
    disk_area = math.pi * settings.radius * settings.radius
    estimated = math.ceil(settings.density * geometry.area / disk_area)
    count = max(1, min(max_count, estimated))
    positions, normals = area_weighted_triangle_samples(rng, geometry, count)
    valid = _within_surface_angle(normals, up_axis_vector(stage_up), settings.max_surface_angle)
    return _build_records(
        positions,
        normals,
        valid,
        target,
        settings,
        assets,
        rng,
        padding_index,
        None,
        stage_up,
        stage_meters_per_unit,
        layer,
    )


def erase_candidates(
    stage: Usd.Stage,
    target: ScatterTarget,
    center_world: Gf.Vec3d,
    radius: float,
    scope: EraseScope,
    brush_asset_paths: Sequence[str],
) -> list[Sdf.Path]:
    """Placements of ``target.prototype_root`` whose world position lies within ``radius`` of ``center_world``.

    Args:
        stage: Stage to inspect.
        target: Resolved scatter target.
        center_world: Brush center in world space.
        radius: Brush radius.
        scope: ``ALL_SCATTERED`` keeps every placement, ``BRUSH_ASSETS`` keeps only placements whose referenced asset
            is one of ``brush_asset_paths``.
        brush_asset_paths: Absolute asset paths of the brush, used with ``EraseScope.BRUSH_ASSETS``.

    Returns:
        The prim paths to erase.
    """
    return placement_pool(stage, target, scope, brush_asset_paths).take_within(center_world, radius)


def author_placements(
    layer: Sdf.Layer,
    records: Sequence[PlacementRecord],
    materials: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> list[str]:
    """Author the placement prims described by ``records`` into ``layer`` with pure Sdf edits.

    Each container is ensured as a ``def Xform`` carrying the scatter marker and brush id; each placement is a
    ``def Xform`` with a prepended reference, the Remix reference and scatter markers, the asset path and explicit
    translate / rotateXYZ / scale ops. Existing prim names are left untouched.

    When ``materials`` holds an entry for a record's asset, the texture inputs it names are authored as overrides
    beneath the placement, so the whole subtree still belongs to the placement and undo removes it with the prim.

    Args:
        layer: Layer to author in.
        records: Placements to author.
        materials: Relative asset path to the shader overrides to author, as built by
            ``lightspeed.trex.scatter.core.materials.build_materials_payload``.

    Returns:
        The paths of the prims that were created.
    """
    created: list[str] = []
    with Sdf.ChangeBlock():
        for record in records:
            if not Sdf.Path.IsValidIdentifier(record.prim_name):
                carb.log_warn(f"{_LOG_PREFIX} Skipping scatter placement with invalid prim name '{record.prim_name}'")
                continue
            container = _ensure_container_spec(layer, Sdf.Path(record.container_path), record.brush_id)
            if record.prim_name in container.nameChildren:
                continue
            spec = Sdf.CreatePrimInLayer(layer, container.path.AppendChild(record.prim_name))
            spec.specifier = Sdf.SpecifierDef
            spec.typeName = _XFORM_TYPE_NAME
            spec.referenceList.prependedItems = [Sdf.Reference(assetPath=record.asset_rel_path)]
            _set_attribute(spec, IS_REMIX_REF_ATTR, Sdf.ValueTypeNames.Bool, True, custom=True)
            _set_attribute(spec, IS_REMIX_SCATTER_ATTR, Sdf.ValueTypeNames.Bool, True, custom=True)
            _set_attribute(spec, SCATTER_ASSET_ATTR, Sdf.ValueTypeNames.String, record.asset_rel_path, custom=True)
            _set_transform_ops(spec, record.translate, record.rotate_xyz, record.scale)
            _set_attribute(
                spec,
                "xformOpOrder",
                Sdf.ValueTypeNames.TokenArray,
                Vt.TokenArray(list(_XFORM_OP_ORDER)),
                variability=Sdf.VariabilityUniform,
            )
            if record.origin:
                _set_attribute(
                    spec,
                    SCATTER_ORIGIN_ATTR,
                    Sdf.ValueTypeNames.DoubleArray,
                    Vt.DoubleArray(list(record.origin)),
                    custom=True,
                )
            _set_attribute(spec, SCATTER_KEY_ATTR, Sdf.ValueTypeNames.UInt64, record.key, custom=True)
            if materials:
                _author_material_overrides(layer, spec, materials.get(record.asset_rel_path))
            created.append(str(spec.path))
    return created


def author_placement_transforms(layer: Sdf.Layer, transforms: Sequence[PlacementTransform]) -> list[str]:
    """Author the transform ops of placements that already exist, refreshing each one's transform checksum.

    Only the three ops and the checksum are touched, so everything else a placement carries, including its reference
    and any material override beneath it, is left exactly as it was. Paths without a spec in ``layer`` are skipped:
    a placement authored in another layer is not this layer's to change.

    Args:
        layer: Layer to author in, normally the stage's edit target.
        transforms: The placements to rewrite.

    Returns:
        The paths that were written.
    """
    written: list[str] = []
    with Sdf.ChangeBlock():
        for transform in transforms:
            spec = layer.GetPrimAtPath(Sdf.Path(transform.prim_path))
            if spec is None:
                continue
            _set_transform_ops(
                spec,
                transform.translate,
                transform.rotate_xyz,
                transform.scale,
                brush_owned=transform.brush_owned,
            )
            if transform.origin:
                _set_attribute(
                    spec,
                    SCATTER_ORIGIN_ATTR,
                    Sdf.ValueTypeNames.DoubleArray,
                    Vt.DoubleArray(list(transform.origin)),
                    custom=True,
                )
            if transform.yaw is not None:
                _set_attribute(spec, SCATTER_YAW_ATTR, Sdf.ValueTypeNames.Double, float(transform.yaw), custom=True)
            if transform.turn is not None:
                _set_attribute(
                    spec,
                    SCATTER_TURN_ATTR,
                    Sdf.ValueTypeNames.DoubleArray,
                    Vt.DoubleArray([float(value) for value in transform.turn]),
                    custom=True,
                )
            written.append(transform.prim_path)
    return written


def _set_transform_ops(
    spec: Sdf.PrimSpec,
    translate: Sequence[float],
    rotate_xyz: Sequence[float],
    scale: Sequence[float],
    brush_owned: bool = True,
) -> None:
    """Author the three transform ops on a placement spec, stamping them as the brush's unless told otherwise.

    Leaving the checksum stale is what keeps a placement the user moved by hand recognizable as theirs even after a
    move has carried it along with the rest of its patch.
    """
    _set_attribute(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*translate))
    _set_attribute(spec, "xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*rotate_xyz))
    _set_attribute(spec, "xformOp:scale", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*scale))
    if not brush_owned:
        return
    _set_attribute(
        spec,
        SCATTER_TRANSFORM_KEY_ATTR,
        Sdf.ValueTypeNames.UInt64,
        transform_key(translate, rotate_xyz, scale),
        custom=True,
    )


def _author_material_overrides(
    layer: Sdf.Layer, placement_spec: Sdf.PrimSpec, overrides: Sequence[Mapping[str, Any]] | None
) -> None:
    """Author the texture inputs of ``overrides`` on shader overs beneath ``placement_spec``.

    ``Sdf.CreatePrimInLayer`` leaves every prim it creates with the default ``over`` specifier, which is exactly what
    an override chain onto a referenced asset needs. Inputs the placement already carries an opinion for are never
    touched, so a hand-assigned texture always wins.
    """
    for override in overrides or ():
        shader_rel_path = str(override.get("shader_rel_path", ""))
        if not shader_rel_path:
            continue
        shader_spec = Sdf.CreatePrimInLayer(layer, placement_spec.path.AppendPath(shader_rel_path))
        for name, texture_path in (override.get("inputs") or {}).items():
            _set_attribute_if_absent(shader_spec, name, Sdf.ValueTypeNames.Asset, Sdf.AssetPath(texture_path))
        encoding = override.get("encoding")
        if encoding is not None:
            _set_attribute_if_absent(shader_spec, _ENCODING_INPUT, Sdf.ValueTypeNames.Int, int(encoding))


def remove_placements(layer: Sdf.Layer, prim_paths: Sequence[str], remove_empty_containers: bool = True) -> None:
    """Delete the prim specs at ``prim_paths`` from ``layer``.

    A container whose every child is being removed is deleted in one edit rather than child by child. Sdf removes a
    child by searching and rewriting its parent's list of children, so deleting a large stroke one child at a time
    costs time proportional to the stroke times the container, which is seconds for a stroke of thousands.

    Args:
        layer: Layer to edit.
        prim_paths: Placement prim paths; paths without a spec in ``layer`` are ignored.
        remove_empty_containers: Also drop scatter containers left without children and references.
    """
    with Sdf.ChangeBlock():
        for parent_path, names in _group_by_parent(prim_paths).items():
            parent = layer.GetPrimAtPath(parent_path)
            if parent is None:
                continue
            if remove_empty_containers and _is_removable_container(parent) and _covers_children(parent, names):
                _delete_spec(layer, parent)
                continue
            for name in names:
                if name in parent.nameChildren:
                    del parent.nameChildren[name]
            if remove_empty_containers and _is_removable_container(parent) and len(parent.nameChildren) == 0:
                _delete_spec(layer, parent)


def snapshot_prims(layer: Sdf.Layer, prim_paths: Sequence[str], into: Sdf.Layer | None = None) -> Sdf.Layer:
    """Copy the specs at ``prim_paths`` (and their containers' own definition) into a snapshot layer.

    A container whose every child is being snapshot is copied in one edit, for the same reason
    :func:`remove_placements` deletes it in one.

    Args:
        layer: Layer holding the specs.
        prim_paths: Placement prim paths; paths without a spec in ``layer`` are skipped.
        into: Snapshot layer to accumulate into, so an erase stroke can snapshot stamp by stamp while keeping every
            container it touches restorable; a new anonymous layer is created when None.

    Returns:
        The snapshot layer usable with :func:`restore_prims`: ``into`` when given, otherwise the new anonymous layer.
    """
    snapshot = Sdf.Layer.CreateAnonymous("scatter_snapshot") if into is None else into
    with Sdf.ChangeBlock():
        for parent_path, names in _group_by_parent(prim_paths).items():
            container = layer.GetPrimAtPath(parent_path)
            if (
                container is not None
                and snapshot.GetPrimAtPath(parent_path) is None
                and container.name.startswith(CONTAINER_PREFIX)
                and _covers_children(container, names)
            ):
                _copy_spec(layer, snapshot, parent_path)
                continue
            for name in names:
                path = parent_path.AppendChild(name)
                if layer.GetPrimAtPath(path) is None:
                    continue
                _copy_spec(layer, snapshot, path)
                _copy_container_definition(layer, snapshot, parent_path)
    return snapshot


def restore_prims(layer: Sdf.Layer, snapshot: Sdf.Layer, prim_paths: Sequence[str]) -> None:
    """Copy the specs at ``prim_paths`` from ``snapshot`` back into ``layer``, re-defining removed containers.

    A container that is gone from ``layer`` and whose every snapshot child is being restored comes back in one edit.

    Args:
        layer: Layer to restore into.
        snapshot: Layer produced by :func:`snapshot_prims`.
        prim_paths: Placement prim paths; paths missing from ``snapshot`` are skipped.
    """
    with Sdf.ChangeBlock():
        for parent_path, names in _group_by_parent(prim_paths).items():
            stored = snapshot.GetPrimAtPath(parent_path)
            if (
                stored is not None
                and layer.GetPrimAtPath(parent_path) is None
                and stored.name.startswith(CONTAINER_PREFIX)
                and _covers_children(stored, names)
            ):
                _copy_spec(snapshot, layer, parent_path)
                continue
            for name in names:
                path = parent_path.AppendChild(name)
                if snapshot.GetPrimAtPath(path) is None:
                    continue
                _copy_spec(snapshot, layer, path)
                _copy_container_definition(snapshot, layer, parent_path)


def asset_scale_factor(
    asset: ScatterAssetEntry, settings: ScatterBrushSettings, stage_meters_per_unit: float
) -> float | None:
    """Factor turning a sampled scale value into the multiplier a placement authors.

    ``MULTIPLIER`` samples the multiplier directly, so the factor is 1. ``ABSOLUTE_SIZE`` samples a height in stage
    units, so the factor is the reciprocal of the asset's own height once converted into those units.

    Returns:
        The factor, or None when the asset has no usable measurement. None keeps the placement at the asset's own
        size rather than authoring a height as if it were a multiplier, which is what made unmeasured assets come out
        enormous in the first place.
    """
    if settings.scale_mode != ScaleMode.ABSOLUTE_SIZE:
        return 1.0
    height = asset.natural_height * (asset.meters_per_unit / stage_meters_per_unit)
    return 1.0 / height if height > _MIN_NATURAL_HEIGHT else None


def _within_surface_angle(normals: np.ndarray, reference, max_angle: float) -> np.ndarray:
    """Return which unit normals turn no more than ``max_angle`` degrees away from ``reference``.

    Args:
        normals: (K, 3) unit normals; rows of zeros (invalid samples) only pass when nothing is filtered.
        reference: Direction to measure from; need not be normalized.
        max_angle: Largest accepted angle in degrees; 180 or more accepts everything.

    Returns:
        (K,) boolean mask.
    """
    count = len(normals)
    if max_angle >= 180.0:
        return np.ones(count, dtype=bool)
    unit = np.array([float(reference[0]), float(reference[1]), float(reference[2])], dtype=np.float64)
    length = float(np.linalg.norm(unit))
    if length <= 0.0:
        return np.ones(count, dtype=bool)
    # Slack for cos() itself: cos(90 degrees) is 6e-17 rather than 0, which would drop a surface exactly at the limit.
    return (np.asarray(normals, dtype=np.float64).reshape(count, 3) @ (unit / length)) >= math.cos(
        math.radians(max_angle)
    ) - _ANGLE_COSINE_SLACK


def _as_float_triple(values: Sequence[float]) -> tuple[float, float, float]:
    """Return ``values`` as a tuple of three Python floats."""
    x, y, z = (float(value) for value in values)
    return (x, y, z)


def _as_array(vector: Gf.Vec3d | None) -> np.ndarray | None:
    """Return the vector as a float64 numpy array, or None when no vector is given."""
    if vector is None:
        return None
    return np.array([vector[0], vector[1], vector[2]], dtype=np.float64)


def _normalized_url(path: str) -> str:
    """Return the asset path normalized for comparison, with forward slashes."""
    return omni.client.normalize_url(path).replace("\\", "/")


def _iter_placement_prims(stage: Usd.Stage, prototype_root: Sdf.Path | str) -> Iterator[Usd.Prim]:
    """Yield every placement prim under the scatter containers that are children of ``prototype_root``."""
    root = stage.GetPrimAtPath(Sdf.Path(prototype_root))
    if not root:
        return
    for container in root.GetChildren():
        if not container.HasAttribute(IS_REMIX_SCATTER_ATTR):
            continue
        for prim in container.GetChildren():
            if prim.HasAttribute(IS_REMIX_SCATTER_ATTR) and prim.HasAttribute(IS_REMIX_REF_ATTR):
                yield prim


def _placement_world_position(prim: Usd.Prim, parent_world: Gf.Matrix4d) -> np.ndarray:
    """Return the world position of a placement: its local translation composed with ``parent_world``."""
    local = UsdGeom.Xformable(prim).GetLocalTransformation().ExtractTranslation()
    world = parent_world.Transform(local)
    return np.array([world[0], world[1], world[2]], dtype=np.float64)


def _resolve_placement_asset(prim: Usd.Prim) -> str | None:
    """Return the absolute path of the asset a placement references, resolved against its authoring layer."""
    for spec in prim.GetPrimStack():
        attr_spec = spec.attributes.get(SCATTER_ASSET_ATTR)
        if attr_spec is None or not attr_spec.default:
            continue
        return spec.layer.ComputeAbsolutePath(str(attr_spec.default))
    return None


def _build_records(
    positions: np.ndarray,
    normals: np.ndarray,
    valid: np.ndarray,
    target: ScatterTarget,
    settings: ScatterBrushSettings,
    assets: Sequence[ScatterAssetEntry],
    rng: np.random.Generator,
    padding_index: PaddingIndex,
    heading_world: Gf.Vec3d | None,
    stage_up: UpAxis,
    stage_meters_per_unit: float,
    layer: Sdf.Layer,
) -> list[PlacementRecord]:
    """Turn the ``valid`` sample positions into placement records for ``target``.

    Shared tail of :func:`generate_stamp` and :func:`generate_flood`: padding rejection, weighted asset choice,
    parent-space transform composition and asset path relativization. Returns an empty list as soon as no asset can
    be chosen.
    """
    container_path = str(container_path_for(target.prototype_root, settings.slug()))
    relative_asset_paths: dict[str, str] = {}
    records: list[PlacementRecord] = []
    for position, normal, is_valid in zip(positions, normals, valid):
        if not is_valid:
            continue
        if settings.padding > 0.0 and not padding_index.is_free(position, settings.padding):
            continue
        asset = choose_asset(rng, assets)
        if asset is None:
            return []
        if settings.padding > 0.0:
            padding_index.add(position)
        origin = to_parent_space_origin(
            Gf.Vec3d(*(float(value) for value in position)),
            Gf.Vec3d(*(float(value) for value in normal)),
            target.parent_world,
            heading_world,
        )
        # Drawn from the stamp so the stroke stays reproducible, then used on its own: the stamp generator is shared
        # by every placement of the stamp, so nothing could replay one placement's draws from it afterwards.
        key = int(rng.integers(0, _PLACEMENT_KEY_LIMIT))
        translate, rotate_xyz, scale = compose_parent_space_transform(
            origin,
            target.parent_world,
            settings,
            placement_rng(key, settings.seed),
            stage_up,
            asset.up_axis,
            asset_scale_factor(asset, settings, stage_meters_per_unit),
        )
        if asset.path not in relative_asset_paths:
            relative_asset_paths[asset.path] = make_relative_asset_path(layer, asset.path)
        records.append(
            PlacementRecord(
                container_path=container_path,
                prim_name=new_placement_name(),
                asset_rel_path=relative_asset_paths[asset.path],
                asset_abs_path=asset.path,
                translate=_as_float_triple(translate),
                rotate_xyz=_as_float_triple(rotate_xyz),
                scale=_as_float_triple(scale),
                brush_id=settings.preset_name,
                origin=origin.to_values(),
                key=key,
            )
        )
    return records


def _ensure_container_spec(layer: Sdf.Layer, container_path: Sdf.Path, brush_id: str) -> Sdf.PrimSpec:
    """Return the container spec at ``container_path``, creating or completing it as a marked ``def Xform``."""
    spec = layer.GetPrimAtPath(container_path)
    if spec is None:
        spec = Sdf.CreatePrimInLayer(layer, container_path)
    if spec.specifier != Sdf.SpecifierDef:
        spec.specifier = Sdf.SpecifierDef
    if not spec.typeName:
        spec.typeName = _XFORM_TYPE_NAME
    if IS_REMIX_SCATTER_ATTR not in spec.attributes:
        _set_attribute(spec, IS_REMIX_SCATTER_ATTR, Sdf.ValueTypeNames.Bool, True, custom=True)
    if brush_id and SCATTER_BRUSH_ID_ATTR not in spec.attributes:
        _set_attribute(spec, SCATTER_BRUSH_ID_ATTR, Sdf.ValueTypeNames.String, brush_id, custom=True)
    return spec


def _set_attribute(
    spec: Sdf.PrimSpec,
    name: str,
    type_name: Sdf.ValueTypeName,
    value: Any,
    custom: bool = False,
    variability: Sdf.Variability = Sdf.VariabilityVarying,
) -> Sdf.AttributeSpec:
    """Author ``value`` as the default of attribute ``name`` on ``spec``, creating the attribute spec when missing."""
    attr_spec = spec.attributes.get(name)
    if attr_spec is None:
        attr_spec = Sdf.AttributeSpec(spec, name, type_name, variability)
    if custom:
        attr_spec.custom = True
    attr_spec.default = value
    return attr_spec


def _set_attribute_if_absent(
    spec: Sdf.PrimSpec, name: str, type_name: Sdf.ValueTypeName, value: Any
) -> Sdf.AttributeSpec | None:
    """Author ``value`` on ``spec`` only when the layer has no opinion for ``name`` yet.

    Unlike :func:`_set_attribute` this never replaces an existing value, which is what keeps the brush from clobbering
    a texture the user assigned by hand.
    """
    if name in spec.attributes:
        return None
    attr_spec = Sdf.AttributeSpec(spec, name, type_name, Sdf.VariabilityVarying)
    attr_spec.default = value
    return attr_spec


def _group_by_parent(prim_paths: Sequence[str]) -> dict[Sdf.Path, list[str]]:
    """Return the child names of ``prim_paths`` grouped by parent path, both in first-seen order."""
    grouped: dict[Sdf.Path, list[str]] = {}
    for prim_path in prim_paths:
        path = Sdf.Path(prim_path)
        grouped.setdefault(path.GetParentPath(), []).append(path.name)
    return grouped


def _covers_children(spec: Sdf.PrimSpec, names: Sequence[str]) -> bool:
    """Whether ``names`` holds every child ``spec`` has, so the whole spec can be handled in one edit."""
    wanted = set(names)
    children = spec.nameChildren
    return len(children) <= len(wanted) and wanted.issuperset(children.keys())


def _is_removable_container(spec: Sdf.PrimSpec) -> bool:
    """Whether ``spec`` is a scatter container the brush may delete once it holds nothing of its own."""
    return spec.name.startswith(CONTAINER_PREFIX) and not spec.hasReferences


def _delete_spec(layer: Sdf.Layer, spec: Sdf.PrimSpec) -> None:
    """Delete one prim spec, with everything beneath it, from its parent."""
    parent = spec.nameParent or layer.pseudoRoot
    del parent.nameChildren[spec.name]


def _copy_spec(src_layer: Sdf.Layer, dst_layer: Sdf.Layer, path: Sdf.Path) -> None:
    """Copy the spec at ``path``, with everything beneath it, between two layers at the same path."""
    Sdf.CreatePrimInLayer(dst_layer, path)
    Sdf.CopySpec(src_layer, path, dst_layer, path)


def _copy_container_definition(src_layer: Sdf.Layer, dst_layer: Sdf.Layer, container_path: Sdf.Path) -> None:
    """Copy the specifier, type and attributes of a scatter container spec without touching its children."""
    src = src_layer.GetPrimAtPath(container_path)
    dst = dst_layer.GetPrimAtPath(container_path)
    if src is None or dst is None or not src.name.startswith(CONTAINER_PREFIX):
        return
    if src.specifier == Sdf.SpecifierDef and dst.specifier != Sdf.SpecifierDef:
        dst.specifier = Sdf.SpecifierDef
    if src.typeName and not dst.typeName:
        dst.typeName = src.typeName
    for name, attr_spec in src.attributes.items():
        if name in dst.attributes:
            continue
        Sdf.CopySpec(src_layer, attr_spec.path, dst_layer, attr_spec.path)
