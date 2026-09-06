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
    "PlacementRecord",
    "author_placements",
    "container_path_for",
    "erase_candidates",
    "existing_placement_points",
    "generate_flood",
    "generate_stamp",
    "make_relative_asset_path",
    "new_placement_name",
    "remove_placements",
    "restore_prims",
    "snapshot_prims",
]

import math
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
)
from .geometry import MeshSurfaceCache, SurfaceSample, area_weighted_triangle_samples
from .sampling import (
    PaddingIndex,
    choose_asset,
    compose_parent_space_transform,
    falloff_weight,
    sample_disk,
    tangent_basis,
)
from .settings import EraseScope, ScatterAssetEntry, ScatterBrushSettings, UpAxis
from .targets import ScatterTarget

_LOG_PREFIX = "[lightspeed.trex.scatter.core]"
_XFORM_TYPE_NAME = "Xform"
_XFORM_OP_ORDER = ("xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale")


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
        )


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
    layer: Sdf.Layer,
) -> list[PlacementRecord]:
    """Generate the placements of one brush stamp centered on ``center``.

    ``ceil(density)`` candidates are drawn in the brush disk, accepted against the falloff curve scaled by the brush
    strength, projected back onto the capture mesh, filtered by padding and finally turned into parent-space records.

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
    return _build_records(
        positions, normals, valid, target, settings, assets, rng, padding_index, heading_world, stage_up, layer
    )


def generate_flood(
    cache: MeshSurfaceCache,
    target: ScatterTarget,
    settings: ScatterBrushSettings,
    assets: Sequence[ScatterAssetEntry],
    rng: np.random.Generator,
    padding_index: PaddingIndex,
    stage_up: UpAxis,
    layer: Sdf.Layer,
    max_count: int,
) -> list[PlacementRecord]:
    """Generate placements covering the whole capture mesh of ``target``.

    The count is ``density`` per brush-disk area, scaled to the mesh area and capped by ``max_count``.

    Args:
        cache: Mesh cache providing the geometry of ``target.mesh_path``.
        target: Resolved scatter target.
        settings: Brush settings.
        assets: Brush assets to choose from.
        rng: Random generator driving every random decision of the flood.
        padding_index: Spatial index of already placed points; accepted points are added to it.
        stage_up: Up axis of the stage.
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
    valid = np.ones(len(positions), dtype=bool)
    return _build_records(
        positions, normals, valid, target, settings, assets, rng, padding_index, None, stage_up, layer
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
    center = _as_array(center_world)
    wanted = {_normalized_url(path) for path in brush_asset_paths} if scope == EraseScope.BRUSH_ASSETS else None
    candidates: list[Sdf.Path] = []
    for prim in _iter_placement_prims(stage, target.prototype_root):
        if np.linalg.norm(_placement_world_position(prim, target.parent_world) - center) > radius:
            continue
        if wanted is not None:
            asset_path = _resolve_placement_asset(prim)
            if asset_path is None or _normalized_url(asset_path) not in wanted:
                continue
        candidates.append(prim.GetPath())
    return candidates


def author_placements(layer: Sdf.Layer, records: Sequence[PlacementRecord]) -> list[str]:
    """Author the placement prims described by ``records`` into ``layer`` with pure Sdf edits.

    Each container is ensured as a ``def Xform`` carrying the scatter marker and brush id; each placement is a
    ``def Xform`` with a prepended reference, the Remix reference and scatter markers, the asset path and explicit
    translate / rotateXYZ / scale ops. Existing prim names are left untouched.

    Args:
        layer: Layer to author in.
        records: Placements to author.

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
            _set_attribute(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*record.translate))
            _set_attribute(spec, "xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*record.rotate_xyz))
            _set_attribute(spec, "xformOp:scale", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*record.scale))
            _set_attribute(
                spec,
                "xformOpOrder",
                Sdf.ValueTypeNames.TokenArray,
                Vt.TokenArray(list(_XFORM_OP_ORDER)),
                variability=Sdf.VariabilityUniform,
            )
            created.append(str(spec.path))
    return created


def remove_placements(layer: Sdf.Layer, prim_paths: Sequence[str], remove_empty_containers: bool = True) -> None:
    """Delete the prim specs at ``prim_paths`` from ``layer``.

    Args:
        layer: Layer to edit.
        prim_paths: Placement prim paths; paths without a spec in ``layer`` are ignored.
        remove_empty_containers: Also drop scatter containers left without children and references.
    """
    with Sdf.ChangeBlock():
        parent_paths: list[Sdf.Path] = []
        for prim_path in prim_paths:
            spec = layer.GetPrimAtPath(Sdf.Path(prim_path))
            if spec is None:
                continue
            parent = spec.nameParent or layer.pseudoRoot
            parent_path = parent.path
            name = spec.name
            if name in parent.nameChildren:
                del parent.nameChildren[name]
            if parent_path not in parent_paths:
                parent_paths.append(parent_path)
        if not remove_empty_containers:
            return
        for container_path in parent_paths:
            container = layer.GetPrimAtPath(container_path)
            if container is None or not container.name.startswith(CONTAINER_PREFIX):
                continue
            if len(container.nameChildren) > 0 or container.hasReferences:
                continue
            container_parent = container.nameParent or layer.pseudoRoot
            del container_parent.nameChildren[container.name]


def snapshot_prims(layer: Sdf.Layer, prim_paths: Sequence[str], into: Sdf.Layer | None = None) -> Sdf.Layer:
    """Copy the specs at ``prim_paths`` (and their containers' own definition) into a snapshot layer.

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
        for prim_path in prim_paths:
            path = Sdf.Path(prim_path)
            if layer.GetPrimAtPath(path) is None:
                continue
            Sdf.CreatePrimInLayer(snapshot, path)
            Sdf.CopySpec(layer, path, snapshot, path)
            _copy_container_definition(layer, snapshot, path.GetParentPath())
    return snapshot


def restore_prims(layer: Sdf.Layer, snapshot: Sdf.Layer, prim_paths: Sequence[str]) -> None:
    """Copy the specs at ``prim_paths`` from ``snapshot`` back into ``layer``, re-defining removed containers.

    Args:
        layer: Layer to restore into.
        snapshot: Layer produced by :func:`snapshot_prims`.
        prim_paths: Placement prim paths; paths missing from ``snapshot`` are skipped.
    """
    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            path = Sdf.Path(prim_path)
            if snapshot.GetPrimAtPath(path) is None:
                continue
            Sdf.CreatePrimInLayer(layer, path)
            Sdf.CopySpec(snapshot, path, layer, path)
            _copy_container_definition(snapshot, layer, path.GetParentPath())


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
        padding_index.add(position)
        translate, rotate_xyz, scale = compose_parent_space_transform(
            Gf.Vec3d(*(float(value) for value in position)),
            Gf.Vec3d(*(float(value) for value in normal)),
            target.parent_world,
            settings,
            rng,
            stage_up,
            asset.up_axis,
            heading_world,
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
