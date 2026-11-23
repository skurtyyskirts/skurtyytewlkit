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

__all__ = ["ScatterGroupsCore", "group_settings"]

import asyncio
import hashlib
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path

import carb
import omni.kit.commands
import omni.kit.undo
import omni.usd
from lightspeed.common.constants import MATERIAL_INPUTS_DIFFUSE_TEXTURE, MESH_NAME_PREFIX, ROOTNODE_MESHES
from lightspeed.trex.utils.common.asset_utils import is_asset_ingested
from omni.flux.utils.common.materials import get_bound_material
from pxr import Sdf, Usd, UsdShade
from pydantic import ValidationError

from .assets import (
    asset_comparison_key,
    normalize_asset_path,
    read_asset_meters_per_unit,
    read_asset_natural_extent,
    read_asset_up_axis,
)
from .constants import GROUP_PRESET_PREFIX, SCATTER_GROUPS_LAYER_DATA_KEY, SURFACE_PREVIEWS_SUBDIRECTORY
from .controller import flood_targets
from .geometry import MeshSurfaceCache
from .group_models import (
    ScatterAmount,
    ScatterGroupAssetModel,
    ScatterGroupCreateRequestModel,
    ScatterGroupListResponseModel,
    ScatterGroupModel,
    ScatterGroupResponseModel,
    ScatterGroupTuningModel,
    ScatterGroupUpdateRequestModel,
    ScatterSurfaceListResponseModel,
    ScatterSurfaceModel,
)
from .materials import build_materials_payload, resolve_asset_texture_set
from .placement import make_relative_asset_path
from .previews import write_contact_sheets, write_texture_preview
from .settings import ScatterAssetEntry, ScatterBrushSettings
from .targets import build_prototype_target, find_capture_mesh, get_prototype_root, instance_count

_LOG_PREFIX = "[lightspeed.trex.scatter.core]"

# Each amount is a density per 50-unit brush disk and a cap on the whole group. Heavy stops at 10,000 placements, the
# most one stroke is budgeted to render; light and medium leave room for the rest of the scene.
_AMOUNT_SETTINGS = {
    ScatterAmount.LIGHT: {"density": 3.0, "flood_max_instances": 1500},
    ScatterAmount.MEDIUM: {"density": 10.0, "flood_max_instances": 4000},
    ScatterAmount.HEAVY: {"density": 30.0, "flood_max_instances": 10000},
}
# Tuning fields named after what they mean to a request rather than after the brush setting they set.
_TUNING_SETTING_NAMES = {"max_instances": "flood_max_instances"}


@dataclass(frozen=True)
class _Surface:
    """A captured material, the albedo texture it renders with and the prototypes bound to it."""

    material_path: str
    texture_path: str | None
    prototypes: tuple[Sdf.Path, ...]


def group_settings(group: ScatterGroupModel) -> ScatterBrushSettings:
    """Build the brush settings a scatter group floods with.

    The group starts from the default brush settings, not from the user's current ones, so retuning the brush in the
    Scatter window never changes how a group scatters. The preset name gives the group its own containers.

    Args:
        group: The group to build settings for.

    Returns:
        The validated settings.

    Raises:
        ValueError: When the tuning contradicts itself, such as ``size_min`` above ``size_max``.
    """
    values = {
        "preset_name": f"{GROUP_PRESET_PREFIX}{group.name}",
        "randomize_seed": False,
        "seed": group.seed,
        **_AMOUNT_SETTINGS[group.amount],
    }
    for field, value in group.tuning.model_dump(exclude_none=True).items():
        values[_TUNING_SETTING_NAMES.get(field, field)] = value
    try:
        return ScatterBrushSettings.model_validate(values)
    except ValidationError as error:
        raise ValueError(f"Invalid scatter group tuning: {error}") from error


class ScatterGroupsCore:
    """Create, retune, list and remove scatter groups in one USD context, and list the surfaces they can fill."""

    def __init__(self, context_name: str = ""):
        """
        Args:
            context_name: Name of the USD context the groups are authored in.
        """
        self._context_name = context_name

    async def get_surfaces(self, previews: bool = True) -> ScatterSurfaceListResponseModel:
        """List every captured material with the albedo texture it renders with and the meshes bound to it.

        Args:
            previews: Write a PNG preview per texture and numbered contact sheets of them all.

        Returns:
            The surfaces, numbered in material path order as their contact sheet tiles are.

        Raises:
            ValueError: When no stage is open.
        """
        stage = self._get_stage()
        surfaces = _collect_surfaces(stage)
        preview_paths: dict[int, Path] = {}
        sheets: list[Path] = []
        if previews and surfaces:
            numbered = [(number, surface.texture_path) for number, surface in enumerate(surfaces, start=1)]
            preview_paths, sheets = await asyncio.to_thread(_write_previews, _previews_directory(stage), numbered)
        return ScatterSurfaceListResponseModel(
            surfaces=[
                ScatterSurfaceModel(
                    number=number,
                    material_path=surface.material_path,
                    texture_path=surface.texture_path,
                    preview_path=preview_paths[number].as_posix() if number in preview_paths else None,
                    mesh_prim_paths=[str(path) for path in surface.prototypes],
                    instance_count=sum(instance_count(stage, path) for path in surface.prototypes),
                )
                for number, surface in enumerate(surfaces, start=1)
            ],
            contact_sheet_paths=[sheet.as_posix() for sheet in sheets],
        )

    def get_groups(self) -> ScatterGroupListResponseModel:
        """List the scatter groups recorded in the layers of the stage.

        Raises:
            ValueError: When no stage is open.
        """
        stage = self._get_stage()
        return ScatterGroupListResponseModel(groups=[_response(layer, group) for layer, group in _read_groups(stage)])

    def create_group(self, request: ScatterGroupCreateRequestModel) -> ScatterGroupResponseModel:
        """Create a scatter group in the edit target layer and fill its surfaces, in one undo step.

        Args:
            request: The group to create.

        Returns:
            The group as recorded, with the number of placements it authored.

        Raises:
            ValueError: When no stage is open, the name is taken, or a target or asset cannot be used.
        """
        stage = self._get_stage()
        slug = _group_slug(request.name)
        for _, existing in _read_groups(stage):
            if _group_slug(existing.name) == slug:
                raise ValueError(
                    f"A scatter group named '{existing.name}' already exists; update it or choose another name"
                )
        group = ScatterGroupModel(
            name=request.name,
            amount=request.amount,
            tuning=request.tuning,
            assets=request.assets,
            material_paths=request.material_paths,
            prim_paths=request.prim_paths,
            target_prim_paths=[],
            seed=secrets.randbits(31) if request.seed is None else request.seed,
            placement_count=0,
            max_instances=0,
        )
        return self._scatter(stage, stage.GetEditTarget().GetLayer(), group, replace=False)

    def update_group(self, name: str, request: ScatterGroupUpdateRequestModel) -> ScatterGroupResponseModel:
        """Change a scatter group and scatter it again in place, in one undo step.

        Args:
            name: Name of the group.
            request: The changes; anything left out keeps its current value.

        Returns:
            The group as recorded after the change.

        Raises:
            LookupError: When no group has that name.
            ValueError: When no stage is open, or the change leaves no assets or no surfaces, or cannot be used.
        """
        stage = self._get_stage()
        layer, group = _find_group(stage, name)
        tuning = group.tuning.model_dump(exclude_none=True)
        if request.amount is not None:
            tuning.pop("density", None)
            tuning.pop("max_instances", None)
        if request.tuning is not None:
            tuning.update(request.tuning.model_dump(exclude_none=True))

        removed = {asset_comparison_key(normalize_asset_path(path)) for path in request.remove_asset_paths}
        assets = {
            asset_comparison_key(normalize_asset_path(asset.path)): asset
            for asset in group.assets
            if asset_comparison_key(normalize_asset_path(asset.path)) not in removed
        }
        for asset in request.add_assets:
            assets[asset_comparison_key(normalize_asset_path(asset.path))] = asset
        if not assets:
            raise ValueError(f"Scatter group '{group.name}' needs at least one asset")

        material_paths = group.material_paths if request.material_paths is None else request.material_paths
        prim_paths = group.prim_paths if request.prim_paths is None else request.prim_paths
        if not material_paths and not prim_paths:
            raise ValueError(f"Scatter group '{group.name}' needs at least one material path or prim path")

        changed = group.model_copy(
            update={
                "amount": group.amount if request.amount is None else request.amount,
                "tuning": ScatterGroupTuningModel.model_validate(tuning),
                "assets": list(assets.values()),
                "material_paths": material_paths,
                "prim_paths": prim_paths,
                "seed": secrets.randbits(31) if request.reseed else group.seed,
            }
        )
        return self._scatter(stage, layer, changed, replace=True)

    def remove_group(self, name: str) -> ScatterGroupResponseModel:
        """Remove a scatter group's placements and record, in one undo step.

        Args:
            name: Name of the group.

        Returns:
            The group as it was recorded before it was removed.

        Raises:
            LookupError: When no group has that name.
            ValueError: When no stage is open.
        """
        stage = self._get_stage()
        layer, group = _find_group(stage, name)
        with omni.kit.undo.group():
            omni.kit.commands.execute(
                "ScatterRemoveBrushCommand", layer_identifier=layer.identifier, brush_slug=_group_slug(group.name)
            )
            omni.kit.commands.execute(
                "ScatterSetGroupRecordCommand", layer_identifier=layer.identifier, group_name=group.name, record=None
            )
        return _response(layer, group)

    def _get_stage(self) -> Usd.Stage:
        """Return the stage of the context.

        Raises:
            ValueError: When no stage is open.
        """
        stage = omni.usd.get_context(self._context_name).get_stage()
        if stage is None:
            raise ValueError("Open a project before scattering")
        return stage

    def _scatter(
        self, stage: Usd.Stage, layer: Sdf.Layer, group: ScatterGroupModel, replace: bool
    ) -> ScatterGroupResponseModel:
        """Resolve a group's surfaces and assets, then flood them and record the group in one undo step.

        Everything that can be refused is checked before the first edit, so a refused request changes nothing.

        Args:
            stage: Stage to scatter on.
            layer: Layer the group authors in.
            group: The group to scatter.
            replace: Remove the group's current placements first.

        Returns:
            The group as recorded.
        """
        settings = group_settings(group)
        prototypes = _resolve_prototypes(stage, group.material_paths, group.prim_paths)
        targets = [target for path in prototypes if (target := build_prototype_target(stage, path)) is not None]
        if not targets:
            raise ValueError("None of the given surfaces holds a captured mesh to scatter on")
        assets = [_prepare_asset(asset) for asset in group.assets]
        materials = build_materials_payload(layer, assets) if settings.assign_textures else {}

        cache = MeshSurfaceCache(lambda: stage)
        try:
            with omni.kit.undo.group():
                if replace:
                    omni.kit.commands.execute(
                        "ScatterRemoveBrushCommand", layer_identifier=layer.identifier, brush_slug=settings.slug()
                    )
                outcome = flood_targets(
                    stage, self._context_name, layer, cache, targets, settings, assets, materials, group.seed, 0
                )
                recorded = group.model_copy(
                    update={
                        "target_prim_paths": [str(target.prototype_root) for target in targets],
                        "placement_count": outcome.placed,
                        "max_instances": settings.flood_max_instances,
                        # Recorded, because a count below the cap otherwise reads as a surface that is simply full.
                        "prim_limited": outcome.limited,
                    }
                )
                omni.kit.commands.execute(
                    "ScatterSetGroupRecordCommand",
                    layer_identifier=layer.identifier,
                    group_name=recorded.name,
                    record=_record_text(layer, recorded),
                )
        finally:
            cache.destroy()
        return _response(layer, recorded)


def _group_slug(name: str) -> str:
    """Return the preset slug that names a group's containers and instancers."""
    return ScatterBrushSettings(preset_name=f"{GROUP_PRESET_PREFIX}{name}").slug()


def _read_groups(stage: Usd.Stage) -> list[tuple[Sdf.Layer, ScatterGroupModel]]:
    """Return every group recorded in the layer stack, by layer and then by name; unreadable records are logged."""
    groups = []
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        records = layer.customLayerData.get(SCATTER_GROUPS_LAYER_DATA_KEY, {})
        for name in sorted(records):
            try:
                group = ScatterGroupModel.model_validate_json(records[name])
            except ValidationError as error:
                carb.log_warn(
                    f"{_LOG_PREFIX} Skipping unreadable scatter group '{name}' in {layer.identifier}: {error}"
                )
                continue
            assets = [
                asset.model_copy(update={"path": normalize_asset_path(layer.ComputeAbsolutePath(asset.path))})
                for asset in group.assets
            ]
            groups.append((layer, group.model_copy(update={"name": name, "assets": assets})))
    return groups


def _find_group(stage: Usd.Stage, name: str) -> tuple[Sdf.Layer, ScatterGroupModel]:
    """Return the layer and record of the group named ``name``.

    Raises:
        LookupError: When no group has that name.
    """
    for layer, group in _read_groups(stage):
        if group.name == name:
            return layer, group
    raise LookupError(f"No scatter group is named '{name}'; list them with remix_get_scatter_groups")


def _record_text(layer: Sdf.Layer, group: ScatterGroupModel) -> str:
    """Serialize a group for the custom layer data, with asset paths relative to the layer so the mod stays portable."""
    assets = [asset.model_copy(update={"path": make_relative_asset_path(layer, asset.path)}) for asset in group.assets]
    return group.model_copy(update={"assets": assets}).model_dump_json()


def _response(layer: Sdf.Layer, group: ScatterGroupModel) -> ScatterGroupResponseModel:
    """Describe a recorded group together with the layer it is authored in."""
    return ScatterGroupResponseModel(
        **group.model_dump(),
        layer_id=layer.identifier,
        capped=group.max_instances > 0 and group.placement_count >= group.max_instances,
    )


def _collect_surfaces(stage: Usd.Stage) -> list[_Surface]:
    """Return every material that captured mesh prototypes are bound to, in material path order."""
    meshes = stage.GetPrimAtPath(ROOTNODE_MESHES)
    if not meshes.IsValid():
        return []
    prototypes: dict[str, list[Sdf.Path]] = {}
    materials: dict[str, UsdShade.Material] = {}
    for prototype in meshes.GetChildren():
        if not prototype.GetName().startswith(MESH_NAME_PREFIX):
            continue
        mesh_path = find_capture_mesh(stage, prototype.GetPath())
        if mesh_path is None:
            continue
        material = get_bound_material(stage.GetPrimAtPath(mesh_path))
        if material is None:
            continue
        material_path = str(material.GetPath())
        materials[material_path] = material
        prototypes.setdefault(material_path, []).append(prototype.GetPath())
    return [_Surface(path, _albedo_texture(materials[path]), tuple(prototypes[path])) for path in sorted(prototypes)]


def _albedo_texture(material: UsdShade.Material) -> str | None:
    """Return the resolved albedo texture of the first shader in ``material`` that binds one."""
    input_name = MATERIAL_INPUTS_DIFFUSE_TEXTURE.removeprefix("inputs:")
    for prim in Usd.PrimRange(material.GetPrim()):
        shader = UsdShade.Shader(prim)
        if not shader:
            continue
        shader_input = shader.GetInput(input_name)
        value = shader_input.Get() if shader_input else None
        if value and value.path:
            return value.resolvedPath or None
    return None


def _resolve_prototypes(stage: Usd.Stage, material_paths: list[str], prim_paths: list[str]) -> list[Sdf.Path]:
    """Return the prototypes a group fills: those bound to its materials, then those of its prims, each once.

    Raises:
        ValueError: When a material has no captured mesh bound to it, or a prim is outside the capture.
    """
    surfaces = {surface.material_path: surface.prototypes for surface in _collect_surfaces(stage)}
    prototypes: list[Sdf.Path] = []
    for material_path in material_paths:
        if material_path not in surfaces:
            raise ValueError(
                f"No captured mesh is bound to '{material_path}'; list the surfaces with remix_get_scatter_surfaces"
            )
        prototypes.extend(surfaces[material_path])
    for prim_path in prim_paths:
        prototype = get_prototype_root(prim_path)
        if prototype is None or not stage.GetPrimAtPath(prototype).IsValid():
            raise ValueError(f"'{prim_path}' is not a captured mesh or instance")
        prototypes.append(prototype)
    return list(dict.fromkeys(prototypes))


def _prepare_asset(asset: ScatterGroupAssetModel) -> ScatterAssetEntry:
    """Measure an asset and resolve its textures the way adding it to the brush palette does.

    Raises:
        ValueError: When the file is not an ingested asset or holds no geometry.
    """
    path = normalize_asset_path(asset.path)
    if not is_asset_ingested(path, ignore_invalid_paths=False):
        raise ValueError(f"'{path}' is not an ingested asset; ingest it with remix_ingest_model_asset first")
    extent = read_asset_natural_extent(path)
    if extent == (0.0, 0.0, 0.0):
        raise ValueError(f"'{path}' cannot be read or holds no geometry")
    entry = ScatterAssetEntry(
        path=path,
        weight=asset.weight,
        up_axis=read_asset_up_axis(path),
        meters_per_unit=read_asset_meters_per_unit(path),
        natural_extent=extent,
    )
    entry.textures = resolve_asset_texture_set(path)
    return entry


def _previews_directory(stage: Usd.Stage) -> Path:
    """Return the preview directory of a stage, one per project so two projects never share stale previews."""
    digest = hashlib.sha1(stage.GetRootLayer().identifier.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / SURFACE_PREVIEWS_SUBDIRECTORY / digest


def _write_previews(directory: Path, numbered: list[tuple[int, str | None]]) -> tuple[dict[int, Path], list[Path]]:
    """Write one preview per texture and the contact sheets, replacing what an earlier listing left.

    Args:
        directory: The stage's preview directory.
        numbered: ``(number, texture path)`` per surface.

    Returns:
        The preview of each surface number that has one, and the contact sheets.
    """
    if directory.is_dir():
        for stale in directory.glob("*.png"):
            stale.unlink()
    previews: dict[int, Path] = {}
    written: dict[str, Path] = {}
    for number, texture_path in numbered:
        if texture_path is None:
            continue
        if texture_path not in written:
            output = directory / f"surface_{number:04d}.png"
            if not write_texture_preview(texture_path, output):
                continue
            written[texture_path] = output
        previews[number] = written[texture_path]
    sheets = write_contact_sheets([(number, previews.get(number)) for number, _ in numbered], directory)
    return previews, sheets
