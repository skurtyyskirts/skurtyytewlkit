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
    "ScatterFloodCommand",
    "ScatterInstanceCommand",
    "ScatterReapplyCommand",
    "ScatterRemoveBrushCommand",
    "ScatterSetGroupRecordCommand",
    "ScatterSplitCommand",
    "ScatterStrokeCommand",
    "ScatterStrokeKind",
]

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

import carb
import omni.kit.commands
import omni.usd
from pxr import Sdf, Usd

from .constants import SCATTER_GROUPS_LAYER_DATA_KEY
from .instancing import InstancingEdit, brush_spec_paths, merge_loose_placements, split_instancers
from .placement import (
    PlacementRecord,
    PlacementTransform,
    author_placement_transforms,
    author_placements,
    prune_inert_overs,
    remove_placements,
    restore_prims,
    snapshot_prims,
)

_LOG_PREFIX = "[lightspeed.trex.scatter.core]"


class ScatterStrokeKind(StrEnum):
    """What a stroke did: placed new prims or erased existing ones."""

    PLACE = "PLACE"
    ERASE = "ERASE"


class ScatterStrokeCommand(omni.kit.commands.Command):
    """One brush stroke.

    PLACE: the records were already authored live while the mouse moved, so the first ``do()`` is a no-op. ``undo()``
    snapshots the created prims as they stand and removes them, and redo restores that snapshot. Snapshotting at undo
    time rather than replaying the records is what lets a stroke that was tuned while it was the brush preview come
    back tuned: those edits are folded into the stroke instead of being undo steps of their own. ERASE: the prims were
    already removed live and a snapshot of their specs is given, so the first ``do()`` is a no-op, ``undo()`` restores
    them from the snapshot and redo removes them again. With ``already_applied=False`` the first ``do()`` performs the
    edit itself.

    A PLACE stroke with ``instance_others`` also merges every other loose placement of its layer into the instancer of
    its patch when it is first done: the stroke before it has stopped being the preview, so nothing edits it as prims
    any more. The merge belongs to this stroke's undo step, so undoing the stroke gives that earlier stroke back as
    prims as well. An ERASE stroke undoes and repeats the instancer entries it erased the same way.
    """

    def __init__(
        self,
        context_name: str,
        layer_identifier: str,
        kind: str,
        records: Sequence[Mapping[str, Any] | PlacementRecord] | None = None,
        prim_paths: Sequence[str] | None = None,
        snapshot_layer: Sdf.Layer | None = None,
        already_applied: bool = True,
        materials: Mapping[str, Any] | None = None,
        instance_others: bool = False,
        erased_entries: Sequence[InstancingEdit] | None = None,
    ):
        """Create the command.

        Args:
            context_name: Name of the USD context the stroke was painted in.
            layer_identifier: Identifier of the layer holding the placements (the stroke's edit layer).
            kind: ``"PLACE"`` or ``"ERASE"`` (see :class:`ScatterStrokeKind`).
            records: PLACE only: the placement records (``PlacementRecord.to_dict()`` payloads or records).
            prim_paths: ERASE only: the erased prim paths.
            snapshot_layer: ERASE only: anonymous layer holding copies of the erased specs.
            already_applied: True when the edit already happened live and the first ``do()`` must not repeat it.
            materials: PLACE only: the shader overrides to author, kept on the command so a redo still has them once
                the asset has left the brush palette.
            instance_others: PLACE only: merge every other loose placement of the layer into its patch's instancer.
            erased_entries: ERASE only: the instancer entries the stroke erased, already removed when
                ``already_applied`` is True.
        """
        self._context_name = context_name
        self._layer_identifier = layer_identifier
        self._kind = ScatterStrokeKind(kind)
        self._records = _coerce_records(records)
        self._materials = dict(materials or {})
        self._prim_paths = [str(path) for path in prim_paths or []]
        self._snapshot_layer = snapshot_layer
        self._skip_next_do = already_applied
        self._created_paths = [record.prim_path for record in self._records] if already_applied else []
        # PLACE only: the stroke as it stood when it was last undone, which a redo restores.
        self._undone_layer: Sdf.Layer | None = None
        self._instance_others = instance_others
        # PLACE only: the merges the first do() made, which undo reverts and redo repeats as they were.
        self._instancing: list[InstancingEdit] | None = None
        self._erased_entries = list(erased_entries or [])

    @property
    def kind(self) -> ScatterStrokeKind:
        """Whether the stroke placed or erased."""
        return self._kind

    @property
    def layer_identifier(self) -> str:
        """Identifier of the layer the stroke edits."""
        return self._layer_identifier

    @property
    def created_paths(self) -> list[str]:
        """PLACE only: the prim paths the stroke created; empty for an erase or before the first ``do()``."""
        return list(self._created_paths)

    @property
    def instanced_count(self) -> int:
        """PLACE only: how many earlier placements the stroke merged into instancers."""
        return sum(len(edit.block) for edit in self._instancing or ())

    def do(self) -> list[str]:
        """Apply the stroke unless it was already applied live; returns the affected prim paths."""
        if self._skip_next_do:
            self._skip_next_do = False
            if self._kind == ScatterStrokeKind.PLACE:
                self._instance(_find_layer(self._layer_identifier))
                return list(self._created_paths)
            return list(self._prim_paths)
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            return []
        if self._kind == ScatterStrokeKind.PLACE:
            if self._undone_layer is not None:
                restore_prims(layer, self._undone_layer, self._created_paths)
                self._undone_layer = None
            else:
                self._created_paths = author_placements(layer, self._records, self._materials)
            self._instance(layer)
            return list(self._created_paths)
        if self._snapshot_layer is None:
            self._snapshot_layer = snapshot_prims(layer, self._prim_paths)
        remove_placements(layer, self._prim_paths)
        for edit in self._erased_entries:
            edit.uninstance(layer)
        return list(self._prim_paths)

    def undo(self) -> None:
        """Revert the stroke: remove placed prims or restore erased prims from the snapshot."""
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            return
        if self._kind == ScatterStrokeKind.PLACE:
            self._undone_layer = snapshot_prims(layer, self._created_paths)
            remove_placements(layer, self._created_paths)
            for edit in reversed(self._instancing or ()):
                edit.uninstance(layer)
            return
        for edit in reversed(self._erased_entries):
            edit.instance(layer)
        if self._snapshot_layer is None:
            if self._prim_paths:
                carb.log_warn(f"{_LOG_PREFIX} No snapshot available to restore the erased scatter placements")
            return
        restore_prims(layer, self._snapshot_layer, self._prim_paths)

    def _instance(self, layer: Sdf.Layer | None) -> None:
        """Merge the other loose placements of the layer the first time, and repeat exactly that merge on redo."""
        if not self._instance_others or layer is None:
            return
        if self._instancing is None:
            self._instancing = merge_loose_placements(
                layer, _context_stage(self._context_name), exclude=self._created_paths
            )
            return
        for edit in self._instancing:
            edit.instance(layer)


class ScatterFloodCommand(omni.kit.commands.Command):
    """Author a batch of flood placements; undo removes them again.

    With ``instance`` the flood is merged straight into the instancers of its patches together with every other loose
    placement of the layer: nothing previews a flood, so nothing needs it as prims.
    """

    def __init__(
        self,
        context_name: str,
        layer_identifier: str,
        records: Sequence[Mapping[str, Any] | PlacementRecord],
        materials: Mapping[str, Any] | None = None,
        instance: bool = False,
    ):
        """Create the command.

        Args:
            context_name: Name of the USD context the flood was requested in.
            layer_identifier: Identifier of the layer to author the placements in.
            records: The placement records (``PlacementRecord.to_dict()`` payloads or records).
            materials: The shader overrides to author, kept on the command so a redo still has them once the asset
                has left the brush palette.
            instance: Merge the flood and every other loose placement of the layer into their patches' instancers.
        """
        self._context_name = context_name
        self._layer_identifier = layer_identifier
        self._records = _coerce_records(records)
        self._materials = dict(materials or {})
        self._created_paths: list[str] = []
        self._instance = instance
        self._instancing: list[InstancingEdit] | None = None

    def do(self) -> list[str]:
        """Author the placements; returns the created prim paths."""
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            self._created_paths = []
            return []
        self._created_paths = author_placements(layer, self._records, self._materials)
        if self._instance:
            if self._instancing is None:
                self._instancing = merge_loose_placements(layer, _context_stage(self._context_name))
            else:
                for edit in self._instancing:
                    edit.instance(layer)
        return list(self._created_paths)

    def undo(self) -> None:
        """Remove the placements created by ``do()``."""
        if not self._created_paths:
            return
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            return
        for edit in reversed(self._instancing or ()):
            edit.uninstance(layer)
        remove_placements(layer, self._created_paths)


class ScatterInstanceCommand(omni.kit.commands.Command):
    """Merge loose placements into the instancers of their patches; undo turns them back into prims.

    The brush merges a stroke on its own once the next stroke is painted; this is for the rest: the last stroke of a
    session, scatter painted before instancing existed, and a patch that was split to be edited.
    """

    def __init__(self, context_name: str, layer_identifier: str, prototype_roots: Sequence[str] | None = None):
        """Create the command.

        Args:
            context_name: Name of the USD context the placements belong to.
            layer_identifier: Identifier of the layer holding the placements (the edit target).
            prototype_roots: Prototypes to merge under; None merges every loose placement of the layer.
        """
        self._context_name = context_name
        self._layer_identifier = layer_identifier
        self._prototype_roots = None if prototype_roots is None else [str(path) for path in prototype_roots]
        self._edits: list[InstancingEdit] | None = None

    @property
    def instanced_count(self) -> int:
        """How many placements the command merged."""
        return sum(len(edit.block) for edit in self._edits or ())

    def do(self) -> int:
        """Merge the placements, or repeat the merge exactly on redo; returns how many were merged."""
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            return 0
        if self._edits is None:
            self._edits = merge_loose_placements(
                layer, _context_stage(self._context_name), prototype_roots=self._prototype_roots
            )
        else:
            for edit in self._edits:
                edit.instance(layer)
        return self.instanced_count

    def undo(self) -> None:
        """Turn the merged entries back into the placements they were."""
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            return
        for edit in reversed(self._edits or ()):
            edit.uninstance(layer)


class ScatterSplitCommand(omni.kit.commands.Command):
    """Turn instancers back into loose placements the brush can edit; undo merges them again."""

    def __init__(self, context_name: str, layer_identifier: str, instancer_paths: Sequence[str]):
        """Create the command.

        Args:
            context_name: Name of the USD context the instancers belong to.
            layer_identifier: Identifier of the layer authoring the instancers (the edit target).
            instancer_paths: Prototype-side paths of the instancers to split.
        """
        self._context_name = context_name
        self._layer_identifier = layer_identifier
        self._instancer_paths = [str(path) for path in instancer_paths]
        self._edits: list[InstancingEdit] | None = None

    @property
    def split_paths(self) -> list[str]:
        """The placements the split created; empty before the first ``do()``."""
        return [path for edit in self._edits or () for path in edit.loose_paths]

    def do(self) -> list[str]:
        """Split the instancers, or repeat the split exactly on redo; returns the placement paths created."""
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            return []
        if self._edits is None:
            self._edits = split_instancers(layer, self._instancer_paths)
        else:
            for edit in self._edits:
                edit.uninstance(layer)
        return self.split_paths

    def undo(self) -> None:
        """Merge the split placements back into their instancers."""
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            return
        for edit in reversed(self._edits or ()):
            edit.instance(layer)


class ScatterReapplyCommand(omni.kit.commands.Command):
    """One live edit of placements that already exist.

    The transforms were already authored while the setting was being dragged, so the first ``do()`` is a no-op and
    the whole drag collapses into this single undo step. ``undo()`` writes the transforms back as they were before
    the edit and redo writes the edited ones again. With ``already_applied=False`` the first ``do()`` performs the
    edit itself.
    """

    def __init__(
        self,
        context_name: str,
        layer_identifier: str,
        before: Sequence[Mapping[str, Any] | PlacementTransform],
        after: Sequence[Mapping[str, Any] | PlacementTransform],
        already_applied: bool = True,
    ):
        """Create the command.

        Args:
            context_name: Name of the USD context the placements belong to.
            layer_identifier: Identifier of the layer holding the placements (the edit target).
            before: The transforms the placements had before the edit.
            after: The transforms the edit gave them.
            already_applied: True when the edit already happened live and the first ``do()`` must not repeat it.
        """
        self._context_name = context_name
        self._layer_identifier = layer_identifier
        self._before = _coerce_transforms(before)
        self._after = _coerce_transforms(after)
        self._skip_next_do = already_applied

    def do(self) -> list[str]:
        """Author the edited transforms unless they were already authored live; returns the affected prim paths."""
        if self._skip_next_do:
            self._skip_next_do = False
            return [transform.prim_path for transform in self._after]
        return self._author(self._after)

    def undo(self) -> None:
        """Write the transforms the placements had before the edit back into the layer."""
        self._author(self._before)

    def _author(self, transforms: Sequence[PlacementTransform]) -> list[str]:
        """Author a set of transforms into the command's layer, or nothing when the layer is unavailable."""
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            return []
        return author_placement_transforms(layer, transforms)


class ScatterRemoveBrushCommand(omni.kit.commands.Command):
    """Remove every container and instancer one preset authored in a layer; undo puts them back as they were.

    Whole specs are copied and deleted, so an instancer comes back with its prototypes and per-entry arrays, and a
    container with every loose placement in it.
    """

    def __init__(self, layer_identifier: str, brush_slug: str):
        """Create the command.

        Args:
            layer_identifier: Identifier of the layer holding the preset's scatter.
            brush_slug: Prim-name-safe identifier of the preset (see ``ScatterBrushSettings.slug``).
        """
        self._layer_identifier = layer_identifier
        self._brush_slug = brush_slug
        self._paths: list[Sdf.Path] = []
        self._snapshot: Sdf.Layer | None = None

    def do(self) -> list[str]:
        """Remove the preset's containers and instancers; returns the removed spec paths."""
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            return []
        self._paths = brush_spec_paths(layer, self._brush_slug)
        self._snapshot = Sdf.Layer.CreateAnonymous("scatter_brush_snapshot")
        for path in self._paths:
            Sdf.CreatePrimInLayer(self._snapshot, path.GetParentPath())
            Sdf.CopySpec(layer, path, self._snapshot, path)
        with Sdf.ChangeBlock():
            for path in self._paths:
                del layer.GetPrimAtPath(path.GetParentPath()).nameChildren[path.name]
            for path in self._paths:
                prune_inert_overs(layer, path.GetParentPath())
        return [str(path) for path in self._paths]

    def undo(self) -> None:
        """Copy the removed containers and instancers back into the layer."""
        layer = _find_layer(self._layer_identifier)
        if layer is None or self._snapshot is None:
            return
        for path in self._paths:
            Sdf.CreatePrimInLayer(layer, path.GetParentPath())
            Sdf.CopySpec(self._snapshot, path, layer, path)


class ScatterSetGroupRecordCommand(omni.kit.commands.Command):
    """Record, replace or forget one scatter group in a layer's custom layer data; undo restores the previous entry."""

    def __init__(self, layer_identifier: str, group_name: str, record: str | None):
        """Create the command.

        Args:
            layer_identifier: Identifier of the layer the group authors its scatter in.
            group_name: Name of the group; ``omni.kit.commands.execute`` reserves ``name`` for the command's own.
            record: JSON text of the group's record, or None to forget the group.
        """
        self._layer_identifier = layer_identifier
        self._name = group_name
        self._record = record
        self._previous: str | None = None

    def do(self) -> None:
        """Write the group's record, keeping the entry it replaces for undo."""
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            return
        self._previous = _set_group_record(layer, self._name, self._record)

    def undo(self) -> None:
        """Write back the entry the group had before ``do()``."""
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            return
        _set_group_record(layer, self._name, self._previous)


def _set_group_record(layer: Sdf.Layer, name: str, record: str | None) -> str | None:
    """Set or clear one group's entry in the custom layer data of ``layer`` and return the entry it replaced."""
    data = dict(layer.customLayerData)
    groups = dict(data.get(SCATTER_GROUPS_LAYER_DATA_KEY, {}))
    previous = groups.get(name)
    if record is None:
        groups.pop(name, None)
    else:
        groups[name] = record
    if groups:
        data[SCATTER_GROUPS_LAYER_DATA_KEY] = groups
    else:
        data.pop(SCATTER_GROUPS_LAYER_DATA_KEY, None)
    layer.customLayerData = data
    return previous


def _context_stage(context_name: str) -> Usd.Stage | None:
    """Return the stage of a USD context, or None when the context or its stage is gone."""
    context = omni.usd.get_context(context_name)
    return None if context is None else context.get_stage()


def _find_layer(identifier: str) -> Sdf.Layer | None:
    """Return the layer with ``identifier``, opening it from disk when needed; None (logged) when unavailable."""
    layer = Sdf.Layer.Find(identifier)
    if layer is None and not Sdf.Layer.IsAnonymousLayerIdentifier(identifier):
        layer = Sdf.Layer.FindOrOpen(identifier)
    if layer is None:
        carb.log_warn(f"{_LOG_PREFIX} Scatter layer '{identifier}' is not available; skipping the scatter edit")
    return layer


def _coerce_records(records: Sequence[Mapping[str, Any] | PlacementRecord] | None) -> list[PlacementRecord]:
    """Return the records as ``PlacementRecord`` instances, converting command payload dictionaries."""
    return [
        (record if isinstance(record, PlacementRecord) else PlacementRecord.from_dict(record))
        for record in records or []
    ]


def _coerce_transforms(
    transforms: Sequence[Mapping[str, Any] | PlacementTransform] | None,
) -> list[PlacementTransform]:
    """Return the transforms as ``PlacementTransform`` instances, converting command payload dictionaries."""
    return [
        (transform if isinstance(transform, PlacementTransform) else PlacementTransform.from_dict(transform))
        for transform in transforms or []
    ]
