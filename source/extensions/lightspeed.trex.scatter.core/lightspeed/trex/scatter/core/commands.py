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

__all__ = ["ScatterFloodCommand", "ScatterReapplyCommand", "ScatterStrokeCommand", "ScatterStrokeKind"]

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

import carb
import omni.kit.commands
from pxr import Sdf

from .placement import (
    PlacementRecord,
    PlacementTransform,
    author_placement_transforms,
    author_placements,
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

    PLACE: the records were already authored live while the mouse moved, so the first ``do()`` is a no-op, ``undo()``
    removes the created prims and redo re-authors them. ERASE: the prims were already removed live and a snapshot of
    their specs is given, so the first ``do()`` is a no-op, ``undo()`` restores them from the snapshot and redo removes
    them again. With ``already_applied=False`` the first ``do()`` performs the edit itself.
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

    def do(self) -> list[str]:
        """Apply the stroke unless it was already applied live; returns the affected prim paths."""
        if self._skip_next_do:
            self._skip_next_do = False
            return list(self._created_paths if self._kind == ScatterStrokeKind.PLACE else self._prim_paths)
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            return []
        if self._kind == ScatterStrokeKind.PLACE:
            self._created_paths = author_placements(layer, self._records, self._materials)
            return list(self._created_paths)
        if self._snapshot_layer is None:
            self._snapshot_layer = snapshot_prims(layer, self._prim_paths)
        remove_placements(layer, self._prim_paths)
        return list(self._prim_paths)

    def undo(self) -> None:
        """Revert the stroke: remove placed prims or restore erased prims from the snapshot."""
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            return
        if self._kind == ScatterStrokeKind.PLACE:
            remove_placements(layer, self._created_paths)
            return
        if self._snapshot_layer is None:
            carb.log_warn(f"{_LOG_PREFIX} No snapshot available to restore the erased scatter placements")
            return
        restore_prims(layer, self._snapshot_layer, self._prim_paths)


class ScatterFloodCommand(omni.kit.commands.Command):
    """Author a batch of flood placements; undo removes them again."""

    def __init__(
        self,
        context_name: str,
        layer_identifier: str,
        records: Sequence[Mapping[str, Any] | PlacementRecord],
        materials: Mapping[str, Any] | None = None,
    ):
        """Create the command.

        Args:
            context_name: Name of the USD context the flood was requested in.
            layer_identifier: Identifier of the layer to author the placements in.
            records: The placement records (``PlacementRecord.to_dict()`` payloads or records).
            materials: The shader overrides to author, kept on the command so a redo still has them once the asset
                has left the brush palette.
        """
        self._context_name = context_name
        self._layer_identifier = layer_identifier
        self._records = _coerce_records(records)
        self._materials = dict(materials or {})
        self._created_paths: list[str] = []

    def do(self) -> list[str]:
        """Author the placements; returns the created prim paths."""
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            self._created_paths = []
            return []
        self._created_paths = author_placements(layer, self._records, self._materials)
        return list(self._created_paths)

    def undo(self) -> None:
        """Remove the placements created by ``do()``."""
        if not self._created_paths:
            return
        layer = _find_layer(self._layer_identifier)
        if layer is None:
            return
        remove_placements(layer, self._created_paths)


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
        record if isinstance(record, PlacementRecord) else PlacementRecord.from_dict(record) for record in records or []
    ]


def _coerce_transforms(
    transforms: Sequence[Mapping[str, Any] | PlacementTransform] | None,
) -> list[PlacementTransform]:
    """Return the transforms as ``PlacementTransform`` instances, converting command payload dictionaries."""
    return [
        transform if isinstance(transform, PlacementTransform) else PlacementTransform.from_dict(transform)
        for transform in transforms or []
    ]
