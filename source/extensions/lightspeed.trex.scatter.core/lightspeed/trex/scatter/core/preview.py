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
    "InstancerViewportPreview",
    "PreviewState",
    "clear_instancer_preview",
    "preview_path_for",
    "sync_instancer_preview",
]

import asyncio
from dataclasses import dataclass

import carb
import numpy as np
import omni.kit.app
import omni.usd
from lightspeed.common.constants import ROOTNODE_MESHES
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, Vt

from .constants import (
    INSTANCED_ROTATIONS_ATTR,
    INSTANCED_TRANSLATES_ATTR,
    INSTANCER_PREFIX,
    IS_REMIX_SCATTER_ATTR,
    PREVIEW_ENTRY_PREFIX,
    PREVIEW_SUFFIX,
    SCATTER_PREVIEW_OF_ATTR,
)
from .instancing import is_scatter_instancer, quaternions_to_euler_xyz, subscribe_instancer_changes

_LOG_PREFIX = "[lightspeed.trex.scatter.core]"
_XFORM_TYPE_NAME = "Xform"
_XFORM_OP_ORDER = ("xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale")
_PARENT_OP_ORDER = ("xformOp:transform",)
_VISIBILITY_ATTR = "visibility"
_MESHES_ROOT = Sdf.Path(ROOTNODE_MESHES)
# Depth of /RootNode/meshes/mesh_<HASH>/<child>, the level instancers and their stand-ins live at.
_PROTOTYPE_DEPTH = _MESHES_ROOT.pathElementCount + 1


@dataclass
class PreviewState:
    """What a viewport stand-in shows of one instancer, so the next sync writes only what changed.

    Attributes:
        ids: (N,) entry ids, one prim each.
        values: (N, 9) translate, rotateXYZ and scale of each entry.
        prototypes: (N,) object array of the prototype path each entry references.
    """

    ids: np.ndarray
    values: np.ndarray
    prototypes: np.ndarray

    @classmethod
    def empty(cls) -> PreviewState:
        return cls(np.empty(0, dtype=np.int64), np.empty((0, 9)), np.empty(0, dtype=object))


def preview_path_for(instancer_path: Sdf.Path | str) -> Sdf.Path:
    """Return the path of an instancer's viewport stand-in, a sibling of it."""
    path = Sdf.Path(str(instancer_path))
    return path.GetParentPath().AppendChild(f"{path.name}{PREVIEW_SUFFIX}")


def sync_instancer_preview(
    stage: Usd.Stage, layer: Sdf.Layer, instancer_path: Sdf.Path | str, shown: PreviewState | None = None
) -> PreviewState | None:
    """Make an instancer's viewport stand-in in ``layer`` show exactly the instancer's entries.

    The stand-in is one ``Xform`` per entry, internally referencing the entry's prototype and carrying its translate,
    rotateXYZ and scale, beside an ``over`` that hides the instancer itself. Only entries that appeared, left or
    changed since ``shown`` are written, so an erase stamp costs the entries it erased rather than the whole patch.

    Args:
        stage: Stage the instancer is composed on.
        layer: Layer to author the stand-in in; the session layer, so it never reaches the mod.
        instancer_path: Instancer to show.
        shown: What the stand-in showed after the last sync, or None to rebuild it from nothing.

    Returns:
        What the stand-in shows now, or None when there is no instancer any more and the stand-in was removed.
    """
    instancer_path = Sdf.Path(str(instancer_path))
    prim = stage.GetPrimAtPath(instancer_path)
    if not is_scatter_instancer(prim):
        clear_instancer_preview(layer, instancer_path)
        return None
    current = _read_entries(prim)
    preview_path = preview_path_for(instancer_path)
    container = layer.GetPrimAtPath(preview_path)
    if container is None or shown is None:
        if container is not None:
            _delete_spec(layer, container)
        container = _create_container(layer, preview_path, instancer_path)
        shown = PreviewState.empty()
    _hide(layer, instancer_path)
    _set_parent_transform(container, UsdGeom.Xformable(prim).GetLocalTransformation())

    changed, removed = _changed_entries(shown, current)
    for index in changed.tolist():
        _author_entry(layer, preview_path, int(current.ids[index]), current.prototypes[index], current.values[index])
    for entry_id in removed.tolist():
        spec = layer.GetPrimAtPath(preview_path.AppendChild(_entry_name(entry_id)))
        if spec is not None:
            # Deactivated rather than deleted: Sdf removes a child by rewriting its parent's whole list of children,
            # so deleting what one erase stamp took would cost the size of the patch for each entry it took. The
            # spec goes with the stand-in the next time it is rebuilt, and is reused if the entry comes back.
            spec.active = False
    return current


def _changed_entries(shown: PreviewState, current: PreviewState) -> tuple[np.ndarray, np.ndarray]:
    """Compare what a stand-in shows with what its instancer holds now.

    Entries are matched by id with numpy alone. A patch holds tens of thousands of them and this runs at every
    merge and every frame of an erase, where comparing them one by one in Python took longer than the frame.

    Args:
        shown: What the stand-in shows.
        current: What the instancer holds.

    Returns:
        The indices into ``current`` of the entries to author, because they are new or differ from what is shown,
        and the ids the stand-in shows that the instancer no longer holds.
    """
    if not len(shown.ids):
        return np.arange(len(current.ids)), np.empty(0, dtype=np.int64)
    order = np.argsort(shown.ids, kind="stable")
    sorted_ids = shown.ids[order]
    slots = np.minimum(np.searchsorted(sorted_ids, current.ids), len(sorted_ids) - 1)
    before = order[slots]
    unchanged = (
        (sorted_ids[slots] == current.ids)
        & (shown.prototypes[before] == current.prototypes)
        & np.all(shown.values[before] == current.values, axis=1)
    )
    return np.flatnonzero(~unchanged), shown.ids[~np.isin(shown.ids, current.ids)]


def clear_instancer_preview(layer: Sdf.Layer, instancer_path: Sdf.Path | str) -> None:
    """Remove an instancer's viewport stand-in from ``layer`` and stop hiding the instancer there."""
    instancer_path = Sdf.Path(str(instancer_path))
    container = layer.GetPrimAtPath(preview_path_for(instancer_path))
    if container is not None:
        _delete_spec(layer, container)
    spec = layer.GetPrimAtPath(instancer_path)
    if spec is None:
        return
    visibility = spec.attributes.get(_VISIBILITY_ATTR)
    if visibility is not None:
        spec.RemoveProperty(visibility)
    if spec.specifier == Sdf.SpecifierOver and not spec.properties and not spec.nameChildren:
        _delete_spec(layer, spec)


class InstancerViewportPreview:
    """Keeps a viewport stand-in of every scatter instancer on a USD context's stage, in its session layer.

    The toolkit renders with HdRemix, which draws each prototype of a ``UsdGeom.PointInstancer`` once where it stands
    and none of its instances, so instanced scatter would show as a few giant assets. The runtime that plays the mod
    instances it properly. The stand-in shows the toolkit what the game will draw, at the toolkit's usual cost of one
    prim per placement, while the layers the mod is made of hold only the instancer.

    Instancers the brush changes are synced on the next update through ``subscribe_instancer_changes``; a resync on the
    stage, such as a layer muted or a project opened, re-checks the instancers it can have touched. Attribute edits made
    by anything else are not followed, which keeps a live edit of a large stroke from being noticed thousands of times
    a frame; :meth:`refresh` brings everything up to date.
    """

    def __init__(self, usd_context_name: str = ""):
        """
        Args:
            usd_context_name: USD context whose stage gets stand-ins.
        """
        self._usd_context_name = usd_context_name
        self._stage: Usd.Stage | None = None
        self._listener = None
        self._shown: dict[Sdf.Path, PreviewState] = {}
        self._dirty: set[Sdf.Path] = set()
        self._dirty_prototypes: set[Sdf.Path] = set()
        self._rebuild_all = False
        self._syncing = False
        self._flush_task: asyncio.Future | None = None
        self._unsubscribe = subscribe_instancer_changes(self._on_instancer_changed)
        context = omni.usd.get_context(usd_context_name)
        self._stage_event_sub = (
            None
            if context is None
            else context.get_stage_event_stream().create_subscription_to_pop(
                self._on_stage_event, name="ScatterInstancerViewportPreview"
            )
        )
        self._attach(None if context is None else context.get_stage())

    @property
    def shown_instancers(self) -> list[Sdf.Path]:
        """Instancers that currently have a stand-in."""
        return sorted(self._shown, key=str)

    def refresh(self) -> None:
        """Rebuild the stand-in of every instancer on the stage now."""
        self._rebuild_all = True
        self.flush()

    def flush(self) -> None:
        """Sync every stand-in that is due now instead of on the next update."""
        stage = self._stage
        if stage is None:
            self._dirty.clear()
            self._dirty_prototypes.clear()
            self._rebuild_all = False
            return
        if self._rebuild_all:
            targets = set(_stage_instancers(stage)) | set(self._shown)
            fresh = targets
        else:
            targets = set(self._dirty) | {
                path for path in self._shown if path.GetParentPath() in self._dirty_prototypes
            }
            for prototype in self._dirty_prototypes:
                targets.update(_prototype_instancers(stage, prototype))
            fresh = set()
        self._dirty.clear()
        self._dirty_prototypes.clear()
        self._rebuild_all = False
        if not targets:
            return
        layer = stage.GetSessionLayer()
        self._syncing = True
        try:
            with Sdf.ChangeBlock():
                for path in sorted(targets, key=str):
                    shown = None if path in fresh else self._shown.get(path)
                    state = sync_instancer_preview(stage, layer, path, shown)
                    if state is None:
                        self._shown.pop(path, None)
                    else:
                        self._shown[path] = state
        except Exception as exc:  # noqa: BLE001 - a stand-in must never break the update loop it runs on.
            carb.log_error(f"{_LOG_PREFIX} Updating the viewport stand-in of instanced scatter failed: {exc}")
            self._shown.clear()
        finally:
            self._syncing = False

    def destroy(self) -> None:
        """Stop following the stage and take every stand-in back out of the session layer."""
        self._unsubscribe()
        self._stage_event_sub = None
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = None
        self._detach(clear=True)

    def _attach(self, stage: Usd.Stage | None) -> None:
        if stage is not None and stage is self._stage:
            return
        self._detach(clear=False)
        if stage is None:
            return
        self._stage = stage
        self._listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._on_objects_changed, stage)
        self._rebuild_all = True
        self._schedule()

    def _detach(self, clear: bool) -> None:
        stage = self._stage
        if self._listener is not None:
            self._listener.Revoke()
        self._listener = None
        if clear and stage is not None:
            layer = stage.GetSessionLayer()
            self._syncing = True
            try:
                with Sdf.ChangeBlock():
                    for path in list(self._shown):
                        clear_instancer_preview(layer, path)
            finally:
                self._syncing = False
        self._stage = None
        self._shown.clear()
        self._dirty.clear()
        self._dirty_prototypes.clear()

    def _on_stage_event(self, event) -> None:
        if event.type == int(omni.usd.StageEventType.OPENED):
            context = omni.usd.get_context(self._usd_context_name)
            self._attach(None if context is None else context.get_stage())
        elif event.type in (int(omni.usd.StageEventType.CLOSING), int(omni.usd.StageEventType.CLOSED)):
            self._detach(clear=False)

    def _on_instancer_changed(self, _layer: Sdf.Layer, instancer_path: Sdf.Path) -> None:
        if self._stage is None:
            return
        self._dirty.add(Sdf.Path(instancer_path))
        self._schedule()

    def _on_objects_changed(self, notice, stage) -> None:
        if self._syncing or stage is not self._stage:
            return
        scheduled = False
        for path in notice.GetResyncedPaths():
            prim_path = path.GetPrimPath() if path.IsPropertyPath() else path
            if prim_path.pathElementCount < _PROTOTYPE_DEPTH:
                if _MESHES_ROOT.HasPrefix(prim_path) or prim_path == Sdf.Path.absoluteRootPath:
                    self._rebuild_all = True
                    scheduled = True
                continue
            if not prim_path.HasPrefix(_MESHES_ROOT):
                continue
            if prim_path.pathElementCount == _PROTOTYPE_DEPTH:
                self._dirty_prototypes.add(prim_path)
                scheduled = True
                continue
            child = prim_path.GetPrefixes()[_PROTOTYPE_DEPTH]
            if child.name.startswith(INSTANCER_PREFIX) and not child.name.endswith(PREVIEW_SUFFIX):
                self._dirty.add(child)
                scheduled = True
        if scheduled:
            self._schedule()

    def _schedule(self) -> None:
        if self._flush_task is not None and not self._flush_task.done():
            return
        try:
            self._flush_task = asyncio.ensure_future(self._flush_next_update())
        except RuntimeError:
            # No event loop to wait on, as in a bare script: sync straight away instead.
            self.flush()

    async def _flush_next_update(self) -> None:
        await omni.kit.app.get_app().next_update_async()
        self.flush()


def _read_entries(prim: Usd.Prim) -> PreviewState:
    """Read what an instancer renders, per entry, as the composed stage has it."""
    instancer = UsdGeom.PointInstancer(prim)
    positions = instancer.GetPositionsAttr().Get()
    count = 0 if positions is None else len(positions)
    if not count:
        return PreviewState.empty()
    translates = _array(prim.GetAttribute(INSTANCED_TRANSLATES_ATTR), count, 3)
    if translates is None:
        translates = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    rotations = _array(prim.GetAttribute(INSTANCED_ROTATIONS_ATTR), count, 3)
    if rotations is None:
        orientations = _array(instancer.GetOrientationsAttr(), count, 4)
        rotations = np.zeros((count, 3)) if orientations is None else quaternions_to_euler_xyz(orientations)
    scales = _array(instancer.GetScalesAttr(), count, 3)
    if scales is None:
        scales = np.ones((count, 3))
    ids = _array(instancer.GetIdsAttr(), count, 0)
    ids = np.arange(count, dtype=np.int64) if ids is None else ids.astype(np.int64)
    targets = [str(path) for path in instancer.GetPrototypesRel().GetForwardedTargets()]
    indices = _array(instancer.GetProtoIndicesAttr(), count, 0)
    table = np.array([*targets, ""], dtype=object)
    if indices is None:
        prototypes = np.full(count, targets[0] if len(targets) == 1 else "", dtype=object)
    else:
        indices = indices.astype(np.int64)
        prototypes = table[np.where((indices >= 0) & (indices < len(targets)), indices, len(targets))]
    keep = prototypes != ""
    values = np.concatenate([translates, rotations, scales], axis=1)
    return PreviewState(ids[keep], values[keep], prototypes[keep])


def _array(attribute: Usd.Attribute, count: int, width: int) -> np.ndarray | None:
    """Return an attribute's array as float64 numpy, or None when it is missing or does not hold ``count`` entries."""
    value = attribute.Get() if attribute else None
    if value is None:
        return None
    values = np.asarray(value, dtype=np.float64)
    if width:
        if values.size != count * width:
            return None
        return values.reshape(count, width)
    return values if len(values) == count else None


def _author_entry(layer: Sdf.Layer, preview_path: Sdf.Path, entry_id: int, prototype: str, values: np.ndarray) -> None:
    """Author one stand-in prim: a reference to the entry's prototype carrying the entry's three ops."""
    spec = Sdf.CreatePrimInLayer(layer, preview_path.AppendChild(_entry_name(entry_id)))
    if spec.HasActive():
        # The entry was erased and is back, as after undoing the erase: its stand-in was only switched off.
        spec.ClearActive()
    spec.specifier = Sdf.SpecifierDef
    spec.typeName = _XFORM_TYPE_NAME
    spec.referenceList.prependedItems = [Sdf.Reference(primPath=Sdf.Path(prototype))]
    _set(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*(float(v) for v in values[0:3])))
    _set(spec, "xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*(float(v) for v in values[3:6])))
    _set(spec, "xformOp:scale", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*(float(v) for v in values[6:9])))
    _set(
        spec,
        "xformOpOrder",
        Sdf.ValueTypeNames.TokenArray,
        Vt.TokenArray(list(_XFORM_OP_ORDER)),
        variability=Sdf.VariabilityUniform,
    )


def _create_container(layer: Sdf.Layer, preview_path: Sdf.Path, instancer_path: Sdf.Path) -> Sdf.PrimSpec:
    """Create the stand-in container, marked so that pickers and the brush skip everything beneath it."""
    spec = Sdf.CreatePrimInLayer(layer, preview_path)
    spec.specifier = Sdf.SpecifierDef
    spec.typeName = _XFORM_TYPE_NAME
    _set(spec, IS_REMIX_SCATTER_ATTR, Sdf.ValueTypeNames.Bool, True, custom=True)
    _set(spec, SCATTER_PREVIEW_OF_ATTR, Sdf.ValueTypeNames.String, str(instancer_path), custom=True)
    return spec


def _hide(layer: Sdf.Layer, instancer_path: Sdf.Path) -> None:
    """Hide the instancer in ``layer``, so HdRemix does not draw its prototypes where they stand."""
    spec = layer.GetPrimAtPath(instancer_path) or Sdf.CreatePrimInLayer(layer, instancer_path)
    attribute = spec.attributes.get(_VISIBILITY_ATTR)
    if attribute is None:
        attribute = Sdf.AttributeSpec(spec, _VISIBILITY_ATTR, Sdf.ValueTypeNames.Token, Sdf.VariabilityVarying)
    if attribute.default != UsdGeom.Tokens.invisible:
        attribute.default = UsdGeom.Tokens.invisible


def _set_parent_transform(container: Sdf.PrimSpec, local: Gf.Matrix4d) -> None:
    """Give the stand-in the instancer's own transform, which the brush leaves at identity."""
    if Gf.IsClose(local, Gf.Matrix4d(1.0), 1e-12):
        if "xformOp:transform" in container.attributes:
            container.RemoveProperty(container.attributes["xformOp:transform"])
            container.RemoveProperty(container.attributes["xformOpOrder"])
        return
    _set(container, "xformOp:transform", Sdf.ValueTypeNames.Matrix4d, local)
    _set(
        container,
        "xformOpOrder",
        Sdf.ValueTypeNames.TokenArray,
        Vt.TokenArray(list(_PARENT_OP_ORDER)),
        variability=Sdf.VariabilityUniform,
    )


def _stage_instancers(stage: Usd.Stage) -> list[Sdf.Path]:
    """Return every scatter instancer on the stage."""
    meshes = stage.GetPrimAtPath(_MESHES_ROOT)
    found = []
    for prototype in meshes.GetChildren() if meshes else ():
        found.extend(_prototype_instancers(stage, prototype.GetPath()))
    return found


def _prototype_instancers(stage: Usd.Stage, prototype_root: Sdf.Path) -> list[Sdf.Path]:
    """Return the scatter instancers directly under a prototype."""
    root = stage.GetPrimAtPath(prototype_root)
    if not root:
        return []
    return [
        child.GetPath()
        for child in root.GetChildren()
        if child.GetName().startswith(INSTANCER_PREFIX) and is_scatter_instancer(child)
    ]


def _entry_name(entry_id: int) -> str:
    return f"{PREVIEW_ENTRY_PREFIX}{entry_id:012x}"


def _set(
    spec: Sdf.PrimSpec,
    name: str,
    type_name: Sdf.ValueTypeName,
    value,
    custom: bool = False,
    variability: Sdf.Variability = Sdf.VariabilityVarying,
) -> None:
    attribute = spec.attributes.get(name)
    if attribute is None:
        attribute = Sdf.AttributeSpec(spec, name, type_name, variability)
    if custom:
        attribute.custom = True
    attribute.default = value


def _delete_spec(layer: Sdf.Layer, spec: Sdf.PrimSpec) -> None:
    parent = spec.nameParent or layer.pseudoRoot
    del parent.nameChildren[spec.name]
