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

__all__ = ["ScatterGizmoTool", "scatter_gizmo_factory"]

from typing import TYPE_CHECKING, Any

import carb
import carb.settings
import omni.usd
from lightspeed.trex.scatter.core import (
    MeshSurfaceCache,
    ReapplyScope,
    ScatterMode,
    ScopeDragSession,
    get_scatter_brush_controller,
    resolve_scope,
    scope_pivot,
    turn_matrix,
    turn_to_degrees,
)
from omni.kit.manipulator.transform import (
    Axis,
    Operation,
    RotateChangedGesture,
    RotateDragGesturePayload,
    SimpleTransformModel,
    TransformDragGesturePayload,
    TransformManipulator,
    TranslateChangedGesture,
    TranslateDragGesturePayload,
)
from pxr import Gf, Usd

if TYPE_CHECKING:
    from collections.abc import Callable

    from lightspeed.trex.scatter.core import MoveOutcome
    from omni.kit.widget.viewport.api import ViewportAPI

_LOG_PREFIX = "[lightspeed.trex.viewports.shared.widget]"
# The operation the viewport toolbar and the W / E / R keys write; the gizmo follows it so it behaves like the prim
# one rather than inventing a second way to switch between moving and turning.
_TRANSFORM_OP_SETTING = "/app/transform/operation"
_OPERATIONS = {"move": Operation.TRANSLATE, "rotate": Operation.ROTATE}
_MANIPULATOR_SIZE = 1.0


def scatter_gizmo_factory(desc: dict[str, Any]) -> ScatterGizmoTool:
    """Create the scatter move gizmo for a viewport from the scene layer factory arguments."""
    return ScatterGizmoTool(
        desc.get("viewport_api"),
        desc.get("is_active_fn"),
        desc.get("usd_context_name"),
    )


class ScatterGizmoTool:
    """Viewport half of the scatter move gizmo for one viewport.

    Shows the standard transform gizmo at the centre of the painted scatter the user has selected, and turns a drag
    of it into one undoable move of that whole patch. The scope comes from ``resolve_scope``, the same rules the
    Scatter window's live editing uses, resolved here rather than read from the window so the gizmo works whether or
    not that window is open.

    The gizmo is hidden while the brush is painting or erasing, because a handle sitting on the patch would take the
    drag the brush is waiting for.
    """

    def __init__(
        self,
        viewport_api: ViewportAPI,
        is_active_fn: Callable[[], bool] | None,
        usd_context_name: str | None,
    ):
        """
        Args:
            viewport_api: Viewport the gizmo is drawn in.
            is_active_fn: Returns whether this viewport is the active one; the gizmo stays hidden otherwise.
            usd_context_name: USD context the moves are authored in; defaults to the viewport's context.
        """
        self._viewport_api = viewport_api
        self._is_active_fn = is_active_fn
        self._usd_context_name = viewport_api.usd_context_name if usd_context_name is None else usd_context_name

        self._visible = True
        self._scope = ReapplyScope()
        self._controller = get_scatter_brush_controller()
        self._cache: MeshSurfaceCache | None = MeshSurfaceCache(self._stage)

        self._session = ScopeDragSession(self._usd_context_name, self._cache)
        # The turn the patch already stood at when this drag opened, so consecutive drags add up rather than each
        # one starting the patch over at no turn.
        self._turn_at_start = Gf.Matrix4d(1.0)
        self._pivot = Gf.Vec3d(0.0, 0.0, 0.0)

        self._model = SimpleTransformModel()
        self._model.global_mode = True
        # Built in the factory's context, so the gizmo belongs to this viewport's scene layer.
        self._manipulator = TransformManipulator(
            size=_MANIPULATOR_SIZE,
            enabled=False,
            axes=Axis.ALL,
            model=self._model,
            gestures=[_TranslateGesture(self), _RotateGesture(self)],
        )

        settings = carb.settings.get_settings()
        self._operation_subscription = settings.subscribe_to_node_change_events(
            _TRANSFORM_OP_SETTING, lambda *_: self.refresh()
        )
        self._mode_subscription = self._controller.subscribe_mode_changed(lambda *_: self.refresh())
        self._stage_event_subscription = (
            omni.usd.get_context(self._usd_context_name)
            .get_stage_event_stream()
            .create_subscription_to_pop(self._on_stage_event, name="ScatterGizmo")
        )
        self.refresh()

    @property
    def name(self) -> str:
        """Layer name shown by the viewport layer system."""
        return "Scatter Move Gizmo"

    @property
    def categories(self) -> list[str]:
        """Layer categories used by ``find_viewport_layer``."""
        return ["manipulator"]

    @property
    def visible(self) -> bool:
        """Whether the layer system shows this gizmo."""
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = bool(value)
        self.refresh()

    @property
    def scope(self) -> ReapplyScope:
        """Placements a drag of the gizmo would move."""
        return self._scope

    @property
    def pivot(self) -> Gf.Vec3d:
        """World point the gizmo sits on, which a turn of the patch is measured about."""
        return self._pivot

    def refresh(self) -> None:
        """Resolve the selection again and show the gizmo on it when a drag would have somewhere to land."""
        if self._manipulator is None:
            return
        stage = self._stage()
        usd_context = omni.usd.get_context(self._usd_context_name)
        self._scope = (
            ReapplyScope() if stage is None else resolve_scope(stage, usd_context, self._controller.settings)
        )
        operation = _OPERATIONS.get(carb.settings.get_settings().get(_TRANSFORM_OP_SETTING))
        showing = bool(
            self._visible
            and operation is not None
            and stage is not None
            and not self._scope.is_empty
            and self._controller.mode == ScatterMode.OFF
            and (self._is_active_fn is None or self._is_active_fn())
        )
        if showing:
            pivot = scope_pivot(stage, stage.GetEditTarget().GetLayer(), self._scope)
            showing = pivot is not None
            if showing:
                self._pivot = pivot
                self._model.set_operation(operation)
                self._model.set_floats(self._model.get_item("translate"), [pivot[0], pivot[1], pivot[2]])
        self._manipulator.enabled = showing

    def begin_drag(self) -> None:
        """Take the patch as it stands, so every step of the drag is measured from there rather than from the last."""
        if self._session.begin(self._scope, self._conform):
            self._turn_at_start = turn_matrix(self._controller.selection_turn(self._scope))

    def update_drag(self, delta: Gf.Matrix4d) -> None:
        """Move the patch to where a drag of ``delta`` from the start of the gesture would put it."""
        self._report(self._session.update(delta))
        self._publish_turn(delta)

    def end_drag(self) -> None:
        """Commit the drag as one undoable move and put the gizmo back on the patch where it now sits."""
        self._session.commit()
        self.refresh()

    def cancel_drag(self) -> None:
        """Put the patch back as it was and close the drag without committing anything."""
        self._session.cancel()
        self._controller.set_selection_turn(self._scope, turn_to_degrees(self._turn_at_start))
        self.refresh()

    def _publish_turn(self, delta: Gf.Matrix4d) -> None:
        """Tell the Scatter window's turn fields where this drag has aimed the patch.

        Only the rotation of the drag is reported, so sliding a patch leaves the turn it already stands at alone.
        """
        rotation = delta.ExtractRotation()
        if abs(rotation.GetAngle()) < 1e-9:
            return
        total = self._turn_at_start * Gf.Matrix4d().SetRotate(rotation)
        self._controller.set_selection_turn(self._scope, turn_to_degrees(total))

    def destroy(self):
        """Close any drag and drop every subscription and scene item."""
        self.cancel_drag()
        self._operation_subscription = None
        self._mode_subscription = None
        self._stage_event_subscription = None
        if self._cache is not None:
            self._cache.destroy()
        self._cache = None
        if self._manipulator is not None:
            self._manipulator.enabled = False
        self._manipulator = None
        self._model = None

    @property
    def _conform(self) -> bool:
        """Whether the brush is set to conform to the surface, which decides if a slide may re-aim the patch."""
        return self._controller.settings.conform_to_surface

    def _stage(self) -> Usd.Stage | None:
        """Return the stage of the gizmo's USD context, or None when there is none."""
        usd_context = omni.usd.get_context(self._usd_context_name)
        return None if usd_context is None else usd_context.get_stage()

    def _on_stage_event(self, event) -> None:
        """Follow the selection, and close a drag the stage is about to pull the ground out from under."""
        if event.type == int(omni.usd.StageEventType.SELECTION_CHANGED):
            self.refresh()
        elif event.type in (int(omni.usd.StageEventType.CLOSING), int(omni.usd.StageEventType.OPENED)):
            self.cancel_drag()

    def _report(self, outcome: MoveOutcome) -> None:
        """Log what a drag could not do; the Scatter window shows the same through the controller's status."""
        notes = outcome.describe()
        if notes:
            carb.log_info(f"{_LOG_PREFIX} Scatter move: {notes}")


class _PatchGesture:
    """What the gizmo's two gestures have in common: one drag of one handle.

    A manipulator hands every gesture registered on it whatever payload the dispatch carries, and only the drag of
    the handle a gesture stands for carries the fields saying how far it was dragged. The toolbar's own gesture and
    the scene's dispatches carry a plain payload instead, which would otherwise be read for a distance it does not
    hold, or start a drag the user never made. Each handler therefore acts only on its own kind of drag.
    """

    _PAYLOAD_TYPE: type = TransformDragGesturePayload

    def __init__(self, tool: ScatterGizmoTool):
        super().__init__()
        self._tool = tool

    def _drag_payload(self) -> Any | None:
        """Return the payload of this gesture's own drag, or None when the dispatch is something else."""
        payload = self.gesture_payload
        return payload if isinstance(payload, self._PAYLOAD_TYPE) else None

    def on_began(self):
        """Start the drag."""
        if self._drag_payload() is not None:
            self._tool.begin_drag()

    def on_ended(self):
        """Commit the drag."""
        if self._drag_payload() is not None:
            self._tool.end_drag()

    def on_canceled(self):
        """Put the patch back."""
        if self._drag_payload() is not None:
            self._tool.cancel_drag()


class _TranslateGesture(_PatchGesture, TranslateChangedGesture):
    """Turns a drag of the translate arrows into a move of the whole patch."""

    _PAYLOAD_TYPE = TranslateDragGesturePayload

    def on_changed(self):
        """Move the patch to the total distance dragged so far."""
        payload = self._drag_payload()
        if payload is None:
            return
        moved = payload.moved
        self._tool.update_drag(Gf.Matrix4d().SetTranslate(Gf.Vec3d(moved[0], moved[1], moved[2])))


class _RotateGesture(_PatchGesture, RotateChangedGesture):
    """Turns a drag of the rotate rings into a turn of every placement of the patch where it stands."""

    _PAYLOAD_TYPE = RotateDragGesturePayload

    def on_changed(self):
        """Aim the patch by the total angle dragged so far, about the axis the ring stands for.

        The rotation is sent on its own, with no pivot: ``move_scope`` turns each placement where it stands, so
        sandwiching it between translations to and from the handle would only claim a movement it does not make.
        The handle itself still sits on the middle of the patch, which is what the rings are drawn around.
        """
        payload = self._drag_payload()
        if payload is None:
            return
        axis = Gf.Vec3d(payload.axis[0], payload.axis[1], payload.axis[2])
        if axis.GetLength() < 1e-9:
            return
        self._tool.update_drag(Gf.Matrix4d().SetRotate(Gf.Rotation(axis, payload.angle)))
