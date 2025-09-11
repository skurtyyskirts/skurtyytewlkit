"""
* SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    "ScatterBrushButtonGroup",
    "create_button_instance",
    "delete_button_instance",
    "scatter_brush_factory",
]

import asyncio
import time
from dataclasses import dataclass
import contextlib
from typing import TYPE_CHECKING, Callable, Optional

import carb
import omni.kit.commands
import omni.kit.undo
import omni.ui as ui
import omni.usd
from lightspeed.trex.app.style import style
from lightspeed.common import constants as _constants
from lightspeed.trex.utils.common.asset_utils import is_asset_ingested as _is_asset_ingested
from lightspeed.trex.hotkeys import TrexHotkeyEvent
from lightspeed.trex.hotkeys import get_global_hotkey_manager as _get_global_hotkey_manager
from pxr import Gf, Sdf, Usd, UsdGeom
from omni.kit.widget.toolbar.widget_group import WidgetGroup
from lightspeed.layer_manager.core import LayerManagerCore as _LayerManagerCore
from lightspeed.layer_manager.core import LayerType as _LayerType
from omni.kit.notification_manager import NotificationStatus, post_notification
import omni.client

try:
    # Optional utility that enforces authoring in mod/replacement layer and creates anchors per mesh
    from tools.custom import mesh_anchor_resolver as _anchor_resolver  # type: ignore
except Exception:  # pragma: no cover
    _anchor_resolver = None  # type: ignore

from .teleport import PointMousePicker  # reuse picker and screen->NDC mapping

if TYPE_CHECKING:
    from omni.kit.widget.viewport.api import ViewportAPI


# Heuristic distance to consider a hit as "sky" or otherwise too far to be meaningful
MAX_PICK_DISTANCE = 1.0e5


@dataclass
class BrushSettings:
    # UI-exposed settings (read from carb.settings on demand)
    radius: float = 0.25
    spacing: float = 0.25
    density: float = 1.0
    random_yaw_min_deg: float = -15.0
    random_yaw_max_deg: float = 15.0
    uniform_scale_min: float = 1.0
    uniform_scale_max: float = 1.0
    align_to_normal: bool = False  # HdRemix query doesn't return normals; keep optional
    offset_along_normal: float = 0.0

    # Core behavior
    min_interval_ms: int = 120  # place at most every N ms when dragging
    group_parent_name: str = "ScatterAnchors"
    # Current asset to scatter (absolute or project-relative USD path)
    asset_path: str | None = None


_scatter_button_group: ScatterBrushButtonGroup | None = None


class _ToggleModel(ui.AbstractValueModel):
    def __init__(self):
        super().__init__()
        self._value = False

    def get_value_as_bool(self):
        return bool(self._value)

    def set_value(self, v):
        new_val = bool(v)
        if new_val != self._value:
            self._value = new_val
            self._value_changed()


class ScatterBrushButtonGroup(WidgetGroup):
    """Scatter Brush toolbar button (toggle)."""

    name = "scatter_brush"

    def __init__(self):
        super().__init__()
        self._button: Optional[ui.ToolButton] = None
        self._model = _ToggleModel()
        self.__on_toggled: list[Callable[[bool], None]] = []

        # subscribe to model changes to propagate
        self._model.add_value_changed_fn(lambda *_: self._notify(self._model.get_value_as_bool()))
        # bridge to settings for external observers
        def _settings_bridge():
            try:
                carb.settings.get_settings().set(
                    'exts."lightspeed.trex.tools.scatter_brush".enabled', bool(self._model.get_value_as_bool())
                )
            except Exception:
                pass
        self._model.add_value_changed_fn(lambda *_: _settings_bridge())

    def _notify(self, value: bool):
        for cb in list(self.__on_toggled):
            try:
                cb(value)
            except Exception:  # noqa
                carb.log_warn("ScatterBrushButtonGroup subscriber raised")

    def subscribe_toggled(self, callback: Callable[[bool], None]):
        self.__on_toggled.append(callback)
        return callback

    def unsubscribe_toggled(self, callback: Callable[[bool], None]):
        with contextlib.suppress(ValueError):
            self.__on_toggled.remove(callback)

    def get_style(self):
        # expects a style key Button.Image::scatter_brush (lowercase to match style key)
        return {f"Button.Image::{self.name}": style.default.get(f"Button.Image::{self.name}", {})}

    def _on_mouse_released(self, _button):
        # toggle state
        self._acquire_toolbar_context()
        if self._is_in_context() and self._button is not None and self._button.enabled:
            current = self._model.get_value_as_bool()
            self._model.set_value(not current)

    def create(self, default_size: ui.Length):
        self._button = ui.ToolButton(
            model=self._model,
            name=self.name,
            identifier=self.name,
            tooltip="Scatter Brush (paint to place objects under the mouse)",
            width=default_size,
            height=default_size,
            mouse_released_fn=lambda x, y, b, _: self._on_mouse_released(b),
        )
        return {self.name: self._button}

    def clean(self):
        super().clean()
        self._button = None
        self.__on_toggled = []


def create_button_instance():
    global _scatter_button_group
    _scatter_button_group = ScatterBrushButtonGroup()
    return _scatter_button_group


def delete_button_instance():
    global _scatter_button_group
    _scatter_button_group = None


class ScatterBrush:
    """Handle scattering while active; intended to be created via factory per viewport."""

    def __init__(self, viewport_api: "ViewportAPI"):
        self._viewport_api = viewport_api
        self._settings = BrushSettings()
        # Settings bridge: allow another extension to set current asset and parameters in carb.settings
        # These keys are intentionally simple; UI agent will write to them.
        self._settings_iface = carb.settings.get_settings()
        # Read primary settings from the Scatter Brush UI extension settings root to avoid drift
        # Keep legacy fallback to the older prefix if needed
        self._settings_prefix_tools = 'exts."lightspeed.trex.tools.scatter_brush"'
        self._settings_prefix_legacy = "/exts/lightspeed.trex.scatter_brush"

        # overlay frame (same pattern as Teleporter) for coordinate conversions if needed later
        self.__viewport_frame = ui.Frame()

        self._active: bool = False
        self._picker = PointMousePicker(self._viewport_api, self.__viewport_frame, self._on_pick)
        self._last_place_time_ms: int = 0
        self._pending_pick: bool = False
        self._mouse_was_down: bool = False
        self._undo_group_open: bool = False

        # Register hotkey to toggle painting (reuse teleport hotkey? no, keep separate if available)
        # Not defining a new TrexHotkeyEvent; toolbar toggle is primary control.
        # Subscribe to toolbar toggle
        if _scatter_button_group:
            _scatter_button_group.subscribe_toggled(self.set_active)

        # Ensure selection of viewport (avoid painting when inactive viewport)
        self._hotkey_subscription = _get_global_hotkey_manager().subscribe_hotkey_event(
            TrexHotkeyEvent.F, lambda: None
        )

        # Background loop to sample mouse while active
        self._task: Optional[asyncio.Task] = asyncio.ensure_future(self._run_loop())

    def destroy(self):
        with contextlib.suppress(Exception):
            if self._task:
                self._task.cancel()

    # Required for compatibility with scene layer
    @property
    def visible(self):
        return True

    @property
    def categories(self):
        return ("tools",)

    @property
    def name(self):
        return "Scatter Brush"

    def set_active(self, value: bool):
        self._active = bool(value)

    async def _run_loop(self):
        # periodic sampler; when active and LMB is pressed, request a pick under mouse
        input_interface = carb.input.acquire_input_interface()
        import omni.appwindow as _appwindow

        app_window = _appwindow.get_default_app_window()
        mouse = app_window.get_mouse()
        while True:
            await asyncio.sleep(0.03)
            if not self._active:
                # Close any open undo group if we became inactive
                if self._undo_group_open:
                    with contextlib.suppress(Exception):
                        omni.kit.undo.end_group()
                    self._undo_group_open = False
                continue

            mouse_down = input_interface.get_mouse_value(mouse, carb.input.MouseInput.LEFT_BUTTON) > 0
            # Manage undo group across press lifecycle
            if mouse_down and not self._mouse_was_down:
                with contextlib.suppress(Exception):
                    omni.kit.undo.begin_group()
                self._undo_group_open = True
            elif (not mouse_down) and self._mouse_was_down:
                if self._undo_group_open:
                    with contextlib.suppress(Exception):
                        omni.kit.undo.end_group()
                    self._undo_group_open = False
            self._mouse_was_down = mouse_down

            # Only act if left mouse button is pressed
            if not mouse_down:
                continue
            # Avoid overlapping pick requests
            if self._pending_pick:
                continue
            self._pending_pick = True
            try:
                # Pick at current mouse position
                self._picker.pick()
            finally:
                # _on_pick will run async context via callback; small delay allows callback to schedule
                await asyncio.sleep(0)
                self._pending_pick = False

    def _on_pick(self, path: str, position: carb.Double3 | None, _pixels: carb.Uint2):
        # place object if rate-limited and valid position
        if position is None:
            return
        # Ignore clearly invalid picks (no path)
        if not path:
            return
        now_ms = int(time.time() * 1000)
        if now_ms - self._last_place_time_ms < self._settings.min_interval_ms:
            return
        self._last_place_time_ms = now_ms
        try:
            self._place_at_world_position(Gf.Vec3d(position[0], position[1], position[2]), path)
        except Exception:  # noqa
            carb.log_warn("Scatter brush placement failed")

    def _ensure_group_parent(self, stage: Usd.Stage) -> Usd.Prim:
        # Create (or get) a scope for scatter anchors near the default prim
        default_prim = stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            parent_path = default_prim.GetPath()
        else:
            parent_path = Sdf.Path("/World")
            if not stage.GetPrimAtPath(parent_path).IsValid():
                UsdGeom.Xform.Define(stage, parent_path)
        group_path = parent_path.AppendPath(self._settings.group_parent_name)
        prim = stage.GetPrimAtPath(group_path)
        if not prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrimCommand",
                prim_path=group_path,
                prim_type="Scope",
                select_new_prim=False,
                context_name=self._viewport_api.usd_context_name,
            )
            prim = stage.GetPrimAtPath(group_path)
        return prim

    @staticmethod
    def _mesh_root_from_path(prim_path: str) -> Sdf.Path | None:
        import re
        # constants.REGEX_MESH_PATH matches the mesh root path
        pattern = re.compile(_constants.REGEX_MESH_PATH)
        match = pattern.match(str(prim_path))
        if match:
            return Sdf.Path(match.group(0))
        # Try trimming to parent until match
        try_path = Sdf.Path(prim_path)
        while try_path and try_path != Sdf.Path.absoluteRootPath:
            if pattern.match(str(try_path)):
                return try_path
            try_path = try_path.GetParentPath()
        return None

    def _is_invalid_category_target(self, stage: Usd.Stage, mesh_root: Sdf.Path) -> bool:
        """Return True if the mesh root or its immediate prim has a category that should be ignored."""
        prim = stage.GetPrimAtPath(mesh_root)
        if not prim or not prim.IsValid():
            return True
        invalid_attrs = [
            _constants.REMIX_CATEGORIES["World UI"]["attr"],
            _constants.REMIX_CATEGORIES["Sky"]["attr"],
            _constants.REMIX_CATEGORIES["Ignore"]["attr"],
        ]
        for attr_name in invalid_attrs:
            attr = prim.GetAttribute(attr_name)
            if attr.IsValid() and bool(attr.Get()):
                return True
        return False

    def _place_at_world_position(self, world_pos: Gf.Vec3d, picked_path: str):
        stage = self._viewport_api.stage
        # Resolve selected asset from settings; only proceed if present
        asset_path = (
            self._settings_iface.get_as_string(self._settings_prefix_tools + ".asset_usd_path")
            or self._settings_iface.get_as_string(self._settings_prefix_legacy + "/asset_path")
            or self._settings.asset_path
        )
        if not asset_path:
            return
        # Ensure asset is ingested (or part of capture, which is always allowed)
        if not _is_asset_ingested(asset_path):
            post_notification("Scatter Brush: Selected asset is not ingested or invalid.", status=NotificationStatus.WARNING)
            return

        # Ignore far-away hits (likely sky)
        try:
            cam_path = self._viewport_api.camera_path
            if cam_path:
                cam_prim = UsdGeom.Camera(stage.GetPrimAtPath(cam_path))
                if cam_prim:
                    cam_xf = cam_prim.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    cam_pos = cam_xf.ExtractTranslation()
                    import math as _math

                    if _math.dist(cam_pos, world_pos) > MAX_PICK_DISTANCE:
                        return
        except Exception:
            pass

        # Only allow scatter on captured meshes. If invalid, ignore event.
        mesh_root = self._mesh_root_from_path(picked_path) if picked_path else None
        if mesh_root is None:
            return
        if self._is_invalid_category_target(stage, mesh_root):
            return

        # Ensure we are authoring on the replacement (mod) layer to avoid capture edits
        try:
            _LayerManagerCore(self._viewport_api.usd_context_name).set_edit_target_layer(
                _LayerType.replacement, do_undo=False
            )
        except Exception:
            pass

        # Author as PointInstancer under a per-anchor parent beneath the picked mesh
        parent_prim: Usd.Prim
        if _anchor_resolver is not None and picked_path:
            try:
                anchor_path = _anchor_resolver.resolve_or_create_anchor_for_hit(str(picked_path), stage=stage)
                parent_prim = stage.GetPrimAtPath(anchor_path)
            except Exception:
                parent_prim = self._ensure_anchor_under_mesh(stage, mesh_root)
        else:
            parent_prim = self._ensure_anchor_under_mesh(stage, mesh_root)

        parent_to_world = (
            UsdGeom.Xformable(parent_prim).ComputeParentToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        )
        local_translation = parent_to_world.Transform(world_pos)

        # Ensure a PointInstancer exists for the chosen asset
        pi_prim = self._ensure_point_instancer(parent_prim, asset_path)
        self._append_instance(pi_prim, local_translation, parent_prim, picked_path, world_pos)

    def _ensure_point_instancer(self, parent_prim: Usd.Prim, asset_path: str) -> Usd.Prim:
        stage = parent_prim.GetStage()

        # Validate referenced asset exists before creating PI/prototype
        try:
            url = omni.client.normalize_url(asset_path)
            res, _info = omni.client.stat(url)
            if res != omni.client.Result.OK:
                post_notification(
                    f"Scatter Brush: Cannot reference asset. File not found: {asset_path}",
                    status=NotificationStatus.ERROR,
                )
                raise RuntimeError("Asset not found")
        except Exception:
            # If normalize/stat fails for any reason, avoid creating broken references
            raise

        # One PI per asset to keep arrays simple for MVP. Name derived from asset basename
        asset_name = str(asset_path).split("/")[-1].split(".")[0]
        pi_path = parent_prim.GetPath().AppendPath(f"PI_{asset_name}")
        pi = stage.GetPrimAtPath(pi_path)
        if not pi.IsValid():
            UsdGeom.PointInstancer.Define(stage, pi_path)
            # Author prototype reference sub-prim once
            proto_root = pi_path.AppendPath("Prototypes").AppendPath(asset_name)
            if not stage.GetPrimAtPath(proto_root).IsValid():
                omni.kit.commands.execute(
                    "CreatePrimCommand",
                    prim_path=proto_root,
                    prim_type="Xform",
                    select_new_prim=False,
                    context_name=self._viewport_api.usd_context_name,
                )
                # Add reference to asset
                success, _ = omni.kit.commands.execute(
                    "AddReference",
                    stage=stage,
                    prim_path=str(proto_root),
                    reference=Sdf.Reference(assetPath=asset_path),
                )
                if not success:
                    post_notification(
                        "Scatter Brush: Failed to add reference to asset. The USD may be invalid or missing dependencies.",
                        status=NotificationStatus.ERROR,
                    )
                    # Clean up the created proto xform to avoid dangling prims
                    with contextlib.suppress(Exception):
                        omni.kit.commands.execute(
                            "DeletePrimsCommand",
                            paths=[str(proto_root)],
                            context_name=self._viewport_api.usd_context_name,
                        )
                    raise RuntimeError("Failed to reference asset")
            # Bind prototypes rel
            pi_geom = UsdGeom.PointInstancer(pi)
            rel = pi_geom.GetPrototypesRel()
            rel.SetTargets([proto_root])
        return pi

    def _append_instance(
        self,
        pi_prim: Usd.Prim,
        local_translation: Gf.Vec3d,
        parent_prim: Usd.Prim,
        picked_path: str,
        world_pos: Gf.Vec3d,
    ) -> None:
        pi = UsdGeom.PointInstancer(pi_prim)
        time_code = Usd.TimeCode.Default()
        # Read existing arrays
        positions = list(pi.GetPositionsAttr().Get(time_code) or [])
        proto_indices = list(pi.GetProtoIndicesAttr().Get(time_code) or [])
        orientations_attr = pi.GetOrientationsAttr()
        scales_attr = pi.GetScalesAttr()
        orientations = list(orientations_attr.Get(time_code) or [])
        scales = list(scales_attr.Get(time_code) or [])

        # Compute randomized yaw and scale (support both legacy and UI settings)
        yaw_min = self._settings_iface.get_as_float(self._settings_prefix_legacy + "/random_yaw_min_deg") or self._settings.random_yaw_min_deg
        yaw_max = self._settings_iface.get_as_float(self._settings_prefix_legacy + "/random_yaw_max_deg") or self._settings.random_yaw_max_deg
        # If UI provides a simple random_yaw toggle, use a default range
        try:
            random_yaw_enabled = bool(self._settings_iface.get(self._settings_prefix_tools + ".random_yaw"))
        except Exception:
            random_yaw_enabled = True
        if random_yaw_enabled and yaw_min == self._settings.random_yaw_min_deg and yaw_max == self._settings.random_yaw_max_deg:
            yaw_min, yaw_max = -15.0, 15.0

        scale_min = (
            self._settings_iface.get_as_float(self._settings_prefix_tools + ".random_scale_min")
            or self._settings_iface.get_as_float(self._settings_prefix_legacy + "/uniform_scale_min")
            or self._settings.uniform_scale_min
        )
        scale_max = (
            self._settings_iface.get_as_float(self._settings_prefix_tools + ".random_scale_max")
            or self._settings_iface.get_as_float(self._settings_prefix_legacy + "/uniform_scale_max")
            or self._settings.uniform_scale_max
        )

        import random, math

        yaw_deg = random.uniform(yaw_min, yaw_max)
        yaw_rad = math.radians(yaw_deg)

        # Determine orientation: optionally align to an estimated surface normal
        try:
            align_enabled = bool(self._settings_iface.get(self._settings_prefix_tools + ".align_to_normals"))
        except Exception:
            align_enabled = self._settings.align_to_normal
        max_angle = (
            self._settings_iface.get_as_float(self._settings_prefix_tools + ".max_surface_angle_deg")
            or 90.0
        )

        # Default normal is world up
        normal_world = Gf.Vec3d(0.0, 0.0, 1.0)
        mesh_root = self._mesh_root_from_path(picked_path) if picked_path else None
        if mesh_root is not None:
            # Use the mesh root transform's +Z as a proxy for surface normal
            try:
                mesh_prim = parent_prim.GetStage().GetPrimAtPath(mesh_root)
                xf = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                normal_world = xf.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
                if normal_world.GetLength() > 0:
                    normal_world = normal_world.GetNormalized()
                else:
                    normal_world = Gf.Vec3d(0.0, 0.0, 1.0)
            except Exception:
                normal_world = Gf.Vec3d(0.0, 0.0, 1.0)

        # Reject surfaces steeper than the max angle relative to world up
        try:
            dot = max(-1.0, min(1.0, float(Gf.Dot(normal_world, Gf.Vec3d(0.0, 0.0, 1.0)))))
            angle_deg = math.degrees(math.acos(dot))
            if angle_deg > max_angle:
                return
        except Exception:
            pass

        if align_enabled:
            # Rotate +Z to normal_world, then apply random twist about the normal
            base_rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), normal_world).GetQuat()
            twist_rot = Gf.Rotation(normal_world, yaw_deg if random_yaw_enabled else 0.0).GetQuat()
            qd = twist_rot * base_rot  # Quatd
            q = Gf.Quatf(float(qd.GetReal()), Gf.Vec3f(qd.GetImaginary()[0], qd.GetImaginary()[1], qd.GetImaginary()[2]))
        else:
            # Quaternion around Z for MVP (assuming up=Z in Remix capture)
            q = Gf.Quatf(float(math.cos(yaw_rad * 0.5)), Gf.Vec3f(0.0, 0.0, float(math.sin(yaw_rad * 0.5))))

        s = random.uniform(scale_min, scale_max)

        positions.append(Gf.Vec3f(local_translation))
        proto_indices.append(0)
        orientations.append(q)
        scales.append(Gf.Vec3f(s, s, s))

        # Write back arrays
        pi.GetPositionsAttr().Set(positions)
        pi.GetProtoIndicesAttr().Set(proto_indices)
        orientations_attr.Set(orientations)
        scales_attr.Set(scales)

    def _ensure_anchor_under_mesh(self, stage: Usd.Stage, mesh_root: Sdf.Path) -> Usd.Prim:
        """Create or return an anchor Xform under the mesh root in the replacement layer.

        The anchor is a child Xform with the Remix `IsRemixRef` attribute set to True so it
        behaves like other Remix-authored references. We then place the point instancer as
        a child of this anchor.
        """
        # Name the anchor deterministically
        anchor_path = mesh_root.AppendPath("scatter_anchor")
        prim = stage.GetPrimAtPath(anchor_path)
        if not prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrimCommand",
                prim_path=str(anchor_path),
                prim_type="Xform",
                select_new_prim=False,
                context_name=self._viewport_api.usd_context_name,
            )
            prim = stage.GetPrimAtPath(anchor_path)
            # Mark as Remix-created
            omni.kit.commands.execute(
                "CreateUsdAttributeOnPath",
                attr_path=prim.GetPath().AppendProperty(_constants.IS_REMIX_REF_ATTR),
                attr_type=Sdf.ValueTypeNames.Bool,
                attr_value=True,
                usd_context_name=self._viewport_api.usd_context_name,
            )
        return prim


def scatter_brush_factory(desc: dict[str, object]):
    viewport_api = desc.get("viewport_api")
    return ScatterBrush(viewport_api)
