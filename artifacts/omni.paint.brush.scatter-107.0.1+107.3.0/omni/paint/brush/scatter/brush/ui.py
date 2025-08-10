# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import copy
import os
from typing import Callable
import weakref
import carb
import carb.settings
import carb.events
import omni.kit.actions.core
import omni.kit.commands
import omni.kit.ui
import omni.kit.undo
import omni.kit.window
import omni.ui as ui
import omni.usd
import omni.paint.system.core as pc
from omni.paint.system.ui import ExpandPanel, create_standard_parameter_panel, SettingsBuilder, is_usd_file_path
from .scale_distribution import ScaleDistributionCtrl
from .constant import *
from .utils import *
from .brush import INSTANCING


def is_valid_prim_path(path):
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    return prim.IsA(UsdGeom.Xformable)


def is_valid_path(path):
    return path == "" or is_usd_file_path(path) or is_valid_prim_path(path)


def get_selections():
    selections = []
    # viewport
    for prim_selection in omni.usd.get_context().get_selection().get_selected_prim_paths():
        selections.append(prim_selection)

    # content_browser
    try:
        content_window = omni.kit.window.content_browser.get_content_window()
        content_selections = content_window.get_current_selections()
        for select in content_selections:
            if is_usd_file_path(select):
                selections.append(select)
    except Exception:
        pass

    # asset
    try:
        asset_browser = omni.kit.browser.asset.get_instance()
        if asset_browser:
            # OM-76853: we need a API to get selections in browser
            if hasattr(asset_browser._window, '_widget'):
                if hasattr(asset_browser._window._widget, '_browser_widget'):
                    widget = asset_browser._window._widget._browser_widget
                    if widget:
                        for select in widget.detail_selection:
                            if is_usd_file_path(select.url):
                                selections.append(select.url)
    except Exception:
        pass

    # asset store
    try:
        asset_store = omni.kit.browser.asset_store.get_instance()
        if asset_store:
            # OM-76853: we need a API to get selections in browser
            if hasattr(asset_store._window, '_widget'):
                if hasattr(asset_store._window._widget, 'detail_selection'):
                    for select in asset_store._window._widget.detail_selection:
                        if is_usd_file_path(select.url):
                            selections.append(select.url)
    except Exception:
        pass

    return selections


class AssetsPanel(ExpandPanel):
    FRAME_OFFSET = 90
    MIN_HEIGHT = 150
    MAX_DEFAULT_HEIGHT = 200

    def __init__(self, title, title_width=0, expand=True, on_param_changed_fn: callable = None):
        self._expand = expand
        self._on_param_changed_fn = on_param_changed_fn
        self._asset_panel_drag_height = ui.Pixel(self.MIN_HEIGHT + self.FRAME_OFFSET)
        self._usd_context = omni.usd.get_context()
        self._stage_sub = self._usd_context.get_stage_event_stream().create_subscription_to_pop(
            self._on_stage, name="scatter brush"
        )
        self._asset_slots = []
        self._brush_event_sub = pc.PainterEventStream().subscribe_to_event_stream_by_type(
            pc.PainterEventType.BRUSH_PARAM_CHANGED, self._on_brush_setting_changed
        )
        super().__init__(title, title_width, expand)

        # actions
        self._action_registry = omni.kit.actions.core.get_action_registry()
        self._action_registry.register_action("omni.paint.brush.scatter", "add_asset", self._on_add_asset, "Add Asset", "Add Asset")

    def __del__(self):
        self.destroy()

    def destroy(self):
        self._stage_sub = None
        for asset_slot in self._asset_slots:
            asset_slot.destroy()
        self._asset_slots.clear()
        self._action_registry.deregister_action("omni.paint.brush.scatter", "add_asset")

    def build_panel(self):
        self._panel.height = self._asset_panel_drag_height if self._expand else ui.Pixel(0)
        self._brush = None
        with ui.VStack(height=0):
            with ui.HStack(height=0):
                ui.Spacer(width=10)
                ui.Button(
                    f"{omni.kit.ui.get_custom_glyph_code('${glyphs}/menu_context.svg')} Add Asset",
                    width=ui.Fraction(1),
                    name="add",
                    clicked_fn=lambda *_: self._on_add_asset(),
                    accept_drop_fn=self._on_accept_drop,
                    drop_fn=self._on_drop,
                    tooltip="add setected prims in viewport and assets in browsers"
                )
                ui.Spacer(width=10)
            ui.Spacer(height=10)

            with ui.VStack(height=0):
                self._asset_panel = ui.ScrollingFrame(
                    name="asset_frame",
                    height=self._panel.height - self.FRAME_OFFSET,
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    style={"ScrollingFrame::asset_frame": {"secondary_color": 0xFF9E9E9E}},
                    accept_drop_fn=self._on_accept_drop,
                    drop_fn=self._on_drop,
                )

                self._asset_panel.set_build_fn(self._build_asset_panel)

    def update(self, brush):
        self._brush = brush
        self._clear_unavailable_stage_assets()

        # update panel with brush assets
        self._update_asset_panel()

    def has_asset_path(self, path):
        for asset in self._brush["assets"]:
            if asset["path"] == path:
                return True

        return False

    def set_panel_height(self, height):
        if self._panel.collapsed:
            return height

        if height.value < self.MIN_HEIGHT:
            height = ui.Pixel(self.MIN_HEIGHT)

        self._asset_panel_drag_height = height
        self._panel.height = self._asset_panel_drag_height
        self._asset_panel.height = ui.Pixel(height.value - self.FRAME_OFFSET)

        # return the actual height
        return height

    def get_panel_height(self):
        return self._asset_panel_drag_height

    def _on_accept_drop(self, path):
        return isinstance(path, str)

    def _on_drop(self, path):
        if self._brush is None:
            carb.log_warn("No brush selected.")
            return

        if self._on_accept_drop(path.mime_data):
            self._on_asset_selected(path.mime_data)

    def _on_add_asset(self):
        if self._brush is None:
            carb.log_warn("No brush selected.")
            return

        selected_paths = get_selections()
        for selected in selected_paths:
            self._on_asset_selected(selected)

        if len(selected_paths) == 0:
            self._on_asset_selected("")

    def _on_asset_selected(self, path):
        if self._brush is not None:
            if self.has_asset_path(path):
                carb.log_warn(f"Asset '{path}' already added")
                return

            if not is_valid_path(path):
                carb.log_warn(f"Asset '{path}' is not valid for paint")
                return

            asset = {"path": path, "thumbnail": "", "weight": 1.0, "enabled": True}

            assets_list = copy.copy(self._brush["assets"])
            assets_list.append(asset)

            omni.kit.commands.execute(
                "ChangeBrushParamCommand",
                brush=self._brush,
                param="assets",
                value=assets_list,
                prev_value=self._brush["assets"],
            )

    def _clear_unavailable_stage_assets(self):
        if self._asset_slots:
            usd_assets = set()
            for asset_slot in self._asset_slots:
                if not asset_slot.is_stage_path:
                    usd_assets.add(asset_slot.asset_path)

            new_asset_list = []
            for usd in usd_assets:
                for asset in self._brush["assets"]:
                    if asset["path"] == usd:
                        new_asset_list.append(asset)
                        break
            self._brush["assets"] = new_asset_list

    def on_stage_changed(self):
        self._clear_unavailable_stage_assets()
        self._update_asset_panel()

    # Callback for stage event
    def _on_stage(self, stage_event):
        if stage_event.type == int(omni.usd.StageEventType.OPENED):
            self.on_stage_changed()

    def _update_asset_panel(self):
        self._asset_panel.rebuild()

    def _build_asset_panel(self):
        for slot in self._asset_slots:
            slot.destroy()
        self._asset_slots.clear()
        if self._brush is None:
            return

        with ui.VStack(height=0, spacing=10):
            for i in range(len(self._brush["assets"])):
                SettingsBuilder.build_asset_setting(self._brush, "assets", i, None, widget_kwargs={"has_cancel": True})

    def _on_brush_setting_changed(self, event: carb.events.IEvent):
        key = event.payload["path"]

        if key.startswith("assets/"):
            self._update_asset_panel()


class ParamsUi:
    def __init__(self, brush, on_param_changed_fn, parent_window=None):
        self._brush = brush
        self._on_param_changed_fn = on_param_changed_fn
        brush_manager = pc.get_instance().get_brush_manager()
        self._original_brush = copy.deepcopy(brush_manager.get_brush_original(brush))

        ui.Spacer(height=5)

        self._brush_assets_panel = None
        self._brush_params_panel = None
        self._assets_panel_expand = True
        self._prev_drag_y_offset = None
        self._scale_distribution_ctrl = None

        if parent_window is not None:

            def on_window_width_change(self_proxy, change):
                if self_proxy:
                    self_proxy._on_window_width_changed(change)

            parent_window.set_width_changed_fn(lambda change: on_window_width_change(weakref.proxy(self), change))

        self._settings_frame = ui.Frame()
        self._settings_frame.set_build_fn(self._build_settings_fn)

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self._brush_assets_panel:
            self._brush_assets_panel.destroy()
            self._brush_assets_panel = None

        if self._scale_distribution_ctrl:
            self._scale_distribution_ctrl.destroy()
            self._scale_distribution_ctrl = None

        if self._brush_params_panel:
            self._brush_params_panel.destroy()
            self._brush_params_panel = None

    def _on_window_width_changed(self, change):
        if self._scale_distribution_ctrl:
            self._scale_distribution_ctrl.on_width_changed(change)

    def _build_settings_fn(self):
        with ui.VStack(spacing=0, height=0):
            with ui.ZStack(height=0):
                self._brush_assets_panel = AssetsPanel(
                    "Assets",
                    title_width=20,
                    expand=self._assets_panel_expand,
                    on_param_changed_fn=self._on_param_changed_fn,
                )

                self._brush_assets_panel.update(self._brush)

                # the dragable Splitter line between assets and parameters
                def on_dragged(length):
                    actual_height = self._brush_assets_panel.set_panel_height(length)
                    self._drag.offset_y = actual_height

                self._drag = ui.Placer(draggable=True, drag_axis=ui.Axis.Y, offset_y_changed_fn=on_dragged)

                with self._drag:
                    rec = ui.Rectangle(height=4, style_type_name_override="Splitter")

                if self._assets_panel_expand:
                    if self._prev_drag_y_offset is not None:
                        self._drag.offset_y = self._prev_drag_y_offset
                    else:
                        self._drag.offset_y = ui.Pixel(
                            min(
                                len(self._brush["assets"]) * 61 + AssetsPanel.FRAME_OFFSET,
                                AssetsPanel.MAX_DEFAULT_HEIGHT + AssetsPanel.FRAME_OFFSET,
                            )
                        )
                else:
                    self._drag.offset_y = 0

                rec.visible = not self._brush_assets_panel._panel.collapsed
                self._drag.draggable = not self._brush_assets_panel._panel.collapsed

                def on_assets_collapsed(collapsed: bool):
                    self._assets_panel_expand = not collapsed
                    if collapsed:
                        self._prev_drag_y_offset = self._drag.offset_y
                    self._settings_frame.rebuild()

                self._brush_assets_panel._panel.set_collapsed_changed_fn(on_assets_collapsed)

            with ui.VStack():
                ui.Spacer(height=5)
                self._brush_params_panel = create_standard_parameter_panel(
                    self._brush, on_param_changed_fn=self._on_brush_setting_changed
                )
                self._brush_params_panel.add_custom_ui("scale", self._build_scale_distribution_ui)
                self._brush_params_panel.rebuild()

    # Custom Build Scale Distribution Ctrl
    def _build_scale_distribution_ui(self, path: str, param, on_param_changed_fn: Callable):
        self._scale_distribution_ctrl = ScaleDistributionCtrl(self._brush, self._original_brush, on_param_changed_fn)
        return self._scale_distribution_ctrl

    def _on_brush_setting_changed(self, key, value):
        if key == "physics":
            omni.kit.undo.begin_group()

            if value:
                if self._brush["instancing"] == INSTANCING.POINT:
                    omni.kit.commands.execute(
                        "ChangeBrushParamCommand",
                        brush=self._brush,
                        param="instancing",
                        value=INSTANCING.ASSET,
                        prev_value=INSTANCING.POINT,
                    )
                    self._brush["prev_instancing"] = INSTANCING.POINT
            else:
                if self._brush.get("prev_instancing", "") == INSTANCING.POINT:
                    omni.kit.commands.execute(
                        "ChangeBrushParamCommand",
                        brush=self._brush,
                        param="instancing",
                        value=INSTANCING.POINT,
                        prev_value=self._brush["instancing"],
                    )
                    self._brush["prev_instancing"] = self._brush["instancing"]

            omni.kit.undo.end_group()
