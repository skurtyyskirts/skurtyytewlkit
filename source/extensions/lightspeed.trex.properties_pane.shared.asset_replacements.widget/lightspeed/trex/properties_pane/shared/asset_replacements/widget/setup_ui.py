"""
* SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import functools
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Tuple
import os

import omni.client
import carb
import omni.usd
from lightspeed.common.constants import GAME_READY_ASSETS_FOLDER as _GAME_READY_ASSETS_FOLDER
from lightspeed.common.constants import REMIX_CAPTURE_FOLDER as _REMIX_CAPTURE_FOLDER
from lightspeed.layer_manager.core import LayerManagerCore as _LayerManagerCore
from lightspeed.layer_manager.core import LayerType as _LayerType
from lightspeed.trex.material.core.shared import Setup as _MaterialCore
from lightspeed.trex.material_properties.shared.widget import SetupUI as _MaterialPropertiesWidget
from lightspeed.trex.mesh_properties.shared.widget import SetupUI as _MeshPropertiesWidget
from lightspeed.trex.replacement.core.shared import Setup as _AssetReplacementCore
from lightspeed.trex.replacement.core.shared.layers import AssetReplacementLayersCore as _AssetReplacementLayersCore
from lightspeed.trex.selection_tree.shared.widget import SetupUI as _SelectionTreeWidget
from lightspeed.trex.utils.common.prim_utils import is_instance as _is_instance
from lightspeed.trex.utils.widget import TrexMessageDialog as _TrexMessageDialog
from lightspeed.common import constants as _constants
from lightspeed.trex.utils.common.asset_utils import is_asset_ingested as _is_asset_ingested
from omni import ui
from omni.flux.bookmark_tree.model.usd import UsdBookmarkCollectionModel as _UsdBookmarkCollectionModel
from omni.flux.bookmark_tree.widget import BookmarkTreeWidget as _BookmarkTreeWidget
from omni.flux.layer_tree.usd.widget import LayerModel as _LayerModel
from omni.flux.layer_tree.usd.widget import LayerTreeWidget as _LayerTreeWidget
from omni.flux.utils.common import Event as _Event
from omni.flux.utils.common import EventSubscription as _EventSubscription
from omni.flux.utils.common import reset_default_attrs as _reset_default_attrs
from omni.flux.utils.widget.collapsable_frame import (
    PropertyCollapsableFrameWithInfoPopup as _PropertyCollapsableFrameWithInfoPopup,
)
from pxr import Sdf, Tf


class CollapsiblePanels(Enum):
    BOOKMARKS = 0
    HISTORY = 1
    LAYERS = 2
    MATERIAL_PROPERTIES = 3
    MESH_PROPERTIES = 4
    SELECTION = 5
    SCATTER_BRUSH = 6


class AssetReplacementsPane:
    def __init__(self, context_name: str):
        """Nvidia StageCraft Components Pane"""

        self._default_attr = {
            "_replacement_core": None,
            "_layers_core": None,
            "_root_frame": None,
            "_layer_tree_widget": None,
            "_bookmark_tree_widget": None,
            "_selection_history_widget": None,
            "_selection_tree_widget": None,
            "_mesh_properties_widget": None,
            "_material_converted_sub": None,
            "_material_properties_widget": None,
            "_layer_collapsable_frame": None,
            "_bookmarks_collapsable_frame": None,
            "_selection_history_collapsable_frame": None,
            "_selection_collapsable_frame": None,
            "_mesh_properties_collapsable_frame": None,
            "_material_properties_collapsable_frame": None,
            "_scatter_collapsable_frame": None,
            "_scatter_assets": None,
            "_scatter_asset_combo": None,
            "_sub_tree_selection_changed": None,
            "_sub_go_to_ingest_tab1": None,
            "_sub_go_to_ingest_tab2": None,
            "_sub_go_to_ingest_tab3": None,
            "_sub_stage_event": None,
            "_layer_validation_error_msg": None,
            "_collapsible_frame_states": None,
        }
        for attr, value in self._default_attr.items():
            setattr(self, attr, value)

        self._context_name = context_name
        self._replacement_core = _AssetReplacementCore(context_name)
        self._layers_core = _AssetReplacementLayersCore(context_name)
        self._material_core = _MaterialCore(context_name)

        self._material_converted_sub = None

        self._layer_validation_error_msg = ""

        self._collapsible_frame_states = {}

        self.__create_ui()

        self.__on_go_to_ingest_tab = _Event()

    def _go_to_ingest_tab(self):
        """Call the event object that has the list of functions"""
        self.__on_go_to_ingest_tab()

    def subscribe_go_to_ingest_tab(self, func):
        """
        Return the object that will automatically unsubscribe when destroyed.
        """
        return _EventSubscription(self.__on_go_to_ingest_tab, func)

    def __create_ui(self):
        self._root_frame = ui.Frame()
        with self._root_frame:
            with ui.ScrollingFrame(
                name="PropertiesPaneSection",
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
            ):
                with ui.VStack():
                    ui.Spacer(height=ui.Pixel(56))

                    with ui.HStack():
                        ui.Spacer(width=ui.Pixel(8), height=ui.Pixel(0))
                        with ui.VStack():
                            ui.Spacer(height=ui.Pixel(8))
                            self._layer_collapsable_frame = _PropertyCollapsableFrameWithInfoPopup(
                                "LAYERS",
                                info_text="Visual representation of the USD layers in the mod.\n\n"
                                "NOTE: USD parent layers aggregate children layers so they have stronger opinions.\n\n"
                                "- Layers can be drag/dropped to change the hierarchy\n"
                                "- The active edit target can be changed by clicking the layer icon aligned "
                                "with the layer\n"
                                "- Layers can be deleted by clicking the Delete button aligned with the "
                                "layer\n"
                                "- Layers can be locked by clicking the Lock button aligned with the layer\n"
                                "- Layers can be muted by clicking the Mute button aligned with the layer\n"
                                "- Layers can be saved by clicking the Save button if they are not locked "
                                "and muted\n"
                                "- New layers can be created by clicking the Create button at the bottom of "
                                "the widget\n"
                                "- Existing layers can be imported by clicking the Import button at the "
                                "bottom of the widget",
                                collapsed=True,
                            )
                            self._collapsible_frame_states[CollapsiblePanels.LAYERS] = False
                            with self._layer_collapsable_frame:
                                model = _LayerModel(
                                    self._context_name,
                                    layer_creation_validation_fn=functools.partial(self.__validate_file_path, False),
                                    layer_creation_validation_failed_callback=functools.partial(
                                        self.__validation_error_callback, False
                                    ),
                                    layer_import_validation_fn=functools.partial(self.__validate_file_path, True),
                                    layer_import_validation_failed_callback=functools.partial(
                                        self.__validation_error_callback, True
                                    ),
                                    exclude_remove_fn=self._layers_core.get_layers_exclude_remove,
                                    exclude_lock_fn=self._layers_core.get_layers_exclude_lock,
                                    exclude_mute_fn=self._layers_core.get_layers_exclude_mute,
                                    exclude_edit_target_fn=self._layers_core.get_layers_exclude_edit_target,
                                    exclude_add_child_fn=self._layers_core.get_layers_exclude_add_child,
                                    exclude_move_fn=self._layers_core.get_layers_exclude_move,
                                )
                                self._layer_tree_widget = _LayerTreeWidget(model=model)
                            self._layer_collapsable_frame.root.set_collapsed_changed_fn(
                                functools.partial(
                                    self.__on_collapsable_frame_changed,
                                    CollapsiblePanels.LAYERS,
                                    self._layer_tree_widget,
                                )
                            )

                            ui.Spacer(height=ui.Pixel(16))

                            # Scatter Brush settings panel
                            self._scatter_collapsable_frame = _PropertyCollapsableFrameWithInfoPopup(
                                "SCATTER BRUSH",
                                info_text=(
                                    "Scatter pre-ingested assets with a brush.\n"
                                    "- Select a USD asset from your project's assets/ingested folder.\n"
                                    "- Use the toolbar brush button to paint into the viewport.\n"
                                ),
                                collapsed=False,
                            )
                            self._collapsible_frame_states[CollapsiblePanels.SCATTER_BRUSH] = True
                            with self._scatter_collapsable_frame:
                                with ui.VStack(spacing=6, style={"margin": 4}):
                                    ui.Label("Asset (ingested)", style_type_name_override="PropertiesWidgetLabel")
                                    self._scatter_assets = self._scan_ingested_assets()
                                    names = [name for name, _ in self._scatter_assets] or ["<no ingested USD assets found>"]
                                    self._scatter_asset_combo = ui.ComboBox(
                                        0, *names, style_type_name_override="PropertiesWidgetField"
                                    )
                                    self._scatter_asset_combo.model.add_value_changed_fn(
                                        self._on_scatter_asset_changed
                                    )

                                    ui.Spacer(height=ui.Pixel(4))
                                    with ui.HStack(spacing=6):
                                        ui.Label("Radius", width=ui.Pixel(80),
                                                 style_type_name_override="PropertiesWidgetLabel")
                                        radius = ui.FloatDrag(0.25, min=0.0, max=100.0, step=0.05,
                                                              style_type_name_override="PropertiesWidgetField")
                                        radius.model.add_value_changed_fn(
                                            lambda *_: self._set_setting("/radius", radius.model.as_float)
                                        )
                                    with ui.HStack(spacing=6):
                                        ui.Label("Spacing", width=ui.Pixel(80),
                                                 style_type_name_override="PropertiesWidgetLabel")
                                        spacing = ui.FloatDrag(0.25, min=0.0, max=100.0, step=0.05,
                                                               style_type_name_override="PropertiesWidgetField")
                                        spacing.model.add_value_changed_fn(
                                            lambda *_: self._set_setting("/spacing", spacing.model.as_float)
                                        )
                                    with ui.HStack(spacing=6):
                                        ui.Label("Density", width=ui.Pixel(80),
                                                 style_type_name_override="PropertiesWidgetLabel")
                                        density = ui.FloatDrag(1.0, min=0.01, max=50.0, step=0.1,
                                                              style_type_name_override="PropertiesWidgetField")
                                        density.model.add_value_changed_fn(
                                            lambda *_: self._set_setting("/density", density.model.as_float)
                                        )
                                    with ui.HStack(spacing=6):
                                        ui.Label("Yaw Min", width=ui.Pixel(80),
                                                 style_type_name_override="PropertiesWidgetLabel")
                                        yaw_min = ui.FloatDrag(-15.0, min=-180.0, max=180.0, step=1.0,
                                                              style_type_name_override="PropertiesWidgetField")
                                        yaw_min.model.add_value_changed_fn(
                                            lambda *_: self._set_setting("/random_yaw_min_deg", yaw_min.model.as_float)
                                        )
                                    with ui.HStack(spacing=6):
                                        ui.Label("Yaw Max", width=ui.Pixel(80),
                                                 style_type_name_override="PropertiesWidgetLabel")
                                        yaw_max = ui.FloatDrag(15.0, min=-180.0, max=180.0, step=1.0,
                                                              style_type_name_override="PropertiesWidgetField")
                                        yaw_max.model.add_value_changed_fn(
                                            lambda *_: self._set_setting("/random_yaw_max_deg", yaw_max.model.as_float)
                                        )
                                    with ui.HStack(spacing=6):
                                        ui.Label("Scale Min", width=ui.Pixel(80),
                                                 style_type_name_override="PropertiesWidgetLabel")
                                        smin = ui.FloatDrag(1.0, min=0.01, max=10.0, step=0.05,
                                                            style_type_name_override="PropertiesWidgetField")
                                        smin.model.add_value_changed_fn(
                                            lambda *_: self._set_setting("/uniform_scale_min", smin.model.as_float)
                                        )
                                    with ui.HStack(spacing=6):
                                        ui.Label("Scale Max", width=ui.Pixel(80),
                                                 style_type_name_override="PropertiesWidgetLabel")
                                        smax = ui.FloatDrag(1.0, min=0.01, max=10.0, step=0.05,
                                                            style_type_name_override="PropertiesWidgetField")
                                        smax.model.add_value_changed_fn(
                                            lambda *_: self._set_setting("/uniform_scale_max", smax.model.as_float)
                                        )
                                    with ui.HStack(spacing=6):
                                        ui.Label("Align Normal", width=ui.Pixel(80),
                                                 style_type_name_override="PropertiesWidgetLabel")
                                        align = ui.CheckBox(False)
                                        align.model.add_value_changed_fn(
                                            lambda *_: self._set_setting("/align_to_normal", align.model.as_bool)
                                        )
                                    with ui.HStack(spacing=6):
                                        ui.Label("Normal Offset", width=ui.Pixel(80),
                                                 style_type_name_override="PropertiesWidgetLabel")
                                        offset = ui.FloatDrag(0.0, min=-1.0, max=1.0, step=0.01,
                                                             style_type_name_override="PropertiesWidgetField")
                                        offset.model.add_value_changed_fn(
                                            lambda *_: self._set_setting("/offset_along_normal", offset.model.as_float)
                                        )

                            self._bookmarks_collapsable_frame = _PropertyCollapsableFrameWithInfoPopup(
                                "BOOKMARKS",
                                info_text="A tree of all the bookmark collections and bookmarked items.\n\n"
                                "- Creating a bookmark will automatically parent the new item to the "
                                "currently selected collection.\n"
                                "- Both Collections and Items are sorted alphabetically.\n"
                                "- Both Items and Collections can be drag/dropped to change the hierarchy.\n"
                                "- Collections can be renamed by double-clicking on them.\n"
                                "- Items can be added or removed from collections by using the add/remove "
                                "buttons aligned with the desired collection.\n"
                                "- Selecting an item in the viewport will change the bookmark selection if "
                                "the item was bookmarked.\n"
                                "- Selecting an item in the bookmarks will select the associated item in the "
                                "viewport.\n",
                                collapsed=True,
                            )
                            self._collapsible_frame_states[CollapsiblePanels.BOOKMARKS] = False
                            with self._bookmarks_collapsable_frame:
                                model = _UsdBookmarkCollectionModel(self._context_name)
                                self._bookmark_tree_widget = _BookmarkTreeWidget(model=model)
                            self._bookmarks_collapsable_frame.root.set_collapsed_changed_fn(
                                functools.partial(
                                    self.__on_collapsable_frame_changed,
                                    CollapsiblePanels.BOOKMARKS,
                                    self._bookmark_tree_widget,
                                )
                            )

                            # TODO REMIX-4102: Re-enable after reworking the selection history
                            # ui.Spacer(height=ui.Pixel(16))
                            #
                            # self._selection_history_collapsable_frame = _PropertyCollapsableFrameWithInfoPopup(
                            #     "SELECTION HISTORY",
                            #     info_text=f"- The history has a maximum length of "
                            #     f"{_UsdSelectionHistoryModel.MAX_LIST_LENGTH}.\n"
                            #     "- Clicking on an item will select it in the viewport.\n",
                            #     collapsed=True,
                            # )
                            # self._collapsible_frame_states[CollapsiblePanels.HISTORY] = False
                            # with self._selection_history_collapsable_frame:
                            #     model = _UsdSelectionHistoryModel(self._context_name)
                            #     self._selection_history_widget = _SelectionHistoryWidget(model=model)
                            # self._selection_history_collapsable_frame.root.set_collapsed_changed_fn(
                            #     functools.partial(
                            #         self.__on_collapsable_frame_changed,
                            #         CollapsiblePanels.HISTORY,
                            #         self._selection_history_widget,
                            #     )
                            # )

                            ui.Spacer(height=ui.Pixel(16))

                            self._selection_collapsable_frame = _PropertyCollapsableFrameWithInfoPopup(
                                "SELECTION",
                                info_text="Tree that will shows your current viewport selection.\n\n"
                                "- The top item is the 'prototype'. A prototype is what represent the asset\n"
                                "- The prototype contains USD reference(s) (linked USD file(s)).\n"
                                "- Multiple USD reference(s) can be added or removed.\n"
                                "- USD prim hierarchy of each reference can be seen.\n"
                                "- Clicking on a prim in the hierarchy will select the prin in the viewport.\n"
                                "- 'Instances' item shows where the asset is instanced in your stage.\n"
                                "- Clicking on an instance (with a prim item selected in the hierarchy) will select "
                                "it in the viewport.\n",
                                collapsed=False,
                            )
                            self._collapsible_frame_states[CollapsiblePanels.SELECTION] = True
                            with self._selection_collapsable_frame:
                                self._selection_tree_widget = _SelectionTreeWidget(self._context_name)

                            ui.Spacer(height=ui.Pixel(16))

                            self._mesh_properties_collapsable_frame = _PropertyCollapsableFrameWithInfoPopup(
                                "OBJECT PROPERTIES",
                                info_text="Mesh properties of the selected mesh(es).\n\n"
                                "- Will show different property depending of the selection from the "
                                "selection panel above. If the selection panel is hiding, it will show nothing.\n"
                                "- A blue circle tells that the property has a different value than the default one.\n"
                                "- A darker background tells that the property has override(s) from layer(s).\n"
                                "- Override(s) can be removed. The list shows the stronger layer (top) to "
                                "the weaker layer (bottom).\n",
                                collapsed=False,
                                pinnable=True,
                                pinned_text_fn=self._get_selection_pin_name,
                                unpinned_fn=self._refresh_mesh_properties_widget,
                            )
                            self._collapsible_frame_states[CollapsiblePanels.MESH_PROPERTIES] = True
                            with self._mesh_properties_collapsable_frame:
                                self._mesh_properties_widget = _MeshPropertiesWidget(self._context_name)
                            self._mesh_properties_collapsable_frame.root.set_collapsed_changed_fn(
                                functools.partial(
                                    self.__on_collapsable_frame_changed,
                                    CollapsiblePanels.MESH_PROPERTIES,
                                    self._mesh_properties_widget,
                                    refresh_fn=self._refresh_mesh_properties_widget,
                                )
                            )

                            ui.Spacer(height=ui.Pixel(16))

                            self._material_properties_collapsable_frame = _PropertyCollapsableFrameWithInfoPopup(
                                "MATERIAL PROPERTIES",
                                info_text="Material properties of the selected mesh(es).\n\n"
                                "- Will show material properties depending of the selection from the "
                                "selection panel above. If the selection panel is hiding, it will show nothing.\n"
                                "- A blue circle tells that the property has a different value than the default one.\n"
                                "- A darker background tells that the property has override(s) from layer(s).\n"
                                "- Override(s) can be removed. The list shows the stronger layer (top) to "
                                "the weaker layer (bottom).\n",
                                collapsed=False,
                                pinnable=True,
                                pinned_text_fn=lambda: self._get_selection_pin_name(for_materials=True),
                                unpinned_fn=self._refresh_material_properties_widget,
                            )
                            self._collapsible_frame_states[CollapsiblePanels.MATERIAL_PROPERTIES] = True
                            with self._material_properties_collapsable_frame:
                                self._material_properties_widget = _MaterialPropertiesWidget(self._context_name)
                                self._material_converted_sub = (
                                    self._material_properties_widget.subscribe_on_material_changed(
                                        self._refresh_material_properties_widget
                                    )
                                )
                            self._material_properties_collapsable_frame.root.set_collapsed_changed_fn(
                                functools.partial(
                                    self.__on_collapsable_frame_changed,
                                    CollapsiblePanels.MATERIAL_PROPERTIES,
                                    self._material_properties_widget,
                                    refresh_fn=self._refresh_material_properties_widget,
                                )
                            )

                            ui.Spacer(height=ui.Pixel(16))

        self._sub_tree_selection_changed = self._selection_tree_widget.subscribe_tree_selection_changed(
            self._on_tree_selection_changed
        )

        self._sub_go_to_ingest_tab1 = self._mesh_properties_widget.subscribe_go_to_ingest_tab(self._go_to_ingest_tab)
        self._sub_go_to_ingest_tab2 = self._selection_tree_widget.subscribe_go_to_ingest_tab(self._go_to_ingest_tab)
        self._sub_go_to_ingest_tab3 = self._material_properties_widget.subscribe_go_to_ingest_tab(
            self._go_to_ingest_tab
        )

        self._refresh_mesh_properties_widget()
        self._refresh_material_properties_widget()
        self._apply_initial_scatter_settings()

    def __on_collapsable_frame_changed(
        self, collapsible_panel_type, widget, collapsed, refresh_fn: Callable[[], Any] = None
    ):
        value = not collapsed
        self._collapsible_frame_states[collapsible_panel_type] = value
        widget.show(value)
        if refresh_fn is not None and value:
            refresh_fn()

    @property
    def selection_tree_widget(self):
        return self._selection_tree_widget

    # -------- Scatter helpers --------
    def _settings_iface(self):
        return carb.settings.get_settings()

    def _settings_prefix(self) -> str:
        return "/exts/lightspeed.trex.scatter_brush"

    def _set_setting(self, key_suffix: str, value: Any):
        try:
            self._settings_iface().set(self._settings_prefix() + key_suffix, value)
        except Exception:  # noqa
            pass

    def _apply_initial_scatter_settings(self):
        # Ensure defaults exist to avoid None reads in the tool
        defaults = {
            "/radius": 0.25,
            "/spacing": 0.25,
            "/density": 1.0,
            "/random_yaw_min_deg": -15.0,
            "/random_yaw_max_deg": 15.0,
            "/uniform_scale_min": 1.0,
            "/uniform_scale_max": 1.0,
            "/align_to_normal": False,
            "/offset_along_normal": 0.0,
        }
        for k, v in defaults.items():
            if self._settings_iface().get(self._settings_prefix() + k) is None:
                self._set_setting(k, v)
        # Also set current asset if any
        if self._scatter_assets:
            self._on_scatter_asset_changed()

    def _on_scatter_asset_changed(self, *_):
        if not self._scatter_assets or self._scatter_asset_combo is None:
            return
        idx = int(self._scatter_asset_combo.model.get_value_as_int())
        idx = max(0, min(idx, len(self._scatter_assets) - 1))
        _, path = self._scatter_assets[idx]
        self._set_setting("/asset_path", path)

    def _scan_ingested_assets(self) -> List[Tuple[str, str]]:
        ctx = omni.usd.get_context(self._context_name)
        stage_url = ctx.get_stage_url()
        if not stage_url:
            return []
        # Project root is parent folder of stage file
        root_url = omni.client.normalize_url(str(omni.client.Uri(stage_url).get_dirname()))
        ingested_url = omni.client.combine_urls(root_url, _constants.REMIX_INGESTED_ASSETS_FOLDER)
        results: List[Tuple[str, str]] = []

        def walk(url: str):
            res, entries = omni.client.list(url)
            if res != omni.client.Result.OK:
                return
            for e in entries:
                child = omni.client.combine_urls(url, e.relative_path)
                if e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
                    walk(child)
                elif e.flags & omni.client.ItemFlags.READABLE_FILE:
                    lower = e.relative_path.lower()
                    if any(lower.endswith(ext) for ext in _constants.USD_EXTENSIONS):
                        if _is_asset_ingested(child):
                            results.append((e.relative_path, child))
        walk(ingested_url)
        # Sort by name
        results.sort(key=lambda t: t[0].lower())
        return results

    def _get_selection_pin_name(self, for_materials: bool = False) -> str:
        """
        Get a formatted name of the current USD selection for the pin label.

        Args:
            for_materials: Whether to target relevant materials from the selection or use the selection directly.

        Returns:
            A formatted string of the `parent/prim` with handling of no selection, multi-selection, and long paths.
        """
        context = omni.usd.get_context(self._context_name)
        stage = context.get_stage()

        prims = []
        prim_paths = list(set(context.get_selection().get_selected_prim_paths()))
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim:
                continue

            # If material pinning, get mesh from instance to later grab all relevant materials
            if for_materials and _is_instance(prim):
                prim = self._material_properties_widget.get_mesh_from_instance(prim)
                if not prim:
                    continue

            prims.append(prim)

        # Use materials relevant to USD selection for material pinning
        if for_materials:
            prims = [
                stage.GetPrimAtPath(path) for path in self._material_properties_widget.get_materials_from_prims(prims)
            ]

        # Return the appropriate prim name
        if not prims:
            return "None Selected"
        if len(prims) == 1:
            path = prims[0].GetPath()
            formatted_name = f"{path.GetParentPath().name}/{path.name}"  # parent/child
            return formatted_name if len(formatted_name) < 50 else "..." + formatted_name[-50:]  # 50 char limit
        return "Multiple Selected"

    def __validate_existing_layer(self, path):
        try:
            sublayer = Sdf.Layer.FindOrOpen(str(path))
        except Tf.ErrorException:
            sublayer = None
        if not sublayer:
            self._layer_validation_error_msg = f"Unable to open layer {path.name}"
            return False
        # Check if the layer type is a reserved layer type
        layer_manager = _LayerManagerCore(self._context_name)
        layer_type = layer_manager.get_custom_data_layer_type(sublayer)
        if layer_type in [
            _LayerType.replacement.value,
            _LayerType.capture.value,
            _LayerType.capture_baker.value,
            _LayerType.workfile.value,
        ]:
            self._layer_validation_error_msg = (
                f"Layer {path.name}'s layer type ({layer_type}) is reserved by Remix, and cannot be loaded."
            )
            return False

        # Check if the layer is already used
        all_layers = layer_manager.get_layers(layer_type=None)
        if sublayer in all_layers:
            self._layer_validation_error_msg = f"Layer {path.name} is already loaded."
            return False

        return True

    def __validate_file_path(self, existing_file, dirname, filename):
        result = self._replacement_core.is_path_valid(
            omni.client.normalize_url(omni.client.combine_urls(dirname, filename)),
            existing_file=existing_file,
        )
        if result and existing_file and not self.__validate_existing_layer(Path(dirname, filename)):
            return False
        return result

    def __validation_error_callback(self, existing_file, *_):
        fill_word = "imported" if existing_file else "created"
        message = f"The {fill_word} layer file is not valid.\n\n"
        if not self._layer_validation_error_msg:
            message = (
                f"{message}Make sure the {fill_word} layer is a writable USD file and is not located in a "
                f'"{_GAME_READY_ASSETS_FOLDER}" or "{_REMIX_CAPTURE_FOLDER}" directory.\n'
            )
        else:
            message = f"{message}{self._layer_validation_error_msg}"
        _TrexMessageDialog(
            message=message,
            disable_cancel_button=True,
        )

    def _on_tree_selection_changed(self, items):
        if not self._mesh_properties_collapsable_frame.root.collapsed:
            self._refresh_mesh_properties_widget()
        if not self._material_properties_collapsable_frame.root.collapsed:
            self._refresh_material_properties_widget()
        self._mesh_properties_collapsable_frame.root.rebuild()

    def _refresh_mesh_properties_widget(self):
        if self._mesh_properties_collapsable_frame.pinned:
            return
        items = self._selection_tree_widget.get_selection()
        self._mesh_properties_widget.refresh(items)

    def _refresh_material_properties_widget(self):
        if self._material_properties_collapsable_frame.pinned:
            return

        # Grab the selection prims and refresh the properties
        context = omni.usd.get_context(self._context_name)
        prim_paths = list(set(context.get_selection().get_selected_prim_paths()))
        items = [context.get_stage().GetPrimAtPath(prim_path) for prim_path in prim_paths]
        self._material_properties_widget.refresh(items)

    def refresh(self):
        if not self._root_frame.visible:
            return
        self._selection_tree_widget.refresh()
        self._refresh_mesh_properties_widget()
        self._refresh_material_properties_widget()

    def show(self, value):
        # Update the widget visibility
        self._root_frame.visible = value
        self._layer_tree_widget.show(self._collapsible_frame_states[CollapsiblePanels.LAYERS] and value)
        self._bookmark_tree_widget.show(self._collapsible_frame_states[CollapsiblePanels.BOOKMARKS] and value)
        # self._selection_history_widget.show(self._collapsible_frame_states[CollapsiblePanels.HISTORY] and value)
        self._selection_tree_widget.show(self._collapsible_frame_states[CollapsiblePanels.SELECTION] and value)
        self._mesh_properties_widget.show(self._collapsible_frame_states[CollapsiblePanels.MESH_PROPERTIES] and value)
        self._material_properties_widget.show(
            self._collapsible_frame_states[CollapsiblePanels.MATERIAL_PROPERTIES] and value
        )
        if value:
            self.refresh()

    def destroy(self):
        _reset_default_attrs(self)
