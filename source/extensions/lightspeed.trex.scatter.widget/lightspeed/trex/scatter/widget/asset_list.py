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

__all__ = ["ScatterAssetDelegate", "ScatterAssetItem", "ScatterAssetListWidget", "ScatterAssetModel"]

import asyncio
import functools
from collections.abc import Sequence
from pathlib import PurePosixPath

import carb.events
import omni.kit.app
import omni.usd
from lightspeed.trex.asset_replacements.core.shared import Setup as AssetReplacementsCore
from lightspeed.trex.asset_replacements.core.shared.data_models import ReplacementAssetType
from lightspeed.trex.scatter.core import (
    ScatterAssetEntry,
    ScatterBrushController,
    UpAxis,
    asset_crate_supported,
    get_scatter_brush_controller,
    list_ingested_models,
    resolve_reference_asset_from_selection,
)
from lightspeed.trex.utils.widget import accept_asset_if_valid_for_replacement, open_asset_file_picker
from omni import ui
from omni.flux.utils.widget.drag_field import FloatBoundedDrag

from .combo_model import ChoicesComboModel
from .value_model import DragValueModel, field_bounds

_UP_AXES = tuple(UpAxis)
_WEIGHT_MIN, _WEIGHT_MAX = field_bounds(ScatterAssetEntry, "weight")


class ScatterAssetItem(ui.AbstractItem):
    """Tree row for one palette entry; its value models drive the row widgets.

    The row also owns the weight drag field the delegate builds for it, so the clamp hook that field installs on
    ``weight_model`` is released together with the row.

    Attributes:
        entry: Palette entry the row shows. Replaced through ``update``.
        enabled_model: Bound to the enabled check box.
        path_model: Asset path, for callers that read the tree through ``get_item_value_model``.
        weight_model: Bound to the weight drag field, which installs its hard clamp on it.
        up_axis_model: Bound to the up-axis combo box; its index follows ``UpAxis`` order.
    """

    def __init__(self, entry: ScatterAssetEntry):
        super().__init__()
        self.entry = entry
        self.enabled_model = ui.SimpleBoolModel(entry.enabled)
        self.path_model = ui.SimpleStringModel(entry.path)
        self.weight_model = DragValueModel(float(entry.weight))
        self.up_axis_model = ChoicesComboModel([axis.value for axis in _UP_AXES], _UP_AXES.index(entry.up_axis))
        self._weight_field: FloatBoundedDrag | None = None

    @property
    def path(self) -> str:
        """Asset path of the entry."""
        return self.entry.path

    @property
    def display_name(self) -> str:
        """File name shown in the row; the full path stays in the tooltip."""
        return PurePosixPath(self.entry.path).name

    def update(self, entry: ScatterAssetEntry) -> None:
        """Adopt a new entry for the same path and push its values into the models that differ."""
        self.entry = entry
        if self.enabled_model.as_bool != entry.enabled:
            self.enabled_model.set_value(entry.enabled)
        if self.weight_model.as_float != entry.weight:
            self.weight_model.set_value(float(entry.weight))
        index = _UP_AXES.index(entry.up_axis)
        if self.up_axis_model.index_model.as_int != index:
            self.up_axis_model.index_model.set_value(index)

    def attach_weight_field(self, field: FloatBoundedDrag) -> None:
        """Own the drag field built for the weight cell, releasing the one a previous build left behind."""
        self.destroy()
        self._weight_field = field

    def destroy(self) -> None:
        """Release the weight drag field; the tree drops the row widgets themselves."""
        if self._weight_field is not None:
            self._weight_field.destroy()
            self._weight_field = None


class ScatterAssetModel(ui.AbstractItemModel):
    """Flat tree model mirroring ``ScatterBrushController.assets`` and writing row edits back to the controller.

    Every row model change is forwarded as soon as it happens; the round trip is idempotent because the controller
    echoes the same values back and the rows skip values they already hold. Palette changes that keep the same paths
    are pushed into the existing rows so the widget being edited is not rebuilt under the cursor; additions and
    removals rebuild the rows.
    """

    def __init__(self, controller: ScatterBrushController):
        super().__init__()
        self._controller = controller
        self._items: list[ScatterAssetItem] = []
        self._item_subscriptions: list[carb.events.ISubscription] = []
        self._pushing = False
        self._assets_changed_sub = controller.subscribe_assets_changed(self.refresh)
        self.refresh(controller.assets)

    def refresh(self, entries: Sequence[ScatterAssetEntry]) -> None:
        """Mirror the palette entries, updating rows in place when only their values changed."""
        if [entry.path for entry in entries] == [item.path for item in self._items]:
            for item, entry in zip(self._items, entries, strict=True):
                self._push(item, entry)
            return
        self._item_subscriptions.clear()
        for item in self._items:
            item.destroy()
        self._items = [ScatterAssetItem(entry) for entry in entries]
        for item in self._items:
            self._item_subscriptions.extend(
                (
                    item.enabled_model.subscribe_value_changed_fn(functools.partial(self._on_enabled_changed, item)),
                    item.weight_model.subscribe_value_changed_fn(functools.partial(self._on_weight_changed, item)),
                    item.up_axis_model.index_model.subscribe_value_changed_fn(
                        functools.partial(self._on_up_axis_changed, item)
                    ),
                )
            )
        self._item_changed(None)

    def get_item_children(self, item: ui.AbstractItem | None = None) -> list[ScatterAssetItem]:
        """Return the rows for the root and nothing for a row."""
        return list(self._items) if item is None else []

    def get_item_value_model_count(self, item: ui.AbstractItem | None = None) -> int:
        """Return the column count: enabled, path, weight and up axis."""
        return 4

    def get_item_value_model(
        self, item: ScatterAssetItem | None = None, column_id: int = 0
    ) -> ui.AbstractValueModel | None:
        """Return the row model behind a column, or None for the root."""
        if item is None:
            return None
        return (item.enabled_model, item.path_model, item.weight_model, item.up_axis_model.index_model)[column_id]

    def destroy(self) -> None:
        """Drop the controller subscription and the rows with their widgets."""
        self._assets_changed_sub = None
        self._item_subscriptions.clear()
        for item in self._items:
            item.destroy()
        self._items = []

    def _push(self, item: ScatterAssetItem, entry: ScatterAssetEntry) -> None:
        """Update a row from the controller without echoing the change back."""
        self._pushing = True
        try:
            item.update(entry)
        finally:
            self._pushing = False

    def _restore(self, item: ScatterAssetItem) -> None:
        """Put the controller's values back into a row after it rejected an edit."""
        entry = next((entry for entry in self._controller.assets if entry.path == item.path), None)
        if entry is not None:
            self._push(item, entry)

    def _on_enabled_changed(self, item: ScatterAssetItem, model: ui.AbstractValueModel) -> None:
        """Forward an enabled toggle to the controller."""
        if self._pushing:
            return
        if not self._controller.set_asset_enabled(item.path, model.as_bool):
            self._restore(item)

    def _on_weight_changed(self, item: ScatterAssetItem, model: ui.AbstractValueModel) -> None:
        """Forward a weight change to the controller."""
        if self._pushing:
            return
        if not self._controller.set_asset_weight(item.path, model.as_float):
            self._restore(item)

    def _on_up_axis_changed(self, item: ScatterAssetItem, model: ui.AbstractValueModel) -> None:
        """Forward an up-axis selection to the controller."""
        if self._pushing:
            return
        if not self._controller.set_asset_up_axis(item.path, _UP_AXES[model.as_int]):
            self._restore(item)


class ScatterAssetDelegate(ui.AbstractItemDelegate):
    """Builds the enabled, asset, weight and up-axis cells of a ``ScatterAssetItem`` row."""

    _ROW_HEIGHT = ui.Pixel(24)
    _SPACING = ui.Pixel(4)
    _HEADERS = ("", "Asset", "Weight", "Up")

    def build_header(self, column_id: int = 0) -> None:
        """Build the title cell of a column."""
        with ui.HStack(height=self._ROW_HEIGHT, spacing=self._SPACING):
            ui.Spacer(width=0)
            ui.Label(self._HEADERS[column_id], name="ColumnHeader")

    def build_branch(
        self,
        model: ui.AbstractItemModel,
        item: ui.AbstractItem | None = None,
        column_id: int = 0,
        level: int = 0,
        expanded: bool = False,
    ) -> None:
        """Build nothing; the list is flat."""

    def build_widget(
        self,
        model: ui.AbstractItemModel,
        item: ScatterAssetItem | None = None,
        column_id: int = 0,
        level: int = 0,
        expanded: bool = False,
    ) -> None:
        """Build the cell of a row bound to the item's value models.

        Cell identifiers follow ``scatter_asset_row_<index>_<column>`` so UI tests can address one row. The weight
        drag field is handed to the item, which releases it with the row.
        """
        if item is None:
            return
        identifier_prefix = f"scatter_asset_row_{model.get_item_children(None).index(item)}"
        with ui.HStack(height=self._ROW_HEIGHT, spacing=self._SPACING):
            ui.Spacer(width=0)
            match column_id:
                case 0:
                    with ui.VStack(width=0):
                        ui.Spacer()
                        ui.CheckBox(
                            item.enabled_model,
                            width=0,
                            height=0,
                            tooltip="Include this asset when painting and flooding",
                            identifier=f"{identifier_prefix}_enabled",
                        )
                        ui.Spacer()
                case 1:
                    ui.Label(
                        item.display_name,
                        name="PropertiesPaneSectionTreeItem",
                        tooltip=item.path,
                        elided_text=True,
                        identifier=f"{identifier_prefix}_path",
                    )
                case 2:
                    item.attach_weight_field(
                        FloatBoundedDrag(
                            model=item.weight_model,
                            hard_min_value=_WEIGHT_MIN,
                            hard_max_value=_WEIGHT_MAX,
                            step=0.1,
                            tooltip="Relative chance of this asset being chosen for a placement",
                            identifier=f"{identifier_prefix}_weight",
                            # The row model applies edits directly; leaving batch edit on makes the drag query the
                            # model again while it is garbage collected, which logs a warning per row.
                            enable_batch_edit=False,
                        )
                    )
                case 3:
                    ui.ComboBox(
                        item.up_axis_model,
                        tooltip="Up axis authored in the asset file; placements are uprighted on the stage",
                        identifier=f"{identifier_prefix}_up_axis",
                    )
            ui.Spacer(width=0)


class ScatterAssetListWidget:
    """Asset palette of the brush: a tree of ``ScatterAssetItem`` rows and add, browse and remove controls.

    Every added asset goes through ``accept_asset_if_valid_for_replacement`` so assets that were not ingested or live
    outside the project show the shared dialogs before they reach ``ScatterBrushController.add_asset``, and assets
    saved with a USD crate version this build cannot read are refused with a status message.
    """

    _TREE_HEIGHT = ui.Pixel(140)
    _ROW_HEIGHT = ui.Pixel(24)
    _SPACING = ui.Pixel(8)
    _COLUMN_WIDTHS = (ui.Pixel(32), ui.Fraction(1), ui.Pixel(72), ui.Pixel(64))
    _BROWSE_PLACEHOLDER = "Browse ingested models..."

    def __init__(self, context_name: str):
        self._context_name = context_name
        self._controller = get_scatter_brush_controller()
        self._asset_core = AssetReplacementsCore(context_name)
        self._model = ScatterAssetModel(self._controller)
        self._delegate: ScatterAssetDelegate | None = ScatterAssetDelegate()
        self._ingested_models: list[str] = []
        self._refreshing_browse = False
        self._browse_refresh_task: asyncio.Task | None = None
        self._browse_model = ChoicesComboModel([self._BROWSE_PLACEHOLDER])
        self._browse_index_sub = self._browse_model.index_model.subscribe_value_changed_fn(self._on_browse_selected)
        self._stage_event_sub = (
            omni.usd.get_context(context_name)
            .get_stage_event_stream()
            .create_subscription_to_pop(self._on_stage_event, name="lightspeed.trex.scatter.widget asset list")
        )
        self._tree_view: ui.TreeView | None = None
        self._browse_combo: ui.ComboBox | None = None
        self._remove_button: ui.Button | None = None
        self._root = ui.VStack(height=0, spacing=self._SPACING)
        self.__build_ui()
        self.refresh_ingested_models()

    @property
    def model(self) -> ScatterAssetModel:
        """Tree model mirroring the controller's palette."""
        return self._model

    def refresh_ingested_models(self) -> None:
        """Re-list the project's ingested models; the combo box entries are only replaced when the names differ."""
        self._ingested_models = list_ingested_models(self._context_name)
        choices = [self._BROWSE_PLACEHOLDER, *(PurePosixPath(path).name for path in self._ingested_models)]
        if choices == self._browse_model.choices:
            return
        self._refreshing_browse = True
        try:
            self._browse_model.set_choices(choices, 0)
        finally:
            self._refreshing_browse = False

    def destroy(self) -> None:
        """Release subscriptions, the pending re-list, the model, the delegate and the widgets."""
        if self._browse_refresh_task is not None:
            self._browse_refresh_task.cancel()
            self._browse_refresh_task = None
        self._stage_event_sub = None
        self._browse_index_sub = None
        self._model.destroy()
        self._delegate = None
        self._browse_model.destroy()
        self._asset_core.destroy()
        self._tree_view = None
        self._browse_combo = None
        self._remove_button = None
        self._root = None

    def __build_ui(self):
        with self._root:
            with ui.ZStack(height=self._TREE_HEIGHT):
                ui.Rectangle(name="TreePanelBackground")
                with ui.ScrollingFrame(
                    name="PropertiesPaneSection",
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                ):
                    self._tree_view = ui.TreeView(
                        self._model,
                        delegate=self._delegate,
                        root_visible=False,
                        header_visible=True,
                        columns_resizable=True,
                        column_widths=list(self._COLUMN_WIDTHS),
                        selection_changed_fn=self._on_selection_changed,
                        identifier="scatter_asset_list",
                    )
            with ui.HStack(height=self._ROW_HEIGHT, spacing=self._SPACING):
                ui.Button(
                    "Add",
                    clicked_fn=self._open_file_picker,
                    tooltip="Choose an ingested asset file to scatter",
                    identifier="scatter_asset_add",
                )
                ui.Button(
                    "Add Selection",
                    clicked_fn=self._add_from_selection,
                    tooltip="Add the asset referenced by the selected prim",
                    identifier="scatter_asset_add_selection",
                )
                self._browse_combo = ui.ComboBox(
                    self._browse_model,
                    mouse_pressed_fn=self._on_browse_pressed,
                    tooltip="Add a model from the project's ingested assets",
                    identifier="scatter_asset_browse_ingested",
                )
                self._remove_button = ui.Button(
                    "Remove",
                    clicked_fn=self._remove_selected,
                    enabled=False,
                    tooltip="Remove the selected assets from the brush",
                    identifier="scatter_asset_remove",
                )

    def _open_file_picker(self) -> None:
        """Open the shared asset file picker for USD models."""
        open_asset_file_picker("Select an asset to scatter", ReplacementAssetType.MESH, self._accept, lambda *_: None)

    def _add_from_selection(self) -> None:
        """Add the asset referenced by the selected prim, or explain why nothing was added."""
        path = resolve_reference_asset_from_selection(omni.usd.get_context(self._context_name))
        if path is None:
            self._controller.post_status("Select a prim that references an ingested asset", is_error=True)
            return
        self._accept(path)

    def _remove_selected(self) -> None:
        """Remove the selected rows from the palette."""
        for item in list(self._tree_view.selection):
            self._controller.remove_asset(item.path)

    def _accept(self, path: str) -> None:
        """Run the replacement validation dialogs on a path and add it to the palette when accepted."""
        usd_context = omni.usd.get_context(self._context_name)
        stage = usd_context.get_stage()
        if stage is None:
            self._controller.post_status("Open a project before adding assets to the brush", is_error=True)
            return
        accept_asset_if_valid_for_replacement(
            path,
            stage.GetEditTarget().GetLayer(),
            self._asset_core,
            usd_context,
            accept_handler=self._add_accepted_asset,
        )

    def _add_accepted_asset(self, path: str) -> None:
        """Add a validated asset to the palette, unless its USD crate version is newer than this build can read."""
        if not asset_crate_supported(path):
            self._controller.post_status(
                f"{path} was saved with a USD crate version this build cannot read", is_error=True
            )
            return
        self._controller.add_asset(path)

    def _on_selection_changed(self, selection: list[ScatterAssetItem]) -> None:
        """Enable the remove button only while rows are selected."""
        self._remove_button.enabled = bool(selection)

    def _on_browse_pressed(self, *_) -> None:
        """Schedule a re-list for the next frame.

        Listing the ingested directory is a blocking walk through ``omni.client``, and replacing the combo box
        entries from inside its own mouse event would swap the model the widget is handling that event for.
        """
        if self._browse_refresh_task is None or self._browse_refresh_task.done():
            self._browse_refresh_task = asyncio.ensure_future(self._refresh_ingested_models_next_frame())

    @omni.usd.handle_exception
    async def _refresh_ingested_models_next_frame(self) -> None:
        """Re-list the ingested models once the click that asked for it has been handled."""
        await omni.kit.app.get_app().next_update_async()
        if self._root is not None:
            self.refresh_ingested_models()

    def _on_browse_selected(self, model: ui.AbstractValueModel) -> None:
        """Add the chosen ingested model and put the combo box back on its placeholder."""
        if self._refreshing_browse:
            return
        index = model.as_int
        if index <= 0 or index > len(self._ingested_models):
            return
        path = self._ingested_models[index - 1]
        self._refreshing_browse = True
        try:
            model.set_value(0)
        finally:
            self._refreshing_browse = False
        self._accept(path)

    def _on_stage_event(self, event: carb.events.IEvent) -> None:
        """Re-list the ingested models when a project opens."""
        if event.type == int(omni.usd.StageEventType.OPENED):
            self.refresh_ingested_models()
