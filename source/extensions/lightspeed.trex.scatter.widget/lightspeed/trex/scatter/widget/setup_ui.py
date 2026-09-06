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

__all__ = ["ScatterPane"]

import dataclasses
import functools
import secrets
from collections.abc import Callable
from enum import Enum

import omni.usd
from lightspeed.trex.scatter.core import (
    ApplyTo,
    EraseScope,
    Falloff,
    MeshSurfaceCache,
    ScatterBrushSettings,
    ScatterMode,
    TargetMode,
    get_scatter_brush_controller,
    instance_count,
    validated_anchor_prototype,
)
from lightspeed.trex.utils.widget import TrexMessageDialog as _TrexMessageDialog
from lightspeed.trex.utils.widget import WorkspaceWidget as _WorkspaceWidget
from omni import ui
from omni.flux.utils.widget.collapsable_frame import PropertyCollapsableFrame as _PropertyCollapsableFrame
from omni.flux.utils.widget.collapsable_frame import (
    PropertyCollapsableFrameWithInfoPopup as _PropertyCollapsableFrameWithInfoPopup,
)
from omni.flux.utils.widget.drag_field import FloatBoundedDrag as _FloatBoundedDrag
from omni.flux.utils.widget.drag_field import IntBoundedDrag as _IntBoundedDrag

from .asset_list import ScatterAssetListWidget
from .presets_ui import PresetsWidget
from .value_model import DragValueModel as _DragValueModel
from .value_model import field_bounds

_ANCHOR_NONE_TEXT = "<none>"
_STATUS_STYLE = "PropertiesPaneSectionTreeItem"
_STATUS_ERROR_STYLE = "PropertiesPaneSectionTreeItemError"
# Floods above the default cap can take a while to author and render, so they ask for confirmation first.
_FLOOD_CONFIRM_THRESHOLD = ScatterBrushSettings.model_fields["flood_max_instances"].default


def _enum_labels(members: tuple[Enum, ...]) -> list[str]:
    """Return the combo box labels for enum members, for example ``HIT_SURFACE`` becomes ``Hit Surface``."""
    return [str(member.value).replace("_", " ").title() for member in members]


@dataclasses.dataclass
class _Binding:
    """Two-way link between one settings field and the value model of its control.

    Attributes:
        model: Value model the control edits.
        read: Converts the model value into the settings value.
        write: Pushes a settings value into the model.
    """

    model: ui.AbstractValueModel
    read: Callable[[ui.AbstractValueModel], object]
    write: Callable[[ui.AbstractValueModel, object], None]


class ScatterPane(_WorkspaceWidget):
    """Scatter window content: the brush mode header and the settings sections bound to the brush controller.

    Every control edits ``ScatterBrushController.settings`` through ``update_settings`` and follows the controller's
    ``settings_changed`` event, so the pane, the presets and the viewport tool never disagree. Controls carry
    ``scatter_<name>`` identifiers for UI tests.
    """

    _ROW_HEIGHT = ui.Pixel(24)
    _HEADER_HEIGHT = ui.Pixel(32)
    _LABEL_WIDTH = ui.Percent(45)
    _SPACING_XS = ui.Pixel(4)
    _SPACING_MD = ui.Pixel(8)
    _SPACING_LG = ui.Pixel(16)

    def __init__(self, context_name: str):
        """Build the pane inside the current ``omni.ui`` container.

        Args:
            context_name: USD context whose stage the brush paints; floods and the anchor selection use it too.
        """
        super().__init__()
        self._context_name = context_name
        self._controller = get_scatter_brush_controller()
        self._surface_cache = MeshSurfaceCache(self._get_stage)
        self._syncing = False
        self._instance_warning_key: tuple[TargetMode, str] | None = None
        self._anchor_instance_count = 0

        self._bindings: dict[str, _Binding] = {}
        self._field_subs: list = []
        self._drag_fields: list = []
        self._section_frames: list = []

        self._root_widget: ui.Frame | None = None
        self._paint_model = ui.SimpleBoolModel(self._controller.mode == ScatterMode.PAINT)
        self._erase_model = ui.SimpleBoolModel(self._controller.mode == ScatterMode.ERASE)
        self._paint_button: ui.ToolButton | None = None
        self._erase_button: ui.ToolButton | None = None
        self._flood_button: ui.Button | None = None
        self._status_label: ui.Label | None = None
        self._per_axis_scale_stack: ui.VStack | None = None
        self._seed_field: _IntBoundedDrag | None = None
        self._reroll_button: ui.Button | None = None
        self._anchor_path_label: ui.Label | None = None
        self._anchor_use_selection_button: ui.Button | None = None
        self._flood_estimate_label: ui.Label | None = None
        self._instance_warning_label: ui.Label | None = None
        self._asset_list_widget: ScatterAssetListWidget | None = None
        self._presets_widget: PresetsWidget | None = None

        self.__create_ui()

        self._paint_sub = self._paint_model.subscribe_value_changed_fn(
            functools.partial(self._on_mode_toggled, ScatterMode.PAINT)
        )
        self._erase_sub = self._erase_model.subscribe_value_changed_fn(
            functools.partial(self._on_mode_toggled, ScatterMode.ERASE)
        )
        self._settings_changed_sub = self._controller.subscribe_settings_changed(self._on_settings_changed)
        self._mode_changed_sub = self._controller.subscribe_mode_changed(self._on_mode_changed)
        self._assets_changed_sub = self._controller.subscribe_assets_changed(self._on_assets_changed)
        self._status_message_sub = self._controller.subscribe_status_message(self._on_status_message)

    def __create_ui(self):
        settings = self._controller.settings
        sections = (
            self.__build_brush_section,
            self.__build_placement_section,
            self.__build_scale_section,
            self.__build_random_section,
            self.__build_target_section,
            self.__build_assets_section,
            self.__build_presets_section,
        )
        self._root_widget = ui.Frame()
        with self._root_widget:
            with ui.ScrollingFrame(
                name="WorkspaceBackground",
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
            ):
                with ui.VStack():
                    ui.Spacer(height=self._SPACING_XS)
                    with ui.HStack():
                        ui.Spacer(width=self._SPACING_XS)
                        with ui.VStack():
                            self.__build_header()
                            for build_section in sections:
                                ui.Spacer(height=self._SPACING_LG)
                                build_section(settings)
                            ui.Spacer(height=self._SPACING_LG)
                        ui.Spacer(width=self._SPACING_XS)
                    ui.Spacer(height=self._SPACING_XS)
        self.__refresh_derived(settings, force_instance_warning=True)

    def __build_header(self):
        with ui.HStack(height=self._HEADER_HEIGHT, spacing=self._SPACING_MD):
            self._paint_button = ui.ToolButton(
                model=self._paint_model,
                text="Paint",
                identifier="scatter_mode_paint",
                tooltip="Paint the brush assets onto the surface under the cursor. Shift+drag erases instead.",
            )
            self._erase_button = ui.ToolButton(
                model=self._erase_model,
                text="Erase",
                identifier="scatter_mode_erase",
                tooltip="Erase scattered placements under the brush.",
            )
            self._flood_button = ui.Button(
                "Flood",
                identifier="scatter_flood",
                tooltip="Fill the anchor prototype or the selected captured meshes with placements in one undo step.",
                clicked_fn=self._on_flood_clicked,
            )
        ui.Spacer(height=self._SPACING_XS)
        self._status_label = ui.Label(
            "",
            identifier="scatter_status",
            height=0,
            word_wrap=True,
            style_type_name_override=_STATUS_STYLE,
        )

    def __build_section(self, title: str, info_text: str) -> _PropertyCollapsableFrameWithInfoPopup:
        frame = _PropertyCollapsableFrameWithInfoPopup(title, info_text=info_text, collapsed=False)
        self._section_frames.append(frame)
        return frame

    def __build_brush_section(self, settings: ScatterBrushSettings):
        frame = self.__build_section(
            "BRUSH",
            "Size and shape of one brush stamp.\n\n"
            "- Radius is in stage units\n"
            "- Density is the number of candidates per stamp and Strength the share of them that is kept\n"
            "- Spacing is the distance between stamps along a stroke\n"
            "- Padding keeps placements apart from each other",
        )
        with frame:
            with ui.VStack(height=0, spacing=self._SPACING_XS):
                self.__build_float_row("Radius", "radius", "scatter_radius", 1.0, "Brush radius in stage units.")
                self.__build_enum_row(
                    "Falloff", "falloff", "scatter_falloff", Falloff, "Acceptance curve from the center to the edge."
                )
                self.__build_float_row(
                    "Density", "density", "scatter_density", 0.5, "Candidates drawn in the brush disk per stamp."
                )
                self.__build_float_row(
                    "Strength", "strength", "scatter_strength", 0.01, "Share of the candidates that is accepted."
                )
                self.__build_float_row(
                    "Spacing", "stamp_spacing", "scatter_spacing", 1.0, "Distance between two stamps along a stroke."
                )
                self.__build_float_row(
                    "Padding", "padding", "scatter_padding", 0.5, "Minimum distance between two placements."
                )

    def __build_placement_section(self, settings: ScatterBrushSettings):
        frame = self.__build_section(
            "PLACEMENT",
            "How each placement sits on the surface.\n\n"
            "- Conform aligns the asset up axis with the surface normal\n"
            "- Align to stroke turns the asset along the stroke direction\n"
            "- Rotation ranges are in degrees; the advanced ranges tilt the asset around X and Y",
        )
        with frame:
            with ui.VStack(height=0, spacing=self._SPACING_XS):
                self.__build_float_row(
                    "Vertical Offset",
                    "vertical_offset",
                    "scatter_vertical_offset",
                    0.5,
                    "Offset along the surface normal in stage units.",
                )
                self.__build_bool_row(
                    "Conform to Surface",
                    "conform_to_surface",
                    "scatter_conform",
                    "Align the asset up axis with the surface normal.",
                )
                self.__build_bool_row(
                    "Align to Stroke",
                    "align_to_stroke",
                    "scatter_align_stroke",
                    "Turn the asset to face the stroke direction.",
                )
                self.__build_float_row(
                    "Rotation Z Min", "rotation_z_min", "scatter_rot_z_min", 1.0, "Lowest random yaw in degrees."
                )
                self.__build_float_row(
                    "Rotation Z Max", "rotation_z_max", "scatter_rot_z_max", 1.0, "Highest random yaw in degrees."
                )
                advanced_frame = _PropertyCollapsableFrame("ADVANCED ROTATION", collapsed=True)
                self._section_frames.append(advanced_frame)
                with advanced_frame:
                    with ui.VStack(height=0, spacing=self._SPACING_XS):
                        self.__build_float_row(
                            "Rotation X Min", "rotation_x_min", "scatter_rot_x_min", 1.0, "Lowest tilt around X."
                        )
                        self.__build_float_row(
                            "Rotation X Max", "rotation_x_max", "scatter_rot_x_max", 1.0, "Highest tilt around X."
                        )
                        self.__build_float_row(
                            "Rotation Y Min", "rotation_y_min", "scatter_rot_y_min", 1.0, "Lowest tilt around Y."
                        )
                        self.__build_float_row(
                            "Rotation Y Max", "rotation_y_max", "scatter_rot_y_max", 1.0, "Highest tilt around Y."
                        )

    def __build_scale_section(self, settings: ScatterBrushSettings):
        frame = self.__build_section(
            "SCALE",
            "Random scale applied to each placement.\n\n"
            "- Bias pushes the distribution toward the minimum (-1) or the maximum (1)\n"
            "- Weight sharpens (above 1) or flattens (below 1) the distribution\n"
            "- Disable Uniform to set a range per axis",
        )
        with frame:
            with ui.VStack(height=0, spacing=self._SPACING_XS):
                self.__build_bool_row(
                    "Random Scale", "scale_enabled", "scatter_scale_enabled", "Apply a random scale to placements."
                )
                self.__build_bool_row(
                    "Uniform", "scale_uniform", "scatter_scale_uniform", "Use one range for all three axes."
                )
                self.__build_float_row("Scale Min", "scale_min", "scatter_scale_min", 0.01, "Lowest uniform scale.")
                self.__build_float_row("Scale Max", "scale_max", "scatter_scale_max", 0.01, "Highest uniform scale.")
                self.__build_float_row(
                    "Bias", "scale_bias", "scatter_scale_bias", 0.01, "Skew toward the minimum (-1) or maximum (1)."
                )
                self.__build_float_row(
                    "Weight", "scale_weight", "scatter_scale_weight", 0.1, "Sharpness of the scale distribution."
                )
                self._per_axis_scale_stack = ui.VStack(height=0, spacing=self._SPACING_XS)
                with self._per_axis_scale_stack:
                    for axis in ("x", "y", "z"):
                        axis_label = axis.upper()
                        self.__build_float_row(
                            f"Scale {axis_label} Min",
                            f"scale_{axis}_min",
                            f"scatter_scale_{axis}_min",
                            0.01,
                            f"Lowest scale along {axis_label}.",
                        )
                        self.__build_float_row(
                            f"Scale {axis_label} Max",
                            f"scale_{axis}_max",
                            f"scatter_scale_{axis}_max",
                            0.01,
                            f"Highest scale along {axis_label}.",
                        )

    def __build_random_section(self, settings: ScatterBrushSettings):
        frame = self.__build_section(
            "RANDOM",
            "Seed of the stroke sampler.\n\n"
            "- With Randomize every stroke draws a fresh seed\n"
            "- A fixed seed replays the same pattern for the same stroke",
        )
        with frame:
            with ui.VStack(height=0, spacing=self._SPACING_XS):
                with ui.HStack(height=self._ROW_HEIGHT, spacing=self._SPACING_MD):
                    seed_tooltip = "Seed used when Randomize is off."
                    ui.Label("Seed", width=self._LABEL_WIDTH, tooltip=seed_tooltip)
                    self._seed_field = self.__build_int_field("seed", "scatter_seed", 1, seed_tooltip)
                    self._reroll_button = ui.Button(
                        "Reroll",
                        width=ui.Pixel(64),
                        identifier="scatter_reroll",
                        tooltip="Draw a new random seed.",
                        clicked_fn=self._on_reroll_clicked,
                    )
                self.__build_bool_row(
                    "Randomize", "randomize_seed", "scatter_randomize_seed", "Draw a fresh seed for every stroke."
                )

    def __build_target_section(self, settings: ScatterBrushSettings):
        frame = self.__build_section(
            "TARGET",
            "Which captured meshes receive placements.\n\n"
            "- Hit Surface authors under the mesh under the cursor\n"
            "- Anchor always authors under the anchor prototype; Use Selection sets it from the viewport selection\n"
            "- Flood fills the anchor or the selected meshes up to the cap\n\n"
            "Placements authored under a prototype appear under every instance of that prototype.",
        )
        with frame:
            with ui.VStack(height=0, spacing=self._SPACING_XS):
                self.__build_enum_row(
                    "Apply To", "apply_to", "scatter_apply_to", ApplyTo, "Paint on every mesh or only on the selection."
                )
                self.__build_enum_row(
                    "Target Mode",
                    "target_mode",
                    "scatter_target_mode",
                    TargetMode,
                    "Author under the hit mesh or always under the anchor prototype.",
                )
                with ui.HStack(height=self._ROW_HEIGHT, spacing=self._SPACING_MD):
                    ui.Label("Anchor", width=self._LABEL_WIDTH, tooltip="Prototype used by the Anchor target mode.")
                    self._anchor_path_label = ui.Label(
                        _ANCHOR_NONE_TEXT,
                        identifier="scatter_anchor_path",
                        elided_text=True,
                        tooltip="Prototype used by the Anchor target mode.",
                    )
                    self._anchor_use_selection_button = ui.Button(
                        "Use Selection",
                        width=0,
                        identifier="scatter_anchor_use_selection",
                        tooltip="Use the first selected captured mesh or instance as the anchor.",
                        clicked_fn=self._on_anchor_use_selection_clicked,
                    )
                self.__build_enum_row(
                    "Erase Scope",
                    "erase_scope",
                    "scatter_erase_scope",
                    EraseScope,
                    "Erase every scattered placement or only the brush assets.",
                )
                with ui.HStack(height=self._ROW_HEIGHT, spacing=self._SPACING_MD):
                    ui.Label("Flood Cap", width=self._LABEL_WIDTH, tooltip="Maximum placements created by one flood.")
                    self.__build_int_field(
                        "flood_max_instances", "scatter_flood_cap", 10, "Maximum placements created by one flood."
                    )
                    self._flood_estimate_label = ui.Label(
                        "",
                        width=0,
                        identifier="scatter_flood_estimate",
                        tooltip=(
                            "Most prims one flood adds: the flood cap, multiplied in Anchor mode by the number of "
                            "instances that show the anchor prototype."
                        ),
                    )
                self._instance_warning_label = ui.Label(
                    "",
                    name="Warning",
                    identifier="scatter_instance_warning",
                    height=0,
                    word_wrap=True,
                    visible=False,
                )

    def __build_assets_section(self, settings: ScatterBrushSettings):
        frame = self.__build_section(
            "ASSETS",
            "Ingested assets the brush chooses from, weighted by their Weight.\n\n"
            "- Disabled rows are skipped\n"
            "- Up Axis corrects assets authored with a different up axis than the stage",
        )
        with frame:
            self._asset_list_widget = ScatterAssetListWidget(self._context_name)

    def __build_presets_section(self, settings: ScatterBrushSettings):
        frame = self.__build_section("PRESETS", "Named brush presets stored as files in the user documents folder.")
        with frame:
            self._presets_widget = PresetsWidget()

    def __build_row(self, label: str, tooltip: str, build_control: Callable[[], None]) -> ui.HStack:
        row = ui.HStack(height=self._ROW_HEIGHT, spacing=self._SPACING_MD)
        with row:
            ui.Label(label, width=self._LABEL_WIDTH, tooltip=tooltip)
            build_control()
        return row

    def __build_float_row(self, label: str, field_name: str, identifier: str, step: float, tooltip: str) -> ui.HStack:
        return self.__build_row(label, tooltip, lambda: self.__build_float_field(field_name, identifier, step, tooltip))

    def __build_bool_row(self, label: str, field_name: str, identifier: str, tooltip: str) -> ui.HStack:
        return self.__build_row(label, tooltip, lambda: self.__build_bool_field(field_name, identifier, tooltip))

    def __build_enum_row(
        self, label: str, field_name: str, identifier: str, enum_type: type[Enum], tooltip: str
    ) -> ui.HStack:
        return self.__build_row(
            label, tooltip, lambda: self.__build_enum_field(field_name, identifier, enum_type, tooltip)
        )

    def __build_float_field(self, field_name: str, identifier: str, step: float, tooltip: str) -> _FloatBoundedDrag:
        low, high = field_bounds(ScatterBrushSettings, field_name)
        model = _DragValueModel(float(getattr(self._controller.settings, field_name)))
        field = _FloatBoundedDrag(
            model=model,
            min=low,
            max=high,
            step=step,
            hard_min_value=low,
            hard_max_value=high,
            identifier=identifier,
            tooltip=tooltip,
            # The model applies edits directly; leaving batch edit on makes the drag query the model again while
            # it is garbage collected, which logs a warning per field.
            enable_batch_edit=False,
        )
        self._drag_fields.append(field)
        self.__bind(field_name, model, lambda m: m.as_float, lambda m, value: m.set_value(float(value)), end_edit=True)
        return field

    def __build_int_field(self, field_name: str, identifier: str, step: int, tooltip: str) -> _IntBoundedDrag:
        low, high = field_bounds(ScatterBrushSettings, field_name)
        model = _DragValueModel(int(getattr(self._controller.settings, field_name)))
        field = _IntBoundedDrag(
            model=model,
            min=low,
            max=high,
            step=step,
            hard_min_value=low,
            hard_max_value=high,
            identifier=identifier,
            tooltip=tooltip,
            # Same reason as the float fields: no batch edit, no model query during garbage collection.
            enable_batch_edit=False,
        )
        self._drag_fields.append(field)
        self.__bind(field_name, model, lambda m: m.as_int, lambda m, value: m.set_value(int(value)), end_edit=True)
        return field

    def __build_bool_field(self, field_name: str, identifier: str, tooltip: str) -> ui.CheckBox:
        model = ui.SimpleBoolModel(getattr(self._controller.settings, field_name))
        checkbox = ui.CheckBox(model=model, width=0, identifier=identifier, tooltip=tooltip)
        ui.Spacer()
        self.__bind(field_name, model, lambda m: m.as_bool, lambda m, value: m.set_value(bool(value)))
        return checkbox

    def __build_enum_field(self, field_name: str, identifier: str, enum_type: type[Enum], tooltip: str) -> ui.ComboBox:
        members = tuple(enum_type)
        current = getattr(self._controller.settings, field_name)
        combo = ui.ComboBox(members.index(current), *_enum_labels(members), identifier=identifier, tooltip=tooltip)
        self.__bind(
            field_name,
            combo.model.get_item_value_model(),
            lambda m: members[m.as_int],
            lambda m, value: m.set_value(members.index(value)),
        )
        return combo

    def __bind(
        self,
        field_name: str,
        model: ui.AbstractValueModel,
        read: Callable[[ui.AbstractValueModel], object],
        write: Callable[[ui.AbstractValueModel, object], None],
        end_edit: bool = False,
    ):
        self._bindings[field_name] = _Binding(model=model, read=read, write=write)
        on_edit = functools.partial(self._on_field_edited, field_name)
        self._field_subs.append(model.subscribe_value_changed_fn(on_edit))
        if end_edit:
            self._field_subs.append(model.subscribe_end_edit_fn(on_edit))

    def _get_stage(self):
        """Return the stage of the pane's USD context, or None when the context or the stage does not exist."""
        usd_context = omni.usd.get_context(self._context_name)
        return usd_context.get_stage() if usd_context is not None else None

    def _on_field_edited(self, field_name: str, model: ui.AbstractValueModel):
        """Forward a user edit to the controller; a rejected value is rolled back to the controller's value.

        The controller notifies every change back through ``settings_changed``, so edits made while the pane itself
        pushes values, and edits that do not change the value, are ignored to keep the round trip idempotent.

        Args:
            field_name: ``ScatterBrushSettings`` field bound to the model.
            model: Value model that changed.
        """
        if self._syncing:
            return
        value = self._bindings[field_name].read(model)
        if value == getattr(self._controller.settings, field_name):
            return
        if not self._controller.update_settings(**{field_name: value}):
            self.__write_field(field_name, getattr(self._controller.settings, field_name))

    def _on_mode_toggled(self, mode: ScatterMode, model: ui.AbstractValueModel):
        """Switch the controller mode when a header toggle button changes and re-sync both buttons.

        Args:
            mode: Mode the toggled button stands for.
            model: Bool model of the toggled button.
        """
        if self._syncing:
            return
        if model.as_bool:
            self._controller.set_mode(mode)
        elif self._controller.mode == mode:
            self._controller.set_mode(ScatterMode.OFF)
        self.__push_mode(self._controller.mode)

    def _on_settings_changed(self, settings: ScatterBrushSettings):
        """Push controller settings into the controls while the pane is visible.

        Args:
            settings: Settings the controller adopted.
        """
        if self.window_visible:
            self.__push_settings(settings)

    def _on_mode_changed(self, mode: ScatterMode):
        """Reflect a controller mode change on the header buttons while the pane is visible.

        Args:
            mode: New interaction mode.
        """
        if self.window_visible:
            self.__push_mode(mode)

    def _on_assets_changed(self, _assets: list):
        """Refresh the derived labels when the asset palette changes while the pane is visible.

        The palette does not change which prototype is anchored, so the instance count keeps its cached value.
        """
        if self.window_visible:
            self.__refresh_derived(self._controller.settings)

    def _on_status_message(self, text: str, is_error: bool):
        """Show the latest controller status line, in the error style when it reports a problem.

        Args:
            text: Status text meant for the user.
            is_error: Whether the status reports a problem.
        """
        if self._status_label is None:
            return
        self._status_label.text = text
        self._status_label.style_type_name_override = _STATUS_ERROR_STYLE if is_error else _STATUS_STYLE

    def _on_flood_clicked(self):
        """Flood the default targets, after a confirmation when the cap exceeds the default."""
        cap = self._controller.settings.flood_max_instances
        flood = functools.partial(self._controller.flood, self._context_name, self._surface_cache)
        if cap <= _FLOOD_CONFIRM_THRESHOLD:
            flood()
            return
        _TrexMessageDialog(
            message=(
                f"Flooding can create up to {cap} placements, which may take a while to author and to render.\n\n"
                "Continue with the flood?"
            ),
            title="Flood Surfaces",
            ok_label="Flood",
            ok_handler=flood,
        )

    def _on_reroll_clicked(self):
        """Replace the seed with a fresh 31-bit value."""
        self._controller.update_settings(seed=secrets.randbits(31))

    def _on_anchor_use_selection_clicked(self):
        """Make the first selected prototype the anchor; the controller reports an empty selection."""
        self._controller.set_anchor_from_selection(self._context_name)

    def show(self, visible: bool):
        """Show or hide the pane; showing re-reads the controller so changes made while hidden are reflected.

        Args:
            visible: Whether the containing window is visible.
        """
        super().show(visible)
        if self._root_widget is not None:
            self._root_widget.visible = visible
        if visible:
            self.__push_settings(self._controller.settings, force_instance_warning=True)
            self.__push_mode(self._controller.mode)

    def destroy(self):
        """Release the controller subscriptions, the surface cache and the child widgets."""
        self._settings_changed_sub = None
        self._mode_changed_sub = None
        self._assets_changed_sub = None
        self._status_message_sub = None
        self._paint_sub = None
        self._erase_sub = None
        self._field_subs.clear()
        self._bindings.clear()
        for field in self._drag_fields:
            field.destroy()
        self._drag_fields.clear()
        for frame in self._section_frames:
            frame.destroy()
        self._section_frames.clear()
        for widget in (self._asset_list_widget, self._presets_widget):
            if widget is not None:
                widget.destroy()
        self._asset_list_widget = None
        self._presets_widget = None
        if self._surface_cache is not None:
            self._surface_cache.destroy()
            self._surface_cache = None
        self._root_widget = None
        self._mark_destroyed()

    def __write_field(self, field_name: str, value: object):
        binding = self._bindings.get(field_name)
        if binding is None:
            return
        self._syncing = True
        try:
            binding.write(binding.model, value)
        finally:
            self._syncing = False

    def __push_settings(self, settings: ScatterBrushSettings, force_instance_warning: bool = False):
        self._syncing = True
        try:
            for field_name, binding in self._bindings.items():
                binding.write(binding.model, getattr(settings, field_name))
        finally:
            self._syncing = False
        self.__refresh_derived(settings, force_instance_warning=force_instance_warning)

    def __push_mode(self, mode: ScatterMode):
        self._syncing = True
        try:
            self._paint_model.set_value(mode == ScatterMode.PAINT)
            self._erase_model.set_value(mode == ScatterMode.ERASE)
        finally:
            self._syncing = False

    def __refresh_derived(self, settings: ScatterBrushSettings, force_instance_warning: bool = False):
        self._per_axis_scale_stack.visible = not settings.scale_uniform
        seed_editable = not settings.randomize_seed
        self._seed_field.enabled = seed_editable
        self._reroll_button.enabled = seed_editable
        self._anchor_path_label.text = settings.anchor_prototype_path or _ANCHOR_NONE_TEXT
        self.__refresh_instance_warning(settings, force_instance_warning)
        # Every placement authored under the anchor prototype renders under each of its instances.
        prims = settings.flood_max_instances * max(self._anchor_instance_count, 1)
        self._flood_estimate_label.text = f"Up to {prims} prims"

    def __refresh_instance_warning(self, settings: ScatterBrushSettings, force: bool):
        # Counting instances walks /RootNode/instances, so only redo it when the anchor inputs change.
        key = (settings.target_mode, settings.anchor_prototype_path)
        if not force and key == self._instance_warning_key:
            return
        self._instance_warning_key = key
        count = 0
        prototype = None
        if settings.target_mode == TargetMode.ANCHOR:
            stage = self._get_stage()
            if stage is not None:
                prototype = validated_anchor_prototype(stage, settings.anchor_prototype_path)
                if prototype is not None:
                    count = instance_count(stage, prototype)
        self._anchor_instance_count = count
        replicates = count > 1
        self._instance_warning_label.visible = replicates
        self._instance_warning_label.text = (
            f"Placements replicate onto {count} instances of {prototype.name}" if replicates else ""
        )
