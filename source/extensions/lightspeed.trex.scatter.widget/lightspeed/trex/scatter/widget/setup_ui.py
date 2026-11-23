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
from collections.abc import Callable, Sequence
from enum import Enum

import omni.usd
from lightspeed.trex.scatter.core import (
    ApplyTo,
    EraseScope,
    Falloff,
    MeshSurfaceCache,
    ReapplyScope,
    ScaleMode,
    ScatterBrushSettings,
    ScatterMode,
    ScopeDragSession,
    TargetMode,
    get_reapply_driver,
    get_scatter_brush_controller,
    instance_count,
    scene_scatter_cost,
    selected_prototypes,
    stroke_prim_limit,
    turn_matrix,
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
from .asset_up_axis import AssetUpAxisWidget
from .presets_ui import PresetsWidget
from .value_model import DragValueModel as _DragValueModel
from .value_model import field_bounds

_ANCHOR_NONE_TEXT = "<none>"
_LIVE_EDIT_TOOLTIP = (
    "Apply placement, scale and seed changes to painted scatter, as well as to the next stroke.\n\n"
    "- Right after a stroke, changes apply to that stroke: tune it by eye, and one undo removes it with everything "
    "changed on it while the settings stay as you left them\n"
    "- Selecting anything ends that, and changes then apply to the selection, each as an undo step of its own\n"
    "- Select painted placements or a scatter container to edit exactly those\n"
    "- With Apply To set to All, select a mesh or an instance to edit what the current preset painted on it; with "
    "Selected, the selected mesh is only where the next stroke goes\n"
    "- Brush and density settings always apply to the next stroke only, never to what is already painted"
)
_TURN_TOOLTIP = (
    "Degrees the selected scatter has been turned by since it was selected, about the stage axes.\n\n"
    "- The viewport's rotate rings write these, and typing here turns the selection the same way\n"
    "- Every placement turns where it stands; the patch is not carried around a pivot\n"
    "- A new selection has not been turned yet, so these read zero again, as they do after an undo or a redo\n"
    "- A turn stays with the scatter it was given to; use Base Rotation to change every stroke painted from now on"
)
_BASE_ROTATION_TOOLTIP = (
    "Degrees every placement is turned about its own root before its random rotation, about the stage {axis} axis."
    "\n\n"
    "- The fix for an asset that comes out lying down or leaning: it applies to every stroke painted from now on\n"
    "- Right after a stroke, the viewport's rotate rings set these and the stroke follows live\n"
    "- Saved with the preset"
)
_SECOND_LEAN_TOOLTIP = (
    "{bound} random lean about the other horizontal axis, in degrees: Y on a Z-up stage, Z on a Y-up stage."
)
_TURN_LIMIT = 360.0
_STATUS_STYLE = "PropertiesPaneSectionTreeItem"
_STATUS_ERROR_STYLE = "PropertiesPaneSectionTreeItemError"
_RESET_TOOLTIP = (
    "Put every brush setting back to its default: paint on the selected mesh, assets standing straight up.\n\n"
    "- The asset list, the saved presets, the current preset name and Edit Selection Live are kept\n"
    "- Right after a stroke, that stroke follows the reset the way it follows a preset; one undo still removes it"
)
_UP_AXIS_TOOLTIP = (
    "Which axis each asset was modelled standing along, as its file says, and the axis the brush stands it up by.\n\n"
    "- Match the file and the asset stands straight up\n"
    "- A choice that disagrees with the file scatters the asset lying on its side and is flagged\n"
    "- Same setting as the Up column of the asset list"
)
_INSTANCE_STROKES_TOOLTIP = (
    "Merge painted placements into one PointInstancer per patch, which the game draws from one copy of each asset's "
    "geometry instead of one replacement per blade.\n\n"
    "- The stroke just painted stays as prims, so it can be tuned and undone in one step\n"
    "- Painting the next stroke, or flooding, merges everything else; undoing that stroke gives it back as prims\n"
    "- Off, every placement stays a prim of its own"
)
_INSTANCE_ALL_TOOLTIP = (
    "Merge every loose placement of the edit layer now, including the stroke just painted: the last stroke of a "
    "session, scatter painted before instancing existed, or a patch that was split. One undo step."
)
_SPLIT_TOOLTIP = (
    "Turn the selected instanced scatter back into placements the brush and the gizmo can edit, and select them. "
    "Select the instancer, or the mesh it is painted on. Painting the next stroke merges them again."
)
# Floods above the default cap can take a while to author and render, so they ask for confirmation first.
_FLOOD_CONFIRM_THRESHOLD = ScatterBrushSettings.model_fields["flood_max_instances"].default
# Settings that shape the range a random size is drawn from, and so do nothing while Random Scale is off.
_RANDOM_SCALE_FIELDS = (
    "size_max",
    "scale_uniform",
    "scale_min",
    "scale_max",
    "scale_x_min",
    "scale_x_max",
    "scale_y_min",
    "scale_y_max",
    "scale_z_min",
    "scale_z_max",
    "scale_bias",
    "scale_weight",
)


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
        # Shared with the viewport gizmo, so a change is applied once whichever of them made it.
        self._reapply_driver = get_reapply_driver(context_name)
        # The turn fields drive the same move the viewport gizmo does, through a session of their own so that a
        # drag of a field is one undo step the way a drag of a ring is.
        self._turn_session = ScopeDragSession(context_name, self._surface_cache)
        self._turn_at_edit_start: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._turn_fields: list = []
        # Held as well as the fields: a drag field hands back a bare C++ model once the Python one it was
        # built with has been collected, which loses both this model's behaviour and its own cleanup.
        self._turn_models: list = []
        self._syncing = False
        self._in_field_edit = False
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
        self._reset_button: ui.Button | None = None
        self._status_label: ui.Label | None = None
        self._size_stack: ui.VStack | None = None
        self._multiplier_scale_stack: ui.VStack | None = None
        self._uniform_scale_stack: ui.VStack | None = None
        self._per_axis_scale_stack: ui.VStack | None = None
        # Controls by the settings field they edit, for the ones that are greyed out while they have no effect.
        self._field_widgets: dict[str, ui.Widget] = {}
        self._seed_field: _IntBoundedDrag | None = None
        self._reroll_button: ui.Button | None = None
        self._anchor_path_label: ui.Label | None = None
        self._anchor_use_selection_button: ui.Button | None = None
        self._flood_estimate_label: ui.Label | None = None
        self._instance_warning_label: ui.Label | None = None
        self._live_edit_label: ui.Label | None = None
        self._instancing_label: ui.Label | None = None
        self._instancing_scope: ReapplyScope | None = None
        self._instance_all_button: ui.Button | None = None
        self._split_button: ui.Button | None = None
        self._asset_list_widget: ScatterAssetListWidget | None = None
        self._up_axis_widget: AssetUpAxisWidget | None = None
        self._presets_widget: PresetsWidget | None = None

        self.__create_ui()
        self.__refresh_turn_fields()

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
        self._scope_changed_sub = self._reapply_driver.subscribe_scope_changed(self._on_scope_changed)
        self._selection_turn_sub = self._controller.subscribe_selection_turn_changed(self._on_selection_turn_changed)
        self._stroke_committed_sub = self._controller.subscribe_stroke_committed(self._on_stroke_committed)
        self._history_stepped_sub = self._controller.subscribe_history_stepped(self._on_history_stepped)
        usd_context = omni.usd.get_context(context_name)
        self._stage_event_sub = (
            None
            if usd_context is None
            else usd_context.get_stage_event_stream().create_subscription_to_pop(
                self._on_stage_event, name="lightspeed.trex.scatter.widget pane"
            )
        )

    def __create_ui(self):
        settings = self._controller.settings
        sections = (
            self.__build_brush_section,
            self.__build_placement_section,
            self.__build_scale_section,
            self.__build_random_section,
            self.__build_target_section,
            self.__build_output_section,
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
                tooltip=(
                    "Paint the enabled assets onto the selected meshes, or onto whatever is under the cursor when "
                    "Apply To is set to All. Hold Shift while dragging to erase instead."
                ),
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
                tooltip=(
                    "Fill the selected captured meshes, or the anchor mesh in Anchor mode, with up to Flood Cap "
                    "placements in one undo step. Padding is kept to what is already painted there."
                ),
                clicked_fn=self._on_flood_clicked,
            )
            self._reset_button = ui.Button(
                "Reset to Defaults",
                identifier="scatter_reset_defaults",
                tooltip=_RESET_TOOLTIP,
                clicked_fn=self._on_reset_clicked,
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
            "- Padding keeps placements apart from each other, and is what stops a stroke painted over another from "
            "doubling it\n"
            "- Max Surface Angle skips surfaces steeper than that, such as the wall beside a floor",
        )
        with frame:
            with ui.VStack(height=0, spacing=self._SPACING_XS):
                self.__build_float_row(
                    "Radius",
                    "radius",
                    "scatter_radius",
                    1.0,
                    "Brush radius in stage units. Hold B and scroll over the viewport to resize it.",
                )
                self.__build_enum_row(
                    "Falloff",
                    "falloff",
                    "scatter_falloff",
                    Falloff,
                    "How the chance of keeping a candidate drops from the center of the brush to its edge.",
                )
                self.__build_float_row(
                    "Density",
                    "density",
                    "scatter_density",
                    0.5,
                    "Candidates drawn in the brush disk per stamp; Strength and Falloff decide how many are kept.\n\n"
                    "A fraction is a chance: 0.5 draws one candidate every other stamp.",
                )
                self.__build_float_row(
                    "Strength",
                    "strength",
                    "scatter_strength",
                    0.01,
                    "Share of the candidates that is kept, from 0 (none) to 1 (all).",
                )
                self.__build_float_row(
                    "Spacing", "stamp_spacing", "scatter_spacing", 1.0, "Distance between two stamps along a stroke."
                )
                self.__build_float_row(
                    "Padding",
                    "padding",
                    "scatter_padding",
                    0.5,
                    "Minimum distance between two placements, in stage units.\n\n"
                    "At 0 nothing keeps placements apart, so painting over an area again adds a full layer to it.",
                )
                self.__build_float_row(
                    "Max Surface Angle",
                    "max_surface_angle",
                    "scatter_max_surface_angle",
                    1.0,
                    "Steepest surface, in degrees from the stage up axis, the brush still paints on.\n\n"
                    "A stamp snaps each candidate onto the closest surface, so near a wall or a ledge some land on "
                    "the side of it and stand sideways. Lower this (45 suits grass) to skip them; 180 keeps every "
                    "candidate, which painting on a wall or a ceiling needs.",
                )

    def __build_placement_section(self, settings: ScatterBrushSettings):
        frame = self.__build_section(
            "PLACEMENT",
            "How each placement sits on the surface.\n\n"
            "- Asset Up Axis says which way each asset was modelled; match its file and it stands straight up\n"
            "- Off by default, Conform tilts the asset to follow the slope of the surface\n"
            "- Align to Stroke turns the asset along the stroke direction\n"
            "- Base Rotation stands up or leans every placement the same way, before its random rotation\n"
            "- Rotation ranges are in degrees; the advanced ranges lean the asset over",
        )
        with frame:
            with ui.VStack(height=0, spacing=self._SPACING_XS):
                self.__build_row("Asset Up Axis", _UP_AXIS_TOOLTIP, ui.Spacer)
                self._up_axis_widget = AssetUpAxisWidget(label_width=self._LABEL_WIDTH)
                self.__build_row("Edit Selection Live", _LIVE_EDIT_TOOLTIP, self.__build_live_edit_control)
                for axis, identifier in enumerate(("scatter_turn_x", "scatter_turn_y", "scatter_turn_z")):
                    self.__build_row(
                        f"Turn {'XYZ'[axis]}",
                        _TURN_TOOLTIP,
                        functools.partial(self.__build_turn_field, axis, identifier),
                    )
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
                    "Tilt the asset to follow the slope of the surface; off, every asset stands straight up.",
                )
                self.__build_bool_row(
                    "Align to Stroke",
                    "align_to_stroke",
                    "scatter_align_stroke",
                    "Turn the asset to face the stroke direction. A flood has no direction, so it ignores this.",
                )
                for axis in ("x", "y", "z"):
                    self.__build_float_row(
                        f"Base Rotation {axis.upper()}",
                        f"base_rotation_{axis}",
                        f"scatter_base_rotation_{axis}",
                        1.0,
                        _BASE_ROTATION_TOOLTIP.format(axis=axis.upper()),
                    )
                self.__build_float_row(
                    "Rotation Z Min",
                    "rotation_z_min",
                    "scatter_rot_z_min",
                    1.0,
                    "Lowest random turn about the stage up axis, in degrees.",
                )
                self.__build_float_row(
                    "Rotation Z Max",
                    "rotation_z_max",
                    "scatter_rot_z_max",
                    1.0,
                    "Highest random turn about the stage up axis, in degrees.",
                )
                advanced_frame = _PropertyCollapsableFrame("ADVANCED ROTATION", collapsed=True)
                self._section_frames.append(advanced_frame)
                with advanced_frame:
                    with ui.VStack(height=0, spacing=self._SPACING_XS):
                        self.__build_float_row(
                            "Rotation X Min",
                            "rotation_x_min",
                            "scatter_rot_x_min",
                            1.0,
                            "Lowest random lean about the X axis, in degrees.",
                        )
                        self.__build_float_row(
                            "Rotation X Max",
                            "rotation_x_max",
                            "scatter_rot_x_max",
                            1.0,
                            "Highest random lean about the X axis, in degrees.",
                        )
                        self.__build_float_row(
                            "Rotation Y Min",
                            "rotation_y_min",
                            "scatter_rot_y_min",
                            1.0,
                            _SECOND_LEAN_TOOLTIP.format(bound="Lowest"),
                        )
                        self.__build_float_row(
                            "Rotation Y Max",
                            "rotation_y_max",
                            "scatter_rot_y_max",
                            1.0,
                            _SECOND_LEAN_TOOLTIP.format(bound="Highest"),
                        )

    def __build_live_edit_control(self):
        # Built without the trailing spacer of __build_bool_field so the scope readout sits beside the checkbox
        # rather than at the far edge of the row.
        model = ui.SimpleBoolModel(self._controller.settings.edit_selection_live)
        ui.CheckBox(model=model, width=0, identifier="scatter_edit_selection_live", tooltip=_LIVE_EDIT_TOOLTIP)
        self.__bind("edit_selection_live", model, lambda m: m.as_bool, lambda m, value: m.set_value(bool(value)))
        ui.Spacer(width=self._SPACING_MD)
        self._live_edit_label = ui.Label(
            self._reapply_driver.summary(),
            name=_STATUS_STYLE,
            identifier="scatter_live_edit_scope",
            tooltip=_LIVE_EDIT_TOOLTIP,
        )

    def __build_scale_section(self, settings: ScatterBrushSettings):
        frame = self.__build_section(
            "SCALE",
            "How large each placement is.\n\n"
            "- Absolute Size sets the height in stage units, measured along the asset's up axis\n"
            "- Multiplier scales the asset by a factor of the size it was authored at\n"
            "- Bias pushes the distribution toward the minimum (-1) or the maximum (1)\n"
            "- Weight sharpens (above 1) or flattens (below 1) the distribution\n"
            "- Disable Uniform to set a range per axis",
        )
        with frame:
            with ui.VStack(height=0, spacing=self._SPACING_XS):
                self.__build_enum_row(
                    "Scale Mode",
                    "scale_mode",
                    "scatter_scale_mode",
                    ScaleMode,
                    "Absolute Size gives every placement a height in stage units; Multiplier scales the asset by a "
                    "factor of its own size.",
                )
                self.__build_bool_row(
                    "Random Scale",
                    "scale_enabled",
                    "scatter_scale_enabled",
                    "Vary the size within the range below.\n\n"
                    "Off, every placement is exactly Size Min tall in Absolute Size mode, and keeps the size it was "
                    "modelled at in Multiplier mode.",
                )
                self._size_stack = ui.VStack(height=0, spacing=self._SPACING_XS)
                with self._size_stack:
                    self.__build_float_row(
                        "Size Min",
                        "size_min",
                        "scatter_size_min",
                        1.0,
                        "Lowest height in stage units, and the height of every placement when Random Scale is off.",
                    )
                    self.__build_float_row(
                        "Size Max", "size_max", "scatter_size_max", 1.0, "Highest height in stage units."
                    )
                self._multiplier_scale_stack = ui.VStack(height=0, spacing=self._SPACING_XS)
                with self._multiplier_scale_stack:
                    self.__build_bool_row(
                        "Uniform", "scale_uniform", "scatter_scale_uniform", "Use one range for all three axes."
                    )
                    self._uniform_scale_stack = ui.VStack(height=0, spacing=self._SPACING_XS)
                    with self._uniform_scale_stack:
                        self.__build_float_row(
                            "Scale Min", "scale_min", "scatter_scale_min", 0.01, "Lowest scale factor."
                        )
                        self.__build_float_row(
                            "Scale Max", "scale_max", "scatter_scale_max", 0.01, "Highest scale factor."
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
                self.__build_float_row(
                    "Bias",
                    "scale_bias",
                    "scatter_scale_bias",
                    0.01,
                    "Skew the sizes toward the minimum (-1) or the maximum (1) of the range; 0 leaves them even.",
                )
                self.__build_float_row(
                    "Weight",
                    "scale_weight",
                    "scatter_scale_weight",
                    0.1,
                    "How sizes spread over the range: above 1 gathers them toward the middle, below 1 pushes them "
                    "toward the two ends, and 1 spreads them evenly.",
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
            "- Apply To Selected, the default, paints only on the selected meshes, straight through anything in front "
            "of them; select the ground first. All paints whatever is under the cursor\n"
            "- Hit Surface authors under the mesh under the cursor\n"
            "- Anchor always authors under the anchor prototype; Use Selection sets it from the viewport selection\n"
            "- Flood fills the anchor or the selected meshes up to the cap\n\n"
            "Placements authored under a prototype appear under every instance of that prototype.",
        )
        with frame:
            with ui.VStack(height=0, spacing=self._SPACING_XS):
                self.__build_enum_row(
                    "Apply To",
                    "apply_to",
                    "scatter_apply_to",
                    ApplyTo,
                    "Paint only on the selected meshes, looking through whatever stands in front of them, or on "
                    "whatever mesh is under the cursor.",
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
                        tooltip=(
                            "Use the selected captured mesh or instance as the anchor. With several selected, the "
                            "one whose path comes first is used."
                        ),
                        clicked_fn=self._on_anchor_use_selection_clicked,
                    )
                self.__build_enum_row(
                    "Erase Scope",
                    "erase_scope",
                    "scatter_erase_scope",
                    EraseScope,
                    "Erase any scattered placement under the brush, or only placements of the assets in the ASSETS "
                    "list.",
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
                            "instances that show the anchor prototype, and never more than five times the Prim "
                            "Budget."
                        ),
                    )
                budget_tooltip = (
                    "Prims a single stroke is expected to render.\n\n"
                    "- The counter on the brush turns amber at this number\n"
                    "- At five times it the stroke stops adding placements and the counter turns red; a flood stops "
                    "at the same limit\n"
                    "- A placement on a mesh with several instances counts once per instance\n"
                    "- Raise it to paint more in one stroke"
                )
                with ui.HStack(height=self._ROW_HEIGHT, spacing=self._SPACING_MD):
                    ui.Label("Prim Budget", width=self._LABEL_WIDTH, tooltip=budget_tooltip)
                    self.__build_int_field("prim_budget", "scatter_prim_budget", 500, budget_tooltip)
                self._instance_warning_label = ui.Label(
                    "",
                    name="Warning",
                    identifier="scatter_instance_warning",
                    height=0,
                    word_wrap=True,
                    visible=False,
                )

    def __build_output_section(self, settings: ScatterBrushSettings):
        frame = self.__build_section(
            "OUTPUT",
            "How painted placements are stored.\n\n"
            "- Instanced placements live in one PointInstancer per patch, drawn from one copy of each asset\n"
            "- The stroke just painted stays as prims until the next one, so it can be tuned and undone\n"
            "- Split turns instanced scatter back into prims to edit it with the brush settings and the gizmo",
        )
        with frame:
            with ui.VStack(height=0, spacing=self._SPACING_XS):
                self.__build_bool_row(
                    "Instance Strokes", "instance_strokes", "scatter_instance_strokes", _INSTANCE_STROKES_TOOLTIP
                )
                with ui.HStack(height=self._ROW_HEIGHT, spacing=self._SPACING_MD):
                    ui.Spacer(width=self._LABEL_WIDTH)
                    self._instance_all_button = ui.Button(
                        "Instance All",
                        identifier="scatter_instance_all",
                        tooltip=_INSTANCE_ALL_TOOLTIP,
                        clicked_fn=self._on_instance_all_clicked,
                    )
                    self._split_button = ui.Button(
                        "Split Selection",
                        identifier="scatter_split_selection",
                        tooltip=_SPLIT_TOOLTIP,
                        clicked_fn=self._on_split_clicked,
                    )
                self._instancing_label = ui.Label(
                    "",
                    name=_STATUS_STYLE,
                    identifier="scatter_instancing_summary",
                    height=0,
                    word_wrap=True,
                    tooltip="Placements painted on the stage, and how many of them are instanced.",
                )

    def __build_assets_section(self, settings: ScatterBrushSettings):
        frame = self.__build_section(
            "ASSETS",
            "Ingested assets the brush chooses from, weighted by their Weight.\n\n"
            "- Disabled rows are skipped\n"
            "- Up Axis corrects assets authored with a different up axis than the stage\n"
            "- Maps shows the textures the brush will assign; its button picks a set by hand",
        )
        with frame:
            with ui.VStack(height=0, spacing=self._SPACING_XS):
                self.__build_bool_row(
                    "Assign Textures",
                    "assign_textures",
                    "scatter_assign_textures",
                    "Assign each asset's ingested albedo, normal, roughness and height maps to the prims it paints.\n"
                    "Textures the asset already carries are never replaced.",
                )
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
        self._field_widgets[field_name] = field
        self.__bind(field_name, model, lambda m: m.as_float, lambda m, value: m.set_value(float(value)), end_edit=True)
        return field

    def __build_turn_field(self, axis: int, identifier: str) -> _FloatBoundedDrag:
        """Build one of the three fields showing what the selection has been turned by.

        Bound to the turn rather than to a brush setting: a brush setting decides what the next stroke paints, while
        a turn is an edit to scatter that is already painted, which is why it is not routed through ``__bind``.

        Args:
            axis: 0, 1 or 2 for X, Y or Z.
            identifier: Widget identifier used by the tests.
        """
        model = _DragValueModel(0.0)
        field = _FloatBoundedDrag(
            model=model,
            min=-_TURN_LIMIT,
            max=_TURN_LIMIT,
            step=1.0,
            hard_min_value=-_TURN_LIMIT,
            hard_max_value=_TURN_LIMIT,
            identifier=identifier,
            tooltip=_TURN_TOOLTIP,
            enable_batch_edit=False,
        )
        self._drag_fields.append(field)
        self._turn_fields.append(field)
        self._turn_models.append(model)
        on_edit = functools.partial(self._on_turn_edited, axis)
        self._field_subs.append(model.subscribe_value_changed_fn(on_edit))
        self._field_subs.append(model.subscribe_end_edit_fn(on_edit))
        self._field_subs.append(model.subscribe_end_edit_fn(self._on_turn_edit_finished))
        return field

    def _on_turn_edited(self, _axis: int, _model: ui.AbstractValueModel):
        """Turn the selection to the angles the three fields now hold.

        The move is measured from the turn the selection stood at when this edit opened, so dragging a field is one
        continuous turn rather than a turn per step, exactly as dragging a ring is.

        Args:
            _axis: Axis of the field that changed; the turn is read from all three together.
            _model: Value model that changed; likewise.
        """
        if self._syncing:
            return
        scope = self._reapply_driver.scope
        if scope.is_empty:
            return
        requested = self.__read_turn_fields()
        if requested == self._controller.selection_turn(scope):
            return
        if not self._turn_session.active:
            self._turn_at_edit_start = self._controller.selection_turn(scope)
            if not self._turn_session.begin(scope, self._controller.settings.conform_to_surface):
                return
        delta = turn_matrix(self._turn_at_edit_start).GetInverse() * turn_matrix(requested)
        self._turn_session.update(delta)
        self._controller.set_selection_turn(scope, requested)

    def _on_turn_edit_finished(self, _model: ui.AbstractValueModel):
        """Commit the turn the fields dialled in as one undo step."""
        if self._syncing:
            return
        self._turn_session.commit()

    def _on_selection_turn_changed(self, scope: ReapplyScope, turn: tuple):
        """Show the turn the viewport gizmo has just given the patch this pane is scoped to."""
        if scope != self._reapply_driver.scope:
            return
        self.__write_turn_fields(turn)

    def __read_turn_fields(self) -> tuple[float, float, float]:
        """Return the X, Y, Z degrees the three turn fields hold."""
        return tuple(float(model.as_float) for model in self._turn_models)

    def __write_turn_fields(self, turn: Sequence[float]) -> None:
        """Push a turn into the three fields without treating it as an edit of theirs."""
        if len(self._turn_models) != 3:
            return
        was_syncing, self._syncing = self._syncing, True
        try:
            for model, value in zip(self._turn_models, turn):
                if abs(model.as_float - float(value)) > 1e-9:
                    model.set_value(float(value))
        finally:
            self._syncing = was_syncing

    def __refresh_turn_fields(self) -> None:
        """Show the turn of whatever is selected now, and grey the fields out when nothing is.

        They are greyed out on the stroke just painted as well: a turn stays with the scatter it is given to, and the
        stroke just painted is tuned through the brush settings, Base Rotation included, so later strokes follow.
        """
        scope = self._reapply_driver.scope
        self.__write_turn_fields(self._controller.selection_turn(scope))
        for field in self._turn_fields:
            field.enabled = not scope.is_empty and not self._reapply_driver.previewing

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
        self._field_widgets[field_name] = checkbox
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
            self._field_subs.append(model.subscribe_end_edit_fn(self._on_edit_finished))
            return
        # A checkbox or a combo box has no drag to end, so its value change is the whole edit. Subscribed after the
        # edit itself so the live edit is committed with the change already applied.
        self._field_subs.append(model.subscribe_value_changed_fn(self._on_edit_finished))

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
        self._in_field_edit = True
        try:
            applied = self._controller.update_settings(**{field_name: value})
        finally:
            self._in_field_edit = False
        if not applied:
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
        if not self._in_field_edit:
            # The change came from outside this pane's controls, from a preset or the reroll button, so no field
            # will report the end of an edit and the live edit it opened has to be committed here. The driver hears
            # of the change through the same event, before or after this pane, so it is handed the change first:
            # finishing an edit it had not opened yet would leave that edit open, with no undo step.
            self._reapply_driver.follow_settings(settings)
            self._reapply_driver.edit_finished()

    def _on_edit_finished(self, _model: ui.AbstractValueModel):
        """Commit the live edit a control opened, so one drag of that control is one undo step.

        Args:
            _model: Value model that finished being edited; the live edit spans every field, so it is not needed.
        """
        if self._syncing:
            return
        self._reapply_driver.edit_finished()

    def _on_scope_changed(self, _scope: ReapplyScope):
        """Show what a settings change would update now that the selection or the last live edit changed.

        Args:
            _scope: The new scope; the readout is built from the driver, which also knows the last outcome.
        """
        self.__refresh_live_edit_label()
        self.__refresh_turn_fields()
        # A new scope follows an undo or a selection, either of which can change what is instanced. A live edit
        # reports the same scope at every step of a drag, and recounting the stage at every step would cost a frame.
        if _scope != self._instancing_scope:
            self._instancing_scope = _scope
            self.__refresh_instancing_label()

    def _on_stroke_committed(self, _count: int, _erase: bool):
        """Recount what is painted once a stroke or a flood has committed, since that is when the brush merges.

        Args:
            _count: Placements the stroke changed; the readout counts the whole stage instead.
            _erase: Whether the stroke erased.
        """
        self.__refresh_instancing_label()

    def _on_history_stepped(self):
        """Recount what is painted after an undo or a redo, which adds or removes scatter without a stroke."""
        self.__refresh_instancing_label()

    def _on_stage_event(self, event):
        """Follow a project opening or closing, which changes everything the pane derives from the stage.

        Args:
            event: Stage event of the pane's USD context.
        """
        if event.type not in (int(omni.usd.StageEventType.OPENED), int(omni.usd.StageEventType.CLOSED)):
            return
        self._on_status_message("", False)
        self.__refresh_instancing_label()
        if self.window_visible:
            self.__refresh_derived(self._controller.settings, force_instance_warning=True)

    def _on_instance_all_clicked(self):
        """Merge every loose placement of the edit layer into its patch's instancer."""
        self._controller.instance_all(self._context_name)
        self.__refresh_instancing_label()

    def _on_split_clicked(self):
        """Turn the selected instanced scatter back into placements; the controller reports an empty selection."""
        self._controller.split_selection(self._context_name)
        self.__refresh_instancing_label()

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
        """Flood the default targets, after a confirmation when the flood is a large one.

        Large is a cap above the default, or more prims than the Prim Budget once the instances of the anchor are
        counted: a modest cap on a mesh with many instances renders as much as a large cap on a mesh with one.
        """
        settings = self._controller.settings
        cap = settings.flood_max_instances
        prims = self.__flood_prims(settings, self.__flood_target_instances(settings))
        flood = functools.partial(self._controller.flood, self._context_name, self._surface_cache)
        if cap <= _FLOOD_CONFIRM_THRESHOLD and prims <= settings.prim_budget:
            flood()
            return
        _TrexMessageDialog(
            message=(
                f"Flooding can create up to {cap:,} placements and render up to {prims:,} prims, which may take a "
                "while to author and to render.\n\n"
                "Continue with the flood?"
            ),
            title="Flood Surfaces",
            ok_label="Flood",
            ok_handler=flood,
        )

    def _on_reset_clicked(self):
        """Put every brush setting back to its default once the user confirms; settings changes cannot be undone."""
        _TrexMessageDialog(
            message=(
                "Reset every brush setting to its default?\n\n"
                "The asset list, the saved presets, the current preset name and Edit Selection Live are kept. "
                "Settings changes cannot be undone.\n\n"
                "With Edit Selection Live on, the stroke just painted or the selected scatter follows the reset; "
                "that update can be undone."
            ),
            title="Reset Brush Settings",
            ok_label="Reset",
            ok_handler=self._controller.reset_settings,
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
            self.__refresh_turn_fields()
            self.__refresh_instancing_label()

    def destroy(self):
        """Commit any open live edit, then release the subscriptions, the surface cache and the child widgets."""
        if self._turn_session is not None:
            self._turn_session.commit()
            self._turn_session = None
        self._settings_changed_sub = None
        self._mode_changed_sub = None
        self._assets_changed_sub = None
        self._status_message_sub = None
        self._scope_changed_sub = None
        self._selection_turn_sub = None
        self._stroke_committed_sub = None
        self._history_stepped_sub = None
        self._stage_event_sub = None
        self._paint_sub = None
        self._erase_sub = None
        self._field_subs.clear()
        self._bindings.clear()
        self._field_widgets.clear()
        for field in self._drag_fields:
            field.destroy()
        self._drag_fields.clear()
        self._turn_fields.clear()
        self._turn_models.clear()
        for frame in self._section_frames:
            frame.destroy()
        self._section_frames.clear()
        for widget in (self._asset_list_widget, self._presets_widget, self._up_axis_widget):
            if widget is not None:
                widget.destroy()
        self._asset_list_widget = None
        self._presets_widget = None
        self._up_axis_widget = None
        if self._reapply_driver is not None:
            # Shared with the viewport gizmo, so it outlives the pane; only the edit this pane opened is closed.
            self._reapply_driver.edit_finished()
            self._reapply_driver = None
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

    def __refresh_live_edit_label(self):
        if self._live_edit_label is not None:
            self._live_edit_label.text = self._reapply_driver.summary()

    def __refresh_instancing_label(self):
        # Counted only while the pane shows it: the count walks every prototype of the capture.
        if self._instancing_label is None or not self.window_visible:
            return
        cost = scene_scatter_cost(self._get_stage())
        if not cost.placements:
            self._instancing_label.text = "Nothing painted yet"
            return
        loose = cost.placements - cost.instanced
        self._instancing_label.text = (
            f"{cost.placements:,} placements on {cost.prototypes:,} meshes, rendering {cost.rendered_prims:,} prims: "
            f"{cost.instanced:,} instanced, {loose:,} loose"
        )

    def __push_mode(self, mode: ScatterMode):
        self._syncing = True
        try:
            self._paint_model.set_value(mode == ScatterMode.PAINT)
            self._erase_model.set_value(mode == ScatterMode.ERASE)
        finally:
            self._syncing = False

    def __refresh_derived(self, settings: ScatterBrushSettings, force_instance_warning: bool = False):
        # The per-axis stack lives inside the multiplier stack, so hiding that one hides it too.
        self.__refresh_live_edit_label()
        absolute_size = settings.scale_mode == ScaleMode.ABSOLUTE_SIZE
        self._size_stack.visible = absolute_size
        self._multiplier_scale_stack.visible = not absolute_size
        # One range or three, never both: the range that is not sampled would be a control that does nothing.
        self._uniform_scale_stack.visible = settings.scale_uniform
        self._per_axis_scale_stack.visible = not settings.scale_uniform
        # With Random Scale off nothing is drawn from a range, so everything that shapes one is greyed out.
        for field_name in _RANDOM_SCALE_FIELDS:
            self._field_widgets[field_name].enabled = settings.scale_enabled
        seed_editable = not settings.randomize_seed
        self._seed_field.enabled = seed_editable
        self._reroll_button.enabled = seed_editable
        self._anchor_path_label.text = settings.anchor_prototype_path or _ANCHOR_NONE_TEXT
        self.__refresh_instance_warning(settings, force_instance_warning)
        # Every placement authored under a prototype renders under each instance of that prototype's hash. Only the
        # anchor is known ahead of the flood; in hit-surface mode the multiplier is whatever is selected when it runs.
        if settings.target_mode == TargetMode.ANCHOR and self._anchor_instance_count:
            prims = self.__flood_prims(settings, self._anchor_instance_count)
            self._flood_estimate_label.text = f"Up to {prims:,} prims"
        else:
            self._flood_estimate_label.text = f"Up to {self.__flood_prims(settings, 1):,} prims per instance"

    @staticmethod
    def __flood_prims(settings: ScatterBrushSettings, instances: int) -> int:
        """Return the most prims one flood can render on meshes of ``instances`` instances, within the prim limit."""
        return min(settings.flood_max_instances * max(instances, 1), stroke_prim_limit(settings.prim_budget))

    def __flood_target_instances(self, settings: ScatterBrushSettings) -> int:
        """Return the most instances any mesh a flood started now would fill has, which multiplies what it renders.

        The anchor's count is the one the estimate already shows. Without an anchor a flood fills the selected
        meshes, which are counted here, when the flood is asked for, rather than at every change of the selection.
        """
        if settings.target_mode == TargetMode.ANCHOR:
            return max(self._anchor_instance_count, 1)
        stage = self._get_stage()
        usd_context = omni.usd.get_context(self._context_name)
        if stage is None or usd_context is None:
            return 1
        counts = [instance_count(stage, prototype) for prototype in selected_prototypes(usd_context)]
        return max(counts, default=1) or 1

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
