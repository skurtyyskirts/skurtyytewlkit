# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math
import asyncio
import numpy as np
import omni.kit.app
import omni.kit.undo
from omni import ui
from omni.paint.system.ui import SettingsBuilder

from .constant import CONTROL_HEIGHT, SPINNER_WIDTH

DATA_SCALER = 0.9
X_STEP = 0.01
MIN_X_RANGE = 0.1
ARROW_WIDTH = 6
ARROW_HEIGHT = 9
BIAS_LABEL_WIDTH = 25
BIAS_PADDING = 8
BIAS_PLACER_OFFSET = BIAS_LABEL_WIDTH + BIAS_PADDING
PLOT_PADDING = 4
PADDING_RIGHT = 25

SCALE_KEY = "scale"
FLOAT_DRAG_STYLE_WITH_NO_SLIDER = {"Slider::value": {"secondary_color": 0x0}}


# Get scale distribution data based on gaussian distribution
def get_scale_data(scale, number):
    multi = 1
    while True:
        # Get data based on gaussian distribution
        bias = scale["bias"] * (scale["max"] - scale["min"]) + scale["min"]
        data = np.random.normal(loc=bias, scale=scale["weight"], size=number * multi)
        # Filter data by scale distribution min/max
        filtered = np.where((data <= scale["max"]) & (data >= scale["min"]))
        if len(filtered[0]) >= number:
            # Only return required number of data
            return data[filtered][:number]
        else:
            multi += 1


# Normal distribution (Gaussian distribution) PDF
def gaussian(x, bias, weight):
    weight = max(weight, 0.01)
    return np.exp(-((x - bias) ** 2) / (2 * weight ** 2)) / (math.sqrt(2 * math.pi) * weight)


class ScaleDistributionCtrl:
    def __init__(self, brush: dict, original_brush: dict, on_param_changed_fn):
        self._brush = brush
        self._original_value = original_brush.get(SCALE_KEY, None)
        if self._original_value is None:
            self._original_value = {"enabled": True, "bias": 0.5, "min": 0.9, "max": 1.1, "weight": 0.2}
        self._on_param_changed_fn = on_param_changed_fn
        self._in_dragging = False

        # checkbox
        (
            self._status_switch,
            self._status_switch_model,
        ), self._status_switch_default = SettingsBuilder.build_checkbox_setting2(
            self._brush,
            "scale/enabled",
            "Scale",
            self._original_value["enabled"],
            self._on_status_changed,
            tooltip="The size of the painted assets is distributed in scale",
        )

        ui.Spacer(height=2)

        # body
        self._body = ui.Frame()
        self._body.set_build_fn(self._build_body)
        self._build_body()

    def _is_default(self, widget):
        keys = ["bias", "min", "max", "weight"]
        for key in keys:
            if self._brush[SCALE_KEY][key] != self._original_value[key]:
                return False
        return True

    def _set_default(self, widget):
        omni.kit.undo.begin_group()
        keys = ["bias", "min", "max", "weight"]
        for key in keys:
            omni.kit.commands.execute(
                "ChangeBrushParamCommand",
                brush=self._brush,
                param=SCALE_KEY + "/" + key,
                value=self._original_value[key],
                prev_value=self._brush[SCALE_KEY][key],
            )

        omni.kit.undo.end_group()
        self._body.rebuild()

    def _build_graph_settings(self):
        with ui.HStack():
            ui.Spacer(width=PADDING_RIGHT)
            with ui.ZStack():
                self._build_image()
                self._build_bias()
        return None

    def _build_body(self):
        with self._body:
            _, self._graph_default = SettingsBuilder.build_custom_setting(
                None, self._build_graph_settings, self._is_default, self._set_default
            )
            self._create_scale_image()
            asyncio.ensure_future(self._calculate_bias_pos_async())

    def rebuild(self):
        self._body.rebuild()

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self._status_switch_model:
            self._status_switch_model.destroy()
            self._status_switch_model = None

        if self._min_spin_model:
            self._min_spin_model.destroy()
            self._min_spin_model = None

        if self._weight_spin_model:
            self._weight_spin_model.destroy()
            self._weight_spin_model = None

        if self._max_spin_model:
            self._max_spin_model.destroy()
            self._max_spin_model = None

    def on_width_changed(self, change):
        self._calculate_bias_pos()

    def _build_image(self):
        with ui.VStack():
            ui.Spacer(height=CONTROL_HEIGHT + ARROW_HEIGHT + 2)
            self._scale_plot = ui.Plot(height=120)
            ui.Spacer(height=3)
            with ui.HStack(height=CONTROL_HEIGHT):
                self._min_spin, self._min_spin_model = SettingsBuilder.build_float_drag_widget(
                    self._brush,
                    "scale/min",
                    0.01,
                    self._on_scale_value_changed,
                    widget_kwargs={"style": FLOAT_DRAG_STYLE_WITH_NO_SLIDER},
                )
                self._min_spin.model.add_begin_edit_fn(self._begin_edit)
                self._min_spin.model.add_post_end_edit_fn(self._end_edit)
                ui.Spacer()
                self._weight_spin, self._weight_spin_model = SettingsBuilder.build_float_drag_widget(
                    self._brush,
                    "scale/weight",
                    0.01,
                    self._on_scale_value_changed,
                    widget_kwargs={"style": FLOAT_DRAG_STYLE_WITH_NO_SLIDER, "min": 0.01},
                )
                ui.Spacer()
                self._max_spin, self._max_spin_model = SettingsBuilder.build_float_drag_widget(
                    self._brush,
                    "scale/max",
                    0.01,
                    self._on_scale_value_changed,
                    widget_kwargs={"style": FLOAT_DRAG_STYLE_WITH_NO_SLIDER},
                )
                self._max_spin.model.add_begin_edit_fn(self._begin_edit)
                self._max_spin.model.add_post_end_edit_fn(self._end_edit)
            with ui.HStack(height=CONTROL_HEIGHT):
                ui.Label("Min", width=SPINNER_WIDTH, alignment=ui.Alignment.CENTER)
                ui.Spacer()
                ui.Label("Weight", width=SPINNER_WIDTH, alignment=ui.Alignment.CENTER)
                ui.Spacer()
                ui.Label("Max", width=SPINNER_WIDTH, alignment=ui.Alignment.CENTER)

    def _build_bias(self):
        with ui.VStack():
            # Here the placer x is always offset to left side of plot image
            self._bias_placer = ui.Placer(offset_x=0, offset_y=0, height=CONTROL_HEIGHT)
            self._line_placer = ui.Placer(offset_x=0, offset_y=0)
            with self._bias_placer:
                with ui.VStack(width=BIAS_LABEL_WIDTH + BIAS_PADDING + SPINNER_WIDTH):
                    with ui.HStack(height=CONTROL_HEIGHT):
                        # we only update bias at the end of edit, so don't use setting value model like others
                        with ui.ZStack(width=BIAS_LABEL_WIDTH + BIAS_PADDING):
                            self._drag_rect = ui.Rectangle(name="drag")
                            with ui.HStack():
                                ui.Label("Bias", width=BIAS_LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER)
                                ui.Spacer(width=BIAS_PADDING)

                        self._bias_field = ui.FloatField(name="value")
                        self._bias_field.model.set_value(self._brush[SCALE_KEY]["bias"])
                        self._bias_field.model.add_end_edit_fn(self._bias_end_edit)

                    with ui.HStack():
                        ui.Spacer(width=BIAS_PLACER_OFFSET)
                        with self._line_placer:
                            with ui.VStack(width=ARROW_WIDTH):
                                self._drag_arrow = ui.Triangle(
                                    width=ARROW_WIDTH,
                                    height=ARROW_HEIGHT,
                                    name="default",
                                    alignment=ui.Alignment.CENTER_BOTTOM,
                                )
                                with ui.Placer(offset_x=ARROW_WIDTH / 2, offset_y=0):
                                    self._drag_line = ui.Line(
                                        height=120, width=1, alignment=ui.Alignment.LEFT, name="bias"
                                    )

        self._enable_drag(self._drag_rect)
        self._enable_drag(self._drag_arrow)
        self._enable_drag(self._drag_line)

    def _enable_drag(self, widget):
        widget.set_mouse_pressed_fn(lambda x, y, key, a: self._begin_drag(key, x))
        widget.set_mouse_moved_fn(lambda x, y, key, m: self._dragging(x))
        widget.set_mouse_released_fn(lambda x, y, key, m: self._end_drag(key))

    def _begin_drag(self, key, x):
        if key != 0:
            return

        self._in_dragging = True
        self._drag_begin_x = x
        self._drag_begin_offset = self._from_bias_to_offset()

    def _dragging(self, x):
        if not self._in_dragging:
            return
        offset_x = self._drag_begin_offset + x - self._drag_begin_x
        offset_min = PLOT_PADDING
        offset_max = self._scale_plot.computed_width - PLOT_PADDING
        if offset_x < offset_min:
            offset_x = offset_min
        if offset_x > offset_max:
            offset_x = offset_max
        self._update_bias_pos(offset_x)
        bias = self._from_offset_to_bias(offset_x)
        self._bias_field.model.set_value(bias)
        self._on_scale_value_changed()

    def _end_drag(self, key):
        if key != 0:
            return

        if self._in_dragging:
            offset_x = self._line_placer.offset_x + ARROW_WIDTH / 2
            omni.kit.commands.execute(
                "ChangeBrushParamCommand",
                brush=self._brush,
                param=SCALE_KEY + "/bias",
                value=self._from_offset_to_bias(offset_x),
                prev_value=self._brush[SCALE_KEY]["bias"],
            )
            self._in_dragging = False

    def _create_scale_data(self):
        bias_normalize = self._bias_field.model.get_value_as_float()
        min_value = self._min_spin_model.get_value_as_float()
        max_value = self._max_spin_model.get_value_as_float()
        weight = self._weight_spin_model.get_value_as_float()
        x = np.arange(min_value, max_value, X_STEP)
        value_range = max_value - min_value
        bias = bias_normalize * value_range + min_value
        self._scale_data = gaussian(x, bias, weight)
        return self._scale_data

    def _create_scale_image(self, update_data=True):
        if update_data:
            self._create_scale_data()

        if self._scale_data is None:
            return

        scale_data = DATA_SCALER * self._scale_data
        self._scale_plot.set_data(*scale_data)
        self._scale_plot.scale_min = min(scale_data)
        self._scale_plot.scale_max = max(scale_data) / DATA_SCALER

    def _on_scale_value_changed(self, value=None):
        self._update_value_limitation()
        self._create_scale_image()
        self._graph_default.rebuild()
        # if self._on_param_changed_fn:
        #     self._on_param_changed_fn(SCALE_KEY, self._scale)

    def _on_status_changed(self, status):
        self._status_switch_default.rebuild()

    def _update_value_limitation(self):
        if self._brush[SCALE_KEY]["max"] - self._brush[SCALE_KEY]["min"] < MIN_X_RANGE:
            self._brush[SCALE_KEY]["max"] = self._brush[SCALE_KEY]["min"] + MIN_X_RANGE
        self._min_spin.max = self._brush[SCALE_KEY]["max"] - MIN_X_RANGE
        self._max_spin.min = self._brush[SCALE_KEY]["min"] + MIN_X_RANGE

    def _from_bias_to_offset(self):
        value_offset = self._brush[SCALE_KEY]["bias"]
        image_width = self._scale_plot.computed_width - 2 * PLOT_PADDING
        offset_x = value_offset * image_width + PLOT_PADDING
        return offset_x

    def _from_offset_to_bias(self, offset_x):
        image_width = self._scale_plot.computed_width - 2 * PLOT_PADDING
        bias = (offset_x - PLOT_PADDING) / image_width
        return bias

    async def _calculate_bias_pos_async(self):
        await omni.kit.app.get_app().next_update_async()
        self._calculate_bias_pos()

    def _calculate_bias_pos(self):
        offset_x = self._from_bias_to_offset()
        self._update_bias_pos(offset_x)

    def _update_bias_pos(self, offset_x):
        if self._scale_plot.computed_width <= 0:
            return
        # bias spin position
        if offset_x < SPINNER_WIDTH / 2:
            # make bias spin left alignment with plot image
            self._bias_placer.offset_x = ui.Pixel(-BIAS_PLACER_OFFSET)
        elif offset_x + PLOT_PADDING + SPINNER_WIDTH / 2 > self._scale_plot.computed_width:
            # make bias spin right alignment with plot image
            self._bias_placer.offset_x = ui.Pixel(
                self._scale_plot.computed_width - PLOT_PADDING - BIAS_PLACER_OFFSET - SPINNER_WIDTH
            )
        else:
            # make bias spin center alignment with the line
            self._bias_placer.offset_x = ui.Pixel(offset_x - BIAS_PLACER_OFFSET - SPINNER_WIDTH / 2)

        # line position is always changed to offset
        self._line_placer.offset_x = offset_x - ARROW_WIDTH / 2

    def _begin_edit(self, model):
        omni.kit.undo.begin_group()

    def _end_edit(self, model):
        omni.kit.undo.end_group()

    # this is not drag field, we only update value at end edit
    def _bias_end_edit(self, model):
        bias = self._bias_field.model.get_value_as_float()
        omni.kit.commands.execute(
            "ChangeBrushParamCommand",
            brush=self._brush,
            param=SCALE_KEY + "/bias",
            value=bias,
            prev_value=self._brush[SCALE_KEY]["bias"],
        )
        self._calculate_bias_pos()
        self._on_scale_value_changed()
