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

__all__ = ["TestStrokeSession"]

import contextlib
import dataclasses
import itertools
import pathlib
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np
import omni.kit.commands
import omni.kit.test
import omni.kit.undo
from pxr import Gf, Sdf, Usd

from lightspeed.trex.scatter.core import commands as commands_module
from lightspeed.trex.scatter.core import stroke as stroke_module
from lightspeed.trex.scatter.core.constants import IS_REMIX_SCATTER_ATTR
from lightspeed.trex.scatter.core.placement import PlacementRecord, author_placements, existing_placement_points
from lightspeed.trex.scatter.core.settings import ScatterAssetEntry, ScatterBrushSettings, UpAxis

_LAYER_IDENTIFIER = "anon:stroke_test.usda"
_ASSET_USDA = """#usda 1.0
(
    defaultPrim = "Root"
    upAxis = "Z"
)

def Xform "Root"
{
    def Cube "Cube"
    {
    }
}
"""


def _make_sample(position) -> SimpleNamespace:
    """Build a SurfaceSample-like value at a world position with an up-facing normal."""
    return SimpleNamespace(
        position=np.asarray(position, dtype=np.float64),
        normal=np.array([0.0, 0.0, 1.0]),
        triangle_index=0,
        distance=0.0,
    )


def _make_target(prototype_name: str, instance_count: int = 1) -> SimpleNamespace:
    """Build a ScatterTarget-like value for a prototype hash name."""
    hash_name = prototype_name.removeprefix("mesh_")
    return SimpleNamespace(
        prototype_root=Sdf.Path(f"/RootNode/meshes/{prototype_name}"),
        parent_instance_root=Sdf.Path(f"/RootNode/instances/inst_{hash_name}_0"),
        mesh_path=Sdf.Path(f"/RootNode/instances/inst_{hash_name}_0/mesh"),
        parent_world=Gf.Matrix4d(1.0),
        instance_count=instance_count,
    )


def _second_instance(target: SimpleNamespace, offset=(0.0, 0.0, 0.0)) -> SimpleNamespace:
    """Build the target the same prototype makes through its second instance, which stands ``offset`` away."""
    values = dict(vars(target))
    values["parent_instance_root"] = Sdf.Path(f"{str(target.parent_instance_root)[:-1]}1")
    values["parent_world"] = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*offset))
    return SimpleNamespace(**values)


def _make_hit(position) -> SimpleNamespace:
    """Build a SurfaceHit-like value."""
    return SimpleNamespace(path=Sdf.Path("/RootNode/instances/inst_A_0/mesh"), world_position=Gf.Vec3d(*position))


def _make_record(name: str) -> mock.Mock:
    """Build a PlacementRecord-like value whose to_dict output is recognisable."""
    record = mock.Mock(name=name)
    record.to_dict.return_value = {"prim_name": name}
    return record


def _make_placement_record(target: SimpleNamespace, name: str, asset_path: str, translate) -> PlacementRecord:
    """Build a real placement record under the default container of ``target`` for authoring on a real stage."""
    return PlacementRecord(
        container_path=str(target.prototype_root.AppendChild("scatter_default")),
        prim_name=name,
        asset_rel_path=asset_path,
        asset_abs_path=asset_path,
        translate=tuple(float(value) for value in translate),
        rotate_xyz=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        brush_id="Default",
    )


class TestStrokeSession(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._stack = contextlib.ExitStack()
        enter = self._stack.enter_context

        self.layer = mock.MagicMock(name="layer")
        self.layer.identifier = _LAYER_IDENTIFIER
        self.stage = mock.MagicMock(name="stage")
        self.stage.GetEditTarget.return_value.GetLayer.return_value = self.layer
        self.context = mock.MagicMock(name="usd_context")
        self.context.get_stage.return_value = self.stage
        self.token = object()

        self.cache = mock.MagicMock(name="cache")
        self.cache.closest_point.side_effect = self._project_on_plane
        self.cache.closest_samples.side_effect = self._project_all_on_plane
        # A clock that never moves keeps every walk inside its time budget; the pacing tests move it themselves.
        self.clock = enter(mock.patch.object(stroke_module, "_clock", return_value=0.0))

        self.generate_stamp = enter(mock.patch.object(stroke_module, "generate_stamp", return_value=[]))
        self.author_placements = enter(mock.patch.object(stroke_module, "author_placements", return_value=[]))
        self.build_materials_payload = enter(
            mock.patch.object(stroke_module, "build_materials_payload", return_value={})
        )
        self.placement_pool = enter(mock.patch.object(stroke_module, "placement_pool"))
        # The pool's per-stamp query is what decides the candidates of each erase stamp.
        self.erase_candidates = self.placement_pool.return_value.take_within
        self.erase_candidates.return_value = []
        self.snapshot_prims = enter(mock.patch.object(stroke_module, "snapshot_prims"))
        self.remove_placements = enter(mock.patch.object(stroke_module, "remove_placements"))
        self.existing_placement_points = enter(
            mock.patch.object(stroke_module, "existing_placement_points", return_value=np.empty((0, 3)))
        )
        self.instanced_points = enter(
            mock.patch.object(stroke_module, "instanced_points", return_value=np.empty((0, 3)))
        )
        self.instanced_pool = enter(mock.patch.object(stroke_module, "instanced_pool"))
        # The instanced pool's per-stamp query decides which instancer entries each erase stamp removes.
        self.instanced_candidates = self.instanced_pool.return_value.take_within
        self.instanced_candidates.return_value = {}
        self.eraser_class = enter(mock.patch.object(stroke_module, "InstancedEraser"))
        self.eraser = self.eraser_class.return_value
        self.eraser.edits = []
        self.eraser.erased_count = 0
        self.padding_index_class = enter(mock.patch.object(stroke_module, "PaddingIndex"))
        # The stage is a mock, so the real reader would raise inside the pxr binding.
        self.stage_meters_per_unit = enter(mock.patch.object(stroke_module, "stage_meters_per_unit", return_value=1.0))
        self.begin_interaction = enter(mock.patch.object(stroke_module, "begin_interaction", return_value=self.token))
        self.end_interaction = enter(mock.patch.object(stroke_module, "end_interaction"))
        self.execute = enter(mock.patch.object(stroke_module.omni.kit.commands, "execute", return_value=(True, None)))
        self.get_context = enter(mock.patch.object(stroke_module.omni.usd, "get_context", return_value=self.context))

        self.target_a = _make_target("mesh_A")
        self.target_b = _make_target("mesh_B")
        self.assets = [ScatterAssetEntry(path="rock.usd")]

    async def tearDown(self):
        self._stack.close()

    @staticmethod
    def _project_on_plane(_mesh_path, point, _max_distance) -> SimpleNamespace:
        """Cache stand-in: every point projects onto itself."""
        return _make_sample(np.asarray(point, dtype=np.float64))

    @staticmethod
    def _project_all_on_plane(_mesh_path, points, _max_distance) -> list[SimpleNamespace]:
        """Batched cache stand-in: every point projects onto itself."""
        return [_make_sample(point) for point in np.asarray(points, dtype=np.float64).reshape(-1, 3)]

    def _run_out_of_time_after(self, clock_reads: int) -> None:
        """Make the stroke clock pass every budget once it has been read ``clock_reads`` times."""
        reads = iter(range(clock_reads))
        self.clock.side_effect = lambda: 0.0 if next(reads, None) is not None else 1.0

    def _make_session(self, erase: bool = False, **settings_overrides) -> stroke_module.StrokeSession:
        settings_kwargs = {"stamp_spacing": 10.0, "radius": 50.0, "padding": 0.0}
        settings_kwargs.update(settings_overrides)
        return stroke_module.StrokeSession(
            usd_context_name="",
            settings=ScatterBrushSettings(**settings_kwargs),
            assets=self.assets,
            cache=self.cache,
            erase=erase,
            stage_up=UpAxis.Z,
            seed=7,
            stroke_index=3,
        )

    def _stamp_centers_x(self) -> list[float]:
        return [round(float(call.args[2].position[0]), 6) for call in self.generate_stamp.call_args_list]

    def _register_scatter_commands(self) -> None:
        """Register the scatter commands for this test and restore the extension's own registration afterwards."""
        were_registered = omni.kit.commands.get_command_class("ScatterStroke") is not None
        registered = omni.kit.commands.register_all_commands_in_module(commands_module)

        def _restore() -> None:
            omni.kit.commands.unregister_module_commands(registered)
            if were_registered:
                omni.kit.commands.register_all_commands_in_module(commands_module)

        self.addCleanup(_restore)

    async def test_begin_opens_interaction_and_stamps_once_at_sample(self):
        # Arrange
        session = self._make_session()
        sample = _make_sample((0.0, 0.0, 0.0))

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, sample)

        # Assert
        self.begin_interaction.assert_called_once_with(self.stage)
        self.generate_stamp.assert_called_once()
        self.assertIs(self.generate_stamp.call_args.args[1], self.target_a)
        self.assertIs(self.generate_stamp.call_args.args[2], sample)
        self.assertIsNone(self.generate_stamp.call_args.args[7])
        self.assertTrue(session.active)

    async def test_begin_hands_every_stamp_the_point_the_stroke_is_seen_from(self):
        # Arrange
        camera = Gf.Vec3d(0.0, 0.0, 500.0)
        session = stroke_module.StrokeSession(
            usd_context_name="",
            settings=ScatterBrushSettings(stamp_spacing=10.0, radius=50.0, padding=0.0),
            assets=self.assets,
            cache=self.cache,
            erase=False,
            stage_up=UpAxis.Z,
            seed=7,
            stroke_index=3,
            view_point=camera,
        )

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(self.generate_stamp.call_args.kwargs["view_point"], camera)

    async def test_begin_without_a_view_point_hands_stamps_none(self):
        # Arrange
        session = self._make_session()

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.assertIsNone(self.generate_stamp.call_args.kwargs["view_point"])

    async def test_begin_creates_padding_index_with_cell_size_smaller_than_one_for_a_small_padding(self):
        # Arrange
        session = self._make_session(padding=0.25)

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.padding_index_class.assert_called_once_with(0.25)

    async def test_begin_creates_padding_index_with_padding_cell_size(self):
        # Arrange
        session = self._make_session(padding=12.5)

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.padding_index_class.assert_called_once_with(12.5)

    async def test_begin_seeds_padding_index_with_existing_placements(self):
        # Arrange
        existing = np.array([[1.0, 2.0, 3.0]])
        self.existing_placement_points.return_value = existing
        session = self._make_session(padding=5.0)

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.existing_placement_points.assert_called_once_with(self.stage, self.target_a)
        self.padding_index_class.return_value.add_many.assert_called_once_with(existing)

    async def test_begin_seeds_padding_index_with_instanced_placements_too(self):
        # Arrange
        instanced = np.array([[5.0, 5.0, 0.0], [7.0, 7.0, 0.0]])
        self.instanced_points.return_value = instanced
        session = self._make_session(padding=5.0)

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.instanced_points.assert_called_once_with(self.stage, self.target_a)
        self.padding_index_class.return_value.add_many.assert_called_once_with(instanced)

    async def test_update_on_same_target_does_not_reseed_padding_index(self):
        # Arrange
        session = self._make_session(stamp_spacing=10.0, padding=5.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((20.0, 0.0, 0.0)), self.target_a, _make_sample((20.0, 0.0, 0.0)))

        # Assert
        self.existing_placement_points.assert_called_once_with(self.stage, self.target_a)

    async def test_update_onto_another_instance_of_the_prototype_reads_what_stands_on_that_instance(self):
        # Arrange - scatter authored under a prototype stands on each of its instances, somewhere else in the world
        other_instance = _second_instance(self.target_a, offset=(100.0, 0.0, 0.0))
        session = self._make_session(stamp_spacing=10.0, padding=5.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((10.0, 0.0, 0.0)), other_instance, _make_sample((10.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(
            self.existing_placement_points.call_args_list,
            [mock.call(self.stage, self.target_a), mock.call(self.stage, other_instance)],
        )
        self.assertEqual(
            self.instanced_points.call_args_list,
            [mock.call(self.stage, self.target_a), mock.call(self.stage, other_instance)],
        )

    async def test_update_back_onto_an_instance_pads_against_what_was_painted_through_another_one_since(self):
        # Arrange - the second instance stands 100 away; one placement is painted through each instance in turn
        other_instance = _second_instance(self.target_a, offset=(100.0, 0.0, 0.0))
        first = dataclasses.replace(
            _make_placement_record(self.target_a, "s_000000000001", "rock.usd", (0.0, 0.0, 0.0)),
            origin=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        second = dataclasses.replace(
            _make_placement_record(self.target_a, "s_000000000002", "rock.usd", (10.0, 0.0, 0.0)),
            origin=(10.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        self.generate_stamp.side_effect = [[first], [second], []]
        session = self._make_session(stamp_spacing=10.0, padding=5.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), other_instance, _make_sample((0.0, 0.0, 0.0)))
        session.update(_make_hit((10.0, 0.0, 0.0)), self.target_a, _make_sample((10.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((20.0, 0.0, 0.0)), other_instance, _make_sample((20.0, 0.0, 0.0)))

        # Assert - the placement painted through the first instance, where it stands on the second
        add_many = self.padding_index_class.return_value.add_many
        add_many.assert_called_once()
        np.testing.assert_allclose(add_many.call_args.args[0], [[110.0, 0.0, 0.0]])

    async def test_begin_without_padding_reads_no_existing_placements(self):
        # Arrange
        session = self._make_session(padding=0.0)

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.existing_placement_points.assert_not_called()

    async def test_begin_without_stage_leaves_session_inactive(self):
        # Arrange
        self.context.get_stage.return_value = None
        session = self._make_session()

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.assertFalse(session.active)
        self.begin_interaction.assert_not_called()
        self.generate_stamp.assert_not_called()

    async def test_begin_authors_generated_records_immediately(self):
        # Arrange
        records = [_make_record("s_1"), _make_record("s_2")]
        self.generate_stamp.return_value = records
        session = self._make_session()

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.author_placements.assert_called_once_with(self.layer, records, {})
        self.assertEqual(session.placed_count, 2)

    async def test_begin_builds_the_materials_payload_once_for_the_whole_stroke(self):
        # Arrange
        session = self._make_session(stamp_spacing=10.0)

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        session.update(_make_hit((35.0, 0.0, 0.0)), self.target_a, _make_sample((35.0, 0.0, 0.0)))

        # Assert
        self.build_materials_payload.assert_called_once_with(self.layer, self.assets)

    async def test_begin_with_assign_textures_off_builds_no_payload(self):
        # Arrange
        self.generate_stamp.return_value = [_make_record("s_1")]
        session = self._make_session(assign_textures=False)

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.build_materials_payload.assert_not_called()
        self.assertEqual(self.author_placements.call_args.args[2], {})

    async def test_begin_passes_the_materials_payload_to_the_authoring_call(self):
        # Arrange
        self.build_materials_payload.return_value = {"rock.usd": [{"shader_rel_path": "S"}]}
        self.generate_stamp.return_value = [_make_record("s_1")]
        session = self._make_session()

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(self.author_placements.call_args.args[2], {"rock.usd": [{"shader_rel_path": "S"}]})

    async def test_update_along_straight_segment_stamps_every_spacing(self):
        # Arrange
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((35.0, 0.0, 0.0)), self.target_a, _make_sample((35.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(self._stamp_centers_x(), [0.0, 10.0, 20.0, 30.0])
        self.assertEqual(self.generate_stamp.call_args.args[7], Gf.Vec3d(1.0, 0.0, 0.0))

    async def test_update_carries_segment_remainder_to_next_update(self):
        # Arrange
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        session.update(_make_hit((35.0, 0.0, 0.0)), self.target_a, _make_sample((35.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((42.0, 0.0, 0.0)), self.target_a, _make_sample((42.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(self._stamp_centers_x(), [0.0, 10.0, 20.0, 30.0, 40.0])

    async def test_update_projects_every_interpolated_center_through_cache_in_one_query(self):
        # Arrange
        session = self._make_session(stamp_spacing=10.0, radius=50.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((20.0, 0.0, 0.0)), self.target_a, _make_sample((20.0, 0.0, 0.0)))

        # Assert
        self.cache.closest_samples.assert_called_once()
        mesh_path, points, max_distance = self.cache.closest_samples.call_args.args
        self.assertIs(mesh_path, self.target_a.mesh_path)
        np.testing.assert_allclose(np.asarray(points)[:, 0], [10.0, 20.0])
        self.assertEqual(max_distance, 50.0)
        self.cache.closest_point.assert_not_called()

    async def test_update_skips_stamps_whose_projection_fails(self):
        # Arrange
        self.cache.closest_samples.side_effect = lambda _mesh, points, _distance: [None] * len(points)
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((35.0, 0.0, 0.0)), self.target_a, _make_sample((35.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(self.generate_stamp.call_count, 1)

    async def test_update_with_new_target_resets_segment_and_stamps_once(self):
        # Arrange
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        sample_b = _make_sample((100.0, 0.0, 0.0))

        # Act
        session.update(_make_hit((100.0, 0.0, 0.0)), self.target_b, sample_b)

        # Assert
        self.assertEqual(self.generate_stamp.call_count, 2)
        self.assertIs(self.generate_stamp.call_args.args[1], self.target_b)
        self.assertIs(self.generate_stamp.call_args.args[2], sample_b)
        self.assertIsNone(self.generate_stamp.call_args.args[7])
        self.cache.closest_samples.assert_not_called()

    async def test_update_after_target_change_continues_from_new_sample(self):
        # Arrange
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        session.update(_make_hit((100.0, 0.0, 0.0)), self.target_b, _make_sample((100.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((115.0, 0.0, 0.0)), self.target_b, _make_sample((115.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(self._stamp_centers_x(), [0.0, 100.0, 110.0])

    async def test_update_with_jump_beyond_stamp_budget_restarts_segment_at_new_sample(self):
        # Arrange
        session = self._make_session(stamp_spacing=1.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        far_sample = _make_sample((10000.0, 0.0, 0.0))

        # Act
        session.update(_make_hit((10000.0, 0.0, 0.0)), self.target_a, far_sample)

        # Assert
        self.assertEqual(self._stamp_centers_x(), [0.0, 10000.0])
        self.assertIs(self.generate_stamp.call_args.args[2], far_sample)
        self.assertEqual(self.generate_stamp.call_args.args[7], Gf.Vec3d(1.0, 0.0, 0.0))
        self.cache.closest_samples.assert_not_called()

    async def test_update_after_budget_restart_continues_walking_from_new_sample(self):
        # Arrange
        session = self._make_session(stamp_spacing=1.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        session.update(_make_hit((10000.0, 0.0, 0.0)), self.target_a, _make_sample((10000.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((10002.5, 0.0, 0.0)), self.target_a, _make_sample((10002.5, 0.0, 0.0)))

        # Assert
        self.assertEqual(self._stamp_centers_x(), [0.0, 10000.0, 10001.0, 10002.0])

    async def test_update_with_segment_exactly_at_stamp_budget_walks_every_stamp(self):
        # Arrange
        budget = stroke_module._MAX_STAMPS_PER_UPDATE
        session = self._make_session(stamp_spacing=1.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((float(budget), 0.0, 0.0)), self.target_a, _make_sample((float(budget), 0.0, 0.0)))

        # Assert
        self.assertEqual(self.generate_stamp.call_count, budget + 1)
        self.assertEqual(self._stamp_centers_x()[-1], float(budget))

    async def test_update_out_of_time_stops_after_the_first_stamp_and_leaves_the_rest_pending(self):
        # Arrange
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        # The update reads the clock once for its deadline; every later read finds the budget spent.
        self._run_out_of_time_after(1)

        # Act
        session.update(_make_hit((35.0, 0.0, 0.0)), self.target_a, _make_sample((35.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(self._stamp_centers_x(), [0.0, 10.0])
        self.assertTrue(session.pending)

    async def test_resume_after_running_out_of_time_makes_the_stamps_an_unhurried_walk_makes(self):
        # Arrange
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        self._run_out_of_time_after(1)
        session.update(_make_hit((35.0, 0.0, 0.0)), self.target_a, _make_sample((35.0, 0.0, 0.0)))
        self.clock.side_effect = None

        # Act
        session.resume()

        # Assert
        self.assertEqual(self._stamp_centers_x(), [0.0, 10.0, 20.0, 30.0])
        self.assertFalse(session.pending)

    async def test_update_while_behind_walks_to_every_sample_given_in_order(self):
        # Arrange
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        self._run_out_of_time_after(1)
        session.update(_make_hit((35.0, 0.0, 0.0)), self.target_a, _make_sample((35.0, 0.0, 0.0)))
        self.clock.side_effect = None

        # Act
        session.update(_make_hit((35.0, 30.0, 0.0)), self.target_a, _make_sample((35.0, 30.0, 0.0)))

        # Assert
        centers = [
            tuple(round(float(value), 6) for value in call.args[2].position[:2])
            for call in self.generate_stamp.call_args_list
        ]
        self.assertEqual(
            centers, [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (35.0, 5.0), (35.0, 15.0), (35.0, 25.0)]
        )

    async def test_resume_when_not_behind_does_not_stamp(self):
        # Arrange
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.resume()

        # Assert
        self.assertEqual(self.generate_stamp.call_count, 1)

    async def test_end_while_behind_makes_every_remaining_stamp_before_committing(self):
        # Arrange
        self.generate_stamp.side_effect = lambda *args, **_kwargs: [_make_record(f"s_{args[2].position[0]:g}")]
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        self._run_out_of_time_after(1)
        session.update(_make_hit((35.0, 0.0, 0.0)), self.target_a, _make_sample((35.0, 0.0, 0.0)))

        # Act
        count = session.end()

        # Assert
        self.assertEqual(self._stamp_centers_x(), [0.0, 10.0, 20.0, 30.0])
        self.assertEqual(count, 4)
        self.assertEqual(len(self.execute.call_args.kwargs["records"]), 4)

    async def test_end_when_catching_up_fails_still_commits_what_was_authored(self):
        # Arrange
        record = _make_record("s_0")
        # The first stamp paints, the one the update gets to before running out of time paints nothing, and the one
        # the end catches up with fails.
        self.generate_stamp.side_effect = [[record], [], RuntimeError("stamp failed")]
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        self._run_out_of_time_after(1)
        session.update(_make_hit((35.0, 0.0, 0.0)), self.target_a, _make_sample((35.0, 0.0, 0.0)))

        with mock.patch.object(stroke_module.carb, "log_error") as log_error:
            # Act
            count = session.end()

        # Assert
        self.assertEqual(count, 1)
        self.assertEqual(self.execute.call_args.kwargs["records"], [record])
        self.assertIn("stamp failed", log_error.call_args.args[0])
        self.end_interaction.assert_called_once_with(self.token)
        self.assertFalse(session.pending)
        # The path past the failure was left unpainted, which is what a caller reports
        self.assertTrue(session.failed)
        self.assertTrue(session.cut_short)

    async def test_update_without_movement_does_not_stamp(self):
        # Arrange
        session = self._make_session()
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(self.generate_stamp.call_count, 1)

    async def test_update_before_begin_does_nothing(self):
        # Arrange
        session = self._make_session()

        # Act
        session.update(_make_hit((35.0, 0.0, 0.0)), self.target_a, _make_sample((35.0, 0.0, 0.0)))

        # Assert
        self.generate_stamp.assert_not_called()
        self.assertFalse(session.active)

    async def test_stamps_use_distinct_stamp_indices_for_rng(self):
        # Arrange
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        with mock.patch.object(stroke_module, "stamp_rng", wraps=stroke_module.stamp_rng) as stamp_rng:
            # Act
            session.update(_make_hit((20.0, 0.0, 0.0)), self.target_a, _make_sample((20.0, 0.0, 0.0)))

        # Assert
        self.assertEqual([call.args for call in stamp_rng.call_args_list], [(7, 3, 1), (7, 3, 2)])

    async def test_end_executes_one_place_command_with_all_records(self):
        # Arrange
        records = [_make_record("s_1"), _make_record("s_2")]
        self.generate_stamp.return_value = records
        session = self._make_session()
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        count = session.end()

        # Assert
        self.assertEqual(count, 2)
        self.execute.assert_called_once_with(
            "ScatterStrokeCommand",
            context_name="",
            layer_identifier=_LAYER_IDENTIFIER,
            kind="PLACE",
            records=records,
            already_applied=True,
            materials={},
            instance_others=True,
        )
        self.end_interaction.assert_called_once_with(self.token)
        self.assertFalse(session.active)

    async def test_end_accumulates_records_across_stamps(self):
        # Arrange
        records = [_make_record("s_1"), _make_record("s_2"), _make_record("s_3")]
        self.generate_stamp.side_effect = [[record] for record in records]
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        session.update(_make_hit((20.0, 0.0, 0.0)), self.target_a, _make_sample((20.0, 0.0, 0.0)))

        # Act
        count = session.end()

        # Assert
        self.assertEqual(count, 3)
        self.assertEqual(self.execute.call_args.kwargs["records"], records)

    async def test_end_with_nothing_placed_executes_no_command(self):
        # Arrange
        session = self._make_session()
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        count = session.end()

        # Assert
        self.assertEqual(count, 0)
        self.execute.assert_not_called()
        self.end_interaction.assert_called_once_with(self.token)

    async def test_end_when_author_placements_raised_still_ends_interaction(self):
        # Arrange
        self.generate_stamp.return_value = [_make_record("s_1")]
        self.author_placements.side_effect = RuntimeError("authoring failed")
        # Nothing reached the layer before the failure.
        self.layer.GetPrimAtPath.return_value = None
        session = self._make_session()
        with contextlib.suppress(RuntimeError):
            session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        count = session.end()

        # Assert
        self.assertEqual(count, 0)
        self.execute.assert_not_called()
        self.end_interaction.assert_called_once_with(self.token)
        self.assertFalse(session.active)

    async def test_end_after_author_placements_raised_part_way_commits_the_placements_it_created(self):
        # Arrange
        created, missing = _make_record("s_1"), _make_record("s_2")
        created.prim_path = "/RootNode/meshes/mesh_A/scatter_default/s_1"
        missing.prim_path = "/RootNode/meshes/mesh_A/scatter_default/s_2"
        self.generate_stamp.return_value = [created, missing]
        self.author_placements.side_effect = RuntimeError("authoring failed")
        self.layer.GetPrimAtPath.side_effect = lambda path: mock.Mock() if path == created.prim_path else None
        session = self._make_session()
        with contextlib.suppress(RuntimeError):
            session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        count = session.end()

        # Assert
        self.assertEqual(count, 1)
        self.assertEqual(self.execute.call_args.kwargs["records"], [created])

    async def test_end_when_command_raises_still_ends_interaction(self):
        # Arrange
        self.generate_stamp.return_value = [_make_record("s_1")]
        self.execute.side_effect = RuntimeError("command failed")
        session = self._make_session()
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        with self.assertRaises(RuntimeError):
            session.end()

        # Assert
        self.end_interaction.assert_called_once_with(self.token)
        self.assertFalse(session.active)

    async def test_end_before_begin_returns_zero(self):
        # Arrange
        session = self._make_session()

        # Act
        count = session.end()

        # Assert
        self.assertEqual(count, 0)
        self.execute.assert_not_called()
        self.end_interaction.assert_not_called()

    async def test_end_called_twice_executes_command_once(self):
        # Arrange
        self.generate_stamp.return_value = [_make_record("s_1")]
        session = self._make_session()
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        session.end()

        # Act
        count = session.end()

        # Assert
        self.assertEqual(count, 0)
        self.execute.assert_called_once()
        self.end_interaction.assert_called_once_with(self.token)

    async def test_abort_when_command_raises_logs_error_and_does_not_raise(self):
        # Arrange
        self.generate_stamp.return_value = [_make_record("s_1")]
        self.execute.side_effect = RuntimeError("command failed")
        session = self._make_session()
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        with mock.patch.object(stroke_module.carb, "log_error") as log_error:
            # Act
            session.abort()

        # Assert
        log_error.assert_called_once()
        self.assertIn("command failed", log_error.call_args.args[0])
        self.end_interaction.assert_called_once_with(self.token)
        self.assertFalse(session.active)

    async def test_abort_before_begin_does_not_raise(self):
        # Arrange
        session = self._make_session()

        # Act
        session.abort()

        # Assert
        self.end_interaction.assert_not_called()

    async def test_erase_session_removes_candidates_and_executes_erase_command_with_snapshot(self):
        # Arrange
        erased_path = "/RootNode/meshes/mesh_A/scatter_default/s_000000000001"
        self.erase_candidates.return_value = [Sdf.Path(erased_path)]
        snapshot = mock.Mock(name="snapshot")
        self.snapshot_prims.return_value = snapshot
        session = self._make_session(erase=True)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        count = session.end()

        # Assert
        self.assertEqual(count, 1)
        self.snapshot_prims.assert_called_once_with(self.layer, [erased_path], into=None)
        self.remove_placements.assert_called_once_with(self.layer, [erased_path])
        self.execute.assert_called_once_with(
            "ScatterStrokeCommand",
            context_name="",
            layer_identifier=_LAYER_IDENTIFIER,
            kind="ERASE",
            prim_paths=[erased_path],
            snapshot_layer=snapshot,
            already_applied=True,
            erased_entries=[],
        )
        self.generate_stamp.assert_not_called()

    async def test_erase_stamp_erases_instanced_entries_and_hands_their_edits_to_the_command(self):
        # Arrange
        instancer = Sdf.Path("/RootNode/meshes/mesh_A/instanced_default")
        ids = np.array([4, 7])
        self.instanced_candidates.return_value = {instancer: ids}
        edit = mock.Mock(name="edit")
        self.eraser.edits = [edit]
        self.eraser.erased_count = 2
        session = self._make_session(erase=True)

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        count = session.end()

        # Assert
        self.eraser_class.assert_called_once_with(self.layer)
        self.instanced_pool.assert_called_once_with(
            self.stage, self.layer, self.target_a, session.settings.erase_scope, ["rock.usd"]
        )
        self.eraser.erase.assert_called_once_with(instancer, ids)
        self.assertEqual(count, 2)
        self.execute.assert_called_once_with(
            "ScatterStrokeCommand",
            context_name="",
            layer_identifier=_LAYER_IDENTIFIER,
            kind="ERASE",
            prim_paths=[],
            snapshot_layer=None,
            already_applied=True,
            erased_entries=[edit],
        )

    async def test_place_stroke_without_instancing_asks_the_command_not_to_merge(self):
        # Arrange
        records = [mock.Mock(name="s_1")]
        self.generate_stamp.return_value = records
        session = self._make_session(instance_strokes=False)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.end()

        # Assert
        self.assertFalse(self.execute.call_args.kwargs["instance_others"])

    async def test_erase_stamp_passes_brush_settings_to_the_placement_pool(self):
        # Arrange
        session = self._make_session(erase=True, radius=40.0)
        sample = _make_sample((1.0, 2.0, 3.0))

        # Act
        session.begin(_make_hit((1.0, 2.0, 3.0)), self.target_a, sample)

        # Assert
        self.placement_pool.assert_called_once_with(
            self.stage, self.target_a, session.settings.erase_scope, ["rock.usd"]
        )
        self.erase_candidates.assert_called_once_with(Gf.Vec3d(1.0, 2.0, 3.0), 40.0)

    async def test_erase_stroke_reads_each_prototype_into_a_pool_once(self):
        # Arrange
        session = self._make_session(erase=True, stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((30.0, 0.0, 0.0)), self.target_a, _make_sample((30.0, 0.0, 0.0)))

        # Assert
        self.placement_pool.assert_called_once()
        self.assertEqual(self.erase_candidates.call_count, 4)

    async def test_erase_stroke_reads_a_pool_for_each_instance_of_the_prototype_it_crosses(self):
        # Arrange - the same placements stand somewhere else on each instance, so one pool cannot serve both
        other_instance = _second_instance(self.target_a, offset=(100.0, 0.0, 0.0))
        session = self._make_session(erase=True, stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((10.0, 0.0, 0.0)), other_instance, _make_sample((10.0, 0.0, 0.0)))

        # Assert
        self.assertEqual([call.args[1] for call in self.placement_pool.call_args_list], [self.target_a, other_instance])
        self.assertEqual([call.args[2] for call in self.instanced_pool.call_args_list], [self.target_a, other_instance])

    async def test_erase_stamp_on_another_instance_skips_a_placement_the_stroke_already_erased(self):
        # Arrange - the pool of each instance holds the same placement, and the first stamp erases it
        path = Sdf.Path("/RootNode/meshes/mesh_A/scatter_default/s_000000000001")
        self.erase_candidates.return_value = [path]
        other_instance = _second_instance(self.target_a)
        session = self._make_session(erase=True, stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((10.0, 0.0, 0.0)), other_instance, _make_sample((10.0, 0.0, 0.0)))

        # Assert
        self.remove_placements.assert_called_once_with(self.layer, [str(path)])

    async def test_erase_update_of_several_stamps_writes_the_instancers_once(self):
        # Arrange
        session = self._make_session(erase=True, stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        self.eraser.flush.reset_mock()

        # Act
        session.update(_make_hit((30.0, 0.0, 0.0)), self.target_a, _make_sample((30.0, 0.0, 0.0)))

        # Assert
        self.eraser.flush.assert_called_once_with()

    async def test_erase_stamp_on_another_instance_writes_the_erased_entries_before_reading_its_pool(self):
        # Arrange - the second pool is read from the layer, so entries erased through the first must be gone from it
        other_instance = _second_instance(self.target_a)
        session = self._make_session(erase=True, stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        order = mock.Mock()
        order.attach_mock(self.eraser.flush, "flush")
        order.attach_mock(self.instanced_pool, "read_pool")

        # Act
        session.update(_make_hit((10.0, 0.0, 0.0)), other_instance, _make_sample((10.0, 0.0, 0.0)))

        # Assert
        steps = [call[0] for call in order.mock_calls if call[0] in ("flush", "read_pool")]
        self.assertEqual(steps, ["flush", "read_pool", "flush"])

    async def test_erase_session_with_no_candidates_executes_no_command(self):
        # Arrange
        session = self._make_session(erase=True)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        count = session.end()

        # Assert
        self.assertEqual(count, 0)
        self.execute.assert_not_called()
        self.snapshot_prims.assert_not_called()
        self.remove_placements.assert_not_called()

    async def test_erase_session_accumulates_later_stamps_into_the_first_snapshot_layer(self):
        # Arrange
        first_path = "/RootNode/meshes/mesh_A/scatter_default/s_1"
        second_path = "/RootNode/meshes/mesh_B/scatter_default/s_2"
        self.erase_candidates.side_effect = [[Sdf.Path(first_path)], [Sdf.Path(second_path)]]
        snapshot = mock.Mock(name="snapshot")
        self.snapshot_prims.return_value = snapshot
        session = self._make_session(erase=True)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((100.0, 0.0, 0.0)), self.target_b, _make_sample((100.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(
            self.snapshot_prims.call_args_list,
            [mock.call(self.layer, [first_path], into=None), mock.call(self.layer, [second_path], into=snapshot)],
        )
        self.assertEqual(session.placed_count, 2)

    async def test_erase_stroke_across_two_prototypes_undo_restores_both_containers_as_defined_prims(self):
        # Arrange
        # This regression drives the real erase pipeline (candidates, snapshot, removal, command, undo) on a real
        # stage, so the collaborator mocks installed by setUp are released first.
        self._stack.close()
        self._register_scatter_commands()
        omni.kit.undo.clear_stack()
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        asset_path = pathlib.Path(temp_dir.name) / "cube.usda"
        asset_path.write_text(_ASSET_USDA, encoding="utf-8")
        stage = Usd.Stage.CreateInMemory()
        layer = stage.GetRootLayer()
        for target in (self.target_a, self.target_b):
            stage.DefinePrim(target.prototype_root, "Xform")
        record_a = _make_placement_record(self.target_a, "s_a", asset_path.as_posix(), (0.0, 0.0, 0.0))
        record_b = _make_placement_record(self.target_b, "s_b", asset_path.as_posix(), (1000.0, 0.0, 0.0))
        author_placements(layer, [record_a, record_b])
        session = self._make_session(erase=True)
        usd_context = SimpleNamespace(get_stage=lambda: stage)
        with mock.patch.object(stroke_module.omni.usd, "get_context", return_value=usd_context):
            session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
            session.update(_make_hit((1000.0, 0.0, 0.0)), self.target_b, _make_sample((1000.0, 0.0, 0.0)))
            erased = session.end()
        self.assertEqual(erased, 2)
        self.assertIsNone(layer.GetPrimAtPath(record_b.container_path))

        # Act
        omni.kit.undo.undo()

        # Assert
        for record in (record_a, record_b):
            container = layer.GetPrimAtPath(record.container_path)
            self.assertEqual(container.specifier, Sdf.SpecifierDef)
            self.assertEqual(container.typeName, "Xform")
            self.assertEqual(container.attributes[IS_REMIX_SCATTER_ATTR].default, True)
            self.assertTrue(stage.GetPrimAtPath(record.container_path).IsDefined())
            self.assertTrue(stage.GetPrimAtPath(record.prim_path).IsDefined())
        self.assertEqual(existing_placement_points(stage, self.target_a).shape, (1, 3))
        self.assertEqual(existing_placement_points(stage, self.target_b).shape, (1, 3))

    async def test_erase_stamp_skips_candidates_without_spec_in_edit_layer(self):
        # Arrange
        local_path = "/RootNode/meshes/mesh_A/scatter_default/s_local"
        foreign_path = "/RootNode/meshes/mesh_A/scatter_default/s_foreign"
        self.erase_candidates.return_value = [Sdf.Path(local_path), Sdf.Path(foreign_path)]
        self.layer.GetPrimAtPath.side_effect = lambda path: None if path == Sdf.Path(foreign_path) else mock.Mock()
        session = self._make_session(erase=True)

        with mock.patch.object(stroke_module.carb, "log_warn") as log_warn:
            # Act
            session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.snapshot_prims.assert_called_once_with(self.layer, [local_path], into=None)
        self.remove_placements.assert_called_once_with(self.layer, [local_path])
        self.assertEqual(session.placed_count, 1)
        log_warn.assert_called_once()
        self.assertIn(_LAYER_IDENTIFIER, log_warn.call_args.args[0])

    async def test_erase_stroke_with_only_foreign_layer_candidates_warns_once_and_erases_nothing(self):
        # Arrange
        self.erase_candidates.return_value = [Sdf.Path("/RootNode/meshes/mesh_A/scatter_default/s_foreign")]
        self.layer.GetPrimAtPath.return_value = None
        session = self._make_session(erase=True, stamp_spacing=10.0)

        with mock.patch.object(stroke_module.carb, "log_warn") as log_warn:
            session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

            # Act
            session.update(_make_hit((20.0, 0.0, 0.0)), self.target_a, _make_sample((20.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(self.erase_candidates.call_count, 3)
        log_warn.assert_called_once()
        self.assertEqual(session.placed_count, 0)
        self.snapshot_prims.assert_not_called()
        self.remove_placements.assert_not_called()

    async def test_placed_count_counts_records_across_stamps(self):
        # Arrange
        self.generate_stamp.side_effect = [[_make_record("s_1"), _make_record("s_2")], [_make_record("s_3")]]
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((10.0, 0.0, 0.0)), self.target_a, _make_sample((10.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(session.placed_count, 3)

    async def test_placed_count_for_erase_session_counts_removed_prims(self):
        # Arrange
        self.erase_candidates.return_value = [Sdf.Path("/RootNode/meshes/mesh_A/scatter_default/s_1")]
        self.snapshot_prims.return_value = mock.Mock(name="snapshot")
        session = self._make_session(erase=True)

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(session.placed_count, 1)

    async def test_rendered_prims_before_begin_returns_zero(self):
        # Arrange
        session = self._make_session()

        # Act
        rendered = session.rendered_prims

        # Assert
        self.assertEqual(rendered, 0)

    async def test_rendered_prims_with_single_instance_target_matches_placed_count(self):
        # Arrange
        self.generate_stamp.return_value = [_make_record("a"), _make_record("b")]
        session = self._make_session()

        # Act
        session.begin(
            _make_hit((0.0, 0.0, 0.0)), _make_target("mesh_A", instance_count=1), _make_sample((0.0, 0.0, 0.0))
        )

        # Assert
        self.assertEqual(session.rendered_prims, session.placed_count)

    async def test_rendered_prims_with_multi_instance_target_multiplies_authored_count(self):
        # Arrange
        self.generate_stamp.return_value = [_make_record("a"), _make_record("b")]
        session = self._make_session()

        # Act
        session.begin(
            _make_hit((0.0, 0.0, 0.0)), _make_target("mesh_A", instance_count=50), _make_sample((0.0, 0.0, 0.0))
        )

        # Assert
        self.assertEqual(session.rendered_prims, 100)

    async def test_rendered_prims_with_zero_instance_target_counts_placements_once(self):
        # Arrange
        self.generate_stamp.return_value = [_make_record("a")]
        session = self._make_session()

        # Act
        session.begin(
            _make_hit((0.0, 0.0, 0.0)), _make_target("mesh_A", instance_count=0), _make_sample((0.0, 0.0, 0.0))
        )

        # Assert
        self.assertEqual(session.rendered_prims, 1)

    async def test_rendered_prims_with_erase_stroke_stays_zero(self):
        # Arrange
        self.erase_candidates.return_value = [Sdf.Path("/RootNode/meshes/mesh_A/scatter_default/s_1")]
        session = self._make_session(erase=True)

        # Act
        session.begin(
            _make_hit((0.0, 0.0, 0.0)), _make_target("mesh_A", instance_count=50), _make_sample((0.0, 0.0, 0.0))
        )

        # Assert
        self.assertEqual(session.rendered_prims, 0)

    async def test_stamp_that_would_pass_the_prim_limit_is_trimmed_to_what_fits(self):
        # Arrange
        self.generate_stamp.side_effect = lambda *_args, **_kwargs: [_make_record(f"s_{index}") for index in range(8)]
        session = self._make_session(stamp_spacing=10.0, prim_budget=2)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((10.0, 0.0, 0.0)), self.target_a, _make_sample((10.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(session.prim_limit, 10)
        self.assertEqual(session.placed_count, 10)
        self.assertEqual(len(self.author_placements.call_args.args[1]), 2)
        self.assertTrue(session.limit_reached)

    async def test_stamp_below_the_prim_limit_leaves_the_stroke_painting(self):
        # Arrange
        self.generate_stamp.return_value = [_make_record("s_1")]
        session = self._make_session(prim_budget=2)

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.assertFalse(session.limit_reached)
        self.assertEqual(session.placed_count, 1)

    async def test_update_after_the_prim_limit_was_reached_stamps_nothing_more(self):
        # Arrange
        self.generate_stamp.side_effect = lambda *_args, **_kwargs: [_make_record(f"s_{index}") for index in range(8)]
        session = self._make_session(stamp_spacing=10.0, prim_budget=1)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        stamps_at_the_limit = self.generate_stamp.call_count

        # Act
        session.update(_make_hit((100.0, 0.0, 0.0)), self.target_a, _make_sample((100.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(self.generate_stamp.call_count, stamps_at_the_limit)
        self.assertEqual(session.placed_count, 5)
        self.assertFalse(session.pending)
        self.cache.closest_samples.assert_not_called()

    async def test_reaching_the_prim_limit_part_way_along_a_segment_stops_the_walk_there(self):
        # Arrange
        self.generate_stamp.side_effect = lambda *_args, **_kwargs: [_make_record("s_a"), _make_record("s_b")]
        session = self._make_session(stamp_spacing=10.0, prim_budget=1)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((100.0, 0.0, 0.0)), self.target_a, _make_sample((100.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(self._stamp_centers_x(), [0.0, 10.0, 20.0])
        self.assertEqual(session.placed_count, 5)
        self.assertTrue(session.limit_reached)
        self.assertFalse(session.pending)

    async def test_stamp_on_a_replicated_target_counts_every_instance_against_the_prim_limit(self):
        # Arrange
        self.generate_stamp.return_value = [_make_record(f"s_{index}") for index in range(8)]
        session = self._make_session(prim_budget=2)

        # Act
        session.begin(
            _make_hit((0.0, 0.0, 0.0)), _make_target("mesh_A", instance_count=5), _make_sample((0.0, 0.0, 0.0))
        )

        # Assert
        self.assertEqual(session.placed_count, 2)
        self.assertEqual(session.rendered_prims, 10)
        self.assertTrue(session.limit_reached)

    async def test_begin_on_a_target_with_more_instances_than_the_prim_limit_skips_it_and_notes_it(self):
        # Arrange - one placement on eleven instances already renders more than the limit of 10
        self.generate_stamp.return_value = [_make_record("s_1")]
        session = self._make_session(prim_budget=2)
        replicated = _make_target("mesh_A", instance_count=11)

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), replicated, _make_sample((0.0, 0.0, 0.0)))

        # Assert - the stroke is not at its limit: it has painted nothing, and another mesh may still fit
        self.generate_stamp.assert_not_called()
        self.assertEqual(session.placed_count, 0)
        self.assertFalse(session.limit_reached)
        self.assertEqual(dict(session.blocked_prototypes), {replicated.prototype_root: 11})

    async def test_update_from_a_mesh_with_too_many_instances_onto_one_with_room_paints_there(self):
        # Arrange
        self.generate_stamp.return_value = [_make_record("s_1")]
        session = self._make_session(prim_budget=2)
        session.begin(
            _make_hit((0.0, 0.0, 0.0)), _make_target("mesh_B", instance_count=11), _make_sample((0.0, 0.0, 0.0))
        )

        # Act
        session.update(_make_hit((10.0, 0.0, 0.0)), self.target_a, _make_sample((10.0, 0.0, 0.0)))

        # Assert
        self.generate_stamp.assert_called_once()
        self.assertIs(self.generate_stamp.call_args.args[1], self.target_a)
        self.assertEqual(session.placed_count, 1)

    async def test_can_place_on_a_target_with_more_instances_than_the_stroke_has_room_for_returns_false(self):
        # Arrange - a limit of 10 rendered prims
        session = self._make_session(prim_budget=2)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        answers = [session.can_place_on(_make_target("mesh_B", instance_count=count)) for count in (1, 10, 11)]

        # Assert
        self.assertEqual(answers, [True, True, False])

    async def test_can_place_on_any_target_after_the_stroke_filled_its_prim_limit_returns_false(self):
        # Arrange
        self.generate_stamp.return_value = [_make_record(f"s_{index}") for index in range(8)]
        session = self._make_session(prim_budget=1)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        answer = session.can_place_on(self.target_b)

        # Assert
        self.assertFalse(answer)

    async def test_stroke_stopped_at_the_prim_limit_still_commits_what_it_placed(self):
        # Arrange
        self.generate_stamp.return_value = [_make_record(f"s_{index}") for index in range(8)]
        session = self._make_session(prim_budget=1)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        count = session.end()

        # Assert
        self.assertEqual(count, 5)
        self.assertEqual(len(self.execute.call_args.kwargs["records"]), 5)
        self.assertFalse(session.cut_short)

    async def test_erase_stroke_with_a_tiny_prim_budget_keeps_erasing(self):
        # Arrange
        session = self._make_session(erase=True, stamp_spacing=10.0, prim_budget=1)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((30.0, 0.0, 0.0)), self.target_a, _make_sample((30.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(self.erase_candidates.call_count, 4)
        self.assertFalse(session.limit_reached)

    async def test_end_out_of_time_leaves_the_rest_of_the_path_unpainted_and_reports_it(self):
        # Arrange
        self.generate_stamp.side_effect = lambda *args, **_kwargs: [_make_record(f"s_{args[2].position[0]:g}")]
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        self._run_out_of_time_after(1)
        session.update(_make_hit((55.0, 0.0, 0.0)), self.target_a, _make_sample((55.0, 0.0, 0.0)))
        # The end reads the clock once for its deadline; every later read finds its whole budget spent.
        self.clock.side_effect = itertools.chain([10.0], itertools.repeat(10.0 + stroke_module._END_DRAIN_BUDGET))

        # Act
        count = session.end()

        # Assert
        self.assertEqual(self._stamp_centers_x(), [0.0, 10.0, 20.0])
        self.assertEqual(count, 3)
        self.assertTrue(session.cut_short)
        self.assertFalse(session.pending)
        self.end_interaction.assert_called_once_with(self.token)

    async def test_end_that_catches_up_in_time_is_not_cut_short(self):
        # Arrange
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        self._run_out_of_time_after(1)
        session.update(_make_hit((35.0, 0.0, 0.0)), self.target_a, _make_sample((35.0, 0.0, 0.0)))

        # Act
        session.end()

        # Assert
        self.assertEqual(self._stamp_centers_x(), [0.0, 10.0, 20.0, 30.0])
        self.assertFalse(session.cut_short)
