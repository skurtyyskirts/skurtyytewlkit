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


def _make_target(prototype_name: str) -> SimpleNamespace:
    """Build a ScatterTarget-like value for a prototype hash name."""
    hash_name = prototype_name.removeprefix("mesh_")
    return SimpleNamespace(
        prototype_root=Sdf.Path(f"/RootNode/meshes/{prototype_name}"),
        parent_instance_root=Sdf.Path(f"/RootNode/instances/inst_{hash_name}_0"),
        mesh_path=Sdf.Path(f"/RootNode/instances/inst_{hash_name}_0/mesh"),
        parent_world=Gf.Matrix4d(1.0),
        instance_count=1,
    )


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

        self.generate_stamp = enter(mock.patch.object(stroke_module, "generate_stamp", return_value=[]))
        self.author_placements = enter(mock.patch.object(stroke_module, "author_placements", return_value=[]))
        self.erase_candidates = enter(mock.patch.object(stroke_module, "erase_candidates", return_value=[]))
        self.snapshot_prims = enter(mock.patch.object(stroke_module, "snapshot_prims"))
        self.remove_placements = enter(mock.patch.object(stroke_module, "remove_placements"))
        self.existing_placement_points = enter(
            mock.patch.object(stroke_module, "existing_placement_points", return_value=np.empty((0, 3)))
        )
        self.padding_index_class = enter(mock.patch.object(stroke_module, "PaddingIndex"))
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

    async def test_begin_creates_padding_index_with_cell_size_of_at_least_one(self):
        # Arrange
        session = self._make_session(padding=0.0)

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.padding_index_class.assert_called_once_with(1.0)

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
        session = self._make_session()

        # Act
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Assert
        self.existing_placement_points.assert_called_once_with(self.stage, self.target_a)
        self.padding_index_class.return_value.add_many.assert_called_once_with(existing)

    async def test_update_on_same_target_does_not_reseed_padding_index(self):
        # Arrange
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((20.0, 0.0, 0.0)), self.target_a, _make_sample((20.0, 0.0, 0.0)))

        # Assert
        self.existing_placement_points.assert_called_once_with(self.stage, self.target_a)

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
        self.author_placements.assert_called_once_with(self.layer, records)
        self.assertEqual(session.placed_count, 2)

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

    async def test_update_projects_interpolated_centers_through_cache(self):
        # Arrange
        session = self._make_session(stamp_spacing=10.0, radius=50.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))

        # Act
        session.update(_make_hit((20.0, 0.0, 0.0)), self.target_a, _make_sample((20.0, 0.0, 0.0)))

        # Assert
        self.assertEqual(self.cache.closest_point.call_count, 2)
        for call, expected_x in zip(self.cache.closest_point.call_args_list, (10.0, 20.0)):
            self.assertIs(call.args[0], self.target_a.mesh_path)
            self.assertAlmostEqual(float(call.args[1][0]), expected_x)
            self.assertEqual(call.args[2], 50.0)

    async def test_update_skips_stamps_whose_projection_fails(self):
        # Arrange
        self.cache.closest_point.side_effect = None
        self.cache.closest_point.return_value = None
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
        self.cache.closest_point.assert_not_called()

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
        self.cache.closest_point.assert_not_called()

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
            records=[{"prim_name": "s_1"}, {"prim_name": "s_2"}],
            already_applied=True,
        )
        self.end_interaction.assert_called_once_with(self.token)
        self.assertFalse(session.active)

    async def test_end_accumulates_records_across_stamps(self):
        # Arrange
        self.generate_stamp.side_effect = [[_make_record("s_1")], [_make_record("s_2")], [_make_record("s_3")]]
        session = self._make_session(stamp_spacing=10.0)
        session.begin(_make_hit((0.0, 0.0, 0.0)), self.target_a, _make_sample((0.0, 0.0, 0.0)))
        session.update(_make_hit((20.0, 0.0, 0.0)), self.target_a, _make_sample((20.0, 0.0, 0.0)))

        # Act
        count = session.end()

        # Assert
        self.assertEqual(count, 3)
        self.assertEqual(
            self.execute.call_args.kwargs["records"],
            [{"prim_name": "s_1"}, {"prim_name": "s_2"}, {"prim_name": "s_3"}],
        )

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
        )
        self.generate_stamp.assert_not_called()

    async def test_erase_stamp_passes_brush_settings_to_erase_candidates(self):
        # Arrange
        session = self._make_session(erase=True, radius=40.0)
        sample = _make_sample((1.0, 2.0, 3.0))

        # Act
        session.begin(_make_hit((1.0, 2.0, 3.0)), self.target_a, sample)

        # Assert
        self.erase_candidates.assert_called_once()
        args = self.erase_candidates.call_args.args
        self.assertIs(args[0], self.stage)
        self.assertIs(args[1], self.target_a)
        self.assertEqual(args[2], Gf.Vec3d(1.0, 2.0, 3.0))
        self.assertEqual(args[3], 40.0)
        self.assertEqual(args[4], session.settings.erase_scope)
        self.assertEqual(args[5], ["rock.usd"])

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
