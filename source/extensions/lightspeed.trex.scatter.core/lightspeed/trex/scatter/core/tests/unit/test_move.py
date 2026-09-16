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

import omni.kit.test
from lightspeed.trex.scatter.core.constants import SCATTER_ORIGIN_ATTR, SCATTER_TRANSFORM_KEY_ATTR, SCATTER_YAW_ATTR
from lightspeed.trex.scatter.core.move import MoveOutcome, move_scope, scope_pivot
from lightspeed.trex.scatter.core.placement import author_placement_transforms
from lightspeed.trex.scatter.core.reapply import reapply_transforms
from lightspeed.trex.scatter.core.sampling import twist_degrees
from pxr import Gf, Sdf

from .test_reapply import _ScatterStageTestCase, _asset, _scope_of, _settings

_PARENT_OFFSET_X = 100.0


def _translate(x: float, y: float, z: float) -> Gf.Matrix4d:
    """A drag that only slides the patch."""
    return Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, y, z))


def _yaw(degrees: float, pivot: Gf.Vec3d) -> Gf.Matrix4d:
    """A drag that turns the patch about the stage up axis through a pivot."""
    to_origin = Gf.Matrix4d().SetTranslate(-pivot)
    spin = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d.ZAxis(), degrees))
    return to_origin * spin * Gf.Matrix4d().SetTranslate(pivot)


class _MoveTestCase(_ScatterStageTestCase):
    """The two-patch scatter stage, with helpers for reading and driving a gizmo drag."""

    def positions(self, paths: list[str]) -> list[tuple]:
        """Return the authored translation of every placement."""
        return [self.authored(path)[0] for path in paths]

    def attribute(self, path: str, name: str):
        """Return an authored attribute of a placement, or None when it authors none."""
        attribute = self.layer.GetPrimAtPath(Sdf.Path(path)).attributes.get(name)
        return None if attribute is None else attribute.default

    def drag(self, delta: Gf.Matrix4d, paths: list[str]) -> MoveOutcome:
        """Move the placements by a delta and author the result, as a gizmo drag does."""
        _before, after, outcome = move_scope(self.stage, self.layer, _scope_of(paths), delta, self.cache)
        author_placement_transforms(self.layer, after)
        return outcome


class TestScopePivot(_MoveTestCase):
    async def test_pivot_sits_at_the_centre_of_the_patch_in_world_space(self):
        # Arrange
        positions = [self.authored(path)[0] for path in self.grass]
        expected_x = (min(p[0] for p in positions) + max(p[0] for p in positions)) * 0.5 + _PARENT_OFFSET_X

        # Act
        pivot = scope_pivot(self.stage, self.layer, _scope_of(self.grass))

        # Assert
        self.assertAlmostEqual(pivot[0], expected_x, places=6)

    async def test_pivot_of_an_empty_scope_is_none(self):
        # Arrange
        scope = _scope_of([])

        # Act
        pivot = scope_pivot(self.stage, self.layer, scope)

        # Assert
        self.assertIsNone(pivot)


class TestMoveScope(_MoveTestCase):
    async def test_dragging_slides_every_placement_by_the_delta(self):
        # Arrange
        before = self.positions(self.grass)

        # Act
        outcome = self.drag(_translate(30.0, 40.0, 0.0), self.grass)

        # Assert
        self.assertEqual(outcome.moved, len(self.grass))
        moved = [(new[0] - old[0], new[1] - old[1]) for new, old in zip(self.positions(self.grass), before)]
        self.assertTrue(all(abs(dx - 30.0) < 1e-6 and abs(dy - 40.0) < 1e-6 for dx, dy in moved))

    async def test_dragging_settles_every_placement_back_onto_the_flat_surface(self):
        # Arrange
        before = self.positions(self.grass)

        # Act
        outcome = self.drag(_translate(30.0, 0.0, 250.0), self.grass)

        # Assert - the quad is flat at z=0, so the upward part of the drag is settled away
        self.assertEqual(outcome.settled, len(self.grass))
        self.assertEqual([p[2] for p in self.positions(self.grass)], [p[2] for p in before])

    async def test_dragging_never_touches_a_placement_outside_the_scope(self):
        # Arrange
        untouched = [self.authored(path) for path in self.glass]

        # Act
        self.drag(_translate(75.0, -20.0, 0.0), self.grass)

        # Assert
        self.assertEqual([self.authored(path) for path in self.glass], untouched)

    async def test_dragging_leaves_every_placement_the_size_it_was(self):
        # Arrange
        before = [self.authored(path)[2] for path in self.grass]

        # Act
        self.drag(_translate(30.0, 40.0, 0.0), self.grass)

        # Assert
        self.assertEqual([self.authored(path)[2] for path in self.grass], before)

    async def test_turning_the_patch_orbits_its_placements_about_the_pivot(self):
        # Arrange
        pivot = scope_pivot(self.stage, self.layer, _scope_of(self.grass))
        before = self.positions(self.grass)

        # Act
        self.drag(_yaw(90.0, pivot), self.grass)

        # Assert - a rigid turn keeps every distance to the pivot, in the parent space the ops are authored in
        local_pivot = Gf.Vec3d(pivot[0] - _PARENT_OFFSET_X, pivot[1], pivot[2])
        radius = [Gf.Vec3d(*p) - local_pivot for p in before]
        turned = [Gf.Vec3d(*p) - local_pivot for p in self.positions(self.grass)]
        for old, new in zip(radius, turned):
            self.assertAlmostEqual(old.GetLength(), new.GetLength(), places=4)

    async def test_turning_the_patch_records_the_turn_so_a_later_settings_change_keeps_it(self):
        # Arrange
        pivot = scope_pivot(self.stage, self.layer, _scope_of(self.grass))

        # Act
        self.drag(_yaw(90.0, pivot), self.grass)

        # Assert
        recorded = [self.attribute(path, SCATTER_YAW_ATTR) for path in self.grass]
        self.assertTrue(all(abs(value - 90.0) < 1e-4 for value in recorded))

    async def test_a_settings_change_after_a_turn_leaves_the_patch_turned(self):
        # Arrange
        pivot = scope_pivot(self.stage, self.layer, _scope_of(self.grass))
        self.drag(_yaw(90.0, pivot), self.grass)
        turned = [self.authored(path)[1] for path in self.grass]

        # Act
        self.reapply(_settings(), self.grass)

        # Assert
        self.assertEqual([self.authored(path)[1] for path in self.grass], turned)

    async def test_dragging_moves_the_recorded_surface_with_the_placement(self):
        # Arrange
        before = [self.attribute(path, SCATTER_ORIGIN_ATTR)[0] for path in self.grass]

        # Act
        self.drag(_translate(30.0, 0.0, 0.0), self.grass)

        # Assert
        after = [self.attribute(path, SCATTER_ORIGIN_ATTR)[0] for path in self.grass]
        self.assertTrue(all(abs(new - old - 30.0) < 1e-6 for new, old in zip(after, before)))

    async def test_a_settings_change_after_a_drag_recomposes_the_patch_where_it_now_sits(self):
        # Arrange
        self.drag(_translate(30.0, 40.0, 0.0), self.grass)
        dragged = self.positions(self.grass)

        # Act
        self.reapply(_settings(), self.grass)

        # Assert
        self.assertEqual(self.positions(self.grass), dragged)

    async def test_dragging_does_not_resize_a_patch_when_the_window_settings_have_moved_on(self):
        # Arrange - the panel now says something very different from what painted this patch
        before = [self.authored(path)[2] for path in self.grass]

        # Act
        self.drag(_translate(30.0, 0.0, 0.0), self.grass)

        # Assert - a move transforms what is there, it never re-derives from the settings
        self.assertEqual([self.authored(path)[2] for path in self.grass], before)

    async def test_a_hand_edited_placement_travels_with_the_patch_and_stays_the_users(self):
        # Arrange
        nudged = self.grass[0]
        spec = self.layer.GetPrimAtPath(Sdf.Path(nudged))
        moved_by_hand = Gf.Vec3d(*self.authored(nudged)[0]) + Gf.Vec3d(0.0, 0.0, 12.0)
        spec.attributes["xformOp:translate"].default = moved_by_hand
        stale_key = self.attribute(nudged, SCATTER_TRANSFORM_KEY_ATTR)

        # Act
        outcome = self.drag(_translate(30.0, 0.0, 0.0), self.grass)

        # Assert - it followed the drag, and its checksum is still the stale one that marks it as hand-edited
        self.assertEqual(outcome.hand_edited, 1)
        self.assertAlmostEqual(self.authored(nudged)[0][0], moved_by_hand[0] + 30.0, places=6)
        self.assertEqual(self.attribute(nudged, SCATTER_TRANSFORM_KEY_ATTR), stale_key)

    async def test_a_later_settings_change_still_leaves_a_dragged_hand_edited_placement_alone(self):
        # Arrange
        nudged = self.grass[0]
        spec = self.layer.GetPrimAtPath(Sdf.Path(nudged))
        spec.attributes["xformOp:translate"].default = Gf.Vec3d(*self.authored(nudged)[0]) + Gf.Vec3d(0.0, 0.0, 12.0)
        self.drag(_translate(30.0, 0.0, 0.0), self.grass)

        # Act
        outcome = self.reapply(_settings(size_min=80.0, size_max=80.0), self.grass)

        # Assert
        self.assertEqual(outcome.hand_edited, 1)

    async def test_dragging_gives_a_placement_that_recorded_no_surface_one_of_its_own(self):
        # Arrange
        legacy = self.grass[0]
        spec = self.layer.GetPrimAtPath(Sdf.Path(legacy))
        spec.RemoveProperty(spec.attributes[SCATTER_ORIGIN_ATTR])

        # Act
        outcome = self.drag(_translate(30.0, 0.0, 0.0), [legacy])

        # Assert
        self.assertEqual(outcome.recovered, 1)
        self.assertEqual(len(self.attribute(legacy, SCATTER_ORIGIN_ATTR)), 6)

    async def test_undoing_a_drag_restores_the_transform_the_surface_and_the_turn_together(self):
        # Arrange
        pivot = scope_pivot(self.stage, self.layer, _scope_of(self.grass))
        original = [self.authored(path) for path in self.grass]
        origins = [tuple(self.attribute(path, SCATTER_ORIGIN_ATTR)) for path in self.grass]
        before, after, _outcome = move_scope(
            self.stage, self.layer, _scope_of(self.grass), _yaw(45.0, pivot), self.cache
        )
        author_placement_transforms(self.layer, after)

        # Act
        author_placement_transforms(self.layer, before)

        # Assert
        self.assertEqual([self.authored(path) for path in self.grass], original)
        self.assertEqual([tuple(self.attribute(path, SCATTER_ORIGIN_ATTR)) for path in self.grass], origins)
        self.assertTrue(all(abs(self.attribute(path, SCATTER_YAW_ATTR)) < 1e-9 for path in self.grass))

    async def test_a_reapply_after_an_undone_drag_puts_the_patch_where_it_started(self):
        # Arrange
        original = self.positions(self.grass)
        before, after, _outcome = move_scope(
            self.stage, self.layer, _scope_of(self.grass), _translate(200.0, 0.0, 0.0), self.cache
        )
        author_placement_transforms(self.layer, after)
        author_placement_transforms(self.layer, before)

        # Act
        self.reapply(_settings(), self.grass)

        # Assert
        self.assertEqual(self.positions(self.grass), original)

    async def test_dragging_counts_a_placement_with_no_opinion_in_the_layer_as_foreign(self):
        # Arrange
        missing = f"{self.grass[0].rsplit('/', 1)[0]}/s_ffffffffffff"

        # Act
        outcome = self.drag(_translate(30.0, 0.0, 0.0), [missing])

        # Assert
        self.assertEqual(outcome.foreign_layer, 1)
        self.assertEqual(outcome.moved, 0)


class TestMoveHelpers(omni.kit.test.AsyncTestCase):
    async def test_twist_degrees_of_a_turn_about_the_axis_is_that_turn(self):
        # Arrange
        rotation = Gf.Rotation(Gf.Vec3d.ZAxis(), 35.0)

        # Act
        twist = twist_degrees(rotation, Gf.Vec3d.ZAxis())

        # Assert
        self.assertAlmostEqual(twist, 35.0, places=6)

    async def test_twist_degrees_of_a_turn_across_the_axis_is_zero(self):
        # Arrange
        rotation = Gf.Rotation(Gf.Vec3d.XAxis(), 60.0)

        # Act
        twist = twist_degrees(rotation, Gf.Vec3d.ZAxis())

        # Assert
        self.assertAlmostEqual(twist, 0.0, places=6)

    async def test_twist_degrees_reports_the_direction_of_the_turn(self):
        # Arrange
        rotation = Gf.Rotation(Gf.Vec3d.ZAxis(), -50.0)

        # Act
        twist = twist_degrees(rotation, Gf.Vec3d.ZAxis())

        # Assert
        self.assertAlmostEqual(twist, -50.0, places=6)

    async def test_twist_degrees_about_a_zero_axis_is_zero(self):
        # Arrange
        rotation = Gf.Rotation(Gf.Vec3d.ZAxis(), 35.0)

        # Act
        twist = twist_degrees(rotation, Gf.Vec3d(0.0, 0.0, 0.0))

        # Assert
        self.assertEqual(twist, 0.0)

    async def test_outcome_describe_reports_only_what_did_not_simply_move(self):
        # Arrange
        outcome = MoveOutcome(moved=10, settled=8, unsettled=2, hand_edited=1)

        # Act
        description = outcome.describe()

        # Assert
        self.assertEqual(description, "2 could not reach the surface; 1 hand-edited kept their own transform")

    async def test_outcome_describe_of_a_clean_drag_is_empty(self):
        # Arrange
        outcome = MoveOutcome(moved=10, settled=10)

        # Act
        description = outcome.describe()

        # Assert
        self.assertEqual(description, "")


class TestMoveAndReapplyTogether(_MoveTestCase):
    async def test_a_drag_then_a_resize_keeps_the_patch_where_it_was_dragged(self):
        # Arrange
        self.drag(_translate(120.0, -35.0, 0.0), self.grass)
        dragged = self.positions(self.grass)

        # Act
        _before, after, _outcome = reapply_transforms(
            self.stage,
            self.layer,
            _scope_of(self.grass),
            _settings(size_min=50.0, size_max=50.0),
            [_asset()],
            self.cache,
        )
        author_placement_transforms(self.layer, after)

        # Assert
        self.assertEqual(self.positions(self.grass), dragged)
