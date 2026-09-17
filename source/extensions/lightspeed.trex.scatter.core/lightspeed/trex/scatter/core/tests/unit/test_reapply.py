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

from unittest.mock import Mock

import numpy as np
import omni.kit.test
from lightspeed.trex.scatter.core.constants import CONTAINER_PREFIX, SCATTER_ORIGIN_ATTR, SCATTER_TRANSFORM_KEY_ATTR
from lightspeed.trex.scatter.core.geometry import MeshSurfaceCache
from lightspeed.trex.scatter.core.placement import (
    PlacementTransform,
    author_placement_transforms,
    author_placements,
    generate_stamp,
    transform_key,
)
from lightspeed.trex.scatter.core.reapply import (
    ReapplyOutcome,
    ReapplyScope,
    ScopeEntry,
    changed_reappliable_fields,
    changed_recompose_assets,
    reapply_transforms,
    resolve_scope,
)
from lightspeed.trex.scatter.core.sampling import PaddingIndex
from lightspeed.trex.scatter.core.settings import ScatterAssetEntry, ScatterBrushSettings, UpAxis
from lightspeed.trex.scatter.core.targets import ScatterTarget
from pxr import Gf, Sdf, Usd, UsdGeom

_HASH = "0123456789ABCDEF"
_PROTOTYPE_ROOT = f"/RootNode/meshes/mesh_{_HASH}"
_INSTANCE_ROOT = f"/RootNode/instances/inst_{_HASH}_0"
_MESH_PATH = f"{_INSTANCE_ROOT}/mesh"
_GRASS_CONTAINER = f"{_PROTOTYPE_ROOT}/{CONTAINER_PREFIX}grass"
_GLASS_CONTAINER = f"{_PROTOTYPE_ROOT}/{CONTAINER_PREFIX}glass"
_ASSET_PATH = "C:/assets/blade.usda"
_PARENT_OFFSET = Gf.Vec3d(100.0, 0.0, 0.0)
_HALF_SIZE = 500.0


def _settings(**overrides) -> ScatterBrushSettings:
    """Brush settings with the randomness pinned, so a test can change one thing at a time."""
    values = {
        "preset_name": "Grass",
        "radius": 50.0,
        "density": 6.0,
        "strength": 1.0,
        "padding": 0.0,
        "randomize_seed": False,
        "seed": 7,
        "rotation_z_min": 0.0,
        "rotation_z_max": 360.0,
        "size_min": 10.0,
        "size_max": 20.0,
    }
    values.update(overrides)
    return ScatterBrushSettings(**values)


def _asset() -> ScatterAssetEntry:
    return ScatterAssetEntry(path=_ASSET_PATH, up_axis=UpAxis.Z, meters_per_unit=1.0, natural_extent=(1.0, 1.0, 2.0))


def _target() -> ScatterTarget:
    return ScatterTarget(
        prototype_root=Sdf.Path(_PROTOTYPE_ROOT),
        parent_instance_root=Sdf.Path(_INSTANCE_ROOT),
        mesh_path=Sdf.Path(_MESH_PATH),
        parent_world=Gf.Matrix4d().SetTranslate(_PARENT_OFFSET),
        instance_count=1,
    )


def _build_scene(stage: Usd.Stage) -> None:
    """Author an instance translated by ``_PARENT_OFFSET`` holding a flat quad in the XY plane, plus the prototype.

    The stage is made Z-up and one unit per meter explicitly: a Remix capture is Z-up, and a stage that authors no
    upAxis reads back as Y-up, which would compose every placement against a different axis than it was painted on.
    """
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    instance = UsdGeom.Xform.Define(stage, _INSTANCE_ROOT)
    instance.AddTranslateOp().Set(_PARENT_OFFSET)
    mesh = UsdGeom.Mesh.Define(stage, _MESH_PATH)
    mesh.GetPointsAttr().Set(
        [
            (-_HALF_SIZE, -_HALF_SIZE, 0.0),
            (_HALF_SIZE, -_HALF_SIZE, 0.0),
            (_HALF_SIZE, _HALF_SIZE, 0.0),
            (-_HALF_SIZE, _HALF_SIZE, 0.0),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    stage.DefinePrim(_PROTOTYPE_ROOT, "Xform")


def _paint(stage: Usd.Stage, settings: ScatterBrushSettings, preset_name: str) -> list[str]:
    """Paint one stamp of ``preset_name`` into the stage's root layer and return the placement paths it created."""
    cache = MeshSurfaceCache(lambda: stage)
    sample = cache.closest_point(_MESH_PATH, Gf.Vec3d(100.0, 0.0, 0.0), 10.0)
    records = generate_stamp(
        cache,
        _target(),
        sample,
        settings.model_copy(update={"preset_name": preset_name}),
        [_asset()],
        np.random.default_rng(1),
        PaddingIndex(1.0),
        None,
        UpAxis.Z,
        1.0,
        stage.GetRootLayer(),
    )
    created = author_placements(stage.GetRootLayer(), records)
    cache.destroy()
    return created


def _usd_context(selected: list[str]) -> Mock:
    """A USD context whose selection reports exactly the given prim paths."""
    context = Mock()
    context.get_selection.return_value.get_selected_prim_paths.return_value = selected
    return context


def _scope_of(paths: list[str], container_name: str = f"{CONTAINER_PREFIX}grass") -> ReapplyScope:
    """A scope over the given placements of the single test prototype."""
    return ReapplyScope(
        entries=(
            ScopeEntry(
                prototype_root=Sdf.Path(_PROTOTYPE_ROOT),
                container_names=(container_name,),
                placement_paths=tuple(Sdf.Path(path) for path in paths),
            ),
        )
    )


class _ScatterStageTestCase(omni.kit.test.AsyncTestCase):
    """A stage holding a flat quad with one grass patch and one glass patch on the same prototype."""

    async def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        _build_scene(self.stage)
        self.layer = self.stage.GetRootLayer()
        self.grass = _paint(self.stage, _settings(), "Grass")
        self.glass = _paint(self.stage, _settings(seed=99), "Glass")
        self.cache = MeshSurfaceCache(lambda: self.stage)

    async def tearDown(self):
        self.cache.destroy()
        self.cache = None
        self.stage = None

    def authored(self, path: str) -> tuple:
        """Return the three transform ops the root layer authors for a placement."""
        spec = self.layer.GetPrimAtPath(Sdf.Path(path))
        return (
            tuple(spec.attributes["xformOp:translate"].default),
            tuple(spec.attributes["xformOp:rotateXYZ"].default),
            tuple(spec.attributes["xformOp:scale"].default),
        )

    def reapply(self, settings: ScatterBrushSettings, paths: list[str]) -> ReapplyOutcome:
        """Recompose the placements under ``settings`` and author the result, as a live edit does."""
        _before, after, outcome = reapply_transforms(
            self.stage, self.layer, _scope_of(paths), settings, [_asset()], self.cache
        )
        author_placement_transforms(self.layer, after)
        return outcome


class TestResolveScope(_ScatterStageTestCase):
    async def test_selecting_a_mesh_scopes_to_the_container_of_the_current_preset(self):
        # Arrange
        settings = _settings(preset_name="Grass")

        # Act
        scope = resolve_scope(self.stage, _usd_context([_INSTANCE_ROOT]), settings)

        # Assert
        self.assertEqual([str(path) for path in scope.entries[0].placement_paths], sorted(self.grass))

    async def test_selecting_a_mesh_with_another_preset_scopes_to_that_presets_container(self):
        # Arrange
        settings = _settings(preset_name="Glass")

        # Act
        scope = resolve_scope(self.stage, _usd_context([_INSTANCE_ROOT]), settings)

        # Assert
        self.assertEqual([str(path) for path in scope.entries[0].placement_paths], sorted(self.glass))

    async def test_selecting_a_container_scopes_to_its_placements_whatever_the_preset_is(self):
        # Arrange
        settings = _settings(preset_name="Grass")

        # Act
        scope = resolve_scope(self.stage, _usd_context([_GLASS_CONTAINER]), settings)

        # Assert
        self.assertEqual([str(path) for path in scope.entries[0].placement_paths], sorted(self.glass))

    async def test_selecting_one_placement_scopes_to_that_placement_only(self):
        # Arrange
        selected = self.grass[0]

        # Act
        scope = resolve_scope(self.stage, _usd_context([selected]), _settings())

        # Assert
        self.assertEqual([str(path) for path in scope.entries[0].placement_paths], [selected])

    async def test_selecting_a_placement_through_an_instance_resolves_it_on_the_prototype_side(self):
        # Arrange
        instance_side = self.grass[0].replace(_PROTOTYPE_ROOT, _INSTANCE_ROOT)

        # Act
        scope = resolve_scope(self.stage, _usd_context([instance_side]), _settings())

        # Assert
        self.assertEqual([str(path) for path in scope.entries[0].placement_paths], [self.grass[0]])

    async def test_selecting_a_placement_and_a_mesh_keeps_only_the_placement(self):
        # Arrange
        selected = [self.grass[0], _INSTANCE_ROOT]

        # Act
        scope = resolve_scope(self.stage, _usd_context(selected), _settings())

        # Assert
        self.assertEqual([str(path) for path in scope.entries[0].placement_paths], [self.grass[0]])

    async def test_selecting_nothing_yields_an_empty_scope(self):
        # Arrange
        context = _usd_context([])

        # Act
        scope = resolve_scope(self.stage, context, _settings())

        # Assert
        self.assertTrue(scope.is_empty)

    async def test_selecting_a_mesh_with_no_container_for_the_preset_yields_an_empty_scope(self):
        # Arrange
        settings = _settings(preset_name="Rubble")

        # Act
        scope = resolve_scope(self.stage, _usd_context([_INSTANCE_ROOT]), settings)

        # Assert
        self.assertTrue(scope.is_empty)

    async def test_selecting_a_prim_outside_the_capture_topology_yields_an_empty_scope(self):
        # Arrange
        context = _usd_context(["/World/SomethingElse"])

        # Act
        scope = resolve_scope(self.stage, context, _settings())

        # Assert
        self.assertTrue(scope.is_empty)

    async def test_resolving_without_a_stage_yields_an_empty_scope(self):
        # Arrange
        context = _usd_context([_INSTANCE_ROOT])

        # Act
        scope = resolve_scope(None, context, _settings())

        # Assert
        self.assertTrue(scope.is_empty)


class TestReapplyTransforms(_ScatterStageTestCase):
    async def test_reapplying_unchanged_settings_leaves_every_transform_where_it_is(self):
        # Arrange
        original = [self.authored(path) for path in self.grass]

        # Act
        outcome = self.reapply(_settings(), self.grass)

        # Assert
        self.assertEqual(outcome.reapplied, len(self.grass))
        self.assertEqual([self.authored(path) for path in self.grass], original)

    async def test_reapplying_a_larger_size_grows_every_placement_without_moving_it(self):
        # Arrange
        original = [self.authored(path) for path in self.grass]

        # Act
        self.reapply(_settings(size_min=40.0, size_max=40.0), self.grass)

        # Assert
        updated = [self.authored(path) for path in self.grass]
        self.assertEqual([entry[0] for entry in updated], [entry[0] for entry in original])
        self.assertTrue(all(new[2][2] > old[2][2] for new, old in zip(updated, original)))

    async def test_reapplying_a_vertical_offset_lifts_every_placement_by_that_offset(self):
        # Arrange
        original = [self.authored(path) for path in self.grass]

        # Act
        self.reapply(_settings(vertical_offset=5.0), self.grass)

        # Assert
        lifted = [self.authored(path)[0][2] - old[0][2] for path, old in zip(self.grass, original)]
        self.assertTrue(all(abs(value - 5.0) < 1e-6 for value in lifted))

    async def test_reapplying_a_new_seed_rerolls_the_rotation_of_the_scope(self):
        # Arrange
        original = [self.authored(path)[1] for path in self.grass]

        # Act
        self.reapply(_settings(seed=1234), self.grass)

        # Assert
        self.assertNotEqual([self.authored(path)[1] for path in self.grass], original)

    async def test_reapplying_never_touches_a_placement_outside_the_scope(self):
        # Arrange
        untouched = [self.authored(path) for path in self.glass]

        # Act
        self.reapply(_settings(size_min=90.0, size_max=90.0, vertical_offset=25.0), self.grass)

        # Assert
        self.assertEqual([self.authored(path) for path in self.glass], untouched)

    async def test_reapplying_skips_a_placement_whose_transform_was_changed_by_hand(self):
        # Arrange
        moved = self.grass[0]
        self.layer.GetPrimAtPath(Sdf.Path(moved)).attributes["xformOp:translate"].default = Gf.Vec3d(9.0, 9.0, 9.0)

        # Act
        outcome = self.reapply(_settings(size_min=40.0, size_max=40.0), self.grass)

        # Assert
        self.assertEqual(outcome.hand_edited, 1)
        self.assertEqual(self.authored(moved)[0], (9.0, 9.0, 9.0))

    async def test_reapplying_recovers_a_placement_that_recorded_no_surface(self):
        # Arrange
        legacy = self.grass[0]
        spec = self.layer.GetPrimAtPath(Sdf.Path(legacy))
        spec.RemoveProperty(spec.attributes[SCATTER_ORIGIN_ATTR])
        spec.RemoveProperty(spec.attributes[SCATTER_TRANSFORM_KEY_ATTR])

        # Act
        outcome = self.reapply(_settings(vertical_offset=5.0), [legacy])

        # Assert
        self.assertEqual(outcome.recovered, 1)
        self.assertEqual(outcome.reapplied, 1)

    async def test_reapplying_counts_a_placement_with_no_opinion_in_the_layer_as_foreign(self):
        # Arrange
        missing = f"{_GRASS_CONTAINER}/s_ffffffffffff"

        # Act
        outcome = self.reapply(_settings(), [missing])

        # Assert
        self.assertEqual(outcome.foreign_layer, 1)
        self.assertEqual(outcome.reapplied, 0)

    async def test_reapplying_twice_still_owns_the_placements_it_rewrote(self):
        # Arrange
        settings = _settings(size_min=40.0, size_max=40.0)
        self.reapply(settings, self.grass)

        # Act
        outcome = self.reapply(settings, self.grass)

        # Assert
        self.assertEqual(outcome.hand_edited, 0)
        self.assertEqual(outcome.reapplied, len(self.grass))

    async def test_reapplying_returns_the_transforms_needed_to_undo_the_change(self):
        # Arrange
        original = [self.authored(path) for path in self.grass]
        before, after, _outcome = reapply_transforms(
            self.stage,
            self.layer,
            _scope_of(self.grass),
            _settings(size_min=40.0, size_max=40.0),
            [_asset()],
            self.cache,
        )
        author_placement_transforms(self.layer, after)

        # Act
        author_placement_transforms(self.layer, before)

        # Assert
        self.assertEqual([self.authored(path) for path in self.grass], original)

    async def test_reapplying_an_asset_that_left_the_palette_keeps_the_placement_at_its_own_size(self):
        # Arrange
        original = [self.authored(path)[2] for path in self.grass]

        # Act
        _before, after, outcome = reapply_transforms(
            self.stage, self.layer, _scope_of(self.grass), _settings(size_min=40.0, size_max=40.0), [], self.cache
        )

        # Assert
        self.assertEqual(outcome.reapplied, len(self.grass))
        self.assertNotEqual([transform.scale for transform in after], original)


class TestReapplyHelpers(omni.kit.test.AsyncTestCase):
    async def test_changed_reappliable_fields_reports_a_scale_change(self):
        # Arrange
        previous = _settings()

        # Act
        changed = changed_reappliable_fields(previous, previous.model_copy(update={"size_min": 1.0}))

        # Assert
        self.assertEqual(changed, {"size_min"})

    async def test_changed_reappliable_fields_ignores_a_radius_change(self):
        # Arrange
        previous = _settings()

        # Act
        changed = changed_reappliable_fields(previous, previous.model_copy(update={"radius": 123.0}))

        # Assert
        self.assertEqual(changed, set())

    async def test_changed_recompose_assets_reports_an_up_axis_change(self):
        # Arrange
        entry = _asset()

        # Act
        changed = changed_recompose_assets([entry], [entry.model_copy(update={"up_axis": UpAxis.Y})])

        # Assert
        self.assertEqual(changed, {_ASSET_PATH})

    async def test_changed_recompose_assets_reports_a_measurement_landing(self):
        # Arrange
        entry = _asset().model_copy(update={"natural_extent": (0.0, 0.0, 0.0)})

        # Act
        changed = changed_recompose_assets([entry], [entry.model_copy(update={"natural_extent": (1.0, 1.0, 2.0)})])

        # Assert
        self.assertEqual(changed, {_ASSET_PATH})

    async def test_changed_recompose_assets_ignores_a_texture_change(self):
        # Arrange
        entry = _asset()

        # Act
        changed = changed_recompose_assets(
            [entry], [entry.model_copy(update={"texture_source": "C:/assets/blade_albedo.png"})]
        )

        # Assert
        self.assertEqual(changed, set())

    async def test_changed_recompose_assets_ignores_a_weight_change(self):
        # Arrange
        entry = _asset()

        # Act
        changed = changed_recompose_assets([entry], [entry.model_copy(update={"weight": 9.0})])

        # Assert
        self.assertEqual(changed, set())

    async def test_changed_recompose_assets_ignores_an_asset_being_added_to_the_palette(self):
        # Arrange - a placement references its own file, so a palette gaining an entry cannot alter one
        entry = _asset()

        # Act
        changed = changed_recompose_assets([entry], [entry, entry.model_copy(update={"path": "C:/assets/rock.usda"})])

        # Assert
        self.assertEqual(changed, set())

    async def test_changed_recompose_assets_ignores_an_asset_being_removed_from_the_palette(self):
        # Arrange
        entry = _asset()

        # Act
        changed = changed_recompose_assets([entry], [])

        # Assert
        self.assertEqual(changed, set())

    async def test_transform_key_matches_after_the_transform_is_read_back_from_a_layer(self):
        # Arrange
        translate, rotate, scale = (1.25, -2.5, 3.75), (10.1, 20.2, 30.3), (1.1, 2.2, 3.3)
        stage = Usd.Stage.CreateInMemory()
        placement = UsdGeom.Xform.Define(stage, "/Placement")
        placement.AddTranslateOp().Set(Gf.Vec3d(*translate))
        placement.AddRotateXYZOp().Set(Gf.Vec3f(*rotate))
        placement.AddScaleOp().Set(Gf.Vec3f(*scale))
        spec = stage.GetRootLayer().GetPrimAtPath(Sdf.Path("/Placement"))

        # Act
        key = transform_key(
            spec.attributes["xformOp:translate"].default,
            spec.attributes["xformOp:rotateXYZ"].default,
            spec.attributes["xformOp:scale"].default,
        )

        # Assert
        self.assertEqual(key, transform_key(translate, rotate, scale))

    async def test_scope_describe_names_the_mesh_and_the_container(self):
        # Arrange
        scope = _scope_of([f"{_GRASS_CONTAINER}/s_000000000001"])

        # Act
        description = scope.describe()

        # Assert
        self.assertEqual(description, f"1 placement - mesh_{_HASH} / grass")

    async def test_empty_scope_describes_itself_as_nothing(self):
        # Arrange
        scope = ReapplyScope()

        # Act
        description = scope.describe()

        # Assert
        self.assertEqual(description, "")

    async def test_outcome_describe_reports_only_what_did_not_happen(self):
        # Arrange
        outcome = ReapplyOutcome(reapplied=10, hand_edited=2, recovered=3)

        # Act
        description = outcome.describe()

        # Assert
        self.assertEqual(
            description, "2 hand-edited left alone; 3 re-rolled (painted before the brush recorded surfaces)"
        )

    async def test_outcome_describe_of_a_clean_reapply_is_empty(self):
        # Arrange
        outcome = ReapplyOutcome(reapplied=10)

        # Act
        description = outcome.describe()

        # Assert
        self.assertEqual(description, "")

    async def test_placement_transform_round_trips_through_its_dictionary(self):
        # Arrange
        transform = PlacementTransform(
            prim_path=f"{_GRASS_CONTAINER}/s_000000000001",
            translate=(1.0, 2.0, 3.0),
            rotate_xyz=(4.0, 5.0, 6.0),
            scale=(7.0, 8.0, 9.0),
        )

        # Act
        restored = PlacementTransform.from_dict(transform.to_dict())

        # Assert
        self.assertEqual(restored, transform)
