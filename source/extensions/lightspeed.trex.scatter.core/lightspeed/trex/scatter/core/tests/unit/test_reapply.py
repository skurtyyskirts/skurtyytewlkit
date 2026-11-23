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

import contextlib
from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock

import numpy as np
import omni.kit.test
import omni.usd
from lightspeed.trex.scatter.core import reapply as reapply_module
from lightspeed.trex.scatter.core.constants import CONTAINER_PREFIX, SCATTER_ORIGIN_ATTR, SCATTER_TRANSFORM_KEY_ATTR
from lightspeed.trex.scatter.core.geometry import MeshSurfaceCache
from lightspeed.trex.scatter.core.instancing import is_scatter_instancer, merge_loose_placements
from lightspeed.trex.scatter.core.placement import (
    PlacementTransform,
    author_placement_transforms,
    author_placements,
    generate_stamp,
    transform_key,
)
from lightspeed.trex.scatter.core.reapply import (
    ReapplyDriver,
    ReapplyOutcome,
    ReapplyScope,
    ScopeEntry,
    _PreparedPlacement,
    changed_reappliable_fields,
    changed_recompose_assets,
    prepare_reapply,
    reapply_transforms,
    resolve_scope,
    scope_of_placements,
)
from lightspeed.trex.scatter.core.sampling import PaddingIndex, ParentSpaceOrigin, placement_rng
from lightspeed.trex.scatter.core.settings import ApplyTo, ScatterAssetEntry, ScatterBrushSettings, UpAxis
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
    """Brush settings with the randomness pinned, so a test can change one thing at a time.

    Painted on every mesh and conformed to it, so a selected mesh is a live-edit scope and the recorded normal shapes
    the placement; the Selected mode cases override ``apply_to`` themselves.
    """
    values = {
        "apply_to": ApplyTo.ALL,
        "conform_to_surface": True,
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

    async def test_selecting_instanced_scatter_scopes_to_nothing_rather_than_the_preset_container(self):
        # Arrange
        layer = self.stage.GetRootLayer()
        merge_loose_placements(layer, self.stage, prototype_roots=[_PROTOTYPE_ROOT], exclude=self.glass)
        instancer = f"{_PROTOTYPE_ROOT}/instanced_grass"

        # Act
        scope = resolve_scope(self.stage, _usd_context([instancer]), _settings(preset_name="Glass"))

        # Assert
        self.assertTrue(is_scatter_instancer(self.stage.GetPrimAtPath(instancer)))
        self.assertTrue(scope.is_empty)

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

    async def test_selecting_a_mesh_while_painting_on_the_selection_yields_an_empty_scope(self):
        # The selected mesh is where the next stroke goes, so editing everything already on it would make every
        # setting changed between two strokes rewrite the whole field.
        # Arrange
        settings = _settings(apply_to=ApplyTo.SELECTED)

        # Act
        scope = resolve_scope(self.stage, _usd_context([_INSTANCE_ROOT]), settings)

        # Assert
        self.assertTrue(scope.is_empty)

    async def test_selecting_a_container_while_painting_on_the_selection_scopes_to_its_placements(self):
        # Arrange
        settings = _settings(apply_to=ApplyTo.SELECTED)

        # Act
        scope = resolve_scope(self.stage, _usd_context([_GLASS_CONTAINER]), settings)

        # Assert
        self.assertEqual([str(path) for path in scope.entries[0].placement_paths], sorted(self.glass))

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
        translate, rotate, scale = ((1.25, -2.5, 3.75), (10.1, 20.2, 30.3), (1.1, 2.2, 3.3))
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

    async def test_scope_of_placements_groups_the_placements_by_the_prototype_they_are_authored_under(self):
        # Arrange
        other_prototype = "/RootNode/meshes/mesh_FEDCBA9876543210"
        paths = [
            f"{_GRASS_CONTAINER}/s_000000000002",
            f"{other_prototype}/{CONTAINER_PREFIX}rocks/s_000000000003",
            f"{_GRASS_CONTAINER}/s_000000000001",
        ]

        # Act
        scope = scope_of_placements(paths)

        # Assert
        self.assertEqual(
            [(str(entry.prototype_root), [str(path) for path in entry.placement_paths]) for entry in scope.entries],
            [
                (_PROTOTYPE_ROOT, [f"{_GRASS_CONTAINER}/s_000000000001", f"{_GRASS_CONTAINER}/s_000000000002"]),
                (other_prototype, [f"{other_prototype}/{CONTAINER_PREFIX}rocks/s_000000000003"]),
            ],
        )

    async def test_scope_of_placements_names_the_containers_the_placements_came_from(self):
        # Arrange
        paths = [f"{_GRASS_CONTAINER}/s_000000000001", f"{_GLASS_CONTAINER}/s_000000000002"]

        # Act
        scope = scope_of_placements(paths)

        # Assert
        self.assertEqual(scope.entries[0].container_names, (f"{CONTAINER_PREFIX}glass", f"{CONTAINER_PREFIX}grass"))

    async def test_scope_of_placements_leaves_out_paths_outside_the_capture_topology(self):
        # Arrange
        paths = ["/World/Props/s_000000000001", f"{_GRASS_CONTAINER}/s_000000000002"]

        # Act
        scope = scope_of_placements(paths)

        # Assert
        self.assertEqual(scope, _scope_of([f"{_GRASS_CONTAINER}/s_000000000002"]))

    async def test_scope_of_placements_rewrites_an_instance_side_path_to_the_prototype_side(self):
        # Arrange
        paths = [f"{_INSTANCE_ROOT}/{CONTAINER_PREFIX}grass/s_000000000003"]

        # Act
        scope = scope_of_placements(paths)

        # Assert
        self.assertEqual(scope, _scope_of([f"{_GRASS_CONTAINER}/s_000000000003"]))

    async def test_scope_of_placements_with_no_placement_under_a_prototype_is_empty(self):
        # Arrange
        paths = ["/World/Props/s_000000000001"]

        # Act
        scope = scope_of_placements(paths)

        # Assert
        self.assertTrue(scope.is_empty)


class TestPreparedReapply(_ScatterStageTestCase):
    """A live edit reads its scope once and composes it at every step."""

    def prepare(self):
        return prepare_reapply(self.stage, self.layer, _scope_of(self.grass), _settings(), [_asset()], self.cache)

    def fresh_reapply(self, settings: ScatterBrushSettings, assets: list[ScatterAssetEntry]):
        _before, after, _outcome = reapply_transforms(
            self.stage, self.layer, _scope_of(self.grass), settings, assets, self.cache
        )
        return after

    async def test_compose_at_every_step_of_an_edit_matches_a_fresh_reapply_at_that_step(self):
        # Arrange - sizes dragged, then the seed changed, all on the one prepared scope
        steps = [_settings(size_min=size, size_max=size) for size in (30.0, 40.0, 50.0)]
        steps.append(_settings(size_min=50.0, size_max=50.0, seed=1234))
        steps.append(_settings(seed=5, rotation_x_min=-30.0, rotation_x_max=30.0))
        expected = [self.fresh_reapply(step, [_asset()]) for step in steps]
        prepared = self.prepare()

        # Act
        composed = [prepared.compose(step, [_asset()])[0] for step in steps]

        # Assert
        self.assertEqual(composed, expected)

    async def test_compose_at_unchanged_settings_returns_the_transforms_the_scope_had_when_prepared(self):
        # Arrange
        prepared = self.prepare()

        # Act
        after, _outcome = prepared.compose(_settings(), [_asset()])

        # Assert
        self.assertEqual(after, prepared.before)

    async def test_compose_with_a_changed_palette_up_axis_turns_the_placements_by_the_new_axis(self):
        # Arrange - the edit opened with the Z-up entry; the user then marks the asset Y-up
        prepared = self.prepare()
        y_up = _asset().model_copy(update={"up_axis": UpAxis.Y})
        expected = self.fresh_reapply(_settings(), [y_up])

        # Act
        after, _outcome = prepared.compose(_settings(), [y_up])

        # Assert
        self.assertEqual(after, expected)
        self.assertNotEqual([t.rotate_xyz for t in after], [t.rotate_xyz for t in prepared.before])

    async def test_compose_reports_the_placements_the_preparation_left_out(self):
        # Arrange
        moved = self.grass[0]
        self.layer.GetPrimAtPath(Sdf.Path(moved)).attributes["xformOp:translate"].default = Gf.Vec3d(9.0, 9.0, 9.0)
        prepared = self.prepare()

        # Act
        after, outcome = prepared.compose(_settings(size_min=40.0, size_max=40.0), [_asset()])

        # Assert
        self.assertEqual(outcome.hand_edited, 1)
        self.assertEqual(outcome.reapplied, len(self.grass) - 1)
        self.assertNotIn(moved, [transform.prim_path for transform in after])


def _prepared_placement(key: int) -> _PreparedPlacement:
    """A prepared placement that only matters for its random key."""
    return _PreparedPlacement(
        path=Sdf.Path(f"{_GRASS_CONTAINER}/s_000000000001"),
        parent_world=Gf.Matrix4d(1.0),
        origin=ParentSpaceOrigin(position=Gf.Vec3d(0.0, 0.0, 0.0), normal=Gf.Vec3d(0.0, 0.0, 1.0)),
        key=key,
        yaw=0.0,
        turn=None,
        asset_key=_ASSET_PATH,
        fallback_asset=_asset(),
    )


class TestReplayedDraws(omni.kit.test.AsyncTestCase):
    """A replay stands in for a placement's generator, so it has to draw exactly what that generator draws."""

    # A uniform draw over (low, high), or None for a plain random(); eight is the most a placement may draw.
    _STEPS = [(-180.0, 180.0), (0.0, 360.0), (-12.5, 7.25), None, (0.8, 1.2), (5.0, 5.0), (89.5, 90.5), None]

    async def test_draws_replay_bit_for_bit_what_the_placement_generator_draws(self):
        # Arrange
        keys = [int(key) for key in np.random.default_rng(11).integers(0, 2**63, size=40, dtype=np.uint64)]
        seeds = (0, 7, 2**31 - 1)
        expected = []
        for key in keys:
            for seed in seeds:
                generator = placement_rng(key, seed)
                expected.append(
                    [float(generator.random() if step is None else generator.uniform(*step)) for step in self._STEPS]
                )

        # Act
        replayed = []
        for key in keys:
            placement = _prepared_placement(key)
            for seed in seeds:
                draws = placement.draws(seed)
                replayed.append([draws.random() if step is None else draws.uniform(*step) for step in self._STEPS])

        # Assert
        self.assertEqual(replayed, expected)

    async def test_draws_under_another_seed_replay_the_generator_of_that_seed(self):
        # Arrange
        placement = _prepared_placement(123456789)
        placement.draws(7)
        expected = [float(value) for value in placement_rng(123456789, 8).random(3)]

        # Act
        draws = placement.draws(8)

        # Assert
        self.assertEqual([draws.random() for _ in range(3)], expected)

    async def test_draws_past_what_a_placement_may_draw_raise_index_error(self):
        # Arrange
        draws = _prepared_placement(1).draws(0)
        for _ in range(8):
            draws.random()

        # Act / Assert
        with self.assertRaises(IndexError):
            draws.random()


def _brush(settings: ScatterBrushSettings) -> Mock:
    """A brush controller stand-in whose settings, preview and folding the test decides."""
    brush = Mock(name="controller")
    brush.settings = settings
    brush.assets = [_asset()]
    brush.preview_scope.return_value = None
    brush.folds_into_preview.return_value = False
    return brush


class TestReapplyDriver(_ScatterStageTestCase):
    async def setUp(self):
        await super().setUp()
        self._stack = contextlib.ExitStack()
        self.usd_context = _usd_context([_GRASS_CONTAINER])
        self.usd_context.get_stage.return_value = self.stage
        self._stack.enter_context(
            mock.patch.object(reapply_module.omni.usd, "get_context", return_value=self.usd_context)
        )
        self.execute = self._stack.enter_context(mock.patch.object(reapply_module.omni.kit.commands, "execute"))
        self.brush = _brush(_settings())
        self.driver = ReapplyDriver(self.brush, "", self.cache)

    async def tearDown(self):
        self.driver.destroy()
        self._stack.close()
        await super().tearDown()

    def change_settings(self, **updates) -> None:
        """Change the brush settings and tell the driver, as the controller does."""
        self.brush.settings = self.brush.settings.model_copy(update=updates)
        self.brush.subscribe_settings_changed.call_args.args[0](self.brush.settings)

    def change_asset(self, **updates) -> None:
        """Change the palette's one asset and tell the driver, as the controller does."""
        self.brush.assets = [self.brush.assets[0].model_copy(update=updates)]
        self.brush.subscribe_assets_changed.call_args.args[0](self.brush.assets)

    def preview(self, paths: list[str], container_name: str = f"{CONTAINER_PREFIX}grass") -> ReapplyScope:
        """Make the controller report a stroke preview over the given placements and let the driver follow it."""
        scope = _scope_of(sorted(paths), container_name)
        self.brush.preview_scope.return_value = scope
        self.driver.refresh_scope()
        return scope

    async def test_follow_settings_called_again_with_the_same_settings_changes_nothing_more(self):
        # Arrange - the Scatter window calls it as well as the controller, in whichever order the two are notified
        self.change_settings(size_min=40.0, size_max=40.0)
        authored = [self.authored(path) for path in self.grass]

        # Act
        with mock.patch.object(reapply_module, "author_placement_transforms") as author:
            self.driver.follow_settings(self.brush.settings)

        # Assert
        author.assert_not_called()
        self.assertEqual([self.authored(path) for path in self.grass], authored)
        self.assertTrue(self.driver.active)

    async def test_edit_finished_after_follow_settings_commits_a_change_the_controller_has_not_notified_yet(self):
        # Arrange - the window's handler ran first: it follows the change itself before finishing the edit
        self.brush.settings = self.brush.settings.model_copy(update={"size_min": 40.0, "size_max": 40.0})
        self.driver.follow_settings(self.brush.settings)

        # Act
        self.driver.edit_finished()

        # Assert
        self.execute.assert_called_once()
        self.assertEqual(self.execute.call_args.args[0], "ScatterReapplyCommand")
        self.assertFalse(self.driver.active)

    async def test_an_asset_up_axis_change_recomposes_the_scope_and_commits_it_as_one_command(self):
        # Arrange
        original = [self.authored(path) for path in self.grass]

        # Act
        self.change_asset(up_axis=UpAxis.Y)

        # Assert - no control reports the end of a palette edit, so the change has to commit itself
        self.assertNotEqual([self.authored(path) for path in self.grass], original)
        self.execute.assert_called_once()
        self.assertEqual(self.execute.call_args.args[0], "ScatterReapplyCommand")
        self.assertFalse(self.driver.active)

    async def test_an_asset_change_no_placement_can_follow_opens_no_edit(self):
        # Arrange
        original = [self.authored(path) for path in self.grass]

        # Act
        self.change_asset(weight=9.0)

        # Assert
        self.assertEqual([self.authored(path) for path in self.grass], original)
        self.execute.assert_not_called()
        self.assertFalse(self.driver.active)

    async def test_an_asset_change_during_a_held_edit_stays_part_of_that_edit(self):
        # Arrange - a gizmo drag holds the edit open
        self.driver.hold()

        # Act
        self.change_asset(up_axis=UpAxis.Y)

        # Assert
        self.execute.assert_not_called()
        self.assertTrue(self.driver.active)

    async def test_hold_keeps_edit_finished_from_committing_the_open_edit(self):
        # Arrange
        self.driver.hold()
        self.change_settings(size_min=40.0, size_max=40.0)

        # Act
        self.driver.edit_finished()

        # Assert
        self.execute.assert_not_called()
        self.assertTrue(self.driver.active)

    async def test_release_of_the_last_hold_commits_the_open_edit(self):
        # Arrange
        self.driver.hold()
        self.change_settings(size_min=40.0, size_max=40.0)
        self.driver.edit_finished()

        # Act
        self.driver.release()

        # Assert
        self.execute.assert_called_once()
        self.assertEqual(self.execute.call_args.args[0], "ScatterReapplyCommand")
        self.assertFalse(self.driver.active)

    async def test_release_with_another_hold_left_keeps_the_edit_open(self):
        # Arrange
        self.driver.hold()
        self.driver.hold()
        self.change_settings(size_min=40.0, size_max=40.0)

        # Act
        self.driver.release()

        # Assert
        self.execute.assert_not_called()
        self.assertTrue(self.driver.active)

    async def test_edit_finished_on_a_preview_that_folds_into_its_stroke_executes_no_command(self):
        # Arrange
        scope = self.preview(self.grass)
        self.brush.folds_into_preview.return_value = True
        original = [self.authored(path) for path in self.grass]
        self.change_settings(size_min=40.0, size_max=40.0)

        # Act
        self.driver.edit_finished()

        # Assert - the edit stays in the layer as part of the stroke instead of becoming an undo step of its own
        self.execute.assert_not_called()
        self.brush.folds_into_preview.assert_called_once_with("", self.layer.identifier, scope)
        self.assertNotEqual([self.authored(path) for path in self.grass], original)

    async def test_edit_finished_on_a_preview_that_does_not_fold_executes_a_reapply_command(self):
        # Arrange
        self.preview(self.grass)
        self.change_settings(size_min=40.0, size_max=40.0)

        # Act
        self.driver.edit_finished()

        # Assert
        self.execute.assert_called_once()
        self.assertEqual(self.execute.call_args.args[0], "ScatterReapplyCommand")

    async def test_edit_finished_on_a_selection_executes_a_reapply_command_without_asking_to_fold(self):
        # Arrange
        self.brush.folds_into_preview.return_value = True
        self.change_settings(size_min=40.0, size_max=40.0)

        # Act
        self.driver.edit_finished()

        # Assert
        self.execute.assert_called_once()
        self.brush.folds_into_preview.assert_not_called()

    async def test_refresh_scope_with_a_preview_and_live_editing_on_follows_the_preview(self):
        # Arrange
        preview = _scope_of(sorted(self.glass), f"{CONTAINER_PREFIX}glass")
        self.brush.preview_scope.return_value = preview

        # Act
        self.driver.refresh_scope()

        # Assert
        self.assertTrue(self.driver.previewing)
        self.assertEqual(self.driver.scope, preview)

    async def test_refresh_scope_with_live_editing_off_falls_back_to_the_selection(self):
        # Arrange
        self.brush.settings = _settings(edit_selection_live=False)
        self.brush.preview_scope.return_value = _scope_of(sorted(self.glass), f"{CONTAINER_PREFIX}glass")

        # Act
        self.driver.refresh_scope()

        # Assert
        self.assertFalse(self.driver.previewing)
        self.assertEqual([str(path) for path in self.driver.scope.entries[0].placement_paths], sorted(self.grass))

    async def test_refresh_scope_without_a_preview_follows_the_selection(self):
        # Arrange
        self.preview(self.glass, f"{CONTAINER_PREFIX}glass")
        self.brush.preview_scope.return_value = None

        # Act
        self.driver.refresh_scope()

        # Assert
        self.assertFalse(self.driver.previewing)
        self.assertEqual([str(path) for path in self.driver.scope.entries[0].placement_paths], sorted(self.grass))

    async def test_turning_live_editing_off_moves_the_scope_from_the_preview_to_the_selection(self):
        # Arrange
        self.preview(self.glass, f"{CONTAINER_PREFIX}glass")

        # Act
        self.change_settings(edit_selection_live=False)

        # Assert
        self.assertFalse(self.driver.previewing)
        self.assertEqual([str(path) for path in self.driver.scope.entries[0].placement_paths], sorted(self.grass))

    async def test_switching_to_paint_on_the_selection_drops_a_selected_mesh_from_the_scope(self):
        # Arrange
        self.usd_context.get_selection.return_value.get_selected_prim_paths.return_value = [_INSTANCE_ROOT]
        self.driver.refresh_scope()

        # Act
        self.change_settings(apply_to=ApplyTo.SELECTED)

        # Assert
        self.assertTrue(self.driver.scope.is_empty)

    async def test_a_preview_change_commits_the_edit_opened_on_the_old_scope(self):
        # Arrange
        self.change_settings(size_min=40.0, size_max=40.0)
        self.brush.preview_scope.return_value = _scope_of(sorted(self.glass), f"{CONTAINER_PREFIX}glass")

        # Act
        self.brush.subscribe_preview_changed.call_args.args[0]()

        # Assert
        self.execute.assert_called_once()
        self.assertTrue(self.driver.previewing)

    async def test_a_selection_change_ends_the_preview(self):
        # Arrange
        self.preview(self.grass)
        subscribe = self.usd_context.get_stage_event_stream.return_value.create_subscription_to_pop
        event = SimpleNamespace(type=int(omni.usd.StageEventType.SELECTION_CHANGED))

        # Act
        subscribe.call_args.args[0](event)

        # Assert
        self.brush.end_preview.assert_called_once()

    async def test_summary_of_a_preview_names_it_the_last_stroke(self):
        # Arrange
        self.preview(self.grass)

        # Act
        summary = self.driver.summary()

        # Assert
        self.assertTrue(summary.startswith("Last stroke: "), summary)

    async def test_destroy_of_a_driver_that_owns_its_cache_destroys_the_cache(self):
        # Arrange
        cache = Mock(name="cache")
        driver = ReapplyDriver(self.brush, "", cache, owns_cache=True)

        # Act
        driver.destroy()

        # Assert
        cache.destroy.assert_called_once()

    async def test_destroy_of_a_driver_that_borrows_its_cache_leaves_the_cache_alone(self):
        # Arrange
        cache = Mock(name="cache")
        driver = ReapplyDriver(self.brush, "", cache)

        # Act
        driver.destroy()

        # Assert
        cache.destroy.assert_not_called()
