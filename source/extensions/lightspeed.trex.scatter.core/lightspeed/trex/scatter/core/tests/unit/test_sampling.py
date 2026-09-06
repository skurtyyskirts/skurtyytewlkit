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

import math

import numpy as np
import omni.kit.test
from pxr import Gf, Usd, UsdGeom

from lightspeed.trex.scatter.core import sampling
from lightspeed.trex.scatter.core.settings import Falloff, ScatterAssetEntry, ScatterBrushSettings, UpAxis

_SQRT_HALF = math.sqrt(0.5)


def _settings(**overrides) -> ScatterBrushSettings:
    """Deterministic brush: no random rotation, no scale, conform on, align off, no offset."""
    values = {
        "rotation_x_min": 0.0,
        "rotation_x_max": 0.0,
        "rotation_y_min": 0.0,
        "rotation_y_max": 0.0,
        "rotation_z_min": 0.0,
        "rotation_z_max": 0.0,
        "scale_enabled": False,
        "conform_to_surface": True,
        "align_to_stroke": False,
        "vertical_offset": 0.0,
    }
    values.update(overrides)
    return ScatterBrushSettings(**values)


def _rotation_matrix(axis: Gf.Vec3d, degrees: float) -> Gf.Matrix4d:
    """Row-vector rotation matrix about an axis."""
    return Gf.Matrix4d().SetRotate(Gf.Rotation(axis, degrees))


def _author_local_transform(
    stage: Usd.Stage, prim_path: str, translate: Gf.Vec3d, rotate_xyz: Gf.Vec3f, scale: Gf.Vec3f
) -> Gf.Matrix4d:
    """Author translate/rotateXYZ/scale ops on a real Xform and return what UsdGeom computes from them."""
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xformable = UsdGeom.Xformable(xform.GetPrim())
    xformable.AddTranslateOp().Set(translate)
    xformable.AddRotateXYZOp().Set(rotate_xyz)
    xformable.AddScaleOp().Set(scale)
    return xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _identity_parent() -> Gf.Matrix4d:
    """Parent transform of a placement whose instance sits at the world origin."""
    return Gf.Matrix4d(1.0)


class TestSamplingFunctions(omni.kit.test.AsyncTestCase):
    def _assert_vec_close(self, actual, expected, tolerance: float = 1e-5):
        """Compare two 3-component vectors component-wise."""
        for index in range(3):
            self.assertAlmostEqual(
                float(actual[index]),
                float(expected[index]),
                delta=tolerance,
                msg=f"component {index}: {tuple(actual)} != {tuple(expected)}",
            )

    def _assert_matrix_close(self, actual: Gf.Matrix4d, expected: Gf.Matrix4d, tolerance: float = 1e-4):
        """Compare two 4x4 matrices element-wise."""
        for row in range(4):
            for column in range(4):
                self.assertAlmostEqual(
                    actual[row][column],
                    expected[row][column],
                    delta=tolerance,
                    msg=f"[{row}][{column}]:\nactual={actual}\nexpected={expected}",
                )

    async def test_stamp_rng_with_same_seed_stroke_and_stamp_returns_identical_draws(self):
        # Arrange
        first = sampling.stamp_rng(seed=1234, stroke_index=7, stamp_index=3)
        second = sampling.stamp_rng(seed=1234, stroke_index=7, stamp_index=3)

        # Act
        draws = (first.random(8), second.random(8))

        # Assert
        np.testing.assert_array_equal(draws[0], draws[1])

    async def test_stamp_rng_with_different_stamp_index_returns_different_draws(self):
        # Arrange
        first = sampling.stamp_rng(seed=1234, stroke_index=7, stamp_index=3)
        second = sampling.stamp_rng(seed=1234, stroke_index=7, stamp_index=4)

        # Act
        draws = (first.random(8), second.random(8))

        # Assert
        self.assertFalse(np.array_equal(draws[0], draws[1]))

    async def test_falloff_weight_for_each_kind_returns_expected_curve_value(self):
        cases = [
            (Falloff.CONSTANT, 0.0, 1.0),
            (Falloff.CONSTANT, 0.5, 1.0),
            (Falloff.CONSTANT, 1.0, 1.0),
            (Falloff.CONSTANT, 1.5, 0.0),
            (Falloff.LINEAR, 0.0, 1.0),
            (Falloff.LINEAR, 0.5, 0.5),
            (Falloff.LINEAR, 1.0, 0.0),
            (Falloff.LINEAR, 1.5, 0.0),
            (Falloff.SMOOTH, 0.0, 1.0),
            (Falloff.SMOOTH, 0.5, 0.5),
            (Falloff.SMOOTH, 1.0, 0.0),
            (Falloff.SMOOTH, 1.5, 0.0),
            (Falloff.SPHERE, 0.0, 1.0),
            (Falloff.SPHERE, 0.5, math.sqrt(0.75)),
            (Falloff.SPHERE, 1.0, 0.0),
            (Falloff.SPHERE, 1.5, 0.0),
            (Falloff.GAUSSIAN, 0.0, 1.0),
            (Falloff.GAUSSIAN, 0.5, math.exp(-0.5 * (0.5 / 0.4) ** 2)),
            (Falloff.GAUSSIAN, 1.0, math.exp(-0.5 * (1.0 / 0.4) ** 2)),
            (Falloff.GAUSSIAN, 1.5, 0.0),
        ]
        for kind, distance, expected in cases:
            with self.subTest(title=f"kind={kind.value} t={distance}"):
                # Arrange
                # (inputs come from the case tuple)

                # Act
                weight = sampling.falloff_weight(kind, distance)

                # Assert
                self.assertAlmostEqual(weight, expected, places=9)

    async def test_falloff_weight_with_negative_distance_returns_center_weight(self):
        # Arrange
        kind = Falloff.LINEAR

        # Act
        weight = sampling.falloff_weight(kind, -0.5)

        # Assert
        self.assertEqual(weight, 1.0)

    async def test_falloff_weight_with_unknown_kind_raises_value_error(self):
        # Arrange
        kind = "NOT_A_FALLOFF"

        # Act / Assert
        with self.assertRaises(ValueError):
            sampling.falloff_weight(kind, 0.5)

    async def test_sample_disk_returns_points_inside_unit_disk_with_uniform_area_density(self):
        # Arrange
        rng = np.random.default_rng(42)
        count = 20000

        # Act
        points = sampling.sample_disk(rng, count)

        # Assert
        radii_squared = np.sum(points * points, axis=1)
        self.assertEqual(points.shape, (count, 2))
        self.assertEqual(points.dtype, np.float64)
        self.assertLessEqual(float(radii_squared.max()), 1.0)
        self.assertAlmostEqual(float(radii_squared.mean()), 0.5, delta=0.01)
        self.assertAlmostEqual(float(points[:, 0].mean()), 0.0, delta=0.02)
        self.assertAlmostEqual(float(points[:, 1].mean()), 0.0, delta=0.02)

    async def test_sample_disk_with_zero_count_returns_empty_array(self):
        # Arrange
        rng = np.random.default_rng(1)

        # Act
        points = sampling.sample_disk(rng, 0)

        # Assert
        self.assertEqual(points.shape, (0, 2))

    async def test_tangent_basis_for_axis_aligned_and_arbitrary_normals_returns_orthonormal_frame(self):
        cases = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([0.3, -0.7, 0.2]),
            np.array([-2.0, 0.5, 4.0]),
        ]
        for normal in cases:
            with self.subTest(title=f"normal={normal.tolist()}"):
                # Arrange
                unit_normal = normal / np.linalg.norm(normal)

                # Act
                tangent, bitangent = sampling.tangent_basis(normal)

                # Assert
                self.assertAlmostEqual(float(np.linalg.norm(tangent)), 1.0, places=9)
                self.assertAlmostEqual(float(np.linalg.norm(bitangent)), 1.0, places=9)
                self.assertAlmostEqual(float(np.dot(tangent, unit_normal)), 0.0, places=9)
                self.assertAlmostEqual(float(np.dot(bitangent, unit_normal)), 0.0, places=9)
                self.assertAlmostEqual(float(np.dot(tangent, bitangent)), 0.0, places=9)
                np.testing.assert_allclose(np.cross(tangent, bitangent), unit_normal, atol=1e-9)

    async def test_tangent_basis_with_hint_aligns_tangent_with_projected_hint(self):
        # Arrange
        normal = np.array([0.0, 0.0, 1.0])
        hint = np.array([1.0, 1.0, 5.0])

        # Act
        tangent, bitangent = sampling.tangent_basis(normal, hint)

        # Assert
        np.testing.assert_allclose(tangent, [_SQRT_HALF, _SQRT_HALF, 0.0], atol=1e-9)
        np.testing.assert_allclose(bitangent, [-_SQRT_HALF, _SQRT_HALF, 0.0], atol=1e-9)

    async def test_tangent_basis_with_hint_parallel_to_normal_falls_back_to_orthonormal_frame(self):
        # Arrange
        normal = np.array([0.0, 0.0, 1.0])
        hint = np.array([0.0, 0.0, 3.0])

        # Act
        tangent, bitangent = sampling.tangent_basis(normal, hint)

        # Assert
        self.assertAlmostEqual(float(np.linalg.norm(tangent)), 1.0, places=9)
        self.assertAlmostEqual(float(np.dot(tangent, normal)), 0.0, places=9)
        np.testing.assert_allclose(np.cross(tangent, bitangent), normal, atol=1e-9)

    async def test_tangent_basis_with_zero_normal_raises_value_error(self):
        # Arrange
        normal = np.zeros(3)

        # Act / Assert
        with self.assertRaises(ValueError):
            sampling.tangent_basis(normal)

    async def test_choose_asset_with_weights_returns_entries_in_proportion_within_tolerance(self):
        # Arrange
        rng = np.random.default_rng(7)
        heavy = ScatterAssetEntry(path="heavy.usd", weight=3.0)
        light = ScatterAssetEntry(path="light.usd", weight=1.0)
        draw_count = 4000

        # Act
        chosen = [sampling.choose_asset(rng, [heavy, light]) for _ in range(draw_count)]

        # Assert
        heavy_fraction = sum(1 for entry in chosen if entry is heavy) / draw_count
        self.assertAlmostEqual(heavy_fraction, 0.75, delta=0.04)
        self.assertTrue(all(entry is heavy or entry is light for entry in chosen))

    async def test_choose_asset_skips_disabled_and_zero_weight_entries(self):
        # Arrange
        rng = np.random.default_rng(11)
        enabled = ScatterAssetEntry(path="enabled.usd", weight=1.0)
        disabled = ScatterAssetEntry(path="disabled.usd", weight=100.0, enabled=False)
        weightless = ScatterAssetEntry(path="weightless.usd", weight=0.0)

        # Act
        chosen = [sampling.choose_asset(rng, [disabled, enabled, weightless]) for _ in range(200)]

        # Assert
        self.assertTrue(all(entry is enabled for entry in chosen))

    async def test_choose_asset_with_nothing_enabled_returns_none(self):
        cases = [
            ("empty", []),
            ("all_disabled", [ScatterAssetEntry(path="a.usd", enabled=False)]),
            ("all_zero_weight", [ScatterAssetEntry(path="a.usd", weight=0.0)]),
        ]
        for title, assets in cases:
            with self.subTest(title=title):
                # Arrange
                rng = np.random.default_rng(0)

                # Act
                chosen = sampling.choose_asset(rng, assets)

                # Assert
                self.assertIsNone(chosen)

    async def test_sample_scale_when_scale_disabled_returns_unit_scale(self):
        # Arrange
        rng = np.random.default_rng(3)
        settings = _settings(scale_enabled=False, scale_min=5.0, scale_max=9.0)

        # Act
        scale = sampling.sample_scale(rng, settings)

        # Assert
        self.assertEqual(scale, (1.0, 1.0, 1.0))

    async def test_sample_scale_uniform_stays_within_range_and_is_equal_on_all_axes(self):
        # Arrange
        rng = np.random.default_rng(5)
        settings = _settings(scale_enabled=True, scale_uniform=True, scale_min=0.5, scale_max=2.0)

        # Act
        samples = [sampling.sample_scale(rng, settings) for _ in range(500)]

        # Assert
        for scale in samples:
            self.assertEqual(scale[0], scale[1])
            self.assertEqual(scale[1], scale[2])
            self.assertGreaterEqual(scale[0], 0.5)
            self.assertLessEqual(scale[0], 2.0)

    async def test_sample_scale_with_positive_bias_shifts_mean_toward_max(self):
        # Arrange
        rng = np.random.default_rng(9)
        settings = _settings(scale_enabled=True, scale_uniform=True, scale_min=0.8, scale_max=1.2, scale_bias=1.0)
        # bias 1 maps t -> sqrt(t): E[sqrt(u)] = 2/3, so the mean lands at 0.8 + 0.4 * 2/3
        expected_mean = 0.8 + 0.4 * (2.0 / 3.0)

        # Act
        samples = [sampling.sample_scale(rng, settings)[0] for _ in range(3000)]

        # Assert
        mean = sum(samples) / len(samples)
        self.assertGreater(mean, 1.0)
        self.assertAlmostEqual(mean, expected_mean, delta=0.01)

    async def test_sample_scale_with_negative_bias_shifts_mean_toward_min(self):
        # Arrange
        rng = np.random.default_rng(9)
        settings = _settings(scale_enabled=True, scale_uniform=True, scale_min=0.8, scale_max=1.2, scale_bias=-1.0)
        # bias -1 maps t -> t^2: E[u^2] = 1/3
        expected_mean = 0.8 + 0.4 * (1.0 / 3.0)

        # Act
        samples = [sampling.sample_scale(rng, settings)[0] for _ in range(3000)]

        # Assert
        mean = sum(samples) / len(samples)
        self.assertLess(mean, 1.0)
        self.assertAlmostEqual(mean, expected_mean, delta=0.01)

    async def test_sample_scale_with_large_weight_concentrates_near_midpoint(self):
        # Arrange
        rng = np.random.default_rng(13)
        settings = _settings(scale_enabled=True, scale_uniform=True, scale_min=0.8, scale_max=1.2, scale_weight=10.0)
        # weight 10 maps |2u-1| -> |2u-1|^10 whose mean is 1/11, so the mean deviation from 1.0 is 0.2 / 11
        expected_mean_deviation = 0.2 / 11.0

        # Act
        samples = [sampling.sample_scale(rng, settings)[0] for _ in range(3000)]

        # Assert
        mean_deviation = sum(abs(value - 1.0) for value in samples) / len(samples)
        self.assertAlmostEqual(mean_deviation, expected_mean_deviation, delta=0.01)
        self.assertLess(mean_deviation, 0.05)

    async def test_sample_scale_per_axis_uses_axis_ranges_with_independent_draws(self):
        # Arrange
        rng = np.random.default_rng(17)
        settings = _settings(
            scale_enabled=True,
            scale_uniform=False,
            scale_min=50.0,
            scale_max=60.0,
            scale_x_min=1.0,
            scale_x_max=2.0,
            scale_y_min=3.0,
            scale_y_max=4.0,
            scale_z_min=5.0,
            scale_z_max=6.0,
        )

        # Act
        samples = [sampling.sample_scale(rng, settings) for _ in range(200)]

        # Assert
        for scale_x, scale_y, scale_z in samples:
            self.assertTrue(1.0 <= scale_x <= 2.0, scale_x)
            self.assertTrue(3.0 <= scale_y <= 4.0, scale_y)
            self.assertTrue(5.0 <= scale_z <= 6.0, scale_z)
        normalized = [(x - 1.0, y - 3.0, z - 5.0) for x, y, z in samples]
        self.assertTrue(any(abs(x - y) > 1e-6 or abs(y - z) > 1e-6 for x, y, z in normalized))

    async def test_sample_rotation_degrees_returns_angles_within_configured_ranges(self):
        # Arrange
        rng = np.random.default_rng(19)
        settings = _settings(
            rotation_x_min=-10.0,
            rotation_x_max=10.0,
            rotation_y_min=20.0,
            rotation_y_max=30.0,
            rotation_z_min=0.0,
            rotation_z_max=360.0,
        )

        # Act
        samples = [sampling.sample_rotation_degrees(rng, settings) for _ in range(500)]

        # Assert
        for rotate_x, rotate_y, rotate_z in samples:
            self.assertTrue(-10.0 <= rotate_x <= 10.0, rotate_x)
            self.assertTrue(20.0 <= rotate_y <= 30.0, rotate_y)
            self.assertTrue(0.0 <= rotate_z <= 360.0, rotate_z)
        self.assertGreater(max(sample[2] for sample in samples) - min(sample[2] for sample in samples), 180.0)

    async def test_sample_rotation_degrees_with_fixed_ranges_returns_exact_values(self):
        # Arrange
        rng = np.random.default_rng(23)
        settings = _settings(
            rotation_x_min=5.0,
            rotation_x_max=5.0,
            rotation_y_min=-15.0,
            rotation_y_max=-15.0,
            rotation_z_min=90.0,
            rotation_z_max=90.0,
        )

        # Act
        rotation = sampling.sample_rotation_degrees(rng, settings)

        # Assert
        self.assertEqual(rotation, (5.0, -15.0, 90.0))

    async def test_up_axis_vector_returns_unit_axis_for_each_up_axis(self):
        cases = [(UpAxis.Y, Gf.Vec3d(0.0, 1.0, 0.0)), (UpAxis.Z, Gf.Vec3d(0.0, 0.0, 1.0))]
        for up_axis, expected in cases:
            with self.subTest(title=f"up_axis={up_axis.value}"):
                # Arrange
                # (inputs come from the case tuple)

                # Act
                vector = sampling.up_axis_vector(up_axis)

                # Assert
                self.assertEqual(vector, expected)

    async def test_stage_up_axis_reads_authored_stage_metadata(self):
        cases = [(UsdGeom.Tokens.y, UpAxis.Y), (UsdGeom.Tokens.z, UpAxis.Z)]
        for token, expected in cases:
            with self.subTest(title=f"token={token}"):
                # Arrange
                stage = Usd.Stage.CreateInMemory()
                UsdGeom.SetStageUpAxis(stage, token)

                # Act
                up_axis = sampling.stage_up_axis(stage)

                # Assert
                self.assertEqual(up_axis, expected)

    async def test_asset_up_correction_with_matching_axes_returns_identity(self):
        cases = [UpAxis.Y, UpAxis.Z]
        for up_axis in cases:
            with self.subTest(title=f"up_axis={up_axis.value}"):
                # Arrange
                probe = Gf.Vec3d(0.3, 0.4, 0.5)

                # Act
                correction = sampling.asset_up_correction(up_axis, up_axis)

                # Assert
                self.assertAlmostEqual(correction.GetAngle(), 0.0, places=9)
                self._assert_vec_close(correction.TransformDir(probe), probe)

    async def test_asset_up_correction_maps_y_up_asset_onto_z_up_stage(self):
        # Arrange
        asset_up_vector = Gf.Vec3d(0.0, 1.0, 0.0)

        # Act
        correction = sampling.asset_up_correction(UpAxis.Y, UpAxis.Z)

        # Assert
        self._assert_vec_close(correction.TransformDir(asset_up_vector), Gf.Vec3d(0.0, 0.0, 1.0))

    async def test_asset_up_correction_maps_z_up_asset_onto_y_up_stage(self):
        # Arrange
        asset_up_vector = Gf.Vec3d(0.0, 0.0, 1.0)

        # Act
        correction = sampling.asset_up_correction(UpAxis.Z, UpAxis.Y)

        # Assert
        self._assert_vec_close(correction.TransformDir(asset_up_vector), Gf.Vec3d(0.0, 1.0, 0.0))

    async def test_rotation_to_xyz_degrees_round_trips_through_authored_rotate_xyz_op(self):
        # Rotation about all three axes so that any wrong Euler order or sign shows up in the authored matrix
        rotation = (
            _rotation_matrix(Gf.Vec3d.XAxis(), 20.0)
            * _rotation_matrix(Gf.Vec3d.YAxis(), 35.0)
            * _rotation_matrix(Gf.Vec3d.ZAxis(), 50.0)
        )
        cases = [
            ("matrix4d", rotation),
            ("matrix4d_with_scale_and_translation", Gf.Matrix4d().SetScale(Gf.Vec3d(2.0, 3.0, 4.0)) * rotation),
            ("rotation", rotation.ExtractRotation()),
            ("matrix3d", rotation.ExtractRotationMatrix()),
        ]
        for title, value in cases:
            with self.subTest(title=title):
                # Arrange
                stage = Usd.Stage.CreateInMemory()

                # Act
                rotate_xyz = sampling.rotation_to_xyz_degrees(value)

                # Assert
                self.assertIsInstance(rotate_xyz, Gf.Vec3f)
                authored = _author_local_transform(
                    stage, "/Rotated", Gf.Vec3d(0.0, 0.0, 0.0), rotate_xyz, Gf.Vec3f(1.0, 1.0, 1.0)
                )
                self._assert_matrix_close(authored, rotation)

    async def test_rotation_to_xyz_degrees_with_unsupported_type_raises_type_error(self):
        # Arrange
        value = (20.0, 35.0, 50.0)

        # Act / Assert
        with self.assertRaises(TypeError):
            sampling.rotation_to_xyz_degrees(value)

    async def test_compose_parent_space_transform_round_trips_through_authored_xform_ops(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        settings = _settings(
            rotation_x_min=20.0,
            rotation_x_max=20.0,
            rotation_y_min=35.0,
            rotation_y_max=35.0,
            rotation_z_min=50.0,
            rotation_z_max=50.0,
            scale_enabled=True,
            scale_uniform=True,
            scale_min=1.5,
            scale_max=1.5,
            vertical_offset=2.5,
            conform_to_surface=True,
        )
        parent_world = _rotation_matrix(Gf.Vec3d.ZAxis(), 30.0) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(10.0, 20.0, 30.0))
        position_world = Gf.Vec3d(12.0, -4.0, 33.0)
        normal_world = Gf.Vec3d(1.0, 2.0, 3.0).GetNormalized()
        # The parent is rigid, so its inverse rotates directions and normals alike
        inverse_parent = parent_world.GetInverse()
        normal_parent = inverse_parent.TransformDir(normal_world).GetNormalized()
        expected_translate = inverse_parent.Transform(position_world) + normal_parent * 2.5
        expected_rotation = (
            _rotation_matrix(Gf.Vec3d.XAxis(), 90.0)  # Y-up asset onto the Z-up stage
            * _rotation_matrix(Gf.Vec3d.XAxis(), 20.0)
            * _rotation_matrix(Gf.Vec3d.YAxis(), 35.0)
            * _rotation_matrix(Gf.Vec3d.ZAxis(), 50.0)
            * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d.ZAxis(), normal_parent))
        )
        expected = (
            Gf.Matrix4d().SetScale(Gf.Vec3d(1.5, 1.5, 1.5))
            * expected_rotation
            * Gf.Matrix4d().SetTranslate(expected_translate)
        )

        # Act
        translate, rotate_xyz, scale = sampling.compose_parent_space_transform(
            position_world,
            normal_world,
            parent_world,
            settings,
            np.random.default_rng(3),
            UpAxis.Z,
            UpAxis.Y,
            None,
        )

        # Assert
        self.assertIsInstance(translate, Gf.Vec3d)
        self.assertIsInstance(rotate_xyz, Gf.Vec3f)
        self.assertIsInstance(scale, Gf.Vec3f)
        self._assert_vec_close(scale, Gf.Vec3f(1.5, 1.5, 1.5))
        self._assert_vec_close(translate, expected_translate)
        authored = _author_local_transform(stage, "/Placement", translate, rotate_xyz, scale)
        self._assert_matrix_close(authored, expected, 1e-4)

    async def test_compose_parent_space_transform_with_conform_aligns_stage_up_axis_with_normal(self):
        cases = [
            Gf.Vec3d(1.0, 1.0, 1.0).GetNormalized(),
            Gf.Vec3d(1.0, 0.0, 0.0),
            Gf.Vec3d(0.0, 0.0, 1.0),  # already up: identity rotation
            Gf.Vec3d(0.0, 0.0, -1.0),  # anti-parallel: any 180 degree flip is valid
            Gf.Vec3d(-0.2, 0.9, 0.3).GetNormalized(),
        ]
        for normal in cases:
            with self.subTest(title=f"normal={tuple(normal)}"):
                # Arrange
                stage = Usd.Stage.CreateInMemory()
                settings = _settings(conform_to_surface=True)

                # Act
                translate, rotate_xyz, scale = sampling.compose_parent_space_transform(
                    Gf.Vec3d(1.0, 2.0, 3.0),
                    normal,
                    _identity_parent(),
                    settings,
                    np.random.default_rng(0),
                    UpAxis.Z,
                    UpAxis.Z,
                    None,
                )

                # Assert
                authored = _author_local_transform(stage, "/Placement", translate, rotate_xyz, scale)
                self._assert_vec_close(authored.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)), normal, 1e-4)
                self._assert_vec_close(translate, Gf.Vec3d(1.0, 2.0, 3.0))

    async def test_compose_parent_space_transform_on_y_up_stage_aligns_y_axis_with_normal(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        settings = _settings(conform_to_surface=True)
        normal = Gf.Vec3d(1.0, 1.0, 1.0).GetNormalized()

        # Act
        translate, rotate_xyz, scale = sampling.compose_parent_space_transform(
            Gf.Vec3d(0.0, 0.0, 0.0),
            normal,
            _identity_parent(),
            settings,
            np.random.default_rng(0),
            UpAxis.Y,
            UpAxis.Y,
            None,
        )

        # Assert
        authored = _author_local_transform(stage, "/Placement", translate, rotate_xyz, scale)
        self._assert_vec_close(authored.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0)), normal, 1e-4)

    async def test_compose_parent_space_transform_without_conform_keeps_stage_up_axis(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        settings = _settings(conform_to_surface=False)
        normal = Gf.Vec3d(1.0, 1.0, 1.0).GetNormalized()

        # Act
        translate, rotate_xyz, scale = sampling.compose_parent_space_transform(
            Gf.Vec3d(0.0, 0.0, 0.0),
            normal,
            _identity_parent(),
            settings,
            np.random.default_rng(0),
            UpAxis.Z,
            UpAxis.Z,
            None,
        )

        # Assert
        authored = _author_local_transform(stage, "/Placement", translate, rotate_xyz, scale)
        self._assert_matrix_close(authored, Gf.Matrix4d(1.0), 1e-5)

    async def test_compose_parent_space_transform_with_align_to_stroke_yaws_x_axis_onto_heading(self):
        cases = [
            ("z_up_flat_heading_y", UpAxis.Z, False, Gf.Vec3d(0.0, 0.0, 1.0), Gf.Vec3d(0.0, 1.0, 0.0)),
            ("z_up_flat_heading_neg_x", UpAxis.Z, False, Gf.Vec3d(0.0, 0.0, 1.0), Gf.Vec3d(-1.0, 0.0, 0.0)),
            ("z_up_flat_heading_diagonal", UpAxis.Z, False, Gf.Vec3d(0.0, 0.0, 1.0), Gf.Vec3d(1.0, -1.0, 0.0)),
            ("y_up_flat_heading_z", UpAxis.Y, False, Gf.Vec3d(0.0, 1.0, 0.0), Gf.Vec3d(0.0, 0.0, 1.0)),
            ("z_up_conform_tilted_x", UpAxis.Z, True, Gf.Vec3d(0.0, 1.0, 1.0).GetNormalized(), Gf.Vec3d(1.0, 0.0, 0.0)),
            ("z_up_conform_tilted_y", UpAxis.Z, True, Gf.Vec3d(1.0, 0.0, 1.0).GetNormalized(), Gf.Vec3d(0.0, 1.0, 0.0)),
        ]
        for title, stage_up, conform, normal, heading in cases:
            with self.subTest(title=title):
                # Arrange
                stage = Usd.Stage.CreateInMemory()
                settings = _settings(conform_to_surface=conform, align_to_stroke=True)
                # Heading is not always tangent to the surface: the asset should face its projection on the surface
                projected = heading - normal * Gf.Dot(heading, normal)
                expected_forward = projected.GetNormalized()

                # Act
                translate, rotate_xyz, scale = sampling.compose_parent_space_transform(
                    Gf.Vec3d(0.0, 0.0, 0.0),
                    normal,
                    _identity_parent(),
                    settings,
                    np.random.default_rng(0),
                    stage_up,
                    stage_up,
                    heading,
                )

                # Assert
                authored = _author_local_transform(stage, "/Placement", translate, rotate_xyz, scale)
                self._assert_vec_close(authored.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)), expected_forward, 1e-4)

    async def test_compose_parent_space_transform_with_align_to_stroke_transforms_heading_into_parent_space(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        settings = _settings(conform_to_surface=False, align_to_stroke=True)
        # Parent yawed by 90 degrees: a world +Y heading is the parent's +X, so no yaw is needed in parent space
        parent_world = _rotation_matrix(Gf.Vec3d.ZAxis(), 90.0)

        # Act
        translate, rotate_xyz, scale = sampling.compose_parent_space_transform(
            Gf.Vec3d(0.0, 0.0, 0.0),
            Gf.Vec3d(0.0, 0.0, 1.0),
            parent_world,
            settings,
            np.random.default_rng(0),
            UpAxis.Z,
            UpAxis.Z,
            Gf.Vec3d(0.0, 1.0, 0.0),
        )

        # Assert
        authored = _author_local_transform(stage, "/Placement", translate, rotate_xyz, scale)
        self._assert_vec_close(authored.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)), Gf.Vec3d(1.0, 0.0, 0.0), 1e-4)

    async def test_compose_parent_space_transform_ignores_heading_when_align_to_stroke_is_off_or_degenerate(self):
        cases = [
            ("align_off", False, Gf.Vec3d(0.0, 1.0, 0.0)),
            ("no_heading", True, None),
            ("zero_heading", True, Gf.Vec3d(0.0, 0.0, 0.0)),
            ("heading_along_up", True, Gf.Vec3d(0.0, 0.0, 5.0)),
        ]
        for title, align, heading in cases:
            with self.subTest(title=title):
                # Arrange
                stage = Usd.Stage.CreateInMemory()
                settings = _settings(conform_to_surface=False, align_to_stroke=align)

                # Act
                translate, rotate_xyz, scale = sampling.compose_parent_space_transform(
                    Gf.Vec3d(0.0, 0.0, 0.0),
                    Gf.Vec3d(0.0, 0.0, 1.0),
                    _identity_parent(),
                    settings,
                    np.random.default_rng(0),
                    UpAxis.Z,
                    UpAxis.Z,
                    heading,
                )

                # Assert
                authored = _author_local_transform(stage, "/Placement", translate, rotate_xyz, scale)
                self._assert_matrix_close(authored, Gf.Matrix4d(1.0), 1e-5)

    async def test_compose_parent_space_transform_applies_vertical_offset_along_effective_up(self):
        cases = [
            ("conform_moves_along_normal", True, UpAxis.Z, Gf.Vec3d(1.0, 0.0, 0.0), Gf.Vec3d(4.0, 2.0, 3.0)),
            ("z_up_moves_along_z", False, UpAxis.Z, Gf.Vec3d(1.0, 0.0, 0.0), Gf.Vec3d(1.0, 2.0, 6.0)),
            ("y_up_moves_along_y", False, UpAxis.Y, Gf.Vec3d(1.0, 0.0, 0.0), Gf.Vec3d(1.0, 5.0, 3.0)),
        ]
        for title, conform, stage_up, normal, expected_translate in cases:
            with self.subTest(title=title):
                # Arrange
                settings = _settings(conform_to_surface=conform, vertical_offset=3.0)

                # Act
                translate, _rotate_xyz, _scale = sampling.compose_parent_space_transform(
                    Gf.Vec3d(1.0, 2.0, 3.0),
                    normal,
                    _identity_parent(),
                    settings,
                    np.random.default_rng(0),
                    stage_up,
                    stage_up,
                    None,
                )

                # Assert
                self._assert_vec_close(translate, expected_translate)

    async def test_compose_parent_space_transform_with_transformed_parent_composes_back_to_world_position(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        settings = _settings(
            conform_to_surface=True,
            rotation_z_min=45.0,
            rotation_z_max=45.0,
            scale_enabled=True,
            scale_uniform=False,
            scale_x_min=2.0,
            scale_x_max=2.0,
            scale_y_min=0.5,
            scale_y_max=0.5,
            scale_z_min=1.0,
            scale_z_max=1.0,
        )
        parent_world = (
            Gf.Matrix4d().SetScale(Gf.Vec3d(2.0, 2.0, 2.0))
            * _rotation_matrix(Gf.Vec3d(1.0, 1.0, 0.0).GetNormalized(), 40.0)
            * Gf.Matrix4d().SetTranslate(Gf.Vec3d(-100.0, 50.0, 25.0))
        )
        position_world = Gf.Vec3d(-80.0, 60.0, 40.0)
        normal_world = Gf.Vec3d(0.0, 1.0, 2.0).GetNormalized()

        # Act
        translate, rotate_xyz, scale = sampling.compose_parent_space_transform(
            position_world,
            normal_world,
            parent_world,
            settings,
            np.random.default_rng(0),
            UpAxis.Z,
            UpAxis.Z,
            None,
        )

        # Assert
        self._assert_vec_close(scale, Gf.Vec3f(2.0, 0.5, 1.0))
        authored = _author_local_transform(stage, "/Placement", translate, rotate_xyz, scale)
        world = authored * parent_world
        self._assert_vec_close(world.Transform(Gf.Vec3d(0.0, 0.0, 0.0)), position_world, 1e-4)

    async def test_compose_parent_space_transform_with_scaled_parent_uses_inverse_transpose_for_normal(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        settings = _settings(conform_to_surface=True)
        # Parent stretches X by 2. The parent-space plane with normal (1,1,0) has the world normal (1,2,0):
        # a wrong direction transform would produce (2,1,0) instead.
        parent_world = Gf.Matrix4d().SetScale(Gf.Vec3d(2.0, 1.0, 1.0))
        normal_world = Gf.Vec3d(1.0, 2.0, 0.0).GetNormalized()
        expected_normal_parent = Gf.Vec3d(_SQRT_HALF, _SQRT_HALF, 0.0)

        # Act
        translate, rotate_xyz, scale = sampling.compose_parent_space_transform(
            Gf.Vec3d(0.0, 0.0, 0.0),
            normal_world,
            parent_world,
            settings,
            np.random.default_rng(0),
            UpAxis.Z,
            UpAxis.Z,
            None,
        )

        # Assert
        authored = _author_local_transform(stage, "/Placement", translate, rotate_xyz, scale)
        self._assert_vec_close(authored.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)), expected_normal_parent, 1e-4)

    async def test_compose_parent_space_transform_accepts_numpy_vectors(self):
        # Arrange
        settings = _settings(conform_to_surface=False, vertical_offset=1.0)

        # Act
        translate, rotate_xyz, scale = sampling.compose_parent_space_transform(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 1.0]),
            _identity_parent(),
            settings,
            np.random.default_rng(0),
            UpAxis.Z,
            UpAxis.Z,
            np.array([0.0, 1.0, 0.0]),
        )

        # Assert
        self._assert_vec_close(translate, Gf.Vec3d(1.0, 2.0, 4.0))
        self._assert_vec_close(rotate_xyz, Gf.Vec3f(0.0, 0.0, 0.0))
        self._assert_vec_close(scale, Gf.Vec3f(1.0, 1.0, 1.0))

    async def test_compose_parent_space_transform_with_zero_normal_raises_value_error(self):
        # Arrange
        settings = _settings()

        # Act / Assert
        with self.assertRaises(ValueError):
            sampling.compose_parent_space_transform(
                Gf.Vec3d(0.0, 0.0, 0.0),
                Gf.Vec3d(0.0, 0.0, 0.0),
                _identity_parent(),
                settings,
                np.random.default_rng(0),
                UpAxis.Z,
                UpAxis.Z,
                None,
            )


class TestPaddingIndex(omni.kit.test.AsyncTestCase):
    async def test_is_free_with_point_within_min_distance_across_cell_boundary_returns_false(self):
        # Arrange
        index = sampling.PaddingIndex(cell_size=1.0)
        index.add(np.array([0.9, 0.9, 0.9]))

        # Act
        free = index.is_free(np.array([1.1, 1.1, 1.1]), min_distance=1.0)

        # Assert
        self.assertFalse(free)

    async def test_is_free_with_point_farther_than_min_distance_returns_true(self):
        # Arrange
        index = sampling.PaddingIndex(cell_size=1.0)
        index.add(np.array([0.0, 0.0, 0.0]))

        # Act
        free = index.is_free(np.array([3.0, 0.0, 0.0]), min_distance=1.0)

        # Assert
        self.assertTrue(free)

    async def test_is_free_with_point_in_same_cell_but_farther_than_min_distance_returns_true(self):
        # Arrange
        index = sampling.PaddingIndex(cell_size=10.0)
        index.add(np.array([0.5, 0.5, 0.5]))

        # Act
        free = index.is_free(np.array([9.5, 0.5, 0.5]), min_distance=1.0)

        # Assert
        self.assertTrue(free)

    async def test_is_free_with_min_distance_larger_than_cell_size_checks_far_cells(self):
        # Arrange
        index = sampling.PaddingIndex(cell_size=1.0)
        index.add(np.array([0.0, 0.0, 0.0]))

        # Act
        free = index.is_free(np.array([4.5, 0.0, 0.0]), min_distance=5.0)

        # Assert
        self.assertFalse(free)

    async def test_is_free_with_zero_min_distance_returns_true(self):
        # Arrange
        index = sampling.PaddingIndex(cell_size=1.0)
        index.add(np.array([0.0, 0.0, 0.0]))

        # Act
        free = index.is_free(np.array([0.0, 0.0, 0.0]), min_distance=0.0)

        # Assert
        self.assertTrue(free)

    async def test_is_free_on_empty_index_returns_true(self):
        # Arrange
        index = sampling.PaddingIndex(cell_size=1.0)

        # Act
        free = index.is_free(np.array([0.0, 0.0, 0.0]), min_distance=100.0)

        # Assert
        self.assertTrue(free)

    async def test_add_many_stores_every_point(self):
        # Arrange
        index = sampling.PaddingIndex(cell_size=2.0)
        points = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [5.0, 5.0, 5.0]])

        # Act
        index.add_many(points)

        # Assert
        self.assertEqual(len(index), 5)
        self.assertFalse(index.is_free(np.array([5.2, 5.2, 5.2]), min_distance=1.0))
        self.assertTrue(index.is_free(np.array([2.5, 2.5, 2.5]), min_distance=1.0))

    async def test_is_free_with_many_occupied_cells_finds_neighbours_and_gaps(self):
        # Arrange: a 10x10x10 lattice spaced 3 apart in unit cells so that the neighbourhood lookup is smaller than
        # the number of occupied cells and the grid walk (not the brute-force fallback) answers the query.
        index = sampling.PaddingIndex(cell_size=1.0)
        axis = np.arange(10, dtype=np.float64) * 3.0
        lattice = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
        index.add_many(lattice)

        # Act
        results = (
            index.is_free(np.array([12.4, 15.0, 18.0]), min_distance=1.0),
            index.is_free(np.array([13.5, 16.5, 19.5]), min_distance=1.0),
        )

        # Assert
        self.assertEqual(len(index), 1000)
        self.assertEqual(results, (False, True))

    async def test_add_many_with_empty_array_keeps_index_empty(self):
        # Arrange
        index = sampling.PaddingIndex(cell_size=2.0)

        # Act
        index.add_many(np.empty((0, 3)))

        # Assert
        self.assertEqual(len(index), 0)

    async def test_add_with_gf_vec3d_stores_point(self):
        # Arrange
        index = sampling.PaddingIndex(cell_size=1.0)

        # Act
        index.add(Gf.Vec3d(1.0, 2.0, 3.0))

        # Assert
        self.assertEqual(len(index), 1)
        self.assertFalse(index.is_free(Gf.Vec3d(1.0, 2.0, 3.5), min_distance=1.0))

    async def test_init_with_non_positive_cell_size_clamps_to_minimum_and_still_indexes(self):
        # Arrange
        index = sampling.PaddingIndex(cell_size=0.0)
        index.add(np.array([0.0, 0.0, 0.0]))

        # Act
        free = index.is_free(np.array([0.0, 0.0, 0.0005]), min_distance=0.001)

        # Assert
        self.assertFalse(free)
