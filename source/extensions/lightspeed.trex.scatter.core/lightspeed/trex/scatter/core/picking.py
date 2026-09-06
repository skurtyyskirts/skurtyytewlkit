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

__all__ = [
    "CpuRaySurfacePicker",
    "HdRemixSurfacePicker",
    "SurfaceHit",
    "SurfacePicker",
    "camera_ray_from_ndc",
    "create_surface_picker",
]

import asyncio
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import carb
import carb.settings
import numpy as np
from lightspeed.common.constants import ROOTNODE_INSTANCES
from lightspeed.hydra.remix.core import (
    RemixRequestQueryType,
    RemixSupport,
    is_remix_supported,
    viewport_api_request_query_hdremix,
)
from pxr import Gf, Sdf, Usd, UsdGeom

from .constants import DEFAULT_MAX_PICK_DISTANCE, FORCE_CPU_PICKER_SETTING, MAX_PICK_DISTANCE_SETTING

if TYPE_CHECKING:
    from omni.kit.widget.viewport.api import ViewportAPI

    from .geometry import MeshSurfaceCache

_LOG_PREFIX = "[lightspeed.trex.scatter.core]"
# Padding applied to cached mesh bounds before the ray test so flat meshes (zero-thickness bounds) are not culled by
# floating point noise.
_BBOX_CULL_EPSILON = 1.0e-4
# HdRemix answers a pick within a few frames. Its completion callbacks share one global request id, so another
# consumer picking at the same time can overwrite ours and the answer never arrives; a request older than this is
# treated as lost so the brush does not stay blocked forever.
_STALE_PICK_TIMEOUT = 2.0


@dataclass
class SurfaceHit:
    """Result of a surface pick.

    Attributes:
        path: Path of the picked prim (for HdRemix, the path the renderer reports; for the CPU picker, the mesh prim).
        world_position: Picked point in world space.
    """

    path: Sdf.Path
    world_position: Gf.Vec3d


SurfaceHitCallback = Callable[[SurfaceHit | None], None]


@runtime_checkable
class SurfacePicker(Protocol):
    """Asynchronous "what is under this viewport point" query.

    Implementations deliver exactly one callback per issued pick, with a `SurfaceHit` or None when nothing usable
    is under the point.
    """

    def pick(self, ndc: tuple[float, float], callback: SurfaceHitCallback) -> bool:
        """Request the surface under a viewport point.

        Args:
            ndc: Normalized device coordinates in [-1, 1] x [-1, 1] (y up).
            callback: Called once with the hit, or None when nothing was hit.

        Returns:
            False when the request was not issued (a request is already in flight or the coordinates are invalid);
            the callback is not called in that case.
        """
        ...

    def cancel(self) -> None:
        """Drop any pending request so its callback is never delivered."""
        ...

    @property
    def in_flight(self) -> bool:
        """Whether a request is pending."""
        ...


def camera_ray_from_ndc(viewport_api: ViewportAPI, ndc: tuple[float, float]) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    """Build the world-space camera ray through a viewport point.

    The near and far clip planes are unprojected with the viewport's `ndc_to_world` matrix; the ray starts on the
    near plane and points towards the far plane.

    Args:
        viewport_api: Viewport whose camera defines the ray.
        ndc: Normalized device coordinates in [-1, 1] x [-1, 1].

    Returns:
        (origin, unit direction) in world space.
    """
    x, y = float(ndc[0]), float(ndc[1])
    ndc_to_world = viewport_api.ndc_to_world
    near = ndc_to_world.Transform(Gf.Vec3d(x, y, -1.0))
    far = ndc_to_world.Transform(Gf.Vec3d(x, y, 1.0))
    return near, (far - near).GetNormalized()


def _is_valid_ndc(ndc: Sequence[float]) -> bool:
    """Return whether ``ndc`` is a finite coordinate pair inside the [-1, 1] viewport square."""
    if len(ndc) != 2:
        return False
    return all(math.isfinite(component) and -1.0 <= component <= 1.0 for component in ndc)


def _max_pick_distance_from_settings() -> float:
    """Return the maximum camera-to-hit distance from the shared teleport carb setting, or the default."""
    value = carb.settings.get_settings().get(MAX_PICK_DISTANCE_SETTING)
    return float(value) if value else DEFAULT_MAX_PICK_DISTANCE


def _ray_intersects_aabb(origin: np.ndarray, direction: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> bool:
    """Slab test between a ray and an axis-aligned box, treating the box boundary as inside."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse_direction = 1.0 / direction
        t_lower = (bbox_min - _BBOX_CULL_EPSILON - origin) * inverse_direction
        t_upper = (bbox_max + _BBOX_CULL_EPSILON - origin) * inverse_direction
    # 0 * inf produces NaN when the origin lies exactly on a slab plane of an axis the ray is parallel to.
    t_near = np.where(np.isnan(t_lower) | np.isnan(t_upper), -np.inf, np.minimum(t_lower, t_upper))
    t_far = np.where(np.isnan(t_lower) | np.isnan(t_upper), np.inf, np.maximum(t_lower, t_upper))
    t_enter = max(float(t_near.max()), 0.0)
    t_exit = float(t_far.min())
    return t_exit >= t_enter


def _current_event_loop() -> asyncio.AbstractEventLoop | None:
    """Return the event loop of the calling (main) thread, which is where pick results must be delivered."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        pass
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        return None


class HdRemixSurfacePicker:
    """Pick through the HdRemix renderer (prim path and world position of the pixel under the cursor).

    HdRemix allows a single query at a time, so at most one request is in flight; `pick` returns False while a
    request is pending. HdRemix answers from its own worker thread, so the result is copied there and delivered to
    the caller on the event loop that issued the pick.
    """

    def __init__(self, viewport_api: ViewportAPI, max_distance_getter: Callable[[], float] | None = None):
        """
        Args:
            viewport_api: Viewport whose texture is queried.
            max_distance_getter: Returns the maximum accepted distance between the camera and the hit; hits farther
                away (typically sky domes) are reported as None. Defaults to the shared teleport carb setting.
        """
        self._viewport_api = viewport_api
        self._max_distance_getter = max_distance_getter or _max_pick_distance_from_settings
        self._token = 0
        self._in_flight = False
        self._issued_at = 0.0

    @property
    def in_flight(self) -> bool:
        """Whether an HdRemix query was issued and its result has not been delivered or dropped yet."""
        return self._in_flight

    def pick(self, ndc: tuple[float, float], callback: SurfaceHitCallback) -> bool:
        """Query HdRemix for the prim and world position under ``ndc``; the result reaches ``callback`` later.

        Returns False while a request younger than ``_STALE_PICK_TIMEOUT`` is pending, when ``ndc`` is invalid or maps
        outside the render texture, or when HdRemix rejects the request. An older pending request is dropped first.
        """
        if self._in_flight:
            if time.monotonic() - self._issued_at < _STALE_PICK_TIMEOUT:
                return False
            carb.log_warn(
                f"{_LOG_PREFIX} HdRemix pick did not complete within {_STALE_PICK_TIMEOUT:.0f} s; dropping it"
            )
            self.cancel()
        if not _is_valid_ndc(ndc):
            return False
        pixel, valid = self._viewport_api.map_ndc_to_texture_pixel(ndc)
        if not valid:
            return False

        loop = _current_event_loop()
        if loop is None:
            carb.log_warn(f"{_LOG_PREFIX} HdRemix pick needs an event loop to deliver its result; none is available")
            return False

        self._token += 1
        token = self._token
        self._in_flight = True
        self._issued_at = time.monotonic()

        def on_query_complete(path: str, world_position: carb.Double3 | None, _pixel: carb.Uint2) -> None:
            # HdRemix invokes this from its own worker thread: only plain values are read here, and the hop onto the
            # issuing event loop is the one asyncio call that is safe from a foreign thread.
            position = None if world_position is None else tuple(float(component) for component in world_position)
            try:
                loop.call_soon_threadsafe(self._deliver, token, path, position, callback)
            except RuntimeError as exc:
                carb.log_error(f"{_LOG_PREFIX} Cannot deliver the HdRemix pick result on the event loop: {exc}")
                if token == self._token:
                    self._in_flight = False

        try:
            viewport_api_request_query_hdremix(
                carb.Uint2(int(pixel[0]), int(pixel[1])),
                callback=on_query_complete,
                request_query_type=RemixRequestQueryType.PATH_AND_WORLDPOS,
            )
        except RuntimeError as exc:
            carb.log_warn(f"{_LOG_PREFIX} HdRemix pick request failed: {exc}")
            self._in_flight = False
            return False
        return True

    def cancel(self) -> None:
        """Invalidate the pending request so its HdRemix callback is ignored, and accept new picks."""
        self._token += 1
        self._in_flight = False

    def _deliver(
        self, token: int, path: str, world_position: Sequence[float] | None, callback: SurfaceHitCallback
    ) -> None:
        """Run the user callback on the event loop for the request identified by ``token`` unless it went stale."""
        if token != self._token:
            return
        self._in_flight = False
        try:
            callback(self._to_hit(path, world_position))
        except Exception as exc:  # noqa: BLE001 - the pick callback is arbitrary caller code running on the loop.
            carb.log_error(f"{_LOG_PREFIX} Surface pick callback failed: {exc}")

    def _to_hit(self, path: str, world_position: Sequence[float] | None) -> SurfaceHit | None:
        """Turn an HdRemix answer into a hit, or None for a miss or a hit farther than the maximum pick distance."""
        if not path or world_position is None:
            return None
        position = Gf.Vec3d(float(world_position[0]), float(world_position[1]), float(world_position[2]))
        camera_position = self._viewport_api.transform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        max_distance = self._max_distance_getter()
        distance = (position - camera_position).GetLength()
        if distance > max_distance:
            carb.log_verbose(f"{_LOG_PREFIX} Ignoring pick {distance:.1f} from the camera (> {max_distance}): {path}")
            return None
        return SurfaceHit(Sdf.Path(path), position)


class CpuRaySurfacePicker:
    """Synchronous camera-ray picker against cached mesh triangles.

    Used when HdRemix is unavailable (tests, unsupported hardware). The callback is invoked before `pick` returns
    and `in_flight` is always False.
    """

    def __init__(
        self, viewport_api: ViewportAPI, cache: MeshSurfaceCache, root_paths: Sequence[str] = (ROOTNODE_INSTANCES,)
    ):
        """
        Args:
            viewport_api: Viewport providing the camera matrices and the stage.
            cache: Mesh geometry cache answering the ray queries.
            root_paths: Prim paths under which meshes are searched; the whole stage is searched when none exist.
        """
        self._viewport_api = viewport_api
        self._cache = cache
        self._root_paths = tuple(Sdf.Path(path) for path in root_paths)

    @property
    def in_flight(self) -> bool:
        """Always False: CPU picks complete before ``pick`` returns."""
        return False

    def pick(self, ndc: tuple[float, float], callback: SurfaceHitCallback) -> bool:
        """Cast the camera ray through ``ndc`` against the cached meshes and call ``callback`` with the nearest hit.

        Returns False without calling back when ``ndc`` is invalid or the viewport has no stage.
        """
        if not _is_valid_ndc(ndc):
            return False
        stage = self._viewport_api.stage
        if stage is None:
            carb.log_warn(f"{_LOG_PREFIX} Cannot pick: the viewport has no stage")
            return False
        origin, direction = camera_ray_from_ndc(self._viewport_api, ndc)
        callback(self._nearest_hit(stage, origin, direction))
        return True

    def cancel(self) -> None:
        """Nothing to cancel: picks complete synchronously."""

    def _nearest_hit(self, stage: Usd.Stage, origin: Gf.Vec3d, direction: Gf.Vec3d) -> SurfaceHit | None:
        """Return the closest ray hit over the visible meshes, culling by cached bounds first; None on a miss."""
        origin_array = np.array(origin, dtype=np.float64)
        direction_array = np.array(direction, dtype=np.float64)
        best_distance = math.inf
        best_hit: SurfaceHit | None = None
        for mesh_path in self._iter_visible_mesh_paths(stage):
            geometry = self._cache.get(mesh_path)
            if geometry is None:
                continue
            if not _ray_intersects_aabb(origin_array, direction_array, geometry.bbox_min, geometry.bbox_max):
                continue
            sample = self._cache.raycast(mesh_path, origin, direction)
            if sample is None:
                continue
            position = np.asarray(sample.position, dtype=np.float64)
            distance = float(np.linalg.norm(position - origin_array))
            if distance < best_distance:
                best_distance = distance
                best_hit = SurfaceHit(mesh_path, Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        return best_hit

    def _iter_visible_mesh_paths(self, stage: Usd.Stage) -> Iterator[Sdf.Path]:
        """Yield the visible mesh prim paths under the configured roots, or under the whole stage when none exist."""
        roots = [stage.GetPrimAtPath(path) for path in self._root_paths]
        roots = [root for root in roots if root.IsValid()]
        if not roots:
            roots = [stage.GetPseudoRoot()]
        predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
        for root in roots:
            for prim in Usd.PrimRange(root, predicate):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
                    continue
                yield prim.GetPath()


def create_surface_picker(
    viewport_api: ViewportAPI,
    cache: MeshSurfaceCache,
    picker_factory: Callable[[ViewportAPI, MeshSurfaceCache], SurfacePicker] | None = None,
) -> SurfacePicker:
    """Create the surface picker appropriate for the current renderer.

    Args:
        viewport_api: Viewport the picker queries.
        cache: Mesh geometry cache used by the CPU fallback.
        picker_factory: Overrides the choice entirely when given (tests, controller injection).

    Returns:
        The factory result when given; the CPU picker when the `forceCpuPicker` setting is on or HdRemix is not
        supported; the HdRemix picker otherwise.
    """
    if picker_factory is not None:
        return picker_factory(viewport_api, cache)
    force_cpu = bool(carb.settings.get_settings().get(FORCE_CPU_PICKER_SETTING))
    if force_cpu or is_remix_supported()[0] != RemixSupport.SUPPORTED:
        return CpuRaySurfacePicker(viewport_api, cache)
    return HdRemixSurfacePicker(viewport_api)
