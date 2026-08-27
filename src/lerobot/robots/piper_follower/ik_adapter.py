# Copyright 2026 jokeru8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pluggable EEF/FK/IK interfaces for Piper deployment.

The interface is usable before a Piper URDF/MJCF is selected. Joint-space
deployment remains available; EEF execution fails closed with a clear error
until an adapter is installed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PiperIKAdapter(Protocol):
    """EEF pose uses ``xyz(m) + quaternion(xyzw)``; joints use radians."""

    @property
    def available(self) -> bool: ...

    def solve(self, eef_pose7: np.ndarray, current_joints_rad: np.ndarray) -> np.ndarray | None: ...

    def forward(self, joints_rad: np.ndarray) -> np.ndarray | None: ...


class UnconfiguredPiperIKAdapter:
    """Placeholder used until a real robot model is connected."""

    @property
    def available(self) -> bool:
        return False

    def solve(self, eef_pose7: np.ndarray, current_joints_rad: np.ndarray) -> np.ndarray | None:
        del eef_pose7, current_joints_rad
        raise RuntimeError(
            "Piper EEF control requires an IK adapter. Call robot.set_ik_adapter(...) "
            "after providing a verified Piper URDF/MJCF solver."
        )

    def forward(self, joints_rad: np.ndarray) -> np.ndarray | None:
        del joints_rad
        return None


class CallablePiperIKAdapter:
    """Small adapter for wiring a future solver without coupling deployment code to it."""

    def __init__(
        self,
        solve_fn: Callable[[np.ndarray, np.ndarray], np.ndarray | None],
        *,
        forward_fn: Callable[[np.ndarray], np.ndarray | None] | None = None,
    ) -> None:
        self._solve_fn = solve_fn
        self._forward_fn = forward_fn

    @property
    def available(self) -> bool:
        return True

    def solve(self, eef_pose7: np.ndarray, current_joints_rad: np.ndarray) -> np.ndarray | None:
        result = self._solve_fn(
            np.asarray(eef_pose7, dtype=np.float64),
            np.asarray(current_joints_rad, dtype=np.float64),
        )
        if result is None:
            return None
        joints = np.asarray(result, dtype=np.float64).reshape(-1)
        if joints.shape != (6,):
            raise ValueError(f"IK solve must return 6 joints, got {joints.shape}")
        return joints

    def forward(self, joints_rad: np.ndarray) -> np.ndarray | None:
        if self._forward_fn is None:
            return None
        pose = self._forward_fn(np.asarray(joints_rad, dtype=np.float64))
        if pose is None:
            return None
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.shape != (7,):
            raise ValueError(f"FK must return xyz+quat xyzw (7,), got {pose.shape}")
        return pose
