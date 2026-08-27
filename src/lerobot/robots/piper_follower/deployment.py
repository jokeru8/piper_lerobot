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

"""Peripheral real-robot deployment layer for Piper.

The CAN/SDK driver remains the sole joint command channel. This module owns
policy-space processing, optional EEF-to-joint conversion and robot lifecycle.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .async_buffer import StreamActionBuffer
from .gripper_filter import GripperHysteresisWindow, PiperGripperFilterConfig
from .ik_adapter import PiperIKAdapter, UnconfiguredPiperIKAdapter
from .trajectory import (
    EEF_ACTION_DIM,
    JOINT_ACTION_DIM,
    JOINT_DIM,
    StraightnessRetimeConfig,
    ema_action_chunk,
    retime_by_straightness,
    upsample_trajectory,
)

logger = logging.getLogger(__name__)

StopAt = Literal["hold", "home"]
StartFrom = Literal["home", "current"]


@dataclass
class PiperAsyncBufferConfig:
    """Marvin async-buffer settings in control-loop steps."""

    enabled: bool = False
    delay: int = 3
    exec_horizon: int = 40
    min_smooth_steps: int = 8

    def __post_init__(self) -> None:
        if not 0 <= self.delay < self.exec_horizon:
            raise ValueError("async buffer requires 0 <= delay < exec_horizon")
        if self.min_smooth_steps < 1:
            raise ValueError("min_smooth_steps must be >= 1")


@dataclass
class PiperDeploymentConfig:
    """Piper policy deployment settings exposed under ``robot.deployment``."""

    # Use str instead of Literal here because draccus 0.10 cannot decode Literal
    # fields from CLI. __post_init__ still validates the accepted values.
    control_mode: str = "joint"
    chunk_upsample: int = 1
    ema_alpha: float = 1.0
    retime: StraightnessRetimeConfig = field(default_factory=StraightnessRetimeConfig)
    gripper: PiperGripperFilterConfig = field(default_factory=PiperGripperFilterConfig)
    async_buffer: PiperAsyncBufferConfig = field(default_factory=PiperAsyncBufferConfig)

    start_from: str = "home"
    stop_at: str = "home"
    home_position: tuple[float, float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    home_hz: float = 10.0
    home_interp_steps: int = 10
    home_settle_s: float = 1.5

    # Optional final command gate. None preserves the current verified driver behaviour.
    max_joint_step_rad: float | None = None
    disable_action_write: bool = False

    def __post_init__(self) -> None:
        if self.control_mode not in ("joint", "eef"):
            raise ValueError("control_mode must be 'joint' or 'eef'")
        if self.chunk_upsample < 1:
            raise ValueError("chunk_upsample must be >= 1")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.start_from not in ("home", "current"):
            raise ValueError("start_from must be 'home' or 'current'")
        if self.stop_at not in ("hold", "home"):
            raise ValueError("stop_at must be 'hold' or 'home'")
        if len(self.home_position) != JOINT_ACTION_DIM:
            raise ValueError(f"home_position must contain {JOINT_ACTION_DIM} values")
        if not np.all(np.isfinite(np.asarray(self.home_position, dtype=np.float64))):
            raise ValueError("home_position contains NaN or infinity")
        if self.home_hz <= 0.0:
            raise ValueError("home_hz must be > 0")
        if self.home_interp_steps < 1:
            raise ValueError("home_interp_steps must be >= 1")
        if self.home_settle_s < 0.0:
            raise ValueError("home_settle_s must be >= 0")
        if self.max_joint_step_rad is not None and self.max_joint_step_rad <= 0.0:
            raise ValueError("max_joint_step_rad must be > 0 when set")


class PiperActionPipeline:
    """Chunk pipeline: retime -> upsample -> cross-chunk causal EMA."""

    def __init__(
        self,
        config: PiperDeploymentConfig,
        *,
        ik_adapter: PiperIKAdapter | None = None,
    ) -> None:
        self.config = config
        self._ik: PiperIKAdapter = ik_adapter or UnconfiguredPiperIKAdapter()
        self._last_executed: np.ndarray | None = None
        self._lock = threading.Lock()
        self._warned_missing_fk = False

    def set_ik_adapter(self, adapter: PiperIKAdapter) -> None:
        self._ik = adapter
        self._warned_missing_fk = False

    def mark_executed(self, policy_action: np.ndarray) -> None:
        movement_dim = JOINT_DIM if self.config.control_mode == "joint" else 7
        with self._lock:
            self._last_executed = np.asarray(policy_action, dtype=np.float64)[:movement_dim].copy()

    def reset(self) -> None:
        with self._lock:
            self._last_executed = None

    def _joint_metric_xyz(self, actions: np.ndarray) -> np.ndarray | None:
        points: list[np.ndarray] = []
        for action in actions:
            pose = self._ik.forward(np.asarray(action[:JOINT_DIM], dtype=np.float64))
            if pose is None:
                if not self._warned_missing_fk:
                    logger.warning("Piper joint retime skipped: no FK implementation is configured")
                    self._warned_missing_fk = True
                return None
            points.append(np.asarray(pose[:3], dtype=np.float64))
        return np.stack(points)

    def process_chunk(self, actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions)
        mode = self.config.control_mode
        if self.config.retime.enabled:
            metric = None if mode == "eef" else self._joint_metric_xyz(values)
            if mode == "eef" or metric is not None:
                values = retime_by_straightness(
                    values,
                    self.config.retime,
                    mode,
                    metric_xyz=metric,
                )
        values = upsample_trajectory(values, self.config.chunk_upsample, mode)
        with self._lock:
            seed = None if self._last_executed is None else self._last_executed.copy()
        # As in Marvin main, the async ensemble owns chunk seams. Applying a
        # second cross-chunk EMA would add avoidable lag.
        ema_alpha = 1.0 if self.config.async_buffer.enabled else self.config.ema_alpha
        return ema_action_chunk(values, ema_alpha, mode, seed=seed)


class PiperDeploymentController:
    """Own policy actions and home/hold lifecycle without changing the SDK layer."""

    def __init__(
        self,
        config: PiperDeploymentConfig,
        *,
        read_positions: Callable[[], dict[str, float]],
        write_positions: Callable[[list[float]], None],
        ik_adapter: PiperIKAdapter | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._read_positions = read_positions
        self._write_positions = write_positions
        self._sleep = sleep_fn
        self.pipeline = PiperActionPipeline(config, ik_adapter=ik_adapter)
        self._ik: PiperIKAdapter = ik_adapter or UnconfiguredPiperIKAdapter()
        self._lock = threading.RLock()
        self._last_joint_action: np.ndarray | None = None
        self._last_policy_action: np.ndarray | None = None
        self._halted = False
        self._closed = False
        self._stop_at: StopAt = config.stop_at
        self._gripper = GripperHysteresisWindow(config.gripper)
        if config.async_buffer.enabled and config.ema_alpha < 1.0:
            logger.info(
                "Piper async buffer owns chunk seams; configured ema_alpha=%.3f is disabled",
                config.ema_alpha,
            )

    def set_ik_adapter(self, adapter: PiperIKAdapter) -> None:
        with self._lock:
            self._ik = adapter
            self.pipeline.set_ik_adapter(adapter)

    def set_stop_at(self, mode: StopAt) -> None:
        if mode not in ("hold", "home"):
            raise ValueError("stop mode must be 'hold' or 'home'")
        with self._lock:
            self._stop_at = mode

    def halt_policy_actions(self) -> None:
        with self._lock:
            self._halted = True

    def resume_policy_actions(self) -> None:
        with self._lock:
            self._halted = False

    def _read_vector(self) -> np.ndarray:
        values = self._read_positions()
        vector = np.array(
            [*(float(values[f"joint_{index}.pos"]) for index in range(1, 7)), float(values["gripper.pos"])],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(vector)):
            raise RuntimeError("Piper feedback contains NaN or infinity")
        return vector

    def current_joint_action(self) -> np.ndarray:
        with self._lock:
            current = self._read_vector()
            self._last_joint_action = current.copy()
            return current

    def prepare_action_chunk(self, actions: np.ndarray) -> np.ndarray:
        return self.pipeline.process_chunk(actions)

    def make_async_action_buffer(self) -> StreamActionBuffer | None:
        cfg = self.config.async_buffer
        if not cfg.enabled:
            return None
        return StreamActionBuffer(cfg.delay, cfg.exec_horizon, cfg.min_smooth_steps)

    def _limit_joint_step(self, target: np.ndarray, reference: np.ndarray) -> np.ndarray:
        max_step = self.config.max_joint_step_rad
        if max_step is None:
            return target
        out = target.copy()
        out[:JOINT_DIM] = reference[:JOINT_DIM] + np.clip(
            target[:JOINT_DIM] - reference[:JOINT_DIM], -max_step, max_step
        )
        return out

    def _write(self, target: np.ndarray) -> dict[str, float]:
        if not np.all(np.isfinite(target)):
            raise ValueError("refusing to send a non-finite Piper action")
        if not self.config.disable_action_write:
            self._write_positions(target.tolist())
        self._last_joint_action = target.copy()
        return {
            **{f"joint_{index + 1}.pos": float(target[index]) for index in range(JOINT_DIM)},
            "gripper.pos": float(target[-1]),
        }

    def send_joint_target(
        self,
        target: np.ndarray | list[float],
        *,
        joint_alpha: float = 1.0,
        gripper_alpha: float = 1.0,
    ) -> dict[str, float]:
        """Send a joint target with the legacy Piper teleoperation EMA."""

        if not 0.0 < joint_alpha <= 1.0 or not 0.0 < gripper_alpha <= 1.0:
            raise ValueError("joint_alpha and gripper_alpha must be in (0, 1]")
        with self._lock:
            values = np.asarray(target, dtype=np.float64).reshape(-1)
            if values.shape != (JOINT_ACTION_DIM,):
                raise ValueError(f"joint target must have shape ({JOINT_ACTION_DIM},), got {values.shape}")
            reference = self._last_joint_action
            if reference is None:
                reference = self._read_vector()
            command = values.copy()
            command[:JOINT_DIM] = (
                joint_alpha * values[:JOINT_DIM] + (1.0 - joint_alpha) * reference[:JOINT_DIM]
            )
            command[-1] = gripper_alpha * values[-1] + (1.0 - gripper_alpha) * reference[-1]
            command = self._limit_joint_step(command, reference)
            return self._write(command)

    def send_policy_action(self, policy_action: np.ndarray, *, now: float | None = None) -> dict[str, float]:
        """Convert joint/EEF policy space into the sole joint CAN command path."""

        with self._lock:
            if self._halted:
                if self._last_joint_action is None:
                    self._last_joint_action = self._read_vector()
                return self._write(self._last_joint_action)

            action = np.asarray(policy_action, dtype=np.float64).reshape(-1)
            expected = JOINT_ACTION_DIM if self.config.control_mode == "joint" else EEF_ACTION_DIM
            if action.shape[0] < expected:
                raise ValueError(
                    f"{self.config.control_mode} policy action requires {expected} values, got {action.shape}"
                )
            if not np.all(np.isfinite(action)):
                raise ValueError("policy action contains NaN or infinity")

            current = self._last_joint_action
            if current is None:
                current = self._read_vector()

            if self.config.control_mode == "joint":
                joints = action[:JOINT_DIM].copy()
                raw_gripper = float(action[JOINT_DIM])
                movement_accepted = True
                executed_action = action
            else:
                eef_pose = action[:7].copy()
                quat_norm = float(np.linalg.norm(eef_pose[3:7]))
                if quat_norm < 1e-8:
                    raise ValueError("EEF policy action contains a zero-norm quaternion")
                eef_pose[3:7] /= quat_norm
                solution = self._ik.solve(eef_pose, current[:JOINT_DIM])
                if solution is None:
                    logger.warning("Piper IK failed; holding the last good joint command")
                    joints = current[:JOINT_DIM].copy()
                    movement_accepted = False
                else:
                    joints = np.asarray(solution, dtype=np.float64).reshape(JOINT_DIM)
                    movement_accepted = True
                raw_gripper = float(action[7])
                executed_action = action.copy()
                executed_action[:7] = eef_pose

            gripper = self._gripper.update(raw_gripper, now=now)
            target = np.concatenate([joints, [gripper]])
            target = self._limit_joint_step(target, current)
            result = self._write(target)
            if movement_accepted:
                self._last_policy_action = executed_action.copy()
                self.pipeline.mark_executed(executed_action)
            return result

    def send_eef_action(
        self,
        eef_pose7: np.ndarray | list[float],
        *,
        gripper: float,
        now: float | None = None,
    ) -> dict[str, float]:
        if self.config.control_mode != "eef":
            raise RuntimeError("set robot.deployment.control_mode=eef before sending EEF actions")
        return self.send_policy_action(np.concatenate([np.asarray(eef_pose7), [gripper]]), now=now)

    def reset(self) -> None:
        with self._lock:
            current = self._read_vector()
            self._last_joint_action = current.copy()
            self._last_policy_action = None
            self._gripper.reset(float(current[-1]))
            self.pipeline.reset()
            self._halted = False

    def hold(self) -> dict[str, float]:
        """Issue the current/last position once and reject subsequent policy actions."""

        with self._lock:
            self._halted = True
            target = self._last_joint_action
            if target is None:
                target = self._read_vector()
            return self._write(target)

    def go_home(self) -> None:
        """Linearly stream from feedback pose to home, matching Marvin main."""

        with self._lock:
            start = self._read_vector()
            target = np.asarray(self.config.home_position, dtype=np.float64)
            steps = self.config.home_interp_steps
            interval = 1.0 / self.config.home_hz
            logger.info(
                "Piper smooth home: steps=%s hz=%.1f settle=%.2fs",
                steps,
                self.config.home_hz,
                self.config.home_settle_s,
            )
            for index in range(1, steps + 1):
                alpha = index / steps
                waypoint = (1.0 - alpha) * start + alpha * target
                self._write(waypoint)
                if index < steps and not self.config.disable_action_write:
                    self._sleep(interval)
            if self.config.home_settle_s and not self.config.disable_action_write:
                self._sleep(self.config.home_settle_s)

    def initialize_pose(self) -> None:
        self._closed = False
        if self.config.start_from == "home":
            self.go_home()
        self.reset()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._halted = True
            stop_at = self._stop_at
        if stop_at == "home":
            self.go_home()
        else:
            self.hold()
