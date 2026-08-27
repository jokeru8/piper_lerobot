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

"""Trajectory retiming, interpolation and smoothing for Piper deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ControlMode = Literal["joint", "eef"]

JOINT_DIM = 6
JOINT_ACTION_DIM = 7  # six joints + gripper
EEF_POSE_DIM = 7  # xyz + quaternion xyzw
EEF_ACTION_DIM = 8  # pose7 + gripper
_EPS = 1e-8


@dataclass
class StraightnessRetimeConfig:
    """Marvin straightness retime parameters adapted to a single Piper arm."""

    enabled: bool = False
    smooth_window: int = 15
    straightness_window: int = 15
    v_min: float = 1.0
    v_max: float = 2.0
    alpha: float = 1.5
    chord_sigma: float = 0.92
    edge_ratio: float = 0.1
    ramp_ratio: float = 0.05
    v_base: float | None = None

    def __post_init__(self) -> None:
        if self.v_min <= 0.0 or self.v_max < self.v_min:
            raise ValueError("retime requires 0 < v_min <= v_max")
        if not 0.0 <= self.edge_ratio <= 0.5:
            raise ValueError("edge_ratio must be in [0, 0.5]")
        if self.ramp_ratio < 0.0:
            raise ValueError("ramp_ratio must be >= 0")
        if self.v_base is not None and self.v_base <= 0.0:
            raise ValueError("v_base must be > 0")


def _validate_actions(actions: np.ndarray, control_mode: ControlMode) -> np.ndarray:
    if control_mode not in ("joint", "eef"):
        raise ValueError(f"unknown control mode: {control_mode!r}")
    values = np.asarray(actions)
    expected = JOINT_ACTION_DIM if control_mode == "joint" else EEF_ACTION_DIM
    if values.ndim != 2 or values.shape[1] < expected:
        raise ValueError(f"{control_mode} actions must have shape (T, >={expected}), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("actions contain NaN or infinity")
    if control_mode == "eef" and np.any(np.linalg.norm(values[:, 3:7], axis=1) < _EPS):
        raise ValueError("EEF actions contain a zero-norm quaternion")
    return values


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), _EPS)
    return quat / norm


def _slerp(q0: np.ndarray, q1: np.ndarray, amount: float) -> np.ndarray:
    start = _normalize_quat(q0).reshape(4)
    end = _normalize_quat(q1).reshape(4)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    if dot > 0.9995:
        return _normalize_quat(start + amount * (end - start)).reshape(4)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return _normalize_quat(
        np.sin((1.0 - amount) * theta) / sin_theta * start + np.sin(amount * theta) / sin_theta * end
    ).reshape(4)


def _linear_interp(values: np.ndarray, query: np.ndarray, source: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(values.shape[0], -1)
    out = np.empty((len(query), flat.shape[1]), dtype=np.float64)
    for dim in range(flat.shape[1]):
        out[:, dim] = np.interp(query, source, flat[:, dim])
    return out.reshape((len(query),) + values.shape[1:])


def _quat_interp(quats: np.ndarray, query: np.ndarray, source: np.ndarray) -> np.ndarray:
    quats = _normalize_quat(quats)
    out = np.empty((len(query), 4), dtype=np.float64)
    for index, value in enumerate(query):
        if value <= source[0]:
            out[index] = quats[0]
            continue
        if value >= source[-1]:
            out[index] = quats[-1]
            continue
        left = int(np.searchsorted(source, value, side="right") - 1)
        left = int(np.clip(left, 0, len(source) - 2))
        width = source[left + 1] - source[left]
        amount = 0.0 if width < _EPS else float((value - source[left]) / width)
        out[index] = _slerp(quats[left], quats[left + 1], amount)
    return out


def _resample(
    actions: np.ndarray,
    query: np.ndarray,
    source: np.ndarray,
    control_mode: ControlMode,
) -> np.ndarray:
    out = _linear_interp(actions, query, source)
    if control_mode == "eef":
        out[:, 3:7] = _quat_interp(actions[:, 3:7], query, source)
    return out


def upsample_trajectory(actions: np.ndarray, factor: int, control_mode: ControlMode) -> np.ndarray:
    """Interpolate ``T`` actions into ``(T-1)*factor+1`` actions."""

    values = _validate_actions(actions, control_mode)
    if factor < 1:
        raise ValueError("upsample factor must be >= 1")
    if factor == 1 or len(values) <= 1:
        return values.copy()
    source = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    query = np.linspace(0.0, 1.0, (len(values) - 1) * factor + 1, dtype=np.float64)
    return _resample(values, query, source, control_mode).astype(values.dtype, copy=False)


def ema_action_chunk(
    actions: np.ndarray,
    alpha: float,
    control_mode: ControlMode,
    *,
    seed: np.ndarray | None = None,
) -> np.ndarray:
    """Causal chunk EMA seeded from the last executed command; gripper passes through."""

    values = _validate_actions(actions, control_mode)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("EMA alpha must be in (0, 1]")
    if alpha >= 1.0 or len(values) <= 1:
        return values.copy()

    out = values.astype(np.float64, copy=True)
    movement_dim = JOINT_DIM if control_mode == "joint" else EEF_POSE_DIM
    if seed is None:
        previous = out[0, :movement_dim].copy()
    else:
        seed_values = np.asarray(seed, dtype=np.float64).reshape(-1)
        if seed_values.shape[0] < movement_dim:
            raise ValueError(f"EMA seed requires at least {movement_dim} values, got {seed_values.shape}")
        previous = seed_values[:movement_dim].copy()
        if control_mode == "eef" and np.linalg.norm(previous[3:7]) < _EPS:
            raise ValueError("EEF EMA seed contains a zero-norm quaternion")
    for index in range(len(out)):
        raw = values[index, :movement_dim].astype(np.float64)
        if control_mode == "joint":
            filtered = alpha * raw + (1.0 - alpha) * previous
        else:
            xyz = alpha * raw[:3] + (1.0 - alpha) * previous[:3]
            quat = _slerp(previous[3:7], raw[3:7], alpha)
            filtered = np.concatenate([xyz, quat])
        out[index, :movement_dim] = filtered
        previous = filtered
    return out.astype(values.dtype, copy=False)


def _odd_window(window: int, length: int) -> int:
    value = max(1, int(window))
    value += value % 2 == 0
    if value > length:
        value = length if length % 2 else max(1, length - 1)
    return value


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    size = _odd_window(window, len(values))
    if size <= 1 or len(values) < 3:
        return values.copy()
    pad = size // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(size, dtype=np.float64) / size
    out = np.empty_like(values)
    for dim in range(values.shape[1]):
        out[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    out[0], out[-1] = values[0], values[-1]
    return out


def _arc_length(points: np.ndarray) -> np.ndarray:
    delta = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(delta)])


def _local_straightness(points: np.ndarray, window: int) -> np.ndarray:
    length = len(points)
    sigma = np.ones(length, dtype=np.float64)
    if length < 3:
        return sigma
    half = _odd_window(window, length) // 2
    arc = _arc_length(points)
    for index in range(length):
        low, high = max(0, index - half), min(length - 1, index + half)
        path = float(arc[high] - arc[low])
        if path >= _EPS:
            chord = float(np.linalg.norm(points[high] - points[low]))
            sigma[index] = np.clip(chord / path, 0.0, 1.0)
    return sigma


def _speed_gate(progress: np.ndarray, edge_ratio: float, ramp_ratio: float) -> np.ndarray:
    edge = float(np.clip(edge_ratio, 0.0, 0.5))
    ramp = min(max(float(ramp_ratio), 0.0), max(0.0, (1.0 - 2.0 * edge) / 2.0))
    distance_from_edge = np.minimum(progress, 1.0 - progress)
    if ramp <= 0.0:
        return (distance_from_edge >= edge).astype(np.float64)
    x = np.clip((distance_from_edge - edge) / ramp, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _chord_replace(
    actions: np.ndarray,
    sigma: np.ndarray,
    threshold: float,
    window: int,
    control_mode: ControlMode,
) -> np.ndarray:
    if threshold >= 1.0 or len(actions) < 3:
        return actions.copy()
    out = actions.astype(np.float64, copy=True)
    half = _odd_window(window, len(actions)) // 2
    intervals: list[tuple[int, int]] = []
    index = 0
    while index < len(actions):
        low, high = max(0, index - half), min(len(actions) - 1, index + half)
        if high > low and float(np.mean(sigma[low : high + 1])) > threshold:
            intervals.append((low, high))
            index = high + 1
        else:
            index += 1
    if not intervals:
        return out
    merged: list[list[int]] = [list(intervals[0])]
    for low, high in intervals[1:]:
        if low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    for low, high in merged:
        source = np.array([0.0, 1.0])
        query = np.linspace(0.0, 1.0, high - low + 1)
        out[low : high + 1] = _resample(actions[[low, high]], query, source, control_mode)
    return out


def retime_by_straightness(
    actions: np.ndarray,
    config: StraightnessRetimeConfig,
    control_mode: ControlMode,
    *,
    metric_xyz: np.ndarray | None = None,
) -> np.ndarray:
    """Marvin-style EEF straightness retime for a single arm.

    EEF mode takes xyz directly from the action. Joint mode must be supplied an
    FK-derived ``metric_xyz`` array. Without FK the caller should skip retime.
    """

    values = _validate_actions(actions, control_mode).astype(np.float64)
    if not config.enabled or len(values) < 2:
        return values.astype(np.asarray(actions).dtype, copy=False)
    if control_mode == "eef":
        metric_xyz = values[:, :3]
    elif metric_xyz is None:
        raise ValueError("joint-space retime requires FK-derived metric_xyz")
    metric_xyz = np.asarray(metric_xyz, dtype=np.float64)
    if metric_xyz.shape != (len(values), 3):
        raise ValueError(f"metric_xyz must have shape {(len(values), 3)}, got {metric_xyz.shape}")
    if not np.all(np.isfinite(metric_xyz)):
        raise ValueError("metric_xyz contains NaN or infinity")

    smooth_metric = _moving_average(metric_xyz, config.smooth_window)
    sigma = _local_straightness(smooth_metric, config.straightness_window)
    # Replacing a Cartesian chord is only valid when the action itself contains
    # Cartesian poses. In joint mode FK is currently a metric-only port: until
    # an IK/FK model is configured, changing the joints from that metric would
    # no longer describe the same EEF path.
    command = (
        _chord_replace(values, sigma, config.chord_sigma, config.straightness_window, control_mode)
        if control_mode == "eef"
        else values.copy()
    )

    metric_after = command[:, :3] if control_mode == "eef" else metric_xyz
    metric_after = _moving_average(metric_after, config.smooth_window)
    sigma = _local_straightness(metric_after, config.straightness_window)
    delta = np.linalg.norm(np.diff(metric_after, axis=0), axis=1)
    total_distance = float(np.sum(delta))
    if total_distance < _EPS:
        return np.stack([command[0], command[-1]]).astype(np.asarray(actions).dtype, copy=False)

    path = np.concatenate([[0.0], np.cumsum(delta)])
    path_normalized = path / path[-1]
    sigma_segment = 0.5 * (sigma[:-1] + sigma[1:])
    straight_speed = config.v_min + (config.v_max - config.v_min) * np.power(
        np.clip(sigma_segment, 0.0, 1.0), config.alpha
    )
    midpoint = 0.5 * (path_normalized[:-1] + path_normalized[1:])
    gate = _speed_gate(midpoint, config.edge_ratio, config.ramp_ratio)
    base = config.v_min if config.v_base is None else config.v_base
    speed = np.maximum(base + gate * (straight_speed - base), _EPS)

    elapsed = np.concatenate([[0.0], np.cumsum(delta / speed)])
    elapsed /= elapsed[-1]
    mean_speed = float(np.average(speed, weights=np.maximum(delta, _EPS)))
    output_length = max(2, int(round((len(values) - 1) / mean_speed)) + 1)
    time_query = np.linspace(0.0, 1.0, output_length)
    path_query = np.interp(time_query, elapsed, path_normalized)
    out = _resample(command, path_query, path_normalized, control_mode)
    out[0], out[-1] = command[0], command[-1]
    return out.astype(np.asarray(actions).dtype, copy=False)
