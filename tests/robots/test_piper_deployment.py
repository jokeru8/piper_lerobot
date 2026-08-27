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

from __future__ import annotations

import draccus
import numpy as np
import pytest

from lerobot.robots.piper_follower.async_buffer import StreamActionBuffer
from lerobot.robots.piper_follower.config_piper_follower import PIPERFollowerConfig
from lerobot.robots.piper_follower.deployment import (
    PiperAsyncBufferConfig,
    PiperDeploymentConfig,
    PiperDeploymentController,
)
from lerobot.robots.piper_follower.gripper_filter import (
    GripperHysteresisWindow,
    PiperGripperFilterConfig,
)
from lerobot.robots.piper_follower.ik_adapter import CallablePiperIKAdapter
from lerobot.robots.piper_follower.trajectory import (
    StraightnessRetimeConfig,
    ema_action_chunk,
    retime_by_straightness,
    upsample_trajectory,
)


def _feedback(values: np.ndarray) -> dict[str, float]:
    return {
        **{f"joint_{index + 1}.pos": float(values[index]) for index in range(6)},
        "gripper.pos": float(values[6]),
    }


def test_deployment_config_decodes_from_cli() -> None:
    cfg = draccus.parse(
        PIPERFollowerConfig,
        args=[
            "--cameras={}",
            "--deployment.control_mode=joint",
            "--deployment.start_from=current",
            "--deployment.stop_at=hold",
            "--deployment.async_buffer.enabled=true",
            "--deployment.gripper.enabled=true",
        ],
    )
    assert cfg.deployment.control_mode == "joint"
    assert cfg.deployment.start_from == "current"
    assert cfg.deployment.stop_at == "hold"
    assert cfg.deployment.async_buffer.enabled
    assert cfg.deployment.gripper.enabled


def test_joint_upsample_and_ema_preserve_gripper_samples() -> None:
    chunk = np.array([[0.0] * 7, [1.0] * 7], dtype=np.float32)
    upsampled = upsample_trajectory(chunk, factor=3, control_mode="joint")
    assert upsampled.shape == (4, 7)
    np.testing.assert_allclose(upsampled[[0, -1]], chunk)

    filtered = ema_action_chunk(upsampled, 0.5, "joint", seed=np.zeros(6))
    np.testing.assert_allclose(filtered[:, 6], upsampled[:, 6])
    assert filtered[1, 0] < upsampled[1, 0]


def test_async_crossfade_disables_extra_chunk_ema() -> None:
    state = np.zeros(7)
    controller = PiperDeploymentController(
        PiperDeploymentConfig(
            ema_alpha=0.1,
            async_buffer=PiperAsyncBufferConfig(enabled=True),
            start_from="current",
        ),
        read_positions=lambda: _feedback(state),
        write_positions=lambda _: None,
        sleep_fn=lambda _: None,
    )
    chunk = np.array([[0.0] * 7, [1.0] * 7])
    np.testing.assert_allclose(controller.prepare_action_chunk(chunk), chunk)


def test_joint_retime_without_fk_is_safely_skipped() -> None:
    state = np.zeros(7)
    controller = PiperDeploymentController(
        PiperDeploymentConfig(
            retime=StraightnessRetimeConfig(enabled=True),
            start_from="current",
        ),
        read_positions=lambda: _feedback(state),
        write_positions=lambda _: None,
        sleep_fn=lambda _: None,
    )
    chunk = np.array([[0.0] * 7, [1.0] * 7])
    np.testing.assert_allclose(controller.prepare_action_chunk(chunk), chunk)


def test_eef_interpolation_uses_unit_quaternions() -> None:
    chunk = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.08],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.00],
        ]
    )
    result = upsample_trajectory(chunk, factor=4, control_mode="eef")
    assert result.shape == (5, 8)
    np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1.0)
    np.testing.assert_allclose(result[[0, -1]], chunk)


def test_eef_retime_preserves_endpoints() -> None:
    t = np.linspace(0.0, 1.0, 20)
    chunk = np.column_stack(
        [
            t,
            np.zeros_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
            np.ones_like(t),
            t,
        ]
    )
    cfg = StraightnessRetimeConfig(enabled=True, v_min=1.0, v_max=2.0)
    result = retime_by_straightness(chunk, cfg, "eef")
    assert result.shape[1] == 8
    np.testing.assert_allclose(result[0], chunk[0])
    np.testing.assert_allclose(result[-1], chunk[-1])


def test_gripper_schmitt_vote_and_ramp() -> None:
    cfg = PiperGripperFilterConfig(enabled=True, window_n=4, ratio=0.75, ramp_s=0.3)
    filt = GripperHysteresisWindow(cfg, initial=cfg.open_position)

    assert filt.update(0.0, now=1.0) == cfg.open_position
    assert filt.update(0.0, now=1.1) == cfg.open_position
    assert filt.update(0.0, now=1.2) == cfg.open_position
    assert filt.state == cfg.closed_position
    assert filt.update(0.0, now=1.35) == pytest.approx(0.04)
    assert filt.update(0.0, now=1.5) == cfg.closed_position


def test_async_buffer_early_launch_drop_and_blend() -> None:
    buffer = StreamActionBuffer(delay=2, exec_horizon=5, min_smooth_steps=3)
    buffer.integrate_first_chunk(np.arange(6, dtype=float).reshape(-1, 1))
    assert buffer.get_next_action()[0] == 0.0
    assert buffer.get_next_action()[0] == 1.0
    assert buffer.mark_launch_if_ready()
    assert not buffer.mark_launch_if_ready()
    assert buffer.get_next_action()[0] == 2.0

    assert buffer.integrate_new_chunk(np.arange(10.0, 16.0).reshape(-1, 1))
    # One action elapsed after launch, so new[0] is dropped. The first overlap
    # retains the remaining old trajectory and the last overlap reaches new.
    assert buffer.get_next_action()[0] == 3.0
    assert buffer.get_next_action()[0] == pytest.approx(8.0)


def test_async_buffer_launches_early_for_retimed_short_chunk() -> None:
    buffer = StreamActionBuffer(delay=3, exec_horizon=40, min_smooth_steps=3)
    buffer.integrate_first_chunk(np.zeros((3, 7)))
    assert buffer.mark_launch_if_ready()


def test_async_buffer_rearms_after_discarding_a_late_chunk() -> None:
    buffer = StreamActionBuffer(delay=1, exec_horizon=3, min_smooth_steps=2)
    buffer.integrate_first_chunk(np.zeros((5, 7)))
    assert buffer.get_next_action() is not None
    assert buffer.mark_launch_if_ready()
    for _ in range(3):
        assert buffer.get_next_action() is not None
    assert not buffer.integrate_new_chunk(np.zeros((2, 7)))
    assert buffer.mark_launch_if_ready()


def test_smooth_home_streams_verified_joint_path_without_hardware() -> None:
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.08])
    writes: list[np.ndarray] = []
    sleeps: list[float] = []
    cfg = PiperDeploymentConfig(
        home_position=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        home_interp_steps=4,
        home_hz=10.0,
        home_settle_s=0.5,
    )
    controller = PiperDeploymentController(
        cfg,
        read_positions=lambda: _feedback(state),
        write_positions=lambda value: writes.append(np.asarray(value)),
        sleep_fn=sleeps.append,
    )

    controller.go_home()
    assert len(writes) == 4
    np.testing.assert_allclose(writes[0], state * 0.75)
    np.testing.assert_allclose(writes[-1], np.zeros(7))
    assert sleeps == [0.1, 0.1, 0.1, 0.5]


def test_eef_adapter_always_executes_through_joint_writer() -> None:
    state = np.zeros(7)
    writes: list[np.ndarray] = []
    adapter = CallablePiperIKAdapter(
        solve_fn=lambda pose, seed: np.arange(6, dtype=float),
        forward_fn=lambda joints: np.array([joints[0], joints[1], joints[2], 0.0, 0.0, 0.0, 1.0]),
    )
    controller = PiperDeploymentController(
        PiperDeploymentConfig(control_mode="eef", start_from="current", stop_at="hold"),
        read_positions=lambda: _feedback(state),
        write_positions=lambda value: writes.append(np.asarray(value)),
        ik_adapter=adapter,
        sleep_fn=lambda _: None,
    )

    result = controller.send_policy_action(np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.08]))
    assert len(writes) == 1
    np.testing.assert_allclose(writes[0][:6], np.arange(6))
    assert result["gripper.pos"] == pytest.approx(0.08)


def test_eef_without_solver_has_clear_failure_and_no_write() -> None:
    writes: list[list[float]] = []
    controller = PiperDeploymentController(
        PiperDeploymentConfig(control_mode="eef", start_from="current", stop_at="hold"),
        read_positions=lambda: _feedback(np.zeros(7)),
        write_positions=writes.append,
        sleep_fn=lambda _: None,
    )

    with pytest.raises(RuntimeError, match="requires an IK adapter"):
        controller.send_policy_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    assert not writes
