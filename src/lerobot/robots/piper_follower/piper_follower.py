#!/usr/bin/env python

import logging
from functools import cached_property
from typing import Any

import numpy as np
import torch

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors.piper.piper import PiperMotorsBus, PiperMotorsBusConfig, action_pos_dict_from_piper_read
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_piper_follower import PIPERFollowerConfig
from .deployment import PiperDeploymentController, StopAt
from .ik_adapter import PiperIKAdapter

logger = logging.getLogger(__name__)


def get_motor_names(arm: dict[str, Any]) -> list[str]:
    return [motor for bus in arm.values() for motor in bus.motors]


class PIPERFollower(Robot):
    config_class = PIPERFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PIPERFollowerConfig):
        super().__init__(config)
        self.config = config
        self.bus = PiperMotorsBus(
            PiperMotorsBusConfig(
                can_name="can_follower",
                motors={
                    "joint_1": (1, "agilex_piper"),
                    "joint_2": (2, "agilex_piper"),
                    "joint_3": (3, "agilex_piper"),
                    "joint_4": (4, "agilex_piper"),
                    "joint_5": (5, "agilex_piper"),
                    "joint_6": (6, "agilex_piper"),
                    "gripper": (7, "agilex_piper"),
                },
            )
        )
        self.logs = {}
        self._is_connected = False
        self._is_calibrated = False
        self.cameras = make_cameras_from_configs(config.cameras)
        self.deployment = PiperDeploymentController(
            config.deployment,
            read_positions=self.bus.read_action_positions,
            write_positions=self.bus.write,
        )

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        arm_dict = {"follower": self.bus}
        action_names = get_motor_names(arm_dict)
        state_names = get_motor_names(arm_dict)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def _motors_ft(self) -> dict[str, type]:
        """用于 record/replay 的电机动作描述"""
        arm_dict = {"follower": self.bus}
        motor_names = get_motor_names(arm_dict)
        return {f"{name}.pos": float for name in motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """用于 record/replay 的相机图像描述"""
        return {cam_key: (cam.height, cam.width, 3) for cam_key, cam in self.cameras.items()}

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.config.deployment.control_mode == "eef":
            return {
                **{f"eef.{name}": float for name in ("x", "y", "z", "qx", "qy", "qz", "qw")},
                "gripper.pos": float,
            }
        return self._motors_ft

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def configure(self, **kwargs):
        # 不做任何事,过抽象类用
        pass

    @property
    def is_connected(self) -> bool:
        """机器人和所有相机是否都已连接"""
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        """机器人是否已完成标定"""
        return self._is_calibrated

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self, calibrate: bool = True) -> None:
        """Connect piper and cameras"""
        if self._is_connected:
            raise DeviceAlreadyConnectedError("Piper is already connected. Do not run robot.connect() twice.")

        if not self.bus.connect(enable=True):
            raise ConnectionError("Failed to enable Piper follower")
        connected_cameras = []
        try:
            logger.info("Piper follower enabled")
            for name, camera in self.cameras.items():
                camera.connect()
                if not camera.is_connected:
                    raise ConnectionError(f"Failed to connect camera {name}")
                connected_cameras.append(camera)
                logger.info("Piper camera connected: %s", name)

            self._is_connected = True
            if calibrate:
                self.calibrate()
            else:
                self.deployment.reset()
        except Exception:
            self._is_connected = False
            for camera in reversed(connected_cameras):
                try:
                    camera.disconnect()
                except Exception:
                    logger.exception("Failed to disconnect Piper camera during connect rollback")
            try:
                self.bus.connect(enable=False)
            except Exception:
                logger.exception("Failed to disable Piper during connect rollback")
            raise

        logger.info("Piper follower and %s camera(s) ready", len(self.cameras))

    def disconnect(self) -> None:
        """Apply configured hold/home stop behaviour, then disable Piper and cameras."""
        if not self._is_connected:
            return
        first_error: Exception | None = None
        try:
            self.deployment.close()
        except Exception as exc:
            first_error = exc
            logger.exception("Piper hold/home failed; continuing with hardware disable")
        try:
            if not self.bus.connect(enable=False):
                raise RuntimeError("Piper SDK did not confirm disable")
        except Exception as exc:
            first_error = first_error or exc
            logger.exception("Failed to disable Piper follower")
        for name, camera in self.cameras.items():
            try:
                camera.disconnect()
            except Exception as exc:
                first_error = first_error or exc
                logger.exception("Failed to disconnect Piper camera: %s", name)
        self._is_connected = False
        if first_error is not None:
            raise first_error

    def calibrate(self):
        """Initialize from current pose or move smoothly to the configured home."""
        if not self._is_connected:
            raise ConnectionError()

        self.deployment.initialize_pose()
        self._is_calibrated = True  # 标记为已标定

    def get_observation(self) -> dict:
        """Capture current joint positions and camera images"""
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected. Run `robot.connect()` first.")

        obs_dict = action_pos_dict_from_piper_read(self.bus.read())

        # 读取图像
        for name, cam in self.cameras.items():
            obs_dict[name] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Teleoperation-compatible joint action path with the existing EMA settings."""
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected.")
        if self.config.deployment.control_mode == "eef" and "eef.x" in action:
            return self.send_policy_action(action)

        motor_order = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper",
        ]
        target_joints = [float(action[f"{motor}.pos"]) for motor in motor_order]

        ja = float(np.clip(self.config.teleop_joint_alpha, 0.0, 1.0))
        ga = float(np.clip(self.config.teleop_gripper_alpha, 0.0, 1.0))
        # Preserve the verified legacy semantics: alpha <= 0 disables EMA.
        ja = 1.0 if ja <= 0.0 else ja
        ga = 1.0 if ga <= 0.0 else ga
        return self.deployment.send_joint_target(target_joints, joint_alpha=ja, gripper_alpha=ga)

    def policy_action_to_dict(self, action: torch.Tensor | np.ndarray) -> dict[str, float]:
        """Map a policy tensor without assuming joint-space output."""

        values = np.asarray(action.detach().cpu() if isinstance(action, torch.Tensor) else action).reshape(-1)
        if self.config.deployment.control_mode == "joint":
            if values.shape[0] < 7:
                raise ValueError(f"joint policy action requires 7 values, got {values.shape}")
            return {
                **{f"joint_{index + 1}.pos": float(values[index]) for index in range(6)},
                "gripper.pos": float(values[6]),
            }
        if values.shape[0] < 8:
            raise ValueError(f"EEF policy action requires 8 values, got {values.shape}")
        names = ("x", "y", "z", "qx", "qy", "qz", "qw")
        return {
            **{f"eef.{name}": float(values[index]) for index, name in enumerate(names)},
            "gripper.pos": float(values[7]),
        }

    def prepare_policy_action_chunk(self, actions: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Run Marvin-style retime -> upsample -> EMA on a complete policy chunk."""

        is_tensor = isinstance(actions, torch.Tensor)
        if is_tensor:
            device, dtype = actions.device, actions.dtype
            values = actions.detach().cpu().numpy()
        else:
            device, dtype = None, None
            values = np.asarray(actions)
        processed = self.deployment.prepare_action_chunk(values)
        result = torch.from_numpy(np.asarray(processed))
        return result.to(device=device, dtype=dtype) if is_tensor else result

    def make_async_action_buffer(self):
        """Create the optional Marvin-style buffer used by the generic gRPC client."""

        return self.deployment.make_async_action_buffer()

    def send_policy_action(self, action: dict[str, float]) -> dict[str, float]:
        """Policy path: joint or EEF input, always executed by joint CAN commands."""

        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected.")
        if self.config.deployment.control_mode == "joint":
            vector = np.array(
                [*(action[f"joint_{index}.pos"] for index in range(1, 7)), action["gripper.pos"]],
                dtype=np.float64,
            )
        else:
            vector = np.array(
                [
                    *(action[f"eef.{name}"] for name in ("x", "y", "z", "qx", "qy", "qz", "qw")),
                    action["gripper.pos"],
                ],
                dtype=np.float64,
            )
        return self.deployment.send_policy_action(vector)

    def send_eef_action(self, eef_pose7: np.ndarray | list[float], *, gripper: float) -> dict[str, float]:
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected.")
        return self.deployment.send_eef_action(eef_pose7, gripper=gripper)

    def set_ik_adapter(self, adapter: PiperIKAdapter) -> None:
        self.deployment.set_ik_adapter(adapter)

    def halt_policy_actions(self) -> None:
        self.deployment.halt_policy_actions()

    def set_stop_at(self, mode: StopAt) -> None:
        self.deployment.set_stop_at(mode)

    def hold_position(self) -> dict[str, float]:
        return self.deployment.hold()

    def go_home(self) -> None:
        self.deployment.go_home()

    def get_record_action_from_follower(self) -> dict[str, float]:
        """Follower joint/gripper positions in rad (and gripper m) for dataset `action` (scheme B)."""
        return self.bus.read_action_positions()
