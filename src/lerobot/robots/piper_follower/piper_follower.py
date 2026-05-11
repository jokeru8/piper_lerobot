#!/usr/bin/env python

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.piper.piper import PiperMotorsBus, PiperMotorsBusConfig, action_pos_dict_from_piper_read

from ..robot import Robot
from .config_piper_follower import PIPERFollowerConfig

logger = logging.getLogger(__name__)

def get_motor_names(arm: dict[str, Any]) -> list[str]:
    return [motor for arm_key, bus in arm.items() for motor in bus.motors]

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
                }
            )
        )
        self.logs = {}
        self._is_connected = False
        self._is_calibrated = False
        self.cameras = make_cameras_from_configs(config.cameras)
        self._ema_cmd: list[float] | None = None

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
        #return {f"observation.images.{cam_key}": (cam.height, cam.width, 3) for cam_key, cam in self.cameras.items()}
        return {cam_key: (cam.height, cam.width, 3) for cam_key, cam in self.cameras.items()}

    @cached_property
    def action_features(self) -> dict[str, type]:
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
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        """机器人是否已完成标定"""
        return self.bus.is_calibrated

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self) -> None:
        """Connect piper and cameras"""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                "Piper is already connected. Do not run robot.connect() twice."
            )

        self.bus.connect(enable=True)
        print("piper follower connected")

        # connect cameras
        for name in self.cameras:
            self.cameras[name].connect()
            self._is_connected = self._is_connected and self.cameras[name].is_connected
            print(f"camera {name} connected")

        print("All connected")
        self._is_connected = True

        self.calibrate()

    def disconnect(self) -> None:
        """move to home position, disenable piper and cameras"""
        self.bus.safe_disconnect()
        print("piper disable after 5 seconds")
        time.sleep(5)
        self.bus.connect(enable=False)

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self._is_connected = False
        self._ema_cmd = None

    def calibrate(self):
        """move piper to the home position"""
        if not self._is_connected:
            raise ConnectionError()

        self.bus.apply_calibration()
        self._is_calibrated = True  # 标记为已标定
        self._ema_cmd = None

    def get_observation(self) -> dict:
        """Capture current joint positions and camera images"""
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected. Run `robot.connect()` first.")

        obs_dict = action_pos_dict_from_piper_read(self.bus.read())

        # 读取图像
        for name, cam in self.cameras.items():
            #obs_dict[f"observation.images.{name}"] = cam.async_read()
            obs_dict[name] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Receive action dict from record() and send to motor"""
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected.")

        motor_order = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
        target_joints = [float(action[f"{motor}.pos"]) for motor in motor_order]

        ja = min(1.0, max(0.0, float(self.config.teleop_joint_alpha)))
        ga = min(1.0, max(0.0, float(self.config.teleop_gripper_alpha)))
        if ja <= 0.0:
            ja = 1.0
        if ga <= 0.0:
            ga = 1.0

        if ja >= 1.0 - 1e-9 and ga >= 1.0 - 1e-9:
            cmd = list(target_joints)
        else:
            if self._ema_cmd is None:
                init = self.bus.read_action_positions()
                self._ema_cmd = [float(init[f"{m}.pos"]) for m in motor_order]
            for i in range(6):
                self._ema_cmd[i] = ja * target_joints[i] + (1.0 - ja) * self._ema_cmd[i]
            self._ema_cmd[6] = ga * target_joints[6] + (1.0 - ga) * self._ema_cmd[6]
            cmd = self._ema_cmd

        self.bus.write(cmd)
        return {f"{motor_order[i]}.pos": cmd[i] for i in range(len(motor_order))}

    def get_record_action_from_follower(self) -> dict[str, float]:
        """Follower joint/gripper positions in rad (and gripper m) for dataset `action` (scheme B)."""
        return self.bus.read_action_positions()