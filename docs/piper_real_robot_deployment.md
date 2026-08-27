# Piper 真机部署层（参考 Marvin main）

本文记录从 OpenPI Marvin 示例的 `main` 分支复用到 Piper 的真机部署能力。
参考基线为 `main`（commit `47fe1f23`），没有搬运 StreamPI 分支的协议或服务。

## 已复用内容

| Marvin main 能力 | Piper 实现 | 说明 |
| --- | --- | --- |
| 异步推理提前发起、迟到 chunk 对齐、交界融合 | `async_buffer.py` + 通用 gRPC `RobotClient` hook | 用 `delay`、`exec_horizon`、`drop_n` 和线性 cross-fade；不引入 StreamPI |
| EEF 直线度 retime | `trajectory.py` | EEF xyz 计算弦长/弧长直线度，首尾减速门控，EEF 四元数用 xyzw SLERP |
| chunk 插值和 EMA | `trajectory.py` | 顺序为 `retime -> upsample -> EMA`；夹爪不做 EMA；异步融合启用时 EMA 自动为 1 |
| EEF/IK 前端、唯一关节执行通道 | `ik_adapter.py` + `deployment.py` | 策略可输出 `xyz + quat(xyzw) + gripper`，IK 后仍只调用已经验证的 Piper joint CAN writer |
| IK 失败保持 | `deployment.py` | IK 返回 `None` 时保持上一条有效关节命令；失败目标不进入下一 chunk 的 EMA seed |
| 夹爪 Schmitt + 滑窗投票 + ramp | `gripper_filter.py` | 适配 Piper 单位：米；小行程闭合、大行程打开 |
| 启动/停止平滑 home | `deployment.py` | 从反馈关节位置线性插值到 home，控制频率、步数和 settle 均可配置 |
| `s` 原地保持、`h` 平滑回 home、Ctrl+C 保持 | `keyboard_stop.py` + async client | 先 halt policy，再由断开流程执行 hold/home |
| dry command / 最终步长保护 | `deployment.py` | `disable_action_write` 和可选 `max_joint_step_rad` 均在 SDK 外围 |

底层 `PiperMotorsBus.read_action_positions()` 与 `PiperMotorsBus.write()` 仍是唯一反馈/命令边界；
部署层不修改 CAN 单位换算、SDK 命令格式和使能流程。

## 策略动作接口

- Piper 是六轴机械臂；joint action 中只有 6 个关节，额外的第 7 维是夹爪。
- `control_mode=joint`：7 维，`joint_1..joint_6`（rad）+ `gripper`（m）。
- `control_mode=eef`：8 维，`x,y,z`（m）+ `qx,qy,qz,qw` + `gripper`（m）。
- EEF 模式的 LeRobot `action_features` 会自动切换为 8 维 EEF schema。
- EEF 最终仍通过 IK 转为 6 关节目标，再进入现有 CAN writer；不存在单独的 EEF CAN 通道。

目前没有绑定具体 Piper URDF/MJCF。未配置 IK adapter 时，joint 模式可正常使用，EEF 执行会明确报错并拒绝写入；
不会用一个未经验证的模型生成真机命令。

接入后续 IK/FK solver：

```python
from lerobot.robots.piper_follower import CallablePiperIKAdapter

robot.set_ik_adapter(
    CallablePiperIKAdapter(
        solve_fn=lambda eef_pose7, current_joints6: solver.solve(eef_pose7, current_joints6),
        forward_fn=lambda joints6: solver.forward(joints6),
    )
)
```

`solve_fn` 返回 6 维关节弧度或 `None`；`forward_fn` 返回 7 维 `xyz + quat(xyzw)` 或 `None`。
joint 模式的直线度 retime 只有配置 FK 后才启用；没有 FK 时只跳过 retime，插值、异步融合、home 和夹爪过滤仍然生效。

## gRPC 异步推理配置

以下参数加到现有 Piper client 命令中。joint 模式尚未配置 FK 时应保持
`retime.enabled=false`；配置好 IK/FK 后才启用 EEF retime：

```bash
--robot.deployment.control_mode=eef \
--robot.deployment.async_buffer.enabled=true \
--robot.deployment.async_buffer.delay=3 \
--robot.deployment.async_buffer.exec_horizon=40 \
--robot.deployment.async_buffer.min_smooth_steps=8 \
--robot.deployment.chunk_upsample=2 \
--robot.deployment.retime.enabled=true \
--robot.deployment.gripper.enabled=true \
--robot.deployment.gripper.close_threshold=0.024 \
--robot.deployment.gripper.open_threshold=0.040 \
--robot.deployment.gripper.window_n=4 \
--robot.deployment.gripper.ratio=0.75 \
--robot.deployment.gripper.ramp_s=0.3
```

Piper 常用控制频率是 30 Hz，因此默认 `delay=3` 对应约 100 ms；应根据实际模型 P95 推理延迟调整。
`exec_horizon` 使用插值后的控制步数，必须大于 `delay`。如果 retime 后的实际 chunk 更短，buffer 会按实际长度提前触发，避免播放队列饿死。
启用 `chunk_upsample=N` 时，client 的 `fps` 也应设置成期望执行频率，确保每个插值点按该频率下发。

同步 chunk 模式可以使用 `ema_alpha < 1`；异步 buffer 模式按 Marvin main 由 cross-fade 管理 seam，
部署层会强制使用 `ema_alpha=1`，避免重复平滑。

RTC 入口保留 LeRobot 原有 RTC chunk merge，不再叠加第二个 `StreamActionBuffer`；Piper 的 EEF/IK、夹爪过滤、
最终关节 writer 和 hold/home 生命周期仍通过同一个部署 controller 执行。

## Home、停止和试运行

```bash
--robot.deployment.start_from=home \
--robot.deployment.stop_at=home \
--robot.deployment.home_position='[0,0,0,0,0,0,0]' \
--robot.deployment.home_interp_steps=10 \
--robot.deployment.home_hz=10 \
--robot.deployment.home_settle_s=1.5
```

- `start_from=current`：连接后保持当前位置，不执行启动 home。
- `stop_at=hold`：正常退出时原地保持一次，然后失能。
- `stop_at=home`：正常退出时先平滑回 home，再失能。
- 运行 async client 或 RTC 真机入口时，按 `s` 选择 hold 后退出，按 `h` 选择 smooth home 后退出，Ctrl+C 选择 hold。
- `disable_action_write=true`：部署层完成处理但不向 CAN writer 写动作，适合先检查日志和 shape。
- `max_joint_step_rad`：可选的最终逐周期关节步长限幅；默认 `None`，保持原先验证过的 driver 行为。

第一次上真机前应重新确认 `home_position`、夹爪实际行程与阈值，并先以低速、急停可达的方式验证。

## 暂未复用

- StreamPI 分支及其 transport/server：明确不在本次范围。
- Marvin 双臂 enable/mux、Apex mode/velocity HTTP API：Piper 是单臂 CAN/SDK，语义不匹配。
- Marvin 的 MuJoCo/Mink 模型和双臂 collision 配置：等待选定并验证 Piper URDF/MJCF 后，通过现有 IK adapter 接入。
- Marvin 相机/vlahost schema：保留 LeRobot 当前 camera/observation schema。
