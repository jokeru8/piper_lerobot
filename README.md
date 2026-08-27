<div align="center">

# 🤖 Piper LeRobot

**让 AgileX Piper 六轴机械臂使用 LeRobot 完成遥操作、数据采集、训练与真机推理**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LeRobot](https://img.shields.io/badge/LeRobot-0.4.2-FFD21E)](https://github.com/huggingface/lerobot)
[![Robot](https://img.shields.io/badge/Robot-Piper_6DOF%2BGripper-F97316)](#动作定义)
[![License](https://img.shields.io/badge/License-Apache--2.0-22C55E)](LICENSE)
[![Deployment](https://img.shields.io/badge/Deployment-Joint_Ready-16A34A)](#真机策略执行)

[快速安装](#安装) · [硬件准备](#硬件准备) · [数据采集](#采集数据集) · [训练](#训练) · [真机推理](#真机策略执行) · [部署说明](docs/piper_real_robot_deployment.md)

</div>

Piper LeRobot 是面向 AgileX Piper 的完整 LeRobot 集成。项目保留了已验证的 Piper SDK / CAN
控制通道，并在外围加入参考 OpenPI Marvin `main` 的异步推理、轨迹平滑和真机生命周期能力。

## 项目能力

| 工作流 | 支持内容 | 状态 |
| --- | --- | :---: |
| 🎮 主从遥操作 | Piper leader → Piper follower | ✅ |
| 🎥 数据采集 | 三路 OpenCV 相机、LeRobot Dataset v3 | ✅ |
| 🧠 模型训练 | ACT、PI0 / PI0.5、多卡训练 | ✅ |
| ⚡ 真机推理 | 本地 RTC、gRPC 远程异步推理 | ✅ |
| 🧰 部署工具 | retime、插值、EMA、chunk 融合、夹爪过滤、平滑 home | ✅ |
| 🧾 EEF / IK / FK | 8 维 EEF 端口与可插拔 adapter | 🟡 待接入真实模型 |

> [!CAUTION]
> 这是会向真实机械臂发送命令的项目。第一次运行前请确认 CAN 接口、关节零位、
> `home_position` 和夹爪行程，降低速度并确保急停可达。仓库目前没有可用真机做自动化回归，
> 新增部署层完成了无硬件测试，但仍需要维护者在受控环境中完成首次真机验收。

本仓库的异步部署实现只参考 OpenPI Marvin 的 `main` 分支，不包含 StreamPI transport 或服务。

### 动作定义

| 模式 | 动作维度 | 定义 |
| --- | ---: | --- |
| `joint` | 7 | `joint_1 ... joint_6`（rad）+ `gripper`（m） |
| `eef` | 8 | `x y z`（m）+ `qx qy qz qw` + `gripper`（m） |

代码和文档中出现的 7 维 joint action 指的是“6 轴 + 夹爪”，不是七轴机械臂。

### 部署数据流

```mermaid
flowchart LR
    OBS["📷 相机 + 机械臂状态"] --> POLICY["🧠 LeRobot Policy"]
    POLICY --> PIPE["⚡ retime · 插值 · EMA<br/>异步 chunk 融合 · 夹爪过滤"]
    PIPE --> JOINT["7D Joint<br/>6 关节 + 夹爪"]
    PIPE -.->|8D EEF| IK["IK Adapter"]
    IK -.->|6 关节| JOINT
    JOINT --> CAN["🤖 Piper SDK / CAN"]

    classDef ready fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef core fill:#eff6ff,stroke:#2563eb,color:#1e3a8a
    classDef pending fill:#fef3c7,stroke:#d97706,color:#78350f
    class OBS,POLICY,PIPE core
    class JOINT,CAN ready
    class IK pending
```

EEF 路径已预留统一端口，图中虚线部分需要在接入并验证 Piper URDF/MJCF 后启用。

## 安装

推荐 Linux、Python 3.10，并使用独立 Conda 环境：

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install -c conda-forge ffmpeg=7.1.1 -y

git clone https://github.com/jokeru8/piper_lerobot.git
cd piper_lerobot
pip install -e ".[piper]"
```

按用途安装额外依赖：

```bash
# PI0 / PI0.5
pip install -e ".[piper,pi]"

# gRPC 远程异步推理
pip install -e ".[piper,async]"

# 开发与测试
pip install -e ".[piper,test,dev]"
```

`piper` extra 安装 `piper-sdk` 和 `python-can`。如果环境中已手动安装 AgileX SDK，
也可以只执行 `pip install -e .`。

## 硬件准备

### 1. 配置 CAN

代码默认使用：

- `can_master`：Piper leader。
- `can_follower`：Piper follower。
- bitrate：`1000000`。

先查找 USB-CAN 端口：

```bash
bash find_all_can_port.sh
```

将下面的 USB 地址替换为本机输出：

```bash
bash can_activate.sh can_master 1000000 "1-8.1:1.0"
bash can_activate.sh can_follower 1000000 "1-8.4.2:1.0"
```

检查接口：

```bash
ip -details link show can_master
ip -details link show can_follower
```

### 2. 配置相机

建议通过 udev 创建稳定设备名：

- `/dev/video_top`
- `/dev/video_wrist`
- `/dev/video_down`

三个相机尽量不要连接同一个 USB 扩展坞。可以先用 `guvcview` 验证：

```bash
sudo apt install guvcview
guvcview --device=/dev/video_top
guvcview --device=/dev/video_wrist
guvcview --device=/dev/video_down
```

后续命令复用以下相机配置：

```bash
PIPER_CAMERAS='{
  "top": {
    "type": "opencv", "index_or_path": "/dev/video_top",
    "width": 480, "height": 640, "fps": 30, "rotation": 90
  },
  "wrist": {
    "type": "opencv", "index_or_path": "/dev/video_wrist",
    "width": 640, "height": 480, "fps": 30, "rotation": 0
  },
  "down": {
    "type": "opencv", "index_or_path": "/dev/video_down",
    "width": 640, "height": 480, "fps": 30, "rotation": 0
  }
}'
```

## 遥操作

不使用相机的最小测试：

```bash
lerobot-teleoperate \
  --robot.type=piper_follower \
  --robot.id=my_follower_arm \
  --robot.cameras={} \
  --robot.deployment.start_from=current \
  --robot.deployment.stop_at=hold \
  --teleop.type=piper_leader \
  --teleop.id=my_leader_arm \
  --display_data=false
```

确认 home 配置安全后，可以将 `start_from` / `stop_at` 改成 `home`。默认 home action 是：

```text
[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
[0,       0,       0,       0,       0,       0,       0      ]
```

## 采集数据集

```bash
lerobot-record \
  --robot.type=piper_follower \
  --robot.cameras="$PIPER_CAMERAS" \
  --robot.deployment.start_from=current \
  --robot.deployment.stop_at=hold \
  --teleop.type=piper_leader \
  --display_data=true \
  --dataset.repo_id=jokeru/record2 \
  --dataset.single_task="Pick up the apple and put it into the basket." \
  --dataset.fps=30 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=5 \
  --dataset.num_episodes=20 \
  --dataset.push_to_hub=false
```

录制快捷键：

- `→`：提前结束当前 episode / reset，进入下一阶段。
- `←`：取消当前 episode 并重新录制。
- `Esc`：结束录制并执行视频编码及后续保存。

默认数据目录位于 `~/.cache/huggingface/lerobot/<repo_id>`。

### 数据集可视化

```bash
lerobot-dataset-viz \
  --repo-id jokeru/record2 \
  --episode-index 0
```

### 合并数据集

被合并的数据集必须具有完全一致的 features：

```bash
lerobot-edit-dataset \
  --repo_id jokeru/pick_and_place \
  --operation.type merge \
  --operation.repo_ids "['jokeru/record_apple', 'jokeru/record_banana']" \
  --push_to_hub true
```

## Hugging Face Hub

```bash
hf auth login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential
hf auth whoami
```

国内网络可按需设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 训练

### ACT

```bash
lerobot-train \
  --dataset.repo_id=jokeru/record2 \
  --policy.type=act \
  --output_dir=outputs/train/record2 \
  --job_name=act_finetune_pick_apple \
  --policy.device=cuda \
  --policy.repo_id=jokeru/act_pick_apple \
  --wandb.enable=false \
  --batch_size=128 \
  --steps=12000 \
  --num_workers=16
```

`batch_size`、`num_workers` 和训练步数需要根据机器配置调整。

### PI0.5

```bash
lerobot-train \
  --dataset.repo_id=jokeru/record2 \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_libero \
  --policy.repo_id=jokeru/pi05_pick_and_place \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --output_dir=outputs/pi05_training \
  --job_name=pi05_training \
  --wandb.enable=false \
  --batch_size=32 \
  --steps=3000
```

多卡训练示例：

```bash
accelerate launch --num_processes=8 \
  src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=jokeru/record2 \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_libero \
  --policy.repo_id=jokeru/pi05_pick_and_place \
  --policy.device=cuda \
  --output_dir=outputs/pi05_training \
  --job_name=pi05_training \
  --wandb.enable=false \
  --batch_size=32 \
  --steps=3000
```

## 真机策略执行

### 使用 `lerobot-record` 执行策略

该方式可以同时保存真机评估 episode：

```bash
lerobot-record \
  --robot.type=piper_follower \
  --robot.cameras="$PIPER_CAMERAS" \
  --robot.deployment.start_from=current \
  --robot.deployment.stop_at=hold \
  --policy.path=jokeru/act_pick_apple \
  --display_data=true \
  --dataset.repo_id=jokeru/eval_act_pick_and_place \
  --dataset.single_task="Pick up the apple and put it into the basket." \
  --dataset.num_episodes=3 \
  --dataset.episode_time_s=120 \
  --dataset.push_to_hub=false
```

### 本地 RTC

RTC demo 支持 SmolVLA、PI0 和 PI0.5：

```bash
python examples/rtc/eval_with_real_robot.py \
  --policy.path=jokeru/pi05_pick_and_place \
  --policy.device=cuda \
  --robot.type=piper_follower \
  --robot.cameras="$PIPER_CAMERAS" \
  --robot.deployment.start_from=current \
  --robot.deployment.stop_at=hold \
  --task="Pick up the apple and put it into the basket." \
  --duration=120 \
  --fps=30 \
  --device=cuda \
  --rtc.enabled=true \
  --rtc.execution_horizon=5 \
  --action_queue_size_to_get_new_actions=30
```

RTC 保留 LeRobot 自带的 action queue merge，不叠加第二个异步 buffer；Piper 的夹爪过滤、
EEF/IK 执行入口和退出 home/hold 仍通过统一部署层处理。

### 远程异步推理

服务端和真机客户端应使用同一版本的仓库。

服务端：

```bash
CUDA_VISIBLE_DEVICES=1 python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 \
  --port=8080 \
  --fps=30 \
  --inference_latency=0.10 \
  --obs_queue_timeout=1
```

需要 SSH 转发时，在真机客户端运行：

```bash
ssh -L 8080:127.0.0.1:8080 USER@SERVER -N
nc -zv 127.0.0.1 8080
```

真机客户端：

```bash
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=piper_follower \
  --robot.cameras="$PIPER_CAMERAS" \
  --robot.deployment.control_mode=joint \
  --robot.deployment.start_from=current \
  --robot.deployment.stop_at=hold \
  --robot.deployment.async_buffer.enabled=true \
  --robot.deployment.async_buffer.delay=3 \
  --robot.deployment.async_buffer.exec_horizon=40 \
  --robot.deployment.async_buffer.min_smooth_steps=8 \
  --robot.deployment.gripper.enabled=true \
  --robot.deployment.gripper.close_threshold=0.024 \
  --robot.deployment.gripper.open_threshold=0.040 \
  --task="Pick up the apple and put it into the basket." \
  --policy_type=pi05 \
  --pretrained_name_or_path=jokeru/pi05_pick_and_place \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --fps=30 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=false
```

`delay` 和 `exec_horizon` 的单位都是控制步。30 Hz 下 `delay=3` 约为 100 ms；
应根据实际 P95 推理耗时调整。启用异步 buffer 后，chunk seam 由 cross-fade 管理，
部署层会关闭额外的跨 chunk EMA，避免重复平滑。

运行 async client 或 RTC demo 时：

- `s`：停止策略并原地 hold。
- `h`：停止策略、平滑回 home 后退出。
- `Ctrl+C`：原地 hold 后退出。

完整的部署参数、retime 原理和 EEF adapter 示例见
[Piper 真机部署说明](docs/piper_real_robot_deployment.md)。

## EEF / IK 状态

EEF action 端口已经接入，但仓库尚未选择和验证 Piper URDF/MJCF，因此没有默认 IK solver。

- joint 模式可以直接运行。
- EEF 模式必须先调用 `robot.set_ik_adapter(...)`。
- 未配置 solver 时，EEF action 会明确报错，且不会写入 CAN。
- joint 模式的 EEF 直线度 retime 需要 FK；没有 FK 时会安全跳过 retime。
- EEF policy 必须输出 8 维动作，不能直接使用训练为 7 维 joint action 的 checkpoint。

## 测试

无需真机的部署层测试：

```bash
pytest -q tests/robots/test_piper_deployment.py
python -m compileall -q src/lerobot/robots/piper_follower
```

测试覆盖 6DOF + gripper shape、EEF 四元数插值、retime、异步 buffer、夹爪迟滞、
平滑 home 和可插拔 IK adapter。真机测试应在限速和急停保护下单独进行。

## 项目结构

```text
src/lerobot/motors/piper/                 Piper SDK / CAN 边界
src/lerobot/robots/piper_follower/        follower 与真机部署层
src/lerobot/teleoperators/piper_leader/   leader 遥操作入口
src/lerobot/async_inference/              gRPC 异步推理
examples/rtc/                              本地 RTC 示例
docs/piper_real_robot_deployment.md        部署设计与参数说明
```

## 贡献

欢迎提交 issue 和 pull request。涉及真机控制的改动请说明：

- Piper 固件和 `piper-sdk` 版本。
- CAN 硬件、控制频率与相机配置。
- 是否经过真机验证以及采用的限速/急停措施。
- action schema 是 7 维 joint 还是 8 维 EEF。

## License 与致谢

本项目基于 [Hugging Face LeRobot](https://github.com/huggingface/lerobot)，使用
[Apache License 2.0](LICENSE)开源。上游归属和 Piper 扩展的版权说明见 [NOTICE](NOTICE)。
