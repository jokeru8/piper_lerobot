录制得到的图像特征名为 `observation.images.top`、`observation.images.wrist`、`observation.images.down`（不再使用 `observation.images.ground`）。

## Piper 从臂遥操作（平滑 + 方案 B 记录 `action`）

- **关节单位**：`observation` 中机械臂关节与 `write()` 一致，均为 **弧度**（夹爪为米行程量级）；由 SDK 读数统一转换。
- **遥操作平滑**：从臂在 `send_action` 内对主臂目标做 EMA，抑制主臂高频抖动。CLI 可调 `--robot.teleop_joint_alpha`、`--robot.teleop_gripper_alpha`（0–1，越小越平滑、延迟越大；`1.0` 等价于关闭）。
- **方案 B（默认开启）**：`--robot.record_action_from_follower=true` 时，写入数据集的 **`action` 为 `send_action` 之后从臂编码器读回的关节**（非主臂原始读数）。**时间语义**：同一帧内 `observation` 在发送指令**之前**采集，`action` 在发送**之后**采集，因此 `action` 接近「执行本步指令后的从臂位形」，与标准 MDP「同一时刻的 \((s,a)\)」略有错位；训练时按此一致使用即可。若需恢复旧行为（`action` = 主臂经 pipeline 的值），设 `--robot.record_action_from_follower=false`。

# record_apple
````
lerobot-record \
  --robot.type=piper_follower \
  --robot.cameras='{
    "top": {
      "type": "opencv",
      "index_or_path": "/dev/video_top",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": 90
    },
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/video_wrist",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0
    },
    "down": {
      "type": "opencv",
      "index_or_path": "/dev/video_down",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0
    }
  }' \
  --teleop.type=piper_leader \
  --display_data=true \
  --dataset.reset_time_s=5 \
  --dataset.repo_id=jokeru/record_p3_apple \
  --dataset.push_to_hub=true \
  --dataset.num_episodes=50 \
  --dataset.single_task="Pick up the apple and put it into the basket."
````
  
上传

````
hf upload jokeru/record_apple ~/.cache/huggingface/lerobot/jokeru/record_apple \
  --repo-type dataset \
  --revision "v3.0" 
````

# record_orange
````
lerobot-record \
  --robot.type=piper_follower \
  --robot.cameras='{
    "top": {
      "type": "opencv",
      "index_or_path": "/dev/video_top",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": 90
    },
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/video_wrist",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0
    },
    "down": {
      "type": "opencv",
      "index_or_path": "/dev/video_down",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0
    }
  }' \
  --teleop.type=piper_leader \
  --display_data=true \
  --dataset.reset_time_s=5 \
  --dataset.repo_id=jokeru/record_p3_orange \
  --dataset.push_to_hub=true \
  --dataset.num_episodes=50 \
  --dataset.single_task="Pick up the orange and put it into the basket."
````
  
上传

````
hf upload jokeru/record_p3_apple ~/.cache/huggingface/lerobot/jokeru/record_p3_apple \
  --repo-type dataset \
  --revision "v3.0" 
````

# record_banana
````
lerobot-record \
  --robot.type=piper_follower \
  --robot.cameras='{
    "top": {
      "type": "opencv",
      "index_or_path": "/dev/video_top",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": 90
    },
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/video_wrist",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0
    },
    "down": {
      "type": "opencv",
      "index_or_path": "/dev/video_down",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0
    }
  }' \
  --teleop.type=piper_leader \
  --display_data=true \
  --dataset.reset_time_s=5 \
  --dataset.repo_id=jokeru/record_banana \
  --dataset.push_to_hub=true \
  --dataset.num_episodes=50 \
  --dataset.single_task="Pick up the banana and put it into the basket."
````

上传

````
hf upload jokeru/record_banana ~/.cache/huggingface/lerobot/jokeru/record_banana \
  --repo-type dataset \
  --revision "v3.0" 
````

# record_watermelon
````
lerobot-record \
  --robot.type=piper_follower \
  --robot.cameras='{
    "top": {
      "type": "opencv",
      "index_or_path": "/dev/video_top",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": 90
    },
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/video_wrist",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0
    },
    "down": {
      "type": "opencv",
      "index_or_path": "/dev/video_down",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0
    }
  }' \
  --teleop.type=piper_leader \
  --display_data=true \
  --dataset.reset_time_s=5 \
  --dataset.repo_id=jokeru/record_watermelon \
  --dataset.push_to_hub=true \
  --dataset.num_episodes=50 \
  --dataset.single_task="Pick up the watermelon and put it into the basket."
````

上传

````
hf upload jokeru/record_watermelon ~/.cache/huggingface/lerobot/jokeru/record_watermelon \
  --repo-type dataset \
  --revision "v3.0" 
````

# record_tape
````
lerobot-record \
  --robot.type=piper_follower \
  --robot.cameras='{
    "top": {
      "type": "opencv",
      "index_or_path": "/dev/video_top",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": 90
    },
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/video_wrist",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0
    },
    "down": {
      "type": "opencv",
      "index_or_path": "/dev/video_down",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0
    }
  }' \
  --teleop.type=piper_leader \
  --display_data=true \
  --dataset.reset_time_s=5 \
  --dataset.repo_id=jokeru/record_tape \
  --dataset.push_to_hub=true \
  --dataset.num_episodes=50 \
  --dataset.single_task="Pick up the tape and put it into the basket."
````

上传

````
hf upload jokeru/record_tape ~/.cache/huggingface/lerobot/jokeru/record_tape \
  --repo-type dataset \
  --revision "v3.0" 
````