录制得到的图像特征名为 `observation.images.top`、`observation.images.wrist`、`observation.images.down`（不再使用 `observation.images.ground`）。

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