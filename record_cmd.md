# 登陆huggingface

通过运行此命令将您的令牌添加到CLI:

````
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
````


````
HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER
````


# 示例 Put round yellow tape into the brown box.
````
lerobot-record \
  --robot.type=piper_follower \
  --robot.cameras='{
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/video6",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": 90,
    },
    "ground": {
      "type": "opencv",
      "index_or_path": "/dev/video0",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0,
    }
  }' \
  --teleop.type=piper_leader \
  --display_data=true \
  --dataset.reset_time_s=5 \
  --dataset.repo_id=jokeru/record1 \
  --dataset.push_to_hub=true \
  --dataset.num_episodes=2 \
  --dataset.single_task="Pick up round yellow tape and place it into the brown box."
````

# 清除记录

````
rm -r ~/.cache/huggingface/lerobot/jokeru/record1
````

# 可视化数据集

````
python src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id jokeru/record2 \
    --episode-index 0
````