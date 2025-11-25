# piper_lerobot数据集v3格式

[huggingface文档](huggingface.co/docs/lerobot)

## 环境创建

````
conda create -y -n lerobot_v21 python=3.10
conda activate lerobot_v21
conda install ffmpeg=7.1.1
git clone https://github.com/jokeru8/piper_lerobot_v21.git
cd piper_lerobot_v21
pip install -e .
````

## 测试相机

两个相机不能从同一个扩展坞连接电脑

````
sudo apt install guvcview    #安装Guvcview
guvcview --device=/dev/video0  # 测试wrist相机
guvcview --device=/dev/video6  # 测试ground相机
````

## 连接机械臂

"3-7.1:1.0"根据输出的can端口号改为自己的

````
conda activate lerobot
bash find_all_can_port.sh
bash can_activate.sh can_master 1000000 "3-7.1:1.0"
bash can_activate.sh can_follower 1000000 "3-7.2:1.0"
````

## 遥操作

````
lerobot-teleoperate \
    --robot.type=piper_follower \
    --robot.id=my_follower_arm \
    --teleop.type=piper_leader \
    --teleop.id=my_leader_arm \
    --display_data=true
````


## 采集数据集

/dev/video0等参数改为自己对应的端口

````
lerobot-record \
  --robot.type=piper_follower \
  --robot.cameras='{
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/video0",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0,
    },
    "ground": {
      "type": "opencv",
      "index_or_path": "/dev/video6",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0,
    }
  }' \
  --teleop.type=piper_leader \
  --display_data=true \
  --dataset.repo_id=jokeru/record-test \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=5 \
  --dataset.single_task="test"
````

### 其他可选参数:
````
  --dataset.episode_time_s=60 每个数据记录episode的持续时间(默认60秒)，可提前结束。
  --dataset.reset_time_s=60 每episode之后重置环境的时长(默认60秒)。
  --dataset.num_episodes=50 记录的总episode数(默认50)。
````

数据会保存到~/.cache/huggingface/lerobot/jokeru

录制过程中使用键盘控制

### 使用键盘快捷键控制数据采集

按右箭头(→):提前停止当前事件,或重置时间,然后切换到下一个。

按左箭头(→):取消当前事件并重新录制。

按ESC:立即停止会话,编码视频并上传数据集。

## 可视化数据集

````
python src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id jokeru/record1 \
    --episode-index 0
````

## 全部失能

````
python utils/teleop_disable.py
````