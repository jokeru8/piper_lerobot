
#环境创建

‘conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg=7.1.1
git clone https://github.com/jokeru8/piper_lerobot.git
cd piper_lerobot
pip install -e .‘
s

#连接机械臂

‘bash find_all_can_port.sh‘
"3-7.1:1.0"根据输出的can端口号改为自己的
‘bash can_activate.sh can_master 1000000 "3-7.1:1.0"
bash can_activate.sh can_follower 1000000 "3-7.2:1.0"‘


#遥操作
‘lerobot-teleoperate \
    --robot.type=piper_follower \
    --robot.id=my_follower_arm \
    --teleop.type=piper_leader \
    --teleop.id=my_leader_arm \
    --display_data=true‘

#采集数据集
/dev/video4等参数改为自己对应的端口

’lerobot-record \
  --robot.type=piper_follower \
  --robot.cameras='{
    "wrist": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30, "color_mode": "rgb"},
    "ground": {"type": "opencv", "index_or_path": "/dev/video6", "width": 640, "height": 480, "fps": 30, "color_mode": "rgb"}
  }' \
  --teleop.type=piper_leader \
  --display_data=true \
  --dataset.repo_id=jokeru/record-test \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=5 \
  --dataset.single_task="test"‘