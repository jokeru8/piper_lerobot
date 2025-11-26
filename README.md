# piper_lerobot数据集v3格式

[huggingface文档](http://huggingface.co/docs/lerobot)

## 1.环境创建

````
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install -c conda-forge ffmpeg=7.1.1 -y
pip install transformers --upgrade
git clone https://github.com/jokeru8/piper_lerobot.git
cd piper_lerobot
pip install -e .
````

## 2.测试相机

注意两个相机不能从同一个扩展坞连接电脑,否则可能读取会出问题

````
sudo apt install guvcview    #安装Guvcview
guvcview --device=/dev/video6  # 测试wrist相机
guvcview --device=/dev/video0  # 测试ground相机
````

## 3.连接机械臂

"3-7.1:1.0"根据输出的can端口号改为自己的

````
conda activate lerobot
bash find_all_can_port.sh
bash can_activate.sh can_master 1000000 "3-7.1:1.0"
bash can_activate.sh can_follower 1000000 "3-7.2:1.0"
````

## 4.遥操作

````
lerobot-teleoperate \
    --robot.type=piper_follower \
    --robot.id=my_follower_arm \
    --teleop.type=piper_leader \
    --teleop.id=my_leader_arm \
    --display_data=true
````
## 5.登陆huggingface

设置国内镜像加速
````
export HF_ENDPOINT=https://hf-mirror.com
````

通过运行此命令将您的令牌添加到CLI:

````
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
````

验证登录
````
HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER
````

上传数据集到huggingface
````
hf upload jokeru/record2 ~/.cache/huggingface/lerobot/jokeru/record2 --repo-type dataset --revision "v3.0" 
````



## 6.采集数据集

/dev/video0等参数改为自己对应的端口

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
      "index_or_path": "/dev/video4",
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

  ### 6.1其他可选参数:
  ````
    --dataset.episode_time_s=60 每个数据记录episode的持续时间(默认60秒)，可提前结束。
    --dataset.reset_time_s=60 每episode之后重置环境的时长(默认60秒)。
    --dataset.num_episodes=50 记录的总episode数(默认50)。
  ````

  数据会保存到~/.cache/huggingface/lerobot/jokeru

  录制过程中使用键盘控制

  ### 6.2使用键盘快捷键控制数据采集

  按右箭头(→):提前停止当前事件,或重置时间,然后切换到下一个。

  按左箭头(→):取消当前事件并重新录制。

  按ESC:立即停止会话,编码视频并上传数据集。

## 7.可视化数据集

````
python src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id jokeru/record1 \
    --episode-index 0
```` 
这种方法只能看一条episode

也可以用vlc直接看mp4文件
````
vlc *.mp4
````


## 8.全部失能

````
python utils/teleop_disable.py
````

## 9.ACT
### 训练ACT

num_workers、batch_size、steps 等训练参数参照自己的设备
````
lerobot-train \
  --dataset.repo_id=jokeru/record2 \
  --policy.type=act \
  --output_dir=outputs/train/record2 \
  --job_name=act_finetune_pick_apple \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=jokeru/act_pick_apple \
  --batch_size=128 \
  --steps=1_000 \
  --num_workers=128
````

### 测试ACT

#### 仿真环境中测试

````
lerobot-eval \
    --policy.path=jokeru/act_policy \
    --env.type=your_env \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
````

#### 真机测试
[lerobot huggingface真机文档](https://huggingface.co/docs/lerobot/il_robots)

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
      "index_or_path": "/dev/video4",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0,
    }
  }' \
  --display_data=true \
  --dataset.repo_id=jokeru/eval_act_pick_apple \
  --dataset.num_episodes=1 \
  --dataset.single_task="Pick up the apple and put it into the basket." \
  --policy.path=jokeru/act_pick_apple
````

