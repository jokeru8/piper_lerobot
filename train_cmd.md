### pi

多卡训练
````
CUDA_VISIBLE_DEVICES=3,4,5,6,7 nohup accelerate launch --num_processes=5 \
  src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=jokeru/pick_and_place \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_pick_and_place \
    --job_name=pi05_pick_and_place \
    --policy.repo_id=jokeru/pi05_pick_and_place \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --steps=400000 \
    --policy.device=cuda \
    --batch_size=1 > outputs/logs/pi05_pick_and_place.log 2>&1 &
````