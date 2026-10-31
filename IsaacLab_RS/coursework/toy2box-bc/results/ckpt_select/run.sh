#!/bin/bash
# checkpoint selection rule: validation seed 7, 50 rollouts, epochs 1200/1500/1800/2000, both training seeds
cd ~/Robotics_manipulation/IsaacLab_RS
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
export PYTHONUNBUFFERED=1
OUT=coursework/toy2box-bc/results/ckpt_select
for S in 101 111; do
  M=$(ls -d logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0/own20_1000_seed$S/*/models | tail -1)
  for E in 1200 1500 1800 2000; do
    ./isaaclab.sh -p coursework/toy2box-bc/play_diagnose.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
      --checkpoint $M/model_epoch_$E.pth --num_rollouts 50 --horizon 800 --seed 7 \
      --output $OUT/val_seed${S}_ep$E.json --headless > $OUT/val_seed${S}_ep$E.log 2>&1 &
  done
  wait
done
echo CKPT_DONE > $OUT/done
