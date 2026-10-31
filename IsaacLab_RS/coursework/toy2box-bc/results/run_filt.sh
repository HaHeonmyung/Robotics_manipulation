#!/bin/bash
# source-quality experiment: filtered set vs random-drop control, then checkpoint validation for multi78_1000
cd ~/Robotics_manipulation/IsaacLab_RS
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
export PYTHONUNBUFFERED=1
R=coursework/toy2box-bc/results
python -I coursework/toy2box-bc/select_demos.py datasets/multi78_filt.hdf5 datasets/multi78.hdf5:0,1,12,14,22,28,55,64 > $R/src_stats/build_filt.log 2>&1
python -I coursework/toy2box-bc/select_demos.py datasets/multi78_rand.hdf5 datasets/multi78.hdf5:5,33,38,49,51,53,62,65 > $R/src_stats/build_rand.log 2>&1
coursework/toy2box-bc/run_baseline.sh datasets/multi78_filt.hdf5 filt70_1000 1000 > $R/filt70_1000.log 2>&1
coursework/toy2box-bc/run_baseline.sh datasets/multi78_rand.hdf5 rand70_1000 1000 > $R/rand70_1000.log 2>&1
# checkpoint validation (seed 7, 50 rollouts) for multi78_1000, plus the missing own20 seed111 ep1800
OUT=$R/ckpt_select
for S in 101 111; do
  M=$(ls -d logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0/multi78_1000_seed$S/*/models | tail -1)
  for E in 1200 1500 1800 2000; do
    ./isaaclab.sh -p coursework/toy2box-bc/play_diagnose.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --checkpoint $M/model_epoch_$E.pth --num_rollouts 50 --horizon 800 --seed 7 --output $OUT/m78_val_seed${S}_ep$E.json --headless > $OUT/m78_val_seed${S}_ep$E.log 2>&1 &
    sleep 40
  done
  wait
done
M=$(ls -d logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0/own20_1000_seed111/*/models | tail -1)
./isaaclab.sh -p coursework/toy2box-bc/play_diagnose.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --checkpoint $M/model_epoch_1800.pth --num_rollouts 50 --horizon 800 --seed 7 --output $OUT/val_seed111_ep1800.json --headless > $OUT/val_seed111_ep1800.log 2>&1
echo FILT_DONE > $R/filt_done
