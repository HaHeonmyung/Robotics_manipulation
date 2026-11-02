#!/bin/bash
# Run the submitted eval_command.txt lines verbatim (same checkout, paths as in the repo) and keep the logs.
cd ~/Robotics_manipulation/IsaacLab_RS
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
export PYTHONUNBUFFERED=1
R=coursework/toy2box-bc/results/final_repro
git rev-parse HEAD > $R/commit.txt
for S in 101 111; do
  ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --checkpoint ./coursework/toy2box-bc/final_weights/multi78_1000_seed${S}_epoch2000.pth --num_rollouts 100 --horizon 800 --seed 42 --headless > $R/play_seed${S}.log 2>&1 &
  sleep 40
done
wait
grep -H "Success rate" $R/play_seed101.log $R/play_seed111.log > $R/summary.txt
echo REPRO_DONE > $R/done
