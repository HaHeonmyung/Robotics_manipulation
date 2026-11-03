#!/bin/bash
# Run the submitted eval_command.txt (relobs) verbatim with the stock play.py and keep the logs.
cd ~/Robotics_manipulation/IsaacLab_RS
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
export PYTHONUNBUFFERED=1
R=coursework/toy2box-bc/results/final_repro_relobs
git rev-parse HEAD > $R/commit.txt
for S in 101 111; do
  ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-RelObs-v0 --checkpoint ./coursework/toy2box-bc/final_weights/relobs_1000_seed${S}_epoch2000.pth --num_rollouts 100 --horizon 800 --seed 42 --headless > $R/play_seed${S}.log 2>&1 &
  sleep 60
done
wait
for S in 101 111; do grep -h "Successful trials\|Success rate" $R/play_seed$S.log | sed "s/^/seed $S: /"; done > $R/summary.txt
cat $R/commit.txt >> $R/summary.txt
echo DONE > $R/done
