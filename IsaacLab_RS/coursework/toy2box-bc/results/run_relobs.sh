#!/bin/bash
cd ~/Robotics_manipulation/IsaacLab_RS
R=coursework/toy2box-bc/results
while [ ! -e $R/final_repro/done ]; do sleep 60; done
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
export PYTHONUNBUFFERED=1
TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-RelObs-v0
mkdir -p $R/relobs_1000
for S in 101 111; do
  ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task $TASK --algo bc --dataset datasets/multi78_relobs_mimic_1000.hdf5 --seed $S --name relobs_1000_seed$S > $R/relobs_1000/03_train_seed$S.log 2>&1
done
for S in 101 111; do
  CK=$(ls -d logs/robomimic/$TASK/relobs_1000_seed$S/*/models | sort | tail -1)/model_epoch_2000.pth
  echo "seed $S checkpoint: $CK" >> $R/relobs_1000/summary.txt
  ./isaaclab.sh -p coursework/toy2box-bc/play_diagnose.py --task $TASK --checkpoint $CK --num_rollouts 100 --horizon 800 --seed 42 --output $R/relobs_1000/eval_seed$S.json --save_traj $R/relobs_1000/eval_seed${S}_traj.npz --headless > $R/relobs_1000/04_eval_seed$S.log 2>&1 &
  sleep 40
done
wait
python3 coursework/toy2box-bc/aggregate.py $R/relobs_1000/eval_seed101.json $R/relobs_1000/eval_seed111.json | tee -a $R/relobs_1000/summary.txt
echo RELOBS_DONE > $R/relobs_done
