#!/bin/bash
# After rand500: (1) multi78 with a second Mimic generation seed -> train/eval (run-variance estimate)
#                (2) final weights evaluated at a second eval seed (7), 100 rollouts (robustness)
cd ~/Robotics_manipulation/IsaacLab_RS
R=coursework/toy2box-bc/results
while [ ! -e $R/after_filt_done ]; do sleep 120; done
sleep 60
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
export PYTHONUNBUFFERED=1
# (0) grasp-window oversampling: same Mimic 1000 + grasp-window copies x2 (sample share 15% -> 35%)
coursework/toy2box-bc/run_baseline.sh datasets/multi78_graspwin_mimic_1000.hdf5 graspwin_1000 1000 --skip-mimic > $R/graspwin_1000.log 2>&1
mkdir -p $R/multi78_rep2
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 --headless --num_envs 100 --generation_num_trials 1000 --seed 2 --input_file datasets/multi78_1000_annotated.hdf5 --output_file datasets/multi78_rep2_mimic_1000.hdf5 > $R/multi78_rep2/02_generate.log 2>&1
coursework/toy2box-bc/run_baseline.sh datasets/multi78_rep2_mimic_1000.hdf5 multi78_rep2 1000 --skip-mimic > $R/multi78_rep2.log 2>&1
mkdir -p $R/final_seed7
for S in 101 111; do
  ./isaaclab.sh -p coursework/toy2box-bc/play_diagnose.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --checkpoint coursework/toy2box-bc/final_weights/multi78_1000_seed${S}_epoch2000.pth --num_rollouts 100 --horizon 800 --seed 7 --output $R/final_seed7/eval_seed${S}.json --headless > $R/final_seed7/eval_seed${S}.log 2>&1 &
  sleep 40
done
wait
echo REP_DONE > $R/rep_done
