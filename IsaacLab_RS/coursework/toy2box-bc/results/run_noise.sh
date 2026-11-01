#!/bin/bash
cd ~/Robotics_manipulation/IsaacLab_RS
R=coursework/toy2box-bc/results
while [ ! -e $R/rand750_done ]; do sleep 120; done
sleep 60
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
export PYTHONUNBUFFERED=1
mkdir -p $R/noise03_1000
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-Noise03-v0 --headless --num_envs 100 --generation_num_trials 1000 --seed 1 --input_file datasets/multi78_1000_annotated.hdf5 --output_file datasets/multi78_noise03_mimic_1000.hdf5 > $R/noise03_1000/02_generate.log 2>&1
coursework/toy2box-bc/run_baseline.sh datasets/multi78_noise03_mimic_1000.hdf5 noise03_1000 1000 --skip-mimic > $R/noise03_1000.log 2>&1
echo NOISE_DONE > $R/noise_done
