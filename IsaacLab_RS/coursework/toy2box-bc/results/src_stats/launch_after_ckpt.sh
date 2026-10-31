#!/bin/bash
# wait until the checkpoint-selection evals are gone (they share /tmp/agibot.urdf with RMPFlow init), then generate
cd ~/Robotics_manipulation/IsaacLab_RS
while [ ! -e coursework/toy2box-bc/results/ckpt_select/done ]; do sleep 60; done
sleep 30
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
export PYTHONUNBUFFERED=1
export MIMIC_SRC_LOG=$(pwd)/coursework/toy2box-bc/results/src_stats/src_log.jsonl
rm -f $MIMIC_SRC_LOG datasets/src_stats78_mimic_400*.hdf5
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 --headless --num_envs 100 --generation_num_trials 400 --input_file datasets/multi78_1000_annotated.hdf5 --output_file datasets/src_stats78_mimic_400.hdf5 > coursework/toy2box-bc/results/src_stats/02_generate.log 2>&1
echo GEN_DONE > coursework/toy2box-bc/results/src_stats/done
