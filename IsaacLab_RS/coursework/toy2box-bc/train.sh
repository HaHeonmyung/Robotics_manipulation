#!/bin/bash
# Train BC-RNN with the TA config on one dataset and one seed.
# Usage (from IsaacLab_RS):  coursework/toy2box-bc/train.sh <dataset.hdf5> <seed> <run_name> [extra train.py args]
# Writes logs/robomimic/<task>/<run_name>_seed<seed>/<timestamp>/ and prints the final checkpoint path.
set -e
cd "$(dirname "$0")/../.."
DATA=${1:?dataset}; SEED=${2:?seed}; RUN=${3:?run name}; shift 3
TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
NAME=${RUN}_seed${SEED}
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
export PYTHONUNBUFFERED=1
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
    --task $TASK --algo bc --dataset "$DATA" --seed "$SEED" --name "$NAME" "$@"
CK=$(ls -d logs/robomimic/$TASK/$NAME/*/models | sort | tail -1)/model_epoch_2000.pth
echo "FINAL_CHECKPOINT=$CK"
