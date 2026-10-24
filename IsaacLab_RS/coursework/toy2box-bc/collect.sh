#!/bin/bash
# Record human demos by keyboard into datasets/team{TEAM}_{NAME}_s{SESSION}.hdf5
# Usage (from IsaacLab_RS):  coursework/toy2box-bc/collect.sh <team> <name> <session> [num_demos=5]
# Refuses to overwrite: record_demos.py opens the file in "w" mode and would wipe it.
set -e
cd "$(dirname "$0")/../.."
TEAM=${1:?team number, e.g. 03}; NAME=${2:?operator name}; SESSION=${3:?session number}; N=${4:-5}
OUT=./datasets/team${TEAM}_${NAME}_s${SESSION}.hdf5
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT (use a new session number)"; exit 1; fi
mkdir -p datasets
echo "recording $N successful demos -> $OUT   (K gripper, R discard, W/S A/D Q/E move)"
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
    --dataset_file "$OUT" --num_demos "$N" --teleop_device keyboard
ls -lh "$OUT"
./isaaclab.sh -p coursework/toy2box-bc/summarize_demos.py "$OUT"
