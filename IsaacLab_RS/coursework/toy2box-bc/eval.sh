#!/bin/bash
# Evaluate one checkpoint the way the assignment fixes it: seed 42, 100 rollouts, horizon 800,
# and classify every failure by stage. Prints the success rate and stage counts.
# Usage (from IsaacLab_RS):  coursework/toy2box-bc/eval.sh <checkpoint.pth> <out.json> [seed=42]
set -e
cd "$(dirname "$0")/../.."
CK=${1:?checkpoint}; OUT=${2:?output json}; SEED=${3:-42}
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
export PYTHONUNBUFFERED=1
./isaaclab.sh -p coursework/toy2box-bc/play_diagnose.py \
    --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
    --checkpoint "$CK" --num_rollouts 100 --horizon 800 --seed "$SEED" \
    --output "$OUT" --save_traj "${OUT%.json}_traj.npz" --headless
