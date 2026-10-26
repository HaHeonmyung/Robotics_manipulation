#!/bin/bash
# Same recording as collect.sh, plus Top / Front / Side viewports opened automatically.
# The recorder is the unmodified scripts/tools/record_demos.py; open_views.py only adds
# display windows through Kit's --exec, so the saved data is identical to collect.sh.
# Usage (from IsaacLab_RS):  coursework/toy2box-bc/collect_v2.sh <team> <name> <session> [num_demos=5]
set -e
cd "$(dirname "$0")/../.."
TEAM=${1:?team number, e.g. 03}; NAME=${2:?operator name}; SESSION=${3:?session number}; N=${4:-5}
OUT=./datasets/team${TEAM}_${NAME}_s${SESSION}.hdf5
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT (use a new session number)"; exit 1; fi
mkdir -p datasets
VIEWS=$(pwd)/coursework/toy2box-bc/open_views.py
echo "recording $N successful demos -> $OUT   (K gripper, R discard, W/S A/D Q/E move)"
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
    --dataset_file "$OUT" --num_demos "$N" --teleop_device keyboard \
    --kit_args "--exec $VIEWS"
ls -lh "$OUT"
./isaaclab.sh -p coursework/toy2box-bc/summarize_demos.py "$OUT"
