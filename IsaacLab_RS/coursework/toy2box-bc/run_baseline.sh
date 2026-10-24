#!/bin/bash
# Full assignment pipeline for one condition: human demos -> Mimic -> BC (seed 101 & 111) -> eval (seed 42).
# Usage (from IsaacLab_RS):
#   coursework/toy2box-bc/run_baseline.sh <human_demos.hdf5> <tag> [mimic_trials=500] [--skip-mimic]
# Outputs under coursework/toy2box-bc/results/<tag>/: step logs, eval_seed101.json, eval_seed111.json, summary.txt
# Mimic output: datasets/<tag>_mimic_<trials>.hdf5 (reused if it already exists).
# With --skip-mimic the human demos are used for training directly.
set -e
cd "$(dirname "$0")/../.."
HUMAN=${1:?human demo hdf5}; TAG=${2:?tag}; TRIALS=${3:-500}; MODE=${4:-}
RUNDIR=coursework/toy2box-bc/results/$TAG; mkdir -p "$RUNDIR"
MIMIC_TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
export PYTHONUNBUFFERED=1
echo "== $TAG started $(date)" | tee -a "$RUNDIR/summary.txt"

if [ "$MODE" = "--skip-mimic" ]; then
    TRAIN_DATA=$HUMAN
else
    ANN=datasets/${TAG}_annotated.hdf5
    TRAIN_DATA=datasets/${TAG}_mimic_${TRIALS}.hdf5
    if [ ! -e "$TRAIN_DATA" ]; then
        if [ ! -e "$ANN" ]; then
            ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
                --task $MIMIC_TASK --auto --input_file "$HUMAN" --output_file "$ANN" --headless \
                > "$RUNDIR/01_annotate.log" 2>&1
        fi
        ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
            --task $MIMIC_TASK --headless --num_envs 100 --generation_num_trials "$TRIALS" \
            --input_file "$ANN" --output_file "$TRAIN_DATA" > "$RUNDIR/02_generate.log" 2>&1
    fi
fi
echo "train data: $TRAIN_DATA" | tee -a "$RUNDIR/summary.txt"

for SEED in 101 111; do
    coursework/toy2box-bc/train.sh "$TRAIN_DATA" $SEED "$TAG" > "$RUNDIR/03_train_seed$SEED.log" 2>&1
    CK=$(grep FINAL_CHECKPOINT "$RUNDIR/03_train_seed$SEED.log" | tail -1 | cut -d= -f2)
    echo "seed $SEED checkpoint: $CK" | tee -a "$RUNDIR/summary.txt"
    coursework/toy2box-bc/eval.sh "$CK" "$RUNDIR/eval_seed$SEED.json" 42 > "$RUNDIR/04_eval_seed$SEED.log" 2>&1 &
done
wait
python3 coursework/toy2box-bc/aggregate.py "$RUNDIR"/eval_seed101.json "$RUNDIR"/eval_seed111.json | tee -a "$RUNDIR/summary.txt"
echo "== $TAG finished $(date)" | tee -a "$RUNDIR/summary.txt"
