#!/usr/bin/env bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Run several generate_dataset.py workers in parallel (each with its own seed) and merge their outputs.
#
# Mimic data generation is bound by a single Python thread per process, so running N worker processes
# uses N CPU cores instead of one. Each worker generates ~total_trials/N successful demos.
#
# Example:
#   ./scripts/imitation_learning/isaaclab_mimic/generate_dataset_parallel.sh \
#       --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 \
#       --input_file ./datasets/toy2box_agibot_annotated.hdf5 \
#       --output_file ./datasets/toy2box_agibot_mimic_v1_500.hdf5 \
#       --total_trials 500 --num_workers 4 --num_envs 50
#
# Worker files are written next to the output as <name>_w<i>.hdf5 (and <name>_w<i>_failed.hdf5),
# worker logs to <log_dir>/generate_w<i>.log. Extra arguments after "--" are passed to every worker.

set -euo pipefail

usage() {
    sed -n '8,21p' "$0" | sed 's/^# \{0,1\}//'
    echo "Options: --task --input_file --output_file --total_trials [500] --num_workers [4]"
    echo "         --num_envs [50] --base_seed [1] --log_dir [<output dir>/logs] [-- extra args]"
}

TASK=""
INPUT_FILE=""
OUTPUT_FILE=""
TOTAL_TRIALS=500
NUM_WORKERS=4
NUM_ENVS=50
BASE_SEED=1
LOG_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --input_file) INPUT_FILE="$2"; shift 2 ;;
        --output_file) OUTPUT_FILE="$2"; shift 2 ;;
        --total_trials) TOTAL_TRIALS="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --num_envs) NUM_ENVS="$2"; shift 2 ;;
        --base_seed) BASE_SEED="$2"; shift 2 ;;
        --log_dir) LOG_DIR="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$TASK" || -z "$INPUT_FILE" || -z "$OUTPUT_FILE" ]]; then
    echo "--task, --input_file and --output_file are required."
    usage
    exit 1
fi
if (( NUM_WORKERS < 1 || TOTAL_TRIALS < NUM_WORKERS )); then
    echo "Need 1 <= num_workers <= total_trials."
    exit 1
fi

# run from the Isaac Lab root so that isaaclab.sh and relative paths resolve
ISAACLAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ISAACLAB_DIR"

OUTPUT_FILE="$(realpath -m "$OUTPUT_FILE")"
OUT_DIR="$(dirname "$OUTPUT_FILE")"
OUT_STEM="$(basename "$OUTPUT_FILE" .hdf5)"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

export PYTHONUNBUFFERED=1

PIDS=()
WORKER_FILES=()
for (( i = 0; i < NUM_WORKERS; i++ )); do
    # spread the remainder over the first workers
    trials=$(( TOTAL_TRIALS / NUM_WORKERS + (i < TOTAL_TRIALS % NUM_WORKERS ? 1 : 0) ))
    seed=$(( BASE_SEED + i ))
    worker_file="$OUT_DIR/${OUT_STEM}_w${i}.hdf5"
    log_file="$LOG_DIR/generate_w${i}.log"
    WORKER_FILES+=("$worker_file")

    echo "[worker $i] seed=$seed trials=$trials num_envs=$NUM_ENVS -> $worker_file (log: $log_file)"
    ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
        --task "$TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --generation_num_trials "$trials" \
        --seed "$seed" \
        --input_file "$INPUT_FILE" \
        --output_file "$worker_file" \
        "${EXTRA_ARGS[@]}" > "$log_file" 2>&1 &
    PIDS+=($!)
    # stagger start-up so the workers do not all load the stage at the same moment
    sleep 5
done

echo "Waiting for ${#PIDS[@]} workers: ${PIDS[*]}"
FAILED=0
for (( i = 0; i < ${#PIDS[@]}; i++ )); do
    wait "${PIDS[$i]}" || true
    if [[ ! -s "${WORKER_FILES[$i]}" ]]; then
        echo "[worker $i] produced no output, see $LOG_DIR/generate_w${i}.log"
        FAILED=1
    else
        echo "[worker $i] done: $(grep -a 'successful demos generated' "$LOG_DIR/generate_w${i}.log" | tail -1 | tr -d '\033' | sed 's/\[K//g')"
    fi
done
if (( FAILED )); then
    echo "Some workers failed, not merging."
    exit 1
fi

./isaaclab.sh -p scripts/tools/merge_hdf5_datasets.py --input_files "${WORKER_FILES[@]}" --output_file "$OUTPUT_FILE"
echo "Merged ${#WORKER_FILES[@]} worker datasets into $OUTPUT_FILE"
