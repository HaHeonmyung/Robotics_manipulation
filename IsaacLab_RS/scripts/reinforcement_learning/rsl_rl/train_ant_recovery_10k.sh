#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "먼저 conda activate lerobot-arena를 실행하세요." >&2
    exit 1
fi

ANT_LAB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "$ANT_LAB_ROOT"
mkdir -p logs/ant_terrain_comparison
ANT_TRAIN_LOG="logs/ant_terrain_comparison/recovery_10k_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S).log"

TERM=xterm nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ant-Continuous-Wide-Ray-Recovery-v0 \
    --headless --seed 42 --num_envs 4096 \
    --max_iterations 10000 --run_name recovery_seed42_10k \
    > "$ANT_TRAIN_LOG" 2>&1 < /dev/null &
ANT_TRAIN_PID=$!

echo "학습 프로세스 PID: $ANT_TRAIN_PID"
echo "로그: $ANT_LAB_ROOT/$ANT_TRAIN_LOG"
echo "진행 확인: tail -f '$ANT_LAB_ROOT/$ANT_TRAIN_LOG'"
echo "체크포인트: $ANT_LAB_ROOT/logs/rsl_rl/ant_continuous_wide_ray_recovery/"
