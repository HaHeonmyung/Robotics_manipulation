# 초기 착지 및 탐색 유지 실험

학습 task: `Isaac-Ant-Continuous-Wide-Ray-Landing-v0`

이전 Wide-Ray와 동일한 247차원 관측, 3.2 × 2.0 m Height map, 연속 혼합 지형,
커리큘럼, 보상 및 물리 설정을 사용한다. 지형 생성기가 반환하는 시작점 높이
`env_origins.z`에 Ant 기본 root 높이 0.5 m를 더하는 기존 리셋 방식도 유지한다.
피라미드 정상·역피라미드 바닥·박스 중앙 플랫폼의 시작 높이는 이미 다르다.

변경한 두 조건:

- `flight` 종료: 리셋 직후 첫 발 접촉까지 최대 1.0초를 허용한다. 한 번이라도
  발 접촉이 확인되면 기존대로 모든 발이 0.35초 넘게 미접촉할 때 종료한다.
  1.0초 안에 한 번도 접촉하지 못하면 종료한다. 개별 환경의 실시간 step counter를
  사용하므로 RSL-RL의 초기 episode length 무작위화에 영향을 받지 않는다.
- PPO `entropy_coef`: 0.0 → 0.005. 탐색을 유지하도록 장려하는 실험값이며,
  행동 std의 하한을 보장하거나 더 좋은 성능을 보장하지 않는다.

torso_height, tilt, 코스 경계, 16초 timeout과 공중 체류 보상 페널티는 기존과 같다.
초기 grace 중에도 몸통 높이·기울기·경계에 의한 종료는 적용된다.
이 실험은 Height map 크기만을 비교하는 실험이 아니라, 초기 착지와 탐색을 함께
조정한 실험이다. 두 변경의 효과를 따로 보려면 같은 Landing task로
`agent.algorithm.entropy_coef=0.0`을 지정한 학습도 비교한다.

## 새로 10,000회 학습

```bash
conda activate lerobot-arena
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Continuous-Wide-Ray-Landing-v0 \
  --headless --seed 42 --num_envs 4096 \
  --max_iterations 10000 --run_name landing_entropy_seed42_10k
```

새 실험 디렉터리 `logs/rsl_rl/ant_continuous_wide_ray_landing/`에 저장한다.
`--resume` 없이 새 정책을 학습하므로 학습된 작은 std를 불러오지 않는다.

터미널을 닫아도 실행되게 백그라운드에서 시작하려면:

```bash
bash scripts/reinforcement_learning/rsl_rl/train_ant_landing_10k.sh
```

화면에 PID, 로그 경로, 진행 확인 명령이 출력된다.

## 기존 모델로 착지 변경만 GUI 확인

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Ant-Continuous-Wide-Ray-Landing-Eval-v0 \
  --num_envs 8 --seed 24 --camera_mode overview --log_resets --real-time \
  --checkpoint logs/rsl_rl/ant_continuous_wide_ray/2026-10-05_21-25-33_wide_ray_fresh_seed42_10k/model_3300.pt
```

가중치를 그대로 사용하고 착지 종료 판정만 바꿔 확인한다. 이 명령은 정책을 학습하지 않는다.

## 평가 비교

기존 결과 20.545785와 비교하려면 기존 `Isaac-Ant-Walk-Hard-Wide-Ray-Score-Eval-v0`
환경에서 seed 24, 100개 환경으로 새 모델과 이전 모델을 평가한다.
Landing-Score-Eval에서 평가한다면 두 모델 모두 해당 환경에서 다시 평가한다.
종료 조건이 다른 환경의 점수를 직접 비교하면 안 된다.

initial flight 재종료가 줄어드는지는 첫 에피소드 평가뿐 아니라 여러 번의
자동 리셋도 관찰한다. timeout 자체는 전진·완주 성공을 의미하지 않는다.

## 실행 검증

첫 착지 유예, 접촉 후 0.35초 제한, 미착지 1.0초 제한 및 일부 환경의 독립적인
reset clock을 검사했다. 학습/두 평가 환경 설정은 flight 종료 항목 외 기존과 같고,
PPO 설정은 실험 디렉터리와 entropy_coef 외 기존과 같음을 확인했다.

GPU에서 기존 model_3300을 사용하여 64개 환경을 1,050 step(시뮬레이션 17.5초) 실행했다.
반복 리셋을 포함한 이 짧은 검사에서 flight 종료는 관찰되지 않았다. 이는 안정적인
보행 또는 점수 향상의 검증을 뜻하지 않는다. 이어서 새 task로 64개 환경, PPO 4회를
실행하고 저장까지 정상 완료했다. 검증용 모델은 성능 비교에 사용하지 않는다.
