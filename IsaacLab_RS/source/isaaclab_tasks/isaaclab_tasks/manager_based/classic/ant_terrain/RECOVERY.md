# Boxes 정체 및 회복 동작 보상 실험

기존에는 stuck penalty가 없다. prolonged_flight는 **모든 발 링크가 미접촉**일 때만
발동하므로 발 하나를 들어 빼는 동작을 직접 벌하지 않는다. 보상 페널티와 flight 종료
조건은 별개여서 페널티 가중치만 줄여도 0.35초 후 초기화 문제는 해결되지 않는다.

새 Recovery task는 다음을 적용한다.

| 항목 | 설정 |
| --- | --- |
| 기본 환경 | Wide-Ray + 첫 착지 최대 1초 유예 |
| 정책 입력 | 기존 247차원 / Height map 3.2 × 2.0 m, 187개 높이 값 |
| progress | 접촉을 유지하며 에피소드 최고 X를 새로 갱신한 거리 / dt, 최대 1 m/s 상당, weight +5 |
| stuck | 최근 2초 동안 에피소드 최고 X가 0.10 m 미만 증가하면 1, weight −0.5 |
| 첫 2초 / 첫 착지 전 | stuck 페널티를 적용하지 않음 |
| prolonged_flight | 기존 −2, 모든 발 미접촉 0.15초 초과 |
| flight 종료 | 첫 착지 후 기존 0.35초 제한 |
| PPO entropy_coef | 기존 Wide 모델과 같은 0.0 (Landing 실험의 0.005를 사용하지 않음) |

뒤로 물러날 때 progress는 0이므로 즉시 음의 전진 보상을 받지는 않는다.
원래 최고 위치로 돌아와도 중복 보상을 받지 않고, 이를 넘어 새로 전진해야 보상을 받는다.
앞뒤로 왕복하며 progress를 반복 수집하는 문제를 방지한다. 공중에서 이동한 최고 X도
기록하여 착지 순간 비행 거리를 접촉 보상으로 한꺼번에 수집하지 못하게 한다.

stuck은 현재 속도나 다리가 움직이는지만 보지 않는다. 최고 전진 거리의 2초 변화량을
측정하므로 제자리 발 움직임과 같은 위치를 왕복하는 동작도 정체로 판정할 수 있다.
잠깐의 회복 동작에는 2초 window를 주지만, 0.05 m/s 미만의 느린 전진도 페널티 대상이
될 수 있다. window·거리·가중치는 검증할 실험값이며 효과를 보장하지 않는다.

상태는 환경별 reset에서 초기화되고, 별도 실제 step counter를 사용하므로 RSL-RL의
무작위 초기 episode length에 영향을 받지 않는다. stuck 자체로 에피소드를 종료하지 않는다.
모든 지형에 동일하게 적용하며, boxes만 특혜를 주는 보상은 아니다.

## 학습

처음부터 학습하며 초기 std는 1.0이다. std의 하한을 새로 고정하는 변경은 없다.
학습 지형, 커리큘럼, 행동·토크 제한, 몸체 높이·기울기·경계 종료 및 다른 보상은 유지한다.

```bash
conda activate lerobot-arena
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Continuous-Wide-Ray-Recovery-v0 \
  --headless --seed 42 --num_envs 4096 \
  --max_iterations 10000 --run_name recovery_seed42_10k
```

저장 폴더: `logs/rsl_rl/ant_continuous_wide_ray_recovery/`

동일 조건으로 백그라운드에서 실행하려면:

```bash
bash scripts/reinforcement_learning/rsl_rl/train_ant_recovery_10k.sh
```

PID와 로그 경로가 출력된다. 로그의 `Episode_Reward/stuck` 항이 새 정체 페널티이다.

## 비교 실험

정체 페널티만 추가하려면 `Isaac-Ant-Continuous-Wide-Ray-Stuck-v0`를 학습한다.
여기서는 기존 부호 있는 전진 속도 보상을 유지한다. 저장 폴더는
`logs/rsl_rl/ant_continuous_wide_ray_stuck/`이다. 두 학습의 seed·환경 수·iteration을 맞춘다.

Landing 종료 조건을 유지한 무수정 보상 대조군은 Landing task에
`agent.algorithm.entropy_coef=0.0`을 지정해 학습한다. 기존 model_2400은 entropy=0.005이고
학습량도 달라 이 새 보상의 효과만 분리해 검증하는 대조군이 아니다.

필요한 경우 동일 task에 `env.rewards.prolonged_flight.weight=-1.0`을 지정해 **별도 실험**한다.
기본 Recovery 실험에서는 이 항을 변경하지 않았다. 격한 도약이 늘어나는지도 확인한다.

## 평가

과제 비교 점수에는 항상 기본 Ant 보상을 사용한다. 기존 20.545785와 비교할 때는
`Isaac-Ant-Walk-Hard-Wide-Ray-Score-Eval-v0`, seed 24, 100개 환경을 사용한다.
초기 착지를 허용한 평가를 비교한다면 두 모델 모두 같은
`Isaac-Ant-Walk-Hard-Wide-Ray-Recovery-Score-Eval-v0`에서 평가한다.

새 학습의 Mean reward는 보상 정의가 바뀌었으므로 기존 Mean reward와 직접 비교하지 않는다.
동일 평가 점수, 전진 거리, 박스에서의 회복 동작 및 종료 사유를 함께 확인한다.
초기착지+탐색+보상 변경을 Height map 크기만의 효과로 설명하지 않는다.

## 구현 검증

전진·후퇴·왕복에서 frontier 보상을 확인하고, 공중 이동 후 착지 순간 거리를 중복
보상하지 않음을 검사했다. 정지·미세 왕복·옆으로만 이동·미착지·정상 전진 및 부분 reset,
terrain origin 변경과 무작위 초기 episode length에서도 stuck window의 동작을 확인했다.

Recovery 학습 설정은 Landing 설정에서 progress와 stuck 항만 다르고,
Stuck 비교군은 stuck 항만 다름을 검사했다. 평가 환경은 동일한 Landing 평가 설정이며
기본 Ant 보상에 stuck 항을 추가하지 않았다. PPO entropy_coef=0.0을 확인했다.

GPU에서 64개 환경, PPO 8회 실행·저장을 완료했고, `Episode_Reward/stuck`이 실제
학습 로그에 기록됨을 확인했다. 이 검증용 모델은 boxes 탈출 성능 개선의 증거가 아니다.
