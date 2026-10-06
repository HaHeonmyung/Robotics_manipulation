# 최고 난도 Ant 높이 관측 비교

주의: 아래 기존 모델의 보상과 시간 종료 비율은 정상 보행 성공률이 아니다. GUI에서 점프/비행이 관찰됐고, 계측에서도 긴 비접촉 구간이 확인됐다. 기존 실험은 보행 성공 사례로 해석하지 않는다.

두 학습 task는 Ant의 보상, 행동, PPO 설정, 마찰, 지형 종류와 난도, 환경 수를 동일하게 사용한다. `ROUGH_TERRAINS_CFG`의 난도 범위를 `(0.9, 1.0)`으로 고정해 모든 지형 타일을 10단계에 해당하는 범위에서 생성한다. 별도 실험 이름으로 체크포인트를 저장한다.

| 용도 | Task | 관측 수 |
| --- | --- | ---: |
| 센서 없음 학습 | `Isaac-Ant-Terrain-Hard-v0` | 60 |
| RayCaster 학습 | `Isaac-Ant-Terrain-Hard-Ray-v0` | 123 |
| 센서 없음 평가 | `Isaac-Ant-Terrain-Hard-Eval-v0` | 60 |
| RayCaster 평가 | `Isaac-Ant-Terrain-Hard-Ray-Eval-v0` | 123 |

RayCaster는 Ant 몸통 기준 `1.6 × 1.2 m` 영역을 `0.2 m` 간격으로 훑어 63개의 높이값을 정책 관측에 추가한다. 평가 task는 두 모델에 동일하게 계단 폭, 상자 간격, 거친 지형의 높이 간격, 마찰계수를 바꾼다. 평가 시 `--seed 24`를 사용한다. 이 구성은 제공되지 않은 조교의 숨겨진 평가 지형을 재현하는 것이 아니라 팀의 자체 평가 지형이다.

학습 명령은 Isaac Lab 루트에서 실행한다. 두 실행 모두 seed, 환경 수, 반복 횟수를 동일하게 둔다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-Terrain-Hard-v0 --headless --seed 42 --num_envs 4096 --max_iterations 1000 --run_name hard_seed42
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-Terrain-Hard-Ray-v0 --headless --seed 42 --num_envs 4096 --max_iterations 1000 --run_name hard_ray_seed42
```

평가는 각각 같은 관측 크기의 평가 task를 사용한다. 아래 스크립트는 100개 환경의 첫 에피소드 보상 합계와 에피소드 길이의 평균, 표준편차 및 시간 종료 비율을 출력한다. `--output_csv`로 환경별 결과도 저장한다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --task Isaac-Ant-Terrain-Hard-Eval-v0 --headless --seed 24 --num_envs 100 --checkpoint logs/rsl_rl/ant_terrain_hard/2026-10-05_12-24-39_hard_seed42/model_999.pt --output_csv logs/ant_terrain_comparison/seed24_hard.csv
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --task Isaac-Ant-Terrain-Hard-Ray-Eval-v0 --headless --seed 24 --num_envs 100 --checkpoint logs/rsl_rl/ant_terrain_hard_ray/2026-10-05_12-35-13_hard_ray_seed42/model_999.pt --output_csv logs/ant_terrain_comparison/seed24_hard_ray.csv
```

## 자체 평가 결과

두 최고 난도 모델은 각각 seed 42, 환경 4096개, PPO 1000회로 학습했다. 보상 함수를 바꾸지 않았으므로 아래 누적 보상은 원래 Ant 보상 기준이다. 수치는 첫 에피소드 100개의 평균 ± 표준편차이다.

| 평가 seed | 센서 없음 보상 | RayCaster 보상 | 센서 없음 길이 | RayCaster 길이 | 시간 종료 비율: 센서 없음 → RayCaster |
| --- | ---: | ---: | ---: | ---: | ---: |
| 24 (과제 지정) | 14.34 ± 6.71 | 15.72 ± 8.04 | 381 ± 312 | 434 ± 345 | 19% → 27% |
| 25 (추가) | 13.27 ± 6.21 | 14.66 ± 7.84 | 406 ± 325 | 502 ± 377 | 24% → 38% |
| 26 (추가) | 13.72 ± 6.03 | 13.95 ± 7.52 | 451 ± 345 | 490 ± 378 | 29% → 38% |

환경별 타일 인덱스는 각 seed에서 두 모델 간 100/100개가 일치했다. 인덱스는 지형 생성기의 행/열 번호이며 실제 지형 종류를 나타내는 라벨은 아니다. RayCaster의 보상이 더 높은 환경은 seed 24/25/26에서 각각 61/60/54개였다. 300개 전체에서 보상 평균 차이는 +1.00이며, RayCaster만 시간 종료에 도달한 환경은 40개, 센서 없는 모델만 도달한 환경은 9개였다. 보상 개선은 작고 seed 26에서는 거의 사라졌다. 비행으로 전진 보상을 얻을 수 있으므로 이 수치로 보행 성능 개선을 주장할 수 없다.

과거 쉬운 혼합 지형 모델을 seed 24의 같은 평가 task에 적용하면 보상 8.00 ± 7.96, 길이 410 ± 373, 시간 종료 28%였다. 이 모델은 최고 난도 비교 실험의 학습 조건을 공유하지 않으므로 참고 기준으로만 사용한다.

## 비행 진단과 수정 실험

2026-10-05, 기존 평가 지형 seed 24, 환경 100개, 첫 에피소드에서 몸체 접촉과 수직 속도를 추가 계측했다. 접촉은 전체 몸체의 접촉력 1 N 초과로 판정했고 60 Hz로 샘플링했다. 정지 대조군은 센서 없는 기존 평가 task에서 관절 입력을 모두 0으로 설정했다.

| 계측 | 센서 없음 | RayCaster | 입력 0 대조군 |
| --- | ---: | ---: | ---: |
| 0.3초 초과 연속 비접촉을 보인 환경 수 | 80/100 | 48/100 | 10/100 |
| 최대 수직 속도의 절댓값 (m/s) | 11.54 | 11.99 | 5.00 |
| 최대 정책 출력 절댓값 | 19.02 | 21.50 | 0 |

결과 파일은 `logs/ant_terrain_comparison/audit_hard.json`, `audit_hard_ray.json`, `audit_zero.json`이다. 비접촉에는 점프뿐 아니라 지형에서 떨어지는 경우도 포함된다. 정책 출력은 실제 모터 토크가 아니다. ImplicitActuator의 `applied_torque` 기록이 0이므로 이 기록으로 실제 토크를 추정하지 않는다. 특정 모서리의 충돌이 원인인지까지는 이번 계측으로 판정할 수 없다.

수정 task는 기존 체크포인트를 재현할 수 있도록 별도로 등록했다. 두 정책의 차이는 높이 스캔 관측 추가 여부이며 나머지 수정 조건은 동일하다.

| 용도 | Task | 관측 수 |
| --- | --- | ---: |
| 센서 관측 없음 학습 | `Isaac-Ant-Walk-Hard-v0` | 60 |
| RayCaster 관측 학습 | `Isaac-Ant-Walk-Hard-Ray-v0` | 123 |
| 센서 관측 없음 평가 | `Isaac-Ant-Walk-Hard-Eval-v0` | 60 |
| RayCaster 관측 평가 | `Isaac-Ant-Walk-Hard-Ray-Eval-v0` | 123 |

수정 사항:

- 정책 행동을 ±1로 제한하고 처리된 관절 토크와 시뮬레이터 모터 한도를 ±7.5 Nm로 설정한다.
- 출발 지점의 높이를 쓰던 넘어짐 판정을 몸통 바로 아래 현재 지면과의 높이 차이로 바꾼다. 두 환경 모두 판정용 단일 ray를 사용하지만 이 값은 정책 관측에 넣지 않는다.
- 네 발 중 하나라도 접촉한 상태의 목표 방향 수평 속도로 전진 보상을 계산한다. 수직 속도, 행동 변화, 0.15초 초과 전 발 비접촉에 벌점을 준다.
- 몸통이 1.2 rad 이상 기울거나 모든 발이 0.35초 넘게 연속 비접촉이면 종료한다. 이 수치는 실험 설계 선택이며 정상 보행을 보장하는 조건은 아니다. 높은 단차의 정상적인 도약도 제한할 수 있다.

기존 모델을 그대로 재생하면 학습된 행동 자체는 바뀌지 않는다. 아래 두 모델은 처음부터 다시 학습한다. 최고 난도에서 학습이 실패하면 같은 커리큘럼을 두 비교군에 적용하는 후속 실험이 필요하다.

```bash
conda activate lerobot-arena
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-Walk-Hard-v0 --headless --seed 42 --num_envs 4096 --max_iterations 1000 --run_name walk_seed42
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-Walk-Hard-Ray-v0 --headless --seed 42 --num_envs 4096 --max_iterations 1000 --run_name walk_ray_seed42
```

수정 실험은 학습 보상을 변경했다. 과제 지침에 맞춰 자체 평가 시 `--original_ant_reward`를 반드시 사용한다. 두 수정 모델은 동일한 새 종료 조건으로 평가하므로 이전 모델 점수와 직접적인 센서 효과 비교를 하지 않는다. 실제 조교 평가 환경의 종료 조건은 조교가 제공한 조건을 따른다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --task Isaac-Ant-Walk-Hard-Eval-v0 --headless --seed 24 --num_envs 100 --original_ant_reward --checkpoint logs/rsl_rl/ant_walk_hard/학습폴더/model_999.pt --output_csv logs/ant_terrain_comparison/walk_seed24.csv
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --task Isaac-Ant-Walk-Hard-Ray-Eval-v0 --headless --seed 24 --num_envs 100 --original_ant_reward --checkpoint logs/rsl_rl/ant_walk_hard_ray/학습폴더/model_999.pt --output_csv logs/ant_terrain_comparison/walk_ray_seed24.csv
```

수정 모델의 정규 학습 및 보행 성능 평가는 아직 수행하지 않았다. 짧은 실행 확인용 체크포인트는 `smoke_walk`/`smoke_walk_ray`이며 성능 비교에 사용하지 않는다.

실행 검증: 두 학습 task 모두 GPU에서 환경 64개, PPO 4회 반복을 완료했다. 수정 RayCaster 평가 task에서 발 접촉 센서 4개, 높이 ray 63개, 정책 관측 123개를 확인했다. 큰 행동 입력에서도 처리된 관절 입력이 7.5 Nm를 초과하지 않았으며, 모터 한도도 7.5 Nm로 확인했다. 경사 아래로 이동해도 몸통-현재 지면 높이가 정상인 경우 종료하지 않는지, 낮은 몸통 및 ray 미검출을 종료하는지, 비접촉 타이머와 접촉 중 전진 보상이 작동하는지 검사했다. 수정 센서 없는 평가 task에서 실행 확인용 체크포인트로 seed 24, 환경 100개, 원본 Ant 보상 평가 및 CSV 저장도 완료했다. 이것은 실행 확인이며 학습된 보행 성능의 검증은 아니다.

## 경로 이탈을 막은 목표 도달 평가와 커리큘럼

기존 task는 목표 통과와 지형 이탈을 판정하지 않았다. `Timeout fraction`은 시간 제한까지 생존한 비율이며 통과 성공률이 아니다. 외곽 평면에서 움직여 얻은 보상도 누적될 수 있었다.

추가 평가 task `Isaac-Ant-Walk-Hard-Course-Eval-v0` / `Isaac-Ant-Walk-Hard-Ray-Course-Eval-v0`는 시작점 기준 앞쪽 3.3 m를 목표로 둔다. 계단이 끝나는 3 m 지점을 지나 타일 내부 착지 구역에서 종료하도록 했다. 몸통의 x가 ±3.7 m, y가 ±1 m 범위를 벗어나면 즉시 종료한다. 목표에는 발 접촉, 정상 몸통 높이와 기울기 조건도 필요하다. 목표 도달은 코스 완주 기준이며 정상 보행 패턴을 보장하지는 않는다. 전체 지형 배열의 외곽 평면에서 계속 보상을 얻는 것은 막는다.

지형 종류별 분석을 위해 생성기를 1행, 20열, `curriculum=True`로 두고 열마다 지형 종류가 고정되도록 했다. 평가 중 난도 업데이트는 하지 않는다. 기존 무작위 타일 배치와 다르므로 이전 점수와 직접 비교하지 않는다. seed 24, 환경 100개에 대해 각 계단/상자/rough는 20개, 각 경사는 10개씩 검사했다. 보상은 원본 Ant 보상으로 계산했다.

2026-10-05에 사용자가 학습한 센서 모델 `logs/rsl_rl/ant_walk_hard_ray/2026-10-05_13-27-42_walk_ray_seed42/model_999.pt`의 초기 부분 구간 진단 결과다. 이 진단은 목표 2.7 m / 경계 x ±3 m를 썼다. 이후 최종 목표를 3.3 m로 변경했으므로 아래 수치를 최종 코스 완주율로 쓰지 않는다.

| 지형 | 최대 범위 목표 도달 | 낮은 범위 목표 도달 |
| --- | ---: | ---: |
| 피라미드 계단 (중앙에서 내려감) | 15/20 | 15/20 |
| 역피라미드 계단 (중앙에서 올라감) | 0/20 | 16/20 |
| 상자 | 13/20 | 15/20 |
| random rough | 18/20 | 20/20 |
| 피라미드 경사 | 7/10 | 7/10 |
| 역피라미드 경사 | 7/10 | 7/10 |
| 합계 | 60/100 | 80/100 |

최대 난도 범위 `(0.9, 1.0)`의 계단 높이는 21.2~23 cm, 낮은 범위 `(0.0, 0.1)`는 5~6.8 cm이다. 낮은 단차를 오를 수 있다는 결과이지만, 높은 단차가 물리적으로 불가능하다는 증거는 아니다. 난도 스윕은 같은 모델을 다시 학습하지 않고 평가한 것으로 훈련 분포 차이도 영향을 준다. random rough 생성기는 difficulty를 무시하므로 그 지형은 범위를 내려도 높이 범위가 그대로다.

결과 CSV: `logs/ant_terrain_comparison/walk_ray_bounded_hard.csv`, `walk_ray_bounded_low.csv`. CSV의 `end_reason`과 `terrain_shape`를 확인하면 시간 종료와 목표 도달을 구분할 수 있다.

최종 3.3 m 착지 목표로 재평가한 결과는 다음과 같다. 최고 범위 평균 원본 보상 6.314632 ± 6.083755, 낮은 범위 3.383172 ± 1.516515였다. 완주율이 낮은 최고 범위에서 보상은 더 높으므로 보상 순위와 코스 통과율을 혼동하지 않는다.

| 지형 | 최대 범위 최종 목표 도달 | 낮은 범위 최종 목표 도달 |
| --- | ---: | ---: |
| 피라미드 계단 | 15/20 | 15/20 |
| 역피라미드 계단 | 0/20 | 16/20 |
| 상자 | 10/20 | 14/20 |
| random rough | 17/20 | 19/20 |
| 피라미드 경사 | 7/10 | 7/10 |
| 역피라미드 경사 | 6/10 | 7/10 |
| 합계 | 55/100 | 78/100 |

최대 범위 경로 이탈은 6/100, 시간 종료는 20/100이었다. 낮은 범위 경로 이탈은 4/100, 시간 종료는 0/100이었다. 원본 보상은 에피소드 종료까지 누적한다. 최종 CSV는 `walk_ray_exit_hard.csv`, `walk_ray_exit_low.csv`이며 같은 `logs/ant_terrain_comparison` 폴더에 저장했다.

다음 학습 task는 `Isaac-Ant-Course-Curriculum-v0` / `Isaac-Ant-Course-Curriculum-Ray-v0`이다. 경로 제한과 목표 조건을 유지하며, 10행의 계단/상자/경사 난도를 `(0.0, 1.0)`으로 생성하고 모든 환경이 첫 행에서 시작한다. 에피소드 목표 도달 시 한 단계 올라가고 그 외에는 한 단계 내려간다. 최상위 행을 통과하면 Isaac Lab의 기본 난도 업데이트 동작에 따라 임의 행으로 돌아가 낮은 단계도 유지 학습한다. 제자리 생존과 방향 정렬 보너스는 0으로 두고 접촉 중 전진 보상을 강화했으며 목표 보상을 추가했다. 두 비교군의 나머지 조건은 동일하다. random rough에는 별도의 높이 커리큘럼을 적용하지 않았다.

```bash
conda activate lerobot-arena
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-Course-Curriculum-Ray-v0 --headless --seed 42 --num_envs 4096 --max_iterations 1000 --run_name course_ray_seed42
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-Course-Curriculum-v0 --headless --seed 42 --num_envs 4096 --max_iterations 1000 --run_name course_seed42
```

커리큘럼 모델의 최고 난도 GUI 검사는 평가 task와 체크포인트 경로를 직접 지정한다. 예시의 `학습폴더`는 실제 저장 이름으로 바꾼다. 추가 CLI override `--experiment_name`은 checkpoint를 직접 지정했을 때 불필요하다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Ant-Walk-Hard-Ray-Course-Eval-v0 --num_envs 8 --seed 24 --checkpoint logs/rsl_rl/ant_course_curriculum_ray/학습폴더/model_999.pt
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --task Isaac-Ant-Walk-Hard-Ray-Course-Eval-v0 --headless --seed 24 --num_envs 100 --original_ant_reward --checkpoint logs/rsl_rl/ant_course_curriculum_ray/학습폴더/model_999.pt
```

커리큘럼 구성은 짧은 실행 검사만 수행했으며 최고 난도 통과 학습을 완료한 모델은 아직 없다. 지형 밖 이동, 공중 도달, 뒤집힌 자세, 제자리 정지에서는 성공을 인정하지 않는 조건과 성공/실패에 따른 난도 변경을 별도로 검사했다.

## 제공된 평가 스크립트 실행 확인

사용자가 제공한 `/home/research_108_01/Downloads/play_one_episode.py`를 내용 변경 없이 `scripts/reinforcement_learning/rsl_rl/play_one_episode_provided.py`로 복사했다. 같은 폴더의 `cli_args.py`를 import할 수 있도록 배치했으며 기존 자체 평가 스크립트는 유지했다. 파일 비교로 원본과 복사본 내용이 동일한 것을 확인했다.

`Isaac-Ant-Walk-Hard-Ray-Score-Eval-v0` / `Isaac-Ant-Walk-Hard-Score-Eval-v0`를 추가했다. 기존 Walk-Hard 평가 지형 및 종료 조건을 유지하고 보상만 원본 Ant의 7개 항목으로 설정한다. 센서 모델의 관측 크기는 123으로 유지한다. Course 목표/경계 종료는 적용하지 않으며 조교의 숨겨진 평가 환경을 재현한 것은 아니다.

2026-10-05, 완료된 `ant_walk_hard_ray/2026-10-05_13-27-42_walk_ray_seed42/model_999.pt`를 제공 스크립트로 seed 24, 환경 100개에서 실행했다. 첫 에피소드 100/100개가 모두 종료됐으며 누적 보상 평균 19.667724, 표준편차 9.546627, 길이 평균 732.190000, 표준편차 345.428102를 출력했다. 로그는 `logs/ant_terrain_comparison/provided_score_walk_ray.log`이다. 원본 보상으로 산출한 자체 지형 점수이며 코스 성공률 또는 조교의 최종 점수로 해석하지 않는다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode_provided.py --task Isaac-Ant-Walk-Hard-Ray-Score-Eval-v0 --headless --seed 24 --num_envs 100 --checkpoint logs/rsl_rl/ant_walk_hard_ray/2026-10-05_13-27-42_walk_ray_seed42/model_999.pt
```

## 커리큘럼 모델의 지속 보행 진단

커리큘럼 센서 모델 `ant_course_curriculum_ray/2026-10-05_13-58-08_course_ray_seed42/model_950.pt`를 같은 Score-Eval 환경, seed 24, 100개 환경에서 진단했다. 자체 평가 스크립트의 `--track_course_progress`는 목표 방문을 기록만 하며 보상과 종료 조건을 바꾸지 않는다. 제공된 평가 스크립트는 수정하지 않았다.

누적 보상 평균 11.283994, 표준편차 8.997891, 길이 평균 417.709991, 표준편차 334.605927로 사용자의 제공 스크립트 결과를 float32 누적 오차 범위 내에서 재현했다. 종료 원인은 time_out 16개, torso_height 24개, tilt 50개, flight 10개였다. 도중에 정상 자세와 접촉 조건으로 3.3 m 목표를 방문한 환경은 81개이며, 이 중 14개만 시간 제한까지 유지됐다. 목표를 방문한 환경에서 방문 이후 종료까지 걸린 시간은 평균 5.592초였다. 목표 통과 후 지속 보행 안정성에 문제가 남아 있다는 결과이며, 짧은 코스 목표 도달만으로 학습 에피소드를 종료했던 학습 조건이 한 원인일 수 있다.

전체 보상 합 / 전체 시간 합으로 계산한 보상률은 기존 센서 모델 약 1.612/초, 커리큘럼 센서 모델 약 1.621/초였다. 공식 누적 보상 기준 점수는 실제로 낮아졌고, 에피소드가 평균 12.203초에서 6.962초로 짧아진 영향이 크다. 보상률은 원인 분석용이며 과제 점수를 대체하지 않는다.

로그와 환경별 기록: `logs/ant_terrain_comparison/course_ray_long_walk.log`, `course_ray_long_walk.csv`. 이 Score-Eval은 무작위 타일 배치이고 앞선 Course-Eval은 지형 종류를 고정한 열 배치이므로, 목표 방문 81/100과 Course 성공 77/100을 직접적인 성능 변화로 비교하지 않는다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --task Isaac-Ant-Walk-Hard-Ray-Score-Eval-v0 --headless --seed 24 --num_envs 100 --track_course_progress --checkpoint logs/rsl_rl/ant_course_curriculum_ray/2026-10-05_13-58-08_course_ray_seed42/model_950.pt --output_csv logs/ant_terrain_comparison/course_ray_long_walk.csv
```

## 연속 험지 추가 학습 task (2026-10-05)

`Isaac-Ant-Continuous-Ray-v0`와 센서 없는 비교군 `Isaac-Ant-Continuous-v0`를 추가했다. stock 8 m 지형 6개를 한 난도의 48 m 코스로 연결하고, 기존 3.3 m 목표 종료/보상 대신 16초 지속 보행을 학습한다. 16초 정상 종료 + 6 m 순전진이면 승급, 3.3 m 미만 초기 실패 또는 정체는 강등, 그 사이 성적은 유지한다. 접촉 중 +X 전진 보상은 1 m/s에서 포화한다. 센서 모델의 123개 관측, 토크 및 기존 보행 실패 조건은 유지한다. rough도 난도에 따라 높이 범위를 바꾸며, 커리큘럼은 첫 행에서 시작한다.

기존 model_950.pt를 `logs/rsl_rl/ant_continuous_ray/warmstart_course/`에 복사했으며, 64개 환경으로 resume 후 PPO 2회 실행을 완료했다. 36개 코스의 지형 접합부와 출발 위치를 GPU raycast로 검사하고, 승급/강등 및 경계/보상 조건을 별도로 검사했다. 정규 추가 학습은 아직 실행하지 않았으므로 성능 개선을 주장하지 않는다. 학습/평가 명령과 세부 조건은 같은 폴더의 `CONTINUOUS.md`를 따른다. 이전과 비교할 때는 기존 Score-Eval을 유지한다. 새 `Isaac-Ant-Continuous-Ray-Eval-v0`는 별도의 연속 코스 최고 난도 평가용이며 점수를 이전 지형 점수와 직접 비교하지 않는다.

## 연속 험지 10,000회 추가 학습 결과 (2026-10-05)

이후 사용자가 위 task로 10,000회 추가 학습을 완료했다. run은 `ant_continuous_ray/2026-10-05_15-51-53_continuous_ray_seed42`, 최종 체크포인트는 `model_10949.pt`다. 이전 model_950.pt에서 시작해 반복 950~10949를 실행했다. 학습 seed는 42, 환경 수는 4096이다.

마지막 학습 로그의 평균 에피소드 길이는 650.65 step(약 10.84초), 난도 행 번호 평균은 6.4009(0부터 세므로 L7~L8 사이)다. 학습 중 timeout 비율 약 42.02%, 코스 이탈 약 29.27%다. 이는 학습 중 집계이며 최고 난도 평가 성공률이 아니다. TensorBoard의 반복 4950~5949와 9950~10949 평균을 비교하면 난도 행 번호는 6.3448→6.3566, 에피소드 길이는 648.01→651.90 step, 학습 보상은 36.76→37.03으로 후반 개선은 작았다. 최종 행동 노이즈 표준편차는 0.00543이다. 탐색이 줄어든 상태이나 이것만으로 정체 원인을 확정하지 않는다.

기존과 동일한 `Isaac-Ant-Walk-Hard-Ray-Score-Eval-v0`, seed 24, 환경 100개에서 제공된 평가 스크립트로 원본 Ant 보상 평균 16.999134, 표준편차 11.038571, 길이 평균 604.37 step(약 10.07초)을 얻었다. 이전 짧은 코스 모델의 평균 보상 11.283994, 길이 약 6.96초보다 높다. 별도 진단으로 이를 재현했으며 종료 원인은 timeout 30, torso_height 45, tilt 15, flight 10이었다. 이전의 16/24/50/10과 비교하면 기울기 종료가 줄고 몸통 높이 종료가 늘었다. 목표 방문은 79/100, 방문 후 시간 제한 유지 29/79, 방문 이후 유지 시간 평균은 9.625초다. 원본 Score-Eval은 경계 종료가 없으므로 지형 밖 이동 가능성이 남는다.

외곽 평면으로 빠지는 것을 막은 `Isaac-Ant-Continuous-Ray-Eval-v0`에서도 seed 24, 100개를 평가했다. 원본 Ant 보상 평균 15.545883, 표준편차 10.222979, 길이 약 9.81초이며 timeout 33, torso_height 16, tilt 13, flight 12, corridor_out_of_bounds 26이었다. 같은 연속 평가 task의 추가 학습 전 모델은 평균 보상 8.444553, 길이 약 4.96초였다. 개선은 경계가 있는 연속 코스에서도 관측되지만, 33% timeout은 48 m 완주율이 아니라 16초 생존 비율이다. 학습과 평가 예산/seed의 한계가 있고 조교의 숨겨진 지형 결과는 아니다.

결과는 `logs/ant_terrain_comparison/continuous_10k_summary.json`, `continuous_10k_provided_score.log`, `continuous_10k_long_walk.csv`, `continuous_10k_bounded_long_walk.csv`에 보관했다. 다음 진단은 GUI에서 코스 이탈 방향과 몸통 높이 종료가 실제 충돌인지 임계값에 따른 조기 종료인지 확인하는 것이다. 이 결과 확인 중에는 학습 설정이나 종료 조건을 변경하지 않았다.

## 확대 Height map 실험 추가 (2026-10-05)

사용자의 요청에 따라 `wide_ray_env_cfg.py`와 `agents/wide_ray_ppo_cfg.py`를 추가했다. 새 학습 task `Isaac-Ant-Continuous-Wide-Ray-v0`는 기존 연속 환경을 상속하며 센서 격자 크기만 3.2×2.0 m로 확대하고 중심을 앞쪽 0.6 m로 옮긴다. 간격 0.2 m, yaw 정렬과 기존 높이 관측/clipping을 유지한다. 실제 범위는 몸통 기준 x=-1~+2.2 m, y=±1 m, 높이값은 187개, 전체 관측은 247개다. 기존 모델에서 resume하지 않고 처음부터 학습한다.

새 연속 평가와 Score-Eval도 각각 기존 평가 설정을 상속했다. 지형, 보상, 종료, 물리, 커리큘럼 및 PPO 조건은 동일하며 설정 전체의 동등성을 검사했다. 센서 값 검사, 처음부터 PPO 4회(64개 환경), 제공 평가 스크립트 100개 첫 에피소드 실행을 완료했다. 정규 학습과 성능 개선 검증은 아직 하지 않았다. 최근 좁은 센서 모델은 기존 모델에서 이어 학습한 이력이 있으므로, 센서 효과만 분리할 때는 좁은 센서도 같은 예산으로 처음부터 학습해야 한다. 파일과 명령은 같은 폴더의 `WIDE_RAY.md`에 정리했다.
