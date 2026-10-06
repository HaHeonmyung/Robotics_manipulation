# Height map 확대: 연속 험지에서 처음부터 학습

최근 모델 `model_10949.pt`가 사용한 연속 지형, 커리큘럼, 보상, 물리 설정, 실패 조건을 유지한다. 새 파일 `wide_ray_env_cfg.py`는 기존 설정을 상속하고 RayCaster의 격자 크기와 앞쪽 offset만 바꾼다. PPO 설정도 기존과 같으며, 기본 학습 횟수와 실험 폴더만 새로 지정한다.

| 설정 | 기존 | 새 모델 |
| --- | --- | --- |
| Height map 크기 | 1.6×1.2 m | 3.2×2.0 m |
| 격자 간격 | 0.2 m | 0.2 m |
| 몸통 기준 중심 X | 0 m | +0.6 m |
| 몸통 기준 관측 범위 X | -0.8~+0.8 m | -1.0~+2.2 m |
| 몸통 기준 관측 범위 Y | -0.6~+0.6 m | -1.0~+1.0 m |
| 격자/높이 샘플 | 9×7 / 63 | 17×11 / 187 |
| 정책 전체 입력 | 123 | 247 |

관측 범위는 몸통의 yaw에 따라 회전한다. 아래로 향한 ray가 지면 높이를 구하며, 기존과 같은 `torso_z - ground_z - 0.5` 값을 [-1, 1]로 제한해 정책에 입력한다. 멀리 있는 높은 지형은 값이 포화될 수 있지만, 센서 범위를 비교하기 위해 clipping도 유지한다. 범위뿐 아니라 중심 위치도 변경했으므로 이번 비교는 크기 확대와 앞쪽 배치의 결합 효과다.

지형은 48×8 m, 여섯 종류의 8 m 구간, L1~L10 커리큘럼, 최대 16초다. 3.3 m 목표 종료가 없으며, 16초 유지 + 6 m 순전진이면 승급한다. 경계, 토크 제한, 접촉 중 전진 보상 및 실패 조건은 기존 연속 task와 같다. 세부 조건은 `CONTINUOUS.md`를 따른다.

## 처음부터 학습

기존 `.pt`를 resume하지 않는다. 기존 모델의 123개 입력과 새 모델의 247개 입력은 크기가 다르다. 저장 폴더는 `logs/rsl_rl/ant_continuous_wide_ray/`다.

```bash
conda activate lerobot-arena
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Continuous-Wide-Ray-v0 \
  --headless --seed 42 --num_envs 4096 \
  --max_iterations 10000 --run_name wide_ray_fresh_seed42_10k
```

10,000회 완료 시 최종 번호는 9999다. 실험 폴더 이름에는 시작 시각이 붙는다. 평가에서 checkpoint의 `새학습폴더`를 실제 폴더 이름으로 바꾼다. 기존 폴더나 기존 체크포인트는 바꾸지 않는다.

## 새 높이맵 먼저 확인

Ant 한 개와 ground plane에서 실제 학습 센서의 187개 높이값을 확인한다. 체크포인트가 필요하지 않으며 Ant의 자세는 고정한다.

```bash
./isaaclab.sh -p scripts/demos/sensors/ant_height_map_wide.py
```

GUI 격자의 X는 몸통 기준 -1~+2.2 m다. 핑크색 점은 ray hit 지점이다. Ground Z / Policy input 표시를 전환하고 yaw를 돌려 관측 범위가 회전하는지 볼 수 있다. 스냅샷은 `logs/ant_raycaster_wide_debug/`에 저장된다. 기존 `ant_height_map.py`의 기본 동작은 좁은 센서 그대로다.

## 이전 모델과 동일한 Score-Eval

새 task는 기존 Score-Eval과 동일한 지형/보상/물리/종료 설정에 확대 센서만 적용한다. 제공받은 평가 스크립트는 수정하지 않는다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode_provided.py \
  --task Isaac-Ant-Walk-Hard-Wide-Ray-Score-Eval-v0 \
  --headless --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant_continuous_wide_ray/새학습폴더/model_9999.pt
```

최근 좁은 센서 모델의 동일 평가 결과는 보상 16.999134 ± 11.038571, 에피소드 길이 604.37 step(약 10.07초)다. 이 모델은 짧은 코스 모델에서 10,000회 추가 학습했다. 새 모델은 처음부터 학습하므로 초기화와 총 학습 이력이 다르다. 두 모델 비교는 후속 방법의 성능 비교로 해석하며, 센서 효과만 분리하려면 좁은 센서도 아래처럼 처음부터 같은 예산으로 학습한다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Continuous-Ray-v0 \
  --headless --seed 42 --num_envs 4096 \
  --max_iterations 10000 --run_name narrow_ray_fresh_seed42_10k
```

## 경계가 있는 연속 코스 평가 및 GUI

새 연속 평가 task는 기존 연속 평가와 같은 조건이다. 원본 Ant 보상을 사용하고 경계 이탈로 종료하므로, 외곽 평면 이동으로 얻는 보상을 막는다. 이전 모델의 이 평가 점수는 15.545883 ± 10.222979, 16초 유지 33/100이었다. Score-Eval과는 지형과 종료 조건이 다르므로 점수를 섞지 않는다. 두 평가 모두 자체 지형이며 조교의 숨겨진 평가 지형은 아니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Continuous-Wide-Ray-Eval-v0 \
  --headless --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant_continuous_wide_ray/새학습폴더/model_9999.pt \
  --output_csv logs/ant_terrain_comparison/wide_ray_fresh_seed24.csv

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Ant-Continuous-Wide-Ray-Eval-v0 \
  --seed 24 --num_envs 8 \
  --checkpoint logs/rsl_rl/ant_continuous_wide_ray/새학습폴더/model_9999.pt
```

GUI에서 여러 코스를 넓게 보려면 `play.py` 명령에 `--camera_mode overview`를 추가한다.
활성 환경의 위치와 코스 크기에 맞춰 위에서 내려다보는 고정 카메라를 설정한다.
기본 `--camera_mode follow`는 0번 Ant를 따라가는 가까운 화면이다.
`--log_resets`를 추가하면 자동 초기화 시 환경 번호, 에피소드 시간,
종료 관리자가 기록한 종료 사유를 터미널에 출력한다.

## 실행 검증

학습 task와 두 평가 task의 설정 전체를 각각 기존 task와 비교해, 센서 size와 offset 외 차이가 없음을 확인했다. PPO 설정은 실험 폴더/기본 반복 횟수를 제외하고 같다. 처음부터 64개 환경, PPO 4회를 실행하고 실제 관측 247개(높이값 187개)와 resume=false를 확인했다. 이 검증용 모델은 학습 성능 비교에 사용하지 않는다.

Ant 한 개와 평면에서 높이 ray 187/187개가 유효하고 지면 높이가 0±약 2e-6 m인지 확인했다. yaw 90도에서 앞쪽 범위가 회전하는 것을 실제 ray hit의 세계 좌표로 검사했다. 데모가 스냅샷 저장 후 종료 대기에 빠지던 기존 문제는 시뮬레이션 콜백을 해제하도록 수정하고 정상 종료를 확인했다. 제공된 평가 스크립트에서 새 모델과 Score-Eval을 사용해 첫 에피소드 100/100개가 완료되는 것도 확인했다. 정규 10,000회 학습과 성능 비교는 아직 수행하지 않았다.

검증 로그는 logs/ant_terrain_comparison/wide_ray_config_check.log, wide_ray_train_smoke.log, wide_ray_plane_check.log, wide_ray_eval_smoke.log에 보관한다.
