# 연속 험지 보행: 추가 학습

기존 짧은 코스 모델은 3.3 m 목표에서 종료했다. 새 task는 동일한 60개 기본 관측과 63개 RayCaster 높이 관측을 유지하면서, 첫 지형을 벗어난 이후의 지속 보행을 학습한다. 기존 task와 체크포인트는 그대로 유지한다.

## 환경과 커리큘럼

- 한 코스에 Isaac Lab의 8×8 m 지형 6개를 이어 붙인다(48×8 m). 계단, 역계단, 상자, rough, 경사, 역경사를 한 번씩 포함한다. 첫 지형의 중앙에서 출발하고 이후 순서는 생성 시 섞는다.
- 첫 지형 비율은 기존과 같이 각각 20/20/20/20/10/10%다. 코스 내 모든 지형은 같은 난도이며, stock 지형의 높이 0인 테두리를 맞춰 연결한다. 접합부의 짧은 평탄 구간은 유지한다.
- 난도는 L1부터 L10까지다. rough는 stock 함수가 difficulty를 무시하므로 높이 범위를 난도에 따라 별도로 확대한다. 생성한 지형과 순서는 한 실행 동안 고정되어 있고 reset 시마다 새로 생성하지 않는다.
- 3.3 m 통과 종료 및 목표 보상은 없다. 에피소드는 최대 16초이며 몸통 높이, 기울기, 장시간 공중 체공, 코스 이탈이면 종료한다. 코스 경계는 시작점 기준 x=-3.7~43.7 m, y=±1 m다.
- 실제로 16초를 모두 유지하며 순전진 거리 6 m 이상이면 승급한다. 종료 시 3.3 m 미만이고 실패했거나 16초를 모두 소진했으면 강등한다. 그 사이 성적이나 3.3 m 이후 실패는 현재 난도를 유지한다. RSL-RL의 첫 에피소드 길이 무작위화로 생긴 조기 timeout은 승급하지 않는다.
- 접촉 중 +X 속도에 보상을 주고 속도는 ±1 m/s 범위로 제한해 계산한다. 제자리 생존/방향 보상은 0이며 기존 수직 속도, 행동 변화, 공중 체공 벌점과 ±7.5 Nm 모터 한도는 유지한다.

이는 보행 안정성을 개선하기 위한 학습 설계이며, 학습 성능이 개선됐다는 결과는 아직 없다. 기존 모델과 추가 학습 모델의 비교는 같은 평가 task/seed/환경 수로 한다. 추가 학습 모델은 더 많은 경험을 사용하므로 처음부터 같은 예산으로 학습한 모델의 비교와 구분한다.

## 기존 센서 모델에서 이어 학습

기존 모델은 새 실험 폴더의 warmstart_course/model_950.pt로 복사했다. 정책과 옵티마이저는 이어받고 지형 커리큘럼 상태는 새로 시작한다. 아래의 1000회는 추가 반복 횟수다. 관측 크기가 다른 센서 없는 모델은 이 센서 체크포인트에서 resume하지 않는다.

```bash
conda activate lerobot-arena
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Continuous-Ray-v0 \
  --headless --seed 42 --num_envs 4096 \
  --resume --load_run warmstart_course --checkpoint model_950.pt \
  --max_iterations 1000 --run_name continuous_ray_seed42
```

결과는 logs/rsl_rl/ant_continuous_ray/시간_continuous_ray_seed42/에 저장된다. VRAM 부족 시 환경 수를 낮추고 변경 사항을 기록한다.

처음부터 센서 없는 비교군을 학습하려면 task를 Isaac-Ant-Continuous-v0로 바꾸고 resume/load_run/checkpoint 옵션을 제거한다. 센서 모델도 resume 옵션 없이 처음부터 학습할 수 있다.

## 평가

기존 결과와 직접 비교할 때는 기존 Score-Eval을 그대로 사용한다. checkpoint에는 새 run의 실제 .pt 경로를 넣는다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode_provided.py \
  --task Isaac-Ant-Walk-Hard-Ray-Score-Eval-v0 \
  --headless --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant_continuous_ray/새학습폴더/model_실제번호.pt
```

별도의 연속 코스 최고 난도 평가도 추가했다. training과 달리 계단 폭, 상자 폭, rough 높이 간격과 마찰이 바뀌며 난도 업데이트는 하지 않는다. 원본 Ant 보상을 사용하지만 지형/경계가 기존 Score-Eval과 다르므로 점수를 섞지 않는다. 두 평가 모두 팀 자체 지형이며 조교의 숨겨진 지형을 재현하지 않는다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Ant-Continuous-Ray-Eval-v0 --seed 24 --num_envs 8 \
  --checkpoint logs/rsl_rl/ant_continuous_ray/새학습폴더/model_실제번호.pt
```

## 실행 검증

센서 task에서 기존 model_950.pt를 읽고 64개 환경으로 PPO 2회(950, 951)를 완료했다. 123개 관측(높이 ray 63개)을 확인했다. 이는 실행 검증이며 지속 보행 성능 평가가 아니다. 승급/강등/동시 실패+timeout/초기 reset/인위적인 조기 timeout과 경계 및 접촉 중 전진 보상은 별도로 검사했다. 여섯 시작 지형 × 세 난도 × 학습/평가 설정의 36개 코스에서 GPU raycast로 접합부 양쪽의 지면 높이가 0이며 미검출이 없는지, 출발 지면이 유효한지 검사했다.

새 연속 코스 평가 task에서도 기존 model_950.pt와 제공된 평가 스크립트로 첫 에피소드 100/100개가 종료되는 것을 확인했다. 이는 추가 학습 전 모델로 평가 실행을 확인한 것이며 새 모델의 학습 결과가 아니다. 검증 로그는 logs/ant_terrain_comparison/continuous_smoke_resume.log, continuous_geometry_check.log, continuous_eval_warmstart.log에 보관했다.
