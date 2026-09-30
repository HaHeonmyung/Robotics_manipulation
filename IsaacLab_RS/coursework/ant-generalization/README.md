# Ant PPO: 학습하지 않은 물성 조건에서의 보행 일반화

처음 실행하는 분은 [세부 실행 안내서 TXT](실행방법_처음사용자용.txt)를 먼저 읽으세요.

목표는 익숙한 환경의 보상을 높이는 것뿐 아니라, 학습하지 않은 환경에서도 보행을 유지하는 것입니다. 본 과제는 조교의 비공개 평가 환경을 모르므로 자체 스트레스 테스트를 사용합니다. 자체 결과를 조교 평가 결과로 해석하면 안 됩니다.

## 실험 설계

- 기준: 명시적으로 표준화한 nominal 환경의 PPO.
- 제안: 관측/행동/보상/PPO 설정을 바꾸지 않고 마찰, 각 몸체 질량, 수평 속도 교란만 무작위화.
- 학습: 방법별 3개 시드(11, 22, 33), 512개 병렬 환경, rollout 32스텝, 500회 PPO 업데이트.
- 예산: 학습 1회당 8,192,000 transitions. 6회 합계 49,152,000 transitions. 디버그용 20회 파일럿은 본 결과에 포함하지 않음.
- 체크포인트: 최종 고정 예산 결과. 평가 점수로 중간 모델/시드를 고르지 않음.
- 평가: 조건별 100개 환경의 첫 에피소드, 평가 시드 20261007, 최대 16초(960 제어 스텝).
- 학습 물성: 마찰 [0.5, 1.25], 질량 배율 [0.8, 1.2], 4~7초 간격 수평 속도 교란 각 축 ±0.3 m/s.
- 별도 평가: 마찰 0.2, 질량 1.5배, 3초 간격 ±0.8 m/s 교란, 복합 조건(마찰 0.3/질량 1.35배/±0.5 m/s).
- 명목 물성을 두 모델에서 동일하게 맞추기 위해 바닥 마찰 1과 multiply 조합, 로봇 마찰 1을 사용함. 따라서 원본 USD 물성에 완전히 손대지 않은 기준 모델은 아님.
- startup 무작위화는 각 collision shape/body에 독립적으로 적용되고 각 병렬 환경 내에서는 유지됨. 에피소드마다 새로 뽑는 방식은 아님.
- 강한 교란 조건은 학습 범위 밖의 크기와 다른 시간 간격을 사용함.

전체 조건은 `protocol.json`에 학습 전에 기록했습니다. 학습 결과를 보고 범위나 고정 예산을 변경하지 않습니다.

## 환경

IsaacLab 저장소 v2.3.0 (3c6e67bb5c7ada942a6d1884ab69338f57596f77), Isaac Sim 5.1.0, RSL-RL 3.0.1, PyTorch 2.7.0+cu128. 앞서 설정한 `lerobot-arena` Conda 환경을 사용합니다. LeRobot 정책은 이번 Ant PPO 학습에 사용하지 않습니다.

현재 노트북:

```bash
source /home/hhm/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
unset PYTHONPATH PYTHONHOME LD_PRELOAD
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES
```

다른 PC에서는 해당 PC의 IsaacLab 설치용 Conda 환경을 활성화합니다. 사용자명이나 고정 프로젝트 경로를 코드에 넣지 않았습니다.

## 재현

이 폴더에서 실행합니다. 이미 완결된 결과가 있으면 batch가 건너뜁니다. 새 실험은 별도 폴더 복사본을 사용합니다.

```bash
python batch.py
python analyze.py
```

개별 학습:

```bash
python run.py train --condition baseline --num_envs 512 --iterations 500 --seed 11 --headless --output runs/baseline_seed11
python run.py train --condition randomized --num_envs 512 --iterations 500 --seed 11 --headless --output runs/randomized_seed11
```

100개 환경 평가(기본 모델은 checkpoint 경로만 교체):

```bash
python run.py evaluate --condition combined --num_envs 100 --seed 20261007 --checkpoint runs/randomized_seed11/final.pt --headless --output evaluation/manual-combined
```

영상 기록도 100개 환경 평가를 끝까지 수행합니다. 영상은 0번 환경의 첫 에피소드에서 최대 600프레임만 저장하며, 조기 종료 후 다음 에피소드를 이어 붙이지 않습니다.

```bash
python run.py evaluate --condition combined --num_envs 100 --seed 20261007 --checkpoint runs/randomized_seed11/final.pt --headless --video --output videos/randomized-combined
```

관측 차원이 원본 Ant와 같으므로 최종 checkpoint는 원본 `Isaac-Ant-v0` 평가에도 로드할 수 있습니다. 실습 저장소에 강의용 `play_one_episode.py`가 있으면 다음처럼 비교할 수 있습니다. 이 명령은 원본 환경 평가이며 자체 물성 변화 조건은 적용하지 않습니다.

```bash
python scripts/reinforcement_learning/rsl_rl/play_one_episode.py --task Isaac-Ant-v0 --num_envs 100 --checkpoint /absolute/path/to/final.pt --headless
```

## 결과 파일과 해석

- `runs/`: 6개 학습의 체크포인트, TensorBoard 로그, 설정 YAML, 완료 메타데이터.
- `evaluation/`: 학습 시드×방법×평가 조건별 `metrics.json`, 100행 `episodes.csv`, 실제 적용된 `physics.npz`.
- `analysis/`: 시드별 표, 3개 학습 시드 평균/표준편차, 학습곡선, 비교 그래프.
- `videos/`: 결과를 보고 골라낸 최우수 시드가 아닌 사전 지정 시드 11의 영상.

표준편차는 두 종류를 구별합니다. `return_std_population`은 한 정책의 100개 평가 에피소드 간 모집단 표준편차입니다. `sd_across_training_seeds`는 독립적으로 학습한 정책 3개의 평균 보상 간 표본 표준편차입니다. 300개 에피소드를 300개의 독립 학습 실험인 것처럼 해석하지 않습니다.

생존율은 16초 time limit까지 낙상 없이 도달한 비율입니다. 가만히 서 있어도 생존할 수 있으므로 전진량과 보상도 함께 봅니다. 전진량은 자동 reset 위치가 섞이지 않도록 제어 스텝 직전 x축 속도를 적분한 근사값입니다. 테스트에서 잘했다는 이유로 비공개 지형이나 모든 물성에 일반화한다고 단정할 수 없습니다. 현재 자체 검증은 평평한 바닥의 동역학 변화만 다룹니다.

## 제출용 저장소

이 폴더를 `IsaacLab_RS/coursework/ant-generalization`에 포함하면 됩니다. 기존 설치 저장소의 소스는 수정하지 않았습니다. `logs`와 `checkpoint`를 포함하라는 과제 요구에 따라 결과도 함께 관리하되, GitHub의 파일 크기 제한에 맞춰 필요하면 Git LFS를 사용하세요. 제출 저장소는 https://github.com/HaHeonmyung/Robotics_manipulation 입니다. 저장소 안에서는 `IsaacLab_RS/coursework/ant-generalization`에 배치합니다. GitHub에 반영된 상태는 저장소 커밋을 확인하세요.

강의용 평가 스크립트도 실제 실행으로 확인했습니다. 사전 지정 randomized seed 11 체크포인트를 원본 `Isaac-Ant-v0`의 100개 환경에서 평가한 보상은 **78.716152 ± 38.782833**(에피소드 간 모집단 SD)이며 100/100개 첫 에피소드가 완료됐습니다. 이 값은 물성을 통일한 nominal 자체 평가와 별도입니다. 로그는 `original-environment-check.log`입니다.

PPT에는 영상의 실제 프레임이 들어 있으며, 발표 시 `videos/`의 동봉 MP4 파일을 재생하세요. 영상은 PPT에 내장되어 있지 않습니다.
