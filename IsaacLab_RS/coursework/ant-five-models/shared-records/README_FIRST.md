# Ant 다섯 모델 공유용 압축

전체 IsaacLab 대신 실제 Ant 실험에 필요한 파일과 PPT의 다섯 모델만 담았습니다.
이 파일은 **기존 IsaacLab 2.3.0에 복사해서 사용하는 추가 파일 묶음**입니다.
IsaacLab 공통 라이브러리와 Isaac Sim/Conda 설치는 조원 PC에 별도로 있어야 합니다.

## 포함 항목

- `IsaacLab/source/.../classic/ant/`: 기본 Ant 환경 및 PPO 설정.
- `IsaacLab/source/.../classic/ant_terrain/`: 험지, Curriculum, RayCaster, reward shaping, 공통 평가 환경과 모든 내부 참조 코드.
- Ant가 실제로 참조하는 `humanoid/mdp/`와 Ant 로봇 설정 `robots/ant.py`.
- 학습, GUI, 수업 제공 평가, 공통 평가, 긴 코스 평가 스크립트와 CLI 보조 파일.
- Height map 가시화 스크립트, 그래프 생성 스크립트.
- PPT에 사용한 가중치 **다섯 개만**. 원래 `logs/rsl_rl/...` 경로를 유지합니다.
- 각 모델의 `params/` 학습 설정, TensorBoard events 학습 로그, `git/` 변경 기록.
- 다섯 모델의 평가 CSV/JSON 및 최종 `summary.csv`. 각 CSV에 첫 에피소드 100개가 기록되어 있습니다.
- `ppt_results/`: 최종 그래프, 그래프 원자료 및 생성 코드.

중간 체크포인트, 다른 모델의 실험 기록, 캐시, 전체 Git 저장소는 제외했습니다.
학습 시 작성된 `git/IsaacLab.diff`는 작은 실험 기록이므로 유지합니다.
현재 사용한 Ant 코드 스냅샷과 저장된 학습 설정을 함께 전달합니다.
각 학습 시점의 변경 기록은 run의 git/ 폴더에서 확인할 수 있습니다.

## 다른 PC에서 설치

1. IsaacLab 2.3.0 및 호환 Isaac Sim 환경을 먼저 설치합니다.
2. ZIP을 풀면 `Ant_Five_Models_Share/IsaacLab/`이 있습니다.
3. 이 `IsaacLab/` 안의 파일들을 조원 PC의 **기존 IsaacLab 프로젝트 루트에 같은 경로로 복사**합니다.
   다른 프로젝트 디렉터리를 교체하거나 이 묶음만 독립 프로젝트로 실행하는 방식이 아닙니다.
   Linux에서는 아래처럼 원본 설치와 합칠 수 있습니다. 기존 파일이 있으면 동일 경로 파일을 덮어씁니다.

```bash
# 압축을 푼 Ant_Five_Models_Share 폴더에서 실행.
cp -a IsaacLab/. ~/IsaacLab/
conda activate lerobot-arena
cd ~/IsaacLab
./isaaclab.sh --install rsl_rl
```

이미 해당 프로젝트가 같은 Conda 환경에 editable 설치되어 있다면 복사한 소스가 반영됩니다.
`lerobot-arena` 대신 조원 PC의 IsaacLab 가상환경 이름을 써도 됩니다.
원본 주요 버전: IsaacLab 2.3.0, Isaac Sim 5.1.0.0, Python 3.11.16,
PyTorch 2.7.0+cu128, RSL-RL 3.0.1. 추가 버전 기록은 requirements-observed.txt에 있습니다.
이 버전 파일은 Conda/Isaac Sim 전체 설치 파일이 아닙니다.

## 다섯 모델 평가

`model_paths.txt`에 가중치/학습 설정/로그의 상대 경로가 있습니다.
`evaluation_commands.txt`에 다섯 모델 일괄 공통 평가, 모델별 수업 제공 평가, GUI 명령이 있습니다.
복사 후 기존 IsaacLab 루트에서 실행합니다.

PPT 조건: 같은 험지·원본 Ant 보상·seed 24·100개 환경·16초·torso_height ON.
`evaluate_ant_common.py`는 checkpoint에서 관측 차원/센서 구성을 자동 선택하므로
모델별 checkpoint 경로만 바꿔 같은 조건으로 평가할 수 있습니다.
수업 제공 `play_one_episode_provided.py`에서는 ①~③ Common-Eval,
④~⑤ Common-Wide-Ray-Eval task도 맞춰야 합니다.
예전 Continuous-Wide-Ray-Recovery-Eval task는 PPT 공통 환경과 다릅니다.

④는 RayCaster 단독이 아니라 Curriculum + 넓은 RayCaster 모델입니다.
선택 체크포인트는 ①~③ model_999, ④ model_3300, ⑤ model_2800입니다.
훈련 예산/설정은 모델별로 다르므로 기법 하나의 효과만 분리한 비교로 해석하지 않습니다.

기존 평가 기록은 `IsaacLab/logs/ant_terrain_comparison/common_16s_height_on_seed24/`에 있습니다.
CSV는 실제 에피소드 기록이고 JSON에는 당시 조건/지형/초기 상태 해시가 있습니다.
저장 기록의 원래 PC 절대 경로는 실행 당시 메타데이터입니다. 실행에는 안내의 상대 경로를 사용하세요.
학습 곡선은 각 run의 events.out.tfevents.*에 있습니다.
모든 run의 원본 터미널 stdout 파일이 별도 저장된 것은 아닙니다.
`stored_evaluation_results.txt`는 기존 CSV를 읽어 작성한 요약이며 새 평가 로그가 아닙니다.
재평가 결과는 원본 기록과 다른 출력 폴더에 저장하도록 명령을 작성했습니다.

## 무결성

CHECKSUMS.sha256은 압축에 포함된 파일의 SHA-256 목록입니다.
Linux에서는 압축을 푼 Ant_Five_Models_Share 폴더에서 `sha256sum -c CHECKSUMS.sha256`으로 확인합니다.
