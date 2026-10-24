# 실습 과제 2: Agibot Toy2Box 모방학습 성공률 향상 연구

연구 질문(고정): 사람 시범으로 학습한 BC 정책은 Toy2Box에서 왜 성공률이 저조한가?
원인 하나를 진단하고, 그 원인을 겨냥해 성공률을 높여라.

제출: 11월 3일(화) 23:59 LMS (GitHub 링크 · 평가 명령어 · 5분 PPT) / 11월 5일(목) 발표.

## 지켜야 하는 규칙 (모든 조 동일)

- 환경·성공 판정·종료 조건은 바꾸지 않는다.
- 사람 시범은 키보드로 직접 수집, 학습에 쓴 사람 시범 20개 이상 100개 이하.
- Mimic 생성 시범은 학습에 실제 쓴 성공 시범 기준 1,000개 이하.
- 학습 seed 101과 102, 두 번 학습한다. (`train.sh`의 `--seed`가 설정 JSON의 `train.seed`를 바꾼다.)
- 자체 검증: 평가 seed 42, rollout 100회, horizon 800, 성공률 평균 ± 표준편차.
- 비교하는 조건끼리 학습량(epoch·batch·체크포인트 규칙)을 같게 맞추고 보고한다.
- 제출 가중치는 최종 모델 2개(seed 101, seed 102)만 남긴다.

## 흐름과 명령 (모두 `IsaacLab_RS`에서 실행)

1. 시범 수집 (사람이 직접, 세션당 5개 권장)
   `coursework/toy2box-bc/collect.sh 03 kim 1 5` → `datasets/team03_kim_s1.hdf5`
   끝나면 건강검진표가 바로 출력된다. 세션 파일 합치기:
   `./isaaclab.sh -p scripts/tools/merge_hdf5_datasets.py --input_files datasets/team03_kim_s1.hdf5 datasets/team03_lee_s1.hdf5 --output_file datasets/team03_all.hdf5`
2. 데이터 점검
   `./isaaclab.sh -p coursework/toy2box-bc/summarize_demos.py datasets/team03_all.hdf5`
   replay 확인: `./isaaclab.sh -p scripts/tools/replay_demos.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --dataset_file datasets/team03_all.hdf5 --num_envs 1 --headless`
3. Baseline (Mimic 500 → BC seed 101·102 → 평가 seed 42 + 실패 단계 분류)
   `nohup coursework/toy2box-bc/run_baseline.sh datasets/team03_all.hdf5 baseline 500 > coursework/toy2box-bc/results/baseline.log 2>&1 &`
   결과: `coursework/toy2box-bc/results/baseline/summary.txt`, `eval_seed101.json`, `eval_seed102.json`
4. 실패 진단
   `eval.sh`는 `play_diagnose.py`로 100회를 돌리면서 실패를 접근·파지·운반·배치로 나눈다.
   `python3 coursework/toy2box-bc/aggregate.py results/baseline/eval_seed101.json results/baseline/eval_seed102.json`
5. 통제 실험: 한 요인만 바꾼 데이터/설정으로 같은 스크립트를 다른 tag로 다시 돌린다.
   `run_baseline.sh datasets/team03_filtered.hdf5 exp_noregrasp 500`
6. 최종 평가 명령 (제출용, 조교 형식)
   ```
   ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
       --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
       --checkpoint <GitHub에 올린 최종 모델 경로> --num_rollouts 100 --horizon 800 --seed 42 --headless
   ```

## 실패 단계 분류 기준 (`play_diagnose.py`)

| 단계 | 조건 |
|---|---|
| approach | 손–트럭 거리가 한 번도 0.08 m 안으로 들어오지 않음 |
| grasp | 트럭까지 갔지만 파지 신호(`subtask_terms/grasp`)가 한 번도 켜지지 않음 |
| transport | 파지했지만 트럭이 상자 xy 0.12 m 안에 오기 전에 놓치거나 떨어뜨림 |
| place | 트럭이 상자 근처까지 갔지만 성공 조건(xy < 0.10, 높이, 그리퍼 열림)이 안 맞음 |

성공률 계산은 `play.py`와 같은 루프·seed를 쓰므로 `play.py` 결과와 같다.

## 파일

- `collect.sh` 시범 수집 래퍼 (덮어쓰기 방지)
- `summarize_demos.py` 시범 길이·그리퍼 전환·멈춤 비율
- `train.sh` BC 학습 (seed, 이름 지정)
- `eval.sh` / `play_diagnose.py` 평가 + 실패 단계 분류
- `aggregate.py` seed 평균 ± 표준편차, 실패 분포
- `run_baseline.sh` 한 조건 전체 파이프라인
- `results/<tag>/` 각 조건의 로그·결과 JSON
