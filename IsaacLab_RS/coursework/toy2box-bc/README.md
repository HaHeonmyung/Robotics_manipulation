# 실습 과제 2: Agibot Toy2Box 모방학습 성공률 향상 연구

연구 질문(고정): 사람 시범으로 학습한 BC 정책은 Toy2Box에서 왜 성공률이 저조한가?
원인 하나를 진단하고, 그 원인을 겨냥해 성공률을 높여라.

제출: 11월 3일(화) 23:59 LMS (GitHub 링크 · 평가 명령어 · 5분 PPT) / 11월 5일(목) 발표.

## 최종 결과 요약 (2026-11-02)

연구 축 B(데이터 증강). 진단: 실패의 47%가 파지 단계, 78건이 트럭 5 cm 앞에서 멈춘 시간 초과 → 파지 직전 상태를 덮는 학습 데이터 부족.
주 가설: Mimic 생성 수를 늘리면 파지 직전 상태 커버리지가 늘어 성공률이 오른다 (지지). 보조 가설(원본 선별, action noise)과 A축 편집 실험(정지 구간, 파지 구간 가중)은 모두 기각.

| 조건 | 사람 시범 | Mimic | 성공률 (seed 101 / 111, 평가 seed 42, 100회, horizon 800, epoch 2000) |
|---|---|---|---|
| baseline | 20 | 500 | 27.0 ± 3.0 (30 / 24) |
| own20_1000 | 20 | 1,000 | 32.5 ± 1.5 (34 / 31) |
| rand500 | 78 | 임의 500 | 20.5 ± 0.5 (20 / 21) |
| rand750 | 78 | 임의 750 | 32.0 ± 1.0 (31 / 33) |
| **multi78_1000 (제출)** | 78 | 1,000 | **40.0 ± 2.0 (38 / 42)**, 평가 seed 7: 39.0 ± 3.0 |
| multi78_rep2 | 78 | 1,000 (생성 seed 2) | 38.0 ± 5.0 (43 / 33) |
| filt70_1000 / rand70_1000 | 70 / 70 | 1,000 | 44.5 ± 3.5 / 44.0 ± 5.0 (가설 2 기각) |
| noise03_1000 | 78 | 1,000 (noise 0.03) | 41.5 ± 0.5 (가설 4 기각) |

### 데이터 출처
- 사람 시범 80개: 조작자 4명(우리 조 2명 + 다른 조 2명, 데이터 교환), 키보드, 각 20개. replay 재현 실패 2개 제외 → 78개 (`datasets/multi78.hdf5`, 원본별 출처는 파일의 `data.attrs["source"]`).
- 1차 세션(조작자 1명) 22개 중 20개가 baseline·own20 조건에 사용.

### 학습량
- robomimic BC-RNN(GMM), 조교 config `bc_rnn_low_dim_agibot_toy2box.json` 그대로: 2,000 epoch × epoch당 100 gradient step × batch 100, seq_length 10. 모든 조건 동일, 체크포인트는 epoch 2000.
- Mimic 생성: `generate_dataset.py --num_envs 100 --generation_num_trials N`, 성공 N개까지 생성(multi78_1000: 2,452회 시도 중 1,000, 40.8%).

### 제출물
- 가중치: `final_weights/multi78_1000_seed101_epoch2000.pth`, `final_weights/multi78_1000_seed111_epoch2000.pth` (SHA-256은 `final_weights/SHA256SUMS`).
- 평가 명령: `eval_command.txt`. 재현 로그: `results/final_repro/`.
- 사전 등록한 판정 기준: `PLAN.md`. 모든 조건의 평가 JSON: `results/<조건>/eval_seed{101,111}.json`.

## 지켜야 하는 규칙 (모든 조 동일)

- 환경·성공 판정·종료 조건은 바꾸지 않는다.
- 사람 시범은 키보드로 직접 수집, 학습에 쓴 사람 시범 20개 이상 100개 이하.
- Mimic 생성 시범은 학습에 실제 쓴 성공 시범 기준 1,000개 이하.
- 학습 seed 101과 111, 두 번 학습한다. (`train.sh`의 `--seed`가 설정 JSON의 `train.seed`를 바꾼다.)
- 자체 검증: 평가 seed 42, rollout 100회, horizon 800, 성공률 평균 ± 표준편차.
- 비교하는 조건끼리 학습량(epoch·batch·체크포인트 규칙)을 같게 맞추고 보고한다.
- 제출 가중치는 최종 모델 2개(seed 101, seed 111)만 남긴다.

## 흐름과 명령 (모두 `IsaacLab_RS`에서 실행)

1. 시범 수집 (사람이 직접, 세션당 5개 권장)
   `coursework/toy2box-bc/collect.sh 03 kim 1 5` → `datasets/team03_kim_s1.hdf5`
   끝나면 건강검진표가 바로 출력된다. 세션 파일 합치기:
   `./isaaclab.sh -p scripts/tools/merge_hdf5_datasets.py --input_files datasets/team03_kim_s1.hdf5 datasets/team03_lee_s1.hdf5 --output_file datasets/team03_all.hdf5`
2. 데이터 점검
   `./isaaclab.sh -p coursework/toy2box-bc/summarize_demos.py datasets/team03_all.hdf5`
   replay 확인: `./isaaclab.sh -p scripts/tools/replay_demos.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --dataset_file datasets/team03_all.hdf5 --num_envs 1 --headless`
3. Baseline (Mimic 500 → BC seed 101·111 → 평가 seed 42 + 실패 단계 분류)
   `nohup coursework/toy2box-bc/run_baseline.sh datasets/team03_all.hdf5 baseline 500 > coursework/toy2box-bc/results/baseline.log 2>&1 &`
   결과: `coursework/toy2box-bc/results/baseline/summary.txt`, `eval_seed101.json`, `eval_seed111.json`
4. 실패 진단
   `eval.sh`는 `play_diagnose.py`로 100회를 돌리면서 실패를 접근·파지·운반·배치로 나눈다.
   `python3 coursework/toy2box-bc/aggregate.py results/baseline/eval_seed101.json results/baseline/eval_seed111.json`
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
