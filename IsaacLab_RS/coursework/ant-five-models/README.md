# Ant 다섯 모델 제출 및 공통 평가

최종 제출 모델은 Recovery `model_2800.pt`입니다. 공유된 다섯 모델, 학습 설정과
TensorBoard 기록, 공통 평가 원자료, 그래프 생성 코드 및 최신 발표 PPT를 함께 제공합니다.
이 자료를 기준으로 제출 프로젝트를 구성했으며 이전 일반화 연구 파일은 포함하지 않습니다.

## 최종 모델 평가

저장소 루트의 [evaluation_command.txt](../../../evaluation_command.txt)를 제출합니다.
설치 완료 후 `IsaacLab_RS` 폴더에서 해당 명령어를 실행합니다.
사용 task는 `Isaac-Ant-Common-Wide-Ray-Eval-v0`, seed 24, 100개 환경,
16초 제한, torso_height ON, 원본 Ant RewardsCfg입니다.

수업 제공 평가 파일은 첫 에피소드 누적 보상과 steps의 mean/std를 출력합니다.
PPT의 접지 최대 전진 거리는 별도 공통 evaluator가 측정합니다.
두 evaluator의 초기 상태 처리도 구분해야 하므로 서로 다른 실행의 점수가 비트 단위로
같다고 가정하지 않습니다. 학습 가중치를 변경하는 평가나 재학습을 수행하지 않습니다.

## PPT의 저장된 공통 평가 기록

| 모델 | 체크포인트 | 관측 차원 | 누적 보상 평균 ± 표준편차 | 접지 최대 전진 거리 평균 |
| --- | --- | ---: | ---: | ---: |
| 평지 학습 | model_999 | 60 | 2.874839 ± 2.327993 | 1.465756 m |
| 험지 직접 학습 | model_999 | 60 | 9.172407 ± 5.130262 | 3.772204 m |
| 커리큘럼 | model_999 | 60 | 6.121553 ± 3.937864 | 3.352082 m |
| 커리큘럼 + 넓은 RayCaster | model_3300 | 247 | 10.336969 ± 10.883198 | 6.963393 m |
| 커리큘럼 + 넓은 RayCaster + Recovery | model_2800 | 247 | 11.253564 ± 11.336109 | 7.473385 m |

이 표는 조원이 공유한 저장 기록입니다. 원자료는
`../../logs/ant_terrain_comparison/common_16s_height_on_seed24/`에 있고,
각 모델 CSV에 첫 에피소드 100개가 기록되어 있습니다. 표준편차는 환경 간 값이며
신뢰구간이 아닙니다. 저장 JSON의 공유 설정·지형 mesh·초기 상태 해시는 다섯 모델이 같습니다.
학습량과 시작 조건이 서로 달라 기법 하나의 인과 효과를 분리한 비교는 아닙니다.
평균 점수만으로 Boxes 탈출이나 미지 지형 일반화가 해결됐다고 주장하지 않습니다.

## 모델과 학습 기록

`models.json`에 모델별 상대 경로와 SHA-256을 기록했습니다.
모델은 원래 `logs/rsl_rl/...` 위치를 유지합니다. 선택된 다섯 모델과 설정, events,
git diff만 Git에 포함하며 새 학습 로그나 policy export는 제외합니다.
저장 YAML은 당시 학습 설정의 기록입니다. 가중치 파일만으로 환경 설정이 자동 복원되지는 않습니다.

## 다섯 모델 재평가와 그래프

[evaluation_commands.txt](evaluation_commands.txt)의 첫 명령어를 사용하면
다섯 모델을 별도 시뮬레이터 프로세스로 순차 평가합니다. 새 결과는 원자료와 다른
`logs/ant_terrain_comparison/reproduced_ppt_comparison/`에 저장합니다.
evaluator는 관측 차원을 체크포인트에서 선택하고 공유 설정, 지형과 초기 상태 해시를 비교합니다.
원본 평가 CSV/JSON을 덮어쓰지 마세요.

그래프 생성 코드는 `../../scripts/analysis/plot_ant_common_results.py`와
`shared-records/ppt_results/plot_ant_common_results.py`에 있습니다.
그림과 원자료는 `shared-records/ppt_results/`에 보존했습니다.

## 발표와 공유 자료

- [최신 발표 PPT](presentation/6조_로보틱스시뮬레이션_최종수정본.pptx)
- [공유 파일 설명](shared-records/README_FIRST.md)
- [공유 원본 manifest](shared-records/manifest.json)
- [공유 결과 요약](shared-records/stored_evaluation_results.txt)

공유 문서의 `IsaacLab/...` 경로는 이 저장소의 `IsaacLab_RS/...`에 해당합니다.
전체 매핑은 `shared-path-map.json`에 있습니다. `shared-records/CHECKSUMS.sha256`은
원본 ZIP 기준 목록이고, 이 저장소 무결성은 저장소 루트의 `SUBMISSION_SHA256SUMS`로 확인합니다.
원본 PC 절대 경로가 포함된 CSV/JSON/YAML은 당시의 기록이며 실제 실행에는 상대 경로를 사용합니다.
