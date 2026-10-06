# Robotics_manipulation

## 실습 과제 1: 처음 보는 환경에서도 잘 걷는 Ant

최종 제출 자료는 **커리큘럼 + 넓은 RayCaster + Recovery `model_2800.pt`**와
PPT에 사용한 다섯 모델 비교 자료입니다. `IsaacLab_RS/`에 전체 IsaacLab 기반 코드,
환경·학습·평가 코드 및 가중치를 포함했습니다.

- [LMS 제출용 평가 명령어](evaluation_command.txt)
- [다섯 모델과 PPT 평가 기록](IsaacLab_RS/coursework/ant-five-models/README.md)
- [최신 5분 발표 PPT](IsaacLab_RS/coursework/ant-five-models/presentation/6조_로보틱스시뮬레이션_최종수정본.pptx)
- [모델 경로와 SHA-256](IsaacLab_RS/coursework/ant-five-models/models.json)

### 설치

검증 환경: Ubuntu 22.04, NVIDIA GPU, Python 3.11, IsaacLab 2.3.0,
Isaac Sim 5.1, PyTorch 2.7.0 CUDA 12.8, RSL-RL 3.0.1.
Isaac Sim, GPU 드라이버와 가상환경 자체는 저장소에 포함하지 않습니다.
새 PC의 전체 설치 검증은 완료하지 않았습니다.

```bash
git clone https://github.com/HaHeonmyung/Robotics_manipulation.git
conda create -n lerobot-arena python=3.11 -y
conda activate lerobot-arena
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
cd Robotics_manipulation/IsaacLab_RS
chmod +x isaaclab.sh
./isaaclab.sh --install rsl_rl
python -m pip install -r coursework/ant-five-models/requirements-evaluation.txt
```

이미 설치된 환경에서는 해당 환경을 활성화한 뒤 **이 저장소의 `IsaacLab_RS` 안에서**
`./isaaclab.sh --install rsl_rl`을 실행합니다. editable 설치가 다른 프로젝트를 가리키면
의도한 제출 코드 대신 이전 코드가 import될 수 있습니다.
평가 명령은 저장소 루트의 `evaluation_command.txt`에 있습니다.
100개 환경의 첫 에피소드가 모두 끝나면 보상 및 steps의 mean/std가 출력됩니다.

PPT의 기존 공통 평가 기록에서 Recovery는 11.253564 ± 11.336109,
접지 최대 전진 거리 평균은 7.473385m입니다. 기존 Continuous-Recovery-Eval
점수와 공통 평가 점수는 환경·종료 조건이 다르므로 직접 비교하지 않습니다.
다섯 모델 모두 제공 evaluator와 Git 복제본 공통 evaluator에서 각각 100개 첫 에피소드를 정상 완료했습니다.
새 실행 검증 기록은 [verification](IsaacLab_RS/coursework/ant-five-models/verification/README.md)에 있습니다.

원본 IsaacLab의 라이선스와 저작자 표기는 유지했습니다. 출처는 [UPSTREAM.md](UPSTREAM.md)를 참고하세요.
