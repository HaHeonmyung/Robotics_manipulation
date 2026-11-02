# 평가 영상 (play_diagnose.py --video, 평가 seed 42, horizon 800)
- baseline_fail_grasp_hover.mp4: baseline(20개+Mimic 500, seed 101) trial 0, 트럭 앞에서 멈춘 채 800 step 시간 초과 (파지 실패)
- baseline_fail_transport_knock.mp4: baseline trial 1, 파지 직후 트럭을 쳐서 이탈 (운반 실패로 분류)
- filt70_success_1.mp4 / filt70_success_2.mp4: filt70_1000(seed 111) trial 0·1 성공 (635 / 554 step)
- relobs_success_1.mp4 / relobs_success_2.mp4: relobs_1000(seed 111) trial 0·1 성공 (514 / 715 step)
- frames_baseline_vs_relobs.png: 위 영상에서 같은 시각의 프레임을 나란히 놓은 비교 이미지
