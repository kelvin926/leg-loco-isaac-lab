# Go2W 학습 중간 결과 리포트 (2026-02-20)

## 1) 실행 개요
- 태스크: `Isaac-Velocity-Flat-Unitree-Go2W-v0`
- 런 이름: `go2w_re43100_retrain_v1`
- 재시작 기준 체크포인트: `model_43100.pt`
- 분산 학습: `torch.distributed.run --nproc_per_node=2` (GPU 2개)
- 환경 수: `16384`

## 2) 중지 시점 요약
- 최종 저장 체크포인트: `model_50450.pt`
- 마지막 완료 반복: `Learning iteration 50465/58100`
- 총 타임스텝: `5,792,858,112`
- 반복당 시간: 약 `2.32 ~ 2.33s`

## 3) 마지막 반복(50465) 주요 지표
- Mean reward: `19.55`
- Mean episode length: `1000.00`
- Mean action noise std: `0.94`
- Mean value function loss: `0.0339`
- Mean surrogate loss: `0.1667`
- Metrics/base_velocity/error_vel_xy: `0.7077`
- Metrics/base_velocity/error_vel_yaw: `0.6055`
- Episode_Termination/time_out: `0.9992`
- Episode_Termination/base_contact: `0.0008`

## 4) 해석
- 장점:
  - `time_out` 비율이 매우 높고 `base_contact`가 낮아, 넘어짐/즉시 실패는 잘 억제됨.
- 한계:
  - 속도 추종 오차(`error_vel_xy`, `error_vel_yaw`)가 아직 커서, 조종 응답성이 충분히 날카롭지는 않음.
  - `noise std`가 높은 편(`0.94`)이라 후반 안정화 구간 대비 탐험 성향이 큼.

## 5) 참고 사항
- 중지 직후 로그에 `SignalException: ... got signal: 15` 및 `carb.tasking Mutex` assertion이 남음.
- 이는 수동 SIGTERM 종료 시 Isaac Sim/Torch 분산 프로세스 정리 과정에서 흔히 동반되는 종료 로그이며,
  체크포인트(`model_50450.pt`) 저장 자체는 정상 확인됨.

## 6) 다음 권장 액션
- 플레이 검증은 `model_50450.pt` 기준으로 우선 수행.
- 평탄면 미세조정은 다음 순서 권장:
  1. 탐험 강도 완화(초기 노이즈/entropy 점진 감쇠)
  2. 추종 오차 중심 리워드 재가중(특히 yaw)
  3. 자세 수평 유지 항(flat orientation) 유지 + 다리 벌어짐 억제 항 미세 가중
