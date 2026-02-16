# Isaac Lab Go2W Headless Guide (KR)

이 문서는 이 저장소에 추가된 **Unitree Go2W 헤드리스 Docker + RL 학습/플레이 파이프라인**을 바로 실행하기 위한 가이드입니다.

## 1) 무엇이 추가되었나

- Go2W USD 자동 다운로드 스크립트 (Hugging Face)
  - `scripts/assets/download_unitree_go2w.py`
- Go2W 로봇 config
  - `source/isaaclab_assets/isaaclab_assets/robots/unitree_go2w.py`
- Go2W velocity task (train/play)
  - `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2w/`
- RSL-RL 실행 래퍼
  - `scripts/rsl_rl/train.py`
  - `scripts/rsl_rl/play.py`
- play command source 확장 (random/schedule/keyboard)
  - `scripts/reinforcement_learning/rsl_rl/play.py`
- 헤드리스 Docker 기본 설정 지원
  - `docker/.container.cfg.example`
  - `docker/container.py` (없으면 `.container.cfg` 자동 생성)


## 2) Docker 실행 (공용 서버 권장 방식)

### 권장: `--suffix`로 이름 충돌 방지

```bash
cd /path/to/IsaacLab

# 예: isaac-lab-base-hyunseo-250216 이름으로 생성
python3 docker/container.py start base --suffix hyunseo-250216
python3 docker/container.py enter base --suffix hyunseo-250216
python3 docker/container.py stop base --suffix hyunseo-250216
```

### 이미 띄운 컨테이너를 수동 rename 한 경우

```bash
docker rename isaac-lab-base isaac-lab-hyunseo-250216
docker exec -it isaac-lab-hyunseo-250216 bash
```

주의: 수동 rename 후에는 `container.py enter/stop base`와 이름이 다를 수 있습니다.


## 3) NGC 인증 실패 시 (unauthorized)

`python3 docker/container.py start` 중 `nvcr.io` 인증 실패가 나면:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC API Key>
```


## 4) Go2W USD 다운로드

컨테이너 안에서 실행:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/assets/download_unitree_go2w.py
```

기본 저장 위치:

- `source/isaaclab_assets/isaaclab_assets/data/external/unitree_model/Go2W/`

필수 파일:

- `Go2W/usd/go2w.usd`
- `Go2W/usd/configuration/go2w_description_base.usd`
- `Go2W/usd/configuration/go2w_description_physics.usd`
- `Go2W/usd/configuration/go2w_description_sensor.usd`

환경변수 override:

- `ISAACLAB_GO2W_USD_PATH` (직접 USD 경로 지정)
- `ISAACLAB_GO2W_ASSET_DIR` 또는 `ISAACLAB_UNITREE_MODEL_ROOT`


## 5) 환경 등록 확인

```bash
./isaaclab.sh -p scripts/environments/list_envs.py --keyword go2w
```

등록 task:

- `Isaac-Velocity-Flat-Unitree-Go2W-v0` (학습)
- `Isaac-Velocity-Flat-Unitree-Go2W-Play-v0` (플레이)


## 6) 학습 (train)

### 기본

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2W-v0 \
  --headless \
  --max_iterations 15000 \
  --run_name go2w_15k
```

### 150k 학습

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2W-v0 \
  --headless \
  --max_iterations 150000 \
  --run_name go2w_150k
```

### 터미널 종료/원격 끊김에도 계속 실행 (detached)

호스트에서:

```bash
docker exec -d isaac-lab-hyunseo-250216 bash -lc '
  cd /workspace/isaaclab &&
  nohup ./isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2W-v0 \
    --headless \
    --max_iterations 15000 \
    --run_name go2w_posefix_v2_15k \
    --device cuda:0 \
    > /workspace/isaaclab/logs/go2w_posefix_v2_15k.out 2>&1 &
'
```

로그 확인:

```bash
docker exec isaac-lab-hyunseo-250216 bash -lc \
  'tail -f /workspace/isaaclab/logs/go2w_posefix_v2_15k.out'
```

중지:

```bash
docker exec isaac-lab-hyunseo-250216 bash -lc \
  "pkill -f 'go2w_posefix_v2_15k'"
```


## 7) 플레이 (play)

### A. 헤드리스 서버 기본 (schedule command)

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Unitree-Go2W-Play-v0 \
  --headless \
  --command_source schedule
```

### B. 키보드 teleop (WSAD + QE)

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Unitree-Go2W-Play-v0 \
  --headless \
  --command_source keyboard
```

키 매핑:

- `W/S`: `+vx / -vx`
- `A/D`: `+vy / -vy`
- `Q/E`: `+wz / -wz`
- `Space`: 명령 0으로 리셋

주의: non-TTY 헤드리스 환경에서는 keyboard 입력이 불가하면 자동으로 schedule 모드로 fallback 됩니다.


## 8) 영상 저장 (검증용)

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Unitree-Go2W-Play-v0 \
  --headless \
  --command_source schedule \
  --video \
  --video_length 500
```

저장 위치:

- `logs/rsl_rl/unitree_go2w_flat/<run_name>/videos/play/`


## 9) 산출물 위치 요약

- 학습 체크포인트: `logs/rsl_rl/unitree_go2w_flat/<run>/model_*.pt`
- TensorBoard: `logs/rsl_rl/unitree_go2w_flat/<run>/events.out.tfevents.*`
- 플레이 영상: `logs/rsl_rl/unitree_go2w_flat/<run>/videos/play/*.mp4`
- 호스트로 복사한 영상(선택): `docker/artifacts/videos/*.mp4`


## 10) 2 GPU 사용 팁

현재 RSL-RL 실행은 **프로세스당 1 GPU** 사용이 기본입니다.  
2개 GPU를 쓰고 싶으면 실험 2개를 병렬로 띄우는 방식이 가장 간단합니다.

예시:

- 실험 A: `--device cuda:0 --run_name go2w_gpu0`
- 실험 B: `--device cuda:1 --run_name go2w_gpu1`
