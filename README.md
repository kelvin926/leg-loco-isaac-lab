# Go2W–OpenArm Loco-Manipulation in Isaac Lab

This repository builds on [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) and contains my additions for Go2W–OpenArm wheeled-legged loco-manipulation experiments.

[Project overview and results](https://home.hyunseo.kr/projects/kalo/) · [KALO paper](https://doi.org/10.1109/AIM65483.2026.11658017) · [Go2W setup guide (Korean)](README_GO2W.md)

## Research and release scope

The workspace supports simulation-based experiments on constrained manipulator motion for early-launch behavior in a wheeled-legged platform. It includes robot/task configuration and training/evaluation utilities. It is a related experimental workspace; the complete KALO paper-reproduction package is not publicly released.

## What I added

- Unitree Go2W asset/configuration support
- Go2W–OpenArm task setup for loco-manipulation experiments
- RSL-RL training and evaluation scripts
- Command-source extensions for scheduled and keyboard-based evaluation
- Evaluation utilities for early-launch experiments

## Reference environment

- Ubuntu 22.04 with a compatible NVIDIA GPU and driver
- Isaac Lab **2.3.2**, as recorded in [VERSION](VERSION)
- Isaac Sim **5.1.0**, as configured in [docker/.env.base](docker/.env.base)
- Python and RSL-RL dependencies from the simulator/Isaac Lab environment

The version values describe the checked-in workspace and container configuration. The [setup guide](README_GO2W.md) covers container startup, assets, registration checks, training, playback, and video output.

## Quick start

Clone the repository and follow the [Docker and asset setup](README_GO2W.md) first. After entering the configured environment, run these commands from the Isaac Lab repository root:

```bash
# Download the Go2W asset and check environment registration.
./isaaclab.sh -p scripts/assets/download_unitree_go2w.py
./isaaclab.sh -p scripts/environments/list_envs.py --keyword go2w

# Train the flat-terrain Go2W locomotion task.
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2W-v0 \
  --headless --max_iterations 15000

# Play a trained task with scheduled velocity commands.
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Unitree-Go2W-Play-v0 \
  --headless --command_source schedule
```

These commands exercise the Go2W locomotion pipeline. They do not reproduce the paper's full Go2W–OpenArm launch evaluation. Checkpoint selection and additional playback options are documented in the [guide](README_GO2W.md).

## Upstream and licenses

The project is derived from NVIDIA Isaac Lab. See the repository [LICENSE](LICENSE), [additional license](LICENSE-mimic), and [dependency/asset license notices](docs/licenses/) for the corresponding components. Robot assets and simulator distributions retain their own terms.
