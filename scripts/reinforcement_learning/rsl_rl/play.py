# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--command_source",
    type=str,
    default="random",
    choices=["random", "schedule", "keyboard"],
    help=(
        "How to drive velocity commands during play. "
        "'random': environment command sampling, 'schedule': deterministic command schedule, "
        "'keyboard': terminal WSAD+QE command updates."
    ),
)
parser.add_argument(
    "--command_term",
    type=str,
    default="base_velocity",
    help="Command term name to overwrite when using --command_source schedule|keyboard.",
)
parser.add_argument(
    "--command_vx_limit",
    type=float,
    default=1.5,
    help="Absolute limit for commanded vx in schedule/keyboard modes.",
)
parser.add_argument(
    "--command_vy_limit",
    type=float,
    default=1.0,
    help="Absolute limit for commanded vy in schedule/keyboard modes.",
)
parser.add_argument(
    "--command_wz_limit",
    type=float,
    default=1.5,
    help="Absolute limit for commanded wz in schedule/keyboard modes.",
)
parser.add_argument(
    "--command_lin_step",
    type=float,
    default=0.2,
    help="Per-key increment for vx and vy in keyboard mode.",
)
parser.add_argument(
    "--command_yaw_step",
    type=float,
    default=0.2,
    help="Per-key increment for wz in keyboard mode.",
)
parser.add_argument(
    "--schedule_profile",
    type=str,
    default="wasd_qe",
    choices=["wasd_qe", "legacy"],
    help="Schedule profile for --command_source schedule.",
)
parser.add_argument(
    "--schedule_hold_seconds",
    type=float,
    default=3.0,
    help="How long to hold each scheduled command in seconds.",
)
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=None,
    help="Set camera eye position for the viewport (x y z).",
)
parser.add_argument(
    "--camera_target",
    type=float,
    nargs=3,
    default=None,
    help="Set camera target/look-at position for the viewport (x y z).",
)
parser.add_argument(
    "--camera_follow",
    action="store_true",
    default=False,
    help="Enable robot-following camera (updates every simulation step).",
)
parser.add_argument(
    "--camera_follow_env_idx",
    type=int,
    default=0,
    help="Environment index to follow when --camera_follow is enabled.",
)
parser.add_argument(
    "--camera_follow_offset",
    type=float,
    nargs=3,
    default=[-1.2, 0.0, 0.55],
    help="Camera offset in robot frame (x y z) when --camera_follow is enabled.",
)
parser.add_argument(
    "--camera_follow_target_height",
    type=float,
    default=0.25,
    help="Target height offset above robot base for follow camera.",
)
parser.add_argument(
    "--camera_follow_full_orientation",
    action="store_true",
    default=False,
    help="Rotate follow offset with full base orientation instead of yaw-only.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import atexit
import os
import select
import time
import tty
from dataclasses import dataclass

try:
    import termios
except ImportError:
    termios = None

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_apply, quat_apply_yaw

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@dataclass
class _CommandLimits:
    vx: float
    vy: float
    wz: float


class _TerminalWsadCommandSource:
    """Terminal WSAD+QE command source for headless play."""

    def __init__(self, limits: _CommandLimits, lin_step: float, yaw_step: float):
        self._limits = limits
        self._lin_step = lin_step
        self._yaw_step = yaw_step
        self._command = torch.zeros(3, dtype=torch.float32)
        self._enabled = False
        self._fd: int | None = None
        self._original_termios = None

        if termios is None or not sys.stdin.isatty():
            return

        self._fd = sys.stdin.fileno()
        self._original_termios = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        atexit.register(self.close)
        self._enabled = True
        self._print_help()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _print_help(self):
        print("[INFO] Keyboard command mode enabled (terminal):")
        print("       W/S: +vx/-vx, A/D: +vy/-vy, Q/E: +wz/-wz")
        print("       Space: reset command to zero, H: print help")

    def poll(self) -> torch.Tensor:
        if not self._enabled:
            return self._command

        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        while ready:
            key = sys.stdin.read(1).lower()
            if key == "w":
                self._command[0] += self._lin_step
            elif key == "s":
                self._command[0] -= self._lin_step
            elif key == "a":
                self._command[1] += self._lin_step
            elif key == "d":
                self._command[1] -= self._lin_step
            elif key == "q":
                self._command[2] += self._yaw_step
            elif key == "e":
                self._command[2] -= self._yaw_step
            elif key == " ":
                self._command.zero_()
            elif key == "h":
                self._print_help()
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)

        self._command[0].clamp_(-self._limits.vx, self._limits.vx)
        self._command[1].clamp_(-self._limits.vy, self._limits.vy)
        self._command[2].clamp_(-self._limits.wz, self._limits.wz)
        return self._command

    def close(self):
        if self._enabled and self._fd is not None and self._original_termios is not None and termios is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_termios)
        self._enabled = False


def _scheduled_command(
    step: int,
    limits: _CommandLimits,
    device: str,
    phase_steps: int,
    profile: str,
) -> torch.Tensor:
    # Default schedule: W/A/S/D/Q/E sequence for deterministic command-tracking checks.
    safe_phase_steps = max(1, phase_steps)
    if profile == "wasd_qe":
        phase = (step // safe_phase_steps) % 6
        if phase == 0:  # W
            cmd = (0.8 * limits.vx, 0.0, 0.0)
        elif phase == 1:  # A
            cmd = (0.0, 0.8 * limits.vy, 0.0)
        elif phase == 2:  # S
            cmd = (-0.8 * limits.vx, 0.0, 0.0)
        elif phase == 3:  # D
            cmd = (0.0, -0.8 * limits.vy, 0.0)
        elif phase == 4:  # Q
            cmd = (0.0, 0.0, 0.8 * limits.wz)
        else:  # E
            cmd = (0.0, 0.0, -0.8 * limits.wz)
    elif profile == "legacy":
        phase = (step // safe_phase_steps) % 8
        if phase == 0:
            cmd = (0.8 * limits.vx, 0.0, 0.0)
        elif phase == 1:
            cmd = (-0.8 * limits.vx, 0.0, 0.0)
        elif phase == 2:
            cmd = (0.0, 0.8 * limits.vy, 0.0)
        elif phase == 3:
            cmd = (0.0, -0.8 * limits.vy, 0.0)
        elif phase == 4:
            cmd = (0.0, 0.0, 0.8 * limits.wz)
        elif phase == 5:
            cmd = (0.0, 0.0, -0.8 * limits.wz)
        elif phase == 6:
            cmd = (0.6 * limits.vx, 0.4 * limits.vy, 0.5 * limits.wz)
        else:
            cmd = (0.0, 0.0, 0.0)
    else:
        raise ValueError(f"Unknown schedule profile: {profile}")
    return torch.tensor(cmd, dtype=torch.float32, device=device)


def _apply_velocity_command_override(command_term, command: torch.Tensor):
    term_command = command_term.command
    if term_command.shape[1] < 3:
        raise ValueError(f"Command term must have at least 3 dimensions, got shape={tuple(term_command.shape)}")

    term_command[:, 0] = command[0]
    term_command[:, 1] = command[1]
    term_command[:, 2] = command[2]

    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False
    if hasattr(command_term, "time_left"):
        command_term.time_left[:] = 1.0e9


def _update_follow_camera(
    env,
    robot,
    env_index: int,
    offset_local: torch.Tensor,
    target_height: float,
    full_orientation: bool,
):
    root_pos = robot.data.root_pos_w[env_index]
    root_quat = robot.data.root_quat_w[env_index]
    quat_batch = root_quat.unsqueeze(0)
    offset_batch = offset_local.unsqueeze(0)
    if full_orientation:
        world_offset = quat_apply(quat_batch, offset_batch)[0]
    else:
        world_offset = quat_apply_yaw(quat_batch, offset_batch)[0]
    eye = (root_pos + world_offset).detach().cpu().tolist()
    target = (root_pos + torch.tensor([0.0, 0.0, target_height], device=root_pos.device)).detach().cpu().tolist()
    env.sim.set_camera_view(eye=eye, target=target)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Set a custom camera view if requested.
    if args_cli.camera_eye is not None and args_cli.camera_target is not None:
        env.unwrapped.sim.set_camera_view(eye=args_cli.camera_eye, target=args_cli.camera_target)

    # Optional camera-follow mode.
    camera_follow_robot = None
    camera_follow_offset = None
    camera_follow_env_index = args_cli.camera_follow_env_idx
    if args_cli.camera_follow:
        if "robot" not in env.unwrapped.scene.keys():
            raise ValueError("Follow camera requires a robot asset named 'robot' in the scene.")
        if camera_follow_env_index < 0 or camera_follow_env_index >= env.unwrapped.scene.num_envs:
            raise ValueError(
                f"Follow camera env index out of range: {camera_follow_env_index}"
                f" (num_envs={env.unwrapped.scene.num_envs})"
            )
        camera_follow_robot = env.unwrapped.scene["robot"]
        camera_follow_offset = torch.tensor(args_cli.camera_follow_offset, dtype=torch.float32, device=env.unwrapped.device)
        print(
            "[INFO] Follow camera enabled:"
            f" env={camera_follow_env_index}, offset={args_cli.camera_follow_offset},"
            f" target_height={args_cli.camera_follow_target_height:.3f},"
            f" full_orientation={args_cli.camera_follow_full_orientation}"
        )

    # Optional external command control for command-conditioned locomotion tasks.
    command_source = args_cli.command_source
    command_limits = _CommandLimits(
        vx=args_cli.command_vx_limit,
        vy=args_cli.command_vy_limit,
        wz=args_cli.command_wz_limit,
    )
    command_term = None
    keyboard_source = None
    if command_source != "random":
        if not hasattr(env.unwrapped, "command_manager"):
            raise ValueError("Selected command override mode, but environment does not have a command manager.")
        command_term = env.unwrapped.command_manager.get_term(args_cli.command_term)
        if command_source == "keyboard":
            keyboard_source = _TerminalWsadCommandSource(
                limits=command_limits,
                lin_step=args_cli.command_lin_step,
                yaw_step=args_cli.command_yaw_step,
            )
            if not keyboard_source.enabled:
                print("[INFO] Keyboard mode requested but terminal raw input is unavailable. Falling back to schedule.")
                command_source = "schedule"

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    schedule_phase_steps = max(1, int(round(args_cli.schedule_hold_seconds / dt)))
    if command_source == "schedule":
        print(
            "[INFO] Command schedule mode:"
            f" profile={args_cli.schedule_profile}, hold={args_cli.schedule_hold_seconds:.2f}s"
            f" ({schedule_phase_steps} steps at dt={dt:.4f}s)"
        )
    step_count = 0

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if command_term is not None:
                if command_source == "keyboard":
                    command = keyboard_source.poll().to(device=env.unwrapped.device)
                else:
                    command = _scheduled_command(
                        step_count,
                        command_limits,
                        device=env.unwrapped.device,
                        phase_steps=schedule_phase_steps,
                        profile=args_cli.schedule_profile,
                    )
                _apply_velocity_command_override(command_term, command)

            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if camera_follow_robot is not None:
                _update_follow_camera(
                    env.unwrapped,
                    camera_follow_robot,
                    env_index=camera_follow_env_index,
                    offset_local=camera_follow_offset,
                    target_height=args_cli.camera_follow_target_height,
                    full_orientation=args_cli.camera_follow_full_orientation,
                )
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            step_count += 1
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if keyboard_source is not None:
        keyboard_source.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
