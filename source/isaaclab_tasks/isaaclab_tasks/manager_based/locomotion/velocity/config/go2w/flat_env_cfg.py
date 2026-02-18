# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_assets.robots.unitree_go2w import UNITREE_GO2W_CFG


@configclass
class UnitreeGo2WFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Flat-ground velocity tracking task for Unitree Go2W."""

    def __post_init__(self):
        super().__post_init__()

        # Scene
        self.scene.robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None

        # Observations and curriculum
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # Commands: random command-conditioned locomotion for training
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.resampling_time_range = (6.0, 12.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # Actions: leg posture (position) + wheel spin (velocity)
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            scale=0.12,
            use_default_offset=True,
        )
        self.actions.joint_vel = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=[".*_foot_joint"],
            scale=24.0,
            use_default_offset=False,
        )

        # Rewards tuned for stable velocity tracking
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(hip|thigh|calf)"
        self.rewards.undesired_contacts.weight = -1.2
        self.rewards.track_lin_vel_xy_exp.weight = 2.2
        self.rewards.track_ang_vel_z_exp.weight = 1.1
        self.rewards.flat_orientation_l2.weight = -4.5
        self.rewards.lin_vel_z_l2.weight = -2.5
        self.rewards.ang_vel_xy_l2.weight = -0.25
        self.rewards.dof_torques_l2.weight = -8.0e-5
        self.rewards.dof_acc_l2.weight = -2.0e-7
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-5.0,
            params={"target_height": 0.45},
        )
        self.rewards.leg_joint_deviation = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.22,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                )
            },
        )
        self.rewards.leg_joint_deviation_stand = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-0.45,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.08,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                ),
            },
        )
        self.rewards.leg_joint_vel_l2 = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.0025,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                )
            },
        )
        self.rewards.leg_joint_torque_l2 = RewTerm(
            func=mdp.joint_torques_l2,
            weight=-1.2e-4,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                )
            },
        )

        # Keep episode resets strict enough to avoid crawling postures.
        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
        )
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 0.55},
        )
        self.terminations.low_base_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.22},
        )
        self.terminations.leg_contact = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


@configclass
class UnitreeGo2WFlatEnvCfg_PLAY(UnitreeGo2WFlatEnvCfg):
    """Play config for Unitree Go2W with deterministic, externally injected commands."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        # Use a robot-relative close camera so headless video output is readable.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (1.8, -1.2, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.35)

        # For play mode, keep command term alive but stop random resampling.
        # Commands are overwritten from scripts/rsl_rl/play.py.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

        # Keep play deterministic
        self.events.physics_material = None
