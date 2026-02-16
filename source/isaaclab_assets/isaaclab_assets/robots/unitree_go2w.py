# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree Go2W (wheeled Go2) robot."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

DEFAULT_GO2W_USD_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "external" / "unitree_model" / "Go2W" / "usd" / "go2w.usd"
)
"""Default repo-local Go2W USD path used when no environment variable override is provided."""

GO2W_USD_PATH = os.environ.get("ISAACLAB_GO2W_USD_PATH", str(DEFAULT_GO2W_USD_PATH))
"""Path to the Go2W USD file.

Override with ``ISAACLAB_GO2W_USD_PATH`` when using a custom asset location.
"""


UNITREE_GO2W_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2W_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        # Leg joints are position-controlled for posture stabilization.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit_sim=80.0,
            velocity_limit_sim=35.0,
            stiffness=25.0,
            damping=0.8,
            friction=0.0,
        ),
        # Wheel joints are velocity-controlled to realize planar motion commands.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=80.0,
            stiffness=0.0,
            damping=2.0,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2W using a repo-local USD asset."""
