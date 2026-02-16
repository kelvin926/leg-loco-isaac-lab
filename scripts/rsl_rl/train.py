#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility wrapper for RSL-RL train script."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

TARGET_DIR = Path(__file__).resolve().parents[1] / "reinforcement_learning" / "rsl_rl"
TARGET_SCRIPT = TARGET_DIR / "train.py"

if __name__ == "__main__":
    sys.path.insert(0, str(TARGET_DIR))
    runpy.run_path(str(TARGET_SCRIPT), run_name="__main__")
