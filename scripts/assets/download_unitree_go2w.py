#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Download the Unitree Go2W USD asset folder from Hugging Face into the Isaac Lab repository."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

HF_REPO_ID = "unitreerobotics/unitree_model"
HF_REPO_TYPE = "dataset"
HF_ALLOW_PATTERNS = ["Go2W/**"]

REQUIRED_FILES = (
    Path("usd/go2w.usd"),
    Path("usd/configuration/go2w_description_base.usd"),
    Path("usd/configuration/go2w_description_physics.usd"),
    Path("usd/configuration/go2w_description_sensor.usd"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Unitree Go2W USD files into this Isaac Lab repository.")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help=(
            "Directory where the Go2W folder should be stored. "
            "Defaults to source/isaaclab_assets/isaaclab_assets/data/external/unitree_model/Go2W."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if required files already exist.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=HF_REPO_ID,
        help=f"Hugging Face dataset repo id (default: {HF_REPO_ID}).",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_target_dir(args: argparse.Namespace) -> Path:
    if args.target_dir is not None:
        return args.target_dir.expanduser().resolve()

    env_asset_dir = os.environ.get("ISAACLAB_GO2W_ASSET_DIR")
    if env_asset_dir:
        return Path(env_asset_dir).expanduser().resolve()

    repo_root = resolve_repo_root()
    env_asset_root = os.environ.get("ISAACLAB_UNITREE_MODEL_ROOT")
    if env_asset_root:
        return Path(env_asset_root).expanduser().resolve() / "Go2W"

    return (
        repo_root
        / "source"
        / "isaaclab_assets"
        / "isaaclab_assets"
        / "data"
        / "external"
        / "unitree_model"
        / "Go2W"
    )


def validate_go2w_layout(go2w_dir: Path) -> list[str]:
    missing = []
    for rel_path in REQUIRED_FILES:
        full_path = go2w_dir / rel_path
        if not full_path.is_file():
            missing.append(str(full_path))

    configuration_dir = go2w_dir / "usd" / "configuration"
    if not configuration_dir.is_dir() or not any(configuration_dir.iterdir()):
        missing.append(str(configuration_dir))

    return missing


def ensure_huggingface_hub_available():
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ModuleNotFoundError:
        print("[ERROR] 'huggingface_hub' is required but not installed.")
        print(
            "[ERROR] Install it with: ./isaaclab.sh -p -m pip install 'huggingface_hub>=0.26,<1.0'"
        )
        print(
            "[ERROR] If you are using Docker, rebuild the image after this patch so dependency is pre-installed."
        )
        raise



def download_go2w(repo_id: str, go2w_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    if go2w_dir.name != "Go2W":
        raise RuntimeError(
            f"Target directory must end with 'Go2W' to preserve expected layout, got: {go2w_dir}"
        )

    unitree_root_dir = go2w_dir.parent
    unitree_root_dir.mkdir(parents=True, exist_ok=True)

    download_kwargs = {
        "repo_id": repo_id,
        "repo_type": HF_REPO_TYPE,
        "allow_patterns": HF_ALLOW_PATTERNS,
        "local_dir": str(unitree_root_dir),
    }

    try:
        snapshot_download(local_dir_use_symlinks=False, **download_kwargs)
    except TypeError:
        # For newer versions where `local_dir_use_symlinks` was removed.
        snapshot_download(**download_kwargs)



def main() -> int:
    args = parse_args()
    go2w_dir = resolve_target_dir(args)

    print(f"[INFO] Target Go2W directory: {go2w_dir}")

    missing_before = validate_go2w_layout(go2w_dir)
    if not args.force and not missing_before:
        print("[INFO] Required Go2W files already exist. Skipping download.")
        return 0

    if args.force and go2w_dir.exists():
        print(f"[INFO] --force enabled. Removing existing directory: {go2w_dir}")
        shutil.rmtree(go2w_dir)

    try:
        ensure_huggingface_hub_available()
    except ModuleNotFoundError:
        return 1

    print(f"[INFO] Downloading '{args.repo_id}/Go2W' from Hugging Face...")
    download_go2w(repo_id=args.repo_id, go2w_dir=go2w_dir)

    missing_after = validate_go2w_layout(go2w_dir)
    if missing_after:
        print("[ERROR] Go2W download finished but required files are missing:")
        for path in missing_after:
            print(f"  - {path}")
        return 2

    print("[INFO] Go2W asset download and validation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
