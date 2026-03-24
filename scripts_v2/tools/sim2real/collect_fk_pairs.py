# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect (joint_pos, ee_pose) pairs from the Isaac Lab physics engine.

Uses IK-based workspace randomization (identical to training resets) to
sample diverse, reachable joint configurations.  For each configuration the
physics-engine wrist_3_link pose (in the robot base frame) is recorded.

The companion script (diffusion_policy/test_fk_comparison.py) then runs
our calibrated analytical FK on the same joint angles and compares,
verifying that the sim and real FK agree to < 0.01 mm per dimension.

Usage:
    python scripts_v2/tools/sim2real/collect_fk_pairs.py \\
        --num_samples 50 --output fk_pairs.npz

    # more samples via larger parallel batch:
    python scripts_v2/tools/sim2real/collect_fk_pairs.py \\
        --num_samples 200 --output fk_pairs.npz --settle_steps 100
"""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect FK verification pairs from simulation.")
parser.add_argument(
    "--num_samples", "-n", type=int, default=50, help="Number of IK-solved joint configurations to collect"
)
parser.add_argument("--output", "-o", type=str, default="fk_pairs.npz", help="Output npz file path")
parser.add_argument("--settle_steps", type=int, default=200, help="Physics steps to settle after each reset")
parser.add_argument(
    "--num_resets", type=int, default=1, help="Number of env resets (total pairs = num_samples * num_resets)"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import uwlab_tasks  # noqa: F401, E402
from uwlab_tasks.manager_based.manipulation.omnireset import mdp as task_mdp  # noqa: E402
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.sysid_cfg import (  # noqa: E402
    SysidEnvCfg,
)


@configclass
class FkPairsEventCfg:
    """Reset events: IK-based EE workspace randomization matching training."""

    reset_everything = EventTerm(
        func=task_mdp.reset_scene_to_default,
        mode="reset",
        params={},
    )

    reset_end_effector_pose = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset_offset": None,
            "pose_range_b": {
                "x": (0.3, 0.7),
                "y": (-0.4, 0.4),
                "z": (0.0, 0.5),
                "roll": (0.0, 0.0),
                "pitch": (np.pi / 4, 3 * np.pi / 4),
                "yaw": (np.pi / 2, 3 * np.pi / 2),
            },
            "robot_ik_cfg": SceneEntityCfg(
                "robot",
                joint_names=["shoulder.*", "elbow.*", "wrist.*"],
                body_names="robotiq_base_link",
            ),
        },
    )


def main():
    env_cfg = SysidEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_samples
    env_cfg.events = FkPairsEventCfg()

    env = gym.make("OmniReset-Ur5eRobotiq2f85-Sysid-v0", cfg=env_cfg)
    device = env.unwrapped.device

    robot = env.unwrapped.scene["robot"]
    ee_idx = robot.body_names.index("wrist_3_link")

    arm_dim = 6
    gripper_dim = 1
    zero_action = torch.zeros(args_cli.num_samples, arm_dim + gripper_dim, device=device)

    all_joint_pos = []
    all_ee_pos = []
    all_ee_quat = []
    all_ee_aa = []

    for r in range(args_cli.num_resets):
        obs, _ = env.reset()

        for _ in range(args_cli.settle_steps):
            obs, _, _, _, _ = env.step(zero_action)

        # Read all envs at once
        joint_pos = robot.data.joint_pos[:, :6].cpu().numpy()  # (N, 6)
        ee_pos_w = robot.data.body_link_pos_w[:, ee_idx]  # (N, 3)
        ee_quat_w = robot.data.body_link_quat_w[:, ee_idx]  # (N, 4)
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            ee_pos_w,
            ee_quat_w,
        )
        ee_aa_b = math_utils.axis_angle_from_quat(ee_quat_b)

        all_joint_pos.append(joint_pos)
        all_ee_pos.append(ee_pos_b.cpu().numpy())
        all_ee_quat.append(ee_quat_b.cpu().numpy())
        all_ee_aa.append(ee_aa_b.cpu().numpy())

        print(f"  Reset {r+1}/{args_cli.num_resets}: collected {len(joint_pos)} pairs")

    all_joint_pos = np.concatenate(all_joint_pos, axis=0)
    all_ee_pos = np.concatenate(all_ee_pos, axis=0)
    all_ee_quat = np.concatenate(all_ee_quat, axis=0)
    all_ee_aa = np.concatenate(all_ee_aa, axis=0)

    np.savez(
        args_cli.output,
        joint_pos=all_joint_pos,
        ee_pos=all_ee_pos,
        ee_quat=all_ee_quat,
        ee_rot_aa=all_ee_aa,
    )

    n = len(all_joint_pos)
    print(f"\nSaved {n} pairs to {args_cli.output}")
    print(f"  joint_pos  : {all_joint_pos.shape}")
    print(f"  ee_pos     : {all_ee_pos.shape}  (meters, robot base frame)")
    print(f"  ee_quat    : {all_ee_quat.shape}  (w,x,y,z)")
    print(f"  ee_rot_aa  : {all_ee_aa.shape}  (axis-angle, radians)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
