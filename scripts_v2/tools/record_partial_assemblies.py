# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to record partial assemblies using IsaacLab framework."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import torch
import yaml
from tqdm import tqdm
from typing import cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record partial assemblies for object pairs.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="UW-FBLeg-PartialAssemblies-v0", help="Name of the task.")
parser.add_argument(
    "--dataset_dir", type=str, default="./partial_assembly_datasets/", help="Directory to save assembly results."
)
parser.add_argument(
    "--num_trajectories", type=int, default=1, help="Number of physics trajectories to run for pose discovery."
)
parser.add_argument("--pos_similarity_threshold", type=float, default=0.001, help="Threshold for pose similarity.")
parser.add_argument(
    "--ori_similarity_threshold", type=float, default=0.01, help="Threshold for orientation similarity."
)

AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import time

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.reset_states.mdp.utils import compute_assembly_hash
from uwlab_tasks.utils.hydra import hydra_task_compose

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    """Main function to record partial assemblies."""
    # create directory if it does not exist
    if not os.path.exists(args_cli.dataset_dir):
        os.makedirs(args_cli.dataset_dir, exist_ok=True)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # make sure environment is non-deterministic for diverse pose discovery
    env_cfg.seed = None

    # Create environment
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped

    # Get USD paths for hash computation
    insertive_usd_path = env_cfg.scene.insertive_object.spawn.usd_path
    receptive_usd_path = env_cfg.scene.receptive_object.spawn.usd_path

    # Compute hash for this object combination
    dataset_hash = compute_assembly_hash(insertive_usd_path, receptive_usd_path)

    # Update info.yaml with this hash and USD paths
    info_file = os.path.join(args_cli.dataset_dir, "info.yaml")
    info_data = {}
    if os.path.exists(info_file):
        with open(info_file) as f:
            info_data = yaml.safe_load(f) or {}

    info_data[dataset_hash] = {
        "insertive_object_usd_path": insertive_usd_path,
        "receptive_object_usd_path": receptive_usd_path,
    }

    with open(info_file, "w") as f:
        yaml.dump(info_data, f, default_flow_style=False)

    print(f"Recording partial assemblies for hash: {dataset_hash}")
    print(f"Insertive: {insertive_usd_path}")
    print(f"Receptive: {receptive_usd_path}")

    # Reset environment (this will position objects at assembled state)
    env.reset()

    # Initialize pose tracking
    recorded_poses = []
    all_recorded_poses = None  # Keep track of ALL recorded poses for uniqueness checking

    # Run pose discovery
    episode_count = 0
    actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)

    # Create progress bar
    pbar = tqdm(total=args_cli.num_trajectories, desc="Trajectories", unit="episodes")
    start_time = time.time()
    total_poses_collected = 0

    while episode_count < args_cli.num_trajectories:
        # Step environment (forces will be applied automatically by pose_discovery event)
        _, rewards, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated

        # Reset environments that are done and update progress
        if dones.any():
            episodes_completed = dones.sum().item()
            episode_count += episodes_completed
            pbar.update(episodes_completed)

        # Get all pose data and use rewards from step
        if "current_pose_data" in env.extras["log"]:
            valid_mask = rewards > 0

            if valid_mask.any():
                # Filter pose data to only valid environments
                all_poses_data = env.extras["log"]["current_pose_data"]
                valid_poses_data = {key: all_poses_data[key][valid_mask] for key in all_poses_data.keys()}

                # Calculate relative poses for similarity checking
                relative_poses = valid_poses_data["relative_pose"]

                # Check uniqueness against ALL previously recorded poses
                if all_recorded_poses is not None:
                    # Calculate distance matrices between all pairs
                    relative_pos = relative_poses[:, :3]  # (N, 3)
                    relative_quat = relative_poses[:, 3:]  # (N, 4)
                    all_recorded_pos = all_recorded_poses[:, :3]  # (M, 3)
                    all_recorded_quat = all_recorded_poses[:, 3:]  # (M, 4)

                    # Compute distance matrices: (N, M)
                    pos_dists = torch.cdist(relative_pos, all_recorded_pos, p=2)  # Euclidean distance
                    ori_dists = torch.cdist(relative_quat, all_recorded_quat, p=2)  # Euclidean distance

                    # Find minimum distance to any previously recorded pose
                    min_pos_dists = torch.min(pos_dists, dim=1)[0]  # (N,)
                    min_ori_dists = torch.min(ori_dists, dim=1)[0]  # (N,)
                    new_pose_mask = (min_pos_dists > args_cli.pos_similarity_threshold) & (
                        min_ori_dists > args_cli.ori_similarity_threshold
                    )
                else:
                    new_pose_mask = torch.ones(len(relative_poses), dtype=torch.bool, device=env.device)

                # Save new unique poses
                if new_pose_mask.any():
                    new_poses = {key: valid_poses_data[key][new_pose_mask] for key in valid_poses_data.keys()}
                    recorded_poses.append(new_poses)

                    # Update all recorded poses for comparison
                    if all_recorded_poses is None:
                        all_recorded_poses = relative_poses[new_pose_mask]
                    else:
                        all_recorded_poses = torch.cat([all_recorded_poses, relative_poses[new_pose_mask]], dim=0)

                    # Update total poses collected
                    new_count = sum(len(batch["relative_position"]) for batch in recorded_poses)
                    if new_count > total_poses_collected:
                        total_poses_collected = new_count

            else:
                # No valid poses this step, continue
                pass

        # Check if simulation should stop
        if env.sim.is_stopped():
            break

    # Save any remaining poses
    if recorded_poses:
        _save_poses_to_dataset(recorded_poses, args_cli.dataset_dir, dataset_hash)

    pbar.close()

    print("Partial assembly recording complete!")
    print(f"Trajectories completed: {episode_count}")
    print(f"Poses recorded: {total_poses_collected}")
    print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")
    if episode_count > 0:
        print(f"Average poses per trajectory: {total_poses_collected / episode_count:.1f}")

    env.close()


def _save_poses_to_dataset(pose_batches: list, dataset_dir: str, dataset_hash: str) -> None:
    """Save pose batches to Torch dataset (.pt)."""
    if not pose_batches:
        return

    # Concatenate all batches into single arrays
    all_poses = {}
    for key in pose_batches[0].keys():
        all_poses[key] = torch.cat([batch[key] for batch in pose_batches], dim=0).cpu()

    # Save as Torch .pt file
    output_file = os.path.join(dataset_dir, f"{dataset_hash}.pt")
    torch.save(all_poses, output_file)

    print(f"Saved {len(all_poses['relative_position'])} poses to {output_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
