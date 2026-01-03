# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run grasp sampling using IsaacLab framework."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import time
import yaml
from tqdm import tqdm
from typing import cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Grasp sampling for end effector on objects.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="OmniReset-Robotiq2f85-GraspSampling-v0", help="Name of the task.")
parser.add_argument("--dataset_dir", type=str, default="./grasp_datasets/", help="Directory to save grasp results.")
parser.add_argument("--num_grasps", type=int, default=500, help="Number of grasp candidates to evaluate.")

AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.recorder_manager import DatasetExportMode

from uwlab.utils.datasets.torch_dataset_file_handler import TorchDatasetFileHandler

import uwlab_tasks  # noqa: F401
import uwlab_tasks.manager_based.manipulation.reset_states.mdp as task_mdp
from uwlab_tasks.utils.hydra import hydra_task_compose

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    """Main function to run grasp sampling."""
    # create directory if it does not exist
    if not os.path.exists(args_cli.dataset_dir):
        os.makedirs(args_cli.dataset_dir, exist_ok=True)

    # Get USD path for hash computation
    object_usd_path = env_cfg.scene.object.spawn.usd_path

    # Compute hash for this object
    dataset_hash = task_mdp.utils.compute_assembly_hash(object_usd_path)

    # Update info.yaml with this hash and USD path
    info_file = os.path.join(args_cli.dataset_dir, "info.yaml")
    info_data = {}
    if os.path.exists(info_file):
        with open(info_file) as f:
            info_data = yaml.safe_load(f) or {}

    info_data[dataset_hash] = {"object_usd_path": object_usd_path}

    with open(info_file, "w") as f:
        yaml.dump(info_data, f, default_flow_style=False)

    print(f"Recording grasps for hash: {dataset_hash}")
    print(f"Object: {object_usd_path}")

    # Configure recorder for hash-based saving
    env_cfg.recorders = task_mdp.GraspRelativePoseRecorderManagerCfg(
        robot_name="robot",
        object_name="object",
        gripper_body_name="robotiq_base_link",
    )
    env_cfg.recorders.dataset_export_dir_path = args_cli.dataset_dir
    env_cfg.recorders.dataset_filename = f"{dataset_hash}.pt"
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = TorchDatasetFileHandler

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # make sure environment is non-deterministic, so we don't get redundant trajectories between datasets!
    env_cfg.seed = None

    # Create environment
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped

    # Reset environment (this will trigger grasp sampling event)
    env.reset()

    # Run grasp sampling
    num_grasps_evaluated = 0
    current_successful_grasps = 0

    # Create progress bar for successful grasps
    pbar = tqdm(total=args_cli.num_grasps, desc="Successful grasps", unit="grasps")
    actions = -torch.ones(env.action_space.shape, device=env.device, dtype=torch.float32)

    start_time = time.time()

    while current_successful_grasps < args_cli.num_grasps:
        # Step environment (this will evaluate grasps in parallel across environments)
        _, _, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated

        # Update progress based on successful grasps
        new_successful_count = env.recorder_manager.exported_successful_episode_count
        if new_successful_count > current_successful_grasps:
            increment = new_successful_count - current_successful_grasps
            current_successful_grasps = new_successful_count
            pbar.update(increment)

        # Count total grasps evaluated (sum across all environments)
        num_grasps_evaluated += dones.sum().item()

        # Check if simulation should stop
        if env.sim.is_stopped():
            break

    pbar.close()

    # Get final statistics
    final_successful_grasps = env.recorder_manager.exported_successful_episode_count

    print("Grasp sampling complete!")
    print(f"Total grasps evaluated: {num_grasps_evaluated}")
    print(f"Successful grasps: {final_successful_grasps}")
    if num_grasps_evaluated > 0:
        print(f"Success rate: {final_successful_grasps / num_grasps_evaluated:.2%}")
        print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
