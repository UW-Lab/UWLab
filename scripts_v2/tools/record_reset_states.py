# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to record reset states using IsaacLab framework."""

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
parser = argparse.ArgumentParser(description="Record reset states for object pairs.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task", type=str, default="OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0", help="Name of the task."
)
parser.add_argument(
    "--dataset_dir", type=str, default="./reset_state_datasets/", help="Directory to save reset state results."
)
parser.add_argument(
    "--num_reset_states", type=int, default=100, help="Number of reset states to record. Set to 0 for infinite."
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
    """Main function to record reset states."""
    # create directory if it does not exist
    if not os.path.exists(args_cli.dataset_dir):
        os.makedirs(args_cli.dataset_dir, exist_ok=True)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # make sure environment is non-deterministic for diverse pose discovery
    env_cfg.seed = None

    # Get USD paths for hash computation
    insertive_usd_path = env_cfg.scene.insertive_object.spawn.usd_path
    receptive_usd_path = env_cfg.scene.receptive_object.spawn.usd_path
    reset_state_hash = task_mdp.utils.compute_assembly_hash(insertive_usd_path, receptive_usd_path)

    # Update info.yaml with this hash and USD paths
    info_file = os.path.join(args_cli.dataset_dir, "info.yaml")
    info_data = {}
    if os.path.exists(info_file):
        with open(info_file) as f:
            info_data = yaml.safe_load(f) or {}

    info_data[reset_state_hash] = {
        "insertive_object_usd_path": insertive_usd_path,
        "receptive_object_usd_path": receptive_usd_path,
    }

    with open(info_file, "w") as f:
        yaml.dump(info_data, f, default_flow_style=False)

    print(f"Recording reset states for hash: {reset_state_hash}")
    print(f"Insertive: {insertive_usd_path}")
    print(f"Receptive: {receptive_usd_path}")

    # Setup recording configuration
    output_dir = args_cli.dataset_dir
    output_file_name = f"{reset_state_hash}.pt"

    env_cfg.recorders = task_mdp.StableStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = TorchDatasetFileHandler

    # create environment
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    env.reset()

    # Run reset state sampling
    num_reset_conditions_evaluated = 0
    current_successful_reset_conditions = 0
    actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    if "ObjectAnywhereEEGrasped" in args_cli.task or "ObjectRestingEEGrasped" in args_cli.task:
        actions[:, -1] = -1.0
    else:
        actions[:, -1] = (
            torch.randint(0, 2, (env.num_envs,), device=env.device, dtype=torch.float32) * 2 - 1
        )  # Randomly choose between -1 and 1

    # Create progress bar
    pbar = tqdm(total=args_cli.num_reset_states, desc="Successful reset states", unit="reset states")

    start_time = time.time()

    while current_successful_reset_conditions < args_cli.num_reset_states:
        # Step environment (this will evaluate grasps in parallel across environments)
        _, _, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated
        done_idx = torch.where(dones)[0]

        # Reset actions for environments that are done
        if done_idx.numel() > 0 and not (
            "ObjectAnywhereEEGrasped" in args_cli.task or "ObjectRestingEEGrasped" in args_cli.task
        ):
            actions[done_idx, -1] = (
                torch.randint(0, 2, (done_idx.numel(),), device=env.device, dtype=torch.float32) * 2 - 1
            )

        # Update progress based on successful reset conditions
        new_successful_count = env.recorder_manager.exported_successful_episode_count
        if new_successful_count > current_successful_reset_conditions:
            increment = new_successful_count - current_successful_reset_conditions
            current_successful_reset_conditions = new_successful_count
            pbar.update(increment)

        # Count total reset conditions evaluated (sum across all environments)
        num_reset_conditions_evaluated += dones.sum().item()

        if env.sim.is_stopped():
            break

    pbar.close()

    # Get final statistics
    final_successful_reset_conditions = env.recorder_manager.exported_successful_episode_count
    print("Reset state recording complete!")
    print(f"Total reset conditions evaluated: {num_reset_conditions_evaluated}")
    print(f"Successful reset conditions: {final_successful_reset_conditions}")
    if num_reset_conditions_evaluated > 0:
        print(f"Success rate: {final_successful_reset_conditions / num_reset_conditions_evaluated:.2%}")
        print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
