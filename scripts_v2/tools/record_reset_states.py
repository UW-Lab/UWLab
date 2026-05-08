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
    "--dataset_dir", type=str, default="./Datasets/OmniReset/", help="Root Datasets/OmniReset/ directory."
)
parser.add_argument(
    "--reset_type",
    type=str,
    default=None,
    help="Reset type name (e.g. ObjectAnywhereEEAnywhere). Auto-inferred from --task if omitted.",
)
parser.add_argument(
    "--num_reset_states", type=int, default=100, help="Number of reset states to record. Set to 0 for infinite."
)
parser.add_argument("--video", action="store_true", default=False, help="Record video of the env.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (env steps).")
parser.add_argument("--video_dir", type=str, default=None, help="Output dir for the video (defaults under --dataset_dir).")

AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# Cameras must be enabled for video capture; mirror play.py.
if args_cli.video:
    args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Drop Isaac Sim's pip_prebundle so conda env's trimesh/rtree win on later imports.
# Patch omni.ext.FastFinder so it ignores trimesh/rtree, drop prebundle paths, evict
# prebundle-sourced trimesh/rtree already loaded.
import sys as _sys
try:
    from omni.ext._impl.fast_importer import FastFinder as _FastFinder
    _orig_find_spec = _FastFinder.find_spec
    def _patched_find_spec(*args, **kwargs):
        fullname = kwargs.get("fullname")
        if fullname is None:
            for a in args:
                if isinstance(a, str):
                    fullname = a
                    break
        if isinstance(fullname, str):
            root = fullname.split(".", 1)[0]
            if root in {"trimesh", "rtree"}:
                return None
        return _orig_find_spec(*args, **kwargs)
    _FastFinder.find_spec = _patched_find_spec
except Exception:
    pass
_sys.path[:] = [p for p in _sys.path if "pip_prebundle" not in p]
for _mod_name in list(_sys.modules):
    if _mod_name == "trimesh" or _mod_name.startswith("trimesh.") or \
       _mod_name == "rtree" or _mod_name.startswith("rtree."):
        _mod = _sys.modules.get(_mod_name)
        _mod_file = getattr(_mod, "__file__", None) or ""
        if "pip_prebundle" in _mod_file:
            del _sys.modules[_mod_name]

"""Rest everything else."""

import gymnasium as gym
import time

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.recorder_manager import DatasetExportMode

from uwlab.utils.datasets.torch_dataset_file_handler import TorchDatasetFileHandler

import uwlab_tasks  # noqa: F401
import uwlab_tasks.manager_based.manipulation.omnireset.mdp as task_mdp
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

    # Derive pair directory and reset type for output path
    insertive_usd_path = env_cfg.scene.insertive_object.spawn.usd_path
    pair = task_mdp.utils.compute_pair_dir(insertive_usd_path)
    # receptive_usd_path = env_cfg.scene.receptive_object.spawn.usd_path
    # pair = task_mdp.utils.compute_pair_dir(insertive_usd_path, receptive_usd_path)

    # Auto-infer reset_type from task name if not provided
    reset_type = args_cli.reset_type
    if reset_type is None:
        for candidate in [
            "ObjectAnywhereEEAnywhere",
            "ObjectRestingEEGrasped",
            "ObjectAnywhereEEGrasped",
            "ObjectPartiallyAssembledEEGrasped",
        ]:
            if candidate in args_cli.task:
                reset_type = candidate
                break
        if reset_type is None:
            raise ValueError(f"Could not infer reset_type from task '{args_cli.task}'. Pass --reset_type explicitly.")

    print(f"Recording reset states for: {pair} / {reset_type}")
    print(f"Insertive: {insertive_usd_path}")
    # print(f"Receptive: {receptive_usd_path}")

    # Setup recording configuration
    output_dir = os.path.join(args_cli.dataset_dir, "Resets", pair)
    os.makedirs(output_dir, exist_ok=True)
    output_file_name = f"resets_{reset_type}.pt"

    env_cfg.recorders = task_mdp.StableStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = TorchDatasetFileHandler

    # create environment
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        video_dir = args_cli.video_dir or os.path.join(args_cli.dataset_dir, "videos", reset_type)
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to {video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Step through the wrapped env so RecordVideo intercepts each step; keep a
    # handle on the unwrapped ManagerBasedRLEnv for action_space / recorder_manager.
    base_env = cast(ManagerBasedRLEnv, env.unwrapped)
    env.reset()

    # Run reset state sampling
    num_reset_conditions_evaluated = 0
    current_successful_reset_conditions = 0
    actions = torch.zeros(base_env.action_space.shape, device=base_env.device, dtype=torch.float32)
    if "ObjectAnywhereEEGrasped" in args_cli.task or "ObjectRestingEEGrasped" in args_cli.task:
        actions[:, -1] = -1.0
    else:
        actions[:, -1] = (
            torch.randint(0, 2, (base_env.num_envs,), device=base_env.device, dtype=torch.float32) * 2 - 1
        )  # Randomly choose between -1 and 1

    # Create progress bar
    pbar = tqdm(total=args_cli.num_reset_states, desc="Successful reset states", unit="reset states")

    start_time = time.time()

    # When recording a video, ensure we run long enough to capture video_length steps
    # even if the requested --num_reset_states would otherwise be hit sooner.
    video_steps_done = 0
    while current_successful_reset_conditions < args_cli.num_reset_states or (
        args_cli.video and video_steps_done < args_cli.video_length
    ):
        if args_cli.video:
            video_steps_done += 1
        # Step environment (this will evaluate grasps in parallel across environments)
        _, _, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated
        done_idx = torch.where(dones)[0]

        # Reset actions for environments that are done
        if done_idx.numel() > 0 and not (
            "ObjectAnywhereEEGrasped" in args_cli.task or "ObjectRestingEEGrasped" in args_cli.task
        ):
            actions[done_idx, -1] = (
                torch.randint(0, 2, (done_idx.numel(),), device=base_env.device, dtype=torch.float32) * 2 - 1
            )

        # Update progress based on successful reset conditions
        new_successful_count = base_env.recorder_manager.exported_successful_episode_count
        if new_successful_count > current_successful_reset_conditions:
            increment = new_successful_count - current_successful_reset_conditions
            current_successful_reset_conditions = new_successful_count
            pbar.update(increment)

        # Count total reset conditions evaluated (sum across all environments)
        num_reset_conditions_evaluated += dones.sum().item()

        if base_env.sim.is_stopped():
            break

    pbar.close()

    # Get final statistics
    final_successful_reset_conditions = base_env.recorder_manager.exported_successful_episode_count
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
