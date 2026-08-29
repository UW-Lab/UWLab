# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a trained diffusion policy."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play policy trained using diffusion policy for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to diffusion policy checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel.")
parser.add_argument(
    "--num_trajectories",
    type=int,
    default=100,
    help="Number of trajectories to evaluate. If None, run until simulation is stopped.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--use_amp", action="store_true", default=False, help="Use automatic mixed precision.")
parser.add_argument("--save_video", action="store_true", default=False, help="Save video of the policy.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, remaining_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import random
import torch
from contextlib import nullcontext
from tqdm import tqdm

import dill
import hydra
import imageio
import isaaclab_tasks  # noqa: F401
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# Diffusion policy imports
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

# Import the Diffusion policy wrapper
from uwlab_rl.wrappers.diffusion import DiffusionPolicyWrapper

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_policy(ckpt_path: str, device: torch.device, use_ema: bool = False) -> BaseImagePolicy:
    with open(ckpt_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    return policy.eval().to(device)


def _discover_cameras(obs_dict, env):
    """Return (cam_keys, scene_cam_names) for video recording."""
    cam_keys = sorted(k for k in obs_dict["policy"] if "rgb" in k)
    if cam_keys:
        return cam_keys, []
    scene_cam_names = sorted(
        name
        for name, sensor in env.unwrapped.scene._sensors.items()
        if hasattr(sensor, "data") and hasattr(sensor.data, "output") and "rgb" in sensor.data.output
    )
    if scene_cam_names:
        print(f"Using scene cameras for video: {scene_cam_names}")
    return cam_keys, scene_cam_names


def _capture_frame(obs_dict, env, env_idx: int, cam_keys: list, scene_cam_names: list) -> np.ndarray | None:
    """Capture and concatenate camera images for one environment."""
    imgs = []
    if cam_keys:
        for cam in cam_keys:
            img = obs_dict["policy"][cam][env_idx].detach().cpu().permute(1, 2, 0).numpy()
            imgs.append((img * 255).clip(0, 255).astype("uint8"))
    elif scene_cam_names:
        for cam_name in scene_cam_names:
            img = env.unwrapped.scene._sensors[cam_name].data.output["rgb"][env_idx].detach().cpu().numpy()
            if img.shape[0] in [1, 3, 4] and img.shape[0] < img.shape[1]:
                img = img.transpose(1, 2, 0)
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype("uint8")
            if img.shape[-1] == 4:
                img = img[..., :3]
            imgs.append(img)
    return np.concatenate(imgs, axis=1) if imgs else None


def _count_successes(env, reset_ids: torch.Tensor, term_names: list[str]) -> int:
    count = 0
    term_dones = env.unwrapped.termination_manager._term_dones[reset_ids]
    for term_row in term_dones:
        active = term_row.nonzero(as_tuple=False).flatten().cpu().tolist()
        if any(term_names[idx] == "success" for idx in active):
            count += 1
    return count


def _collect_metrics(infos: dict, episode_metrics: dict):
    if "log" not in infos:
        return
    for key, value in infos["log"].items():
        if key.startswith("Metrics/") or key.startswith("Episode_Reward/"):
            episode_metrics.setdefault(key, []).append(value)


def _print_results(episodes: int, successful_episodes: int, episode_metrics: dict):
    print("\nFinal Statistics:")
    print(f"Total trajectories evaluated: {episodes}")
    if successful_episodes > 0 or "Episode_Termination/success" in episode_metrics:
        print(f"Successful trajectories: {successful_episodes}")
        print(f"Success rate: {successful_episodes / episodes * 100:.2f}%")
    else:
        print("Success rate: Not calculable (success metric not found in environment)")
    if episode_metrics:
        print("\nAverage Metrics:")
        for metric_name, values in sorted(episode_metrics.items()):
            if values:
                floats = [float(v) if isinstance(v, torch.Tensor) else v for v in values]
                print(f"{metric_name}: {sum(floats) / len(floats):.4f}")


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    """Run a trained diffusion policy with Isaac Lab environment."""
    _set_seeds(args_cli.seed)

    device = torch.device(args_cli.device if args_cli.device else "cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.concatenate_terms = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    policy = _load_policy(args_cli.checkpoint, device)
    wrapped_policy = DiffusionPolicyWrapper(policy, device, n_obs_steps=policy.n_obs_steps, num_envs=args_cli.num_envs)

    obs_dict, _ = env.reset()
    dones = torch.ones(args_cli.num_envs, dtype=torch.bool, device=device)
    wrapped_policy.reset((dones > 0).nonzero(as_tuple=False).reshape(-1))

    term_names = env.unwrapped.termination_manager._term_names  # type: ignore
    assert "success" in term_names, "Success term not found in termination manager"

    episodes, steps, successful_episodes = 0, 0, 0
    episode_metrics: dict = {}

    pbar = None
    if args_cli.num_trajectories is not None:
        pbar = tqdm(total=args_cli.num_trajectories, desc="Evaluating trajectories (Success: 0.00%)")

    # Video recording state
    cam_keys, scene_cam_names, env_frames, frames_to_save = [], [], [], []
    if args_cli.save_video:
        cam_keys, scene_cam_names = _discover_cameras(obs_dict, env)
        env_frames = [[] for _ in range(args_cli.num_envs)]

    while simulation_app.is_running():
        if args_cli.num_trajectories is not None and episodes >= args_cli.num_trajectories:
            print(f"\nReached target number of trajectories ({args_cli.num_trajectories}). Stopping evaluation.")
            break

        with torch.inference_mode(), torch.autocast(device_type=device.type) if args_cli.use_amp else nullcontext():
            actions = wrapped_policy.predict_action(obs_dict)

            if args_cli.save_video:
                for i in range(args_cli.num_envs):
                    frame = _capture_frame(obs_dict, env, i, cam_keys, scene_cam_names)
                    if frame is not None:
                        env_frames[i].append(frame)

            step_result = env.step(actions)
            if len(step_result) == 4:
                obs_dict, rewards, dones, infos = step_result
            else:
                obs_dict, rewards, terminated, truncated, infos = step_result
                dones = terminated | truncated

            steps += 1

            if isinstance(dones, torch.Tensor):
                new_ids = (dones > 0).nonzero(as_tuple=False)
                episodes += len(new_ids)
            elif dones:
                new_ids = [0]
                episodes += 1
            else:
                new_ids = []

            if isinstance(dones, torch.Tensor) and dones.any():
                reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
                successful_episodes += _count_successes(env, reset_ids, term_names)
                wrapped_policy.reset(reset_ids)
                _collect_metrics(infos, episode_metrics)
                steps = 0

                if args_cli.save_video:
                    for i in reset_ids:
                        frames_to_save.extend(env_frames[i])
                        env_frames[i] = []
                    imageio.mimsave("policy_cameras.mp4", frames_to_save, fps=10, codec="libx264")

                if pbar is not None:
                    pbar.update(len(new_ids))
                    rate = (successful_episodes / episodes * 100) if episodes > 0 else 0.0
                    pbar.set_description(f"Evaluating trajectories (Success: {rate:.2f}%)")

    _print_results(episodes, successful_episodes, episode_metrics)
    if pbar is not None:
        pbar.close()
    env.close()


if __name__ == "__main__":
    # run the main function - the decorator handles parameter passing
    main()  # type: ignore
    # close sim app
    simulation_app.close()
