# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a trained diffusion policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# Shim rsl_rl: lti env's editable install points at UWLab-ICL's fork (dict obs);
# UWLab uses the UW-Lab/rsl_rl fork (Tensor obs).
_UWLAB_RSL_RL = "/mnt/storage/lti/UWLab/.uwlab_rsl_rl"
if os.path.isdir(_UWLAB_RSL_RL):
    sys.meta_path[:] = [
        f for f in sys.meta_path
        if not (isinstance(f, type) and getattr(f, "__module__", "").startswith("__editable___rsl_rl"))
    ]
    if _UWLAB_RSL_RL not in sys.path:
        sys.path.insert(0, _UWLAB_RSL_RL)
    for _mod_name in list(sys.modules):
        if _mod_name == "rsl_rl" or _mod_name.startswith("rsl_rl."):
            del sys.modules[_mod_name]

# Shim diffusion_policy: lti env's editable install points at UWLab-ICL's fork
# (which wraps mean_head/log_std_head inside an OutputHead). UWLab's submodule
# uses flat mean_head/log_std_head, and train.py wrote checkpoints in that
# format because it ran from inside UWLab/diffusion_policy/ where sys.path[0]
# beats the editable finder. eval_distilled_policy.py runs from a different
# cwd, so without this shim it loads UWLab-ICL's class and fails on state-dict
# key mismatch when loading the ckpt.
_UWLAB_DP = "/mnt/storage/lti/UWLab/diffusion_policy"
if os.path.isdir(os.path.join(_UWLAB_DP, "diffusion_policy")):
    sys.meta_path[:] = [
        f for f in sys.meta_path
        if not (isinstance(f, type) and getattr(f, "__module__", "").startswith("__editable___diffusion_policy"))
    ]
    if _UWLAB_DP not in sys.path:
        sys.path.insert(0, _UWLAB_DP)
    for _mod_name in list(sys.modules):
        if _mod_name == "diffusion_policy" or _mod_name.startswith("diffusion_policy."):
            del sys.modules[_mod_name]

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play policy trained using diffusion policy for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to diffusion policy checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel.")
parser.add_argument("--num_trajectories", type=int, default=100, help="Number of trajectories to evaluate. If None, run until simulation is stopped.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--use_amp", action="store_true", default=False, help="Use automatic mixed precision.")
parser.add_argument("--save_video", action="store_true", default=False, help="Save video of the policy.")
parser.add_argument("--episode_length_s", type=float, default=24.0, help="Episode length in seconds.")
parser.add_argument("--exp_name", type=str, default="diffusion_policy_eval", help="Experiment name for logging.")
parser.add_argument("--wandb_project", type=str, default="diffusion_policy_eval_new", help="WandB project name for logging.")
parser.add_argument("--wandb_group", type=str, default="default_group", help="WandB group name for logging.")
parser.add_argument("--iteration", type=int, default=0, help="Iteration number for logging.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, remaining_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Drop Isaac Sim's pip_prebundle so conda env's trimesh/rtree win on later imports.
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
sys.path[:] = [p for p in sys.path if "pip_prebundle" not in p]
for _mod_name in list(sys.modules):
    if _mod_name == "trimesh" or _mod_name.startswith("trimesh.") or \
       _mod_name == "rtree" or _mod_name.startswith("rtree."):
        _mod = sys.modules.get(_mod_name)
        _mod_file = getattr(_mod, "__file__", None) or ""
        if "pip_prebundle" in _mod_file:
            del sys.modules[_mod_name]

"""Rest everything follows."""

import gymnasium as gym
import torch
import dill
import hydra
from contextlib import nullcontext
from tqdm import tqdm
import random
import numpy as np
import wandb 

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from uwlab_tasks.utils.hydra import hydra_task_compose

# Diffusion policy imports
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# Import the Diffusion policy wrapper
from uwlab_rl.wrappers.diffusion import DiffusionPolicyWrapper

# import imageio


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    """Run a trained diffusion policy with Isaac Lab environment."""
    # Set seeds for reproducibility
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check device is available
    device = torch.device("cuda:0")#torch.device(args_cli.device if args_cli.device else 'cuda' if torch.cuda.is_available() else 'cpu')
    policy_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Override configurations with CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    # Set environment seed
    env_cfg.seed = args_cli.seed
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # ckpt_base = args_cli.checkpoint.split("/")[-1].split(".")[0] if args_cli.checkpoint else "none"
    wandb.init(
        project=args_cli.wandb_project,
        group=args_cli.wandb_group,
        config={
            "task": args_cli.task,
            "checkpoint": args_cli.checkpoint,
            "num_envs": args_cli.num_envs,
            "num_trajectories": args_cli.num_trajectories,
            "seed": args_cli.seed,
            "use_amp": args_cli.use_amp,
            "save_video": args_cli.save_video,
            "episode_length_s": args_cli.episode_length_s,
            "exp_name": args_cli.exp_name,
            "iteration": args_cli.iteration,
        },
    )

    # Load diffusion policy checkpoint
    ckpt_path = args_cli.checkpoint
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Load policy based on configuration
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    # print policy summary
    print(policy)

    policy.eval().to(policy_device)

    # Wrap policy to handle Isaac Lab observations
    wrapped_policy = DiffusionPolicyWrapper(policy, policy_device, n_obs_steps=policy.n_obs_steps, num_envs=args_cli.num_envs)

    # reset environment
    obs_dict, _ = env.reset()
    dones = torch.ones(args_cli.num_envs, dtype=torch.bool, device=device)
    reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
    wrapped_policy.reset(reset_ids)

    # Get termination term names to identify success
    term_names = env.unwrapped.termination_manager._term_names  # type: ignore
    assert "success" in term_names, "Success term not found in termination manager"

    episodes = 0
    steps = 0
    successful_episodes = 0  # Track successful episodes

    # Track all episode metrics
    episode_metrics = {}

    ep_returns = torch.zeros(args_cli.num_envs, dtype=torch.float32, device=device)
    success_video_count = 0
    fail_video_count = 0
    render_frames = []
    successes = []

    # Initialize progress bar if num_trajectories is specified
    pbar = None
    if args_cli.num_trajectories is not None:
        pbar = tqdm(total=args_cli.num_trajectories, desc="Evaluating trajectories (Success: 0.00%)")

    # simulate environment
    if args_cli.save_video:
        env_frames = [[] for _ in range(args_cli.num_envs)]
        frames_to_save = []
        cam_keys = sorted([key for key in obs_dict['policy'].keys() if 'rgb' in key])

    while simulation_app.is_running():
        # Check if we've reached the desired number of trajectories
        if args_cli.num_trajectories is not None and episodes >= args_cli.num_trajectories:
            if pbar is not None:
                pbar.close()
            print(f"\nReached target number of trajectories ({args_cli.num_trajectories}). Stopping evaluation.")
            break

        # run everything in inference mode
        with torch.inference_mode(), torch.autocast(device_type=device.type) if args_cli.use_amp else nullcontext():
            # compute actions using wrapped diffusion policy
            episode_steps = env.unwrapped.episode_length_buf
            # first_step_mask = (episode_steps == 0).to(device)
            # action_ids = ~first_step_mask.nonzero(as_tuple=False).reshape(-1).to(device)
            # actions = torch.zeros((args_cli.num_envs, env.action_space.shape[0]), device=device)
            # actions[:, -1] = -1.0  # default noop for last action dimension

            # if len(action_ids) > 0:
            # actions[action_ids] = wrapped_policy.predict_action(obs_dict, action_ids.tolist()).to(device)
            actions = wrapped_policy.predict_action(obs_dict).to(device)

            first_step_mask = (episode_steps == 0)
            if torch.any(first_step_mask):
                actions[first_step_mask, :-1] = 0.0
                actions[first_step_mask, -1] = -1.0  # close gripper

            if args_cli.save_video:
                if len(cam_keys) == 0:
                    frame = env.render()
                    env_frames[0].append(frame)
                else:
                    for i in range(args_cli.num_envs):
                        imgs = []

                        for cam in cam_keys:
                            img = obs_dict['policy'][cam][i].detach().cpu().permute(1, 2, 0).numpy()
                            img = (img * 255).clip(0, 255).astype('uint8')
                            imgs.append(img)
                        frame = np.concatenate(imgs, axis=1)
                        env_frames[i].append(frame)

            # apply actions using environment
            step_result = env.step(actions)
            if len(step_result) == 4:
                obs_dict, rewards, dones, infos = step_result
            else:
                # Handle gymnasium v0.26+ format with 5 return values
                obs_dict, rewards, terminated, truncated, infos = step_result
                dones = terminated | truncated

            steps += 1

            rewards_t = rewards if isinstance(rewards, torch.Tensor) else torch.as_tensor(rewards, device=device)
            ep_returns += rewards_t.to(device)

            # Clear data for completed episodes
            new_ids = []
            if isinstance(dones, torch.Tensor):
                new_ids = (dones > 0).nonzero(as_tuple=False)
                episodes += len(new_ids)
            else:
                # Handle scalar done value
                if dones:
                    episodes += 1
                    new_ids = [0]  # Single episode done

            if isinstance(dones, torch.Tensor) and dones.any():
                reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
                num_new_successes = 0

                term_dones = env.unwrapped.termination_manager._term_dones[reset_ids]  # type: ignore
                ep_success_flags = []
                for env_idx, term_row in enumerate(term_dones):
                    active_term_idx = term_row.nonzero(as_tuple=False)
                    is_success = False
                    if active_term_idx.numel() > 0:
                        # Handle multiple active termination conditions
                        active_term_indices = active_term_idx.flatten().cpu().tolist()
                        for term_idx in active_term_indices:
                            if term_names[term_idx] == "success":
                                num_new_successes += 1
                                is_success = True
                                break  # Count each environment only once
                    ep_success_flags.append(is_success)
                    successes.append(is_success)

                successful_episodes += num_new_successes

                for k, env_id in enumerate(reset_ids.detach().cpu().tolist()):
                    video_log = {}
                    ep_ret = float(ep_returns[env_id].detach().cpu().item())
                    is_success = float(ep_success_flags[k])
                    video_log["eval/episode_return"] = ep_ret
                    video_log["eval/success"] = is_success
                    # video_log["eval/success_rate_running"] = (successful_episodes / episodes) if episodes > 0 else 0.0
                    if args_cli.save_video and len(env_frames[env_id]) > 0:
                        v = np.asarray(env_frames[env_id])  # T,H,W,C uint8
                        v = v.transpose(0, 3, 1, 2)  # T,C,H,W
                        if is_success > 0:
                            video_log[f"eval/success_video_{success_video_count}"] = wandb.Video(v, fps=10, format="mp4")
                            success_video_count += 1
                        else:
                            video_log[f"eval/fail_video_{fail_video_count}"] = wandb.Video(v, fps=10, format="mp4")
                            fail_video_count += 1
                        # video_log[f"eval/video_{video_count}"] = wandb.Video(v, fps=10, format="mp4")
                        # video_count += 1
                    wandb.log(video_log)

                ep_returns[reset_ids] = 0.0

                wrapped_policy.reset(reset_ids)

                # Store metrics for completed episodes
                if "log" in infos:
                    # Store all metrics from this episode
                    for key, value in infos["log"].items():
                        if key.startswith("Metrics/") or key.startswith("Episode_Reward/"):
                            if key not in episode_metrics:
                                episode_metrics[key] = []
                            episode_metrics[key].append(value)

                steps = 0

                if args_cli.save_video:
                    for i in reset_ids:
                        frames_to_save.extend(env_frames[i])
                        env_frames[i] = []
                    # imageio.mimsave("logs/policy_cameras.mp4", frames_to_save, fps=10, codec='libx264')

                # Update progress bar with success rate
                if pbar is not None:
                    pbar.update(len(new_ids))
                    success_rate = (successful_episodes / episodes * 100) if episodes > 0 else 0.0
                    pbar.set_description(f"Evaluating trajectories (Success: {success_rate:.2f}%)")

    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Total trajectories evaluated: {episodes}")
    if successful_episodes > 0 or "Episode_Termination/success" in episode_metrics:
        print(f"Successful trajectories: {successful_episodes}")
        print(f"Success rate: {successful_episodes/episodes*100:.2f}%")
    else:
        print("Success rate: Not calculable (success metric not found in environment)")

    # Print metrics statistics
    if episode_metrics:
        print("\nAverage Metrics:")
        for metric_name, values in sorted(episode_metrics.items()):
            if values:  # Only print if we have values
                values = [float(v) if isinstance(v, torch.Tensor) else v for v in values]
                mean = sum(values) / len(values)
                print(f"{metric_name}: {mean:.4f}")

    final_stats = {
        "eval/num_episodes": episodes,
        "eval/successful_episodes": successful_episodes,
        "eval/success_rate": (successful_episodes / episodes) if episodes > 0 else 0.0,
    }
    if episode_metrics:
        for metric_name, values in sorted(episode_metrics.items()):
            if values:
                vals = [float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v) for v in values]
                final_stats[f"eval/{metric_name}_mean"] = float(np.mean(vals))
    wandb.log(final_stats)
    wandb.finish()

    # Cleanup
    if pbar is not None:
        pbar.close()
    env.close()


if __name__ == "__main__":
    # run the main function - the decorator handles parameter passing
    main()  # type: ignore
    # close sim app
    simulation_app.close()
