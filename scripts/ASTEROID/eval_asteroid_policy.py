# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
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
parser.add_argument("--episode_length_s", type=float, default=24.0, help="Episode length in seconds.")
parser.add_argument("--exp_name", type=str, default="diffusion_policy_eval", help="Experiment name for logging.")
parser.add_argument("--wandb_project", type=str, default="asteroid", help="WandB project name for logging.")
parser.add_argument("--wandb_group", type=str, default=None, help="WandB group name for logging.")
parser.add_argument("--iteration", type=int, default=0, help="Iteration number for logging.")
# append AppLauncher cli args
parser.add_argument(
    "--student_devices",
    type=str,
    default=None,
    help=(
        "Comma-separated devices for student inference (e.g. cuda:1,cuda:2). Default: every visible GPU"
        " except the simulation device, falling back to the simulation device."
    ),
)
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

import isaaclab_tasks  # noqa: F401
import wandb
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from student_policy import MultiDeviceStudent, resolve_student_devices

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose


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

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Override configurations with CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # simulation and action buffers share the sim device; the student may use other GPUs
    device = torch.device(env_cfg.sim.device)
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    # Set environment seed
    env_cfg.seed = args_cli.seed
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

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

    # Load the student, replicated across the student devices
    student_devices = resolve_student_devices(device, args_cli.student_devices)
    print(f"[eval] Student inference devices: {[str(d) for d in student_devices]}")
    wrapped_policy = MultiDeviceStudent.from_checkpoint(
        args_cli.checkpoint, student_devices, args_cli.num_envs, output_device=device, use_amp=args_cli.use_amp
    )
    print(wrapped_policy.shards[0].policy)

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
    successes = []

    # Initialize progress bar if num_trajectories is specified
    pbar = None
    if args_cli.num_trajectories is not None:
        pbar = tqdm(total=args_cli.num_trajectories, desc="Evaluating trajectories (Success: 0.00%)")

    # simulate environment
    if args_cli.save_video:
        env_frames = [[] for _ in range(args_cli.num_envs)]
        cam_keys = sorted([key for key in obs_dict["policy"].keys() if "rgb" in key])

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
            actions = wrapped_policy.predict_action(obs_dict).to(device)

            # first step after reset: hold the arm still with the gripper closed
            first_step_mask = episode_steps == 0
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
                            img = obs_dict["policy"][cam][i].detach().cpu().permute(1, 2, 0).numpy()
                            img = (img * 255).clip(0, 255).astype("uint8")
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
                    if args_cli.save_video and len(env_frames[env_id]) > 0:
                        v = np.asarray(env_frames[env_id])  # T,H,W,C uint8
                        v = v.transpose(0, 3, 1, 2)  # T,C,H,W
                        if is_success > 0:
                            video_log[f"eval/success_video_{success_video_count}"] = wandb.Video(
                                v, fps=10, format="mp4"
                            )
                            success_video_count += 1
                        else:
                            video_log[f"eval/fail_video_{fail_video_count}"] = wandb.Video(v, fps=10, format="mp4")
                            fail_video_count += 1
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
                        env_frames[i] = []

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
