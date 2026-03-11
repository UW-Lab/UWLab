# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from collections.abc import Mapping

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--collect_dataset",
    action="store_true",
    default=False,
    help="Collect state-action pairs during inference and save successful episodes for supervised learning.",
)
parser.add_argument(
    "--dataset_output",
    type=str,
    default=None,
    help="Output .pt path for collected dataset. Defaults to <run_log_dir>/supervised_dataset.pt",
)
parser.add_argument(
    "--num_successful_episodes",
    type=int,
    default=100,
    help="Stop data collection after this many successful episodes.",
)
parser.add_argument(
    "--success_reward_threshold",
    type=float,
    default=0.1,
    help="Episode is successful if its final reward is above this threshold.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from tqdm import tqdm

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def _flatten_observation_value(value) -> torch.Tensor:
    """Flatten a tensor-like observation structure into `[num_envs, flat_dim]`."""
    if isinstance(value, torch.Tensor):
        return value.reshape(value.shape[0], -1)

    if isinstance(value, Mapping) or (hasattr(value, "keys") and hasattr(value, "__getitem__")):
        flat_chunks = [_flatten_observation_value(value[key]) for key in value.keys()]
        if not flat_chunks:
            raise ValueError("Observation mapping is empty and cannot be flattened.")
        return torch.cat(flat_chunks, dim=-1)

    tensor_value = torch.as_tensor(value)
    return tensor_value.reshape(tensor_value.shape[0], -1)


def _extract_policy_state(obs) -> torch.Tensor:
    """Extract and flatten policy observations into `[num_envs, state_dim]`."""
    if isinstance(obs, Mapping) or (hasattr(obs, "keys") and hasattr(obs, "__getitem__")):
        if "policy" in obs:
            state = obs["policy"]
        else:
            # Fall back to first observation entry if policy group is unavailable.
            first_key = next(iter(obs))
            state = obs[first_key]
    else:
        state = obs
    return _flatten_observation_value(state)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    num_envs = env.num_envs
    state_dim = _extract_policy_state(obs).shape[1]
    action_dim = env.num_actions

    successful_episodes = 0
    total_finished_episodes = 0
    data_pbar = None
    collected_state_chunks: list[torch.Tensor] = []
    collected_action_chunks: list[torch.Tensor] = []
    collected_reward_chunks: list[torch.Tensor] = []
    env_ids_device: torch.Tensor | None = None
    episode_lengths: torch.Tensor | None = None
    episode_states: torch.Tensor | None = None
    episode_actions: torch.Tensor | None = None
    episode_rewards: torch.Tensor | None = None
    if args_cli.collect_dataset:
        max_episode_steps = getattr(env.unwrapped, "max_episode_length", None)
        if max_episode_steps is None:
            raise RuntimeError("Dataset collection requires `env.unwrapped.max_episode_length` to be available.")
        max_episode_steps = int(max_episode_steps)
        collection_device = env.unwrapped.device
        env_ids_device = torch.arange(num_envs, device=collection_device, dtype=torch.long)
        episode_lengths = torch.zeros(num_envs, device=collection_device, dtype=torch.long)
        episode_states = torch.empty((num_envs, max_episode_steps, state_dim), device=collection_device, dtype=torch.float32)
        episode_actions = torch.empty((num_envs, max_episode_steps, action_dim), device=collection_device, dtype=torch.float32)
        episode_rewards = torch.empty((num_envs, max_episode_steps), device=collection_device, dtype=torch.float32)
        data_pbar = tqdm(total=args_cli.num_successful_episodes, desc="Collecting successful episodes", unit="episode")

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            state_t = _extract_policy_state(obs)
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

            if args_cli.collect_dataset:
                assert env_ids_device is not None
                assert episode_lengths is not None
                assert episode_states is not None
                assert episode_actions is not None
                assert episode_rewards is not None
                flat_actions = actions.reshape(actions.shape[0], -1)
                rewards_device = rewards.detach()
                dones_device = dones.detach()
                states_device = state_t.detach()
                actions_device = flat_actions.detach()
                step_ids = episode_lengths.clone()
                episode_states[env_ids_device, step_ids] = states_device
                episode_actions[env_ids_device, step_ids] = actions_device
                episode_rewards[env_ids_device, step_ids] = rewards_device
                episode_lengths += 1

                done_ids = torch.nonzero(dones_device, as_tuple=False).squeeze(-1)
                if done_ids.numel() > 0:
                    total_finished_episodes += int(done_ids.numel())

                    successful_done_mask = rewards_device[done_ids] > args_cli.success_reward_threshold
                    successful_done_ids = done_ids[successful_done_mask]
                    remaining_successes = args_cli.num_successful_episodes - successful_episodes

                    if remaining_successes > 0 and successful_done_ids.numel() > 0:
                        successful_done_ids = successful_done_ids[:remaining_successes]
                        successful_lengths = episode_lengths[successful_done_ids]
                        successful_episodes += int(successful_done_ids.numel())
                        if data_pbar is not None:
                            data_pbar.update(int(successful_done_ids.numel()))

                        max_success_len = int(successful_lengths.max().item())
                        valid_steps = (
                            torch.arange(max_success_len, device=successful_lengths.device, dtype=torch.long).unsqueeze(0)
                            < successful_lengths.unsqueeze(1)
                        )
                        collected_state_chunks.append(episode_states[successful_done_ids, :max_success_len][valid_steps].cpu())
                        collected_action_chunks.append(episode_actions[successful_done_ids, :max_success_len][valid_steps].cpu())
                        collected_reward_chunks.append(episode_rewards[successful_done_ids, :max_success_len][valid_steps].cpu())

                    episode_lengths[done_ids] = 0

                if successful_episodes >= args_cli.num_successful_episodes:
                    print(
                        f"[INFO] Collected {successful_episodes} successful episodes "
                        f"(threshold={args_cli.success_reward_threshold})."
                    )
                    break
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    if data_pbar is not None:
        data_pbar.close()

    if args_cli.collect_dataset:
        if collected_state_chunks:
            collected_states = torch.cat(collected_state_chunks, dim=0)
            collected_actions = torch.cat(collected_action_chunks, dim=0)
            collected_rewards = torch.cat(collected_reward_chunks, dim=0)
        else:
            collected_states = torch.empty((0, state_dim), dtype=torch.float32)
            collected_actions = torch.empty((0, action_dim), dtype=torch.float32)
            collected_rewards = torch.empty((0,), dtype=torch.float32)

        dataset_path = args_cli.dataset_output
        if dataset_path is None:
            dataset_path = os.path.join(log_dir, "supervised_dataset.pt")
        dataset_path = os.path.abspath(dataset_path)
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

        payload = {
            "states": collected_states.float(),
            "actions": collected_actions.float(),
            "rewards": collected_rewards.float(),
            "meta": {
                "task": args_cli.task,
                "checkpoint": resume_path,
                "success_reward_threshold": args_cli.success_reward_threshold,
                "target_successful_episodes": args_cli.num_successful_episodes,
                "successful_episodes_collected": successful_episodes,
                "finished_episodes_seen": total_finished_episodes,
                "num_samples": int(collected_states.shape[0]),
            },
        }
        torch.save(payload, dataset_path)
        print(f"[INFO] Saved supervised dataset to: {dataset_path}")
        print(
            "[INFO] Dataset stats: "
            f"samples={payload['meta']['num_samples']}, "
            f"success_eps={successful_episodes}, "
            f"finished_eps={total_finished_episodes}"
        )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
