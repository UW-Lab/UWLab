# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate checkpoint robustness under action noise for distillation selection.

All policies run simultaneously in the same env, each controlling a disjoint
slice of environments. This ensures identical resets/randomization and makes
results independent of checkpoint ordering.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "reinforcement_learning", "rsl_rl")
)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate checkpoints for distillation robustness.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate (split evenly).")
parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint paths to evaluate.")
parser.add_argument("--eval_steps", type=int, default=1000, help="Number of env steps to run.")
parser.add_argument("--action_noise", type=float, default=2.0, help="Std of Gaussian noise added to actions.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.omnireset import mdp as task_mdp
from uwlab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate checkpoints under action noise and rank by success throughput."""
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    num_policies = len(args_cli.checkpoints)
    num_envs = env_cfg.scene.num_envs
    envs_per_policy = num_envs // num_policies
    slices = []
    for i in range(num_policies):
        start = i * envs_per_policy
        end = (i + 1) * envs_per_policy if i < num_policies - 1 else num_envs
        slices.append((start, end))

    env_cfg.terminations.success = DoneTerm(
        func=task_mdp.consecutive_success_state_with_min_length,
        params={"num_consecutive_successes": 5, "min_episode_length": 10},
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    term_names = env.unwrapped.termination_manager._term_names
    assert "success" in term_names, f"'success' not in termination terms: {term_names}"
    success_idx = term_names.index("success")

    policies = []
    policy_nns = []
    for ckpt_path in args_cli.checkpoints:
        resume_path = retrieve_file_path(ckpt_path)
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policies.append(runner.get_inference_policy(device=env.unwrapped.device))
        try:
            policy_nns.append(runner.alg.policy)
        except AttributeError:
            policy_nns.append(runner.alg.actor_critic)

    print(f"\n{'=' * 60}")
    print(f"Running {num_policies} policies across {num_envs} envs")
    print(f"Action noise std: {args_cli.action_noise}")
    print(f"Eval steps: {args_cli.eval_steps}")
    for i, ckpt in enumerate(args_cli.checkpoints):
        s, e = slices[i]
        print(f"  Policy {i}: envs [{s}:{e}] ({e - s} envs) <- {os.path.basename(ckpt)}")
    print(f"{'=' * 60}")

    total_successes = [0] * num_policies
    total_episodes = [0] * num_policies

    obs = env.get_observations()

    for step in range(args_cli.eval_steps):
        with torch.inference_mode():
            action_slices = []
            for i, policy in enumerate(policies):
                s, e = slices[i]
                action_slices.append(policy(obs)[s:e])
            actions = torch.cat(action_slices, dim=0)
            actions = actions + args_cli.action_noise * torch.randn_like(actions)
            obs, _, dones, extras = env.step(actions)
            for pnn in policy_nns:
                pnn.reset(dones)

        if dones.any():
            reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
            term_dones = env.unwrapped.termination_manager._term_dones[reset_ids]

            for env_id, term_row in zip(reset_ids, term_dones):
                eid = env_id.item()
                pidx = next(i for i, (s, e) in enumerate(slices) if s <= eid < e)
                total_episodes[pidx] += 1
                active = term_row.nonzero(as_tuple=False).flatten().cpu().tolist()
                if success_idx in active:
                    total_successes[pidx] += 1

    print(f"\n{'=' * 60}")
    print(f"RANKING BY THROUGHPUT (action_noise={args_cli.action_noise}, steps={args_cli.eval_steps})")
    print(f"{'=' * 60}")
    ranking = []
    for i, ckpt in enumerate(args_cli.checkpoints):
        rate = total_successes[i] / total_episodes[i] if total_episodes[i] > 0 else 0.0
        ranking.append((ckpt, total_successes[i], total_episodes[i], rate))
    ranking.sort(key=lambda x: x[1], reverse=True)
    for rank, (ckpt, succ, eps, rate) in enumerate(ranking, 1):
        print(f"  #{rank}: {os.path.basename(ckpt)}")
        print(f"       successes={succ}  episodes={eps}  rate={rate:.1%}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
