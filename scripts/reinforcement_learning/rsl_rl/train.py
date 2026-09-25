# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--resume_path", type=str, default=None,
    help="Direct path to a checkpoint file to resume from (bypasses log directory search).",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# --- null-space preference critic -------------------------------------------------------------
parser.add_argument("--beta", type=float, default=None, help="Preference step budget. 0 == baseline PPO.")
parser.add_argument(
    "--pref_source", type=str, default=None, choices=["zero", "noise", "action_rate", "ee_height", "terms"],
    help="Preference stream: zero (sanity A), noise (sanity B), action_rate (noise-bait probe), "
         "terms (scripted predicates).",
)
parser.add_argument("--pref_noise_std", type=float, default=None, help="Std for --pref_source=noise.")
parser.add_argument(
    "--pref_terms", type=str, default=None,
    help="Comma-separated RewardManager term names for --pref_source=terms.",
)
parser.add_argument(
    "--critic_arch", type=str, default=None, choices=["shared", "separate"],
    help="Second value head as a widened shared trunk, or an independent critic MLP.",
)
parser.add_argument("--gamma_pref", type=float, default=None, help="Discount for the preference return.")
parser.add_argument(
    "--pref_mask_noise", type=lambda v: v.lower() not in ("0", "false", "no"), default=None,
    help="Layer 1: keep the preference gradient off the exploration-noise params (default true).",
)
parser.add_argument(
    "--pref_detach_noise_features", action="store_true", default=False,
    help="Layer 2: also detach the trunk features feeding the gSDE noise head in the pref surrogate.",
)
parser.add_argument(
    "--projection_mode", type=str, default=None, choices=["gradient", "advantage", "sum"],
    help="gradient = faithful null-space projection; advantage/sum = ablations.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from uwlab_rl.rsl_rl.nullspace import (
    ActionRatePreference,
    DualCriticOnPolicyRunner,
    EndEffectorHeightPreference,
    DualRewardVecEnvWrapper,
    GaussianNoisePreference,
    RewardManagerTermsPreference,
    ZeroPreference,
)

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # --- null-space preference critic CLI overrides ---
    # Applied before sanitize_rsl_rl_cfg, which only strips keys for algorithm classes it can
    # resolve inside rsl_rl.algorithms; NullspacePPO lives in uwlab_rl, so these survive.
    if args_cli.beta is not None:
        agent_cfg.algorithm.beta = args_cli.beta
    if args_cli.gamma_pref is not None:
        agent_cfg.algorithm.gamma_pref = args_cli.gamma_pref
    if args_cli.projection_mode is not None:
        agent_cfg.algorithm.projection_mode = args_cli.projection_mode
    if args_cli.pref_mask_noise is not None:
        agent_cfg.algorithm.pref_mask_noise = args_cli.pref_mask_noise
    if args_cli.pref_detach_noise_features:
        agent_cfg.algorithm.pref_detach_noise_features = True
    if args_cli.critic_arch is not None:
        agent_cfg.policy.critic_arch = args_cli.critic_arch
    if args_cli.pref_source is not None:
        agent_cfg.pref_source = args_cli.pref_source
    if args_cli.pref_noise_std is not None:
        agent_cfg.pref_noise_std = args_cli.pref_noise_std
    if args_cli.pref_terms is not None:
        agent_cfg.pref_term_names = tuple(n.strip() for n in args_cli.pref_terms.split(",") if n.strip())

    # make config compatible with installed rsl-rl version
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if args_cli.resume_path is not None:
        resume_path = retrieve_file_path(args_cli.resume_path)
        agent_cfg.resume = True
    elif agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    if agent_cfg.class_name == "DualCriticOnPolicyRunner":
        # The dual-critic path needs a second reward stream. DualRewardVecEnvWrapper reads the
        # RewardManager's already-materialised per-term buffer rather than restructuring the
        # manager -- OmniReset's `progress_context` term returns zeros but caches state that the
        # reward, terminations, reset curriculum and data-collection configs all read back, so
        # splitting or reweighting the manager silently corrupts the reward.
        source_name = getattr(agent_cfg, "pref_source", "zero")
        if source_name == "zero":
            pref_source = ZeroPreference()
        elif source_name == "noise":
            pref_source = GaussianNoisePreference(
                std=getattr(agent_cfg, "pref_noise_std", 1.0), seed=agent_cfg.seed
            )
        elif source_name == "action_rate":
            # Noise-bait probe: a preference maximally satisfiable by shrinking exploration.
            pref_source = ActionRatePreference()
        elif source_name == "ee_height":
            # High-conflict preference: fights the lift the task requires.
            pref_source = EndEffectorHeightPreference()
        elif source_name == "terms":
            term_names = list(getattr(agent_cfg, "pref_term_names", ()))
            if not term_names:
                raise ValueError("--pref_source=terms requires --pref_terms=<comma,separated,names>")
            pref_source = RewardManagerTermsPreference(term_names)
        else:
            raise ValueError(f"Unknown pref_source: {source_name}")
        print(f"[INFO] Preference reward source: {source_name}")
        env = DualRewardVecEnvWrapper(env, pref_source=pref_source, clip_actions=agent_cfg.clip_actions)
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # In-job wandb with a stable run id (gen_convergence_yaml.py sets WANDB_RUN_ID + WANDB_RESUME=allow):
    # every preemption restart resumes the SAME wandb run instead of opening a new one. On resume wandb
    # loads the run's stored config, and rsl_rl's WandbSummaryWriter then re-sends its own config with
    # values that change on every start (log_dir, env_cfg.log_dir, resume settings). wandb raises
    # ConfigError on a changed value unless allow_val_change=True, which would kill the start before
    # training -- so let config updates overwrite, in this process only.
    if agent_cfg.logger == "wandb" and os.environ.get("WANDB_RUN_ID"):
        import wandb.sdk.wandb_config as _wandb_config

        _config_update = _wandb_config.Config.update

        def _update_allow_change(self, d, allow_val_change=None):
            return _config_update(self, d, allow_val_change=True)

        _wandb_config.Config.update = _update_allow_change
        print(f"[INFO] wandb: resuming run id {os.environ['WANDB_RUN_ID']} (config updates may overwrite)")

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DualCriticOnPolicyRunner":
        runner = DualCriticOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
