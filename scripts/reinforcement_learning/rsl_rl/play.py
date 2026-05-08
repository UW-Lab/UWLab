# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

# Shim rsl_rl: lti conda env's editable install points at UWLab-ICL's fork (dict obs);
# UWLab uses the UW-Lab/rsl_rl fork (Tensor obs). Override before any rsl_rl import.
import os as _os
_UWLAB_RSL_RL = "/mnt/storage/lti/UWLab/.uwlab_rsl_rl"
if _os.path.isdir(_UWLAB_RSL_RL):
    sys.meta_path[:] = [
        f for f in sys.meta_path
        if not (isinstance(f, type) and getattr(f, "__module__", "").startswith("__editable___rsl_rl"))
    ]
    if _UWLAB_RSL_RL not in sys.path:
        sys.path.insert(0, _UWLAB_RSL_RL)
    for _mod_name in list(sys.modules):
        if _mod_name == "rsl_rl" or _mod_name.startswith("rsl_rl."):
            del sys.modules[_mod_name]

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
parser.add_argument("--max_episodes", type=int, default=None,
                    help="Stop after this many completed episodes and write a JSON.")
parser.add_argument("--eval_output", type=str, default=None, help="Path to write per-eval JSON.")
parser.add_argument("--discretize_actions", action="store_true", default=False,
                    help="Snap arm dims to bin grid + binarize gripper sign before env.step.")
parser.add_argument("--num_bins", type=int, default=51, help="Bins per arm dim for --discretize_actions.")
parser.add_argument("--action_bound", type=float, default=25.0, help="Symmetric clip range for arm dims.")
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
import os
import time
import torch

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

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from uwlab_rl.rsl_rl.exporter import export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # make config compatible with installed rsl-rl version
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

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

    # Discretizer setup (UWLab-ICL collect_demos_asteroid scheme): arm dims 0-5
    # snap to evenly-spaced bin centers in [-bound, +bound]; gripper dim 6 sign-
    # thresholded to {-1, +1}. Mirrors the inference-side discretization used by
    # the categorical-head distillation student.
    bin_centers = None
    if args_cli.discretize_actions:
        bin_centers = torch.linspace(
            -args_cli.action_bound, args_cli.action_bound, args_cli.num_bins,
            device=env.unwrapped.device, dtype=torch.float32,
        )
        print(f"[eval] discretize ON: arm bins={args_cli.num_bins} ±{args_cli.action_bound}, gripper sign-thresh")

    # Termination bookkeeping (success vs timeout vs other)
    eval_episodes = 0
    eval_term_success = 0
    eval_term_timeout = 0
    eval_term_other = 0
    has_success_term = False
    has_timeout_term = False
    if args_cli.max_episodes is not None:
        term_mgr = env.unwrapped.termination_manager
        active = list(term_mgr.active_terms)
        has_success_term = "success" in active
        has_timeout_term = "time_out" in active
        print(f"[eval] termination terms: {active}")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if bin_centers is not None:
                a = actions.clamp(-args_cli.action_bound, args_cli.action_bound)
                # arm dims (0..5): snap to nearest bin center
                for d in range(min(6, actions.shape[-1])):
                    diff = (a[:, d:d+1] - bin_centers.unsqueeze(0)).abs()
                    actions[:, d] = bin_centers[diff.argmin(dim=-1).squeeze(-1)]
                # gripper dim (6): hard sign-threshold
                if actions.shape[-1] >= 7:
                    actions[:, 6] = torch.where(
                        actions[:, 6] >= 0,
                        torch.ones_like(actions[:, 6]),
                        -torch.ones_like(actions[:, 6]),
                    )
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

            if args_cli.max_episodes is not None and isinstance(dones, torch.Tensor) and dones.any():
                done_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
                term_mgr = env.unwrapped.termination_manager
                if has_success_term:
                    eval_term_success += int(term_mgr.get_term("success")[done_ids].sum().item())
                if has_timeout_term:
                    eval_term_timeout += int(term_mgr.get_term("time_out")[done_ids].sum().item())
                acc = (
                    (int(term_mgr.get_term("success")[done_ids].sum().item()) if has_success_term else 0)
                    + (int(term_mgr.get_term("time_out")[done_ids].sum().item()) if has_timeout_term else 0)
                )
                eval_term_other += int(done_ids.numel()) - acc
                eval_episodes += int(done_ids.numel())
                rate_s = eval_term_success / max(eval_episodes, 1)
                rate_t = eval_term_timeout / max(eval_episodes, 1)
                print(f"[eval] episodes={eval_episodes}/{args_cli.max_episodes} "
                      f"term_success={eval_term_success} ({rate_s:.4f}) "
                      f"term_timeout={eval_term_timeout} ({rate_t:.4f}) "
                      f"term_other={eval_term_other}", flush=True)
                if eval_episodes >= args_cli.max_episodes:
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

    if args_cli.max_episodes is not None and args_cli.eval_output is not None:
        import json
        rate_s = eval_term_success / max(eval_episodes, 1)
        rate_t = eval_term_timeout / max(eval_episodes, 1)
        result = {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "discretize_actions": bool(args_cli.discretize_actions),
            "num_bins": args_cli.num_bins,
            "action_bound": args_cli.action_bound,
            "num_envs": env_cfg.scene.num_envs,
            "num_episodes": eval_episodes,
            "term_success": eval_term_success,
            "term_success_rate": rate_s,
            "term_timeout": eval_term_timeout,
            "term_timeout_rate": rate_t,
            "term_other": eval_term_other,
        }
        os.makedirs(os.path.dirname(args_cli.eval_output) or ".", exist_ok=True)
        with open(args_cli.eval_output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[eval] wrote {args_cli.eval_output}: term_success={rate_s:.4f}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
