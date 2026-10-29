# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations using exploration policy + expert policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import gymnasium as gym
import numpy as np
import os
import random
import time
import torch
from tqdm import tqdm
from typing import Sequence

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations with exploration + expert policy.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.zarr", help="File path to export recorded demos."
)
parser.add_argument("--num_demos", type=int, default=10, help="Number of demonstrations to record.")
parser.add_argument("--exploration_checkpoint", type=str, default=None, help="Path to exploration policy checkpoint.")
parser.add_argument(
    "--min_exploration_horizon", type=float, default=0.02, help="Minimum exploration horizon for exploration policy."
)
parser.add_argument(
    "--max_exploration_horizon", type=float, default=0.3, help="Maximum exploration horizon for exploration policy."
)
parser.add_argument("--episode_length_s", type=float, default=5.0, help="Episode length in seconds.")
parser.add_argument("--expert_noise", type=float, default=0.0, help="Noise level for the expert policy.")
parser.add_argument(
    "--max_exploration_horizon_for_save",
    type=float,
    default=0.9,
    help=(
        "With an exploration policy, only save successful episodes whose explorer steps make up less than"
        " this fraction of the episode."
    ),
)
parser.add_argument(
    "--stall_reset_after_s",
    type=float,
    default=90.0,
    help="Force env.reset() when no new demo has been recorded for this many seconds. 0 disables.",
)
parser.add_argument(
    "--student_devices",
    type=str,
    default=None,
    help=(
        "Comma-separated devices for student inference (e.g. cuda:1,cuda:2). Default: every visible GPU"
        " except the simulation device, falling back to the simulation device."
    ),
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Print collection telemetry and write per-interval action stats to action_stats.csv.",
)

parser.add_argument("--render", action="store_true", help="Whether to render the environment.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for environment.")
parser.add_argument("--video", action="store_true", default=False, help="Record video of env 0 during collection.")
parser.add_argument("--video_length", type=int, default=400, help="Video length in env steps (only used when --video).")
parser.add_argument(
    "--video_interval",
    type=int,
    default=0,
    help="Restart video recording every N env steps. 0 (default) = single video at step 0.",
)
parser.add_argument(
    "--video_dir", type=str, default=None, help="Directory to write videos. Defaults to <dataset_file>_dir/videos."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# Mirror play.py: cameras must be enabled for the RecordVideo wrapper to render.
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers.recorder_manager import DatasetExportMode

# Import dataset handlers
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from student_policy import MultiDeviceStudent, resolve_student_devices

from uwlab.utils.datasets import ZarrDatasetFileHandler

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.asteroid.mdp.recorders.recorders_cfg import (
    AsteroidActionStateRecorderManagerCfg as ActionStateRecorderManagerCfg,
)
from uwlab_tasks.utils.hydra import hydra_task_compose

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def record_pre_reset(self, env_ids: Sequence[int] | None, force_export_or_skip=None) -> None:
    """Trigger recorder terms for pre-reset functions.

    Args:
        env_ids: The environment ids in which a reset is triggered.
    """
    # Do nothing if no active recorder terms are provided
    if len(self.active_terms) == 0:
        return

    if env_ids is None:
        env_ids = list(range(self._env.num_envs))
    if isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.tolist()

    for term in self._terms.values():
        key, value = term.record_pre_reset(env_ids)
        self.add_to_episodes(key, value, env_ids)

    # Set task success values for the relevant episodes
    success_results = torch.zeros(len(env_ids), dtype=bool, device=self._env.device)
    # Check success indicator from termination terms
    if hasattr(self._env, "termination_manager"):
        if "success" in self._env.termination_manager.active_terms:
            success_results |= self._env.termination_manager.get_term("success")[env_ids]

    # Check episode length condition for success
    episode_lengths = self._env.episode_length_buf[env_ids]
    exploration_lengths = self.exploration_lengths[env_ids]
    exploration_ratios = exploration_lengths / episode_lengths
    success_results = success_results & (exploration_ratios < self.max_exploration_horizon_for_save)
    self.set_success_to_episodes(env_ids, success_results)

    if force_export_or_skip or (force_export_or_skip is None and self.cfg.export_in_record_pre_reset):
        self.export_episodes(env_ids)


def process_agent_cfg(env_cfg, agent_cfg):
    if hasattr(agent_cfg.algorithm, "behavior_cloning_cfg"):
        if agent_cfg.algorithm.behavior_cloning_cfg is None:
            del agent_cfg.algorithm.behavior_cloning_cfg
        else:
            bc_cfg = agent_cfg.algorithm.behavior_cloning_cfg
            if bc_cfg.experts_observation_group_cfg is not None:
                import importlib

                # resolve path to the module location
                mod_name, attr_name = bc_cfg.experts_observation_group_cfg.split(":")
                mod = importlib.import_module(mod_name)
                cfg_cls = mod
                for attr in attr_name.split("."):
                    cfg_cls = getattr(cfg_cls, attr)
                cfg = cfg_cls()
                setattr(env_cfg.observations, "expert_obs", cfg)

    if hasattr(agent_cfg.algorithm, "offline_algorithm_cfg"):
        if agent_cfg.algorithm.offline_algorithm_cfg is None:
            del agent_cfg.algorithm.offline_algorithm_cfg
        else:
            if agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg is None:
                del agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
            else:
                bc_cfg = agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
                if bc_cfg.experts_observation_group_cfg is not None:
                    import importlib

                    # resolve path to the module location
                    mod_name, attr_name = bc_cfg.experts_observation_group_cfg.split(":")
                    mod = importlib.import_module(mod_name)
                    cfg_cls = mod
                    for attr in attr_name.split("."):
                        cfg_cls = getattr(cfg_cls, attr)
                    cfg = cfg_cls()
                    setattr(env_cfg.observations, "expert_obs", cfg)
    return agent_cfg


def sample_exploration_horizons(
    num_envs: int, min_horizon: int, max_horizon: int, device: torch.device
) -> torch.Tensor:
    """Sample exploration horizons for each environment."""
    return torch.randint(min_horizon, max_horizon + 1, (num_envs,), device=device)


class CollectionMonitor:
    """Watches demo-collection progress; optionally prints telemetry and recovers from stalls.

    Every ``check_every`` env steps it (a) with ``debug``, prints a progress line, appends
    per-dimension action statistics to ``csv_path`` and diagnoses stalls, and (b) when no new demo
    has been recorded for ``stall_reset_after_s`` seconds, asks the caller to force an ``env.reset()``.
    It gives up once ``stall_giveup_after`` consecutive forced resets bring no new demos.
    """

    def __init__(
        self,
        debug: bool,
        csv_path: str,
        stall_reset_after_s: float,
        check_every: int = 50,
        no_progress_warn_s: float = 30.0,
        stall_giveup_after: int = 3,
    ):
        self.debug = debug
        self.csv_path = csv_path
        self.stall_reset_after_s = stall_reset_after_s
        self.check_every = check_every
        self.no_progress_warn_s = no_progress_warn_s
        self.stall_giveup_after = stall_giveup_after
        self.step = 0
        self.demo_count = 0
        self.last_demo_wall = time.time()
        self.step_times: list[float] = []
        self.window_dones = 0
        self.window_successes = 0
        self.window_wall = time.time()
        self.csv_header_written = False
        self.stall_reset_count = 0
        self.last_stall_reset_demo_count = 0

    def record_step(self, step_dt: float, dones: torch.Tensor, demo_count: int) -> None:
        self.step += 1
        self.step_times = (self.step_times + [step_dt])[-100:]
        self.window_dones += int(dones.sum().item())
        if demo_count > self.demo_count:
            self.window_successes += demo_count - self.demo_count
            self.demo_count = demo_count
            self.last_demo_wall = time.time()

    def check(self, env, actions: torch.Tensor) -> str | None:
        """Returns ``"reset"`` to force an env reset, ``"giveup"`` to stop collecting, else ``None``."""
        if self.step % self.check_every != 0:
            return None
        now = time.time()
        no_progress_s = now - self.last_demo_wall
        if self.debug:
            self._report(env, actions, now, no_progress_s)
        self.window_dones = 0
        self.window_successes = 0
        self.window_wall = now

        if self.stall_reset_after_s <= 0 or no_progress_s <= self.stall_reset_after_s:
            return None
        if self.demo_count > self.last_stall_reset_demo_count:
            # the previous forced reset helped
            self.stall_reset_count = 0
        self.stall_reset_count += 1
        if self.stall_reset_count > self.stall_giveup_after:
            print(
                f"[collect] {self.stall_reset_count - 1} consecutive forced resets did not unstick collection."
                f" Saving {self.demo_count} demos and exiting.",
                flush=True,
            )
            return "giveup"
        print(
            f"[collect] No new demo for {no_progress_s:.0f}s: forcing env.reset() (attempt"
            f" {self.stall_reset_count}/{self.stall_giveup_after}) at {self.demo_count} demos.",
            flush=True,
        )
        self.last_stall_reset_demo_count = self.demo_count
        self.last_demo_wall = time.time()
        return "reset"

    def _report(self, env, actions: torch.Tensor, now: float, no_progress_s: float) -> None:
        wall_dt = max(now - self.window_wall, 1e-9)
        ep_lens = env.unwrapped.episode_length_buf
        tm = env.unwrapped.termination_manager
        time_outs = int(tm.time_outs.sum().item()) if hasattr(tm, "time_outs") else -1
        term_dones = int(tm.dones.sum().item()) if hasattr(tm, "dones") else -1
        step_med = sorted(self.step_times)[len(self.step_times) // 2] if self.step_times else 0.0
        step_max = max(self.step_times) if self.step_times else 0.0
        act_mean = actions.mean(dim=0).detach().cpu().tolist()
        act_std = actions.std(dim=0).detach().cpu().tolist()
        act_norm = actions.norm(dim=-1).detach().cpu()
        act_norm_med = float(act_norm.median().item())
        act_norm_max = float(act_norm.max().item())
        # fraction of envs with any action dim at the +-1 clip bound
        act_sat = float(((actions.abs() >= 0.999).any(dim=-1)).float().mean().item())
        print(
            f"[debug] step={self.step} demos={self.demo_count} "
            f"window: dones={self.window_dones} successes={self.window_successes} "
            f"({self.window_dones / wall_dt:.1f} dones/s, {self.window_successes / wall_dt:.2f} success/s) | "
            f"ep_len min/mean/max={int(ep_lens.min())}/{float(ep_lens.float().mean()):.1f}/{int(ep_lens.max())} | "
            f"step_dt med/max={step_med * 1000:.0f}/{step_max * 1000:.0f}ms | "
            f"this-step time_outs={time_outs} dones_in_buf={term_dones} | "
            f"act |a| med/max={act_norm_med:.3f}/{act_norm_max:.3f} sat={act_sat:.2f} | "
            f"no_progress={no_progress_s:.0f}s",
            flush=True,
        )
        try:
            if not self.csv_header_written:
                os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
                cols = (
                    ["step", "demos", "no_progress_s"]
                    + [f"act_mean_{i}" for i in range(len(act_mean))]
                    + [f"act_std_{i}" for i in range(len(act_std))]
                    + ["act_norm_med", "act_norm_max", "act_sat_frac"]
                )
                with open(self.csv_path, "w") as f:
                    f.write(",".join(cols) + "\n")
                self.csv_header_written = True
            row = (
                [str(self.step), str(self.demo_count), f"{no_progress_s:.1f}"]
                + [f"{x:.5f}" for x in act_mean]
                + [f"{x:.5f}" for x in act_std]
                + [f"{act_norm_med:.5f}", f"{act_norm_max:.5f}", f"{act_sat:.4f}"]
            )
            with open(self.csv_path, "a") as f:
                f.write(",".join(row) + "\n")
        except OSError as e:
            print(f"[debug] CSV write failed: {e}", flush=True)
        if no_progress_s > self.no_progress_warn_s:
            if step_max > 5.0:
                cause = f"step latency exploded ({step_max * 1000:.0f} ms); the PhysX solver likely hung on an env."
            elif self.window_dones == 0:
                cause = (
                    f"env steps are fast ({step_med * 1000:.0f} ms median) but nothing terminated in this window;"
                    " termination buffers may be stuck."
                )
            elif self.window_successes == 0:
                cause = (
                    f"{self.window_dones} episodes ended but none succeeded; EXPORT_SUCCEEDED_ONLY drops failures,"
                    " so the demo count cannot advance."
                )
            else:
                cause = "indeterminate."
            print(f"[debug] STALL CAUSE: {cause}", flush=True)


@hydra_task_compose(args_cli.task, "rsl_rl_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Collect demonstrations from the environment using exploration + expert policy."""
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    # device = torch.device(args_cli.device if args_cli.device else 'cuda' if torch.cuda.is_available() else 'cpu')

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.basename(args_cli.dataset_file)

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # add recordermanager to save data
    use_zarr_format = args_cli.dataset_file.endswith(".zarr")
    if use_zarr_format:
        dataset_handler = ZarrDatasetFileHandler
    else:
        dataset_handler = HDF5DatasetFileHandler

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = dataset_handler

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # simulation, expert and action buffers share the sim device; the student may use other GPUs
    device = torch.device(env_cfg.sim.device)
    env_cfg.seed = args_cli.seed
    # we want to have the terms in the observations returned as a dictionary
    env_cfg.observations.policy.concatenate_terms = False

    # set episode length based on cli argument
    env_cfg.episode_length_s = args_cli.episode_length_s
    episode_length = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.sim.render_interval))
    max_exploration_horizon = int(args_cli.max_exploration_horizon * episode_length)
    min_exploration_horizon = int(args_cli.min_exploration_horizon * episode_length)

    # add expert obs into env_cfg
    agent_cfg = process_agent_cfg(env_cfg, agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # Optional video recording (mirrors scripts/reinforcement_learning/rsl_rl/play.py).
    # Wrapping must happen *before* RslRlVecEnvWrapper so RecordVideo sees env.step()
    # at the right level. RecordVideo only renders env 0 of the vec env.
    if args_cli.video:
        if args_cli.video_dir is None:
            args_cli.video_dir = os.path.join(os.path.dirname(args_cli.dataset_file), "videos")
        os.makedirs(args_cli.video_dir, exist_ok=True)
        if args_cli.video_interval and args_cli.video_interval > 0:
            video_kwargs = {
                "video_folder": args_cli.video_dir,
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
        else:
            video_kwargs = {
                "video_folder": args_cli.video_dir,
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
        print(f"[collect_demos] Recording video to {args_cli.video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    num_envs = env.num_envs

    # load exploration policy
    exploration_policy = None
    if args_cli.exploration_checkpoint:
        student_devices = resolve_student_devices(device, args_cli.student_devices)
        print(f"[collect] Student inference devices: {[str(d) for d in student_devices]}")
        exploration_policy = MultiDeviceStudent.from_checkpoint(
            args_cli.exploration_checkpoint, student_devices, num_envs, output_device=device
        )

    # load expert policy
    bc = agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
    assert len(bc.experts_path) == 1, "Only one expert is supported for now."
    expert_obs_fn = bc.experts_observation_func
    loader = bc.experts_loader
    if not callable(loader):
        loader = eval(loader)
    expert_policy = loader(bc.experts_path[0]).to(device)
    expert_policy.eval()

    # get expert mask recorder
    recorder_manager = env.unwrapped.recorder_manager
    expert_mask_recorder = recorder_manager._terms.get("record_pre_step_expert_mask")

    # initialize exploration horizons for each env
    exploration_horizons = sample_exploration_horizons(
        num_envs, min_exploration_horizon, max_exploration_horizon, device
    )
    expert_mask_recorder.set_exploration_horizon(exploration_horizons)

    if exploration_policy is not None:
        from types import MethodType

        recorder_manager.record_pre_reset = MethodType(record_pre_reset, recorder_manager)

    exploration_lengths = torch.zeros((num_envs,), device=device, dtype=torch.int32)
    recorder_manager.exploration_lengths = exploration_lengths
    recorder_manager.max_exploration_horizon_for_save = args_cli.max_exploration_horizon_for_save

    # reset environment and get initial observations
    obs_dict, _ = env.reset()

    # reset exploration policy
    if exploration_policy is not None:
        reset_ids = torch.arange(num_envs, device=device)
        exploration_policy.reset(reset_ids)

    rendered_frames = []
    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    monitor = CollectionMonitor(
        debug=args_cli.debug,
        csv_path=os.path.join(output_dir, "action_stats.csv"),
        stall_reset_after_s=args_cli.stall_reset_after_s,
    )
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        pbar = tqdm(total=args_cli.num_demos, desc="Recording Demonstrations", unit="demo")

        while True:
            # get current episode step for each env
            episode_steps = env.unwrapped.episode_length_buf

            # determine which policy to use for each env
            use_exploration = (episode_steps < exploration_horizons) & (exploration_policy is not None)
            exploration_lengths += use_exploration.int()
            recorder_manager.exploration_lengths = exploration_lengths
            use_expert = ~use_exploration

            # compute actions
            actions = torch.zeros((num_envs, env.action_space.shape[-1]), device=device)
            expert_mask = torch.zeros((num_envs, 1), dtype=torch.bool, device=device)
            # exploration policy actions
            if use_exploration.any() and exploration_policy is not None:
                exploration_env_ids = use_exploration.nonzero(as_tuple=False).reshape(-1)
                exploration_obs = {k: v[use_exploration] for k, v in obs_dict["policy"].items()}
                exploration_actions = exploration_policy.predict_action(exploration_obs, exploration_env_ids)
                exploration_actions = exploration_actions.to(device)
                actions[use_exploration] = exploration_actions

            # expert policy actions
            if use_expert.any():
                expert_policy_obs = expert_obs_fn(env)
                if isinstance(expert_policy_obs, dict):
                    expert_policy_obs = torch.cat([v for v in expert_policy_obs.values()], dim=-1)
                mean, std = expert_policy.compute_distribution(expert_policy_obs)
                expert_actions = torch.normal(mean, std)
                if args_cli.expert_noise > 0.0:
                    noise = torch.randn_like(expert_actions) * args_cli.expert_noise
                    expert_actions += noise
                actions[use_expert] = expert_actions[use_expert]
                expert_mask[use_expert] = True

            # mask actions for first step after reset (first image not valid)
            first_step_mask = episode_steps == 0
            if torch.any(first_step_mask):
                actions[first_step_mask, :-1] = 0.0
                actions[first_step_mask, -1] = -1.0  # close gripper

            # set expert mask for recorder
            expert_mask_recorder.set_mask(expert_mask)

            # env stepping
            step_start = time.time()
            obs_dict, _, dones, _ = env.step(actions)
            step_dt = time.time() - step_start

            # handle resets
            if dones.any():
                reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
                # resample exploration horizons for reset envs
                exploration_horizons[reset_ids] = sample_exploration_horizons(
                    len(reset_ids), min_exploration_horizon, max_exploration_horizon, device
                )
                if exploration_policy is not None:
                    exploration_policy.reset(reset_ids)

                exploration_lengths[reset_ids] = 0

            # update demo count
            new_count = env.unwrapped.recorder_manager.exported_successful_episode_count
            if new_count > current_recorded_demo_count:
                pbar.update(new_count - current_recorded_demo_count)
                current_recorded_demo_count = new_count

            monitor.record_step(step_dt, dones, current_recorded_demo_count)
            stall_action = monitor.check(env, actions)
            if stall_action == "giveup":
                break
            if stall_action == "reset":
                # clear PhysX state and re-sample reset states
                obs_dict, _ = env.reset()
                if exploration_policy is not None:
                    exploration_policy.reset(torch.arange(num_envs, device=device))
                exploration_lengths.zero_()
                exploration_horizons[:] = sample_exploration_horizons(
                    num_envs, min_exploration_horizon, max_exploration_horizon, device
                )

            if args_cli.num_demos > 0 and new_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break

            if args_cli.render:
                rendered_frames.append(env.render())

        pbar.close()

    if args_cli.render:
        # save rendered frames as a video
        import cv2

        video_path = os.path.join(output_dir, "recorded_demos_video.mp4")
        height, width, _ = rendered_frames[0].shape
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
        for frame in rendered_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        video_writer.release()
        print(f"Saved recorded demonstrations video at: {video_path}")
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function - the decorator handles parameter passing
    main()  # type: ignore
    # close sim app
    simulation_app.close()
