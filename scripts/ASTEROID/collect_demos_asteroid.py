# Copyright (c) 2024-2025, The Octi Lab Project Developers.
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

"""Script to collect demonstrations using exploration policy + expert policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import os
import gymnasium as gym
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
parser.add_argument(
    "--num_demos", type=int, default=10, help="Number of demonstrations to record."
)
parser.add_argument(
    "--exploration_checkpoint", type=str, default=None, help="Path to exploration policy checkpoint."
)
parser.add_argument(
    "--min_exploration_horizon", type=float, default=0.02, help="Minimum exploration horizon for exploration policy."
)
parser.add_argument(
    "--max_exploration_horizon", type=float, default=0.3, help="Maximum exploration horizon for exploration policy."
)
parser.add_argument(
    "--episode_length_s", type=float, default=5.0, help="Episode length in seconds."
)
parser.add_argument("--expert_noise", type=float, default=0.0, help="Noise level for the expert policy.")

parser.add_argument("--render", action="store_true", help="Whether to render the environment.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for environment.")
parser.add_argument("--video", action="store_true", default=False, help="Record video of env 0 during collection.")
parser.add_argument("--video_length", type=int, default=400, help="Video length in env steps (only used when --video).")
parser.add_argument("--video_interval", type=int, default=0,
                    help="Restart video recording every N env steps. 0 (default) = single video at step 0.")
parser.add_argument("--video_dir", type=str, default=None,
                    help="Directory to write videos. Defaults to <dataset_file>_dir/videos.")

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

import dill
import hydra

from isaaclab.envs import (
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg
)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401

# Import dataset handlers
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab.managers.recorder_manager import DatasetExportMode

from uwlab.utils.datasets import ZarrDatasetFileHandler
from uwlab_tasks.utils.hydra import hydra_task_compose
from uwlab_tasks.manager_based.manipulation.asteroid.mdp.recorders.recorders_cfg import (
    AsteroidActionStateRecorderManagerCfg as ActionStateRecorderManagerCfg,
)

# Diffusion policy imports
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from uwlab_rl.wrappers.diffusion import DiffusionPolicyWrapper

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
    max_exploration_horizon_for_save = 0.9
    success_results = success_results & (exploration_ratios < max_exploration_horizon_for_save)
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


def load_exploration_policy(checkpoint_path: str, device: torch.device, num_envs: int):
    """Load exploration diffusion policy from checkpoint."""
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.eval().to(device)

    wrapped_policy = DiffusionPolicyWrapper(policy, device, n_obs_steps=policy.n_obs_steps, num_envs=num_envs)
    return wrapped_policy


def sample_exploration_horizons(num_envs: int, min_horizon: int, max_horizon: int, device: torch.device) -> torch.Tensor:
    """Sample exploration horizons for each environment."""
    return torch.randint(min_horizon, max_horizon + 1, (num_envs,), device=device)


@hydra_task_compose(args_cli.task, "rsl_rl_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Collect demonstrations from the environment using exploration + expert policy."""
    # device = torch.device(args_cli.device if args_cli.device else 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')
    policy_device = torch.device('cuda:1') if torch.cuda.device_count() > 1 else device

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.basename(args_cli.dataset_file)

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # add recordermanager to save data
    use_zarr_format = args_cli.dataset_file.endswith('.zarr')
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
    env_cfg.seed = None
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
            args_cli.video_dir = os.path.join(
                os.path.dirname(args_cli.dataset_file), "videos"
            )
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
        exploration_policy = load_exploration_policy(args_cli.exploration_checkpoint, policy_device, num_envs)

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
    
    # reset environment and get initial observations
    obs_dict, _ = env.reset()

    # reset exploration policy
    if exploration_policy is not None:
        reset_ids = torch.arange(num_envs, device=device)
        exploration_policy.reset(reset_ids)

    rendered_frames = []
    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    # ---- debug instrumentation ----
    import time as _time_dbg
    _dbg_step = 0
    _dbg_last_demo_count = 0
    _dbg_last_demo_step = 0
    _dbg_last_demo_wall = _time_dbg.time()
    _dbg_step_times: list[float] = []  # rolling step latencies
    _dbg_total_dones = 0
    _dbg_total_successes = 0
    _dbg_window_steps = 0
    _dbg_window_dones = 0
    _dbg_window_successes = 0
    _dbg_window_wall = _time_dbg.time()
    DBG_PRINT_EVERY = 50  # env steps
    DBG_NO_PROGRESS_S = 30.0  # alert when no new demo for this many seconds
    # ---- stall recovery (auto-retry inside this process) ----
    STALL_RESET_AFTER_S = 90.0   # full env.reset() if no demo progress for this long
    STALL_GIVEUP_AFTER = 3       # break after this many resets in a row that don't help
    _stall_reset_count = 0
    _last_stall_reset_demo_count = 0
    _last_stall_reset_step = 0
    # CSV trace of action stats per dbg interval — offline-plottable.
    _dbg_action_csv = os.path.join(
        os.path.dirname(args_cli.dataset_file), "action_stats.csv"
    )
    _dbg_csv_header_written = False
    # --------------------------------
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
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
                exploration_obs = {k: v[use_exploration] for k, v in obs_dict['policy'].items()}
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
            first_step_mask = (episode_steps == 0)
            if torch.any(first_step_mask):
                actions[first_step_mask, :-1] = 0.0
                actions[first_step_mask, -1] = -1.0  # close gripper

            # set expert mask for recorder
            expert_mask_recorder.set_mask(expert_mask)

            # env stepping
            _dbg_t0 = _time_dbg.time()
            obs_dict, _, dones, _ = env.step(actions)
            _dbg_step_dt = _time_dbg.time() - _dbg_t0
            _dbg_step_times.append(_dbg_step_dt)
            if len(_dbg_step_times) > 100:
                _dbg_step_times = _dbg_step_times[-100:]

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
                # expert_mask_recorder.set_exploration_horizon(exploration_horizons)

            # update demo count
            new_count = env.unwrapped.recorder_manager.exported_successful_episode_count
            if new_count > current_recorded_demo_count:
                increment = new_count - current_recorded_demo_count
                current_recorded_demo_count = new_count
                _dbg_window_successes += increment
                _dbg_last_demo_step = _dbg_step
                _dbg_last_demo_wall = _time_dbg.time()
                pbar.update(increment)

            # ---- debug telemetry ----
            _dbg_step += 1
            _dbg_window_steps += 1
            _dbg_n_dones = int(dones.sum().item())
            _dbg_window_dones += _dbg_n_dones
            _dbg_total_dones += _dbg_n_dones

            if _dbg_step % DBG_PRINT_EVERY == 0:
                _now = _time_dbg.time()
                wall_dt = max(_now - _dbg_window_wall, 1e-9)
                ep_lens = env.unwrapped.episode_length_buf
                ep_min, ep_max, ep_mean = (
                    int(ep_lens.min().item()),
                    int(ep_lens.max().item()),
                    float(ep_lens.float().mean().item()),
                )
                # time_out_buf / terminated_buf if available
                tm = env.unwrapped.termination_manager
                _to = int(tm.time_outs.sum().item()) if hasattr(tm, "time_outs") else -1
                _term = int(tm.dones.sum().item()) if hasattr(tm, "dones") else -1
                step_med = sorted(_dbg_step_times)[len(_dbg_step_times) // 2] if _dbg_step_times else 0.0
                step_max = max(_dbg_step_times) if _dbg_step_times else 0.0
                no_progress_s = _now - _dbg_last_demo_wall
                rate = _dbg_window_successes / wall_dt
                done_rate = _dbg_window_dones / wall_dt
                # action stats — track per-dim mean / std / saturation.
                _act_dim = actions.shape[-1]
                _act_mean = actions.mean(dim=0).detach().cpu().tolist()
                _act_std = actions.std(dim=0).detach().cpu().tolist()
                _act_norm = actions.norm(dim=-1).detach().cpu()
                _act_norm_med = float(_act_norm.median().item())
                _act_norm_max = float(_act_norm.max().item())
                # how often does any action dim hit ±1 (the typical clip bound)?
                _act_saturated_frac = float(((actions.abs() >= 0.999).any(dim=-1)).float().mean().item())
                print(
                    f"[dbg] step={_dbg_step} demos={current_recorded_demo_count} "
                    f"window: dones={_dbg_window_dones} successes={_dbg_window_successes} "
                    f"({done_rate:.1f} dones/s, {rate:.2f} success/s) | "
                    f"ep_len min/mean/max={ep_min}/{ep_mean:.1f}/{ep_max} | "
                    f"step_dt med/max={step_med*1000:.0f}/{step_max*1000:.0f}ms | "
                    f"this-step time_outs={_to} dones_in_buf={_term} | "
                    f"act |a| med/max={_act_norm_med:.3f}/{_act_norm_max:.3f} sat={_act_saturated_frac:.2f} | "
                    f"no_progress={no_progress_s:.0f}s",
                    flush=True,
                )
                # CSV row: step,demos,no_progress_s,act_mean[0..D],act_std[0..D],act_norm_median,act_saturated_frac
                try:
                    if not _dbg_csv_header_written:
                        os.makedirs(os.path.dirname(_dbg_action_csv), exist_ok=True)
                        with open(_dbg_action_csv, "w") as _f:
                            cols = (
                                ["step", "demos", "no_progress_s"]
                                + [f"act_mean_{i}" for i in range(_act_dim)]
                                + [f"act_std_{i}" for i in range(_act_dim)]
                                + ["act_norm_med", "act_norm_max", "act_sat_frac"]
                            )
                            _f.write(",".join(cols) + "\n")
                        _dbg_csv_header_written = True
                    with open(_dbg_action_csv, "a") as _f:
                        row = (
                            [str(_dbg_step), str(current_recorded_demo_count), f"{no_progress_s:.1f}"]
                            + [f"{x:.5f}" for x in _act_mean]
                            + [f"{x:.5f}" for x in _act_std]
                            + [f"{_act_norm_med:.5f}", f"{_act_norm_max:.5f}", f"{_act_saturated_frac:.4f}"]
                        )
                        _f.write(",".join(row) + "\n")
                except Exception as _e:
                    print(f"[dbg] CSV write failed: {_e}", flush=True)
                if no_progress_s > DBG_NO_PROGRESS_S:
                    # diagnose stall: split between physics-step-stuck vs success-rate=0
                    if step_max > 5.0:
                        print(f"[dbg] STALL CAUSE: step latency exploded ({step_max*1000:.0f} ms). "
                              "physx solver likely hung on at least one env.", flush=True)
                    elif _dbg_window_dones == 0 and ep_max > 0:
                        print(f"[dbg] STALL CAUSE: env steps fast ({step_med*1000:.0f} ms median) "
                              "but ZERO terminations in window. time_out / abnormal / success "
                              "all stopped firing — termination buffers may be stuck.", flush=True)
                    elif _dbg_window_dones > 0 and _dbg_window_successes == 0:
                        print(f"[dbg] STALL CAUSE: episodes terminating ({_dbg_window_dones} in window) "
                              "but zero successes — all attempts failing. EXPORT_SUCCEEDED_ONLY "
                              "drops failures so demo count can't advance.", flush=True)
                    else:
                        print(f"[dbg] STALL CAUSE: indeterminate. step_med={step_med*1000:.0f}ms "
                              f"dones_window={_dbg_window_dones} successes_window={_dbg_window_successes}",
                              flush=True)
                _dbg_window_steps = 0
                _dbg_window_dones = 0
                _dbg_window_successes = 0
                _dbg_window_wall = _now

                # ---- stall recovery: full env.reset() to clear physx state and
                #      re-sample reset states; break only if multiple consecutive
                #      resets fail to unstick collection.
                if no_progress_s > STALL_RESET_AFTER_S:
                    if current_recorded_demo_count > _last_stall_reset_demo_count:
                        # last reset DID help (we got new demos since); reset counter
                        _stall_reset_count = 0
                    _stall_reset_count += 1
                    if _stall_reset_count > STALL_GIVEUP_AFTER:
                        print(
                            f"[dbg] STALL GIVEUP: {_stall_reset_count} consecutive "
                            f"force-resets did not unstick collection. Saving "
                            f"{current_recorded_demo_count} demos and exiting cleanly.",
                            flush=True,
                        )
                        break
                    print(
                        f"[dbg] STALL RECOVERY: forcing env.reset() (attempt "
                        f"{_stall_reset_count}/{STALL_GIVEUP_AFTER}) at "
                        f"{current_recorded_demo_count} demos to clear physx state.",
                        flush=True,
                    )
                    obs_dict, _ = env.reset()
                    if exploration_policy is not None:
                        exploration_policy.reset(torch.arange(num_envs, device=device))
                    exploration_lengths.zero_()
                    exploration_horizons[:] = sample_exploration_horizons(
                        num_envs, min_exploration_horizon, max_exploration_horizon, device
                    )
                    _last_stall_reset_demo_count = current_recorded_demo_count
                    _last_stall_reset_step = _dbg_step
                    _dbg_last_demo_wall = _time_dbg.time()  # don't immediately re-trigger
            # -------------------------

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
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
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
