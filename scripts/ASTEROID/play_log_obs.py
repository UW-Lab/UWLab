"""Play a trained RSL-RL state policy in the tactile data-collection env, log a
single observation term from the `data_collection` group every step, and save a
matplotlib plot.

Mirrors play.py's preamble (AppLauncher boot, ckpt
load via OnPolicyRunner) but reads obs via the env's ObservationManager so we
can pull from any group, not just the wrapped policy obs.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play policy and log a data_collection obs term.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID (tactile data-collection env).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_*.pt to load.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=200, help="Env steps to roll out and log.")
parser.add_argument("--obs_group", type=str, default="data_collection")
parser.add_argument("--obs_term", type=str, nargs="+", default=["left_knuckle_pos"],
                    help="One or more obs term names to log; plotted on the same figure.")
parser.add_argument("--out_plot", type=str, default="./gripper_pos.png")
parser.add_argument("--video", action="store_true", default=False, help="Record video of the env.")
parser.add_argument("--video_length", type=int, default=None,
                    help="Number of env steps to record. Defaults to --num_steps.")
parser.add_argument("--video_dir", type=str, default="./play_videos",
                    help="Where gym.RecordVideo will write the rl-video-step-0.mp4.")
parser.add_argument("--out_synced_video", type=str, default="./synced_play.mp4",
                    help="Output path for the merged camera+plot video.")
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Cameras must be enabled for video capture; mirror play.py.
if args_cli.video:
    args_cli.enable_cameras = True
    if args_cli.video_length is None:
        args_cli.video_length = args_cli.num_steps

# Hydra reads from sys.argv; isolate its overrides.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import os  # noqa: E402
import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import inspect  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import uwlab_tasks  # noqa: E402,F401
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def _drop_unknown_algorithm_keys(agent_cfg) -> None:
    """Mirror cli_args.sanitize_rsl_rl_cfg: strip alg keys the installed PPO class can't accept."""
    alg_cfg = agent_cfg.algorithm
    class_name = getattr(alg_cfg, "class_name", None)
    if class_name is None:
        return
    from rsl_rl import algorithms
    alg_class = getattr(algorithms, class_name, None)
    if alg_class is None:
        return
    accepted = set(inspect.signature(alg_class.__init__).parameters.keys())
    for key in list(vars(alg_cfg)):
        if key != "class_name" and key not in accepted:
            delattr(alg_cfg, key)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed

    _drop_unknown_algorithm_keys(agent_cfg)

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        os.makedirs(args_cli.video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[INFO] Recording video to {args_cli.video_dir}")

    base_env = env.unwrapped
    wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading actor-only weights from: {args_cli.checkpoint}")
    # Tactile env's critic obs (188 dims at training) differs from current critic dim
    # (155). strict=False does NOT tolerate size mismatches — it only ignores missing
    # / unexpected keys. So drop everything not on the actor side before loading.
    ckpt = torch.load(args_cli.checkpoint, weights_only=False, map_location=agent_cfg.device)
    full_sd = ckpt["model_state_dict"]
    actor_sd = {k: v for k, v in full_sd.items() if k.startswith(("actor.", "actor_obs_normalizer.", "std", "log_std"))}
    # rsl_rl's ActorCritic.load_state_dict wraps nn.Module's and returns a bool, not
    # _IncompatibleKeys, so call super() (nn.Module) directly to get key reports.
    import torch.nn as _nn
    incompat = _nn.Module.load_state_dict(runner.alg.policy, actor_sd, strict=False)
    if any(k.startswith(("actor.", "actor_obs_normalizer.")) for k in incompat.missing_keys):
        raise RuntimeError(f"Actor weights missing after partial load: {incompat.missing_keys}")
    print(f"[INFO] Loaded {len(actor_sd)} actor tensors; "
          f"skipped {len(full_sd) - len(actor_sd)} critic / RND / etc. tensors. "
          f"({len(incompat.missing_keys)} missing, {len(incompat.unexpected_keys)} unexpected.)")
    policy = runner.get_inference_policy(device=base_env.device)

    obs_buf: dict[str, list[np.ndarray]] = {term: [] for term in args_cli.obs_term}

    obs = wrapped.get_observations()
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = wrapped.step(actions)
        # Pull the dict-form group directly from the obs manager
        # (update_history=False so we don't double-tick history buffers).
        group_obs = base_env.observation_manager.compute_group(args_cli.obs_group, update_history=False)
        if not isinstance(group_obs, dict):
            raise RuntimeError(
                f"Group '{args_cli.obs_group}' is not dict-form (concatenate_terms=True?); "
                f"set concatenate_terms=False to capture individual terms."
            )
        for term in args_cli.obs_term:
            if term not in group_obs:
                raise KeyError(
                    f"Term '{term}' not in group '{args_cli.obs_group}'. "
                    f"Available: {list(group_obs.keys())}"
                )
            obs_buf[term].append(group_obs[term].detach().cpu().numpy().copy())

    dt = base_env.step_dt
    fig, ax = plt.subplots(figsize=(10, 4))
    for term in args_cli.obs_term:
        series = np.stack(obs_buf[term], axis=0).squeeze()
        t = np.arange(series.shape[0]) * dt
        if series.ndim == 1:
            ax.plot(t, series, label=term)
        else:
            for i in range(series.shape[-1]):
                ax.plot(t, series[..., i], label=f"{term}[{i}]")
        print(f"[INFO] {term}: shape={series.shape}, min={series.min():.4f} max={series.max():.4f}")
    ax.set_xlabel(f"time (s, env step_dt={dt:.3f})")
    ax.set_ylabel("joint angle (rad)")
    ax.set_title(f"{args_cli.obs_group} terms during play (ckpt={os.path.basename(args_cli.checkpoint)})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.abspath(args_cli.out_plot)
    fig.savefig(out_path, dpi=120)
    print(f"[INFO] Saved plot to: {out_path}")

    env.close()

    if args_cli.video:
        # Locate the file gym.RecordVideo just wrote.
        vid_files = sorted(f for f in os.listdir(args_cli.video_dir) if f.endswith(".mp4"))
        if not vid_files:
            print(f"[WARN] No mp4 found in {args_cli.video_dir}; skipping synced video.")
            return
        cam_path = os.path.join(args_cli.video_dir, vid_files[-1])
        out_synced = os.path.abspath(args_cli.out_synced_video)
        _make_synced_video(
            cam_path=cam_path,
            obs_buf=obs_buf,
            obs_terms=args_cli.obs_term,
            step_dt=dt,
            out_path=out_synced,
            ckpt_name=os.path.basename(args_cli.checkpoint),
        )


def _make_synced_video(cam_path, obs_buf, obs_terms, step_dt, out_path, ckpt_name):
    """Stack the recorded env video on top of a matplotlib plot of `obs_terms`,
    drawing a moving vertical cursor on the plot in lock-step with the video.
    Mirrors diffusion_policy/visualize_episode.py."""
    cap = cv2.VideoCapture(cam_path)
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or (1.0 / step_dt)
    n_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame0 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ret:
        print(f"[WARN] Could not read first frame from {cam_path}; skipping synced video.")
        cap.release()
        return
    cam_h, cam_w = frame0.shape[:2]

    series = {term: np.stack(obs_buf[term], axis=0).squeeze() for term in obs_terms}
    N = len(next(iter(series.values())))
    ratio = vid_fps / (1.0 / step_dt)  # video frames per env step

    print(f"[INFO] Synced video: {n_vid_frames} video frames @ {vid_fps:.1f} fps, "
          f"{N} env steps @ {1/step_dt:.1f} Hz (ratio {ratio:.2f}x)")

    plot_h = 280
    plot_dpi = 100
    fig, ax = plt.subplots(figsize=(cam_w / plot_dpi, plot_h / plot_dpi), dpi=plot_dpi)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#2a2a2a")
    colors = ["#e67e22", "#3498db", "#2ecc71", "#9b59b6", "#e74c3c"]
    t = np.arange(N)
    for i, term in enumerate(obs_terms):
        s = series[term]
        if s.ndim == 1:
            ax.plot(t, s, color=colors[i % len(colors)], lw=1.0, label=term)
        else:
            for j in range(s.shape[-1]):
                ax.plot(t, s[:, j], color=colors[(i + j) % len(colors)], lw=1.0, label=f"{term}[{j}]")
    ax.set_xlim(0, max(N - 1, 1))
    ax.set_ylabel("obs", color="#cccccc", fontsize=8)
    ax.set_xlabel(f"env step ({step_dt*1000:.0f} ms)", color="#cccccc", fontsize=8)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    ax.grid(True, alpha=0.2, color="#888888")
    ax.legend(loc="upper right", fontsize=7, facecolor="#333333", edgecolor="#555555", labelcolor="white")
    ax.set_title(f"{ckpt_name}", color="#cccccc", fontsize=8)
    fig.tight_layout(pad=0.6)

    fig.canvas.draw()
    # matplotlib >=3.10 dropped tostring_rgb; buffer_rgba works on all recent versions.
    plot_bg = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()  # H x W x 3 (RGB)
    plot_bg_bgr = cv2.cvtColor(plot_bg, cv2.COLOR_RGB2BGR)

    bbox = ax.get_window_extent()
    img_h = plot_bg.shape[0]
    ax_x0 = int(bbox.x0)
    ax_x1 = int(bbox.x1)
    ax_y0_img = img_h - int(bbox.y1)
    ax_y1_img = img_h - int(bbox.y0)
    plt.close(fig)

    out_h = cam_h + plot_h
    out_w = cam_w
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, vid_fps, (out_w, out_h))
    print(f"[INFO] Writing {out_path}  ({out_w}x{out_h} @ {vid_fps:.1f} fps)")

    for frame_idx in range(n_vid_frames):
        env_idx = min(int(round(frame_idx / max(ratio, 1e-6))), N - 1)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        plot_frame = plot_bg_bgr.copy()
        x_norm = env_idx / max(N - 1, 1)
        x_px = ax_x0 + int(x_norm * (ax_x1 - ax_x0))
        cv2.line(plot_frame, (x_px, ax_y0_img), (x_px, ax_y1_img), (255, 255, 255), 1)
        if plot_frame.shape[1] != out_w:
            plot_frame = cv2.resize(plot_frame, (out_w, plot_h))
        writer.write(np.vstack([frame, plot_frame]))

    writer.release()
    cap.release()
    print(f"[INFO] Saved synced video to: {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
