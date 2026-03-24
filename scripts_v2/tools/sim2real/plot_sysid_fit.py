# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Visualize system identification fit by replaying real waypoints through
the manager-based Sysid env (same RelCartesianOSCAction as sysid_ur5e_osc.py).

Loads a CMA-ES checkpoint, applies best params to the sim, runs closed-loop
replay, and plots sim vs. real joint trajectories.

Usage:
    python scripts_v2/tools/sim2real/plot_sysid_fit.py --headless \
        --checkpoint logs/sysid/YYYYMMDD_HHMMSS/checkpoint_0200.pt \
        --real_data sysid_data_real.pt
"""

import argparse
import matplotlib
import numpy as np
import os
import torch

matplotlib.use("Agg")
import gymnasium as gym
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Plot sysid fit")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to sysid checkpoint .pt")
parser.add_argument("--real_data", type=str, required=True, help="Path to real data .pt")
parser.add_argument("--max_steps", type=int, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.utils.math import subtract_frame_transforms

from uwlab_assets.robots.ur5e_robotiq_gripper.kinematics import ARM_JOINT_NAMES, EE_BODY_NAME, NUM_ARM_JOINTS

import uwlab_tasks  # noqa: F401  # register gym envs
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.sysid_cfg import SysidEnvCfg
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import settle_robot, target_pose_to_action

# ============================================================================
# Parameter application (same as sysid_ur5e_osc.py)
# ============================================================================


def apply_params(robot, params, arm_joint_ids, num_joints, device):
    """Apply 25-element param vector (single env) to robot."""
    N = 1
    env_ids = torch.arange(N, device=device)
    p = torch.tensor(params, device=device, dtype=torch.float32).unsqueeze(0)

    armature_full = torch.zeros(N, num_joints, device=device)
    static_friction_full = torch.zeros(N, num_joints, device=device)
    dynamic_friction_full = torch.zeros(N, num_joints, device=device)
    viscous_friction_full = torch.zeros(N, num_joints, device=device)
    armature_full[:, arm_joint_ids] = p[:, 0:6]
    static_fric = p[:, 6:12]
    dynamic_ratio = p[:, 12:18]
    static_friction_full[:, arm_joint_ids] = static_fric
    dynamic_friction_full[:, arm_joint_ids] = dynamic_ratio * static_fric
    viscous_friction_full[:, arm_joint_ids] = p[:, 18:24]
    robot.write_joint_armature_to_sim(armature_full, env_ids=env_ids)
    robot.write_joint_friction_coefficient_to_sim(
        static_friction_full,
        joint_dynamic_friction_coeff=dynamic_friction_full,
        joint_viscous_friction_coeff=viscous_friction_full,
        env_ids=env_ids,
    )

    delay_int = int(round(float(p[0, 24])))
    arm_actuator = robot.actuators["arm"]
    delay_tensor = torch.tensor([delay_int], device=device, dtype=torch.int)
    arm_actuator.positions_delay_buffer.set_time_lag(delay_tensor)
    arm_actuator.velocities_delay_buffer.set_time_lag(delay_tensor)
    arm_actuator.efforts_delay_buffer.set_time_lag(delay_tensor)


# ============================================================================
# Closed-loop replay
# ============================================================================


def closed_loop_replay(
    env,
    wp_step_indices,
    wp_target_pos,
    wp_target_quat,
    initial_joint_pos,
    arm_joint_ids,
    ee_frame_idx,
    sim_dt,
    T_steps,
    headless=True,
):
    """Run closed-loop replay using env's RelCartesianOSC. Returns sim trajectory dict."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    sim = unwrapped.sim
    device = unwrapped.device
    action_dim = unwrapped.action_manager.total_action_dim
    W = wp_step_indices.shape[0]

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    default_joint_pos[:, arm_joint_ids] = initial_joint_pos.unsqueeze(0)
    default_joint_vel[:] = 0.0
    env.reset()
    settle_robot(robot, sim, default_joint_pos, default_joint_vel, arm_joint_ids, sim_dt, headless=headless)

    sim_positions, sim_velocities, sim_ee_positions = [], [], []
    wp_idx = 0

    for t in range(T_steps):
        while wp_idx + 1 < W and t >= wp_step_indices[wp_idx + 1]:
            wp_idx += 1

        ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
        ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
        )
        target_pos = wp_target_pos[wp_idx].unsqueeze(0)
        target_quat = wp_target_quat[wp_idx].unsqueeze(0)

        action_arm = target_pose_to_action(ee_pos_b, ee_quat_b, target_pos, target_quat)
        action = torch.cat([action_arm, torch.zeros(1, action_dim - 6, device=device)], dim=-1)
        env.step(action)

        joint_pos = robot.data.joint_pos[:, arm_joint_ids]
        joint_vel = robot.data.joint_vel[:, arm_joint_ids]
        sim_positions.append(joint_pos[0].cpu().numpy().copy())
        sim_velocities.append(joint_vel[0].cpu().numpy().copy())
        sim_ee_positions.append(ee_pos_b[0].cpu().numpy().copy())

        if (t + 1) % max(1, T_steps // 20) == 0:
            print(f"  step {t+1}/{T_steps} ({100*(t+1)/T_steps:.0f}%)")

    return {
        "joint_positions": np.array(sim_positions),
        "joint_velocities": np.array(sim_velocities),
        "ee_positions": np.array(sim_ee_positions),
    }


# ============================================================================
# Plotting
# ============================================================================

JOINT_NAMES_SHORT = ["Shoulder Pan", "Shoulder Lift", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]


def plot_overlay(real_joints, sim_joints, dt, save_path="sysid_fit.png"):
    """Plot sim vs real joint positions."""
    T = real_joints.shape[0]
    time_axis = np.arange(T) * dt

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    for j in range(NUM_ARM_JOINTS):
        ax = axes[j]
        ax.plot(time_axis, np.degrees(real_joints[:, j]), "b-", linewidth=1.0, label="Real", alpha=0.8)
        ax.plot(time_axis, np.degrees(sim_joints[:, j]), "r-", linewidth=1.0, label="Sim", alpha=0.8)
        ax.set_title(JOINT_NAMES_SHORT[j], fontsize=11)
        ax.set_ylabel("deg")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-2].set_xlabel("Time (s)")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Sysid Fit: Sim vs Real Joint Trajectories", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved overlay plot: {save_path}")
    plt.close(fig)


def plot_error(real_joints, sim_joints, dt, save_path="sysid_fit_error.png"):
    """Plot per-joint error over time."""
    T = real_joints.shape[0]
    time_axis = np.arange(T) * dt
    error_deg = np.degrees(sim_joints - real_joints)

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    for j in range(NUM_ARM_JOINTS):
        ax = axes[j]
        ax.plot(time_axis, error_deg[:, j], "k-", linewidth=0.8, alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        rmse_j = np.sqrt(np.mean(error_deg[:, j] ** 2))
        ax.set_title(f"{JOINT_NAMES_SHORT[j]}  (RMSE={rmse_j:.2f}°)", fontsize=11)
        ax.set_ylabel("Error (deg)")
        ax.grid(True, alpha=0.3)

    axes[-2].set_xlabel("Time (s)")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Sysid Fit: Per-Joint Error", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved error plot:   {save_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================


def main():
    args = args_cli
    device_str = args.device

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    best_params = ckpt["best_params"]
    best_score = ckpt["best_score"]
    ckpt_args = ckpt.get("args", {})
    print(f"  Score (MSE): {best_score:.6f}  RMSE: {np.degrees(np.sqrt(best_score)):.4f}°")
    print(f"  Checkpoint args: sim_dt={ckpt_args.get('sim_dt', 'N/A')}")

    # Print best params
    arm = best_params[:6]
    sfric = best_params[6:12]
    dratio = best_params[12:18]
    vfric = best_params[18:24]
    delay = round(float(best_params[24]))
    print(f"\n  {'Joint':<20s} {'Armature':>10s} {'SFric':>10s} {'DRatio':>10s} {'VFric':>10s}")
    for i, name in enumerate(JOINT_NAMES_SHORT):
        print(f"  {name:<20s} {arm[i]:10.4f} {sfric[i]:10.4f} {dratio[i]:10.4f} {vfric[i]:10.4f}")
    print(f"  Motor delay: {delay} steps")

    # Load real data
    print(f"\nLoading real data: {args.real_data}")
    real_data = torch.load(args.real_data, map_location="cpu", weights_only=False)
    real_joint_pos = real_data["joint_positions"]
    initial_joint_pos = real_data["initial_joint_pos"]
    wp_step_indices = real_data["waypoint_step_indices"]
    wp_target_pos = real_data["waypoint_target_pos"]
    wp_target_quat = real_data["waypoint_target_quat"]
    dt = real_data["dt"]

    T_steps = real_joint_pos.shape[0]
    if args.max_steps is not None:
        T_steps = min(T_steps, args.max_steps)

    print(f"  {T_steps} steps ({T_steps*dt:.2f}s), dt={dt*1000:.1f}ms")

    # Move to GPU
    real_joint_pos_np = real_joint_pos[:T_steps].numpy()
    initial_joint_pos_dev = initial_joint_pos.to(device_str).float()
    wp_step_indices = wp_step_indices.to(device_str).long()
    wp_target_pos = wp_target_pos.to(device_str).float()
    wp_target_quat = wp_target_quat.to(device_str).float()

    # Create env (same as sysid_ur5e_osc.py)
    env_cfg = SysidEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.0
    delay_max = max(delay, 5)
    _effort_lim = {
        "shoulder_pan_joint": 150.0,
        "shoulder_lift_joint": 150.0,
        "elbow_joint": 150.0,
        "wrist_1_joint": 28.0,
        "wrist_2_joint": 28.0,
        "wrist_3_joint": 28.0,
    }
    _vel_lim = {
        "shoulder_pan_joint": 1.5708,
        "shoulder_lift_joint": 1.5708,
        "elbow_joint": 1.5708,
        "wrist_1_joint": 3.1415,
        "wrist_2_joint": 3.1415,
        "wrist_3_joint": 3.1415,
    }
    env_cfg.scene.robot.actuators["arm"] = DelayedPDActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness=0.0,
        damping=0.0,
        effort_limit=_effort_lim,
        velocity_limit=_vel_lim,
        min_delay=0,
        max_delay=delay_max,
    )
    env = gym.make("OmniReset-Ur5eRobotiq2f85-Sysid-v0", cfg=env_cfg)
    env.reset()

    unwrapped = env.unwrapped
    robot: Articulation = unwrapped.scene["robot"]
    device = unwrapped.device
    arm_joint_ids = robot.find_joints(ARM_JOINT_NAMES)[0]
    ee_frame_idx = robot.find_bodies(EE_BODY_NAME)[0][0]
    num_joints = robot.num_joints
    sim_dt = env_cfg.sim.dt

    # Apply best params
    print(f"\nApplying best params (delay={delay})...")
    apply_params(robot, best_params, arm_joint_ids, num_joints, device)

    # Run closed-loop replay
    print(f"\nRunning closed-loop replay ({T_steps} steps)...")
    result = closed_loop_replay(
        env,
        wp_step_indices,
        wp_target_pos,
        wp_target_quat,
        initial_joint_pos_dev,
        arm_joint_ids,
        ee_frame_idx,
        sim_dt,
        T_steps,
        headless=args_cli.headless,
    )

    sim_joints = result["joint_positions"]
    real_joints = real_joint_pos_np

    # Compute per-joint RMSE
    error_deg = np.degrees(sim_joints - real_joints)
    print(f"\n{'='*60}")
    print("Per-joint RMSE (deg)")
    print("=" * 60)
    for j in range(NUM_ARM_JOINTS):
        rmse_j = np.sqrt(np.mean(error_deg[:, j] ** 2))
        print(f"  {JOINT_NAMES_SHORT[j]:<16s}: {rmse_j:.4f}")

    rmse_total = np.sqrt(np.mean(error_deg**2))
    mae_total = np.mean(np.abs(error_deg))
    max_total = np.max(np.abs(error_deg))
    print(f"  TOTAL          : RMSE={rmse_total:.4f}  MAE={mae_total:.4f}  Max={max_total:.4f}")

    # Sysid-equivalent score for comparison with checkpoint
    error_rad = sim_joints - real_joints
    sysid_score = np.mean(np.sum(error_rad**2, axis=1))
    sysid_rmse_deg = np.degrees(np.sqrt(sysid_score))
    print(f"\n  Sysid-equivalent metric: score={sysid_score:.6f}  RMSE={sysid_rmse_deg:.4f}°")
    print(f"  Checkpoint metric:       score={best_score:.6f}  RMSE={np.degrees(np.sqrt(best_score)):.4f}°")
    print("=" * 60)

    # Plot
    out_dir = os.path.dirname(args.checkpoint) if os.path.dirname(args.checkpoint) else "."
    plot_overlay(real_joints, sim_joints, dt, save_path=os.path.join(out_dir, "sysid_fit.png"))
    plot_error(real_joints, sim_joints, dt, save_path=os.path.join(out_dir, "sysid_fit_error.png"))


if __name__ == "__main__":
    main()
    simulation_app.close()
