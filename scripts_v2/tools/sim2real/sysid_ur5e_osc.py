# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
System Identification for UR5e using CMA-ES (Closed-Loop Replay).

Uses the manager-based env (OmniReset-Ur5eRobotiq2f85-Sysid-v0) so the same
RelCartesianOSCAction as RL is used — no duplicate OSC. PACE-style integration.

Parameters (25 total): armature*6, static_friction*6, dynamic_ratio*6,
  viscous_friction*6, motor_delay*1.

Usage:
    python scripts_v2/tools/sim2real/sysid_ur5e_osc.py --headless --num_envs 512 \
        --real_data sysid_data_real.pt --max_iter 200
"""

import argparse
import gymnasium as gym
import numpy as np
import os
import time
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR5e System Identification via CMA-ES")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--real_data", type=str, required=True)
parser.add_argument("--max_iter", type=int, default=200)
parser.add_argument("--sigma", type=float, default=0.3)
parser.add_argument("--output_dir", type=str, default="logs/sysid")
parser.add_argument("--save_interval", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=None)
# Parameter bounds
parser.add_argument("--armature_min", type=float, default=0.0)
parser.add_argument("--armature_max", type=float, default=10.0)
parser.add_argument("--friction_min", type=float, default=0.0)
parser.add_argument("--friction_max", type=float, default=20.0)
parser.add_argument("--viscous_friction_min", type=float, default=0.0)
parser.add_argument("--viscous_friction_max", type=float, default=20.0)
parser.add_argument(
    "--delay_max", type=int, default=5, help="Max motor delay in physics steps. CMA-ES searches [0, delay_max]."
)

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
# CMA-ES Optimizer
# ============================================================================


class CMAES:
    """Lightweight CMA-ES wrapper using the cmaes library."""

    def __init__(self, num_params, population_size, sigma=0.3, bounds=None):
        from cmaes import CMA

        self.num_params = num_params
        self.population_size = population_size
        self.bounds = np.array(bounds)
        self.optimizer = CMA(
            mean=np.full(num_params, 0.5),
            sigma=sigma,
            population_size=population_size,
            bounds=np.column_stack([np.zeros(num_params), np.ones(num_params)]),
        )
        self._solutions = None

    def ask(self) -> np.ndarray:
        self._solutions = []
        for _ in range(self.population_size):
            self._solutions.append(self.optimizer.ask())
        normalized = np.array(self._solutions)
        return self.bounds[:, 0] + normalized * (self.bounds[:, 1] - self.bounds[:, 0])

    def tell(self, scores: np.ndarray):
        self.optimizer.tell(list(zip(self._solutions, scores.tolist())))

    @property
    def best_params(self) -> np.ndarray:
        mean_normalized = self.optimizer._mean
        return self.bounds[:, 0] + mean_normalized * (self.bounds[:, 1] - self.bounds[:, 0])


# ============================================================================
# Parameter Mapping
# ============================================================================


def build_bounds(args):
    """25 params: [armature*6, static_friction*6, dynamic_ratio*6, viscous_friction*6, delay*1]."""
    bounds = []
    for _ in range(NUM_ARM_JOINTS):
        bounds.append([args.armature_min, args.armature_max])
    for _ in range(NUM_ARM_JOINTS):
        bounds.append([args.friction_min, args.friction_max])
    for _ in range(NUM_ARM_JOINTS):
        bounds.append([0.0, 1.0])  # dynamic_ratio
    for _ in range(NUM_ARM_JOINTS):
        bounds.append([args.viscous_friction_min, args.viscous_friction_max])
    bounds.append([0.0, float(args.delay_max)])  # motor_delay
    return bounds


def apply_params_to_envs(robot, params_tensor, arm_joint_ids, num_joints, device):
    """Apply 25 params to all envs (joint dynamics + per-env motor delay)."""
    N = params_tensor.shape[0]
    env_ids = torch.arange(N, device=device)

    armature_full = torch.zeros(N, num_joints, device=device)
    static_friction_full = torch.zeros(N, num_joints, device=device)
    dynamic_friction_full = torch.zeros(N, num_joints, device=device)
    viscous_friction_full = torch.zeros(N, num_joints, device=device)
    armature_full[:, arm_joint_ids] = params_tensor[:, 0:6]
    static_fric = params_tensor[:, 6:12]
    dynamic_ratio = params_tensor[:, 12:18]
    static_friction_full[:, arm_joint_ids] = static_fric
    dynamic_friction_full[:, arm_joint_ids] = dynamic_ratio * static_fric
    viscous_friction_full[:, arm_joint_ids] = params_tensor[:, 18:24]
    robot.write_joint_armature_to_sim(armature_full, env_ids=env_ids)
    robot.write_joint_friction_coefficient_to_sim(
        static_friction_full,
        joint_dynamic_friction_coeff=dynamic_friction_full,
        joint_viscous_friction_coeff=viscous_friction_full,
        env_ids=env_ids,
    )

    # Motor delay: continuous -> round to int, set per-env on actuator buffers
    delay_int = torch.round(params_tensor[:, 24]).clamp(min=0).to(torch.int)
    arm_actuator = robot.actuators["arm"]
    arm_actuator.positions_delay_buffer.set_time_lag(delay_int)
    arm_actuator.velocities_delay_buffer.set_time_lag(delay_int)
    arm_actuator.efforts_delay_buffer.set_time_lag(delay_int)


# ============================================================================
# Main
# ============================================================================


def main():
    args = args_cli
    device_str = args.device
    N = args.num_envs
    num_params = NUM_ARM_JOINTS * 4 + 1  # 25

    print("\n" + "=" * 60)
    print("UR5e System Identification - CMA-ES (Closed-Loop Replay)")
    print("=" * 60)
    print(f"Envs: {N}  Params: {num_params}  Iters: {args.max_iter}  Sigma: {args.sigma}")
    print("Controller: env's RelCartesianOSC (same as RL)")
    print(f"Motor delay: optimized [0, {args.delay_max}] steps")

    # Load real data
    print(f"\nLoading: {args.real_data}")
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
    W = wp_step_indices.shape[0]

    print(f"  {T_steps} steps ({T_steps*dt:.2f}s), {W} waypoints, dt={dt*1000:.1f}ms")

    # Move to GPU
    real_joint_pos = real_joint_pos[:T_steps].to(device_str).float()
    initial_joint_pos_dev = initial_joint_pos.to(device_str).float()
    wp_step_indices = wp_step_indices.to(device_str).long()
    wp_target_pos = wp_target_pos.to(device_str).float()
    wp_target_quat = wp_target_quat.to(device_str).float()

    # Manager-based env (same RelCartesianOSC as RL)
    env_cfg = SysidEnvCfg()
    env_cfg.scene.num_envs = N
    env_cfg.scene.env_spacing = 2.0
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
        max_delay=args.delay_max,
    )
    env = gym.make("OmniReset-Ur5eRobotiq2f85-Sysid-v0", cfg=env_cfg)
    env.reset()

    unwrapped = env.unwrapped
    robot: Articulation = unwrapped.scene["robot"]
    sim = unwrapped.sim
    device = unwrapped.device
    arm_joint_ids = robot.find_joints(ARM_JOINT_NAMES)[0]
    ee_frame_idx = robot.find_bodies(EE_BODY_NAME)[0][0]
    num_joints = robot.num_joints
    sim_dt = env_cfg.sim.dt
    action_dim = unwrapped.action_manager.total_action_dim  # 7 (arm 6 + gripper 1)

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    default_joint_pos[:, arm_joint_ids] = initial_joint_pos_dev.unsqueeze(0).expand(N, -1)
    default_joint_vel[:] = 0.0

    bounds = build_bounds(args)
    cmaes = CMAES(num_params=num_params, population_size=N, sigma=args.sigma, bounds=bounds)

    print(
        f"\nBounds: armature[{args.armature_min},{args.armature_max}] "
        f"friction[{args.friction_min},{args.friction_max}] "
        f"dyn_ratio[0,1] viscous[{args.viscous_friction_min},{args.viscous_friction_max}] "
        f"delay[0,{args.delay_max}]"
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}\n")

    best_score_ever = float("inf")
    best_params_ever = None
    history = []

    for iteration in range(args.max_iter):
        iter_start = time.time()

        params_np = cmaes.ask()
        params_tensor = torch.tensor(params_np, device=device, dtype=torch.float32)
        apply_params_to_envs(robot, params_tensor, arm_joint_ids, num_joints, device)

        env.reset()
        settle_robot(robot, sim, default_joint_pos, default_joint_vel, arm_joint_ids, sim_dt, headless=True)

        scores = torch.zeros(N, device=device)
        wp_idx = 0

        for t in range(T_steps):
            while wp_idx + 1 < W and t >= wp_step_indices[wp_idx + 1]:
                wp_idx += 1

            ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
            ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
            )
            target_pos = wp_target_pos[wp_idx].unsqueeze(0).expand(N, -1)
            target_quat = wp_target_quat[wp_idx].unsqueeze(0).expand(N, -1)

            action_arm = target_pose_to_action(ee_pos_b, ee_quat_b, target_pos, target_quat)
            action = torch.cat([action_arm, torch.zeros(N, action_dim - 6, device=device)], dim=-1)
            env.step(action)

            joint_pos = robot.data.joint_pos[:, arm_joint_ids]
            scores += torch.sum((joint_pos - real_joint_pos[t].unsqueeze(0)) ** 2, dim=1)

        scores = scores / T_steps
        scores_np = scores.cpu().numpy()
        cmaes.tell(scores_np)

        min_score = scores_np.min()
        mean_score = scores_np.mean()
        iter_time = time.time() - iter_start

        if min_score < best_score_ever:
            best_score_ever = min_score
            best_params_ever = params_np[scores_np.argmin()]

        history.append(
            {"iteration": iteration, "min": float(min_score), "mean": float(mean_score), "best": float(best_score_ever)}
        )

        best_delay = round(float(best_params_ever[24]))
        rmse_deg = np.degrees(np.sqrt(best_score_ever))
        print(
            f"[{iteration+1:3d}/{args.max_iter}] "
            f"min={min_score:.6f} mean={mean_score:.6f} best={best_score_ever:.6f} "
            f"({rmse_deg:.3f}\u00b0 delay={best_delay}) {iter_time:.1f}s"
        )

        if (iteration + 1) % args.save_interval == 0:
            ckpt = {
                "best_params": best_params_ever,
                "best_score": best_score_ever,
                "iteration": iteration + 1,
                "history": history,
                "bounds": bounds,
                "args": vars(args),
            }
            ckpt_path = os.path.join(output_dir, f"checkpoint_{iteration+1:04d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"  -> {ckpt_path}")

    # Final results
    print(f"\n{'='*60}")
    print(f"DONE  RMSE: {np.degrees(np.sqrt(best_score_ever)):.4f}\u00b0")
    print(f"{'='*60}")

    arm = best_params_ever[:6]
    sfric = best_params_ever[6:12]
    dratio = best_params_ever[12:18]
    dfric = dratio * sfric
    vfric = best_params_ever[18:24]
    delay = round(float(best_params_ever[24]))

    print(f"\n  {'Joint':<25s} {'Arm':>8s} {'SFric':>8s} {'DRat':>8s} {'DFric':>8s} {'VFric':>8s}")
    for i, name in enumerate(ARM_JOINT_NAMES):
        print(f"  {name:<25s} {arm[i]:8.4f} {sfric[i]:8.4f} {dratio[i]:8.4f} {dfric[i]:8.4f} {vfric[i]:8.4f}")
    print(f"\n  Motor delay: {delay} steps ({delay*sim_dt*1000:.0f}ms at {1/sim_dt:.0f}Hz)")

    final = {
        "best_params": best_params_ever,
        "best_score": best_score_ever,
        "best_armature": arm.tolist(),
        "best_friction": sfric.tolist(),
        "best_dynamic_ratio": dratio.tolist(),
        "best_dynamic_friction": dfric.tolist(),
        "best_viscous_friction": vfric.tolist(),
        "best_delay": delay,
        "history": history,
        "bounds": bounds,
        "args": vars(args),
    }
    final_path = os.path.join(output_dir, "final_results.pt")
    torch.save(final, final_path)
    print(f"\nSaved: {final_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
