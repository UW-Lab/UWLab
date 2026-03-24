# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive sim2real camera alignment for UR5e + Robotiq 2F-85.

Renders the simulation camera view, blends it with a real reference image,
and lets you move/rotate the camera and adjust focal length with the keyboard.
Press 'p' to print the final (pos, rot, focal_length) values that you can
paste into data_collection_rgb_cfg.py.

Mirrors the sysid workflow:
  scripts_v2/tools/sim2real/sysid_ur5e_osc.py   →  tunes physics params
  scripts_v2/tools/sim2real/align_cameras.py    →  tunes camera poses

Usage (front camera example):
    python scripts_v2/tools/sim2real/align_cameras.py \
        --enable_cameras \
        --camera front_camera \
        --real_image /path/to/real_front.png \
        --joint_angles -12.0 -80.0 63.0 -30.6 -97.9 174.3

Usage (wrist camera example):
    python scripts_v2/tools/sim2real/align_cameras.py \
        --enable_cameras \
        --camera wrist_camera \
        --real_image /path/to/real_wrist.png \
        --joint_angles -12.0 -80.0 63.0 -30.6 -97.9 174.3

Keyboard controls:
    w/x         move +/- X           i/k  pitch +/-
    a/d         move +/- Y           j/l  yaw +/-
    up/down     move +/- Z           u/o  roll +/-
    left/right  focal length -/+
    1/2         blend ratio -/+  (0 = all sim, 1 = all real)
    +/-         position step size +/-
    r           reset camera to initial pose
    p           print camera params & save current view
    q           quit
"""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sim2Real camera alignment tool.")
parser.add_argument(
    "--camera",
    type=str,
    default="front_camera",
    choices=["front_camera", "side_camera", "wrist_camera"],
    help="Which camera to align",
)
parser.add_argument("--real_image", type=str, default=None, help="Path to reference real RGB image (png/jpg)")
parser.add_argument(
    "--joint_angles",
    type=float,
    nargs=6,
    default=[2.28, -95.58, 99.07, -93.36, -86.57, 4.33],
    help="Arm joint angles in degrees (6 joints). Default matches real_env.py default init pose.",
)
parser.add_argument("--gripper_pos", type=float, default=1.0, help="Gripper position (0=closed, 1=open)")
parser.add_argument("--warmup_steps", type=int, default=30, help="Simulation warmup steps before interaction")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402

from pxr import Gf, UsdGeom  # noqa: E402

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.camera_align_cfg import (
    CameraAlignEnvCfg,
)

# ---- RGB key lookup ----
CAMERA_TO_RGB = {
    "front_camera": "front_rgb",
    "side_camera": "side_rgb",
    "wrist_camera": "wrist_rgb",
}


class CameraAligner:
    """Interactive controller: keyboard → camera pose → blended view."""

    def __init__(self, env, camera_key, real_img, fig, ax):
        self.env = env
        self.camera_key = camera_key
        self.rgb_key = CAMERA_TO_RGB[camera_key]
        self.fig = fig
        self.ax = ax
        self.real_img = real_img  # (H, W, 3) float [0,1]

        self.camera = self.env.unwrapped.scene._sensors[camera_key]

        # Read initial LOCAL pose from the USD prim XformOps (offset relative to parent).
        # We work in local space because USD XformOps are authoritative and survive
        # the USD→Fabric sync that happens each sim step (unlike Fabric-only writes).
        prim = self.camera._sensor_prims[0]
        xformable = UsdGeom.Xformable(prim)
        self._xform_ops = {op.GetOpType(): op for op in xformable.GetOrderedXformOps()}
        self.pos = np.array(self._xform_ops[UsdGeom.XformOp.TypeTranslate].Get(), dtype=np.float64)
        quat = self._xform_ops[UsdGeom.XformOp.TypeOrient].Get()
        self.rot = np.array([quat.GetReal(), *quat.GetImaginary()], dtype=np.float64)

        # Tuning step sizes
        self.pos_step = 0.005
        self.rot_step = 0.005
        self.focal_step = 0.1
        self.blend = 0.5

        # We'll store the most recent obs
        self.obs = None
        self.action = None

    # ---- quaternion ↔ euler helpers (OpenGL convention) ----
    @staticmethod
    def quat_to_euler(q):
        w, x, y, z = q
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([roll, pitch, yaw])

    @staticmethod
    def euler_to_quat(e):
        r, p, y = e / 2.0
        cr, cp, cy = np.cos(r), np.cos(p), np.cos(y)
        sr, sp, sy = np.sin(r), np.sin(p), np.sin(y)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ])

    # ---- update sim ----
    def apply_camera_pose(self):
        w, x, y, z = self.rot.tolist()
        self._xform_ops[UsdGeom.XformOp.TypeTranslate].Set(Gf.Vec3d(*self.pos.tolist()))
        self._xform_ops[UsdGeom.XformOp.TypeOrient].Set(Gf.Quatd(w, x, y, z))

    def step_and_render(self):
        self.obs, _, _, _, _ = self.env.step(self.action)

    # ---- visualize ----
    def update_view(self):
        sim_rgb = self.obs["policy"][self.rgb_key]
        # (1, C, H, W) or (1, H, W, C) – handle both
        img = sim_rgb[0]
        if img.shape[0] in (3, 4):
            img = img.permute(1, 2, 0)
        img = img.cpu().numpy().astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0

        # Resize real to match sim if needed
        real = self.real_img
        if real is not None:
            if real.shape[:2] != img.shape[:2]:
                from PIL import Image

                real = (
                    np.array(Image.fromarray((real * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0])))
                    / 255.0
                )
            blended = (1 - self.blend) * img[..., :3] + self.blend * real[..., :3]
        else:
            blended = img[..., :3]

        self.ax.clear()
        self.ax.imshow(np.clip(blended, 0, 1))
        info = f"cam={self.camera_key}  blend={self.blend:.2f}  step={self.pos_step:.4f}"
        self.ax.set_title(info, fontsize=9)
        self.ax.axis("off")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ---- keyboard ----
    def on_key(self, event):
        k = event.key
        changed = True

        # --- position ---
        if k == "w":
            self.pos[0] += self.pos_step
        elif k == "x":
            self.pos[0] -= self.pos_step
        elif k == "a":
            self.pos[1] += self.pos_step
        elif k == "d":
            self.pos[1] -= self.pos_step
        elif k == "up":
            self.pos[2] += self.pos_step
        elif k == "down":
            self.pos[2] -= self.pos_step

        # --- rotation ---
        elif k in ("i", "k", "j", "l", "u", "o"):
            e = self.quat_to_euler(self.rot)
            if k == "i":
                e[1] += self.rot_step
            elif k == "k":
                e[1] -= self.rot_step
            elif k == "j":
                e[2] += self.rot_step
            elif k == "l":
                e[2] -= self.rot_step
            elif k == "u":
                e[0] += self.rot_step
            elif k == "o":
                e[0] -= self.rot_step
            self.rot = self.euler_to_quat(e)

        # --- focal length ---
        elif k in ("left", "right"):
            prim = self.camera._sensor_prims[0]
            fl = prim.GetFocalLengthAttr().Get()
            fl += self.focal_step if k == "right" else -self.focal_step
            prim.GetFocalLengthAttr().Set(fl)
            print(f"focal_length={fl:.2f}")

        # --- blend ---
        elif k == "1":
            self.blend = max(0.0, self.blend - 0.1)
        elif k == "2":
            self.blend = min(1.0, self.blend + 0.1)

        # --- step size ---
        elif k == "+":
            self.pos_step *= 2.0
            print(f"pos_step={self.pos_step:.5f}")
        elif k == "-":
            self.pos_step /= 2.0
            print(f"pos_step={self.pos_step:.5f}")

        # --- reset ---
        elif k == "r":
            self.pos = np.array(self.camera.cfg.offset.pos, dtype=np.float64)
            rot_cfg = self.camera.cfg.offset.rot
            self.rot = np.array(rot_cfg, dtype=np.float64)
            print("Reset to initial camera pose")

        # --- print / save ---
        elif k == "p":
            self._print_params()
            changed = False

        # --- quit ---
        elif k == "q":
            plt.close(self.fig)
            return

        else:
            changed = False

        if changed:
            self.apply_camera_pose()
            self.step_and_render()
            self.update_view()

    def _print_params(self):
        prim = self.camera._sensor_prims[0]
        fl = prim.GetFocalLengthAttr().Get()
        euler = self.quat_to_euler(self.rot)

        print("\n" + "=" * 60)
        print(f"Camera: {self.camera_key}")
        print(f"Offset pos (x, y, z):          ({self.pos[0]:.7f}, {self.pos[1]:.7f}, {self.pos[2]:.7f})")
        print(
            f"Quaternion (w, x, y, z):       ({self.rot[0]:.8f}, {self.rot[1]:.8f}, {self.rot[2]:.8f},"
            f" {self.rot[3]:.8f})"
        )
        print(f"Euler (roll, pitch, yaw) rad:  ({euler[0]:.6f}, {euler[1]:.6f}, {euler[2]:.6f})")
        print(f"Focal length:                  {fl:.2f}")
        print()
        print("--- Paste into data_collection_rgb_cfg.py ---")
        print("--- (same values for BOTH TiledCameraCfg.OffsetCfg AND BaseRGBEventCfg) ---")
        print(f"    pos=({self.pos[0]:.7f}, {self.pos[1]:.7f}, {self.pos[2]:.7f}),")
        print(f"    rot=({self.rot[0]:.8f}, {self.rot[1]:.8f}, {self.rot[2]:.8f}, {self.rot[3]:.8f}),")
        print(f"    focal_length={fl:.2f}")
        print("=" * 60 + "\n")

        # save screenshot
        sim_rgb = self.obs["policy"][self.rgb_key][0]
        if sim_rgb.shape[0] in (3, 4):
            sim_rgb = sim_rgb.permute(1, 2, 0)
        img = sim_rgb.cpu().numpy()
        if img.max() > 1.5:
            img = (img / 255.0).clip(0, 1)
        out_path = f"camera_align_{self.camera_key}.png"
        plt.imsave(out_path, img[..., :3])
        print(f"Saved sim view to {out_path}")


def main():
    # Create the camera-alignment environment
    env_cfg = CameraAlignEnvCfg()

    # Override default joint positions to match the real robot pose.
    # joint_angles are in degrees; convert to radians for the init_state.
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    joint_rads = [float(np.deg2rad(a)) for a in args_cli.joint_angles]
    for name, rad in zip(joint_names, joint_rads):
        env_cfg.scene.robot.init_state.joint_pos[name] = rad

    env = gym.make("OmniReset-Ur5eRobotiq2f85-CameraAlign-v0", cfg=env_cfg)
    device = env.unwrapped.device

    # Send zero OSC deltas so the robot holds the init joint config.
    arm_dim = 6
    gripper_dim = 1
    action = torch.zeros(1, arm_dim + gripper_dim, device=device)
    action[0, -1] = args_cli.gripper_pos  # gripper open

    # Reset and warm up
    print(f"Warming up simulation for {args_cli.warmup_steps} steps...")
    obs, _ = env.reset()

    for _ in range(args_cli.warmup_steps):
        obs, _, _, _, _ = env.step(action)

    # Load real reference image
    real_img = None
    if args_cli.real_image is not None:
        real_img = plt.imread(args_cli.real_image)[..., :3].astype(np.float32)
        if real_img.max() > 1.5:
            real_img = real_img / 255.0
        print(f"Loaded reference image: {args_cli.real_image}  shape={real_img.shape}")
    else:
        print("No --real_image provided; showing sim-only view. Press 1/2 to adjust blend ratio once you supply one.")

    # Set up matplotlib — clear default keybindings that conflict with controls
    for key in plt.rcParams:
        if key.startswith("keymap."):
            plt.rcParams[key] = []
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.ion()

    aligner = CameraAligner(env, args_cli.camera, real_img, fig, ax)
    aligner.action = action
    aligner.obs = obs

    fig.canvas.mpl_connect("key_press_event", aligner.on_key)

    aligner.update_view()
    print("\nCamera alignment ready. Use keyboard to adjust (see --help for keys).")
    plt.show(block=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
