# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.devices import DeviceBase
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG, FRAME_MARKER_CFG

from uwlab.utils.math import create_axis_remap_function

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from .teleop_cfg import TeleopCfg


@dataclass
class TeleopState:
    device_interface: DeviceBase
    axis_remap: tuple[str, str, str]
    command_type: Literal["position", "pose"]
    attach_body: SceneEntityCfg
    attach_scope: Literal["self", "descendants"]
    pose_reference_body: SceneEntityCfg
    debug_vis: bool = False
    num_command_body: int = None
    reorient_func: Callable = None
    ref_pos_b: torch.Tensor = None
    ref_quat_b: torch.Tensor = None
    attach_pos_b: torch.Tensor = None
    attach_quat_b: torch.Tensor = None

    def update_ref(self, robot: Articulation):
        ref_body_id = self.pose_reference_body.body_ids
        ref_pos_b, ref_quat_b = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            robot.data.body_link_pos_w[:, ref_body_id, :].view(-1, 3),
            robot.data.body_link_quat_w[:, ref_body_id, :].view(-1, 4),
        )
        self.ref_pos_b = ref_pos_b.repeat_interleave(self.num_command_body, dim=0)
        self.ref_quat_b = ref_quat_b.repeat_interleave(self.num_command_body, dim=0)

    def update_attach(self, robot: Articulation):
        attach_body_id = self.attach_body.body_ids
        self.attach_pos_b, self.attach_quat_b = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            robot.data.body_link_pos_w[:, attach_body_id, :].view(-1, 3),
            robot.data.body_link_quat_w[:, attach_body_id, :].view(-1, 4),
        )
        self.attach_pos_b = self.attach_pos_b.repeat_interleave(self.num_command_body, dim=0)
        self.attach_quat_b = self.attach_quat_b.repeat_interleave(self.num_command_body, dim=0)

    def combine_frame_on_root(
        self, robot: Articulation, command_pos_b: torch.Tensor, command_quat_b: torch.Tensor | None
    ):
        command_pos_w, command_quat_w = math_utils.combine_frame_transforms(
            robot.data.root_pos_w.repeat_interleave(self.num_command_body, dim=0),
            robot.data.root_quat_w.repeat_interleave(self.num_command_body, dim=0),
            command_pos_b,
            command_quat_b,
        )
        return command_pos_w, command_quat_w

    def combine_frame_on_ref_b(self, command_pos: torch.Tensor, command_rot: torch.Tensor | None):
        if command_rot is not None and command_rot.shape[1] == 4:
            command_rot = math_utils.axis_angle_from_quat(command_rot[:, 3:])

        command_pos, command_rot = self.reorient_func(command_pos, command_rot)
        if command_rot is not None:
            command_rot = math_utils.quat_from_euler_xyz(command_rot[:, 0], command_rot[:, 1], command_rot[:, 2])
        command_pos_b, command_quat_b = math_utils.combine_frame_transforms(
            t01=self.ref_pos_b,
            q01=self.ref_quat_b,
            t12=command_pos,
            q12=command_rot,
        )
        return command_pos_b, command_quat_b

    def combine_frame_on_attach_b(self, command_pos: torch.Tensor, command_quat: torch.Tensor | None):
        if self.attach_scope == "descendants":
            command_pos_b, command_quat_b = math_utils.combine_frame_transforms(
                t01=self.attach_pos_b,
                q01=self.attach_quat_b,
                t12=command_pos,
                q12=command_quat,
            )
        else:
            command_pos_b = self.attach_pos_b + command_pos
            if command_quat is None:
                return command_pos_b, None
            command_quat_b = math_utils.quat_mul(command_quat, self.attach_quat_b)
        return command_pos_b, command_quat_b


class Teleop:
    def __init__(self, cfg: TeleopCfg, env):
        self._env = env
        self.num_envs = env.unwrapped.num_envs
        self.cfg = cfg
        devices = cfg.teleop_devices
        self.teleops: list[TeleopState] = []
        for device in devices.values():
            teleop_device = TeleopState(
                device_interface=device.teleop_interface_cfg.class_type(device.teleop_interface_cfg, env.device),
                axis_remap=device.reference_axis_remap,
                command_type=device.command_type,
                attach_body=device.attach_body,
                attach_scope=device.attach_scope,
                pose_reference_body=device.pose_reference_body,
                debug_vis=device.debug_vis,
            )
            self.teleops.append(teleop_device)
            teleop_device.device_interface.reset()
            teleop_command = teleop_device.device_interface.advance()
            teleop_command = (teleop_command,) if not isinstance(teleop_command, tuple) else teleop_command
            teleop_device.num_command_body = teleop_command[0].shape[0]

            total_command_bodys = self.num_envs * teleop_device.num_command_body
            teleop_device.ref_pos_b = torch.zeros((total_command_bodys, 3), device=env.device)
            teleop_device.ref_quat_b = torch.zeros((total_command_bodys, 4), device=env.device)
            teleop_device.attach_pos_b = torch.zeros((total_command_bodys, 3), device=env.device)
            teleop_device.attach_quat_b = torch.zeros((total_command_bodys, 4), device=env.device)

            device.attach_body.resolve(env.unwrapped.scene)
            device.pose_reference_body.resolve(env.unwrapped.scene)

        self.device = env.device
        self.command_w = None

        frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
        frame_marker_cfg.markers["frame"].scale = (0.025, 0.025, 0.025)
        cuboid_marker_cfg = CUBOID_MARKER_CFG.copy()  # type: ignore
        cuboid_marker_cfg.markers["cuboid"].size = (0.01, 0.01, 0.01)
        self.pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/device_pose"))
        self.position_marker = VisualizationMarkers(cuboid_marker_cfg.replace(prim_path="/Visuals/device_position"))
        self.robot = self._env.unwrapped.scene["robot"]

    @property
    def command(self):
        return self.command_w

    def add_callback(self, key: str, func: Callable):
        # check keys supported by callback
        for teleop_device in self.teleops:
            teleop_device.device_interface.add_callback(key, func)

    def reset(self):
        robot = self._env.unwrapped.scene["robot"]
        for teleop_device in self.teleops:
            teleop_device.device_interface.reset()

            teleop_device.update_ref(robot)
            teleop_device.update_attach(robot)
            teleop_device.reorient_func = create_axis_remap_function(*teleop_device.axis_remap, device=self.device)

    def advance(self):
        self.command_b = []
        self.visualization_pose = []
        self.visualization_position = []
        for teleop_device in self.teleops:
            teleop_command = teleop_device.device_interface.advance()
            teleop_command = (teleop_command,) if not isinstance(teleop_command, tuple) else teleop_command
            teleop_command = self._world_command_to_robot_command(teleop_device, *teleop_command)
            self.command_b.append(teleop_command)

        self._debug_vis()
        return torch.cat(self.command_b, dim=1)

    def _debug_vis(self):
        if len(self.visualization_pose) > 0:
            pose = torch.cat(self.visualization_pose, dim=0).view(-1, 7)
            self.pose_marker.visualize(
                translations=pose[:, :3],
                orientations=pose[:, 3:],
            )

        if len(self.visualization_position) > 0:
            position = torch.cat(self.visualization_position, dim=0).view(-1, 3)
            self.position_marker.visualize(
                translations=position,
            )

    def _world_command_to_robot_command(self, teleop_device: TeleopState, command: torch.Tensor, *args):
        robot = self._env.unwrapped.scene["robot"]
        command_shape = command.shape[1]

        teleop_device.update_ref(robot)
        if teleop_device.attach_scope == "descendants":
            teleop_device.update_attach(robot)

        command = command.repeat(self.num_envs, 1)
        if command_shape == 3:
            command_pos_b, _ = teleop_device.combine_frame_on_ref_b(command[:, :3], None)
            command_pos_b, _ = teleop_device.combine_frame_on_attach_b(command_pos_b, None)

            if teleop_device.debug_vis:
                pos_w, _ = teleop_device.combine_frame_on_root(robot, command_pos_b, None)
                self.visualization_position.append(pos_w)

            command_res = command_pos_b.view(self.num_envs, -1)
        else:
            command_pos_b, command_quat_b = teleop_device.combine_frame_on_ref_b(command[:, :3], command[:, 3:])
            command_pos_b, command_quat_b = teleop_device.combine_frame_on_attach_b(command_pos_b, command_quat_b)

            if teleop_device.debug_vis:
                pos_w, quat_w = teleop_device.combine_frame_on_root(robot, command_pos_b, command_quat_b)
                self.visualization_pose.append(torch.cat([pos_w, quat_w], dim=1))

            command_res = torch.cat([command_pos_b, command_quat_b], dim=1).view(self.num_envs, -1)

        if len(args) > 0:
            arg_cat = torch.cat(args, dim=1).repeat(self.num_envs, 1)
            command_res = torch.cat([command_res, arg_cat], dim=1)

        return command_res
