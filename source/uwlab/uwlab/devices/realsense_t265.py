# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from enum import IntEnum
from typing import TYPE_CHECKING

from isaaclab.devices import DeviceBase
from isaaclab.utils.math import compute_pose_error

from uwlab.utils.math import create_axis_remap_function

if TYPE_CHECKING:
    from .device_cfg import RealsenseT265Cfg


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


class RealsenseT265(DeviceBase):
    def __init__(
        self,
        cfg: RealsenseT265Cfg,
        device="cuda:0",
    ):
        import pyrealsense2 as rs

        self.cam_device_id = cfg.cam_device_id
        self.device = device
        self._additional_callbacks = dict()
        self.pipeline = rs.pipeline()  # type: ignore
        self.config = rs.config()  # type: ignore

        self.ctx = rs.context()  # type: ignore
        self.t265_pipeline = rs.pipeline(self.ctx)  # type: ignore
        self.t265_config = rs.config()  # type: ignore
        self.t265_config.enable_device(self.cam_device_id)
        self.t265_config.enable_stream(rs.stream.pose)  # type: ignore
        self.init_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
        self.current_pose = self.init_pose.clone()
        self.t265_pipeline.start(self.t265_config)
        self.axis_remap_fn = create_axis_remap_function(forward="z", left="-x", up="y", device=self.device)

        self.quat_error = None

    def reset(self):
        self.current_pose = self.init_pose.clone()
        t265_frames = self.t265_pipeline.wait_for_frames()
        pose_frame = t265_frames.get_pose_frame()
        pose_data = pose_frame.get_pose_data()
        self.initial_quat = torch.tensor(
            [[pose_data.rotation.w, pose_data.rotation.x, pose_data.rotation.y, pose_data.rotation.z]],
            device=self.device,
        )
        self.initial_pos = torch.tensor(
            [[pose_data.translation.x, pose_data.translation.y, pose_data.translation.z]], device=self.device
        )

    def add_callback(self, key, func):
        # camera does not have any callbacks
        pass

    def advance(self):
        """
        Advance the device state, read the device translational and rotational data,
        then transform them xyz rpy that makes intuitive sense in Isaac Sim world coordinates.
        """
        t265_frames = self.t265_pipeline.wait_for_frames()
        pose_frame = t265_frames.get_pose_frame()
        pose_data = pose_frame.get_pose_data()

        curr_quat = torch.tensor(
            [[pose_data.rotation.w, pose_data.rotation.x, pose_data.rotation.y, pose_data.rotation.z]],
            device=self.device,
        )
        curr_pos = torch.tensor(
            [[pose_data.translation.x, pose_data.translation.y, pose_data.translation.z]], device=self.device
        )
        del_xyz, del_rpy = compute_pose_error(
            curr_pos, curr_quat, self.initial_pos, self.initial_quat, rot_error_type="axis_angle"
        )
        del_xyz, del_rpy = self.axis_remap_fn(del_xyz, del_rpy)

        self.current_pose[:, :3] = del_xyz
        self.current_pose[:, 3:] = del_rpy
        return self.current_pose
