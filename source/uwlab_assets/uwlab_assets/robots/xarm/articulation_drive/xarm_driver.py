# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import time
import torch
from typing import TYPE_CHECKING

from uwlab.assets.articulation.articulation_drive import ArticulationDrive

if TYPE_CHECKING:
    from .xarm_driver_cfg import XarmDriverCfg


class XarmDriver(ArticulationDrive):
    def __init__(self, cfg: XarmDriverCfg, data_indices: slice = slice(None)):
        self.device = torch.device("cpu")
        self.cfg = cfg
        self.p_gain_scaler = cfg.p_gain_scaler
        self.work_space_limit = cfg.work_space_limit
        self.data_idx = data_indices
        work_space_limit = torch.tensor(cfg.work_space_limit, device=self.device)
        self.min_limits = work_space_limit[:, 0]
        self.max_limits = work_space_limit[:, 1]
        self.is_radian = cfg.is_radian

        self.current_pos = torch.zeros(1, 5, device=self.device)
        self.current_vel = torch.zeros(1, 5, device=self.device)
        self.current_eff = torch.zeros(1, 5, device=self.device)
        self._prepare()

    @property
    def ordered_joint_names(self):
        return ["joint" + str(i) for i in range(1, 6)]

    def close(self):
        self._arm.disconnect()

    def read_dof_states(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Blocking call to get_joint_states, storing the data in local torch Tensors."""
        code = 1
        while code != 0:
            code, (pos, vel, eff) = self._arm.get_joint_states(is_radian=True)
            if code != 0:
                self._log(f"Warning: get_joint_states returned code={code}", is_error=True)
                self._prepare()
        pos = torch.tensor(pos[:5], device=self.device).view(1, -1)
        vel = torch.tensor(vel[:5], device=self.device).view(1, -1)
        eff = torch.tensor(eff[:5], device=self.device).view(1, -1)
        self.current_pos[:] = pos
        self.current_vel[:] = vel
        self.current_eff[:] = eff
        return pos, vel, eff

    def _get_forward_kinematics(self, pos: torch.Tensor):
        code = 1
        while code != 0:
            code, pose = self._arm.get_forward_kinematics(pos[0].tolist(), input_is_radian=self.is_radian)
            if code != 0:
                self._log(f"Warning: get_forward_kinematics returned code={code}", is_error=True)
                self._prepare()
        ee_pose = torch.tensor(pose, device=self.device) / 1000  # convert to meter
        return ee_pose

    def write_dof_targets(self, pos_target: torch.Tensor, vel_target: torch.Tensor, eff_target: torch.Tensor):
        # Non-blocking motion
        position_error = pos_target - self.current_pos
        command = position_error * self.p_gain_scaler + self.current_pos

        ee_pose = self._get_forward_kinematics(command)
        within_workspace_limits = ((ee_pose[:3] > self.min_limits) & (ee_pose[:3] < self.max_limits)).all()

        if not within_workspace_limits:
            print(f"Arm action {ee_pose[:3]} canceled: arm target end-effector position is out of workspace limits.")
            return
        code = 1
        while code != 0:
            code = self._arm.set_servo_angle_j(command[0].tolist(), is_radian=True, wait=False)
            if code != 0:
                self._log(f"Warning: set_servo_angle_j returned code={code}", is_error=True)
                self._prepare()

    def set_dof_stiffnesses(self, stiffnesses):
        pass

    def set_dof_armatures(self, armatures):
        pass

    def set_dof_frictions(self, frictions):
        pass

    def set_dof_dampings(self, dampings):
        pass

    def set_dof_limits(self, limits):
        pass

    def _prepare(self):
        from xarm.wrapper import XArmAPI

        self._arm = XArmAPI(port=self.cfg.ip, is_radian=self.is_radian)
        self._arm.clean_error()
        self._arm.clean_warn()
        self._arm.motion_enable(enable=True)
        self._arm.set_mode(1)
        time.sleep(0.50)
        self._arm.set_collision_sensitivity(0)
        self._arm.set_state(0)
        time.sleep(0.50)

    def _log(self, msg, is_error=False):
        """
        Simple logging mechanism.
        In real code, use 'logging' module or other logging frameworks.
        """
        prefix = "[ERROR]" if is_error else "[INFO]"
        entry = f"{prefix} {time.strftime('%H:%M:%S')}: {msg}"
        print(entry)
