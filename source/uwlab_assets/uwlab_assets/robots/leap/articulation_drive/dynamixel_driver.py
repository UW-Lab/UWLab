# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from uwlab.assets.articulation.articulation_drive import ArticulationDrive

if TYPE_CHECKING:
    from .dynamixel_driver_cfg import DynamixelDriverCfg


class DynamixelDriver(ArticulationDrive):
    def __init__(self, cfg: DynamixelDriverCfg, data_indices: slice = slice(None)):
        self.device = cfg.device
        assert self.device == "cpu", "Dynamixel driver only supports CPU mode"
        self.cfg = cfg
        self.data_idx = data_indices
        hand_kI = cfg.hand_kI
        hand_curr_lim = cfg.hand_curr_lim
        self.offset = 3.14159

        # Initialize client in CPU-only environment
        self.joint_idx = torch.arange(16, device=self.device)
        self.joint_integral = torch.ones(len(self.joint_idx), device=self.device) * hand_kI
        self.magic = torch.ones(len(self.joint_idx), device=self.device) * 5
        self.joint_curr_lim = torch.ones(len(self.joint_idx), device=self.device) * hand_curr_lim
        self._prepare()

    @property
    def ordered_joint_names(self):
        return ["j" + str(i) for i in range(16)]

    def close(self):
        self.dxl_client.disconnect()

    def read_dof_states(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = torch.tensor((self.dxl_client.read_pos() - self.offset), device=self.device).view(1, -1)
        vel = torch.tensor(self.dxl_client.read_vel(), device=self.device).view(1, -1)
        eff = torch.zeros_like(pos, device=self.device)
        return pos, vel, eff

    def write_dof_targets(self, pos_target: torch.Tensor, vel_target: torch.Tensor, eff_target: torch.Tensor):
        pos_target = pos_target[0] + self.offset
        self.dxl_client.write_desired_pos(self.joint_idx.tolist(), pos_target.numpy())

    def set_dof_stiffnesses(self, stiffnesses):
        self.dxl_client.sync_write(self.joint_idx.tolist(), stiffnesses[0].tolist(), 84, 2)  # Pgain stiffness

    def set_dof_armatures(self, armatures):
        pass

    def set_dof_frictions(self, frictions):
        pass

    def set_dof_dampings(self, dampings):
        self.dxl_client.sync_write(self.joint_idx.tolist(), dampings[0].tolist(), 80, 2)

    def set_dof_limits(self, limits):
        pass

    def _prepare(self):
        from .dynamixel_client import DynamixelClient

        self.dxl_client = DynamixelClient(
            port=self.cfg.port,
            motor_ids=[i for i in range(16)],
            baudrate=4000000,
        )
        self.dxl_client.connect()
        self.dxl_client.sync_write(self.joint_idx.tolist(), self.joint_integral.tolist(), 82, 2)  # Igain
        self.dxl_client.sync_write(self.joint_idx.tolist(), self.magic.tolist(), 11, 1)
        self.dxl_client.set_torque_enabled(self.joint_idx.tolist(), True)

        # Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(self.joint_idx.tolist(), self.joint_curr_lim.tolist(), 102, 2)

        print("Dynamixel driver initialized")
