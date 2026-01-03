# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

import pymodbus.client as ModbusClient
import urx
from pymodbus.constants import Endian
from pymodbus.framer import Framer
from pymodbus.payload import BinaryPayloadBuilder

from uwlab.assets.articulation.articulation_drive import ArticulationDrive

if TYPE_CHECKING:
    from .ur_driver_cfg import URDriverCfg


class URDriver(ArticulationDrive):
    def __init__(self, cfg: URDriverCfg, data_indices: slice = slice(None)):
        self.device = torch.device("cpu")
        self.cfg = cfg
        # self.work_space_limit = cfg.work_space_limit
        self.data_idx = data_indices

        self.current_pos = torch.zeros(1, 6, device=self.device)
        self.current_vel = torch.zeros(1, 6, device=self.device)
        self.current_eff = torch.zeros(1, 6, device=self.device)

    @property
    def ordered_joint_names(self):
        return [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

    def _prepare(self):
        # Initialize urx connection
        self.arm = urx.Robot(self.cfg.ip, use_rt=True)

        # Initialize Modbus Client
        self.modbus_client = ModbusClient.ModbusTcpClient(self.cfg.ip, port=self.cfg.port, framer=Framer.SOCKET)
        self.modbus_client.connect()

    def sendModbusValues(self, values):
        # Values will be divided by 100 in URScript
        values = np.array(values) * 100
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
        # Loop through each pose value and write it to a register
        for i in range(6):
            builder.reset()
            builder.add_16bit_int(int(values[i]))
            payload = builder.to_registers()
            self.modbus_client.write_register(128 + i, payload[0])

    def write_dof_targets(self, pos_target: torch.Tensor, vel_target: torch.Tensor, eff_target: torch.Tensor):
        # Non-blocking motion

        self.sendModbusValues(pos_target[0].tolist())

    def read_dof_states(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Blocking call to get_joint_states, storing the data in local torch Tensors."""
        joint_pos = np.array(self.arm.getj())
        pos = torch.tensor(joint_pos, device=self.device)
        self.current_pos[:] = pos
        return pos, self.current_vel, self.current_eff

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


# uncomment below code to run the worker
# if __name__ == "__main__":
#     # Create the worker
#     class Cfg:
#         ip = "192.168.1.2"
#         port = 602
#     driver = URDriver(cfg=Cfg())
#     driver._prepare()
#     pos, vel, eff = driver.read_dof_states()
#     print(pos, vel, eff)
#     driver.write_dof_targets(pos, vel, eff)

#     while True:
#         time.sleep(1)
