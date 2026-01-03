# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from uwlab.assets.articulation.articulation_drive import ArticulationDrive

if TYPE_CHECKING:
    from .robotiq_driver_cfg import RobotiqDriverCfg


class RobotiqDriver(ArticulationDrive):
    def __init__(self, cfg: RobotiqDriverCfg, data_indices: slice = slice(None)):
        self.device = torch.device("cpu")
        self.cfg = cfg
        # self.work_space_limit = cfg.work_space_limit
        self.data_idx = data_indices
        self.num_joint = len(self.ordered_joint_names)
        self.current_pos = torch.zeros(1, self.num_joint, device=self.device)
        self.current_vel = torch.zeros(1, self.num_joint, device=self.device)
        self.current_eff = torch.zeros(1, self.num_joint, device=self.device)

    @property
    def ordered_joint_names(self):
        return (
            [
                "finger_joint",
                "right_outer_knuckle_joint",
                # "left_outer_finger_joint", # urdf does not have this joint
                "right_outer_finger_joint",
                "left_inner_finger_joint",
                "right_inner_finger_joint",
                "left_inner_finger_knuckle_joint",
                "right_inner_finger_knuckle_joint",
            ],
        )

    def _prepare(self):
        # Initialize urx connection
        pass

    def write_dof_targets(self, pos_target: torch.Tensor, vel_target: torch.Tensor, eff_target: torch.Tensor):
        # Non-blocking motion
        pass

    def read_dof_states(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Blocking call to get_joint_states, storing the data in local torch Tensors."""
        return self.current_pos, self.current_vel, self.current_eff

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
