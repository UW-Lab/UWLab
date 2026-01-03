# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject


@configclass
class Offset:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @property
    def pose(self) -> tuple[float, float, float, float, float, float, float]:
        return self.pos + self.quat

    def apply(self, root: RigidObject | Articulation) -> tuple[torch.Tensor, torch.Tensor]:
        data = root.data.root_pos_w
        pos_w, quat_w = math_utils.combine_frame_transforms(
            root.data.root_pos_w,
            root.data.root_quat_w,
            torch.tensor(self.pos).to(data.device).repeat(data.shape[0], 1),
            torch.tensor(self.quat).to(data.device).repeat(data.shape[0], 1),
        )
        return pos_w, quat_w

    def combine(self, pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_data = pos.shape[0]
        device = pos.device
        return math_utils.combine_frame_transforms(
            pos,
            quat,
            torch.tensor(self.pos).to(device).repeat(num_data, 1),
            torch.tensor(self.quat).to(device).repeat(num_data, 1),
        )

    def subtract(self, pos_w: torch.Tensor, quat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        offset_pos = torch.tensor(self.pos).to(pos_w.device).repeat(pos_w.shape[0], 1)
        offset_quat = torch.tensor(self.quat).to(pos_w.device).repeat(pos_w.shape[0], 1)
        inv_offset_pos = -math_utils.quat_apply(math_utils.quat_inv(offset_quat), offset_pos)
        inv_offset_quat = math_utils.quat_inv(offset_quat)
        return math_utils.combine_frame_transforms(pos_w, quat_w, inv_offset_pos, inv_offset_quat)
