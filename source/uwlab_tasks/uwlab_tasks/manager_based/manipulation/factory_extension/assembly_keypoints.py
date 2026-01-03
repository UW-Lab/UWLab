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


@configclass
class KeyPointsNistBoard:
    bolt_m16: Offset = Offset(pos=(0.145, -0.1495, -0.01))
    hole_8mm: Offset = Offset(pos=(-0.0895, 0.0, 0.01))
    gear_base: Offset = Offset(pos=(0.145, 0.021, 0.0092), quat=(0.70711, 0.0, 0.0, -0.70711))
    small_gear: Offset = Offset(pos=(0.145, 0.021, 0.0092), quat=(0.70711, 0.0, 0.0, -0.70711))
    medium_gear: Offset = Offset(pos=(0.145, 0.021, 0.0092), quat=(0.70711, 0.0, 0.0, -0.70711))
    large_gear: Offset = Offset(pos=(0.145, 0.021, 0.0092), quat=(0.70711, 0.0, 0.0, -0.70711))


@configclass
class KeyPointsBoltM16:
    bolt_tip_offset: Offset = Offset(pos=(0, 0, 0.035))
    bolt_base_offset: Offset = Offset(pos=(0, 0, 0.01))
    screwed_nut_offset: Offset = Offset(pos=(0, 0, 0.032))


@configclass
class KeyPointsNutM16:
    center_axis_bottom: Offset = Offset(pos=(0.0, 0.0, 0.01))
    center_axis_middle: Offset = Offset(pos=(0.0, 0.0, 0.0165))
    center_axis_top: Offset = Offset(pos=(0.0, 0.0, 0.023))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.01), quat=(0.70711, 0.0, 0.0, -0.70711))
    grasp_diameter: float = 0.024


@configclass
class KeyPointsGearBase:
    small_gear_tip_offset: Offset = Offset(pos=(0.05075, 0.0, 0.025))
    small_gear_assembled_bottom_offset = Offset(pos=(0.05075, 0.0, 0.005))
    medium_gear_tip_offset: Offset = Offset(pos=(0.02025, 0.0, 0.025))
    medium_gear_assembled_bottom_offset = Offset(pos=(0.02025, 0.0, 0.005))
    large_gear_tip_offset: Offset = Offset(pos=(-0.03025, 0.0, 0.025))
    large_gear_assembled_bottom_offset = Offset(pos=(-0.03025, 0.0, 0.005))


@configclass
class KeyPointsSmallGear:
    center_axis_bottom: Offset = Offset(pos=(0.05075, 0.0, 0.005))
    center_axis_top: Offset = Offset(pos=(0.05075, 0.0, 0.03))
    grasp_point: Offset = Offset(pos=(0.05075, 0.0, 0.022))
    grasp_diameter: float = 0.03


@configclass
class KeyPointsMediumGear:
    center_axis_bottom: Offset = Offset(pos=(0.02025, 0.0, 0.005))
    center_axis_top: Offset = Offset(pos=(0.02025, 0.0, 0.03))
    grasp_point: Offset = Offset(pos=(0.02025, 0.0, 0.022), quat=(0.70711, 0.0, 0.0, -0.70711))
    grasp_diameter: float = 0.03


@configclass
class KeyPointsLargeGear:
    center_axis_bottom: Offset = Offset(pos=(-3.025e-2, 0.0, 0.005))
    center_axis_top: Offset = Offset(pos=(-3.025e-2, 0.0, 0.03))
    grasp_point: Offset = Offset(pos=(-3.025e-2, 0.0, 0.022))
    grasp_diameter: float = 0.03


@configclass
class KeyPointsHole8MM:
    hole_tip_offset: Offset = Offset(pos=(0, 0, 0.025))
    inserted_peg_base_offset: Offset = Offset(pos=(0, 0, 0.0))


@configclass
class KeyPointsPeg8MM:
    center_axis_bottom: Offset = Offset(pos=(0.0, 0.0, 0.0))
    center_axis_middle: Offset = Offset(pos=(0.0, 0.0, 0.025))
    center_axis_top: Offset = Offset(pos=(0.0, 0.0, 0.05))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.035))
    grasp_diameter: float = 0.007986


@configclass
class KeyPointPandaHand:
    object_grasped_point: Offset = Offset(pos=(0.0, 0.0, 0.107), quat=(0.0, 0.0, 1.0, 0.0))


KEYPOINTS_NISTBOARD = KeyPointsNistBoard()
KEYPOINTS_BOLTM16 = KeyPointsBoltM16()
KEYPOINTS_NUTM16 = KeyPointsNutM16()
KEYPOINTS_GEARBASE = KeyPointsGearBase()
KEYPOINTS_SMALLGEAR = KeyPointsSmallGear()
KEYPOINTS_MEDIUMGEAR = KeyPointsMediumGear()
KEYPOINTS_LARGEGEAR = KeyPointsLargeGear()
KEYPOINTS_HOLE8MM = KeyPointsHole8MM()
KEYPOINTS_PEG8MM = KeyPointsPeg8MM()
KEYPOINTS_PANDAHAND = KeyPointPandaHand()
