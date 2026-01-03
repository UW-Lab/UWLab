# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

SPOT_JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(
    asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True
)

ARM_DEFAULT_JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(
    asset_name="arm", joint_names=[".*"], scale=0.2, use_default_offset=True
)
