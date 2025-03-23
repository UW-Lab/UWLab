# Copyright (c) 2024-2025, The UW Lab Project Developers.
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
