# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab_assets.robots.anymal as anymal
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

from ... import balance_beams_env, stepping_beams_env, stepping_stones_env


@configclass
class AnymalDActionsCfg:
    actions = JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class AnyMalDEnvMixin:
    actions: AnymalDActionsCfg = AnymalDActionsCfg()

    def __post_init__(self: stepping_stones_env.SteppingStoneLocomotionEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()  # type: ignore
        self.scene.robot = anymal.ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore


@configclass
class SteppingStoneAnymalDEnvCfg(AnyMalDEnvMixin, stepping_stones_env.SteppingStoneLocomotionEnvCfg):
    pass


@configclass
class BalanceBeamsAnymalDEnvCfg(AnyMalDEnvMixin, balance_beams_env.BalanceBeamsLocomotionEnvCfg):
    pass


@configclass
class SteppingBeamsAnymalDEnvCfg(AnyMalDEnvMixin, stepping_beams_env.SteppingBeamsLocomotionEnvCfg):
    pass
