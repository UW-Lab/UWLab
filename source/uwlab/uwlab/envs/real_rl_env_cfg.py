# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from uwlab.scene import SceneContextCfg

from .real_rl_env import RealRLEnv


@configclass
class RealRLEnvCfg(ManagerBasedRLEnvCfg):
    class_type = RealRLEnv

    device = "cpu"

    scene: SceneContextCfg = MISSING
