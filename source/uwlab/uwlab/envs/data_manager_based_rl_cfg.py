# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass


@configclass
class DataManagerBasedRLEnvCfg(ManagerBasedRLEnvCfg):
    data: object | None = None
