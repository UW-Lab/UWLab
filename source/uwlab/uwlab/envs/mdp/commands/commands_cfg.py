# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .categorical_command import CategoricalCommand


@configclass
class CategoricalCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = CategoricalCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    num_category: int = MISSING
