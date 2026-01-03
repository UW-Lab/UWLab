# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .data_manager import DataTerm


@configclass
class DataTermCfg:
    """Configuration for a data generator term."""

    class_type: type[DataTerm] = MISSING
    """The associated data term class to use.

    The class should inherit from :class:`isaaclab.managers.data_manager.DataTerm`.
    """

    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""
