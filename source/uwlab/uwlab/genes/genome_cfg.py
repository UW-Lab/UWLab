# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.utils import configclass

from .genome import Genome


@configclass
class GenomeCfg:
    class_type: Callable[..., Genome] = Genome

    genomic_mutation_profile: dict = MISSING  # type: ignore

    genomic_constraint_profile: dict = MISSING  # type: ignore

    seed: int = 32
