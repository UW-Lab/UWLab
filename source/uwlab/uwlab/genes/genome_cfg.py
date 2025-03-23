# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

from isaaclab.utils import configclass

from .genome import Genome


@configclass
class GenomeCfg:
    class_type: Callable[..., Genome] = Genome

    genomic_mutation_profile: dict = MISSING  # type: ignore

    genomic_constraint_profile: dict = MISSING  # type: ignore

    seed: int = 32
