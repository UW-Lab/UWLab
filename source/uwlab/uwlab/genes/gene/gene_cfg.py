# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from uwlab.genes.gene import gene


@configclass
class GeneOperatorBaseCfg:
    class_type: type = MISSING  # type: ignore
    # The group flag indicates which group this gene belongs to.
    group: str = "any"
    phase: list[Literal["init", "mutate", "breed"]] = MISSING  # type: ignore
    # The function to be used for mutations, to be defined externally.
    mutation_func: Callable = MISSING  # type: ignore
    # The arguments that supplies mutation_func.
    mutation_args: tuple[any, ...] = MISSING  # type: ignore

    mutation_rate: float = 1


@configclass
class TupleGeneBaseCfg(GeneOperatorBaseCfg):
    element_length: int = MISSING  # type: ignore

    element_idx: int = MISSING  # type: ignore

    # Defines the type of tuple operation:
    # 'descend': Each subsequent tuple element must be less than the previous one.
    # 'ascend': Each subsequent tuple element must be greater than the previous one.
    # 'equal': All elements in the tuple are the same.
    # 'symmetric': The tuple consists of exactly two values, 0th value being the negative of the 1st value.
    tuple_type: Literal["descend", "ascend", "equal", "symmetric"] = MISSING  # type: ignore


@configclass
class TerrainGeneCfg(GeneOperatorBaseCfg):
    class_type: type = gene.TerrainGeneOperator


@configclass
class FloatGeneCfg(GeneOperatorBaseCfg):
    class_type: type = gene.FloatGeneOperator

    fmin: float = -float("inf")

    fmax: float = float("inf")


@configclass
class IntGeneCfg(FloatGeneCfg):
    class_type: type = gene.IntGeneOperator


@configclass
class StrGeneCfg(GeneOperatorBaseCfg):
    str_list: list[str] = MISSING  # type: ignore


@configclass
class StrTupleGeneCfg(TupleGeneBaseCfg):
    # Not yet implemented.
    pass


# Define the configuration class for floating-point tuple gene operations.
@configclass
class FloatTupleGeneCfg(TupleGeneBaseCfg):
    class_type: type = gene.FloatTupleGeneOperator

    # Minimum possible values for each element in the tuple, defaulting to negative infinity.
    fmin: tuple[float, ...] = tuple(-float("inf") for _ in range(2))

    # Maximum possible values for each element in the tuple, defaulting to positive infinity.
    fmax: tuple[float, ...] = tuple(float("inf") for _ in range(2))


@configclass
class IntTupleGeneCfg(FloatTupleGeneCfg):
    # not_yet_implemented
    pass
