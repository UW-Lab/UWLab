# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uwlab.terrains.terrain_generator_cfg import MultiOriginTerrainGeneratorCfg

    from .gene_cfg import FloatGeneCfg, FloatTupleGeneCfg, GeneOperatorBaseCfg, TerrainGeneCfg


class GeneOperatorBase:
    cfg: GeneOperatorBaseCfg

    def __init__(self, retrive_args, cfg: GeneOperatorBaseCfg, rng: np.random.Generator):
        self.retrive_args = retrive_args
        self.mutation_args = cfg.mutation_args
        self.mutation_func = cfg.mutation_func
        self.group = cfg.group
        self.rng = rng

    def get(self, source):
        raise NotImplementedError

    def set(self, val):
        raise NotImplementedError

    def mutate(self, source):
        raise NotImplementedError

    def breed(self, this_source, other_source):
        raise NotImplementedError

    def traverse_operations(self, src, operation_list, arg_list):
        for operation, arg in zip(operation_list, arg_list):
            src = operation(src, arg)
        return src

    def _set_attr(self, target, key, val):
        if isinstance(target, dict):
            target[key] = val
        else:
            setattr(target, key, val)


class FloatGeneOperator(GeneOperatorBase):
    cfg: FloatGeneCfg

    def __init__(self, retrive_args, cfg: FloatGeneCfg, rng: np.random.Generator):
        super().__init__(retrive_args, cfg, rng)
        self.fmin = cfg.fmin
        self.fmax = cfg.fmax
        self.mutation_rate = cfg.mutation_rate

    def get(self, source):
        return self.traverse_operations(source, *self.retrive_args[:2])

    def set(self, source, value):
        if value < self.fmin or value > self.fmax:
            raise ValueError("you are trying to set a value out of bound")
        self._set_func(source, value, *self.retrive_args)

    def mutate(self, source):
        val = self.get(source)
        new_val = self.mutation_func(self.rng, val, self.mutation_rate, *self.mutation_args)
        new_val = np.clip(new_val, self.fmin, self.fmax).item()
        self.set(source, new_val)

    def breed(self, this_source, other_source):
        this_val = self.get(this_source)
        other_val = self.get(other_source)
        new_val = (this_val + other_val) / 2
        self.set(this_source, new_val)

    def _set_func(self, src_env, v, ops, args):
        self._set_attr(self.traverse_operations(src_env, ops[:-1], args[:-1]), args[-1], float(v))


class IntGeneOperator(FloatGeneOperator):
    def set(self, source, value):
        if value < self.fmin or value > self.fmax:
            raise ValueError("you are trying to set a value out of bound")
        self._set_func(source, value, *self.retrive_args)

    def _set_func(self, src_env, v, ops, args):
        self._set_attr(self.traverse_operations(src_env, ops[:-1], args[:-1]), args[-1], int(v))


class FloatTupleGeneOperator(GeneOperatorBase):
    cfg: FloatTupleGeneCfg

    def __init__(self, retrive_args, cfg: FloatTupleGeneCfg, rng: np.random.Generator):
        super().__init__(retrive_args, cfg, rng)
        self.mutation_rate = cfg.mutation_rate
        self.fmin = cfg.fmin
        self.fmax = cfg.fmax
        self.element_length = cfg.element_length
        self.element_idx: int = cfg.element_idx

    def get(self, source) -> float:
        return self.traverse_operations(source, *self.retrive_args[:2])[self.element_idx]

    def set(self, source, value):
        self._set_float_tuple_func(source, value, *self.retrive_args)

    def mutate(self, source):
        val = self.get(source)
        new_val = self.mutation_func(self.rng, val, self.mutation_rate, *self.mutation_args)
        new_val = np.clip(new_val, self.fmin[self.element_idx], self.fmax[self.element_idx]).item()
        self.set(source, new_val)

    def breed(self, this_source, other_source):
        val = self.get(this_source)
        other_val = self.get(other_source)
        self.set(this_source, (val + other_val) / 2)

    def _set_float_tuple_func(self, src_env, v, ops, args):
        # Retrieve the original tuple and convert it to a list to make it mutable
        val_list = list(getattr(self.traverse_operations(src_env, ops[:-1], args[:-1]), args[-1]))
        # Modify the value at the specified index
        val_list[self.element_idx] = v
        # Convert the list back to a tuple
        val_tuple = tuple(val_list)
        # Set the modified tuple back to the source environment
        self._set_attr(self.traverse_operations(src_env, ops[:-1], args[:-1]), args[-1], val_tuple)


class TerrainGeneOperator(GeneOperatorBase):
    cfg: TerrainGeneCfg

    def __init__(self, retrive_args, cfg: TerrainGeneCfg, rng: np.random.Generator):
        super().__init__(retrive_args, cfg, rng)
        self.mutation_rate = cfg.mutation_rate

    def get(self, source):
        return self.traverse_operations(source, *self.retrive_args[:2])

    def set(self, source, value):
        self._set_func(source, value, *self.retrive_args)

    def mutate(self, source):
        val = self.get(source)
        new_val = self.mutation_func(self.rng, val, self.mutation_rate, *self.mutation_args)
        self.set(source, new_val)

    def breed(self, this_source, other_source):
        this_val: MultiOriginTerrainGeneratorCfg = self.get(this_source)
        other_val: MultiOriginTerrainGeneratorCfg = self.get(other_source)
        num_sub_terrains = len(this_val.sub_terrains) + len(other_val.sub_terrains)
        width = np.ceil(np.sqrt(num_sub_terrains)).item()
        this_val.num_cols = int(width)
        this_val.num_rows = int(width)
        this_val.sub_terrains.update(other_val.sub_terrains)
        self.set(this_source, this_val)

    def _set_func(self, src_env, v, ops, args):
        self._set_attr(self.traverse_operations(src_env, ops[:-1], args[:-1]), args[-1], v)
