# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch


# MUTATION FUNCTIONS
def add_constant(rng: np.random.Generator, val, mutation_rate: float, constant):
    sign = -1 if rng.random() < 0.5 else 1
    new_val = val + sign * constant * mutation_rate
    return new_val


def add_fraction(rng: np.random.Generator, val, mutation_rate: float, fraction):
    sign = -1 if rng.random() < 0.5 else 1
    val_offset = sign * fraction * val
    new_val = val + val_offset * mutation_rate
    return new_val


def random_int(rng: np.random.Generator, val, mutation_rate: float, imin, imax):
    new_val = rng.integers(imin, imax)
    return new_val


def random_float(rng: np.random.Generator, val, mutation_rate: float, fmin, fmax):
    new_val = rng.random() * (fmax - fmin) + fmin
    return new_val


def random_selection(rng: np.random.Generator, val, mutation_rate: float, selection_list: list):
    rand_int = rng.integers(0, len(selection_list))
    return selection_list[rand_int]


def random_dict(rng: np.random.Generator, val, mutation_rate: float, dict: dict):
    # select a random pair of key, value from the dictionary
    # return as a dictionary
    key = random_selection(rng, val, mutation_rate, list(dict.keys()))
    value = dict[key]
    return {key: value}


def mutate_terrain_cfg(rng: np.random.Generator, val, mutation_rate, cfg):
    key = random_selection(rng, val, mutation_rate, list(cfg.sub_terrains.keys()))
    value = cfg.sub_terrains[key]
    sub_terrain = {key: value}
    cfg.sub_terrains = sub_terrain
    return cfg


# BREEDING FUNCTIONS


def breed_terrain_cfg(this_val, other_val):
    num_sub_terrains = len(this_val.sub_terrains) + len(other_val.sub_terrains)
    width = np.ceil(np.sqrt(num_sub_terrains))
    this_val.num_cols = width
    this_val.num_rows = width
    this_val.sub_terrains.update(other_val.sub_terrains)


def value_distribution(
    values: list[float],
    distribute_to_n_values: int,
    value_to_distribute: float | None = None,
    equal_distribution: bool = False,
) -> list[float]:
    """Redistributes the total sum of values to the top n values based on their initial proportion."""
    if distribute_to_n_values <= 0 or distribute_to_n_values > len(values):
        raise ValueError("distribute_to_n_values must be greater than 0 and less than or equal to the length of values")

    # Get indices of the top 'n' values
    top_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:distribute_to_n_values]
    top_sum = sum(values[i] for i in top_indices)
    if value_to_distribute:
        total_value_to_distribute = value_to_distribute
    else:
        total_value_to_distribute = sum(values)

    # Calculate proportions and distribute the total sum accordingly
    proportion = np.zeros(len(values))
    if equal_distribution:
        proportion[top_indices] = np.array([1 / distribute_to_n_values for _ in top_indices])
    else:
        proportion[top_indices] = np.array([values[i] / top_sum for i in top_indices])
    output = proportion * total_value_to_distribute
    return output.tolist()


def probability_distribution(vals: list[float], distribute_to_n_values: int) -> list[float]:
    """Converts redistributed values into a valid probability distribution using softmax and returns it as a list of floats."""
    # First, redistribute the values
    redistributed_vals = value_distribution(vals, distribute_to_n_values)

    # Convert to a torch tensor
    logits = torch.tensor(redistributed_vals, dtype=torch.float)

    # Apply softmax to convert logits into probabilities
    probabilities = torch.softmax(logits, dim=0)

    # Convert the tensor back to a list of floats
    probabilities_list = probabilities.tolist()

    return probabilities_list
