# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation-layout migration for OmniReset state-policy checkpoints."""

from __future__ import annotations

import copy
import torch
from typing import Any

LEGACY_LAYOUT = (
    ("insertive_asset_in_receptive_asset_frame", 6),
    ("prev_actions", 7),
    ("joint_pos", 12),
    ("end_effector_pose", 6),
    ("insertive_asset_pose", 6),
    ("receptive_asset_pose", 6),
)
EA_LAYOUT = LEGACY_LAYOUT[1:] + LEGACY_LAYOUT[:1]
PERMUTATION = tuple(range(6, 43)) + tuple(range(6))
LAYOUT_KEY = "uwlab_observation_layout"


def convert_omnireset_checkpoint(
    checkpoint: dict[str, Any],
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source_sha256: str,
) -> dict[str, Any]:
    """Convert a legacy 43-feature checkpoint to EA declaration order without retraining.

    Args:
        checkpoint: Loaded actor, critic, normalizer and Adam checkpoint state.
        actor: Actor constructed with the checkpoint's original model configuration.
        critic: Critic constructed with the checkpoint's original model configuration.
        optimizer: Actor-then-critic Adam optimizer with checkpoint state already loaded.
        source_sha256: SHA-256 of the original checkpoint file.

    Returns:
        A separate checkpoint with reordered input weights, statistics, optimizer moments and layout metadata.
    """
    if LAYOUT_KEY in checkpoint:
        raise ValueError("Checkpoint already has observation-layout metadata; refusing double conversion.")
    if len(source_sha256) != 64:
        raise ValueError("Expected the original checkpoint's SHA-256.")
    bytes.fromhex(source_sha256)
    names = {
        id(parameter): (role, name)
        for role, model in (("actor", actor), ("critic", critic))
        for name, parameter in model.named_parameters()
    }
    saved_groups = checkpoint["optimizer_state_dict"]["param_groups"]
    if len(optimizer.param_groups) != len(saved_groups):
        raise ValueError("Optimizer parameter groups do not match the checkpoint.")
    input_ids = {}
    for group, saved_group in zip(optimizer.param_groups, saved_groups):
        if len(group["params"]) != len(saved_group["params"]):
            raise ValueError("Optimizer parameter counts do not match the checkpoint.")
        for parameter, parameter_id in zip(group["params"], saved_group["params"]):
            role, name = names[id(parameter)]
            slot = checkpoint["optimizer_state_dict"]["state"][parameter_id]
            if slot["exp_avg"].shape != parameter.shape or slot["exp_avg_sq"].shape != parameter.shape:
                raise ValueError(f"Optimizer state shape mismatch for {role}.{name}.")
            if name == "mlp.0.weight":
                input_ids[f"{role}_state_dict"] = parameter_id
    if len(input_ids) != 2 or len(set(input_ids.values())) != 2:
        raise ValueError("Expected separate actor and critic input weights.")
    output = copy.deepcopy(checkpoint)
    for key, parameter_id in input_ids.items():
        state = output[key]
        weight = state["mlp.0.weight"]
        if weight.ndim != 2 or weight.shape[1] != 43:
            raise ValueError(f"Expected 43 input features for {key}.")
        p = torch.tensor(PERMUTATION, device=weight.device)
        state["mlp.0.weight"] = weight.index_select(1, p).clone()
        for name in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
            value = state[name]
            if value.shape != (1, 43):
                raise ValueError(f"Unexpected normalizer shape for {key}/{name}.")
            state[name] = value.index_select(1, p.to(value.device)).clone()
        slot = output["optimizer_state_dict"]["state"][parameter_id]
        if not {"step", "exp_avg", "exp_avg_sq"} <= slot.keys() or not set(slot) <= {
            "step",
            "exp_avg",
            "exp_avg_sq",
            "max_exp_avg_sq",
        }:
            raise ValueError("Only Adam/AMSGrad optimizer state is supported.")
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name in slot:
                value = slot[name]
                if value.shape != weight.shape:
                    raise ValueError(f"Unexpected optimizer shape for {key}/{name}.")
                slot[name] = value.index_select(1, p.to(value.device)).clone()
    output[LAYOUT_KEY] = {
        "schema_version": 1,
        "source_layout": "omnireset43_beta_annotated_first",
        "target_layout": "omnireset43_ea_declaration_order",
        "source_terms": list(LEGACY_LAYOUT),
        "target_terms": list(EA_LAYOUT),
        "new_index_to_old_index": list(PERMUTATION),
        "source_checkpoint_sha256": source_sha256,
        "optimizer_input_weight_parameter_ids": input_ids,
        "optimizer_state_migrated": True,
    }
    return output
