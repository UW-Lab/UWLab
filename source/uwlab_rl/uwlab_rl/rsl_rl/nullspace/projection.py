# Copyright (c) 2026, Null-space preference critic project.
# SPDX-License-Identifier: BSD-3-Clause

"""Null-space projection of a preference gradient against a task gradient.

The method in one line: move along the task gradient as usual, and add only the component of the
preference gradient that is orthogonal to it, so the preference term produces no *first-order*
change in task value.

    ĝ_task = g_task / ‖g_task‖
    g      = g_task + β · (I − ĝ_task ĝ_taskᵀ) · g_pref

Everything here operates on flat vectors so it is independent of the parameter layout, and is
pure/side-effect-free so it can be unit-tested without a simulator.
"""

from __future__ import annotations

import torch

# Floor on ‖g_task‖ used when normalizing. The projector degenerates as g_task → 0 (exactly where
# the method matters most, once task success has converged), so this is a real numerical guard,
# not decoration -- see the "known costs" section of the design notes.
DEFAULT_EPS = 1e-12


def flatten_grads(grads: list[torch.Tensor | None], params: list[torch.Tensor]) -> torch.Tensor:
    """Flatten a list of per-parameter grads into one vector, treating None as zeros.

    ``torch.autograd.grad(..., allow_unused=True)`` returns None for parameters that did not
    participate in the graph (e.g. gSDE's ``log_std`` when the surrogate does not touch it).
    """
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)
    ])


def unflatten_to(vec: torch.Tensor, params: list[torch.Tensor]) -> list[torch.Tensor]:
    """Split a flat vector back into tensors shaped like ``params``."""
    out, i = [], 0
    for p in params:
        n = p.numel()
        out.append(vec[i : i + n].view_as(p))
        i += n
    return out


def project_nullspace(
    g_task: torch.Tensor,
    g_pref: torch.Tensor,
    beta: float,
    eps: float = DEFAULT_EPS,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine task and preference gradients with the preference term projected orthogonally.

    Args:
        g_task: flat task gradient.
        g_pref: flat preference gradient.
        beta: preference step budget. ``beta == 0`` returns ``g_task`` exactly (bitwise), which is
            what makes the β=0 sanity run a true no-op.
        eps: floor on ‖g_task‖.

    Returns:
        (combined gradient, diagnostics). Diagnostics are cheap scalars worth logging every
        update: they are how you tell "preference is being squeezed out" from "preference is
        driving the update", and they are the early-warning signal for reward hacking.
    """
    task_norm = torch.linalg.vector_norm(g_task)
    pref_norm = torch.linalg.vector_norm(g_pref)

    # cosine similarity before projection: +1 means preference already agrees with the task
    # (projection removes almost everything), 0 means it is already orthogonal (projection is a
    # no-op and preference is "free"), -1 means it directly opposes success.
    denom = task_norm * pref_norm
    cos_before = (torch.dot(g_task, g_pref) / denom).item() if denom > eps else 0.0

    if beta == 0.0:
        return g_task, {
            "g_task_norm": task_norm.item(),
            "g_pref_norm": pref_norm.item(),
            "cos_before": cos_before,
            "pref_removed_frac": 0.0,
            "pref_contrib_ratio": 0.0,
        }

    # Normalize the task direction, flooring the denominator.
    g_hat = g_task / task_norm.clamp_min(eps)
    # Remove the component of g_pref along g_hat.
    g_pref_perp = g_pref - torch.dot(g_hat, g_pref) * g_hat

    perp_norm = torch.linalg.vector_norm(g_pref_perp)
    # Fraction of the preference gradient the projection discarded. → 1 means preference is
    # (anti)parallel to the task and the null space affords it nothing.
    removed = (1.0 - (perp_norm / pref_norm).item()) if pref_norm > eps else 0.0

    g = g_task + beta * g_pref_perp

    return g, {
        "g_task_norm": task_norm.item(),
        "g_pref_norm": pref_norm.item(),
        "cos_before": cos_before,
        "pref_removed_frac": removed,
        # How much of the final step the preference term accounts for. This is the quantity that
        # self-schedules: small early (task gradient dominates), growing as g_task → 0.
        "pref_contrib_ratio": (beta * perp_norm / task_norm.clamp_min(eps)).item(),
    }
