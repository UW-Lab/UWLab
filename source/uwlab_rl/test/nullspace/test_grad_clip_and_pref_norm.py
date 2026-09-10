# Copyright (c) 2026, Null-space preference critic project.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the gradient-clip coupling fix (NOTES 29).

The bug: rsl_rl clips the whole policy as ONE gradient vector. With a dual critic, a large
preference value loss enters the same norm as the actor's gradient, so clipping scales the actor's
update down -- at beta=0 too, where the preference is meant to have no effect at all. Under
``pref_source=action_rate`` the preference value loss reached 1e5-1e6 against
``max_grad_norm=1.0``, which throttled the actor and pinned the adaptive learning rate at its cap.

``projection.py`` imports nothing but torch, so it is loaded by path (as in ``test_projection.py``)
and none of these tests need Isaac Sim.
"""

from __future__ import annotations

import importlib.util
import io
import pathlib

import pytest
import torch

_PATH = pathlib.Path(__file__).resolve().parents[2] / "uwlab_rl" / "rsl_rl" / "nullspace" / "projection.py"
_spec = importlib.util.spec_from_file_location("nsc_projection_gradclip", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
partition_policy_params = _mod.partition_policy_params
clip_grad_norm_by_group = _mod.clip_grad_norm_by_group

MAX_NORM = 1.0  # the OmniReset agent config's max_grad_norm


class _DualPolicy(torch.nn.Module):
    """Stand-in carrying DualCriticActorCritic's real parameter names."""

    def __init__(self, separate_pref_critic: bool = True) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(4, 2)
        self.log_std = torch.nn.Parameter(torch.zeros(2))  # gSDE noise scale -- belongs to the actor
        self.critic = torch.nn.Linear(4, 1)
        if separate_pref_critic:
            self.critic_pref = torch.nn.Linear(4, 1)


def _fill(params, value: float) -> None:
    for p in params:
        p.grad = torch.full_like(p, value)


def _flat(params) -> torch.Tensor:
    return torch.cat([p.grad.flatten() for p in params]).clone()


def _groups(policy):
    return partition_policy_params(policy.named_parameters())


# -- the bug, reproduced, and the fix ------------------------------------------------------------


def test_global_clip_lets_the_pref_critic_throttle_the_actor():
    """Documents the bug: under one global clip the actor's update shrinks because the
    PREFERENCE CRITIC's gradient is large -- even though the actor's own gradient is small."""
    p = _DualPolicy()
    g = _groups(p)
    _fill(g["actor"] + g["critic"], 0.1)  # actor norm ~0.35: under the limit on its own
    _fill(g["critic_pref"], 1e3)  # the action_rate preference critic's scale
    before = _flat(g["actor"])
    assert before.norm() < MAX_NORM

    torch.nn.utils.clip_grad_norm_(p.parameters(), MAX_NORM)  # upstream behaviour

    assert _flat(g["actor"]).norm() / before.norm() < 1e-2  # crushed by a different module


def test_per_group_clip_leaves_the_actor_untouched_by_the_pref_critic():
    p = _DualPolicy()
    g = _groups(p)
    _fill(g["actor"] + g["critic"], 0.1)
    _fill(g["critic_pref"], 1e3)
    before = _flat(g["actor"])

    norms = clip_grad_norm_by_group(g, MAX_NORM)

    assert torch.allclose(_flat(g["actor"]), before)  # below its own limit -> not scaled at all
    assert _flat(g["critic_pref"]).norm() == pytest.approx(MAX_NORM, rel=1e-4)  # clipped on its own
    assert norms["actor"] == pytest.approx(before.norm().item(), rel=1e-5)  # pre-clip norms reported
    assert norms["critic_pref"] > 1e3


def test_actor_update_is_invariant_to_pref_critic_scale():
    """The property the acceptance run checks at the outcome level: at beta=0 the actor's clipped
    gradient must not depend on how large the preference critic's gradient is. Under the global
    clip it does; under per-group clipping it does not."""
    per_group, global_ = [], []
    for pref_scale in (0.0, 1.0, 1e3, 1e6):
        for mode, out in (("per_group", per_group), ("global", global_)):
            p = _DualPolicy()
            torch.manual_seed(0)
            g = _groups(p)
            _fill(g["actor"] + g["critic"], 0.1)
            _fill(g["critic_pref"], pref_scale)
            if mode == "per_group":
                clip_grad_norm_by_group(g, MAX_NORM)
            else:
                torch.nn.utils.clip_grad_norm_(p.parameters(), MAX_NORM)
            out.append(_flat(g["actor"]))

    assert all(torch.allclose(a, per_group[0]) for a in per_group)
    assert not all(torch.allclose(a, global_[0]) for a in global_)


def test_per_group_still_clips_an_oversized_actor():
    """Per-group clipping must not silently disable clipping for the actor."""
    p = _DualPolicy()
    g = _groups(p)
    _fill(g["actor"], 10.0)
    _fill(g["critic"] + g["critic_pref"], 0.1)

    clip_grad_norm_by_group(g, MAX_NORM)

    assert _flat(g["actor"]).norm() == pytest.approx(MAX_NORM, rel=1e-4)


# -- partition ------------------------------------------------------------------------------------


def test_partition_follows_actor_parameters_and_avoids_the_prefix_trap():
    """``critic`` is a prefix of ``critic_pref``; the preference critic must not land in the task
    critic's group, and ``log_std`` must stay with the actor (as in ``actor_parameters``)."""
    p = _DualPolicy()
    ids = {k: {id(x) for x in v} for k, v in _groups(p).items()}

    assert ids["actor"] == {id(p.actor.weight), id(p.actor.bias), id(p.log_std)}
    assert ids["critic"] == {id(p.critic.weight), id(p.critic.bias)}
    assert ids["critic_pref"] == {id(p.critic_pref.weight), id(p.critic_pref.bias)}


def test_shared_critic_arch_leaves_pref_group_empty_and_is_safe():
    """``critic_arch='shared'`` has no separate preference critic; params without a gradient are
    skipped rather than erroring."""
    p = _DualPolicy(separate_pref_critic=False)
    g = _groups(p)
    assert g["critic_pref"] == []
    _fill(g["critic"], 0.1)  # actor deliberately left with grad=None

    norms = clip_grad_norm_by_group(g, MAX_NORM)

    assert norms["critic_pref"] == 0.0 and norms["actor"] == 0.0 and norms["critic"] > 0.0


# -- preference reward normalisation -------------------------------------------------------------

_networks = pytest.importorskip("rsl_rl.networks")
EDVN = _networks.EmpiricalDiscountedVariationNormalization


def test_pref_reward_normalizer_brings_action_rate_scale_rewards_to_order_one():
    torch.manual_seed(0)
    n = EDVN(shape=1, gamma=0.99)
    n.train()
    for _ in range(300):
        raw = -(torch.rand(4096, 1) * 3e3)  # -||a - a_prev||^2 at the magnitudes that were observed
        out = n(raw)

    assert raw.abs().mean() > 1e3
    assert 1e-3 < out.abs().mean().item() < 10.0


def test_zero_preference_stays_exactly_zero():
    """``pref_source='zero'`` must be untouched: no NaN from a vanishing std, no drift off zero."""
    n = EDVN(shape=1, gamma=0.99)
    n.train()
    for _ in range(200):
        out = n(torch.zeros(4096, 1))

    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))


def test_normalizer_works_under_inference_mode_and_survives_a_checkpoint_round_trip():
    """rsl_rl collects rollouts inside ``torch.inference_mode()`` (on_policy_runner.py:101), so the
    normaliser's buffers are rewritten there. They must still be readable for logging, serialisable
    by ``torch.save``, and loadable into a freshly built policy on resume."""
    n = EDVN(shape=1, gamma=0.99)
    n.train()
    with torch.inference_mode():
        for _ in range(20):
            n(-(torch.rand(512, 1) * 1e3))
    scale = float(n.emp_norm._std)  # read outside inference mode, as update() logging does
    assert scale > 1.0

    buf = io.BytesIO()
    torch.save(n.state_dict(), buf)  # what the runner's save() does
    buf.seek(0)
    fresh = EDVN(shape=1, gamma=0.99)
    fresh.load_state_dict(torch.load(buf))  # what a resumed run does, outside inference mode
    assert float(fresh.emp_norm._std) == pytest.approx(scale)

    fresh.train()
    with torch.inference_mode():
        out = fresh(-(torch.rand(512, 1) * 1e3))
    assert torch.isfinite(out).all()


def test_normalizer_on_the_policy_is_checkpointed_and_adds_no_parameters():
    """Attached to the policy it rides along in ``policy.state_dict()`` -- and, holding only
    buffers, it cannot enter a clip group or the optimizer."""
    p = _DualPolicy()
    p.pref_reward_normalizer = EDVN(shape=1, gamma=0.99)

    assert any(k.startswith("pref_reward_normalizer.emp_norm.") for k in p.state_dict())
    assert not any(n.startswith("pref_reward_normalizer") for n, _ in p.named_parameters())
    grouped = {id(x) for v in _groups(p).values() for x in v}
    assert grouped == {id(x) for x in p.parameters()}  # every parameter grouped exactly once
