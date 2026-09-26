# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import torch
from itertools import chain

import pytest
from rsl_rl.modules.normalization import EmpiricalNormalization

from uwlab_rl.rsl_rl.checkpoint_layout import LAYOUT_KEY, PERMUTATION, convert_omnireset_checkpoint


class Model(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.obs_normalizer = EmpiricalNormalization(43)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(43, 8), torch.nn.ELU(), torch.nn.Linear(8, output_dim))

    def forward(self, obs):
        return self.mlp(self.obs_normalizer(obs))


@pytest.mark.parametrize("amsgrad", [False, True])
def test_checkpoint_layout_preserves_outputs_and_optimizer(amsgrad):
    torch.manual_seed(42)
    actor, critic = Model(14), Model(1)
    optimizer = torch.optim.Adam(chain(actor.parameters(), critic.parameters()), amsgrad=amsgrad)
    observations = torch.randn(16, 43)
    actor.obs_normalizer.update(observations)
    critic.obs_normalizer.update(observations)
    loss = actor(observations).square().mean() + critic(observations).square().mean()
    loss.backward()
    optimizer.step()
    checkpoint = copy.deepcopy({
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": 19,
        "learning_rate": 0.001,
        "infos": {"source": "fixture"},
        "logger_state": {"tot_timesteps": 128},
    })
    converted = convert_omnireset_checkpoint(checkpoint, actor, critic, optimizer, "0" * 64)
    p = torch.tensor(PERMUTATION)
    inverse = torch.argsort(p)
    for model, key in ((actor, "actor_state_dict"), (critic, "critic_state_dict")):
        restored = copy.deepcopy(model)
        restored.load_state_dict(converted[key], strict=True)
        torch.testing.assert_close(restored(observations[:, p]), model(observations), rtol=1e-5, atol=1e-6)
        for name, original in checkpoint[key].items():
            actual = converted[key][name]
            if name in ("mlp.0.weight", "obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
                actual = actual[:, inverse]
            assert torch.equal(actual, original)
        assert torch.equal(checkpoint[key]["mlp.0.weight"], model.state_dict()["mlp.0.weight"])
    input_ids = set(converted[LAYOUT_KEY]["optimizer_input_weight_parameter_ids"].values())
    for identifier, slots in checkpoint["optimizer_state_dict"]["state"].items():
        for name, original in slots.items():
            actual = converted["optimizer_state_dict"]["state"][identifier][name]
            if identifier in input_ids and name != "step":
                actual = actual[:, inverse]
            assert torch.equal(actual, original)
    assert converted["optimizer_state_dict"]["param_groups"] == checkpoint["optimizer_state_dict"]["param_groups"]
    for name in ("iter", "learning_rate", "infos", "logger_state"):
        assert converted[name] == checkpoint[name]
    with pytest.raises(ValueError, match="double conversion"):
        convert_omnireset_checkpoint(converted, actor, critic, optimizer, "0" * 64)
