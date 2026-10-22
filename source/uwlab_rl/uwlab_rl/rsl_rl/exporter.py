# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""JIT export of an rsl-rl >= 5.0 actor with its output distribution.

rsl-rl's own ``OnPolicyRunner.export_policy_to_jit`` exports the mean action only. Demo collection
(``scripts_v2/tools/collect_demos.py``) samples from the expert, so the exported module also needs
``compute_distribution(obs) -> (mean, std)``. Fixed-std Gaussians have their std evaluated once at export
time; heteroscedastic Gaussians (std predicted by the MLP next to the mean) and gSDE compute it per observation
with the same clamping as the distribution, so all of them export exactly.
"""

from __future__ import annotations

import copy
import os
import torch
from torch import nn

from rsl_rl.modules import HeteroscedasticGaussianDistribution


def export_policy_as_jit(actor: nn.Module, path: str, filename: str = "policy.pt") -> None:
    """Export an rsl-rl ``MLPModel`` actor into a TorchScript file with ``forward`` and ``compute_distribution``.

    Args:
        actor: The actor model (``rsl_rl.models.MLPModel``), uncompiled.
        path: The directory to save into.
        filename: The file name. Defaults to "policy.pt".
    """
    os.makedirs(path, exist_ok=True)
    exporter = _TorchPolicyExporter(actor).to("cpu")  # device-neutral artifact; consumers move it as needed
    torch.jit.script(exporter).save(os.path.join(path, filename))


class _TorchPolicyExporter(nn.Module):
    """TorchScript-able snapshot of an ``MLPModel`` actor: normalizer, MLP and output distribution."""

    def __init__(self, actor: nn.Module) -> None:
        super().__init__()
        self.normalizer = copy.deepcopy(actor.obs_normalizer)
        # ``MLP`` is an ``nn.Sequential`` subclass whose ``__init__`` takes positional arguments,
        # so it cannot be sliced; rebuild the split around the last layer explicitly.
        layers = [copy.deepcopy(layer) for layer in actor.mlp]
        self.actor_features = nn.Sequential(*layers[:-1])
        self.actor_final = layers[-1]
        self.epsilon = 1e-6
        self.heteroscedastic = False
        self.gsde = False
        self.log_std = False
        self.std_min, self.std_max = 0.0, 0.0
        self.log_std_min, self.log_std_max = 0.0, 0.0

        dist = actor.distribution
        with torch.no_grad():
            if dist is None:
                self.register_buffer("std", torch.ones(1))
                self.register_buffer("std_matrix", torch.ones(1, 1))
            elif isinstance(dist, HeteroscedasticGaussianDistribution):
                # The MLP outputs [..., 2, num_actions]: mean, then (log-)std, clamped like ``dist.update``.
                self.heteroscedastic = True
                self.log_std = dist.std_type == "log"
                self.std_min, self.std_max = float(dist.std_range[0]), float(dist.std_range[1])
                self.log_std_min, self.log_std_max = float(dist.log_std_range[0]), float(dist.log_std_range[1])
                self.register_buffer("std", torch.ones(1))
                self.register_buffer("std_matrix", torch.ones(1, 1))
            elif hasattr(dist, "_get_std") and getattr(dist, "requires_latent_sde", False):
                # gSDE: marginal std = sqrt(phi(s)^2 @ std_matrix^2), std_matrix is (latent_dim, num_actions)
                # after the distribution's own clamp / full_std / expln handling.
                self.gsde = True
                self.register_buffer("std", torch.ones(1))
                self.register_buffer("std_matrix", dist._get_std().detach().clone())
            else:
                # Gaussian: evaluate the clamped std once from the distribution's parameterization.
                if getattr(dist, "std_type", "scalar") == "log":
                    std = torch.exp(dist.log_std_param.clamp(dist.log_std_range[0], dist.log_std_range[1]))
                else:
                    std = dist.std_param.clamp(dist.std_range[0], dist.std_range[1])
                self.register_buffer("std", std.detach().clone())
                self.register_buffer("std_matrix", torch.ones(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.actor_final(self.actor_features(self.normalizer(x)))
        if self.heteroscedastic:
            return out[..., 0, :]
        return out

    @torch.jit.export
    def compute_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.actor_features(self.normalizer(x))
        out = self.actor_final(features)
        if self.heteroscedastic:
            mean = out[..., 0, :]
            if self.log_std:
                std = torch.exp(out[..., 1, :].clamp(self.log_std_min, self.log_std_max))
            else:
                std = out[..., 1, :].clamp(self.std_min, self.std_max)
        elif self.gsde:
            mean = out
            variance = torch.mm(features**2, self.std_matrix**2)
            std = torch.sqrt(variance + self.epsilon)
        else:
            mean = out
            std = self.std.expand_as(mean)
        return mean, std
