# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
from torch import nn

from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter, _TorchPolicyExporter


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporterExtended(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporterExtended(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _StateDependentPolicyMixin(nn.Module):
    """Mixin class to handle state-dependent policy logic."""

    def _setup_state_dependent_policy(self, policy):
        """Setup state-dependent policy components."""
        self.actor_features = self.actor[:-1]  # type: ignore
        self.actor_final = self.actor[-1]  # type: ignore

        self.register_buffer("log_std", policy.log_std.clone())
        self.epsilon = 1e-6

    def _setup_regular_policy(self, policy):
        """Setup regular policy components."""
        self.actor_features = self.actor[:-1]  # type: ignore
        self.actor_final = self.actor[-1]  # type: ignore

        if hasattr(policy, "std"):
            self.register_buffer("std", policy.std.clone())
        if hasattr(policy, "log_std"):
            self.register_buffer("log_std", policy.log_std.clone())
        if hasattr(policy, "noise_std_type"):
            self.noise_std_type = policy.noise_std_type
        else:
            self.noise_std_type = "scalar"

        # For GSDE, ensure epsilon is set
        if self.noise_std_type == "gsde":
            self.epsilon = 1e-6

    def _ensure_compatibility_attributes(self, policy):
        """Ensure all attributes exist for TorchScript compatibility."""
        if not hasattr(self, "std"):
            if hasattr(policy, "std"):
                self.register_buffer("std", policy.std.clone())
            else:
                # Create a default std tensor
                default_std = torch.ones(policy.num_actions if hasattr(policy, "num_actions") else 1)
                self.register_buffer("std", default_std)

        if not hasattr(self, "log_std"):
            if hasattr(policy, "log_std"):
                self.register_buffer("log_std", policy.log_std.clone())
            else:
                # Create a default log_std tensor
                default_log_std = torch.zeros(policy.num_actions if hasattr(policy, "num_actions") else 1)
                self.register_buffer("log_std", default_log_std)

        if not hasattr(self, "epsilon"):
            self.epsilon = 1e-6

        if not hasattr(self, "noise_std_type"):
            if hasattr(policy, "noise_std_type"):
                self.noise_std_type = policy.noise_std_type
            else:
                self.noise_std_type = "scalar"  # Default fallback

        # Ensure epsilon is set for GSDE
        if self.noise_std_type == "gsde" and not hasattr(self, "epsilon"):
            self.epsilon = 1e-6

    def _compute_distribution(self, observations):
        """Compute mean and std for distribution."""
        if self.is_state_dependent.item():  # type: ignore
            # Use the separated layers
            features = self.actor_features(observations)  # type: ignore
            mean = self.actor_final(features)  # type: ignore

            # Compute variance using exploration matrices and torch.mm
            variance = torch.mm(features**2, torch.exp(self.log_std) ** 2)  # type: ignore
            std = torch.sqrt(variance + self.epsilon)

            return mean, std
        else:
            # Regular ActorCritic logic
            mean = self.actor(observations)  # type: ignore

            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)  # type: ignore
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)  # type: ignore
            elif self.noise_std_type == "gsde":
                # GSDE: log_std is a matrix (hidden_dim, num_actions)
                # Compute features from actor[:-1] (all layers except last)
                features = self.actor_features(observations)  # type: ignore
                # Compute variance: variance = torch.mm(features**2, exp(log_std)**2)
                # features shape: (batch, hidden_dim), log_std shape: (hidden_dim, num_actions)
                variance = torch.mm(features**2, torch.exp(self.log_std) ** 2)  # type: ignore
                std = torch.sqrt(variance + self.epsilon)
            else:
                std = torch.ones_like(mean)

            return mean, std


class _TorchPolicyExporterExtended(_TorchPolicyExporter, _StateDependentPolicyMixin):
    def __init__(self, policy, normalizer=None):
        super().__init__(policy, normalizer)

        # Detect policy type
        is_state_dependent = hasattr(policy, "use_state_dependent_noise") and policy.use_state_dependent_noise
        self.register_buffer("is_state_dependent", torch.tensor(is_state_dependent, dtype=torch.bool))

        if is_state_dependent:
            self._setup_state_dependent_policy(policy)
        else:
            self._setup_regular_policy(policy)

        # Ensure all attributes exist for TorchScript compatibility
        self._ensure_compatibility_attributes(policy)

    @torch.jit.export
    def compute_distribution(self, x):
        observations = self.normalizer(x)
        return self._compute_distribution(observations)


class _OnnxPolicyExporterExtended(_OnnxPolicyExporter, _StateDependentPolicyMixin):
    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__(policy, normalizer, verbose)

        is_state_dependent = hasattr(policy, "use_state_dependent_noise") and policy.use_state_dependent_noise
        self.register_buffer("is_state_dependent", torch.tensor(is_state_dependent, dtype=torch.bool))

        if is_state_dependent:
            self._setup_state_dependent_policy(policy)
        else:
            self._setup_regular_policy(policy)

    @torch.jit.export
    def compute_distribution(self, x):
        observations = self.normalizer(x)
        return self._compute_distribution(observations)
