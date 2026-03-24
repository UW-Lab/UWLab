# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from abc import ABC, abstractmethod
from typing import Any


class ObservationHistoryManager(ABC):
    """Abstract base class for managing observation history."""

    def __init__(self, num_envs: int, n_obs_steps: int, device: torch.device):
        self.num_envs = num_envs
        self.n_obs_steps = n_obs_steps
        self.device = device
        self.history = None
        self.needs_init = set()  # Track environments that need initialization

    @abstractmethod
    def initialize(self, processed_obs: dict[str, torch.Tensor]):
        """Initialize the history with the first observation."""
        pass

    @abstractmethod
    def update(self, processed_obs: dict[str, torch.Tensor]):
        """Update history with new observations."""
        pass

    @abstractmethod
    def get_batch(self, env_indices: list[int]) -> dict[str, torch.Tensor]:
        """Get observation batch for specific environments."""
        pass

    @abstractmethod
    def reset_envs(self, env_indices: list[int]):
        """Reset history for specific environments."""
        pass


class LowDimObservationHistory(ObservationHistoryManager):
    """Manages observation history for low-dimensional policies."""

    def initialize(self, processed_obs: dict[str, torch.Tensor]):
        """Initialize history as a single tensor."""
        obs_shape = processed_obs["obs"].shape
        history_shape = (self.num_envs, self.n_obs_steps, obs_shape[-1])
        self.history = torch.zeros(history_shape, device=self.device, dtype=processed_obs["obs"].dtype)

    def update(self, processed_obs: dict[str, torch.Tensor]):
        """Update history by shifting and adding new observations."""
        if self.history is None:
            self.initialize(processed_obs)
        # Handle environments that need initialization after reset
        if self.needs_init:
            for env_idx in list(self.needs_init):
                # Fill entire history with the first observation
                first_obs = processed_obs["obs"][env_idx : env_idx + 1]  # Keep batch dimension
                for step in range(self.n_obs_steps):
                    self.history[env_idx, step] = first_obs[0]
                self.needs_init.remove(env_idx)
        # Update history by shifting and adding new observations
        self.history[:, :-1] = self.history[:, 1:].clone()
        # Add new observation at the end
        self.history[:, -1] = processed_obs["obs"]

    def get_batch(self, env_indices: list[int]) -> dict[str, torch.Tensor]:
        """Get observation batch for specific environments."""
        if self.history is None:
            return {"obs": torch.zeros((len(env_indices), self.n_obs_steps, 0), device=self.device)}

        # Select observations for specific environments
        env_obs = self.history[env_indices]  # Shape: (batch, n_obs_steps, obs_dim)
        return {"obs": env_obs}

    def reset_envs(self, env_indices: list[int]):
        """Reset history for specific environments."""
        for i in env_indices:
            self.needs_init.add(i)


class ImageObservationHistory(ObservationHistoryManager):
    """Manages observation history for image-based policies."""

    def __init__(self, num_envs: int, n_obs_steps: int, device: torch.device):
        super().__init__(num_envs, n_obs_steps, device)
        self.obs_keys = None

    def initialize(self, processed_obs: dict[str, torch.Tensor]):
        """Initialize history as a dictionary of tensors."""
        self.obs_keys = list(processed_obs.keys())
        self.history = {}
        for key in self.obs_keys:
            # Shape: (num_envs, n_obs_steps, ...)
            obs_shape = processed_obs[key].shape
            history_shape = (self.num_envs, self.n_obs_steps) + obs_shape[1:]
            self.history[key] = torch.zeros(history_shape, device=self.device, dtype=processed_obs[key].dtype)

    def update(self, processed_obs: dict[str, torch.Tensor]):
        """Update history by shifting and adding new observations."""
        if self.history is None:
            self.initialize(processed_obs)
        # Handle environments that need initialization after reset
        if self.needs_init and self.obs_keys is not None:
            for env_idx in list(self.needs_init):
                if env_idx < self.num_envs:
                    # Fill entire history with the first observation for each key
                    for key in self.obs_keys:
                        first_obs = processed_obs[key][env_idx : env_idx + 1]  # Keep batch dimension
                        for step in range(self.n_obs_steps):
                            self.history[key][env_idx, step] = first_obs[0]
                    self.needs_init.remove(env_idx)
        # Update history by shifting and adding new observations
        if self.obs_keys is not None:
            for key in self.obs_keys:
                # Shift history: (num_envs, n_obs_steps-1, ...) -> (num_envs, 1:n_obs_steps, ...)
                self.history[key][:, :-1] = self.history[key][:, 1:].clone()
                # Add new observation at the end
                self.history[key][:, -1] = processed_obs[key]

    def get_batch(self, env_indices: list[int]) -> dict[str, torch.Tensor]:
        """Get observation batch for specific environments."""
        if self.history is None or self.obs_keys is None:
            return {}
        obs_batch = {}
        for key in self.obs_keys:
            # Select observations for specific environments and transpose to (batch, time, ...)
            env_obs = self.history[key][env_indices]  # Shape: (batch, n_obs_steps, ...)
            obs_batch[key] = env_obs
        return obs_batch

    def reset_envs(self, env_indices: list[int]):
        """Reset history for specific environments."""
        for i in env_indices:
            self.needs_init.add(i)


class DiffusionPolicyWrapper:
    """Wraps diffusion policy to handle Isaac Lab environment observations and action execution."""

    def __init__(self, policy, device: torch.device, n_obs_steps: int = 2, num_envs: int = 1):
        """Initialize the policy wrapper.

        Args:
            policy: The diffusion policy to wrap.
            device: Device to run the policy on.
            n_obs_steps: Number of observation steps to maintain in history.
            num_envs: Number of environments to handle.
            execute_horizon: Number of actions to execute from each chunk before
                replanning. None = execute full chunk (open-loop). 1 = replan
                every step (receding horizon).
        """
        self.policy = policy
        self.device = device
        self.n_obs_steps = n_obs_steps
        self.num_envs = num_envs

        # Initialize observation history manager based on policy type
        self.is_image_policy = self._is_image_policy()
        if self.is_image_policy:
            self.obs_history_manager = ImageObservationHistory(num_envs, n_obs_steps, device)
        else:
            self.obs_history_manager = LowDimObservationHistory(num_envs, n_obs_steps, device)

        # Initialize action queue as list of lists for each environment
        self.action_queue = [[] for _ in range(num_envs)]

        # Reset the policy to initialize its internal queues
        self.policy.reset()

    def _is_image_policy(self) -> bool:
        """Detect if this is an image policy based on class name."""
        policy_class_name = self.policy.__class__.__name__.lower()
        image_policy_indicators = ["image", "hybrid", "video"]
        return any(indicator in policy_class_name for indicator in image_policy_indicators)

    def reset(self, reset_ids: torch.Tensor):
        """Reset the policy wrapper and clear observation history and action queue."""
        reset_indices = reset_ids.tolist() if hasattr(reset_ids, "tolist") else reset_ids
        for i in reset_indices:
            self.action_queue[i].clear()

        # Reset observation history for these environments
        if isinstance(reset_indices, torch.Tensor):
            reset_indices = reset_indices.tolist()
        self.obs_history_manager.reset_envs(reset_indices)
        self.policy.reset()

    def predict_action(self, obs_dict: dict[str, Any]) -> torch.Tensor:
        """Predict action given Isaac Lab environment observations.

        Args:
            obs_dict: Raw observations from Isaac Lab environment

        Returns:
            Action tensor for environment execution with shape (num_envs, action_dim)
        """
        # Process observations to format expected by diffusion policy
        processed_obs = self._process_obs(obs_dict)

        # Update observation history with batched operations
        self.obs_history_manager.update(processed_obs)

        # Find environments that need new action chunks
        need_new_actions = [i for i in range(self.num_envs) if len(self.action_queue[i]) == 0]

        if need_new_actions:
            # Get new action chunks for environments that need them
            new_actions = self._get_action_chunks(need_new_actions)

            # Distribute action chunks to respective queues
            for idx, env_idx in enumerate(need_new_actions):
                self.action_queue[env_idx].extend(new_actions[idx])

        # Extract next action for each environment
        actions = torch.zeros(self.num_envs, self.action_queue[0][0].shape[-1], device=self.device, dtype=torch.float32)
        for i in range(self.num_envs):
            actions[i] = self.action_queue[i].pop(0)

        return actions

    def _process_obs(self, obs_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert Isaac Lab observations to format expected by diffusion policy.

        Args:
            obs_dict: Raw observations from environment

        Returns:
            Processed observation dictionary with batched tensors
        """
        # Get policy observations
        if isinstance(obs_dict, dict):
            obs = obs_dict.get("policy", obs_dict)
        else:
            obs = obs_dict

        if self.is_image_policy:
            return self._process_image_obs(obs)
        else:
            return self._process_lowdim_obs(obs)

    def _process_image_obs(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process observations for image-based policies with batched operations.

        Args:
            obs: Raw observations from environment

        Returns:
            Processed observation dictionary for image policy
        """
        processed_obs = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                tensor = value.to(self.device)
            else:
                tensor = torch.tensor(value, device=self.device)
            processed_obs[key] = tensor
        return processed_obs

    def _process_lowdim_obs(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process observations for low-dimensional policies with batched operations.

        Args:
            obs: Raw observations from environment

        Returns:
            Processed observation dictionary for low-dim policy
        """
        # Concatenate all observation components into a single vector
        obs_components = []
        for key in sorted(obs.keys()):
            value = obs[key]
            if isinstance(value, torch.Tensor):
                obs_components.append(value.to(self.device))
            else:
                obs_components.append(torch.tensor(value, device=self.device))

        # Concatenate all components along the feature dimension
        if obs_components:
            obs_tensor = torch.cat(obs_components, dim=-1)
            # Ensure proper shape: (num_envs, features)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            processed_obs = {"obs": obs_tensor}
        else:
            processed_obs = {"obs": torch.zeros((self.num_envs, 0), device=self.device)}

        return processed_obs

    def _get_action_chunks(self, env_indices: list[int]) -> list[torch.Tensor]:
        """Get action chunks for specific environments.

        Args:
            env_indices: List of environment indices that need new action chunks

        Returns:
            List of action chunks for each environment
        """
        # Create observation batch for the environments that need new actions
        obs_batch = self.obs_history_manager.get_batch(env_indices)

        # Get action chunk from policy
        result = self.policy.predict_action(obs_batch)
        if isinstance(result, dict):
            action_chunk = result["action"]
        else:
            action_chunk = result

        # Process action chunk for each environment
        action_chunks = []
        if action_chunk.ndim == 3:
            # Shape: (batch_size, action_chunk_len, action_dim)
            for i in range(action_chunk.shape[0]):
                env_action_chunk = action_chunk[i]  # Shape: (action_chunk_len, action_dim)
                action_chunks.append(env_action_chunk)
        else:
            # Single action case: (batch_size, action_dim) -> list of (1, action_dim) per env
            for i in range(action_chunk.shape[0]):
                action_chunks.append(action_chunk[i].unsqueeze(0))

        return action_chunks
