# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Loading and multi-GPU inference for ASTEROID diffusion-policy students.

The simulator runs on one GPU. The student runs on every *other* visible GPU: with one
extra GPU it simply moves there, and with several the environments are split into
contiguous blocks, each served by its own policy replica, and the replicas run
concurrently. Without an extra GPU the student shares the simulator's device.
"""

from __future__ import annotations

import torch
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import dill
import hydra

from uwlab_rl.wrappers.diffusion import DiffusionPolicyWrapper


def resolve_student_devices(sim_device: str | torch.device, requested: str | None = None) -> list[torch.device]:
    """Devices for student inference.

    Args:
        sim_device: Device the simulator runs on.
        requested: Comma-separated device list (e.g. ``"cuda:1,cuda:2"``) overriding the default
            of every visible CUDA device except ``sim_device``.
    """
    sim_device = torch.device(sim_device)
    if requested:
        return [torch.device(d.strip()) for d in requested.split(",") if d.strip()]
    if sim_device.type == "cuda":
        sim_index = sim_device.index if sim_device.index is not None else torch.cuda.current_device()
        extra = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count()) if i != sim_index]
        if extra:
            return extra
    return [sim_device]


def load_student(checkpoint_path: str, device: torch.device):
    """Load a diffusion-policy student (EMA weights when trained with EMA) onto ``device``."""
    with open(checkpoint_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill)
    cfg = payload["cfg"]
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    return policy.eval().to(device)


class MultiDeviceStudent:
    """:class:`DiffusionPolicyWrapper` interface spread over one or more devices.

    Env ``i`` belongs to the shard holding the contiguous block that contains ``i``; each shard
    has its own policy replica, observation history and action queue. Actions are returned on
    ``output_device`` in the order of the requested envs.
    """

    def __init__(
        self,
        policies: list,
        devices: list[torch.device],
        num_envs: int,
        output_device: str | torch.device,
        use_amp: bool = False,
    ):
        assert len(policies) == len(devices) >= 1
        num_shards = min(len(devices), num_envs)
        self.output_device = torch.device(output_device)
        self.use_amp = use_amp
        self.bounds = [(k * num_envs // num_shards, (k + 1) * num_envs // num_shards) for k in range(num_shards)]
        self.devices = devices[:num_shards]
        self.shards = [
            DiffusionPolicyWrapper(policy, device, n_obs_steps=policy.n_obs_steps, num_envs=end - start)
            for policy, device, (start, end) in zip(policies, self.devices, self.bounds)
        ]
        self.n_obs_steps = self.shards[0].n_obs_steps
        self._executor = ThreadPoolExecutor(max_workers=num_shards) if num_shards > 1 else None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        devices: list[torch.device],
        num_envs: int,
        output_device: str | torch.device,
        use_amp: bool = False,
    ) -> MultiDeviceStudent:
        devices = devices[: max(1, min(len(devices), num_envs))]
        policies = [load_student(checkpoint_path, device) for device in devices]
        return cls(policies, devices, num_envs, output_device, use_amp)

    def _split(self, env_ids: list[int]) -> list[list[tuple[int, int]]]:
        """For each shard, the ``(position in env_ids, local env index)`` pairs it serves."""
        groups: list[list[tuple[int, int]]] = [[] for _ in self.shards]
        for pos, env_id in enumerate(env_ids):
            for k, (start, end) in enumerate(self.bounds):
                if start <= env_id < end:
                    groups[k].append((pos, env_id - start))
                    break
        return groups

    def reset(self, reset_ids) -> None:
        ids = reset_ids.tolist() if hasattr(reset_ids, "tolist") else list(reset_ids)
        for shard, group in zip(self.shards, self._split(ids)):
            if group:
                shard.reset([local for _, local in group])

    def predict_action(self, obs_dict: dict, env_ids=None) -> torch.Tensor:
        """Actions for ``env_ids`` (default: every env), given their observation rows.

        ``obs_dict`` is either ``{"policy": {...}}`` or the policy group itself; every tensor's
        first dimension lines up with ``env_ids``.
        """
        obs = obs_dict.get("policy", obs_dict) if isinstance(obs_dict, dict) else obs_dict
        if env_ids is None:
            env_ids = list(range(self.bounds[-1][1]))
        elif hasattr(env_ids, "tolist"):
            env_ids = env_ids.tolist()
        groups = self._split(env_ids)

        def run(k: int):
            group = groups[k]
            if not group:
                return None
            shard, (start, end) = self.shards[k], self.bounds[k]
            positions = torch.tensor([pos for pos, _ in group])
            local_ids = [local for _, local in group]
            shard_obs = {
                key: value[positions.to(value.device)] if isinstance(value, torch.Tensor) else value
                for key, value in obs.items()
            }
            amp = torch.autocast(device_type=shard.device.type) if self.use_amp else nullcontext()
            with torch.inference_mode(), amp:
                if shard.is_transformer:
                    actions = shard.predict_action(shard_obs, local_ids)
                else:
                    if len(local_ids) != end - start:
                        raise ValueError("non-transformer students must act for every env in their shard")
                    actions = shard.predict_action(shard_obs)
            return positions, actions.to(self.output_device)

        if self._executor is None:
            results = [run(0)]
        else:
            results = list(self._executor.map(run, range(len(self.shards))))

        out = None
        for result in results:
            if result is None:
                continue
            positions, actions = result
            if out is None:
                out = torch.zeros(len(env_ids), actions.shape[-1], device=self.output_device, dtype=actions.dtype)
            out[positions.to(self.output_device)] = actions
        return out
