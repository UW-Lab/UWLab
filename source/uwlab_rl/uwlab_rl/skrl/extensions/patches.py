# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import types
from contextlib import contextmanager
from typing import Any

from isaaclab.envs import ManagerBasedRLEnv
from skrl.agents.torch.base import Agent

from .ext_cfg import ContextInitializerCfg, SupplementaryTrainingCfg


class AgentPatcher:
    def __init__(
        self, locs: dict[str, Any], env: ManagerBasedRLEnv, agent: Agent, suppl_train_cfg: SupplementaryTrainingCfg
    ):
        self.env = env
        self.agent = agent
        self.context = self.context_init(locs, self.env, self.agent, suppl_train_cfg.context_initializers)
        self.dict_mem_func, self.dict_sample_func, self.dict_loss_func = self.init_contextual_func(suppl_train_cfg)

        self.original__update = agent._update
        self.original_memory_add_sample = agent.memory.add_samples
        self.patch_agent()

    def context_init(
        self,
        locs: dict[str, Any],
        env: ManagerBasedRLEnv,
        agent: Agent,
        context_initializers: list[ContextInitializerCfg],
    ):
        context = {}
        for context_initializer in context_initializers:
            context_identifier = context_initializer.context_identifier
            context_initializer_f = context_initializer.context_initializer
            context_params = context_initializer.context_params
            context[context_identifier] = context_initializer_f(env, agent, locs, **context_params)
        return context

    def init_contextual_func(self, suppl_train_cfg: SupplementaryTrainingCfg):
        dict_mem_func = {}
        dict_sample_func = {}
        dict_loss_func = {}

        for sample_term in suppl_train_cfg.supplementary_sample_terms:
            sample_f = sample_term.sample_f_creator(self.env, self.agent, self.context, **sample_term.sample_params)
            sample_size_f = sample_term.sample_size_f_creator(
                self.env, self.agent, self.context, **sample_term.sample_size_params
            )
            dict_sample_func[sample_term.sample_term] = sample_f
            dict_mem_func[sample_term.sample_term] = sample_size_f

        for loss_terms in suppl_train_cfg.supplementary_losses:
            loss_f = loss_terms.loss_f_creator(self.env, self.agent, self.context, **loss_terms.loss_params)
            dict_loss_func[loss_terms.loss_term] = loss_f

        return dict_mem_func, dict_sample_func, dict_loss_func

    def patch_agent(self):
        for term, size_f in self.dict_mem_func.items():
            size = size_f()
            self.agent.memory.create_tensor(name=term, size=size, dtype=torch.float32)
        self.agent._tensors_names += [memory_term for memory_term, _ in self.dict_mem_func.items()]

    def patched_add_samples(self, *args, **kwargs):
        for mem_term, sample_f in self.dict_sample_func.items():
            sample = sample_f(kwargs)
            kwargs[mem_term] = sample
            return self.original_memory_add_sample(**kwargs)

    def patched__update(self, *args, **kwargs):
        """
        Patched _update method that injects 'suppl_loss' if not already provided.
        """
        return self.original__update(*(args[1:]), suppl_loss=self.dict_loss_func)

    def apply_patch(self):
        # Replace the agent's _update method with the patched one
        self.agent._update = types.MethodType(self.patched__update, self.agent)
        self.agent.memory.add_samples = types.MethodType(self.patched_add_samples, self.agent.memory)

    def remove_patch(self):
        # Restore the original _update method
        self.agent._update = self.original__update
        self.agent.memory.add_samples = self.original_memory_add_sample


@contextmanager
def patch_agent_with_supplementary_loss(locs, env, agent, suppl_loss):
    patcher = AgentPatcher(locs, env, agent, suppl_loss)
    patcher.apply_patch()
    try:
        yield
    finally:
        patcher.remove_patch()
