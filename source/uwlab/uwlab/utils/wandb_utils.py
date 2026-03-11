import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
import os

from dataclasses import asdict
from typing import Optional

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from metalearning.isaac.plot_utils import get_figure

import h5py

def log_rl2_stats(returns: torch.Tensor, success: torch.Tensor, num_steps: torch.Tensor, log_step: int, num_envs: int, dataset_success_rate: Optional[float] = None, save_path: Optional[str] = None, log_once=True):
    if log_once and hasattr(log_rl2_stats, 'logged'):
        return

    wandb.define_metric("RL2/step")
    wandb.define_metric("RL2/*", step_metric="RL2/step")

    if not hasattr(log_rl2_stats, 'num_finished_envs'):
        log_rl2_stats.num_finished_envs = 0
        log_rl2_stats.returns = []
        log_rl2_stats.success = []
        log_rl2_stats.num_steps = []

    log_rl2_stats.num_finished_envs += returns.shape[0]
    log_rl2_stats.returns.append(returns.clone())
    log_rl2_stats.success.append(success.clone())
    log_rl2_stats.num_steps.append(num_steps.clone())
    if log_rl2_stats.num_finished_envs < num_envs:
        return

    log_rl2_stats.logged = True

    stats = {}

    returns = torch.cat(log_rl2_stats.returns, dim=0)
    success = torch.cat(log_rl2_stats.success, dim=0)
    num_steps = torch.cat(log_rl2_stats.num_steps, dim=0)
    cum_success = torch.clamp(torch.cumsum(success, dim=1), max=1.0).to(dtype=torch.bool)
    
    # compute consecutive success rate (after a successful episode, what % of environments stay successful?)
    last_was_success = torch.roll(success, 1, dims=1)
    consecutive_success_rate = torch.logical_and(success, last_was_success).sum(dim=0) / torch.clamp(last_was_success.sum(dim=0), min=1)
    consecutive_success_rate[0] = 0.0 # first episode is not consecutive
    stats["RL2/consecutive_success_rate"] = get_figure(
        consecutive_success_rate.unsqueeze(0), "Episodes", "Consecutive Success Rate"
    )
    
    # compute fail to success rate (after an unsuccessful episode, what % of environments become successful?)
    fail_to_success_rate = torch.logical_and(success, ~last_was_success).sum(dim=0) / torch.clamp((~last_was_success).sum(dim=0), min=1)
    fail_to_success_rate[0] = 0.0 # first episode has no previous episode
    stats["RL2/fail_to_success_rate"] = get_figure(
        fail_to_success_rate.unsqueeze(0), "Episodes", "Fail to Success Rate"
    )

    stats["RL2/returns"] = get_figure(
        returns, "Episodes", "Return"
    )
    
    success_rate_ref = None
    if dataset_success_rate is not None:
        success_rate_ref = dataset_success_rate / 100.0
    
    stats["RL2/success"] = get_figure(
        success, "Episodes", "Success", reference_line=success_rate_ref
    )
    
    stats["RL2/num_steps"] = get_figure(
        num_steps, "Episodes", "Number of Steps"
    )
    
    cum_success_rate_ref = None
    if dataset_success_rate is not None:
        cum_success_rate_ref = dataset_success_rate / 100.0
    
    stats["RL2/cum_success"] = get_figure(
        cum_success, "Episodes", "Cumulative Success", reference_line=cum_success_rate_ref
    )

    stats["RL2/max_return"] = returns.max(dim=1)[0].mean()
    stats["RL2/improvement"] = (returns[:, -1] - returns[:, 0]).mean()
    
    diffs = returns[:, 1:] - returns[:, :-1]
    stats["RL2/return_diff_plot"] = get_figure(
        diffs, "Episodes", "Return Difference"
    )
    stats["RL2/num_envs"] = returns.shape[0]
    stats["RL2/step"] = log_step

    wandb.log(stats)

    if save_path is not None:
        # this is called before any other access to the file from the main training script, so we delete any old file and write to it
        # TODO: this is a hack, should be more robust
        if os.path.exists(save_path):
            os.remove(save_path)
        with h5py.File(save_path, 'a') as f:
            f.create_dataset("returns", data=returns.cpu().numpy(), compression="gzip", compression_opts=4)
            f.create_dataset("consecutive_success_rate", data=consecutive_success_rate.cpu().numpy(), compression="gzip", compression_opts=4)
            f.create_dataset("fail_to_success_rate", data=fail_to_success_rate.cpu().numpy(), compression="gzip", compression_opts=4)
            f.create_dataset("success", data=success.cpu().numpy(), compression="gzip", compression_opts=4)
            f.create_dataset("num_steps", data=num_steps.cpu().numpy(), compression="gzip", compression_opts=4)
            f.create_dataset("cum_success", data=cum_success.cpu().numpy(), compression="gzip", compression_opts=4)
            f.attrs["dataset_success_rate"] = dataset_success_rate

    log_rl2_stats.num_finished_envs = 0 # reset for next batch
    log_rl2_stats.returns = []
    log_rl2_stats.success = []
    log_rl2_stats.num_steps = []

