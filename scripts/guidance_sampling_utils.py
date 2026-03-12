"""Shared sampling/model helpers for the active evaluation scripts."""

from __future__ import annotations

import numpy as np
import torch

from scripts.system_eval_utils import load_system_models


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_models(cfg, device):
    return load_system_models(cfg, device)


def run_batch(dpf, torques, noise, length: int, batch_size: int, **kwargs):
    num_samples = torques.shape[0]
    all_states = []
    all_torques = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_torques = torques[start:end]
        batch_noise = noise[start:end] if noise is not None else None
        state, tau_out = dpf.sample_trajectories(
            num_samples=batch_torques.shape[0],
            trajectory_length=length,
            context_fraction=0.5,
            use_ema=True,
            torque=batch_torques,
            initial_noise=batch_noise,
            **kwargs,
        )
        all_states.append(state)
        all_torques.append(tau_out)
    return torch.cat(all_states, dim=0), torch.cat(all_torques, dim=0)
