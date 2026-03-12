"""
Models package for Trajectory DPF.

This package contains:
- architectures: Core model architecture classes.
- trajectory_dpf_model: Main TrajectoryDPF LightningModule.
- train_trajectory_dpf: training CLI entrypoint for TrajectoryDPF.
- trajectory_dpf_training: training/validation helpers.
- trajectory_dpf_sampling: generic diffusion sampling helpers.
- utils: Shared EMA and physics/eval helpers.
"""

from perceiver.model.core import InputAdapter
from src.models.architectures import TrajectoryPerceiverIO, TrajectoryOutputAdapter, OutputQueryProvider
from src.models.utils import EMA

__all__ = [
    "TrajectoryPerceiverIO",
    "TrajectoryOutputAdapter",
    "InputAdapter",
    "OutputQueryProvider",
    "EMA",
]
