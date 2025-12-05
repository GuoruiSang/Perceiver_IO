"""
Models package for Trajectory DPF.

This package contains:
- architectures: Core model architecture classes (TrajectoryPerceiverIO, TrajectoryOutputAdapter)
- trajectory_dpf: Main TrajectoryDPF LightningModule and entry point
- utils: Utility classes (EMA, OutputQueryProvider)
"""

from perceiver.model.core import InputAdapter
from src.models.architectures import TrajectoryPerceiverIO, TrajectoryOutputAdapter
from src.models.utils import OutputQueryProvider, EMA

__all__ = [
    "TrajectoryPerceiverIO",
    "TrajectoryOutputAdapter",
    "InputAdapter",
    "OutputQueryProvider",
    "EMA",
]
