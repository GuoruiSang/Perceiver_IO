"""
Perceiver IO for Trajectory DPF - Refactored Package

A modular implementation of Diffusion Probabilistic Fields for trajectory generation
using Perceiver IO architecture.

Package Structure:
  config          - Centralized configuration constants
  paths           - Centralized path management
  models/         - Model architectures and core implementations
    ├── __init__.py
    ├── architectures.py    - TrajectoryPerceiverIO, TrajectoryOutputAdapter
    ├── trajectory_dpf.py    - TrajectoryDPF LightningModule (main model)
    └── utils.py            - Utility classes (EMA, OutputQueryProvider)
  
  data/           - Data loading and preprocessing
    ├── __init__.py
    └── mujoco_dataset.py    - MuJoCo trajectory dataset
  
  training/       - Training utilities and callbacks
    ├── __init__.py
    ├── callbacks.py    - WandBTrajectoryLogger
    ├── utils.py        - Training functions (normalization, visualization, simulation)
    └── sampler.py      - Trajectory sampling and I/O utilities
  
  utils/          - General utilities
    └── __init__.py

Usage:
  from src.models import TrajectoryDPF, EMA, OutputQueryProvider
  from src.training import WandBTrajectoryLogger, compute_normalization_stats
  from src.config import DEFAULT_BATCH_SIZE
  from src.paths import PROJECT_ROOT, DATA_DIR
"""

__version__ = "1.0.0"
__all__ = []
