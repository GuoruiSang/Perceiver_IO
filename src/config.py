"""
Centralized configuration for Trajectory DPF training and generation.

This module consolidates all hardcoded hyperparameters and references
to centralized path definitions for easy configuration management.
"""

from src.paths import (
    DEFAULT_DATASET_PATH,
    DEFAULT_TEST_DATAPATH_PATH,
    CHECKPOINT_DIR,
    DEFAULT_RESUME_CHECKPOINT,
    DEFAULT_GENERATED_TRAJECTORIES_PATH,
)

# ===========================
# Data and Paths Configuration
# ===========================

DEFAULT_H5_PATH = str(DEFAULT_DATASET_PATH)
DEFAULT_TEST_H5_PATH = str(DEFAULT_TEST_DATAPATH_PATH)
DEFAULT_CHECKPOINT_DIR = str(CHECKPOINT_DIR)
DEFAULT_RESUME_CHECKPOINT = str(DEFAULT_RESUME_CHECKPOINT) if DEFAULT_RESUME_CHECKPOINT is not None else None
DEFAULT_OUTPUT_PATH = str(DEFAULT_GENERATED_TRAJECTORIES_PATH)

# ===========================
# Training Parameters
# ===========================

# Data loading
DEFAULT_BATCH_SIZE = 256
DEFAULT_NUM_WORKERS = 16
DEFAULT_TRAIN_VAL_SPLIT = 0.9  # 90% train, 10% val

# Model training
DEFAULT_EPOCHS = 3000
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_LATENTS = 256
DEFAULT_NUM_LATENT_CHANNELS = 256
DEFAULT_DIFFUSION_STEPS = 1000

# Checkpoint saving
DEFAULT_CHECKPOINT_SAVE_TOP_K = 3
DEFAULT_CHECKPOINT_SAVE_INTERVAL = 50  # every N epochs
DEFAULT_CHECKPOINT_FILENAME_TEMPLATE = 'trajectory_dpf_forward_epoch:{epoch:03d}_val_loss:{val_loss:.4f}'

# Training loop
DEFAULT_LOG_EVERY_N_STEPS = 10

# ===========================
# Sampling/Generation Parameters
# ===========================

DEFAULT_NUM_SAMPLES = 8
DEFAULT_SAMPLER = "ddim"  # choices: ["ddpm", "ddim", "ddpm_legacy"]
DEFAULT_NUM_DIFFUSION_STEPS = 200
DEFAULT_CONTEXT_FRACTION = 0.5
DEFAULT_USE_EMA = True

# ===========================
# W&B (Weights & Biases) Configuration
# ===========================

DEFAULT_WANDB_ENABLED = True
DEFAULT_WANDB_PROJECT = "trajectory-dpf-smooth"
DEFAULT_WANDB_ENTITY = None
DEFAULT_WANDB_RUN_NAME = 'Position&Momentum|Torque(AdaIN)'

# ===========================
# Mode Configuration
# ===========================

DEFAULT_MODE = "generate_samples"  # choices: ["train", "generate_samples"]
DEFAULT_DEVICE = "gpu"  # "gpu" if available, else "cpu" (handled dynamically)

# ===========================
# Normalization Configuration (Data preprocessing)
# ===========================

DEFAULT_NORMALIZATION_RANGE_EPSILON = 1e-2  # Minimum range for stability in min-max normalization
