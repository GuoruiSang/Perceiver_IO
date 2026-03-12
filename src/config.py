"""
Centralized configuration for Trajectory DPF training.
"""

from src.paths import (
    DEFAULT_DATASET_PATH,
    CHECKPOINT_DIR,
    DEFAULT_RESUME_CHECKPOINT,
)

# ===========================
# Data and Paths Configuration
# ===========================

DEFAULT_H5_PATH = str(DEFAULT_DATASET_PATH)
DEFAULT_CHECKPOINT_DIR = str(CHECKPOINT_DIR)
DEFAULT_RESUME_CHECKPOINT = str(DEFAULT_RESUME_CHECKPOINT) if DEFAULT_RESUME_CHECKPOINT is not None else None

# ===========================
# Training Parameters
# ===========================

# Data loading
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 16
DEFAULT_TRAIN_VAL_SPLIT = 0.9  # 90% train, 10% val

# Model training
DEFAULT_EPOCHS = 3000
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_LATENTS = 256
DEFAULT_NUM_LATENT_CHANNELS = 256
DEFAULT_DIFFUSION_STEPS = 1000

# ===========================
# W&B (Weights & Biases) Configuration
# ===========================

DEFAULT_WANDB_ENABLED = True
DEFAULT_WANDB_PROJECT = "trajectory-dpf-smooth"
DEFAULT_WANDB_ENTITY = None
DEFAULT_WANDB_RUN_NAME = 'TrajectoryDPF_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions'

# ===========================
# Mode Configuration
# ===========================

DEFAULT_MODE = "train"

# ===========================
# Normalization Configuration (Data preprocessing)
# ===========================

DEFAULT_NORMALIZATION_RANGE_EPSILON = 1e-2  # Minimum range for stability in min-max normalization
