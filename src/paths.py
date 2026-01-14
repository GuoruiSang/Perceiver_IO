"""
Centralized path management for Trajectory DPF.

This module defines all file paths and directories used in the project,
making it easy to customize data locations, checkpoints, and outputs
without modifying code.

Path Categories:
  - Data paths: Training and validation datasets
  - Checkpoint paths: Model checkpoints and weights
  - Output paths: Generated samples, plots, and logs
"""

from pathlib import Path

# ===========================
# Base Project Paths
# ===========================

PROJECT_ROOT = Path(__file__).parent.parent  # /path/to/Perceiver_IO
SRC_ROOT = Path(__file__).parent  # /path/to/src

# ===========================
# Data Paths
# ===========================

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "traj_40000-steps_4000.h5"
DEFAULT_TEST_DATAPATH_PATH = DATA_DIR / "traj_4000-steps_4000.h5"
# ===========================
# Checkpoint and Model Paths
# ===========================

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_RESUME_CHECKPOINT = CHECKPOINT_DIR / "trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&ContextLengthCap:epoch=239_val_loss:val_loss=0.0598.ckpt"

# ===========================
# Output Paths
# ===========================

OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_GENERATED_TRAJECTORIES_PATH = OUTPUT_DIR / "generated_trajectories.h5"

# ===========================
# Config Paths (optional for future use)
# ===========================

CONFIG_DIR = PROJECT_ROOT / "configs"

# ===========================
# Utility Functions
# ===========================

def ensure_directories_exist() -> None:
    """
    Create all necessary directories if they don't exist.
    
    Call this at the start of the program to ensure all output directories are ready.
    """
    directories = [
        DATA_DIR,
        CHECKPOINT_DIR,
        OUTPUT_DIR,
        CONFIG_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Ensured directory exists: {directory}")


def get_checkpoint_filename(model_name: str = "trajectory_dpf") -> str:
    """
    Generate a standardized checkpoint filename.
    
    Args:
        model_name: Base name for the model checkpoint
    
    Returns:
        A formatted checkpoint filename template
    """
    return f"{model_name}_smooth_pos-no-norm_epoch_fourier:{{epoch:03d}}_val_loss:{{val_loss:.4f}}"


# ===========================
# Path Validation
# ===========================

def validate_data_path(path: Path) -> bool:
    """
    Validate that a data file exists.
    
    Args:
        path: Path to validate
    
    Returns:
        True if path exists, False otherwise
    """
    return path.exists() and path.is_file()


def validate_directory(path: Path) -> bool:
    """
    Validate that a directory exists and is writable.
    
    Args:
        path: Directory path to validate
    
    Returns:
        True if directory exists and is writable
    """
    if not path.exists():
        return False
    if not path.is_dir():
        return False
    # Try to write a test file
    try:
        test_file = path / ".write_test"
        test_file.touch()
        test_file.unlink()
        return True
    except (OSError, PermissionError):
        return False
