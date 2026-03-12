"""
Centralized path constants for the active training/eval project.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "3dof" / "traj_40000-steps_4000.h5"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "3dof" / "dpf"
DEFAULT_RESUME_CHECKPOINT = (
    CHECKPOINT_DIR
    / "trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt"
)
