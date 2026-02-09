#!/usr/bin/env bash
# Train 2DoF fixed-length diffusion model (standard diffusion, not variable-length DPF).
set -euo pipefail

PROJECT_ROOT="/home/gsang/Projects/Perceiver_IO"
cd "$PROJECT_ROOT"

TRAIN_FILE="${TRAIN_FILE:-data/2dof/traj_40000-steps_4000.h5}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/2dof}"
LOG_DIR="${LOG_DIR:-logs}"

FIXED_TRAJ_LENGTH="${FIXED_TRAJ_LENGTH:-1000}"
DEVICES="${DEVICES:-0 1 2}"
EPOCHS="${EPOCHS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
WANDB_PROJECT="${WANDB_PROJECT:-Trajectory-DPF-2DOF}"
RESUME_CKPT="${RESUME_CKPT:-}"

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

echo "=== 2DoF Fixed-Length Training ==="
echo "train file: $TRAIN_FILE"
echo "fixed trajectory length: $FIXED_TRAJ_LENGTH"
echo "devices: $DEVICES"
echo "epochs: $EPOCHS"
echo "checkpoint dir: $CHECKPOINT_DIR"
echo "log file: $LOG_DIR/dpf_2dof_fixed_length.log"

python src/models/trajectory_dpf.py \
  --mode train \
  --h5_path "$TRAIN_FILE" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --devices $DEVICES \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --num_decoder_blocks 4 \
  --num_latents 256 \
  --num_latent_channels 256 \
  --diffusion_steps 1000 \
  --fixed_trajectory_length "$FIXED_TRAJ_LENGTH" \
  --wandb true \
  --wandb_project "$WANDB_PROJECT" \
  --resume_from_checkpoint "$RESUME_CKPT" \
  2>&1 | tee "$LOG_DIR/dpf_2dof_fixed_length.log"
