#!/usr/bin/env bash
# Train fixed-length standard diffusion with transformer backbone.
set -euo pipefail

PROJECT_ROOT="/home/gsang/Projects/Perceiver_IO"
cd "$PROJECT_ROOT"

SYSTEM="${SYSTEM:-2dof}"  # 2dof or 3dof
LOG_DIR="${LOG_DIR:-logs}"

if [[ "$SYSTEM" == "2dof" ]]; then
  TRAIN_FILE="${TRAIN_FILE:-data/2dof/traj_40000-steps_4000.h5}"
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/transformer_2dof}"
  WANDB_PROJECT="${WANDB_PROJECT:-Trajectory-StdDiff-Transformer-2DOF}"
elif [[ "$SYSTEM" == "3dof" ]]; then
  TRAIN_FILE="${TRAIN_FILE:-data/traj_40000-steps_4000.h5}"
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/transformer_3dof}"
  WANDB_PROJECT="${WANDB_PROJECT:-Trajectory-StdDiff-Transformer-3DOF}"
else
  echo "Unsupported SYSTEM=$SYSTEM (expected 2dof or 3dof)"
  exit 1
fi

FIXED_TRAJ_LENGTH="${FIXED_TRAJ_LENGTH:-1000}"
DEVICES="${DEVICES:-0 1}"
EPOCHS="${EPOCHS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
RESUME_CKPT="${RESUME_CKPT:-}"
TIME_LIMIT_SEC="${TIME_LIMIT_SEC:-0}"  # 0 means no timeout

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

CMD=(
  /home/gsang/miniconda3/envs/perceiver/bin/python src/models/trajectory_dpf.py
  --mode train
  --h5_path "$TRAIN_FILE"
  --checkpoint_dir "$CHECKPOINT_DIR"
  --devices $DEVICES
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --num_decoder_blocks 4
  --num_latents 256
  --num_latent_channels 256
  --diffusion_steps 1000
  --fixed_trajectory_length "$FIXED_TRAJ_LENGTH"
  --backbone transformer
  --wandb true
  --wandb_project "$WANDB_PROJECT"
  --resume_from_checkpoint "$RESUME_CKPT"
)

LOG_FILE="$LOG_DIR/std_diffusion_${SYSTEM}_transformer_fixed_time.log"

echo "=== Transformer Fixed-Length Training ==="
echo "system: $SYSTEM"
echo "train file: $TRAIN_FILE"
echo "checkpoint dir: $CHECKPOINT_DIR"
echo "devices: $DEVICES"
echo "time_limit_sec: $TIME_LIMIT_SEC"
echo "log: $LOG_FILE"

if [[ "$TIME_LIMIT_SEC" -gt 0 ]]; then
  timeout "${TIME_LIMIT_SEC}s" "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
  rc=${PIPESTATUS[0]}
  if [[ $rc -eq 124 ]]; then
    echo "[INFO] timeout reached at ${TIME_LIMIT_SEC}s. Training stopped by fixed-time policy." | tee -a "$LOG_FILE"
    exit 0
  fi
  exit "$rc"
else
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
fi
