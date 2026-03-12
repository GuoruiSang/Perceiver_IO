#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gsang/miniconda3/envs/perceiver/bin/python}"
cd "$PROJECT_ROOT"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR" checkpoints/2dof/dpf checkpoints/2dof/hnn

DPF_TRAIN_FILE="${DPF_TRAIN_FILE:-data/2dof/traj_40000-steps_4000.h5}"
HNN_TRAIN_FILE="${HNN_TRAIN_FILE:-data/2dof/traj_40000-steps_4000.h5}"
HNN_VAL_FILE="${HNN_VAL_FILE:-data/2dof/traj_2000-steps_4000.h5}"

echo "=== Train final 2DoF checkpoints ==="
echo "python: $PYTHON_BIN"
echo "dpf train file: $DPF_TRAIN_FILE"
echo "hnn train file: $HNN_TRAIN_FILE"
echo "hnn val file: $HNN_VAL_FILE"

nohup "$PYTHON_BIN" src/models/train_trajectory_dpf.py \
  --mode train \
  --h5_path "$DPF_TRAIN_FILE" \
  --checkpoint_dir checkpoints/2dof/dpf \
  --devices ${DPF_DEVICES:-0 1 2} \
  --epochs "${DPF_EPOCHS:-3000}" \
  --batch_size "${DPF_BATCH_SIZE:-128}" \
  --lr "${DPF_LR:-1e-4}" \
  --num_decoder_blocks 4 \
  --num_latents 256 \
  --num_latent_channels 256 \
  --diffusion_steps 1000 \
  --wandb "${DPF_WANDB:-true}" \
  --wandb_project "${DPF_WANDB_PROJECT:-Trajectory-DPF-2DOF}" \
  --resume_from_checkpoint "${DPF_RESUME_CKPT:-}" \
  > "$LOG_DIR/dpf_2dof_training.log" 2>&1 &

echo "Started 2DoF DPF training. Log: $LOG_DIR/dpf_2dof_training.log"

CUDA_VISIBLE_DEVICES="${HNN_CUDA_VISIBLE_DEVICES:-3}" nohup "$PYTHON_BIN" src/models/HNN.py \
  --mode train \
  --train_file "$HNN_TRAIN_FILE" \
  --test_file "$HNN_VAL_FILE" \
  --checkpoint_dir checkpoints/2dof/hnn \
  --checkpoint_prefix "${HNN_PREFIX:-StructuredHNN-2DOF}" \
  --wandb_name "${HNN_WANDB_NAME:-StructuredHNN-2DOF-Hinge}" \
  --xml_path configs/rigid_arm_hinge_2dof.xml \
  --model_type "${HNN_MODEL_TYPE:-structured}" \
  --hidden_dim "${HNN_HIDDEN_DIM:-256}" \
  --num_layers "${HNN_NUM_LAYERS:-4}" \
  > "$LOG_DIR/hnn_2dof_training.log" 2>&1 &

echo "Started 2DoF HNN training. Log: $LOG_DIR/hnn_2dof_training.log"
