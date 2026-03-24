#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gsang/miniconda3/envs/perceiver/bin/python}"

cd "$PROJECT_ROOT"

TRAIN_H5="${TRAIN_H5:-data/reacher_per_step_branching_train_10000_sources_v1_24cpu/splits/source_group_split_seed0/traj_from_10000_sources_steps_500_train_sources8755_steps_500.h5}"
VAL_H5="${VAL_H5:-data/reacher_per_step_branching_train_10000_sources_v1_24cpu/splits/source_group_split_seed0/traj_from_10000_sources_steps_500_val_sources973_steps_500.h5}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/reacher/dpf_per_step_branching_500}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

"$PYTHON_BIN" src/models/train_trajectory_dpf.py \
  --mode train \
  --h5_path "$TRAIN_H5" \
  --val_h5_path "$VAL_H5" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --devices ${DPF_DEVICES:-0 1 2} \
  --epochs "${DPF_EPOCHS:-3000}" \
  --batch_size "${DPF_BATCH_SIZE:-128}" \
  --num_workers "${DPF_NUM_WORKERS:-16}" \
  --lr "${DPF_LR:-1e-4}" \
  --num_decoder_blocks 4 \
  --num_latents 256 \
  --num_latent_channels 256 \
  --diffusion_steps 1000 \
  --backbone perceiverio \
  --conditioning_mode concat_torque_in_state \
  --query_context_mode clean_prefix_noisy_suffix \
  --token_layout shifted_tau \
  --qpos_representation_override raw \
  --fixed_trajectory_length 500 \
  --wandb "${DPF_WANDB:-true}" \
  --wandb_project "${DPF_WANDB_PROJECT:-trajectory-dpf-smooth}" \
  --wandb_run_name "${DPF_WANDB_RUN_NAME:-TrajectoryDPF_x0Stabilized&AbsoluteTimeEncoding&PerStepBranchingWorkspaceSources10000}" \
  --disable_startup_visualization \
  --disable_wandb_traj_callback \
  --resume_from_checkpoint "${DPF_RESUME_CKPT:-}"
