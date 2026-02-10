#!/usr/bin/env bash
# Launch 2DoF/3DoF transformer training with fixed wall-clock budgets from original DPF runs.
set -euo pipefail

PROJECT_ROOT="/home/gsang/Projects/Perceiver_IO"
cd "$PROJECT_ROOT"

# DPF reference durations (hours -> seconds):
# 2DoF: 24h36m52s = 885... wait exact below.
T_DPF_2DOF_SEC="${T_DPF_2DOF_SEC:-88612}"  # 24h 36m 52s
T_DPF_3DOF_SEC="${T_DPF_3DOF_SEC:-87992}"  # 24h 26m 32s

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

tmux kill-session -t transformer_2dof_fixed_time 2>/dev/null || true
tmux kill-session -t transformer_3dof_fixed_time 2>/dev/null || true

tmux new-session -d -s transformer_2dof_fixed_time \
  "cd $PROJECT_ROOT && CUDA_VISIBLE_DEVICES=0,1 SYSTEM=2dof DEVICES='0 1' TIME_LIMIT_SEC=$T_DPF_2DOF_SEC bash scripts/train_transformer_fixed_length.sh 2>&1 | tee $LOG_DIR/std_diffusion_2dof_gpu01_transformer.log"

tmux new-session -d -s transformer_3dof_fixed_time \
  "cd $PROJECT_ROOT && CUDA_VISIBLE_DEVICES=2,3 SYSTEM=3dof DEVICES='0 1' TIME_LIMIT_SEC=$T_DPF_3DOF_SEC bash scripts/train_transformer_fixed_length.sh 2>&1 | tee $LOG_DIR/std_diffusion_3dof_gpu23_transformer.log"

echo "Started tmux sessions:"
tmux ls | rg "transformer_(2dof|3dof)_fixed_time"
echo "2DoF budget (sec): $T_DPF_2DOF_SEC"
echo "3DoF budget (sec): $T_DPF_3DOF_SEC"
