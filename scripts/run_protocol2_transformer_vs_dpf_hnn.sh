#!/usr/bin/env bash
# Run Protocol2 comparisons using latest transformer checkpoints.
set -euo pipefail

PROJECT_ROOT="/home/gsang/Projects/Perceiver_IO"
cd "$PROJECT_ROOT"

pick_latest_ckpt() {
  local ckpt_dir="$1"
  find "$ckpt_dir" -maxdepth 1 -type f -name "*.ckpt" -printf "%T@ %p\n" \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

CKPT_2DOF="${CKPT_2DOF:-$(pick_latest_ckpt checkpoints/transformer_2dof)}"
CKPT_3DOF="${CKPT_3DOF:-$(pick_latest_ckpt checkpoints/transformer_3dof)}"

if [[ -z "${CKPT_2DOF:-}" || ! -f "$CKPT_2DOF" ]]; then
  echo "2DoF checkpoint not found. Set CKPT_2DOF or wait for training."
  exit 1
fi
if [[ -z "${CKPT_3DOF:-}" || ! -f "$CKPT_3DOF" ]]; then
  echo "3DoF checkpoint not found. Set CKPT_3DOF or wait for training."
  exit 1
fi

echo "Using checkpoints:"
echo "  2DoF: $CKPT_2DOF"
echo "  3DoF: $CKPT_3DOF"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" \
FIXED_DIFFUSION_CKPT="$CKPT_2DOF" \
OUT_ROOT="${OUT_ROOT_2DOF:-output_ablation/protocol2_2dof_transformer_eval_with_hnn}" \
PLOTS_ROOT="${PLOTS_ROOT:-plots}" \
/home/gsang/miniconda3/envs/perceiver/bin/python scripts/run_protocol2_2dof_diffusion_vs_dpf_hnn_boxplots.py

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" \
FIXED_DIFFUSION_CKPT="$CKPT_3DOF" \
OUT_ROOT="${OUT_ROOT_3DOF:-output_ablation/protocol2_3dof_transformer_eval_with_hnn}" \
PLOTS_ROOT="${PLOTS_ROOT:-plots}" \
/home/gsang/miniconda3/envs/perceiver/bin/python scripts/run_protocol2_3dof_diffusion_vs_dpf_boxplots.py
