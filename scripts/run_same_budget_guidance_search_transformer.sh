#!/usr/bin/env bash
# Run same-budget guidance search for transformer fixed diffusion on 2DoF/3DoF.
set -euo pipefail

PROJECT_ROOT="/home/gsang/Projects/Perceiver_IO"
cd "$PROJECT_ROOT"

SYSTEMS="${SYSTEMS:-2dof,3dof}"
OUT_ROOT="${OUT_ROOT:-output_ablation}"

pick_latest_ckpt() {
  local ckpt_dir="$1"
  find "$ckpt_dir" -maxdepth 1 -type f -name "*.ckpt" -printf "%T@ %p\n" \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

IFS=',' read -r -a SYS_ARR <<< "$SYSTEMS"
for sys in "${SYS_ARR[@]}"; do
  sys="$(echo "$sys" | xargs)"
  if [[ -z "$sys" ]]; then
    continue
  fi
  if [[ "$sys" != "2dof" && "$sys" != "3dof" ]]; then
    echo "Unsupported system '$sys' (expected 2dof or 3dof)"
    exit 1
  fi

  ckpt_dir="checkpoints/transformer_${sys}"
  if [[ "$sys" == "2dof" ]]; then
    ckpt="${FIXED_DIFF_CKPT_2DOF:-${FIXED_DIFF_CKPT:-}}"
  else
    ckpt="${FIXED_DIFF_CKPT_3DOF:-${FIXED_DIFF_CKPT:-}}"
  fi
  if [[ -z "$ckpt" ]]; then
    ckpt="$(pick_latest_ckpt "$ckpt_dir")"
  fi
  if [[ -z "${ckpt:-}" || ! -f "$ckpt" ]]; then
    echo "Checkpoint not found for $sys. Expected in $ckpt_dir or set FIXED_DIFF_CKPT."
    exit 1
  fi

  out_dir="${OUT_ROOT}/same_budget_guidance_search_${sys}_transformer"
  echo "[search] system=$sys ckpt=$ckpt out=$out_dir"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" \
    /home/gsang/miniconda3/envs/perceiver/bin/python scripts/run_same_budget_guidance_search_3dof.py \
      --system "$sys" \
      --fixed-diff-ckpt "$ckpt" \
      --out-dir "$out_dir" \
      "$@"
done
