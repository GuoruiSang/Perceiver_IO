#!/usr/bin/env bash
# Post-training pipeline: guidance search + Protocol2 eval + final report plots.
set -euo pipefail

PROJECT_ROOT="/home/gsang/Projects/Perceiver_IO"
cd "$PROJECT_ROOT"

RUN_SEARCH="${RUN_SEARCH:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_REPORT="${RUN_REPORT:-0}"  # 0 by default: keep boxplot-only main flow

if [[ "$RUN_SEARCH" == "1" ]]; then
  echo "[step] same-budget guidance search (2DoF + 3DoF)"
  SYSTEMS="${SYSTEMS:-2dof,3dof}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" \
    bash scripts/run_same_budget_guidance_search_transformer.sh
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  echo "[step] Protocol2 full evaluation + boxplots (2DoF + 3DoF)"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" \
    bash scripts/run_protocol2_transformer_vs_dpf_hnn.sh
fi

if [[ "$RUN_REPORT" == "1" ]]; then
  echo "[step] build final summary csv and report plots"
  /home/gsang/miniconda3/envs/perceiver/bin/python scripts/build_protocol2_transformer_reports.py
fi

echo "[done] post-training pipeline finished"
