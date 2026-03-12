#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gsang/miniconda3/envs/perceiver/bin/python}"
cd "$PROJECT_ROOT"

NUM_WORKERS="${NUM_WORKERS:-24}"

echo "=== Generate core datasets for final alpha=1 reproducibility ==="
echo "python: $PYTHON_BIN"
echo "num_workers: $NUM_WORKERS"

echo ""
echo "[1/2] 2DoF train/val datasets"
"$PYTHON_BIN" scripts/data/generate_dataset_forward.py \
  --torque_policies "sinusoidal:1.0" \
  --xml_path configs/rigid_arm_hinge_2dof.xml \
  --num_trajectories 40000 \
  --num_val 2000 \
  --num_steps 4000 \
  --save_path data/2dof \
  --num_workers "$NUM_WORKERS"

echo ""
echo "[2/2] 3DoF train/val datasets"
"$PYTHON_BIN" scripts/data/generate_dataset_forward.py \
  --torque_policies "sinusoidal:1.0" \
  --xml_path configs/rigid_arm_hinge.xml \
  --num_trajectories 40000 \
  --num_val 2000 \
  --num_steps 4000 \
  --save_path data/3dof \
  --num_workers "$NUM_WORKERS"

echo ""
echo "Core dataset generation complete."
