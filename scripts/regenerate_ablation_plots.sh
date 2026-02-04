#!/bin/bash
# Regenerate all 4 ablation plots from the full dependency chain.
# Skips steps if output files already exist.

set -e
cd "$(dirname "$0")/.."

echo "=== Ablation Plots Regeneration Script ==="

# --- Level 1: Generate H5 trajectories (if missing) ---
TRAJ_DIR="output_ablation/trajectories/original"
if [[ ! -f "$TRAJ_DIR/exp_a_sinusoidal.h5" ]] || \
   [[ ! -f "$TRAJ_DIR/exp_a_gp.h5" ]] || \
   [[ ! -f "$TRAJ_DIR/exp_a_zero.h5" ]] || \
   [[ ! -f "$TRAJ_DIR/exp_b_context_fractions.h5" ]]; then
    echo "[Level 1] Generating H5 trajectories..."
    python scripts/generate_ablation_trajectories.py \
        --model_name original \
        --checkpoint checkpoints/model.ckpt \
        --output_dir output_ablation
else
    echo "[Level 1] H5 trajectories already exist, skipping."
fi

# --- Level 2a: Compute MSE CSVs from H5 (if missing) ---
CSV_DIR="output_ablation/results/original"
if [[ ! -f "$CSV_DIR/exp_a_sinusoidal.csv" ]] || \
   [[ ! -f "$CSV_DIR/exp_a_gp.csv" ]] || \
   [[ ! -f "$CSV_DIR/exp_a_zero.csv" ]] || \
   [[ ! -f "$CSV_DIR/exp_b_context_fractions.csv" ]]; then
    echo "[Level 2a] Computing MSE CSVs from H5..."
    python scripts/compute_ablation_mse.py \
        --input_dir output_ablation/trajectories \
        --output_dir output_ablation/results
else
    echo "[Level 2a] MSE CSVs already exist, skipping."
fi

# --- Level 2b: Compute spline MSE CSV (if missing) ---
SPLINE_CSV="output_ablation/results/latest/mse_spline_torques.csv"
if [[ ! -f "$SPLINE_CSV" ]]; then
    echo "[Level 2b] Computing spline MSE CSV..."
    python scripts/eval_mse_spline.py
else
    echo "[Level 2b] Spline MSE CSV already exists, skipping."
fi

# --- Level 2c: Compute HamRes CSVs (if missing) ---
# Note: HamRes computation requires GPU and takes ~15min per policy
HAMRES_MISSING=0
for policy in sinusoidal gp zero spline; do
    if [[ ! -f "$CSV_DIR/hamres_pct_${policy}.csv" ]]; then
        HAMRES_MISSING=1
        break
    fi
done

if [[ $HAMRES_MISSING -eq 1 ]]; then
    echo "[Level 2c] HamRes CSVs missing - run HamRes computation manually (GPU required)"
    echo "         Scripts: compute_hamres_pct_*.py in scratchpad"
else
    echo "[Level 2c] HamRes CSVs already exist, skipping."
fi

# --- Level 3: Generate plots ---
echo "[Level 3] Generating plots..."
python scripts/plot_ablation_original.py
python scripts/plot_ablation_nmse.py
python scripts/plot_ablation_hamres.py

echo ""
echo "=== Done! Generated plots: ==="
ls -la plots/ablation_*.png
