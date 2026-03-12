#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

PY=/home/gsang/miniconda3/envs/perceiver/bin/python
TS=${1:-$(date +%Y%m%d_%H%M%S)}
ALPHA=1.0

DPF_RUN_ROOT=eval_runs
DIFF_RUN_ROOT=output/eval_runs
PLOT_ROOT=plots/method_comparison_7methods_rmse_alpha1_${TS}
TABLE_ROOT=plots/method_comparison_7methods/tables
LOG_ROOT=logs/alpha1_regen_parallel_${TS}
mkdir -p "$LOG_ROOT" "$TABLE_ROOT"

DPF2_OUT=${DPF_RUN_ROOT}/default_full_sweep_alpha1_dpf2_${TS}
DPF3_OUT=${DPF_RUN_ROOT}/default_full_sweep_3dof_ckpt0010_alpha1_dpf3_${TS}
DIFF2_OUT=${DIFF_RUN_ROOT}/2dof_transformer_eval_with_hnn_alpha1_${TS}
DIFF3_OUT=${DIFF_RUN_ROOT}/3dof_transformer_eval_with_hnn_alpha1_${TS}
DPF2_RESAMP_OUT=${DPF_RUN_ROOT}/resampling_strategy1_n4_alpha1_dpf2_${TS}
DPF3_RESAMP_OUT=${DPF_RUN_ROOT}/resampling_strategy1_n4_alpha1_dpf3_${TS}
DIFF2_RESAMP_OUT=${DIFF_RUN_ROOT}/2dof_transformer_eval_with_hnn_resampling_n4_alpha1_${TS}
DIFF3_RESAMP_OUT=${DIFF_RUN_ROOT}/3dof_transformer_eval_with_hnn_resampling_n4_alpha1_${TS}

echo "[meta] ts=${TS}" | tee "$LOG_ROOT/_meta.log"
echo "[meta] alpha=${ALPHA}" | tee -a "$LOG_ROOT/_meta.log"
echo "[meta] out dpf2=${DPF2_OUT}" | tee -a "$LOG_ROOT/_meta.log"
echo "[meta] out dpf3=${DPF3_OUT}" | tee -a "$LOG_ROOT/_meta.log"
echo "[meta] out diff2=${DIFF2_OUT}" | tee -a "$LOG_ROOT/_meta.log"
echo "[meta] out diff3=${DIFF3_OUT}" | tee -a "$LOG_ROOT/_meta.log"
echo "[meta] out dpf2_resamp=${DPF2_RESAMP_OUT}" | tee -a "$LOG_ROOT/_meta.log"
echo "[meta] out dpf3_resamp=${DPF3_RESAMP_OUT}" | tee -a "$LOG_ROOT/_meta.log"
echo "[meta] out diff2_resamp=${DIFF2_RESAMP_OUT}" | tee -a "$LOG_ROOT/_meta.log"
echo "[meta] out diff3_resamp=${DIFF3_RESAMP_OUT}" | tee -a "$LOG_ROOT/_meta.log"
echo "[meta] plot_root=${PLOT_ROOT}" | tee -a "$LOG_ROOT/_meta.log"

# 1) DPF one-step (2DoF)
(
CUDA_VISIBLE_DEVICES=0 \
SWEEP_SYSTEMS=2dof \
SWEEP_POLICIES=sinusoidal,gp,zero,spline \
SWEEP_LENGTHS_CSV=50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000 \
SWEEP_NUM_SAMPLES=100 \
SWEEP_SEED=10 \
SWEEP_ALPHA_Q=${ALPHA} \
SWEEP_ALPHA_P=${ALPHA} \
SWEEP_OUTPUT_SUBDIR=${DPF2_OUT} \
"$PY" -m scripts.eval.run_dpf_eval_grid
) |& tee "$LOG_ROOT/dpf2.log" &
PID_DPF2=$!

# 2) DPF one-step (3DoF, ckpt0010 default)
(
CUDA_VISIBLE_DEVICES=1 \
SWEEP_SYSTEMS=3dof \
SWEEP_POLICIES=sinusoidal,gp,zero,spline \
SWEEP_LENGTHS_CSV=50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000 \
SWEEP_NUM_SAMPLES=100 \
SWEEP_SEED=10 \
SWEEP_ALPHA_Q=${ALPHA} \
SWEEP_ALPHA_P=${ALPHA} \
SWEEP_OUTPUT_SUBDIR=${DPF3_OUT} \
"$PY" -m scripts.eval.run_dpf_eval_grid
) |& tee "$LOG_ROOT/dpf3.log" &
PID_DPF3=$!

# 3) Diffusion one-step (2DoF)
(
CUDA_VISIBLE_DEVICES=2 \
EVAL_SYSTEMS=2dof \
GUIDANCE_ALPHA_Q=${ALPHA} \
GUIDANCE_ALPHA_P=${ALPHA} \
POLICIES_CSV=sinusoidal,gp,zero,spline \
EVAL_LENGTHS_CSV=50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000 \
EVAL_NUM_SAMPLES=100 \
EVAL_SEED=10 \
FIXED_DIFFUSION_CKPT=${PROJECT_ROOT}/checkpoints/2dof/transformer_diffusion/trajectory_dpf_StateOnlyAdaLN_x0Stabilized\&AbsoluteTimeEncoding\&FixedTrajLength1000\&UniformContext\&EncoderNone\&DecoderAttentions_backbone-transformer:epoch=2999_val_loss:val_loss=0.0009.ckpt \
OUT_ROOT=${DIFF2_OUT} \
PLOTS_ROOT=plots/_tmp_alpha1_diff2_${TS} \
"$PY" -m scripts.eval.run_diffusion_eval_grid
) |& tee "$LOG_ROOT/diff2.log" &
PID_DIFF2=$!

# 4) Diffusion one-step (3DoF, structured-999 HNN)
(
CUDA_VISIBLE_DEVICES=3 \
EVAL_SYSTEMS=3dof \
GUIDANCE_ALPHA_Q=${ALPHA} \
GUIDANCE_ALPHA_P=${ALPHA} \
POLICIES_CSV=sinusoidal,gp,zero,spline \
EVAL_LENGTHS_CSV=50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000 \
EVAL_NUM_SAMPLES=100 \
EVAL_SEED=10 \
FIXED_DIFFUSION_CKPT=${PROJECT_ROOT}/checkpoints/3dof/transformer_diffusion/trajectory_dpf_StateOnlyAdaLN_x0Stabilized\&AbsoluteTimeEncoding\&FixedTrajLength1000\&UniformContext\&EncoderNone\&DecoderAttentions_backbone-transformer:epoch=2999_val_loss:val_loss=0.0008.ckpt \
HNN_CKPT_3DOF=${PROJECT_ROOT}/checkpoints/3dof/hnn/StructuredHNN-dim256-traj40000-epoch-epoch=999.ckpt \
OUT_ROOT=${DIFF3_OUT} \
PLOTS_ROOT=plots/_tmp_alpha1_diff3_${TS} \
"$PY" -m scripts.eval.run_diffusion_eval_grid
) |& tee "$LOG_ROOT/diff3.log" &
PID_DIFF3=$!

echo "[meta] launched pids dpf2=${PID_DPF2} dpf3=${PID_DPF3} diff2=${PID_DIFF2} diff3=${PID_DIFF3}" | tee -a "$LOG_ROOT/_meta.log"

wait "$PID_DPF2"
wait "$PID_DPF3"
wait "$PID_DIFF2"
wait "$PID_DIFF3"

echo "[meta] phase1 one-step runs done; launching resampling m=4" | tee -a "$LOG_ROOT/_meta.log"

# 5) DPF resampling m=4 (2DoF)
(
CUDA_VISIBLE_DEVICES=0 \
GUIDANCE_PRESET=custom \
GUIDANCE_METHOD=strategy1 \
GUIDANCE_NUM_CANDIDATES=4 \
GUIDANCE_ENERGY_MODE=robust_hamres \
GUIDANCE_TRUST_LAMBDA=1e-3 \
GUIDANCE_HAMRES_SMOOTH_SIGMA=0.5 \
GUIDANCE_HAMRES_DELTA=2.0 \
GUIDANCE_HAMRES_MIN_SCALE_Q=1e-3 \
GUIDANCE_HAMRES_MIN_SCALE_P=1e-3 \
SWEEP_SYSTEMS=2dof \
SWEEP_POLICIES=sinusoidal,gp,zero,spline \
SWEEP_LENGTHS_CSV=50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000 \
SWEEP_NUM_SAMPLES=100 \
SWEEP_SEED=10 \
SWEEP_ALPHA_Q=${ALPHA} \
SWEEP_ALPHA_P=${ALPHA} \
SWEEP_OUTPUT_SUBDIR=${DPF2_RESAMP_OUT} \
"$PY" -m scripts.eval.run_dpf_eval_grid
) |& tee "$LOG_ROOT/dpf2_resampling_n4.log" &
PID_DPF2_RESAMP=$!

# 6) DPF resampling m=4 (3DoF)
(
CUDA_VISIBLE_DEVICES=1 \
GUIDANCE_PRESET=custom \
GUIDANCE_METHOD=strategy1 \
GUIDANCE_NUM_CANDIDATES=4 \
GUIDANCE_ENERGY_MODE=robust_hamres \
GUIDANCE_TRUST_LAMBDA=1e-3 \
GUIDANCE_HAMRES_SMOOTH_SIGMA=0.5 \
GUIDANCE_HAMRES_DELTA=2.0 \
GUIDANCE_HAMRES_MIN_SCALE_Q=1e-3 \
GUIDANCE_HAMRES_MIN_SCALE_P=1e-3 \
SWEEP_SYSTEMS=3dof \
SWEEP_POLICIES=sinusoidal,gp,zero,spline \
SWEEP_LENGTHS_CSV=50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000 \
SWEEP_NUM_SAMPLES=100 \
SWEEP_SEED=10 \
SWEEP_ALPHA_Q=${ALPHA} \
SWEEP_ALPHA_P=${ALPHA} \
SWEEP_OUTPUT_SUBDIR=${DPF3_RESAMP_OUT} \
"$PY" -m scripts.eval.run_dpf_eval_grid
) |& tee "$LOG_ROOT/dpf3_resampling_n4.log" &
PID_DPF3_RESAMP=$!

# 7) Diffusion resampling m=4 (2DoF)
(
CUDA_VISIBLE_DEVICES=2 \
EVAL_SYSTEMS=2dof \
GUIDANCE_PRESET=custom \
GUIDANCE_METHOD=strategy1 \
GUIDANCE_NUM_CANDIDATES=4 \
GUIDANCE_ENERGY_MODE=robust_hamres \
GUIDANCE_TRUST_LAMBDA=1e-3 \
GUIDANCE_HAMRES_SMOOTH_SIGMA=0.5 \
GUIDANCE_HAMRES_DELTA=2.0 \
GUIDANCE_HAMRES_MIN_SCALE_Q=1e-3 \
GUIDANCE_HAMRES_MIN_SCALE_P=1e-3 \
GUIDANCE_ALPHA_Q=${ALPHA} \
GUIDANCE_ALPHA_P=${ALPHA} \
POLICIES_CSV=sinusoidal,gp,zero,spline \
EVAL_LENGTHS_CSV=50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000 \
EVAL_NUM_SAMPLES=100 \
EVAL_SEED=10 \
FIXED_DIFFUSION_CKPT=${PROJECT_ROOT}/checkpoints/2dof/transformer_diffusion/trajectory_dpf_StateOnlyAdaLN_x0Stabilized\&AbsoluteTimeEncoding\&FixedTrajLength1000\&UniformContext\&EncoderNone\&DecoderAttentions_backbone-transformer:epoch=2999_val_loss:val_loss=0.0009.ckpt \
OUT_ROOT=${DIFF2_RESAMP_OUT} \
PLOTS_ROOT=plots/_tmp_alpha1_diff2_resampling_n4_${TS} \
"$PY" -m scripts.eval.run_diffusion_eval_grid
) |& tee "$LOG_ROOT/diff2_resampling_n4.log" &
PID_DIFF2_RESAMP=$!

# 8) Diffusion resampling m=4 (3DoF)
(
CUDA_VISIBLE_DEVICES=3 \
EVAL_SYSTEMS=3dof \
GUIDANCE_PRESET=custom \
GUIDANCE_METHOD=strategy1 \
GUIDANCE_NUM_CANDIDATES=4 \
GUIDANCE_ENERGY_MODE=robust_hamres \
GUIDANCE_TRUST_LAMBDA=1e-3 \
GUIDANCE_HAMRES_SMOOTH_SIGMA=0.5 \
GUIDANCE_HAMRES_DELTA=2.0 \
GUIDANCE_HAMRES_MIN_SCALE_Q=1e-3 \
GUIDANCE_HAMRES_MIN_SCALE_P=1e-3 \
GUIDANCE_ALPHA_Q=${ALPHA} \
GUIDANCE_ALPHA_P=${ALPHA} \
POLICIES_CSV=sinusoidal,gp,zero,spline \
EVAL_LENGTHS_CSV=50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000 \
EVAL_NUM_SAMPLES=100 \
EVAL_SEED=10 \
FIXED_DIFFUSION_CKPT=${PROJECT_ROOT}/checkpoints/3dof/transformer_diffusion/trajectory_dpf_StateOnlyAdaLN_x0Stabilized\&AbsoluteTimeEncoding\&FixedTrajLength1000\&UniformContext\&EncoderNone\&DecoderAttentions_backbone-transformer:epoch=2999_val_loss:val_loss=0.0008.ckpt \
HNN_CKPT_3DOF=${PROJECT_ROOT}/checkpoints/3dof/hnn/StructuredHNN-dim256-traj40000-epoch-epoch=999.ckpt \
OUT_ROOT=${DIFF3_RESAMP_OUT} \
PLOTS_ROOT=plots/_tmp_alpha1_diff3_resampling_n4_${TS} \
"$PY" -m scripts.eval.run_diffusion_eval_grid
) |& tee "$LOG_ROOT/diff3_resampling_n4.log" &
PID_DIFF3_RESAMP=$!

echo "[meta] launched pids dpf2_resamp=${PID_DPF2_RESAMP} dpf3_resamp=${PID_DPF3_RESAMP} diff2_resamp=${PID_DIFF2_RESAMP} diff3_resamp=${PID_DIFF3_RESAMP}" | tee -a "$LOG_ROOT/_meta.log"

wait "$PID_DPF2_RESAMP"
wait "$PID_DPF3_RESAMP"
wait "$PID_DIFF2_RESAMP"
wait "$PID_DIFF3_RESAMP"

echo "[meta] all 8 runs done; plotting" | tee -a "$LOG_ROOT/_meta.log"

"$PY" -m scripts.plot.plot_7method_rmse_boxplots \
  --dpf-onestep-2dof-root output/${DPF2_OUT} \
  --dpf-onestep-2dof-hamres output/${DPF2_OUT}/metrics/metrics_per_sample.csv \
  --dpf-onestep-3dof-root output/${DPF3_OUT} \
  --dpf-onestep-3dof-hamres output/${DPF3_OUT}/metrics/metrics_per_sample.csv \
  --dpf-resampling-2dof-root output/${DPF2_RESAMP_OUT} \
  --dpf-resampling-3dof-root output/${DPF3_RESAMP_OUT} \
  --diff-onestep-2dof output/${DIFF2_OUT}/metrics/metrics_per_sample_rmse_hamres.csv \
  --diff-onestep-3dof output/${DIFF3_OUT}/metrics/metrics_per_sample_rmse_hamres.csv \
  --diff-resampling-2dof output/${DIFF2_RESAMP_OUT}/metrics/metrics_per_sample_rmse_hamres.csv \
  --diff-resampling-3dof output/${DIFF3_RESAMP_OUT}/metrics/metrics_per_sample_rmse_hamres.csv \
  --out-dir "${PLOT_ROOT}" |& tee "$LOG_ROOT/plot7_rmse.log"

echo "[done] alpha=1 parallel regen pipeline completed (${TS})" | tee -a "$LOG_ROOT/_meta.log"
