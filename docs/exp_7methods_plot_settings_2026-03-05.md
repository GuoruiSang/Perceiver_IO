# 7-Method RMSE Plot Settings (Active comparison contract)

Date: 2026-03-05
Updated: 2026-03-11

This file now defines the active 7-method experiment contract.

Canonical eval entrypoint:
- `scripts/eval/run_7method_comparison_pipeline.sh`

The active contract uses:
- one-step guidance with `alpha=1` and `guidance_normalize_grad=True`
- matched-data structured HNN checkpoints for both 2DoF and 3DoF
- the checkpoint layout `checkpoints/<dof>/{dpf,hnn,transformer_diffusion}/...`
- per-DoF train/val datasets generated together from one dataset-generation command
- RMSE + HamRes as the active metric set
- PNG-only figures

## Methods and order

1. Unguided DPF
2. Guided DPF (Gradient / one-step)
3. Guided DPF (Resampling, m=4)
4. HNN rollout
5. Unguided Diffusion
6. Guided Diffusion (Gradient / one-step)
7. Guided Diffusion (Resampling, m=4)

## Required outputs

- 7-method RMSE plots:
  - `plots/method_comparison_7methods_rmse_alpha1_<timestamp>`
- One-step rerun logs:
  - `logs/alpha1_regen_parallel_<timestamp>`

Here `alpha=1` is the guidance setting used by the active comparison, not the pipeline name.
- Updated non-sinusoidal table values:
  - `plots/method_comparison_7methods/tables/tables_non_sin_newcfg_onestep_20260305_values.csv`
  - `plots/method_comparison_7methods/tables/tables_non_sin_newcfg_onestep_20260305.tex`

## Final data sources for plots/tables

### DPF one-step (alpha=1)

- 2DoF:
  - `output/eval_runs/default_full_sweep_alpha1_dpf2_<timestamp>`
  - `output/eval_runs/default_full_sweep_alpha1_dpf2_<timestamp>/metrics/metrics_per_sample.csv`
  - `output/eval_runs/default_full_sweep_alpha1_dpf2_<timestamp>/metrics/rmse_range_per_sample_Lle1000.csv`
- 3DoF:
  - `output/eval_runs/default_full_sweep_3dof_ckpt0010_alpha1_dpf3_<timestamp>`
  - `output/eval_runs/default_full_sweep_3dof_ckpt0010_alpha1_dpf3_<timestamp>/metrics/metrics_per_sample.csv`
  - `output/eval_runs/default_full_sweep_3dof_ckpt0010_alpha1_dpf3_<timestamp>/metrics/rmse_range_per_sample_Lle1000.csv`

### DPF resampling (strategy1, m=4)

- 2DoF:
  - `output/eval_runs/resampling_strategy1_n4_alpha1_dpf2_<timestamp>`
  - `output/eval_runs/resampling_strategy1_n4_alpha1_dpf2_<timestamp>/metrics/metrics_per_sample.csv`
  - `output/eval_runs/resampling_strategy1_n4_alpha1_dpf2_<timestamp>/metrics/rmse_range_per_sample_Lle1000.csv`
- 3DoF:
  - `output/eval_runs/resampling_strategy1_n4_alpha1_dpf3_<timestamp>`
  - `output/eval_runs/resampling_strategy1_n4_alpha1_dpf3_<timestamp>/metrics/metrics_per_sample.csv`
  - `output/eval_runs/resampling_strategy1_n4_alpha1_dpf3_<timestamp>/metrics/rmse_range_per_sample_Lle1000.csv`

### Diffusion one-step (alpha=1)

- 2DoF:
  - `output/eval_runs/2dof_transformer_eval_with_hnn_alpha1_<timestamp>/metrics/metrics_per_sample_rmse_hamres.csv`
- 3DoF:
  - `output/eval_runs/3dof_transformer_eval_with_hnn_alpha1_<timestamp>/metrics/metrics_per_sample_rmse_hamres.csv`

### Diffusion resampling (strategy1, m=4)

- 2DoF:
  - `output/eval_runs/2dof_transformer_eval_with_hnn_resampling_n4_alpha1_<timestamp>/metrics/metrics_per_sample_rmse_hamres.csv`
- 3DoF:
  - `output/eval_runs/3dof_transformer_eval_with_hnn_resampling_n4_alpha1_<timestamp>/metrics/metrics_per_sample_rmse_hamres.csv`

## Final guidance settings

- One-step gradient guidance:
  - `guidance_method = strategy2`
  - `guidance_energy_mode = one_step`
  - `guidance_num_candidates = 16`
  - `guidance_normalize_grad = True`
  - `alpha_q = alpha_p = 1.0` for:
    - 2DoF DPF
    - 3DoF DPF
    - 2DoF Diffusion
    - 3DoF Diffusion

- Resampling guidance:
  - `guidance_method = strategy1`
  - `guidance_energy_mode = robust_hamres`
  - `guidance_num_candidates = 4`
  - `guidance_trust_lambda = 1e-3`
  - `guidance_hamres_smooth_sigma = 0.5`
  - `guidance_hamres_delta = 2.0`
  - `guidance_hamres_min_scale_q = guidance_hamres_min_scale_p = 1e-3`

## Active checkpoints

- DPF 2DoF:
  - `checkpoints/2dof/dpf/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0008.ckpt`
- DPF 3DoF:
  - `checkpoints/3dof/dpf/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt`
- Diffusion 2DoF:
  - `checkpoints/2dof/transformer_diffusion/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&FixedTrajLength1000&UniformContext&EncoderNone&DecoderAttentions_backbone-transformer:epoch=2999_val_loss:val_loss=0.0009.ckpt`
- Diffusion 3DoF:
  - `checkpoints/3dof/transformer_diffusion/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&FixedTrajLength1000&UniformContext&EncoderNone&DecoderAttentions_backbone-transformer:epoch=2999_val_loss:val_loss=0.0008.ckpt`
- HNN 2DoF:
  - `checkpoints/2dof/hnn/StructuredHNN-2DOF-epoch-epoch=999.ckpt`
- HNN 3DoF:
  - `checkpoints/3dof/hnn/StructuredHNN-dim256-traj40000-epoch-epoch=999.ckpt`

## Training contract for HNN checkpoints

- 2DoF structured HNN:
  - train file: `data/2dof/traj_40000-steps_4000.h5`
  - val file: `data/2dof/traj_2000-steps_4000.h5`
- 3DoF structured HNN:
  - train file: `data/3dof/traj_40000-steps_4000.h5`
  - val file: `data/3dof/traj_2000-steps_4000.h5`

## Dataset generation contract

- 2DoF dataset generation:
  - `xml_path = configs/rigid_arm_hinge_2dof.xml`
  - `torque_policies = sinusoidal:1.0`
  - `num_trajectories = 40000`
  - `num_val = 2000`
  - `num_steps = 4000`
- 3DoF dataset generation:
  - `xml_path = configs/rigid_arm_hinge.xml`
  - `torque_policies = sinusoidal:1.0`
  - `num_trajectories = 40000`
  - `num_val = 2000`
  - `num_steps = 4000`

For each DoF, do not mix train/val files generated from different runs.
