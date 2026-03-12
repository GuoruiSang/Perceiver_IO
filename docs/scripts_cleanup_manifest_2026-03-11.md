# Scripts Cleanup Manifest

Date: 2026-03-11
Updated: 2026-03-11

## Active cleanup contract

Keep anything required to:
- regenerate the required datasets
- retrain the active checkpoints used by the experiment
- rerun the active 7-method comparison pipeline
- reproduce the 7-method final metrics/content described in `docs/exp_7methods_plot_settings_2026-03-05.md`

Status meanings:
- `keep`: required by the active experiment contract
- `removed`: outside the contract and deleted

## keep

Active 7-method comparison path:
- `scripts/system_eval_utils.py`
- `scripts/diffusion_eval_shared.py`
- `scripts/guidance_eval_config.py`
- `scripts/guidance_sampling_utils.py`
- `scripts/eval/run_dpf_eval_grid.py`
- `scripts/eval/run_diffusion_eval_grid.py`
- `scripts/eval/run_7method_comparison_pipeline.sh`
- `scripts/plot/plot_7method_rmse_boxplots.py`

Dataset generation and loading:
- `scripts/data/dataset.py`
- `scripts/data/generate_dataset_forward.py`
- `scripts/train/generate_2dof_3dof_datasets.sh`

Training launchers:
- `scripts/train/train_2dof_dpf_hnn_models.sh`
- `scripts/train/train_3dof_dpf_hnn_models.sh`
- `scripts/train/train_transformer_diffusion_fixed_length.sh`

Training/reference docs:
- `docs/exp_7methods_plot_settings_2026-03-05.md`
- `docs/fixed_length_2dof_training.md`
- `docs/fixed_length_3dof_training.md`
- `docs/model_StructuredHNN_2026-02-06.md`
- `docs/transformer_baseline_architecture.md`

Required training/eval inputs:
- `data/2dof/traj_40000-steps_4000.h5`
- `data/2dof/traj_2000-steps_4000.h5`
- `data/3dof/traj_40000-steps_4000.h5`
- `data/3dof/traj_2000-steps_4000.h5`
- `data/2dof/sinusoidal/sinusoidal_torques_2000_L1500.h5`
- `data/2dof/gp/gp_torques_1000_L1500_intermediate_v1.h5`
- `data/2dof/spline/spline_torques_1000_L1500_intermediate_v1.h5`
- `data/3dof/sinusoidal/sinusoidal_torques_1000_L1500.h5`
- `data/3dof/gp/gp_torques_1000_L1500.h5`
- `data/3dof/spline/spline_torques_1000_L1500.h5`
- `configs/rigid_arm_hinge_2dof.xml`
- `configs/rigid_arm_hinge.xml`

Active checkpoint targets referenced by code/docs:
- `checkpoints/2dof/dpf/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0008.ckpt`
- `checkpoints/2dof/hnn/StructuredHNN-2DOF-epoch-epoch=999.ckpt`
- `checkpoints/2dof/transformer_diffusion/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&FixedTrajLength1000&UniformContext&EncoderNone&DecoderAttentions_backbone-transformer:epoch=2999_val_loss:val_loss=0.0009.ckpt`
- `checkpoints/3dof/dpf/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt`
- `checkpoints/3dof/hnn/StructuredHNN-dim256-traj40000-epoch-epoch=999.ckpt`
- `checkpoints/3dof/transformer_diffusion/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&FixedTrajLength1000&UniformContext&EncoderNone&DecoderAttentions_backbone-transformer:epoch=2999_val_loss:val_loss=0.0008.ckpt`

Active HNN training defaults:
- `2DoF HNN`: `structured`, train on `data/2dof/traj_40000-steps_4000.h5`, validate on `data/2dof/traj_2000-steps_4000.h5`
- `3DoF HNN`: `structured`, train on `data/3dof/traj_40000-steps_4000.h5`, validate on `data/3dof/traj_2000-steps_4000.h5`

Active dataset-generation contract:
- `2DoF`: generate `traj_40000-steps_4000.h5` and `traj_2000-steps_4000.h5` together from `configs/rigid_arm_hinge_2dof.xml` with `torque_policies=sinusoidal:1.0`
- `3DoF`: generate `traj_40000-steps_4000.h5` and `traj_2000-steps_4000.h5` together under `data/3dof/` from `configs/rigid_arm_hinge.xml` with `torque_policies=sinusoidal:1.0`

Kept experiment artifacts:
- `output/eval_runs/`
- `plots/method_comparison_7methods_rmse_alpha1_20260305_2320`
- `plots/method_comparison_7methods`
- `plots/method_comparison_7methods/tables/tables_non_sin_newcfg_onestep_20260305_values.csv`
- `plots/method_comparison_7methods/tables/tables_non_sin_newcfg_onestep_20260305.tex`
- `logs/alpha1_regen_parallel_20260305_2320`
- `configs/reacher_non_diss_from_dataset.xml`

## removed

Deleted because they were outside the active experiment contract:
- benchmark/speed scripts
- alternate alpha-search and quick-search scripts
- extra transformer/report pipeline scripts not used by the active path
- smoothing ablation script
- Reacher, shifted-task, boundary, inpaint, and suffix-overlap scripts
- `scripts/__pycache__/`
- non-final analysis CSV summaries under the old `analysis/` tree

## Notes

- Matching metrics/content is the validation target; timestamped output directories may differ.
- The active experiment contract now uses matched-data structured HNN checkpoints for both 2DoF and 3DoF.
- The old historical spec was intentionally overwritten to match the active contract.
- Final table artifacts were moved under `plots/method_comparison_7methods/tables/`.
- Active figures are PNG-only.
- Active evaluation metrics are RMSE+HamRes; normalized error metrics are no longer part of the contract.
- The active comparison pipeline now regenerates both one-step and resampling (`m=4`) outputs for all 7 plotted methods.
- `alpha=1` refers to the one-step guidance setting used by the active comparison.
- `scripts/system_eval_utils.py` now holds the shared system/model/metric helpers used by the active scripts.
