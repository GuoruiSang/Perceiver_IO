# Scripts Guide

Top-level `scripts/` keeps active grouped subdirectories plus shared-import modules.

## Main entrypoints
- `scripts/eval/run_7method_comparison_pipeline.sh`
  - full active experiment eval pipeline; use this after checkpoints are ready.
- `scripts/eval/run_dpf_eval_grid.py`
  - DPF-only eval runner across the active policy/length/system grid; used for one-step or resampling depending env guidance settings.
- `scripts/eval/run_diffusion_eval_grid.py`
  - diffusion eval runner across the active policy/length/system grid; used for one-step or resampling depending env guidance settings.
- `scripts/plot/plot_7method_rmse_boxplots.py`
  - final 7-method figure builder from saved eval outputs.
- `scripts/train/generate_2dof_3dof_datasets.sh`
  - regenerate the paired 2DoF and 3DoF datasets used by training.
- `scripts/train/train_2dof_dpf_hnn_models.sh`
  - train the 2DoF DPF and 2DoF structured HNN checkpoints.
- `scripts/train/train_3dof_dpf_hnn_models.sh`
  - train the 3DoF DPF and 3DoF structured HNN checkpoints.
- `scripts/train/train_transformer_diffusion_fixed_length.sh`
  - train one transformer-diffusion checkpoint for a chosen system; also use this for fixed-budget runs.

## Grouped layout
- `scripts/eval/`
  - active evaluation runners plus small helper entrypoints.
- `scripts/plot/`
  - active plotting scripts.
- `scripts/train/`
  - minimal dataset/training launchers required to regenerate the final checkpoints.
- `scripts/data/`
  - dataset loading/generation utilities.

## Shared modules kept at root
- `scripts/system_eval_utils.py`
  - shared system configs, model loading, torque loading, and RMSE/HamRes helpers.
- `scripts/diffusion_eval_shared.py`
  - shared diffusion-vs-DPF/HNN evaluation logic for 2DoF and 3DoF runners.
- `scripts/guidance_eval_config.py`
  - shared guidance presets and env-driven evaluation config.
- `scripts/guidance_sampling_utils.py`
  - shared sampling/model helpers used by the active eval runners.

## Notes
- The active reproducibility contract is the alpha=1 7-method workflow plus regeneration of the required datasets and retraining of the six final checkpoints.
- Older alternate searches, reports, benchmarks, and Reacher scripts remain outside that contract.
- `scripts/system_eval_utils.py` is the shared helper layer for active evaluation code.
- Output root is `output/`; active evaluation pipelines pass `eval_runs/...` subpaths that are resolved under it.
