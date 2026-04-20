# Reacher DPF Training Package 2026-04-20

## Dataset Location

The Reacher DPF train/validation dataset has been copied to the mounted data drive:

`/Data/gsang/hnn_guided_dpf/reacher_dpf/reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1`

Note: this machine exposes the data drive as `/Data` with a capital `D`; lowercase `/data` is not mounted here.

## Files

- `train_traj_40000-steps_1000.h5`
- `val_traj_4000-steps_1000.h5`
- `build_report.json`

The copied dataset size is about `2.7G`.

## Shared Pretrained Checkpoints

For immediate inference without retraining, the trained Reacher DPF checkpoint is available at:

`/Data/gsang/hnn_guided_dpf/reacher_dpf/checkpoints/dpf_exploration_iid_uniform_len1000_v1/trajectory_dpf_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions_cond-concat_torque_in_state_layout-shifted_tau_qpos-raw:epoch=2999_val_loss:val_loss=0.0983.ckpt`

The trained structured Reacher HNN checkpoint for HNN-guided DPF sampling is available at:

`/Data/gsang/hnn_guided_dpf/reacher_dpf/checkpoints/hnn_exploration_iid_uniform_len1000_v1/StructuredHNN-ReacherExploration-IID-epoch-epoch=999.ckpt`

## Training / Inference Notebook

`notebooks/reacher_dpf_training_inference.ipynb`

The notebook includes:

- dataset path and H5 structure checks
- embedded Reacher-only DPF dataset/model/training/sampling code, without importing local `src.*` or `scripts.*` modules
- embedded structured HNN model/training/guidance code, without importing local `src.*` or `scripts.*` modules
- self-contained DPF and HNN training code, without launching repository training scripts
- checkpoint loading from either user-local trained checkpoints or the shared pretrained checkpoints
- small prefix-completion sampling example
- unguided, HNN-guided, target-guided, and combined target/HNN DPF candidate inference against a Reacher target, including both HNN-then-target and target-then-HNN guidance orderings

## Smoke Test Status

Latest smoke test on 2026-04-20 passed:

- notebook JSON and all code cells compile
- shared train/validation H5 paths exist and expose the expected Reacher trajectory fields
- shared DPF and HNN checkpoints load successfully
- tiny DPF and HNN training-step/backward passes run on temporary mini H5 files
- all inference presets run: unguided, HNN, target L1/L2/Linf, HNN-then-target, and target-then-HNN
- best-candidate plotting writes an output PNG successfully
