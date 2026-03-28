# Reacher Exploration IID Evaluation Settings

This note records the final settings for the exploration-trained IID DPF checkpoint evaluation that we kept after trimming the older rollout modes.

## Scope

- Checkpoint family: exploration-trained unconditional DPF
- Kept evaluation mode: `validation_random_source_random_future_target_in_traj`
- Date: `2026-03-28`

## Checkpoint

- Checkpoint path:
  `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/dpf_exploration_iid_uniform_len1000_v1/trajectory_dpf_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions_cond-concat_torque_in_state_layout-shifted_tau_qpos-raw:epoch=2999_val_loss:val_loss=0.0983.ckpt`

## Training Dataset

- Train HDF5:
  `/home/gsang/Projects/hnn_guided_dpf/data/reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/train_traj_40000-steps_1000.h5`
- Val HDF5:
  `/home/gsang/Projects/hnn_guided_dpf/data/reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/val_traj_4000-steps_1000.h5`

Dataset generation properties:

- Torque mode: `iid_uniform`
- Train trajectories: `40000`
- Validation trajectories: `4000`
- Trajectory length: `1000`
- `dt = 0.001`
- Torque scale: `0.2`
- `qvel_scale = 0.8`
- `max_abs_qvel = 4*pi = 12.566370614359172`
- `max_abs_qacc = 10000.0`

## Model / Training Config

These are the key checkpoint training settings taken from the saved W&B config.

- Backbone: `perceiverio`
- Conditioning mode: `concat_torque_in_state`
- Query/context mode: `clean_prefix_noisy_suffix`
- Token layout: `shifted_tau`
- `qpos_representation = raw`
- `qpos_representation_override = raw`
- Diffusion steps: `1000`
- Decoder blocks: `4`
- Num latents: `256`
- Num latent channels: `256`
- Batch size: `160`
- Epochs: `3000`
- Learning rate: `1e-4`
- Num workers: `16`
- `use_ema = true`
- W&B trajectory image logging: enabled

W&B run name:

- `ReacherExploration-iid_uniform-DPF-rawQpos-concatTau-cleanPrefix-shiftedTau-bs160-len1000`

## Kept Evaluation Mode

Script:

- `/home/gsang/Projects/hnn_guided_dpf/scripts/data/run_reacher_goal_prefix_expansion.py`

Only supported task mode now:

- `validation_random_source_random_future_target_in_traj`

Task construction:

1. Pick a validation trajectory by ID.
2. Randomly sample `t_source`.
3. Randomly sample `t_target > t_source` from the same trajectory.
4. Source state is `traj[t_source]`.
5. Goal is the end-effector position at `traj[t_target]`.
6. Enforce a minimum initial source-to-target end-effector distance.

For the main kept benchmark, the minimum initial distance was:

- `0.1 m`

## Main Evaluation Run

Output directory:

- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_iid_ckpt_validation_random_source_random_future_d10cm_10tests`

Summary file:

- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_iid_ckpt_validation_random_source_random_future_d10cm_10tests/reacher_goal_prefix_expansion_summary.json`

Task IDs used:

- `5, 8, 9, 10, 11, 12, 13, 20, 21, 22`

Rollout settings:

- Device: `cuda:5`
- Seed: `0`
- `num_candidates = 8`
- `max_sampling_retries = 3`
- `retry_improvement_margin = 0.001`
- `num_diffusion_steps = 20`
- `lookahead_steps = 64`
- `recent_prefix_cap = 64`
- `goal_tolerance = 0.01`
- `max_prefix_len = 1000`
- `random_future_target_min_initial_distance = 0.1`

Interpretation of `max_prefix_len = 1000`:

- The rollout is allowed to use the full trajectory-length budget `L`.

## Main Result

Aggregate result from the kept 10-task run:

- Success rate: `0.6`
- Mean best goal distance: `0.012289775265395968 m`
- Median best goal distance: `0.00995823179744547 m`
- Best case: `0.009807686361098722 m`
- Worst best distance: `0.023101249465929147 m`
- Mean final goal distance: `0.07002997068103758 m`

Useful reading:

- The model solved `6/10` tasks at `1 cm` tolerance.
- Median best distance was already below the success threshold.
- Some failed runs drifted badly by the end, which is why final-distance statistics are worse than best-distance statistics.

## Kept Result Folders

We kept only the result folders for the retained task mode:

- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_iid_ckpt_validation_random_source_random_future_quick`
- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_iid_ckpt_validation_random_source_random_future_d10cm_5tests`
- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_iid_ckpt_validation_random_source_random_future_d10cm_10tests`

Older rollout-mode result folders were deleted after the cleanup.
