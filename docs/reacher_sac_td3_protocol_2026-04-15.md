# Reacher SAC/TD3 Comparison Protocol

This note records the evaluation protocol we will use when comparing online RL baselines such as SAC and TD3 against the existing DPF/HNN/MuJoCo reacher experiments.

## Goal

Make the comparison fair at the task-evaluation level, even though SAC/TD3 do not use the same internal planning mechanics as the MPC-style methods.

## Fixed Benchmark

- Task mode: `validation_random_source_random_target_across_trajs`
- Fixed task ids: `0,1,2,3,4,5,6,7,8,9`
- Validation dataset:
  `/home/gsang/Projects/hnn_guided_dpf/data/reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/val_traj_4000-steps_1000.h5`
- Minimum initial source-target distance: `0.1 m`
- Cross-trajectory sampling max tries: `128`
- Goal tolerance: `0.01 m`

## Main Evaluation Rule

For SAC and TD3, do not force the controller into the DPF/HNN internal rollout budget such as `lookahead_steps=256`.

Instead, evaluate all methods with the same task-level stopping rule:

- stop immediately on success
- otherwise stop after `500` consecutive executed steps without decreasing goal distance
- use a large hard safety cap for rollout length

The SAC baseline implementation added on April 15, 2026 uses:

- training episode cap: `1000` steps
- evaluation hard cap: `5000` steps
- stall patience: `500` steps

This matches the intent of the current reset-time-index closed-loop reacher experiments: a controller is allowed to keep acting until it succeeds or clearly stalls.

## Result Docs And Runtime Bookkeeping

Finished result docs for this protocol are:

- `/home/gsang/Projects/hnn_guided_dpf/docs/reacher_sac_td3_results_2026-04-16.md`
- `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sac_tune_2026-04-15/sac_tune_summary.md`
- `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sb3_auto_pipeline_2026-04-15/td3_tune/td3_tune_summary.md`

Runtime columns should be interpreted as:

- `wall_clock_to_eval_start_s`: elapsed trainer time right before the fixed-task evaluation started
- `total_wall_clock_s`: total trainer wall-clock from launch to completion, when a saved trainer log is available

The RL evaluator has also been updated so future runs will save `task_wall_clock_seconds` per source-target evaluation task directly.

## Reported Metrics

Main metrics:

- `success_rate`
- `best_goal_mean`
- `final_goal_mean`

Also report budget-truncated metrics for every method:

- `success_rate@256`, `best_goal_mean@256`
- `success_rate@512`, `best_goal_mean@512`
- `success_rate@1000`, `best_goal_mean@1000`

This gives both:

- a full-task comparison
- a rollout-budget-matched comparison

## Shared RL Utilities To Keep

Training split that the future SAC/TD3 baselines should use:

- `/home/gsang/Projects/hnn_guided_dpf/data/reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/train_traj_40000-steps_1000.h5`

Shared env/eval utilities kept for the future SB3 baselines:

- `/home/gsang/Projects/hnn_guided_dpf/src/rl/reacher_goal_rl.py`
- `/home/gsang/Projects/hnn_guided_dpf/scripts/train/train_reacher_sb3.py`
- `/home/gsang/Projects/hnn_guided_dpf/scripts/eval/eval_reacher_sb3_policy.py`

Installed library versions in the active `perceiver` environment:

- `stable-baselines3==2.3.2`
- `gymnasium==0.29.1`
- `torch==2.0.1+cu117`

Observation used by the goal-conditioned RL environment:

- `qpos_raw[2]`
- `mom[2]`
- current end-effector position `ee_xy[2]`
- target position `goal_xy[2]`
- target delta `goal_xy - ee_xy[2]`

Action used by the RL environment:

- 2D torque clipped to `[-0.2, 0.2]`

Default reward kept as the current starting point:

- `10.0 * (prev_goal_distance - next_goal_distance)`
- `-1.0 * next_goal_distance`
- `-0.01 * mean(action^2)`
- `+5.0` on success

Planned starting point for the first SB3 SAC/TD3 runs:

- total env steps: `300000`
- replay / buffer size: `500000`
- batch size: `256`
- training episode cap: `1000`
- evaluation hard cap: `5000`
- stall patience: `500`
- action range: `[-0.2, 0.2]`
- same fixed 10-task benchmark as above

## Important Framing

The RL baselines answer a different question than the DPF/HNN methods.

- DPF/HNN: can a trajectory model or learned dynamics model solve source-target reaching through model-based candidate selection?
- SAC/TD3: can an online-trained goal-conditioned policy solve the same source-target reaching benchmark directly?

That makes SAC/TD3 useful comparison baselines, but they should be labeled as online RL controllers rather than diffusion-model ablations.

## Note On Implementation Choice

We intentionally do not keep the earlier lightweight in-repo SAC implementation as the main baseline.

Reason:

- for serious comparison, SAC and TD3 should be run from a standard library implementation such as `stable-baselines3`
- the kept `reacher_goal_rl.py` module is meant to provide the custom environment and fixed-task evaluator that the standard library trainer can plug into
