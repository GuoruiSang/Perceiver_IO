# Reacher Target-Guidance Sweep (2026-04-06)

This note records the target-guidance sweep run on the capped reacher-task benchmark with HNN guidance disabled.

## Goal

Test whether end-effector-space target guidance improves the closed-loop reacher task, and tune `target_guidance_alpha` on a fixed benchmark.

## Checkpoint And Data

- DPF checkpoint:
  `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/dpf_exploration_iid_uniform_len1000_v1/trajectory_dpf_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions_cond-concat_torque_in_state_layout-shifted_tau_qpos-raw:epoch=2999_val_loss:val_loss=0.0983.ckpt`
- Validation HDF5:
  `/home/gsang/Projects/hnn_guided_dpf/data/reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/val_traj_4000-steps_1000.h5`

## Benchmark Definition

Task mode:
- `validation_random_source_random_target_across_trajs`

Fixed task seeds:
- `0,1,2,3,4`

Fixed rollout / sampling settings:
- `num_candidates = 8`
- `max_sampling_retries = 3`
- `retry_improvement_margin = 0.001`
- `num_diffusion_steps = 20`
- `lookahead_steps = 64`
- `recent_prefix_cap = 64`
- `goal_tolerance = 0.01`
- `max_rollout_steps = 200`
- `random_future_target_min_initial_distance = 0.1`
- `reset_window_time_indices = false`

Target-guidance settings held fixed:
- `target_guidance_time_power = 2.0`
- `target_guidance_normalize_grad = true`

HNN guidance:
- disabled

Primary ranking metric:
- aggregate `best_goal_distance.mean`

Secondary metrics:
- aggregate `final_goal_distance.mean`
- `success_rate`
- replay-space `qpos` / `mom` MSE

## Sweep Results

Summary JSONs:
- Unguided:
  `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_unguided_acrosstraj_cap200_pilot5/reacher_goal_prefix_expansion_summary.json`
- `alpha = 1e-4`:
  `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_target_guidance_acrosstraj_cap200_alpha1e4_pilot5/reacher_goal_prefix_expansion_summary.json`
- `alpha = 1e-3`:
  `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_target_guidance_acrosstraj_cap200_alpha1e3_pilot5/reacher_goal_prefix_expansion_summary.json`
- `alpha = 3e-3`:
  `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_target_guidance_acrosstraj_cap200_alpha3e3_pilot5/reacher_goal_prefix_expansion_summary.json`
- `alpha = 1e-2`:
  `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_target_guidance_acrosstraj_cap200_pilot5/reacher_goal_prefix_expansion_summary.json`
- `alpha = 3e-2`:
  `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_target_guidance_acrosstraj_cap200_alpha3e2_pilot5/reacher_goal_prefix_expansion_summary.json`

Ranked by `best_goal_distance.mean` (lower is better):

| Setting | Success Rate | Best Goal Dist Mean | Final Goal Dist Mean | qpos MSE Mean | mom MSE Mean |
|---|---:|---:|---:|---:|---:|
| `alpha = 3e-3` | `0.0` | `0.2072846930` | `0.2072846930` | `0.0011289191` | `0.2518800450` |
| `alpha = 1e-2` | `0.0` | `0.2098653370` | `0.2098653370` | `0.0010696267` | `0.2283572072` |
| `alpha = 1e-3` | `0.0` | `0.2113578593` | `0.2121002694` | `0.0015416926` | `0.2790872854` |
| `alpha = 1e-4` | `0.0` | `0.2123482514` | `0.2128547397` | `0.0015181028` | `0.2530902678` |
| `unguided` | `0.0` | `0.2128156200` | `0.2134362103` | `0.0013486333` | `0.2326956247` |
| `alpha = 3e-2` | `0.0` | `0.2130400356` | `0.2130400356` | `0.0009345926` | `0.1736016388` |

## Best Recorded Config

Best config on this benchmark:
- `target_guidance_alpha = 3e-3`
- `target_guidance_time_power = 2.0`
- `target_guidance_normalize_grad = true`

Interpretation:
- Target guidance still did **not** achieve any task successes on this capped 5-task benchmark.
- But `alpha = 3e-3` gave the best mean closest-approach distance among the tested settings.
- The improvement over unguided is modest:
  - unguided `best_goal_distance.mean = 0.2128156200`
  - best target-guided `best_goal_distance.mean = 0.2072846930`

## Takeaway

Target guidance helps a bit on this capped benchmark when tuned, but the gain is modest and does not turn failures into successes. The best tested setting is `alpha = 3e-3`, not `1e-2`.
