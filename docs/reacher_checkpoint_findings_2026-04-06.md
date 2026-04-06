# Reacher Checkpoint Findings

Date:

- `2026-04-06`

## Scope

This note summarizes the main discoveries from the recent Reacher DPF evaluation work:

- comparison of three checkpoints
- clarification of the correct train-matched sampling setup
- effect of prefix length and rebasing
- closed-loop reacher-task behavior
- HNN guidance and target guidance results

## Checkpoints

The three checkpoints compared most often were:

1. IID exploration DPF
   `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/dpf_exploration_iid_uniform_len1000_v1/trajectory_dpf_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions_cond-concat_torque_in_state_layout-shifted_tau_qpos-raw:epoch=2999_val_loss:val_loss=0.0983.ckpt`
2. Smooth-random exploration DPF
   `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/dpf_exploration_smooth_random_len1000_v1/trajectory_dpf_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions_cond-concat_torque_in_state_layout-shifted_tau_qpos-raw:epoch=2999_val_loss:val_loss=0.0017.ckpt`
3. Bidirectional DPF
   `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/dpf/trajectory_dpf_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions_cond-concat_torque_in_state_layout-shifted_tau_qpos-raw:epoch=2999_val_loss:val_loss=0.0012.ckpt`

## Main Discovery

The most important correction was a train-test mismatch in how sampling was being interpreted.

Training used:

- `query_context_mode = clean_prefix_noisy_suffix`
- full query trajectory
- clean observed prefix inside the query
- context as a subset of the query

So the default sampling interpretation should be:

- `train_matched_completion`

not the earlier stricter interpretation that behaved like prefix-only completion.

This was updated in:

- `/home/gsang/Projects/hnn_guided_dpf/src/models/trajectory_dpf_sampling.py`

## Callback-Like Reproduction

To understand why some offline samples looked worse than W&B logs, the callback-like sampling path was reproduced.

Relevant assets:

- callback-like reproduction: [callback_like_traj0_prefix500_comparison.jpg](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_bidirectional_callback_repro/callback_like_traj0_prefix500_comparison.jpg)
- callback reproduction summary: [summary.json](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_bidirectional_callback_repro/summary.json)
- original W&B image: [sampled_trajectory_40798_bc05fbce56a7fa8b929e.jpg](/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/dpf/wandb/run-20260316_230030-viqyr8tk/files/media/images/sampled_trajectory_40798_bc05fbce56a7fa8b929e.jpg)

Takeaway:

- the W&B examples were not directly comparable to the earlier fixed-prefix offline renders
- W&B logging used callback-like settings such as a long prefix and `100` diffusion steps
- once matched more carefully, some of the apparent gap was explained by setup differences

## Three-Checkpoint Callback-Like Comparison

A direct callback-like comparison was rendered for the three checkpoints using:

- `traj_0`
- `prefix_len = 500`
- `num_diffusion_steps = 100`

Results:

- summary: [summary.json](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_callback_like/summary.json)
- IID image: [iid_uniform_traj0_prefix500_callback_like.jpg](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_callback_like/iid_uniform_traj0_prefix500_callback_like.jpg)
- smooth-random image: [smooth_random_traj0_prefix500_callback_like.jpg](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_callback_like/smooth_random_traj0_prefix500_callback_like.jpg)
- bidirectional image: [bidirectional_traj0_prefix500_callback_like.jpg](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_callback_like/bidirectional_traj0_prefix500_callback_like.jpg)

Observed total MSE on this case:

- IID exploration DPF: `3.688e-05`
- Smooth-random exploration DPF: `0.005175`
- Bidirectional DPF: `0.069083`

Takeaway:

- on this matched callback-like case, the IID checkpoint was strongest
- the smooth-random checkpoint was decent
- the bidirectional checkpoint was clearly worse than its W&B sample might suggest

## Prefix-Length Sweep

A broader sweep was then run on `traj_0` with prefix lengths:

- `1, 2, 4, 8, 16, 32, 100, 200, 400`

Assets:

- summary table: [summary_table.md](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_prefixlen_sweep/summary_table.md)
- full summary: [summary.json](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_prefixlen_sweep/summary.json)

Representative images:

- IID, prefix 1: [iid_uniform_traj0_prefix0001.jpg](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_prefixlen_sweep/iid_uniform_traj0_prefix0001.jpg)
- IID, prefix 100: [iid_uniform_traj0_prefix0100.jpg](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_prefixlen_sweep/iid_uniform_traj0_prefix0100.jpg)
- Smooth-random, prefix 16: [smooth_random_traj0_prefix0016.jpg](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_prefixlen_sweep/smooth_random_traj0_prefix0016.jpg)
- Smooth-random, prefix 400: [smooth_random_traj0_prefix0400.jpg](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_prefixlen_sweep/smooth_random_traj0_prefix0400.jpg)
- Bidirectional, prefix 32: [bidirectional_traj0_prefix0032.jpg](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_prefixlen_sweep/bidirectional_traj0_prefix0032.jpg)
- Bidirectional, prefix 400: [bidirectional_traj0_prefix0400.jpg](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_prefixlen_sweep/bidirectional_traj0_prefix0400.jpg)

Key trend:

- IID exploration DPF was by far the strongest once the prefix was not tiny
- Smooth-random improved a lot for long prefixes but was unstable at short and medium prefixes
- Bidirectional was the weakest and highly erratic on this sweep

Selected total MSE values:

- IID: `prefix 32 -> 0.004130`, `prefix 100 -> 0.002232`, `prefix 400 -> 0.000097`
- Smooth-random: `prefix 16 -> 0.134356`, `prefix 200 -> 0.023637`, `prefix 400 -> 0.012321`
- Bidirectional: `prefix 32 -> 0.771355`, `prefix 100 -> 0.523207`, `prefix 400 -> 0.163783`

Important caveat:

- this was a single-trajectory, single-sample diagnostic, not a robust average

## Rebased Prefix Evaluation

Another important test was the rebased-start setting:

1. pick validation trajectory `traj_0`
2. choose a start index `i`
3. discard `0..i-1`
4. treat the remaining suffix as a new trajectory starting at timestep `0`
5. keep `prefix_len = 200`
6. set query length to the remaining suffix length

This was tested at:

- `rebase_start = 100, 200, 400`

Assets:

- summary table: [summary_table.md](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_rebased_prefix200_examples/summary_table.md)
- full summary: [summary.json](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_rebased_prefix200_examples/summary.json)

Representative images:

- IID start 100: [iid_uniform_traj0_start0100_len0900_prefix200.jpg](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_rebased_prefix200_examples/iid_uniform_traj0_start0100_len0900_prefix200.jpg)
- IID start 400: [iid_uniform_traj0_start0400_len0600_prefix200.jpg](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_rebased_prefix200_examples/iid_uniform_traj0_start0400_len0600_prefix200.jpg)
- Smooth-random start 200: [smooth_random_traj0_start0200_len0800_prefix200.jpg](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_rebased_prefix200_examples/smooth_random_traj0_start0200_len0800_prefix200.jpg)
- Bidirectional start 100: [bidirectional_traj0_start0100_len0900_prefix200.jpg](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_rebased_prefix200_examples/bidirectional_traj0_start0100_len0900_prefix200.jpg)
- Bidirectional start 400: [bidirectional_traj0_start0400_len0600_prefix200.jpg](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_rebased_prefix200_examples/bidirectional_traj0_start0400_len0600_prefix200.jpg)

Observed total MSE:

- IID: `start100 -> 0.002505`, `start200 -> 0.001732`, `start400 -> 0.000194`
- Smooth-random: `start100 -> 0.027230`, `start200 -> 0.005065`, `start400 -> 0.005810`
- Bidirectional: `start100 -> 0.360731`, `start200 -> 0.279273`, `start400 -> 0.052510`

Takeaway:

- under this rebased-start evaluation, IID was again the strongest
- smooth-random was workable and much better than bidirectional
- bidirectional improved only when the rebased suffix became relatively short

## Closed-Loop Reacher Task

For task evaluation, the retained main benchmark mode became:

- `validation_random_source_random_future_target_in_traj`

The documented settings and kept benchmark folder are described in:

- [reacher_exploration_iid_eval_settings_2026-03-28.md](/home/gsang/Projects/hnn_guided_dpf/docs/reacher_exploration_iid_eval_settings_2026-03-28.md)

Main kept run:

- summary: [reacher_goal_prefix_expansion_summary.json](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_iid_ckpt_validation_random_source_random_future_d10cm_10tests/reacher_goal_prefix_expansion_summary.json)
- example GIF: [traj_0008_t0558_rt0866_workspace.gif](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_iid_ckpt_validation_random_source_random_future_d10cm_10tests/traj_0008_t0558_rt0866_workspace.gif)

Main result:

- success rate: `0.6`
- mean best goal distance: `0.01229 m`

Takeaway:

- the IID exploration checkpoint can solve a meaningful fraction of in-trajectory source-target tasks with at least `10 cm` initial separation

## Harder Cross-Trajectory Task Mode

A harder mode was also added:

- `validation_random_source_random_target_across_trajs`

This samples the source state and target independently from validation trajectories, allowing them to come from different trajectories.

One unguided 10-task run is here:

- summary: [reacher_goal_prefix_expansion_summary.json](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_iid_ckpt_validation_random_source_random_target_across_trajs_d10cm_10tests/reacher_goal_prefix_expansion_summary.json)

Takeaway:

- this harder benchmark is much worse than the in-trajectory benchmark
- broad source-target generalization remains difficult

## HNN Guidance

The HNN-guidance sweep is documented in:

- [reacher_sampling_hnn_guidance_sweep_2026-04-06.md](/home/gsang/Projects/hnn_guided_dpf/docs/reacher_sampling_hnn_guidance_sweep_2026-04-06.md)

Best tested config:

- `guidance_method = strategy2`
- `alpha_q = 1e-3`
- `alpha_p = 1e-3`
- `guidance_trust_lambda = 0`
- `guidance_normalize_grad = true`
- `guidance_joint_update = false`

Full 1000-prefix confirmation:

- report: [reacher_sampling_hnn_guidance_compare.json](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_compare_1000_a1/reacher_sampling_hnn_guidance_compare.json)

Representative example plots:

- [improve_big_traj0096_prefix0049.png](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_examples/improve_big_traj0096_prefix0049.png)
- [improve_mid_traj0034_prefix0023.png](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_examples/improve_mid_traj0034_prefix0023.png)
- [near_tie_traj0592_prefix0980.png](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_examples/near_tie_traj0592_prefix0980.png)
- [worse_big_traj0044_prefix0009.png](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_examples/worse_big_traj0044_prefix0009.png)

Takeaway:

- HNN guidance improved replay MSE only slightly on average
- the effect was high-variance: some examples improved a lot, some worsened badly
- no strong task-level benefit was established

## Target Guidance

Target guidance was tested with HNN guidance turned off.

The target-guidance sweep is documented in:

- [reacher_target_guidance_sweep_2026-04-06.md](/home/gsang/Projects/hnn_guided_dpf/docs/reacher_target_guidance_sweep_2026-04-06.md)

Best tested target-guidance setting on the capped pilot benchmark:

- `target_guidance_alpha = 3e-3`
- `target_guidance_time_power = 2.0`
- `target_guidance_normalize_grad = true`

Comparison folders:

- unguided: `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_unguided_acrosstraj_cap200_pilot5`
- target-guided: `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_target_guidance_acrosstraj_cap200_alpha3e3_pilot5`

Replay-vs-unguided-vs-guided comparison images:

- [src3246_t0179_goal0342_t0236_replay_vs_unguided_vs_guided.png](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_guidance_replay_comparisons/src3246_t0179_goal0342_t0236_replay_vs_unguided_vs_guided.png)
- [src1231_t0075_goal0163_t0016_replay_vs_unguided_vs_guided.png](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_guidance_replay_comparisons/src1231_t0075_goal0163_t0016_replay_vs_unguided_vs_guided.png)
- [src3350_t0109_goal1046_t0298_replay_vs_unguided_vs_guided.png](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_guidance_replay_comparisons/src3350_t0109_goal1046_t0298_replay_vs_unguided_vs_guided.png)

Takeaway:

- target guidance gave only a small numerical improvement on the capped benchmark
- it did not produce a meaningful success-rate gain

## Overall Conclusion

Current evidence points to:

1. The exploration-trained IID checkpoint is the strongest of the tested DPFs under the corrected train-matched completion setup.
2. Smooth-random is the second-best option and looks more stable than the old bidirectional model under rebased evaluation.
3. The old bidirectional checkpoint is comparatively weak outside narrower conditions.
4. HNN guidance and target guidance both provide, at best, small average improvements so far.
5. The train-matched completion interpretation matters; earlier prefix-only-style readings were overly pessimistic.

## Recommended Default References

For future discussion, the most useful reference artifacts are:

- prefix sweep summary: [summary_table.md](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_prefixlen_sweep/summary_table.md)
- rebased prefix summary: [summary_table.md](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_three_checkpoint_rebased_prefix200_examples/summary_table.md)
- main IID task benchmark: [reacher_goal_prefix_expansion_summary.json](/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion_iid_ckpt_validation_random_source_random_future_d10cm_10tests/reacher_goal_prefix_expansion_summary.json)
- HNN-guidance sweep note: [reacher_sampling_hnn_guidance_sweep_2026-04-06.md](/home/gsang/Projects/hnn_guided_dpf/docs/reacher_sampling_hnn_guidance_sweep_2026-04-06.md)
- target-guidance sweep note: [reacher_target_guidance_sweep_2026-04-06.md](/home/gsang/Projects/hnn_guided_dpf/docs/reacher_target_guidance_sweep_2026-04-06.md)
