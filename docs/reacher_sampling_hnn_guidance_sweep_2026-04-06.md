# Reacher Sampling HNN Guidance Sweep

This note records the sampling-only HNN-guidance sweep used to choose a guidance configuration for the IID exploration Reacher DPF checkpoint.

Date:

- `2026-04-06`

## Goal

Compare unguided and HNN-guided prefix completions on replay MSE.

For each sampled validation trajectory:

1. Sample one random prefix length.
2. Generate an unguided completion.
3. Generate a guided completion using the same prefix and the same initial diffusion noise.
4. Compare the generated trajectory against the replay trajectory.

The main metric is:

- full-trajectory MSE over `qpos + mom`

We report:

- `unguided_full_mse_total`
- `guided_full_mse_total`
- `delta_full_mse_total = guided - unguided`
- `guided_better_fraction`

Negative `delta_full_mse_total` means guidance helped.

## Assets

DPF checkpoint:

- `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/dpf_exploration_iid_uniform_len1000_v1/trajectory_dpf_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions_cond-concat_torque_in_state_layout-shifted_tau_qpos-raw:epoch=2999_val_loss:val_loss=0.0983.ckpt`

HNN checkpoint:

- `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/hnn_exploration_iid_uniform_len1000_v1/StructuredHNN-ReacherExploration-IID-epoch-epoch=999.ckpt`

Validation dataset:

- `/home/gsang/Projects/hnn_guided_dpf/data/reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/val_traj_4000-steps_1000.h5`

Benchmark script:

- `/home/gsang/Projects/hnn_guided_dpf/scripts/data/compare_reacher_sampling_hnn_guidance.py`

## Shared Evaluation Settings

- `guidance_method = strategy2`
- `guidance_joint_update = false`
- `num_diffusion_steps = 20` for the main benchmark
- one random prefix per trajectory
- same initial diffusion noise for unguided and guided sampling
- MSE computed over the entire trajectory

## Best Config

Best tested config:

- `alpha_q = 1e-3`
- `alpha_p = 1e-3`
- `guidance_trust_lambda = 0`
- `guidance_normalize_grad = true`
- `guidance_joint_update = false`

This is the config referred to below as `a1`.

## Full 1000-Prefix Confirmation

Output directory:

- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_compare_1000_a1`

Report:

- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_compare_1000_a1/reacher_sampling_hnn_guidance_compare.json`

Log:

- `/home/gsang/Projects/hnn_guided_dpf/logs/reacher_sampling_hnn_guidance_compare_1000_a1.log`

Settings:

- `num_trajectories = 1000`
- `start_index = 0`
- `prefix_min = 1`
- `prefix_max = 999`
- `num_diffusion_steps = 20`
- `guidance_method = strategy2`
- `alpha_q = 1e-3`
- `alpha_p = 1e-3`
- `guidance_trust_lambda = 0`
- `guidance_normalize_grad = true`
- `guidance_joint_update = false`

Aggregate result:

- unguided full MSE total mean: `0.21911107497654891`
- guided full MSE total mean: `0.21756837464817527`
- mean delta (`guided - unguided`): `-0.0015427003283736162`
- guided better fraction: `0.511`

Interpretation:

- Guidance helped slightly on mean full-trajectory MSE.
- The effect was small.
- `qpos` mean MSE improved, while `mom` mean MSE got slightly worse.

## Pilot Sweep

Pilot sweep outputs:

- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_sweep/g1_aq1e3_ap1e3_t0/reacher_sampling_hnn_guidance_compare.json`
- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_sweep/g2_aq1e3_ap3e4_t0/reacher_sampling_hnn_guidance_compare.json`
- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_sweep/g3_aq1e3_ap1e4_t0/reacher_sampling_hnn_guidance_compare.json`
- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_sweep/g4_aq3e3_ap1e3_t0/reacher_sampling_hnn_guidance_compare.json`
- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_sweep/g5_aq3e3_ap3e4_t0/reacher_sampling_hnn_guidance_compare.json`
- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_sweep/g6_aq1e3_ap3e4_t1e4/reacher_sampling_hnn_guidance_compare.json`

Pilot settings:

- `num_trajectories = 300`
- `num_diffusion_steps = 20`

Pilot ranking by mean `delta_full_mse_total`:

1. `g1_aq1e3_ap1e3_t0`: `+0.024632965290140318`
2. `g2_aq1e3_ap3e4_t0`: `+0.024633046231163924`
3. `g3_aq1e3_ap1e4_t0`: `+0.02463309519858934`
4. `g6_aq1e3_ap3e4_t1e4`: `+0.024633118041297586`
5. `g4_aq3e3_ap1e3_t0`: `+0.024633786683043377`
6. `g5_aq3e3_ap3e4_t0`: `+0.02463390037586055`

Pilot takeaway:

- All six configs were nearly identical on this subset.
- All six were slightly worse than unguided on the pilot subset.
- `g1` was still the best among the tested local configs, so it remained the leading candidate for full confirmation.

## Wider Sweep

Finished wider-sweep reports:

- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_sweep_wide/w2_aq1e2_ap1e3_t0/reacher_sampling_hnn_guidance_compare.json`
- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_sweep_wide/w3_aq1e2_ap3e4_t0/reacher_sampling_hnn_guidance_compare.json`

These tested larger `alpha_q` values:

- `w2`: `alpha_q = 1e-2`, `alpha_p = 1e-3`, `trust = 0`
- `w3`: `alpha_q = 1e-2`, `alpha_p = 3e-4`, `trust = 0`

Results:

- `w2` mean delta: `+0.016434471344732374`
- `w3` mean delta: `+0.016434631883967696`

Takeaway:

- Larger `alpha_q` did not improve the benchmark.
- These wider settings were clearly worse than the confirmed `a1` result on the full 1000-prefix benchmark.

Some additional wide-sweep sessions were started and inspected by rolling log averages, then stopped once they were clearly tracking the same or worse behavior. They did not show evidence of beating `a1`.

## Final Recommendation

Use this HNN-guidance config for follow-up reacher-task testing:

- `guidance_method = strategy2`
- `alpha_q = 1e-3`
- `alpha_p = 1e-3`
- `guidance_trust_lambda = 0`
- `guidance_normalize_grad = true`
- `guidance_joint_update = false`

Current conclusion:

- This is the best tested config.
- It improves replay MSE only slightly on the 1000-prefix benchmark.
- The gain is real but small, so downstream task-level gains should be treated as uncertain until measured directly.
