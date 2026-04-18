# Reacher Task Comparison Table

This is a single table of all finished results on the shared 10-task benchmark:

- task mode: `validation_random_source_random_target_across_trajs`
- task ids: `0..9`
- `lookahead_steps=256`
- `stall_patience_steps=500`
- `goal_tolerance=0.01`
- `random_future_target_min_initial_distance=0.1`

All step counts below are normalized executed control steps. Runtime is test-time runtime for the evaluation itself: RL uses fixed-10-task evaluation wall-clock, while MPC/shooting methods use serial online-control runtime.

For the multi-seed RL rows:

- `3-seed best` = mean over each seed's own best checkpoint
- `3-seed final` = mean over the `300k` final checkpoint from each seed

| Family | Method | Number of Candidates | Guidance / Setting | Success Rate | Mean Best Goal Dist | Mean Final Goal Dist | Mean Steps To Best | Mean Steps To Final | Test-Time Runtime |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `RL` | `SAC` | `-` | best checkpoint | `0.700` | `0.0563 m` | `0.0646 m` | `1921.2` | `2071.2` | `13.4 s` |
| `RL` | `TD3` | `-` | best checkpoint | `1.000` | `0.0084 m` | `0.0084 m` | `2112.0` | `2112.0` | `10.2 s` |
| `RL` | `TD3` | `-` | 3-seed best checkpoint mean | `1.000` | `0.0078 m` | `0.0078 m` | `1935.1` | `1935.1` | `8.9 s` |
| `RL` | `TD3+HER` | `-` | 3-seed best checkpoint mean | `1.000` | `0.0071 m` | `0.0071 m` | `1523.1` | `1523.1` | `8.7 s` |
| `RL` | `TD3` | `-` | 3-seed final checkpoint mean | `0.667` | `0.0515 m` | `0.0722 m` | `1537.6` | `1704.2` | `7.8 s` |
| `RL` | `TD3+HER` | `-` | 3-seed final checkpoint mean | `0.067` | `0.1270 m` | `0.1519 m` | `1481.4` | `1921.5` | `10.8 s` |
| `DPF` | unguided | `1` | predicted suffix | `0.000` | `0.0732 m` | `0.1400 m` | `2469.0` | `3193.5` | `138.4 min` |
| `DPF` | unguided | `8` | predicted suffix | `0.300` | `0.0274 m` | `0.0984 m` | `1305.5` | `1748.9` | `68.0 min` |
| `DPF` | unguided | `16` | predicted suffix | `0.800` | `0.0115 m` | `0.0326 m` | `1387.2` | `1487.2` | `68.5 min` |
| `DPF` | unguided | `32` | predicted suffix | `0.700` | `0.0128 m` | `0.0353 m` | `1129.4` | `1325.7` | `111.0 min` |
| `DPF` | unguided | `64` | predicted suffix | `0.600` | `0.0156 m` | `0.0445 m` | `1192.4` | `1392.4` | `195.3 min` |
| `DPF` | unguided | `128` | predicted suffix | `0.900` | `0.0102 m` | `0.0126 m` | `1317.6` | `1367.6` | `364.2 min` |
| `DPF` | guided | `1` | `target_l1` | `0.100` | `0.0704 m` | `0.1404 m` | `2589.6` | `3324.6` | `172.7 min` |
| `DPF` | guided | `1` | `target_l2` | `0.100` | `0.0676 m` | `0.1253 m` | `2597.8` | `3105.1` | `160.6 min` |
| `DPF` | guided | `1` | `target_linf` | `0.200` | `0.0565 m` | `0.1238 m` | `2912.3` | `3659.0` | `176.6 min` |
| `DPF+HNN` | guided | `1` | `hnn only` | `0.000` | `0.0736 m` | `0.1423 m` | `2467.5` | `3181.2` | `219.0 min` |
| `DPF+HNN` | guided | `1` | `target_l1 + hnn` | `0.100` | `0.0543 m` | `0.1195 m` | `2975.8` | `3522.0` | `258.2 min` |
| `DPF+HNN` | guided | `1` | `target_l2 + hnn` | `0.000` | `0.0740 m` | `0.1402 m` | `2401.2` | `3084.3` | `232.0 min` |
| `DPF+HNN` | guided | `1` | `target_linf + hnn` | `0.100` | `0.0567 m` | `0.1311 m` | `2867.7` | `3614.9` | `269.4 min` |
| `DPF` | guided | `8` | `target_l1` | `0.300` | `0.0282 m` | `0.0959 m` | `1220.1` | `1703.4` | `72.6 min` |
| `DPF` | guided | `8` | `target_l2` | `0.300` | `0.0273 m` | `0.0976 m` | `1305.5` | `1685.5` | `71.7 min` |
| `DPF` | guided | `8` | `target_linf` | `0.400` | `0.0273 m` | `0.0786 m` | `1280.1` | `1711.3` | `n/a` |
| `DPF+HNN` | guided | `8` | `hnn only` | `0.200` | `0.0289 m` | `0.1072 m` | `1229.5` | `1745.2` | `103.5 min` |
| `DPF+HNN` | guided | `8` | `target_l1 + hnn` | `0.400` | `0.0275 m` | `0.0745 m` | `1221.9` | `1573.6` | `98.8 min` |
| `DPF+HNN` | guided | `8` | `target_l2 + hnn` | `0.300` | `0.0274 m` | `0.0995 m` | `1305.6` | `1733.9` | `108.7 min` |
| `DPF+HNN` | guided | `8` | `target_linf + hnn` | `0.400` | `0.0272 m` | `0.0788 m` | `1322.2` | `1753.4` | `n/a` |
| `HNN` | iid random shooting | `16` | random shooting | `1.000` | `0.0099 m` | `0.0099 m` | `1120.7` | `1120.7` | `398.5 min` |
| `HNN` | iid random shooting | `32` | random shooting | `1.000` | `0.0099 m` | `0.0099 m` | `1110.7` | `1110.7` | `397.5 min` |
| `HNN` | iid random shooting | `64` | random shooting | `1.000` | `0.0099 m` | `0.0099 m` | `968.3` | `968.3` | `311.3 min` |
| `HNN` | iid random shooting | `128` | random shooting | `1.000` | `0.0099 m` | `0.0099 m` | `849.6` | `849.6` | `267.0 min` |
| `MuJoCo` | iid random shooting | `8` | random shooting | `1.000` | `0.0099 m` | `0.0099 m` | `1270.8` | `1270.8` | `21.6 min` |
| `MuJoCo` | iid random shooting | `16` | random shooting | `1.000` | `0.0099 m` | `0.0099 m` | `1069.5` | `1069.5` | `30.6 min` |
| `MuJoCo` | iid random shooting | `32` | random shooting | `1.000` | `0.0099 m` | `0.0099 m` | `1054.1` | `1054.1` | `54.5 min` |
| `MuJoCo` | iid random shooting | `64` | random shooting | `1.000` | `0.0099 m` | `0.0099 m` | `955.5` | `955.5` | `92.9 min` |
| `MuJoCo` | iid random shooting | `128` | random shooting | `1.000` | `0.0099 m` | `0.0099 m` | `889.3` | `889.3` | `162.9 min` |

## Interpretation

- `TD3` is the strongest finished learned controller overall. It reaches `1.0` success with `0.0084 m` mean best/final goal distance, while also having by far the cheapest test-time evaluation.
- `SAC` is much weaker than `TD3` on this benchmark. It improves over the weaker DPF guided rows, but it is clearly behind `TD3` and also behind the strongest unguided/shooting baselines in accuracy.
- In the 3-seed RL suite, `TD3+HER` slightly improves the best-checkpoint mean over plain `TD3` (`0.0071 m` vs `0.0078 m`), so HER can help peak performance in this goal-conditioned setting.
- But `TD3+HER` is much less stable over long training. Its 3-seed final mean collapses to `0.067` success, while plain `TD3` keeps `0.667` final success.
- `DPF unguided` improves substantially with candidate count, but not monotonically. In this finished sweep, `cand128` is best, while `cand32` and `cand64` are weaker than `cand16`, which suggests noticeable stochasticity and online-selection effects.
- `DPF guided` helps mainly at low candidate count. For `cand1`, target guidance and target+HNN guidance improve over unguided. For `cand8`, guidance lifts success from `0.3` to `0.4`, but the gain is still moderate.
- Within the finished `cand8` guided rows, `target_l1 + hnn`, `target_linf`, and `target_linf + hnn` all tie on success rate at `0.4`. Among them, `target_l1 + hnn` has the best mean final-goal distance, while `target_linf + hnn` has the best mean best-goal distance by a very small margin.
- `HNN` random shooting and `MuJoCo` random shooting are both very strong planning-style baselines. They achieve near-identical accuracy at high candidate count and both outperform all DPF variants on this fixed benchmark.
- `MuJoCo` random shooting is much cheaper than `HNN` random shooting at test time. So the learned HNN dynamics did not buy a runtime advantage here; it mainly acts as a learned replacement for the simulator, but the current implementation is still slower.
- The main test-time tradeoff is therefore: `RL` shifts compute into offline training and is extremely cheap at inference, while `DPF/HNN/MuJoCo` keep large compute in the online decision loop.

## Notes

- Guided DPF was only run for `cand1` and `cand8`, so there are no guided rows for `cand16/32/64/128`.
- The two rerun-completed rows `cand8 target_linf` and `cand8 target_linf + hnn` have runtime listed as `n/a` because the rerun logs did not preserve recoverable wall-clock timing.
- For paper use, you may want to bold the best row within each family or the globally best row for each metric in LaTeX.

## Source Docs

- `/home/gsang/Projects/hnn_guided_dpf/docs/reacher_method_comparison_2026-04-16.md`
- `/home/gsang/Projects/hnn_guided_dpf/docs/reacher_sac_td3_results_2026-04-16.md`
- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_guidance_compare_suite_2026-04-14_final_only_fixed_alpha1e4/cand1_finished_partial_summary.md`
- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_guidance_compare_suite_2026-04-14_final_only_fixed_alpha1e4/cand8_finished_partial_summary.md`
- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_unguided_candidate_sweep_sharded_2026-04-15/sweep_progress.md`
- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_hnn_iid_random_shooting_sweep_sharded_2026-04-15_fast_2workers/sweep_progress.md`
- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_mujoco_iid_random_shooting_sweep_serial_2026-04-15_8cpu/sweep_progress.md`
- `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/td3_her_multiseed_suite_2026-04-17_v2/suite_summary.md`
