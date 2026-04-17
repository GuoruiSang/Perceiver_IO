# Reacher Method Comparison 2026-04-16

This note consolidates the finished reacher-task results into one comparison sheet.

I do **not** use one fully Cartesian table over every method, candidate count, and norm, because that would be misleading:

- `L1/L2/Linf` only apply to target-guided DPF, not unguided DPF
- many guided DPF candidate-count cells were never run
- runtime units are different across RL, shooting baselines, and DPF MPC-style methods

So this sheet uses a structured comparison with consistent rows only for **completed** experiments, plus an explicit missing/not-run section.

## Shared Benchmark

These comparisons all target the same fixed 10-task benchmark unless noted otherwise:

- task mode: `validation_random_source_random_target_across_trajs`
- task ids: `0,1,2,3,4,5,6,7,8,9`
- lookahead / rollout family: `lookahead_steps=256`
- recent prefix cap where applicable: `64`
- `reset_window_time_indices=true` where applicable
- `stall_patience_steps=500`
- `goal_tolerance=0.01`
- `random_future_target_min_initial_distance=0.1`
- `cross_traj_sampling_max_tries=128`
- `max_sampling_retries=3`
- `retry_improvement_margin=0.001`

## Runtime Conventions

- RL rows use `best-checkpoint fixed-10-task evaluation wall-clock`
- DPF cand1/cand8 guidance rows use `serial runtime of that finished 10-task method run`
- DPF unguided candidate-sweep rows use `serial runtime estimate`
- HNN random-shooting sweep rows use `serial runtime estimate`
- MuJoCo random-shooting sweep rows use `serial wall-clock`, which equals wall-clock because that sweep was serial
- step metrics below are normalized to executed control steps; for older MPC/shooting summaries they were recomputed from saved rollout traces because some legacy `steps_taken` fields counted the initial state in the rollout length

## 1. RL Policy Baselines

These use the recommended `best_eval.zip` checkpoints, not the final degraded checkpoints.

| Family | Variant | Candidates | Success Rate | Mean Best Goal Dist | Mean Final Goal Dist | Mean Steps To Best | Mean Steps To Final | Eval Wall-Clock | Mean Task Eval Time | Training Wall-Clock |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `RL` | `SAC best checkpoint` | `-` | `0.700` | `0.056294 m` | `0.064555 m` | `1921.2` | `2071.2` | `13.403 s` | `1.331 s` | `853.3 s` |
| `RL` | `TD3 best checkpoint` | `-` | `1.000` | `0.008437 m` | `0.008437 m` | `2112.0` | `2112.0` | `10.211 s` | `0.999 s` | `539.6 s` |

Sources:

- `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sb3_auto_pipeline_2026-04-15/sac_full/eval_best_checkpoint_2026-04-16/reacher_rl_policy_eval_summary.json`
- `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sb3_auto_pipeline_2026-04-15/td3_full/eval_best_checkpoint_2026-04-16/reacher_rl_policy_eval_summary.json`
- `/home/gsang/Projects/hnn_guided_dpf/docs/reacher_sac_td3_results_2026-04-16.md`

## 2. Random-Shooting Baselines

### 2.1 HNN Random Shooting

| Family | Variant | Candidates | Success Rate | Mean Best Goal Dist | Mean Final Goal Dist | Mean Steps To Best | Mean Steps To Final | Serial Runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `HNN shooting` | `iid random shooting` | `16` | `1.000` | `0.009926 m` | `0.009926 m` | `1120.7` | `1120.7` | `398.5 min` |
| `HNN shooting` | `iid random shooting` | `32` | `1.000` | `0.009943 m` | `0.009943 m` | `1110.7` | `1110.7` | `397.5 min` |
| `HNN shooting` | `iid random shooting` | `64` | `1.000` | `0.009938 m` | `0.009938 m` | `968.3` | `968.3` | `311.3 min` |
| `HNN shooting` | `iid random shooting` | `128` | `1.000` | `0.009861 m` | `0.009861 m` | `849.6` | `849.6` | `267.0 min` |

Source:

- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_hnn_iid_random_shooting_sweep_sharded_2026-04-15_fast_2workers/sweep_progress.md`

### 2.2 MuJoCo Random Shooting

| Family | Variant | Candidates | Success Rate | Mean Best Goal Dist | Mean Final Goal Dist | Mean Steps To Best | Mean Steps To Final | Serial Runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `MuJoCo shooting` | `iid random shooting` | `8` | `1.000` | `0.009892 m` | `0.009892 m` | `1270.8` | `1270.8` | `21.6 min` |
| `MuJoCo shooting` | `iid random shooting` | `16` | `1.000` | `0.009947 m` | `0.009947 m` | `1069.5` | `1069.5` | `30.6 min` |
| `MuJoCo shooting` | `iid random shooting` | `32` | `1.000` | `0.009888 m` | `0.009888 m` | `1054.1` | `1054.1` | `54.5 min` |
| `MuJoCo shooting` | `iid random shooting` | `64` | `1.000` | `0.009877 m` | `0.009877 m` | `955.5` | `955.5` | `92.9 min` |
| `MuJoCo shooting` | `iid random shooting` | `128` | `1.000` | `0.009902 m` | `0.009902 m` | `889.3` | `889.3` | `162.9 min` |

Source:

- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_mujoco_iid_random_shooting_sweep_serial_2026-04-15_8cpu/sweep_progress.md`

## 3. Unguided DPF By Candidate Count

### 3.1 cand1 and cand8

| Family | Variant | Candidates | Success Rate | Mean Best Goal Dist | Mean Final Goal Dist | Mean Steps To Best | Mean Steps To Final | Serial Runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `DPF unguided` | `predicted_suffix` | `1` | `0.000` | `0.0732 m` | `0.1400 m` | `2469.0` | `3193.5` | `138.4 min` |
| `DPF unguided` | `predicted_suffix` | `8` | `0.300` | `0.0274 m` | `0.0984 m` | `1305.5` | `1748.9` | `68.0 min` |

Sources:

- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_guidance_compare_suite_2026-04-14_final_only_fixed_alpha1e4/cand1_finished_partial_summary.md`
- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_guidance_compare_suite_2026-04-14_final_only_fixed_alpha1e4/cand8_finished_partial_summary.md`

### 3.2 Candidate Sweep

| Family | Variant | Candidates | Success Rate | Mean Best Goal Dist | Mean Final Goal Dist | Mean Steps To Best | Mean Steps To Final | Serial Runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `DPF unguided` | `predicted_suffix` | `16` | `0.800` | `0.011464 m` | `0.032585 m` | `1387.2` | `1487.2` | `68.5 min` |
| `DPF unguided` | `predicted_suffix` | `32` | `0.700` | `0.012835 m` | `0.035272 m` | `1129.4` | `1325.7` | `111.0 min` |
| `DPF unguided` | `predicted_suffix` | `64` | `0.600` | `0.015565 m` | `0.044509 m` | `1192.4` | `1392.4` | `195.3 min` |
| `DPF unguided` | `predicted_suffix` | `128` | `0.900` | `0.010246 m` | `0.012643 m` | `1317.6` | `1367.6` | `364.2 min` |

Source:

- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_unguided_candidate_sweep_sharded_2026-04-15/sweep_progress.md`

## 4. Guided DPF, cand1

Here `guidance` refers to target-guidance norm and optional follow-up HNN guidance on top of the DPF rollout.

| Family | Variant | Candidates | Guidance | Success Rate | Mean Best Goal Dist | Mean Final Goal Dist | Mean Steps To Best | Mean Steps To Final | Serial Runtime |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `DPF guided` | `target_l1` | `1` | `target L1` | `0.100` | `0.0704 m` | `0.1404 m` | `2589.6` | `3324.6` | `172.7 min` |
| `DPF guided` | `target_l2` | `1` | `target L2` | `0.100` | `0.0676 m` | `0.1253 m` | `2597.8` | `3105.1` | `160.6 min` |
| `DPF guided` | `target_linf` | `1` | `target Linf` | `0.200` | `0.0565 m` | `0.1238 m` | `2912.3` | `3659.0` | `176.6 min` |
| `DPF + HNN guided` | `hnn` | `1` | `HNN only` | `0.000` | `0.0736 m` | `0.1423 m` | `2467.5` | `3181.2` | `219.0 min` |
| `DPF + HNN guided` | `target_l1_then_hnn` | `1` | `target L1 + HNN` | `0.100` | `0.0543 m` | `0.1195 m` | `2975.8` | `3522.0` | `258.2 min` |
| `DPF + HNN guided` | `target_l2_then_hnn` | `1` | `target L2 + HNN` | `0.000` | `0.0740 m` | `0.1402 m` | `2401.2` | `3084.3` | `232.0 min` |
| `DPF + HNN guided` | `target_linf_then_hnn` | `1` | `target Linf + HNN` | `0.100` | `0.0567 m` | `0.1311 m` | `2867.7` | `3614.9` | `269.4 min` |

Source:

- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_guidance_compare_suite_2026-04-14_final_only_fixed_alpha1e4/cand1_finished_partial_summary.md`

## 5. Guided DPF, cand8

| Family | Variant | Candidates | Guidance | Success Rate | Mean Best Goal Dist | Mean Final Goal Dist | Mean Steps To Best | Mean Steps To Final | Serial Runtime |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `DPF guided` | `target_l1` | `8` | `target L1` | `0.300` | `0.0282 m` | `0.0959 m` | `1220.1` | `1703.4` | `72.6 min` |
| `DPF guided` | `target_l2` | `8` | `target L2` | `0.300` | `0.0273 m` | `0.0976 m` | `1305.5` | `1685.5` | `71.7 min` |
| `DPF guided` | `target_linf` | `8` | `target Linf` | `0.400` | `0.0273 m` | `0.0786 m` | `1280.1` | `1711.3` | `n/a` |
| `DPF + HNN guided` | `hnn` | `8` | `HNN only` | `0.200` | `0.0289 m` | `0.1072 m` | `1229.5` | `1745.2` | `103.5 min` |
| `DPF + HNN guided` | `target_l1_then_hnn` | `8` | `target L1 + HNN` | `0.400` | `0.0275 m` | `0.0745 m` | `1221.9` | `1573.6` | `98.8 min` |
| `DPF + HNN guided` | `target_l2_then_hnn` | `8` | `target L2 + HNN` | `0.300` | `0.0274 m` | `0.0995 m` | `1305.6` | `1733.9` | `108.7 min` |
| `DPF + HNN guided` | `target_linf_then_hnn` | `8` | `target Linf + HNN` | `0.400` | `0.0272 m` | `0.0788 m` | `1322.2` | `1753.4` | `n/a` |

Source:

- `/home/gsang/Projects/hnn_guided_dpf/plots/reacher_guidance_compare_suite_2026-04-14_final_only_fixed_alpha1e4/cand8_finished_partial_summary.md`

## 6. Missing Or Not-Run Cells

These should stay blank rather than be guessed:

- `Unguided DPF × {L1, L2, Linf}`:
  not applicable, because unguided DPF does not use target-guidance norms
- `Guided DPF cand16 / cand32 / cand64 / cand128`:
  not run in the recorded experiment set

## 7. Quick Read

- Best finished RL result: `TD3 best checkpoint`, `success_rate=1.0`, `best_goal_mean=0.008437 m`
- Best finished DPF unguided row: `cand128`, `success_rate=0.9`, `best_goal_mean=0.010246 m`
- Best finished HNN random-shooting row: `cand128`, `success_rate=1.0`, `best_goal_mean=0.009861 m`
- Best finished MuJoCo random-shooting row: `cand64`, `success_rate=1.0`, `best_goal_mean=0.009877 m`
- Best finished cand1 guided DPF row by success: `target_linf`, `0.2`
- Best finished cand1 guided DPF row by mean best-goal distance: `target_l1_then_hnn`, `0.0543 m`
- Best finished cand8 guided DPF row by success: `target_l1_then_hnn`, `target_linf`, and `target_linf_then_hnn`, all `0.4`
