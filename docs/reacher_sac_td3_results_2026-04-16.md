# Reacher SAC/TD3 Results 2026-04-16

This note records the finished SB3 SAC and TD3 runs on the fixed 10-task IID-uniform Reacher benchmark.

It was updated on `2026-04-17` to include the finished 3-seed `TD3` vs `TD3+HER` suite.

## Runtime Definitions

- `wall_clock_to_eval_start_s`: elapsed trainer time recorded right before that fixed-task evaluation started.
- `total_wall_clock_s`: full trainer wall-clock from launch to completion, from the saved trainer log.
- `final_eval_wall_clock_s`: recovered final fixed-10-task evaluation duration for these finished historical runs, computed as `total_wall_clock_s - wall_clock_to_final_eval_start_s`.
- historical note: these finished SAC/TD3 runs store checkpoint-level wall-clock and per-task `steps_taken`, but they did not directly store evaluation duration or per-source-target task wall-clock. The evaluator has now been patched so future runs will save both `evaluation_wall_clock_seconds` and `task_wall_clock_seconds` directly.
- `mean_steps_to_best_goal_dist`: mean executed control-step index at which each task first reached its minimum goal distance during that evaluation.
- `mean_steps_to_final_goal_dist`: mean executed control steps per task for that evaluation.

## Main Paths

- protocol: `/home/gsang/Projects/hnn_guided_dpf/docs/reacher_sac_td3_protocol_2026-04-15.md`
- output root: `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sb3_auto_pipeline_2026-04-15`
- SAC tune summary: `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sac_tune_2026-04-15/sac_tune_summary.md`
- TD3 tune summary: `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sb3_auto_pipeline_2026-04-15/td3_tune/td3_tune_summary.md`
- TD3 vs TD3+HER multiseed suite: `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/td3_her_multiseed_suite_2026-04-17_v2/suite_summary.md`

## Pipeline Runtime

| item | wall_clock_s | wall_clock_min |
|---|---:|---:|
| full SAC+TD3 auto pipeline | 865.282 | 14.421 |
| TD3 vs TD3+HER 3-seed suite | 2616.038 | 43.600 |

## Tune Results

| algorithm | selected variant | selected step | success_rate | best_goal_mean | final_goal_mean | mean_steps_to_best_goal_dist | mean_steps_to_final_goal_dist | wall_clock_to_eval_start_s | extra runtime note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| SAC | `lowlr1e4_ls1000_30k` | 10000 | 0.400 | 0.063795 | 0.117747 | 1646.7 | 1946.7 | 28.366 | SAC tune logs were not retained separately, so only pre-eval wall-clock is available |
| TD3 | `lowlr1e4_noise0p1_30k` | 20000 | 0.900 | 0.007631 | 0.018975 | 2186.6 | 2231.5 | 41.659 | total wall-clock for this 30k run was `85.0 s` |

## Full Training Results

| algorithm | best step | best success_rate | best_goal_mean | best final_goal_mean | wall_clock_to_best_eval_start_s | final step | final success_rate | final best_goal_mean | final final_goal_mean | wall_clock_to_final_eval_start_s | final_eval_wall_clock_s | total_wall_clock_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAC | 210000 | 0.700 | 0.056294 | 0.064555 | 540.833 | 300000 | 0.200 | 0.063283 | 0.149549 | 831.896 | 21.404 | 853.3 |
| TD3 | 30000 | 1.000 | 0.008437 | 0.008437 | 48.958 | 300000 | 0.700 | 0.049028 | 0.074556 | 532.549 | 7.051 | 539.6 |

## TD3 vs TD3+HER 3-Seed Suite

These rows are aggregated over seeds `0,1,2` from:

- `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/td3_her_multiseed_suite_2026-04-17_v2/suite_summary.md`
- `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/td3_her_multiseed_suite_2026-04-17_v2/pipeline_state.json`

### Aggregate Results

| method | checkpoint selection | success_rate mean±std | best_goal_mean mean±std | final_goal_mean mean±std | mean_steps_to_best_goal_dist mean±std | mean_steps_to_final_goal_dist mean±std | eval_wall_clock_s mean±std | mean_task_eval_s mean±std | training_wall_clock_s mean±std |
|---|---|---|---|---|---|---|---|---|---|
| TD3 | best checkpoint per seed | `1.000 ± 0.000` | `0.007830 ± 0.000482` | `0.007830 ± 0.000482` | `1935.1 ± 149.0` | `1935.1 ± 149.0` | `8.857 ± 0.502` | `0.866 ± 0.049` | `601.3 ± 16.4` |
| TD3+HER | best checkpoint per seed | `1.000 ± 0.000` | `0.007087 ± 0.000443` | `0.007087 ± 0.000443` | `1523.1 ± 316.6` | `1523.1 ± 316.6` | `8.749 ± 1.823` | `0.839 ± 0.172` | `1414.1 ± 52.6` |
| TD3 | final checkpoint | `0.667 ± 0.125` | `0.051454 ± 0.022296` | `0.072210 ± 0.025858` | `1537.6 ± 187.0` | `1704.2 ± 171.5` | `7.793 ± 0.750` | `0.760 ± 0.081` | `601.3 ± 16.4` |
| TD3+HER | final checkpoint | `0.067 ± 0.047` | `0.126952 ± 0.012432` | `0.151877 ± 0.009312` | `1481.4 ± 274.0` | `1921.5 ± 269.2` | `10.807 ± 1.394` | `1.072 ± 0.139` | `1414.1 ± 52.6` |

### Per-Seed Best vs Final

| run | method | seed | best step | best success_rate | best_goal_mean | final step | final success_rate | final_goal_mean | run_wall_clock_s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `td3_seed0` | TD3 | 0 | 30000 | 1.000 | 0.008437 | 300000 | 0.700 | 0.049028 | 580.6 |
| `td3_seed1` | TD3 | 1 | 180000 | 1.000 | 0.007257 | 300000 | 0.800 | 0.025440 | 602.6 |
| `td3_seed2` | TD3 | 2 | 20000 | 1.000 | 0.007795 | 300000 | 0.500 | 0.079892 | 620.6 |
| `td3_her_seed0` | TD3+HER | 0 | 90000 | 1.000 | 0.007671 | 300000 | 0.100 | 0.139945 | 1487.7 |
| `td3_her_seed1` | TD3+HER | 1 | 100000 | 1.000 | 0.006993 | 300000 | 0.000 | 0.130713 | 1368.3 |
| `td3_her_seed2` | TD3+HER | 2 | 60000 | 1.000 | 0.006597 | 300000 | 0.100 | 0.110198 | 1386.2 |

## Evaluation Step Metrics

These are normalized executed control steps from the fixed 10-task evaluation summaries.

| algorithm | tune selected step | tune mean_steps_to_best_goal_dist | tune mean_steps_to_final_goal_dist | best checkpoint mean_steps_to_best_goal_dist | best checkpoint mean_steps_to_final_goal_dist | final checkpoint mean_steps_to_best_goal_dist | final checkpoint mean_steps_to_final_goal_dist |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAC | 10000 | 1646.7 | 1946.7 | 1921.2 | 2071.2 | 2636.7 | 3307.9 |
| TD3 | 20000 | 2186.6 | 2231.5 | 2112.0 | 2112.0 | 1320.4 | 1470.4 |

## Checkpoints To Use

| algorithm | recommended checkpoint | reason |
|---|---|---|
| SAC | `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sb3_auto_pipeline_2026-04-15/sac_full/checkpoints/best_eval.zip` | best fixed-task eval happened at `210k`, not at the final checkpoint |
| TD3 | `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sb3_auto_pipeline_2026-04-15/td3_full/checkpoints/best_eval.zip` | TD3 peaked very early at `30k` and degraded afterward |

Latest checkpoints, if needed for completeness:

- SAC latest: `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sb3_auto_pipeline_2026-04-15/sac_full/checkpoints/latest.zip`
- TD3 latest: `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sb3_auto_pipeline_2026-04-15/td3_full/checkpoints/latest.zip`

## Main Takeaways

- TD3 was much stronger than SAC on this benchmark under the standard SB3 MLP setup.
- TD3 reached `success_rate = 1.0` at `30k` with `best_goal_mean = 0.008437`, which is already close to the pure MuJoCo IID random-shooting baseline.
- Both SAC and TD3 degraded after their best checkpoints, so `best_eval.zip` matters more than `latest.zip`.
- SAC improved during training, but even its best checkpoint stayed clearly behind TD3 here.
- The final fixed-10-task evaluation itself was short compared with training: about `21.4 s` for SAC and `7.1 s` for TD3.
- In the 3-seed suite, `TD3+HER` slightly improved the mean best-checkpoint goal distance over plain `TD3` (`0.007087 m` vs `0.007830 m`), but it was much less stable over long training.
- The final-checkpoint gap is large: 3-seed `TD3` final success was `0.667 ± 0.125`, while 3-seed `TD3+HER` final success was only `0.067 ± 0.047`.
- `TD3+HER` was also much slower to train in this setup, with mean per-run wall-clock about `1414 s` vs `601 s` for plain `TD3`.
