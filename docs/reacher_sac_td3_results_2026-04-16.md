# Reacher SAC/TD3 Results 2026-04-16

This note records the finished SB3 SAC and TD3 runs on the fixed 10-task IID-uniform Reacher benchmark.

## Runtime Definitions

- `wall_clock_to_eval_start_s`: elapsed trainer time recorded right before that fixed-task evaluation started.
- `total_wall_clock_s`: full trainer wall-clock from launch to completion, from the saved trainer log.
- historical note: these finished SAC/TD3 runs store checkpoint-level wall-clock and per-task `steps_taken`, but they do not store per-source-target task wall-clock. The evaluator has now been patched so future runs will save `task_wall_clock_seconds` per task directly.

## Main Paths

- protocol: `/home/gsang/Projects/hnn_guided_dpf/docs/reacher_sac_td3_protocol_2026-04-15.md`
- output root: `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sb3_auto_pipeline_2026-04-15`
- SAC tune summary: `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sac_tune_2026-04-15/sac_tune_summary.md`
- TD3 tune summary: `/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sb3_auto_pipeline_2026-04-15/td3_tune/td3_tune_summary.md`

## Pipeline Runtime

| item | wall_clock_s | wall_clock_min |
|---|---:|---:|
| full SAC+TD3 auto pipeline | 865.282 | 14.421 |

## Tune Results

| algorithm | selected variant | selected step | success_rate | best_goal_mean | final_goal_mean | wall_clock_to_eval_start_s | extra runtime note |
|---|---|---:|---:|---:|---:|---:|---|
| SAC | `lowlr1e4_ls1000_30k` | 10000 | 0.400 | 0.063795 | 0.117747 | 28.366 | SAC tune logs were not retained separately, so only pre-eval wall-clock is available |
| TD3 | `lowlr1e4_noise0p1_30k` | 20000 | 0.900 | 0.007631 | 0.018975 | 41.659 | total wall-clock for this 30k run was `85.0 s` |

## Full Training Results

| algorithm | best step | best success_rate | best_goal_mean | best final_goal_mean | wall_clock_to_best_eval_start_s | final step | final success_rate | final best_goal_mean | final final_goal_mean | wall_clock_to_final_eval_start_s | total_wall_clock_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAC | 210000 | 0.700 | 0.056294 | 0.064555 | 540.833 | 300000 | 0.200 | 0.063283 | 0.149549 | 831.896 | 853.3 |
| TD3 | 30000 | 1.000 | 0.008437 | 0.008437 | 48.958 | 300000 | 0.700 | 0.049028 | 0.074556 | 532.549 | 539.6 |

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
