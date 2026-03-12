# output Layout

Single root for all experiment artifacts.

## Top-level folders
- `eval_runs/`
  - Active evaluation run outputs (full metrics, manifests, trajectories).

## Naming policy
- No `protocol1/protocol2` tags in active paths.
- New runs should default to `eval_runs/<run_name>`.
- Final figures live under `plots/`; logs live under `logs/`.
