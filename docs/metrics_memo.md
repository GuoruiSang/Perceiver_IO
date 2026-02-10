# Metrics Computation Memo: Robust HamRes and NRMSE(range)

Last updated: 2026-02-10

## 1. HamRes (Hamiltonian Residual)

### Definition
HamRes measures how well a trajectory satisfies Hamilton's equations, using a robust formulation:

```
r_q = dq/dt - dH/dp
r_p = dp/dt - (-dH/dq + τ)

r_q_norm[d] = r_q[d] / (scale_q[d] + eps_q)
r_p_norm[d] = r_p[d] / (scale_p[d] + eps_p)

rho(x) = delta^2 * (sqrt(1 + (x / delta)^2) - 1)   # pseudo-Huber

HamRes_t = mean_d( rho(r_q_norm[d]) + rho(r_p_norm[d]) )
HamRes = median_t( HamRes_t )
```

- `dq/dt`, `dp/dt`: central difference from smoothed trajectory (`sigma=1.0` by default)
- `dH/dp`, `dH/dq`: autograd from pre-trained HNN
- `τ`: external torque
- `scale_q`, `scale_p`: per-dimension normalization scales
- `eps_q`, `eps_p`: denominator floors (`1e-3` default)
- `delta`: pseudo-Huber transition parameter (`1.0` default)

### Why robust HamRes

- Finite differences amplify small high-frequency noise in `q` and `p`.
- Position and momentum channels often have different scales.
- Squared loss + mean aggregation can be dominated by a few spike timesteps.

The robust version (smoothing + per-dim normalization + pseudo-Huber + median-over-time) is much more stable for trajectory comparison.

---

## 2. NRMSE(range)

### Definition
NRMSE(range) measures trajectory error relative to MuJoCo reconstruction, computed per dimension then averaged:

```
NRMSE_q = mean_d( RMSE_t(q_gen[:,d], q_recon[:,d]) / (range_t(q_recon[:,d]) + eps_q) )
NRMSE_p = mean_d( RMSE_t(p_gen[:,d], p_recon[:,d]) / (range_t(p_recon[:,d]) + eps_p) )

where:
  d: dimension index
  RMSE_t: root mean squared error over time
  range_t: max-min over time
```

- `q_gen`, `p_gen`: generated trajectory
- `q_recon`, `p_recon`: MuJoCo reconstruction from generated initial state + same torque sequence

Lower is better.

---

## 3. HNN Forward Rollout Baseline

Symplectic Euler integration from initial state:

```
p_new = p + dt * (-dH/dq + τ)
q_new = q + dt * dH/dp
```

This is a non-diffusion baseline: pure HNN rollout under the same torque sequence.

---

## 4. Current Comparison Protocol (Active)

This round uses:

- Protocol: **Protocol 2 only** (generate length 1000, then trim to target length)
- Systems: 2DoF, 3DoF
- Policies: sinusoidal, gp, spline, zero
- Lengths: 50, 100, ..., 1000
- Groups:
  - Unguided DPF
  - Guided DPF
  - Unguided Diffusion baseline
  - Guided Diffusion baseline
  - HNN rollout
- Diffusion baseline backbone: `--backbone transformer`
  - architecture details: `docs/transformer_baseline_architecture.md`

### Main scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_transformer_fixed_length.sh` | Train fixed-length diffusion baseline with transformer backbone |
| `scripts/run_fixed_time_transformer_training.sh` | Launch fixed-time transformer runs aligned to original DPF wall-clock budgets |
| `scripts/run_same_budget_guidance_search_3dof.py` | Same-budget guidance search (supports `--system 2dof/3dof`) |
| `scripts/run_same_budget_guidance_search_transformer.sh` | Convenience launcher using latest transformer ckpts for `2dof,3dof` |
| `scripts/run_protocol2_2dof_diffusion_vs_dpf_hnn_boxplots.py` | Protocol-2 evaluation + plots for 2DoF |
| `scripts/run_protocol2_3dof_diffusion_vs_dpf_boxplots.py` | Protocol-2 evaluation + plots for 3DoF |
| `scripts/run_protocol2_transformer_vs_dpf_hnn.sh` | Run both Protocol-2 evaluations using latest transformer ckpts |
| `scripts/build_protocol2_transformer_reports.py` | Build final absolute/ratio/probability summary csv + report plots |
| `scripts/run_protocol2_transformer_posttrain.sh` | One-shot post-train pipeline: search + Protocol-2 eval + final reports |

### Typical usage

```bash
# 1) Fixed-time transformer training
bash scripts/run_fixed_time_transformer_training.sh

# 2) Same-budget guidance search (2DoF + 3DoF)
bash scripts/run_same_budget_guidance_search_transformer.sh \
  --policies sinusoidal,gp,zero,spline --lengths 100,300,700,1000 --seeds 10,11

# 3) Full Protocol-2 comparison
bash scripts/run_protocol2_transformer_vs_dpf_hnn.sh

# 4) Final summary csv + report plots
python scripts/build_protocol2_transformer_reports.py

# Optional one-shot post-train pipeline
bash scripts/run_protocol2_transformer_posttrain.sh
```

---

## 5. Main Outputs (Current)

- Search outputs:
  - `output_ablation/same_budget_guidance_search_2dof_*/search_detail.csv`
  - `output_ablation/same_budget_guidance_search_3dof_*/search_detail.csv`
  - corresponding `summary_by_candidate.csv` and `best_configs.json`
- Protocol-2 outputs:
  - `output_ablation/protocol2_2dof_*`
  - `output_ablation/protocol2_3dof_*`
  - per-sample metrics csv + manifests + boxplots
- Final report outputs:
  - `output_ablation/protocol2_transformer_reports/absolute_metrics_summary.csv`
  - `output_ablation/protocol2_transformer_reports/guidance_ratio_probability_summary.csv`
  - `plots/protocol2_transformer_reports/*.png`

---

## 6. Quick Reference

| Metric | What it measures | Normalization | Lower is better |
|--------|------------------|---------------|-----------------|
| HamRes | Physics consistency (Hamilton's eqs) | robust residual normalization + pseudo-Huber + median-over-time | Yes |
| NRMSE(range) | Trajectory accuracy vs MuJoCo | RMSE / (reference range + eps) | Yes |

---

## 7. Legacy Note

Older files in this repo may still mention:

- NMSE-only summaries
- Protocol 1 (direct length generation)
- fixed-diffusion PerceiverIO baselines

For the current comparison round, follow Section 4 in this memo.
