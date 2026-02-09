# Metrics Computation Memo: HamRes and NMSE

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

- `dq/dt`, `dp/dt`: central difference from smoothed trajectory (`sigma=1.0` Gaussian by default)
- `dH/dp`, `dH/dq`: autograd from pre-trained HNN
- `τ`: external torque
- `scale_q`, `scale_p`: per-dimension normalization scales (prefer HNN training stats if per-dim; otherwise derivative std from current trajectory)
- `eps_q`, `eps_p`: denominator floors (`1e-3` default)
- `delta`: pseudo-Huber transition parameter (`1.0` default)

### Why robust HamRes

- Finite differences amplify small high-frequency noise in `q` and `p`.
- Position and momentum channels often have different scales.
- Squared-loss + mean aggregation can be dominated by a small number of spikes.

The robust version (smoothing + per-dim normalization + pseudo-Huber + median over time) is much more stable for trajectory comparison.

---

## 2. NMSE (Normalized Mean Squared Error)

### Definition
NMSE measures trajectory accuracy relative to MuJoCo physics reconstruction, computed **per-dimension** then averaged:

```
NMSE_q = mean_d( MSE_t(q_gen[:,d], q_recon[:,d]) / Var_t(q_recon[:,d]) )
NMSE_p = mean_d( MSE_t(p_gen[:,d], p_recon[:,d]) / Var_t(p_recon[:,d]) )

where:
  d: dimension index (e.g. 0,1,2 for 3DoF)
  MSE_t: mean squared error over time axis
  Var_t: variance over time axis (of GT reconstruction)
  q_gen, p_gen: generated trajectory (position, momentum)
  q_recon, p_recon: MuJoCo reconstructed trajectory from initial state + torques
```

Per-dimension normalization ensures each degree of freedom contributes equally regardless of scale. The final NMSE is the mean of per-dimension NMSE values.

---

## 3. HNN Forward Rollout NMSE

Symplectic Euler integration from initial state, compared against MuJoCo ground truth:

```
p_new = p + dt * (-dH/dq + τ)
q_new = q + dt * dH/dp
```

This serves as a baseline: pure HNN rollout without the diffusion model.

---

## 4. Current Scripts

| Script | Purpose |
|--------|---------|
| `scripts/compute_ablation_2dof_with_smoothing.py` | Generate DPF trajectories (unguided + guided) and compute NMSE/HamRes metrics for both 2DoF and 3DoF |
| `scripts/compute_hnn_rollout_nmse_unified.py` | Compute HNN forward rollout NMSE baseline for both systems |
| `scripts/plot_hamres_combined.py` | HamRes plot: 1x2 (2DoF, 3DoF), mean +/- std bands, linear scale |
| `scripts/plot_nmse_combined.py` | NMSE plot: 2x2 (2DoF/3DoF x q/p), mean +/- std bands + HNN rollout, linear scale |

### Usage

```bash
# DPF metrics (per GPU, per policy)
CUDA_VISIBLE_DEVICES=0 python scripts/compute_ablation_2dof_with_smoothing.py \
  --system 2dof --policy sinusoidal --smooth_sigma 5.0 --num_samples 100

# HNN rollout baseline
python scripts/compute_hnn_rollout_nmse_unified.py --system 2dof --policy sinusoidal

# Plots
python scripts/plot_hamres_combined.py
python scripts/plot_nmse_combined.py
```

---

## 5. System Configuration

| | 2DoF | 3DoF |
|---|---|---|
| DPF checkpoint | `checkpoints/2dof/trajectory_dpf_...val_loss=0.0008.ckpt` | `checkpoints/trajectory_dpf_...val_loss=0.0010.ckpt` |
| HNN checkpoint | `checkpoints/2dof/SeperableHNN-2DOF-epoch-epoch=999.ckpt` | `checkpoints/StructuredHNN-dim256-epoch-epoch=749.ckpt` |
| MuJoCo XML | `configs/rigid_arm_hinge_2dof.xml` | `configs/rigid_arm_hinge.xml` |
| qpos_dim | 2 | 3 |
| guidance_steps | 10 | 25 |
| guidance_lr | 0.0001 | 0.01 |
| guidance_after_steps | 45 | 45 |

### Torque Data

All torques sourced from 3DoF files (shape [1000, 1500, 3]). 2DoF slices `[:, :, :2]`.

| Policy | Source file |
|--------|------------|
| sinusoidal | `data/sinusoidal_torques_1000_L1500.h5` |
| gp | `data/gp_torques_1000_L1500.h5` |
| spline | `data/spline_torques_1000_L1500.h5` |
| zero | zeros (generated) |

### Constants

- DATA_DT = 0.0002, SIM_DT = 0.0001
- smooth_sigma = 5.0 (Gaussian smoothing on x0 during DDIM sampling)
- num_samples = 100, DDIM 50 steps
- Trajectory lengths: 50, 100, 150, ..., 1500 (30 values)

---

## 6. Output Files

### DPF Metrics (8 CSV files, 30 rows each)

```
output_ablation/results/
├── 2dof_smoothed/
│   └── metrics_{sinusoidal,gp,zero,spline}_sigma5.0.csv
└── 3dof_smoothed/
    └── metrics_{sinusoidal,gp,zero,spline}_sigma5.0.csv
```

Columns: `sigma, trajectory_length, {unguided,guided}_{nmse_q,nmse_p,hamres}_{mean,std,p25,median,p95,p99}`

### HNN Rollout Baseline (8 CSV files, 30 rows each)

```
output_ablation/results/
├── 2dof_smoothed/
│   └── hnn_rollout_nmse_{sinusoidal,gp,zero,spline}.csv
└── 3dof_smoothed/
    └── hnn_rollout_nmse_{sinusoidal,gp,zero,spline}.csv
```

Columns: `trajectory_length, hnn_nmse_q_mean, hnn_nmse_q_std, hnn_nmse_p_mean, hnn_nmse_p_std`

### Plots

- `plots/ablation_hamres_combined.png` — 1x2 (2DoF, 3DoF), mean +/- std bands
- `plots/ablation_nmse_combined.png` — 2x2 (rows: 2DoF/3DoF, cols: NMSE_q/NMSE_p), mean +/- std bands + HNN rollout

---

## 7. Quick Reference

| Metric | What it measures | Normalization | Lower is better |
|--------|------------------|---------------|-----------------|
| HamRes | Physics consistency (Hamilton's eqs) | HNN training variances | Yes |
| NMSE | Trajectory accuracy vs MuJoCo | Per-trajectory variance | Yes |

### GT Reference
- HamRes: ~0.0003 (near zero, perfect physics)
- NMSE: 0 (by definition, GT = reconstruction)
