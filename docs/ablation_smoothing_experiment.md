# Ablation: Dual-Smoothing Experiment

**Date**: 2026-02-06 15:26 CST
**Last updated**: 2026-02-06 17:30 CST

---

## Objective

Generate NMSE and HamRes metrics for 2DoF and 3DoF systems across 4 torque policies and 30 trajectory lengths (50–1500), using a dual-smoothing strategy during diffusion sampling. Produce combined plots with mean +/- std bands (linear scale).

> Note: This is a historical experiment note. The old plotting scripts referenced below were removed during cleanup.
> For the active pipeline, use `docs/metrics_memo.md` and Protocol-2 transformer scripts.

---

## Smoothing Strategy

Gaussian smoothing (`gaussian_filter1d`, sigma=5) applied to the predicted clean trajectory `x0` during DDIM sampling:

- **Unguided** (no HNN guidance): smooth after **every** sampling step (1x per step)
- **Guided** (HNN guidance active): smooth after every sampling step; on steps where guidance is also applied (steps 45–49 of 50), smooth **twice** — once after prediction, once after guidance optimization

Implementation: `src/models/trajectory_dpf.py`, DDIM sampler block (lines ~996–1042).

**Previous approach** (superseded): unguided had no smoothing (sigma=0), guided smoothed only after guidance steps.

---

## Configuration

### Systems

| | 2DoF | 3DoF |
|---|---|---|
| DPF checkpoint | `checkpoints/2dof/trajectory_dpf_...val_loss=0.0008.ckpt` | `checkpoints/trajectory_dpf_...val_loss=0.0010.ckpt` |
| HNN checkpoint | `checkpoints/2dof/SeperableHNN-2DOF-epoch-epoch=999.ckpt` | `checkpoints/SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt` |
| MuJoCo XML | `configs/rigid_arm_hinge_2dof.xml` | `configs/rigid_arm_hinge.xml` |
| qpos_dim | 2 | 3 |

### Guidance Parameters

| | guidance_steps | guidance_lr | guidance_after_steps |
|---|---|---|---|
| 2DoF | 10 | 0.0001 | 45 |
| 3DoF | 25 | 0.01 | 45 |

### Torque Data (unified source)

All torque data from 3DoF files (shape [1000, 1500, 3]). 2DoF slices first 2 dims (`[:, :, :2]`).

| Policy | Source file |
|--------|------------|
| sinusoidal | `data/sinusoidal_torques_1000_L1500.h5` |
| gp | `data/gp_torques_1000_L1500.h5` |
| spline | `data/spline_torques_1000_L1500.h5` |
| zero | zeros (generated) |

### Constants

- DATA_DT = 0.0002
- SIM_DT = 0.0001
- smooth_sigma = 5.0
- num_samples = 100
- DDIM sampler, 50 steps
- Trajectory lengths: 50, 100, 150, ..., 1500 (30 values)

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/compute_ablation_2dof_with_smoothing.py` | Generate trajectories (unguided + guided) and compute NMSE/HamRes metrics |
| `scripts/compute_hnn_rollout_nmse_unified.py` | Compute HNN forward rollout NMSE baseline |
| *(removed)* `scripts/plot_hamres_combined.py` | Old HamRes combined plot script (deprecated) |
| *(removed)* `scripts/plot_nmse_combined.py` | Old NMSE combined plot script (deprecated) |

---

## Execution

### DPF metric generation (8 jobs, 4 GPUs)

Each GPU runs one policy: 2DoF first, then 3DoF sequentially.

```
GPU 0: sinusoidal (2DoF -> 3DoF)
GPU 1: gp         (2DoF -> 3DoF)
GPU 2: zero       (2DoF -> 3DoF)
GPU 3: spline     (2DoF -> 3DoF)
```

Commands:
```bash
cd /home/gsang/Projects/Perceiver_IO
# Per GPU (example for GPU 0):
CUDA_VISIBLE_DEVICES=0 python scripts/compute_ablation_2dof_with_smoothing.py \
  --system 2dof --policy sinusoidal --smooth_sigma 5.0 --num_samples 100
CUDA_VISIBLE_DEVICES=0 python scripts/compute_ablation_2dof_with_smoothing.py \
  --system 3dof --policy sinusoidal --smooth_sigma 5.0 --num_samples 100
```

### HNN rollout baseline (8 jobs)

```bash
python scripts/compute_hnn_rollout_nmse_unified.py --system 2dof --policy sinusoidal
# ... repeat for all (system, policy) combinations
```

### Plot generation

```bash
# historical scripts removed
# use current Protocol-2 plotting pipeline:
bash scripts/run_protocol2_transformer_vs_dpf_hnn.sh
```

---

## Output Files

### Data (CSV)

DPF metrics (8 files, 30 rows each):
- `output_ablation/results/2dof_smoothed/metrics_{sinusoidal,gp,zero,spline}_sigma5.0.csv`
- `output_ablation/results/3dof_smoothed/metrics_{sinusoidal,gp,zero,spline}_sigma5.0.csv`

Columns: `sigma, trajectory_length, {unguided,guided}_{nmse_q,nmse_p,hamres}_{mean,std,p25,median,p95,p99}`

HNN rollout baseline (8 files, 30 rows each):
- `output_ablation/results/2dof_smoothed/hnn_rollout_nmse_{sinusoidal,gp,zero,spline}.csv`
- `output_ablation/results/3dof_smoothed/hnn_rollout_nmse_{sinusoidal,gp,zero,spline}.csv`

Columns: `trajectory_length, hnn_nmse_q_mean, hnn_nmse_q_std, hnn_nmse_p_mean, hnn_nmse_p_std`

### Plots

- `plots/ablation_hamres_combined.png` — 1x2, linear scale, L≤1100
- `plots/ablation_nmse_combined.png` — 2x2, linear scale, L≤1100, includes HNN rollout

---

## Key Code Change

`src/models/trajectory_dpf.py` DDIM sampler — added per-step smoothing block before guidance:

```python
# After x0 = self._predict_x0(x_t, eps, a_bar_t):

# [NEW] Smooth x0 at every sampling step
if smooth_sigma > 0:
    x0_phys = self.denormalize_state(x0)
    x0_smoothed = gaussian_filter1d(x0_phys.cpu().numpy(), sigma=smooth_sigma, axis=1)
    x0_phys = torch.tensor(x0_smoothed, dtype=x0_phys.dtype, device=x0_phys.device)
    x0 = self.normalize_state(x0_phys)

# [EXISTING] Guidance block (unchanged) — includes its own post-guidance smoothing
if hnn is not None and guidance_steps > 0 and ...:
    ...guidance optimization...
    if smooth_sigma > 0:
        ...smooth again after guidance...
```

`scripts/compute_ablation_2dof_with_smoothing.py` — unguided now passes `smooth_sigma=sigma` (was hardcoded to 0.0), moved inside sigma loop.

---

## Preliminary Results (Run 1 — without seed control)

> **Important caveat**: This run does NOT share initial noise between unguided and guided.
> Each call to `sample_trajectories()` draws independent `torch.randn()` noise.
> Guided sample y_n and unguided sample x_n are **not paired** — they start from different
> random states. The guided vs unguided comparison confounds guidance effect with noise
> randomness. A second run with fixed `initial_noise` is needed for proper paired comparison.

### Outlier Analysis

**NMSE distributions are extremely heavy-tailed.** Out of 1440 (system, policy, length, metric) measurements, 857 (59.5%) have mean > 2× median. The worst case: 2dof zero L=300 NMSE_p has mean/median = 16,788×. **Mean is not a reliable summary statistic for NMSE; median should be used.**

**HamRes is well-behaved.** Only 4 out of 480 measurements have mean > 2× median:
- 2dof spline L=800 unguided: mean=96.84, median=0.707 (mean/median=137×, single exploding trajectory)
- 3 mild cases at L=50 (startup transients, ratio 2.3–2.5×)

### Guidance Effect on HamRes (using median)

| System | Guidance improves HamRes | Typical ratio (guided/unguided) |
|--------|--------------------------|--------------------------------|
| **2DoF** | **100%** (120/120) | **0.43** (57% reduction) |
| **3DoF** | **2.5%** (3/120) | **2.00** (doubles HamRes) |

2DoF guidance dramatically reduces Hamiltonian residual. 3DoF guidance consistently **worsens** HamRes — the guided median is ~2× the unguided median across all policies and lengths.

### Guidance Effect on NMSE (using median)

| System | NMSE_q improves | NMSE_p improves | All 3 metrics improve |
|--------|-----------------|-----------------|----------------------|
| **2DoF** | 23.3% | 48.3% | 7.5% |
| **3DoF** | 34.2% | 25.0% | 0.8% |

Typical guided/unguided median ratios:

| System | NMSE_q ratio | NMSE_p ratio |
|--------|-------------|-------------|
| 2DoF | 1.59 (worse) | 1.06 (neutral) |
| 3DoF | 1.26 (worse) | 1.57 (worse) |

Guidance generally **worsens** NMSE. Only 10 out of 240 combinations (4.2%) see all three metrics (HamRes + NMSE_q + NMSE_p) improve simultaneously; 9 of those are from 2DoF.

### Summary by System

**2DoF**: Guidance is a clear win for physics consistency (HamRes ↓57%) at the cost of slightly worse trajectory accuracy (NMSE_q ↑59%, NMSE_p neutral). This is the expected tradeoff — guidance pushes trajectories toward Hamiltonian-consistent dynamics, which may differ from the MuJoCo ground truth due to HNN approximation errors.

**3DoF**: Guidance is harmful across all metrics. HamRes doubles, NMSE worsens. This suggests the 3DoF guidance parameters (lr=0.01, 25 steps) are too aggressive, or the 3DoF HNN has higher approximation errors that corrupt the guidance signal. **The 3DoF guidance parameters need to be re-tuned.**

### Policy Dependence

All four torque policies (sinusoidal, GP, zero, spline) behave similarly within each system. **System (2dof vs 3dof) is the dominant factor**, not the torque policy.

---

## Known Issues and Next Steps

1. **No seed control (critical)**: Unguided and guided samples use independent random noise. Must fix by passing shared `initial_noise` to `sample_trajectories()` and rerun. The `initial_noise` parameter already exists in the API.

2. **Outlier-dominated NMSE means**: Plots should use **median + p25–p75 bands** instead of mean ± std for NMSE. The CSV already contains percentile columns; only plot scripts need modification.

3. **3DoF guidance degrades all metrics**: Need to either re-tune 3DoF guidance parameters (lower lr, fewer steps) or investigate whether the 3DoF HNN quality is sufficient for guidance.

4. **HNN guidance is non-unique**: The energy landscape `E = Σ MSE(Hamilton residuals)` is non-convex. Different initial noise → different x0 → Adam converges to different local minima. This is another reason why paired comparison (shared noise) is important.
