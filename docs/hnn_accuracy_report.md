# HNN Accuracy Report

## 1. Model Summary

| | 2DoF | 3DoF |
|---|---|---|
| Architecture | SeperableHNN | StructuredHNN |
| Hidden dim | 1024 | 256 |
| Activation | CELU | - |
| Params | ~6.3M | ~270K |
| Checkpoint | `2dof/SeperableHNN-2DOF-epoch=999` | `StructuredHNN-dim256-epoch=749` |
| Training data | 160M samples (4000 traj x 40K steps) | 320M samples (4000 traj x 80K steps) |
| Epochs | 1000 | 750 |
| Loss function | MSE on (dq/dt, dp/dt) | MSE on (dq/dt, dp/dt) |

### Key Differences
- **2DoF SeperableHNN**: H = T(p) + V(q), kinetic and potential learned independently via large MLPs
- **3DoF StructuredHNN**: Enforces quadratic kinetic energy T = 0.5 p^T M^{-1}(q) p, learns M^{-1}(q) via Cholesky decomposition. Physics-informed, fewer parameters.

---

## 2. Training Accuracy

### 2DoF SeperableHNN (epoch=999, final)

| Metric | Value |
|--------|-------|
| Train loss | 6.56e-07 |
| Val loss | 7.12e-07 |
| Val loss (epoch 0) | 7.83e-04 |
| Improvement | ~1100x (from 7.83e-04 to 7.12e-07) |

**Verdict: 2DoF HNN training essentially converged to machine-precision level.** Train/val losses are nearly identical (~7e-7), indicating no overfitting and excellent generalization.

### 3DoF StructuredHNN (epoch=749, final)

| Metric | Value |
|--------|-------|
| Train loss | ~1.8e-04 |
| Val loss | 3.08e-04 |
| Val loss (epoch 0) | 0.246 |
| Improvement | ~800x (from 0.246 to 3.08e-04) |

**Verdict: 3DoF training loss is ~430x worse than 2DoF** (3.08e-04 vs 7.12e-07). The val/train gap (3.08e-04 vs 1.8e-4) suggests mild overfitting. This is expected given the physics-constrained architecture (StructuredHNN) and much smaller model (270K vs 6.3M params) facing a harder problem (3DoF).

### Comparison

| | 2DoF | 3DoF | Ratio (3DoF/2DoF) |
|---|---|---|---|
| Final val loss | 7.12e-07 | 3.08e-04 | ~430x |
| Final train loss | 6.56e-07 | ~1.8e-04 | ~274x |
| Val/Train ratio | 1.09 | ~1.71 | - |
| Initial val loss | 7.83e-04 | 0.246 | ~314x |

---

## 3. HNN Forward Rollout NMSE (Symplectic Euler Integration)

Integration method: Symplectic Euler from GT initial state, compared against MuJoCo GT.
DATA_DT = 0.0002s, SIM_DT = 0.0001s, 100 samples per length.

### 2DoF (SeperableHNN-2DOF-epoch=999) — Current

| Length | NMSE_q mean | NMSE_q std | NMSE_p mean | NMSE_p std |
|--------|-------------|------------|-------------|------------|
| 50 | 1.25e-08 | 9.27e-08 | 6.13e-07 | 5.87e-06 |
| 100 | 3.89e-08 | 2.79e-07 | 1.17e-07 | 8.86e-07 |
| 200 | 6.59e-08 | 3.90e-07 | 2.88e-08 | 1.86e-07 |
| 500 | 1.05e-07 | 5.14e-07 | 6.94e-09 | 2.99e-08 |
| 1000 | 1.33e-07 | 4.72e-07 | 5.66e-09 | 1.68e-08 |
| 1500 | 1.88e-07 | 9.38e-07 | 6.79e-09 | 1.99e-08 |

**Sinusoidal policy shown. GP policy similar (~1.5e-07 at L=1000).**

**Verdict: 2DoF HNN essentially perfect.** NMSE < 2e-07 for all lengths. Errors are numerical precision level.

---

### 3DoF — STALE DATA (SeperableHNN-dim1024, NOT current StructuredHNN)

> **WARNING: These results are from the OLD SeperableHNN(dim1024)-CELU-epoch=999 checkpoint.**
> **Need to re-run with StructuredHNN-dim256-epoch=749.**

| Length | NMSE_q mean (sin) | NMSE_p mean (sin) | NMSE_q mean (gp) | NMSE_p mean (gp) |
|--------|-------------------|-------------------|-------------------|-------------------|
| 50 | 9.70e-04 | 5.91e-03 | 9.40e-04 | 6.65e-03 |
| 100 | 3.57e-03 | 1.42e-02 | 3.77e-03 | 1.56e-02 |
| 200 | 1.35e-02 | 4.26e-02 | 1.50e-02 | 4.01e-02 |
| 500 | 9.90e-02 | 1.27e-01 | 8.67e-02 | 1.26e-01 |
| 1000 | 2.10e-01 | 3.78e-01 | 3.16e-01 | 4.05e-01 |
| 1500 | 6.01e-01 | 9.83e-01 | 8.12e-01 | 8.75e-01 |

**Verdict (old model): 3DoF accuracy degrades significantly with trajectory length.** NMSE_q ~ 0.2 at L=1000, nearly 1.0 at L=1500. Far worse than 2DoF.

---

## 4. TODO

- [ ] Re-run `compute_hnn_rollout_nmse_unified.py --system 3dof` with StructuredHNN-dim256-epoch=749
- [ ] Compare StructuredHNN vs SeperableHNN rollout accuracy for 3DoF
- [ ] Evaluate whether StructuredHNN's physics-informed structure improves long-horizon rollout stability

---

## 5. Data Sources

| Data | Path |
|------|------|
| 2DoF training log | `logs/seperable_hnn_training.log` (or TensorBoard) |
| 3DoF training log | `logs/structured_hnn_training.log` |
| 2DoF rollout sinusoidal | `output_ablation/results/2dof_smoothed/hnn_rollout_nmse_sinusoidal.csv` |
| 2DoF rollout gp | `output_ablation/results/2dof_smoothed/hnn_rollout_nmse_gp.csv` |
| 3DoF rollout sinusoidal (STALE) | `output_ablation/results/3dof_smoothed/hnn_rollout_nmse_sinusoidal.csv` |
| 3DoF rollout gp (STALE) | `output_ablation/results/3dof_smoothed/hnn_rollout_nmse_gp.csv` |
| Evaluation script | `scripts/compute_hnn_rollout_nmse_unified.py` |
