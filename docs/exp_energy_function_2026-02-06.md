# Experiment: HNN Energy Function Analysis

**Date**: 2026-02-06
**Systems**: 2-DoF, 3-DoF rigid arm
**Torque policy**: sinusoidal
**Smoothing**: disabled (sigma=0)

---

## Objective

Investigate the relationship between the HNN guidance energy function and trajectory quality metrics (NMSE, HamRes). Two sub-experiments:

1. **Pilot: forward diff vs central diff guidance** — Does the finite-difference scheme matter?
2. **Single-trajectory energy minimization** — What happens when we push the energy to its minimum?

---

## Background

The HNN guidance energy is defined as:

```
E = MSE(dq/dt - dH/dp) + MSE(dp/dt - (-dH/dq + τ))
```

where time derivatives are approximated by either:
- **Central difference**: `(s[t+1] - s[t-1]) / (2Δt)` (2-step method, susceptible to parasitic oscillations in theory)
- **Forward difference**: `(s[t+1] - s[t]) / Δt` (1-step method, no parasitic solutions)

HamRes evaluation always uses central difference, regardless of which scheme is used for guidance.

---

## Experiment 1: Forward Diff vs Central Diff Pilot

### Setup

- 200 samples, fixed seeds (shared `initial_noise` between variants)
- Trajectory lengths: 100, 300, 500, 700, 1000
- 3 variants: unguided, guided (central diff), guided (forward diff)
- Guidance parameters: 2DoF (10 steps, lr=0.0001, after step 45); 3DoF (25 steps, lr=0.01, after step 45)

### Results — 2DoF (median)

| L | NMSE_q | | | NMSE_p | | | HamRes | | |
|--:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| | unguid | central | forward | unguid | central | forward | unguid | central | forward |
| 100 | 4.9e-6 | 1.3e-6 | 1.2e-6 | 8.0e-6 | 2.8e-6 | 2.8e-6 | 309 | 42 | 51 |
| 300 | 5.2e-6 | 1.5e-6 | 1.5e-6 | 1.3e-5 | 6.5e-6 | 6.3e-6 | 296 | 16 | 22 |
| 500 | 1.4e-5 | 7.4e-6 | 6.7e-6 | 2.4e-5 | 1.3e-5 | 1.4e-5 | 268 | 11 | 16 |
| 700 | 3.8e-5 | 2.2e-5 | 2.2e-5 | 3.3e-5 | 2.1e-5 | 2.1e-5 | 335 | 12 | 17 |
| 1000 | 1.1e-4 | 8.3e-5 | 8.1e-5 | 7.8e-5 | 6.7e-5 | 6.7e-5 | 509 | 17 | 23 |

2DoF: Guidance dramatically improves all metrics. Central and forward diff produce nearly identical NMSE. Central diff achieves slightly lower HamRes (expected — HamRes evaluation uses central diff).

### Results — 3DoF (median)

| L | NMSE_q | | | NMSE_p | | | HamRes | | |
|--:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| | unguid | central | forward | unguid | central | forward | unguid | central | forward |
| 100 | 2.2e-4 | 1.0e-4 | 1.1e-4 | 1.0e-3 | 2.2e-4 | 1.5e-4 | 0.68 | 2.35 | 2.22 |
| 300 | 1.5e-3 | 5.0e-4 | 3.9e-4 | 1.5e-3 | 3.7e-4 | 3.6e-4 | 0.47 | 2.10 | 2.02 |
| 500 | 3.7e-3 | 1.0e-3 | 1.2e-3 | 2.9e-3 | 8.5e-4 | 1.0e-3 | 0.45 | 2.02 | 1.99 |
| 700 | 6.0e-3 | 2.6e-3 | 2.1e-3 | 5.7e-3 | 2.6e-3 | 2.2e-3 | 0.44 | 2.04 | 2.02 |
| 1000 | 1.9e-2 | 7.1e-3 | 7.9e-3 | 1.8e-2 | 8.3e-3 | 8.5e-3 | 0.72 | 2.17 | 2.07 |

3DoF: Guidance improves NMSE (2-5x) but consistently **worsens** HamRes (4-5x). Forward and central diff produce very similar results — no practical difference.

### Pilot conclusion

**Forward diff vs central diff makes negligible practical difference.** The theoretical concern about parasitic solutions from central diff does not manifest in practice with the current guidance schedule (only 10-25 Adam steps during the last 5 diffusion steps). The dominant factor is the system (2DoF vs 3DoF), not the finite-difference scheme.

---

## Experiment 2: Single-Trajectory Energy Minimization

### Motivation

The pilot showed guidance helps NMSE but worsens HamRes for 3DoF. To understand why, we take one trajectory and run Adam optimization directly on the energy function until convergence, tracking how NMSE and HamRes evolve.

### Setup

- 1 trajectory from unguided DPF sampling (seed=42, sinusoidal torque, L=500)
- Adam optimizer on `(seq_qpos, seq_mom)` minimizing HNN physics energy
- Convergence: `|ΔE/E| < 1e-7` for 100 consecutive steps, or max 20000 steps
- Both central and forward diff energy variants tested
- NMSE computed against each trajectory's own MuJoCo GT (re-simulated from the trajectory's initial state after optimization)
- **2DoF**: lr=0.0001, converged at ~5000 steps
- **3DoF**: lr=0.0001 (reduced from default 0.01 to avoid Adam oscillation)

### Results — 2DoF

Both central and forward diff converge to E ≈ 0, with all metrics improving monotonically.

**Central diff:**

| Step | Energy | NMSE_q | NMSE_p | HamRes |
|-----:|-------:|-------:|-------:|-------:|
| 0 | 227.20 | 2.80e-5 | 8.61e-5 | 580.19 |
| 500 | 0.151 | 3.79e-6 | 2.29e-5 | 0.413 |
| 1000 | 0.044 | 2.72e-6 | 2.09e-5 | 0.120 |
| 2000 | 0.017 | 1.39e-6 | 1.87e-5 | 0.047 |
| 3000 | 0.013 | 7.21e-7 | 1.73e-5 | 0.035 |
| 5000 | 0.008 | 6.78e-7 | 1.24e-5 | 0.022 |

**Forward diff:**

| Step | Energy | NMSE_q | NMSE_p | HamRes |
|-----:|-------:|-------:|-------:|-------:|
| 0 | 828.77 | 2.80e-5 | 8.61e-5 | 580.19 |
| 500 | 0.476 | 2.55e-6 | 1.86e-5 | 1.276 |
| 1000 | 0.126 | 2.17e-6 | 1.68e-5 | 0.341 |
| 2000 | 0.029 | 1.33e-6 | 1.63e-5 | 0.079 |
| 3000 | 0.016 | 9.40e-7 | 1.60e-5 | 0.043 |
| 5000 | 0.012 | 4.86e-7 | 1.46e-5 | 0.030 |

**2DoF conclusion**: E → 0 implies NMSE → 0 and HamRes → 0. The 2DoF HNN is accurate enough that its energy landscape aligns with true physics. All metrics improve monotonically — no over-optimization.

### Results — 3DoF

**Central diff (lr=0.0001):**

| Step | Energy | NMSE_q | NMSE_p | HamRes |
|-----:|-------:|-------:|-------:|-------:|
| 0 | 47.58 | 2.39e-4 | 3.12e-4 | 0.122 |
| 100 | **25.06** | **1.10e-4** | **1.07e-4** | 0.066 |
| 500 | 22.95 | 2.67e-4 | 3.75e-4 | 0.060 |
| 1000 | 20.41 | 6.77e-4 | 1.26e-3 | 0.053 |
| 2000 | 15.55 | 2.12e-3 | 4.80e-3 | 0.040 |
| 3000 | 11.15 | 3.59e-3 | 9.40e-3 | 0.029 |
| 5000 | 4.45 | 1.05e-3 | **1.20e-2** | 0.012 |

**Forward diff (lr=0.0001):**

| Step | Energy | NMSE_q | NMSE_p | HamRes |
|-----:|-------:|-------:|-------:|-------:|
| 0 | 70.39 | 2.39e-4 | 3.12e-4 | 0.122 |
| 100 | **24.91** | **1.21e-4** | **1.20e-4** | 0.066 |
| 500 | 23.26 | 4.02e-4 | 5.34e-4 | 0.061 |
| 1000 | 21.52 | 7.15e-4 | 1.12e-3 | 0.056 |
| 2000 | 18.07 | 1.14e-3 | 2.27e-3 | 0.047 |
| 3000 | 14.42 | 1.37e-3 | 3.48e-3 | 0.038 |
| 5000 | 7.04 | 5.73e-4 | **6.12e-3** | 0.018 |

**3DoF key observations:**

1. **Energy decreases monotonically** in both schemes, confirming Adam converges properly with lr=0.0001 (unlike lr=0.01 which caused oscillation).

2. **NMSE_q shows a U-curve**: best at step ~100, then worsens. By step 5000, NMSE_q is worse than step 0 for central diff (1.05e-3 vs 2.39e-4), though forward diff shows partial recovery at step 5000.

3. **NMSE_p worsens monotonically** after step 100: from 1.07e-4 (step 100) to 1.20e-2 (step 5000) — a **112× degradation** for central diff. Forward diff is similar: 1.20e-4 → 6.12e-3, a **51× degradation**.

4. **HamRes improves monotonically**: 0.122 → 0.012 (10×). This shows the optimizer is successfully making the trajectory more consistent with the HNN's Hamiltonian — but the HNN's Hamiltonian doesn't match true physics.

5. **Forward diff is slightly less destructive**: at step 5000, forward diff NMSE_p (6.12e-3) is about half of central diff NMSE_p (1.20e-2). Forward diff energy also converges more slowly (7.04 vs 4.45), which may explain the less severe NMSE degradation.

### Learning rate comparison (3DoF central diff)

A separate run with lr=0.01 (the original 3DoF default) showed energy oscillating at E ≈ 3-5 after 5000 steps, with Adam unable to converge smoothly. Reducing to lr=0.0001 confirmed that:
- Energy CAN be reduced further (monotonically, no oscillation)
- But lower energy does NOT mean better trajectory — NMSE worsens regardless of lr
- The lr=0.01 instability was masking the fundamental over-optimization problem

---

## Key Findings

### 1. Forward diff vs central diff: negligible difference

The theoretical concern about parasitic oscillatory solutions from the central difference scheme does not materialize in practice. Both schemes produce nearly identical guidance outcomes. This is likely because:
- The guidance runs for very few Adam steps (10-25) during a limited window of diffusion steps
- The trajectories are already fairly smooth from the diffusion model
- Any parasitic modes are small relative to the signal

### 2. The 2DoF HNN is accurate; the 3DoF HNN has systematic errors

**2DoF**: Pushing energy to zero drives all metrics (NMSE_q, NMSE_p, HamRes) to zero. The HNN's learned Hamiltonian is a faithful approximation of the true system dynamics. Guidance is unambiguously beneficial.

**3DoF**: Energy minimization worsens NMSE after an initial improvement. The HNN's learned Hamiltonian has systematic approximation errors. Trajectories that satisfy the HNN's Hamilton equations diverge from the true MuJoCo dynamics. The HamRes metric (computed using the same HNN) decreases, but this is self-referential — the trajectory is becoming more consistent with a wrong model.

### 3. Optimal guidance is light-touch, not energy minimization

For 3DoF, the best trajectory quality occurs at step ~100 (E ≈ 25), not at the energy minimum (E ≈ 4.5). This suggests:
- Guidance should apply a limited number of optimization steps to avoid over-optimization
- The current 3DoF guidance schedule (25 steps × 5 diffusion steps = 125 effective steps) may already be in the over-optimization regime
- A stopping criterion based on NMSE (if available) or a fixed step budget is more appropriate than convergence to the energy minimum

### 4. 3DoF guidance parameters need re-tuning

The lr=0.01 default for 3DoF is too aggressive on two fronts:
- It causes Adam oscillation (energy fails to converge smoothly)
- Even with smaller lr, the fundamental over-optimization problem means fewer steps (not smaller lr) is the right fix

---

## Implications for the Ablation Study

1. **No need to switch to forward diff** — central diff works equally well in practice
2. **3DoF guidance parameters should be reduced** — fewer steps or earlier stopping to stay in the beneficial region (E ≈ 20-25 for this trajectory)
3. **The 3DoF HamRes improvement from guidance is misleading** — it measures consistency with the HNN (which has errors), not with true physics
4. **Future work**: investigate whether the 3DoF HNN can be improved (more training, larger model) to make guidance as effective as for 2DoF

---

## Scripts Used

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/pilot_fwd_vs_central.py` | 200-sample pilot test (fwd vs central) | Deleted after report |
| `scripts/test_energy_vs_nmse.py` | Single-trajectory energy optimization | Deleted after report |
| `scripts/plot_energy_vs_nmse.py` | Metric evolution plots | Deleted after report |

## Output Files (deleted)

- `output_ablation/results/pilot_fwd_vs_central/{2dof,3dof}_sinusoidal.csv`
- `output_ablation/results/energy_vs_nmse/{2dof,3dof}_{central,forward}.csv`
- `plots/energy_vs_nmse.png`, `plots/energy_traj_{2dof,3dof}.png`
- Log files: `output_ablation/{pilot,test_energy}_*.log`
