# Model: StructuredHNN

**Date**: 2026-02-06
**File**: `src/models/HNN.py` (class `StructuredHNN`)
**System**: 3-DoF rigid arm hinge (`configs/rigid_arm_hinge.xml`)

---

## Motivation

Previous `SeperableHNN` (6.3M params, 1024-wide MLP) on 3DoF showed systematic approximation errors:
- Energy minimization worsens NMSE after ~100 optimization steps (see `exp_energy_function_2026-02-06.md`)
- 2DoF HNN is accurate (E->0 implies metrics->0), but 3DoF is not
- Root cause: the MLP learns an arbitrary T(p,q), failing to capture the exact quadratic structure of kinetic energy

## Architecture

### Core Idea

For any mechanical system, kinetic energy is **exactly** quadratic in momentum:

```
T(q, p) = 0.5 * p^T @ M^{-1}(q) @ p
```

where `M(q)` is the configuration-dependent mass matrix. Instead of learning an arbitrary function T(p,q) with an MLP, we **enforce** this structure by learning `M^{-1}(q)` via Cholesky decomposition.

### Structure

```
StructuredHNN
├── Cholesky Network: q -> L(q)           (learns M^{-1} = L @ L^T, guaranteed SPD)
│   Input:  [q, sin(q), cos(q)]  (dim = 3 * coordinate_dim)
│   Output: dim*(dim+1)/2 parameters      (lower-triangular L)
│   Diagonal: softplus + 1e-4             (ensures positive definiteness)
│
├── Kinetic Energy: T = 0.5 * ||L(q)^T @ p||^2
│   (equivalent to 0.5 * p^T @ M^{-1}(q) @ p, exactly quadratic in p)
│
├── Potential Network: q -> V(q)           (learns gravity potential)
│   Input:  [q, sin(q), cos(q)]  (dim = 3 * coordinate_dim)
│   Output: scalar V
│
└── H(q, p) = T(q, p) + V(q)
```

### Key Design Choices

| Choice | Rationale |
|--------|-----------|
| Cholesky decomposition for M^{-1} | Guarantees symmetric positive definite (physically valid mass matrix) |
| `[q, sin(q), cos(q)]` input features | Gravity potential and inertia tensor depend on trig functions of joint angles for revolute joints |
| SiLU activation | Smoother than CELU/Tanh, better gradient flow for regression |
| Small network (256-wide, 4 layers) | The quadratic structure handles most of the work; network only needs to learn M^{-1}(q) (6 params for 3DoF) and V(q) (1 scalar) |

### Physics Guarantees (vs SeperableHNN)

| Property | SeperableHNN | StructuredHNN |
|----------|-------------|---------------|
| T quadratic in p | Not enforced (learned) | **Enforced exactly** |
| M^{-1}(q) SPD | Not guaranteed | **Guaranteed by construction** |
| dH/dp linear in p | Not enforced | **Enforced exactly** |
| T >= 0 | Not guaranteed | **Guaranteed** (sum of squares) |
| V depends on trig(q) | Must approximate from raw q | **Provided directly** |

### Why This Should Improve 3DoF Accuracy

1. **Reduced function space**: The kinetic network only needs to learn 6 Cholesky parameters as a function of 3 angles (not an arbitrary 6D->1D function). This is a massive reduction in hypothesis space.

2. **Exact gradient structure**: `dH/dp = M^{-1}(q) @ p` is automatically linear in p. The Hamilton equation `dq/dt = dH/dp` is the primary training target -- getting this structure right by construction eliminates a major source of systematic error.

3. **Trigonometric features**: For the rigid arm under gravity (`g = -9.81`), the potential energy is `V = -m*g*z_com(q)` where `z_com = 0.25 * R(q)[2,0]` involves sin/cos of the 3 joint angles. Providing these directly means the V network doesn't need to approximate trig functions internally.

## Parameters

| Config | Default | Notes |
|--------|---------|-------|
| `hidden_dim` | 256 | Width of both Cholesky and V networks |
| `num_layers` | 4 | Depth of both networks |
| `coordinate_dim` | 3 | Number of generalized coordinates |
| Total params | ~270K | vs 6.3M for SeperableHNN |

## Usage

```bash
# Train (default is now structured)
python src/models/HNN.py --mode train \
    --model_type structured \
    --hidden_dim 256 \
    --num_layers 4 \
    --checkpoint_prefix "StructuredHNN-dim256" \
    --wandb_name "StructuredHNN-3D-Hinge"

# Revert to old model for comparison
python src/models/HNN.py --mode train --model_type separable
```

## Generality

The quadratic kinetic energy structure `T = 0.5 * p^T M^{-1}(q) p` holds for **all** standard mechanical systems (any rigid body robot, any DoF). For higher DoF:

| DoF | Cholesky params | Suggested hidden_dim |
|-----|----------------|---------------------|
| 3   | 6              | 256                 |
| 6-7 | 21-28          | 512                 |
| 12+ | 78+            | 1024                |

The `sin/cos` features help for revolute joints (most robots). Prismatic joints don't benefit but aren't hurt.
