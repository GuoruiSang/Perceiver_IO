# Fixed-Length 3DoF Model Training Log

**Date**: 2026-02-06

## Objective

Train a fixed-length (non-DPF) diffusion model for the 3DoF hinge arm, using the same PerceiverIO architecture and the same sinusoidal-only training data. The only difference from the existing DPF model is that training uses a single fixed trajectory length instead of random variable lengths (100-1000).

## Changes Made

### 1. `src/models/trajectory_dpf.py` — Added `--fixed_trajectory_length` CLI argument

Four modifications in the `main()` function:

| Location | Change |
|----------|--------|
| CLI args (~L1174) | Added `--fixed_trajectory_length` argument |
| Dataset loading (~L1509) | `trajectory_length` now follows `--fixed_trajectory_length` when set |
| Model construction (~L1625) | `trajectory_length_training_options` set to `(T_fixed,)` instead of `(100,...,1000)` |
| Naming (~L1700, ~L1666) | Checkpoint filename and W&B run name use `FixedTrajLengthN` instead of `VariableTrajLength` |

No changes to `TrajectoryDPF.__init__()`, model architecture, or any other files. Existing checkpoints are fully compatible.

### 2. Dataset metadata fix — `data_dt` correction

Fixed incorrect `data_dt` metadata in two HDF5 files:

| File | Before | After | Reason |
|------|--------|-------|--------|
| `data/traj_40000-steps_4000.h5` | `data_dt=0.00025` | `data_dt=0.0002` | Should be `dt * skip_steps = 0.0001 * 2` |
| `data/traj_4000-steps_4000.h5` | `data_dt=0.00025` | `data_dt=0.0002` | Same |

The `0.00025` value was a historical bug — the 80k dataset (`traj_80000-steps_4000.h5`) already had the correct value `0.0002`.

## Training Configuration

```bash
python src/models/trajectory_dpf.py \
    --mode train \
    --fixed_trajectory_length 1000 \
    --devices 0 2 3 4 \
    --resume_from_checkpoint "" \
    --epochs 3000
```

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Training data | `data/traj_40000-steps_4000.h5` | 40,000 trajectories, sinusoidal-only torques |
| Test data | `data/traj_4000-steps_4000.h5` | 4,000 trajectories, sinusoidal-only torques |
| Fixed trajectory length | 1000 | Single fixed length (vs DPF's random 100-1000) |
| GPUs | 0, 2, 3, 4 | 4x DDP |
| Epochs | 3000 | Same as DPF |
| Architecture | `ConditionedTrajectoryPerceiverIO` | Identical to DPF |
| num_latents | 256 | Same |
| num_latent_channels | 256 | Same |
| cond_dim | 256 | Same |
| num_decoder_blocks | 4 | Same |
| Diffusion steps | 1000 | Same |
| dt | 0.0001 | Same |
| data_dt | 0.0002 | Corrected from 0.00025 |

### W&B

- Project: `trajectory-dpf-smooth`
- Run name: `...FixedTrajLength1000...`

### Checkpoint naming

```
trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&FixedTrajLength1000&UniformContext&EncoderNone&DecoderAttentions:{epoch:03d}_val_loss:{val_loss:.4f}
```

## DPF vs Fixed-Length Comparison

| Aspect | DPF Model | Fixed-Length Model |
|--------|-----------|-------------------|
| `trajectory_length_training_options` | `(100, 200, ..., 1000)` | `(1000,)` |
| Architecture | `ConditionedTrajectoryPerceiverIO` | Same |
| Diffusion / DDIM / CFG | Yes | Yes |
| Token structure `[state \| diff_enc \| temp_enc]` | Same | Same |
| Torque conditioning (AdaLN) | Same | Same |
| Training data | Sinusoidal-only | Same |
| Trajectory extension capability | Yes (variable-length generalization) | No (fixed-length only) |

## Known Issue

DDP initialization repeats dataset loading and normalization stats computation once per GPU process (4 times total). This is a pre-existing PyTorch Lightning DDP behavior — only affects startup time, not training correctness or speed. Could be optimized in the future by guarding with `if trainer.is_global_zero`.
