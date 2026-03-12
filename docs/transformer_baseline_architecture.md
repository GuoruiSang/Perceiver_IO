# Transformer Baseline Architecture

Last updated: 2026-03-11

This document specifies the standard-diffusion transformer backbone used for the final 2DoF and 3DoF comparisons.

## Entry points

- model class: `src/models/architectures.py` -> `TrajectoryTransformerDiffusion`
- training CLI: `src/models/train_trajectory_dpf.py` -> `--backbone transformer`
- launcher: `scripts/train/train_transformer_diffusion_fixed_length.sh`

## Fixed-budget reproduction

Run the two systems separately with explicit time budgets:

```bash
CUDA_VISIBLE_DEVICES=0,1 SYSTEM=2dof DEVICES="0 1" TIME_LIMIT_SEC=88612 \
  bash scripts/train/train_transformer_diffusion_fixed_length.sh
```

```bash
CUDA_VISIBLE_DEVICES=2,3 SYSTEM=3dof DEVICES="0 1" TIME_LIMIT_SEC=87992 \
  bash scripts/train/train_transformer_diffusion_fixed_length.sh
```

## Forward architecture

For input tokens `x_tokens` with shape `[B, T, C_in]`:

1. `Linear(C_in -> d_model)`
2. `LayerNorm(d_model)`
3. `TransformerEncoder(num_layers)`
4. `LayerNorm(d_model)`
5. `Linear(d_model -> state_dim)`

Project defaults:

- `d_model = num_latent_channels = 256`
- `num_layers = num_decoder_blocks = 4`
- `nhead = 8`
- `dropout = 0.0`
- `activation = GELU`
- `norm_first = True`

## Token construction

Transformer mode uses:

- normalized state
- normalized torque
- diffusion-step embedding
- absolute temporal embedding

Only the state slice is noised during diffusion.

## Final baseline checkpoints

- 2DoF:
  - `checkpoints/2dof/transformer_diffusion/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&FixedTrajLength1000&UniformContext&EncoderNone&DecoderAttentions_backbone-transformer:epoch=2999_val_loss:val_loss=0.0009.ckpt`
- 3DoF:
  - `checkpoints/3dof/transformer_diffusion/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&FixedTrajLength1000&UniformContext&EncoderNone&DecoderAttentions_backbone-transformer:epoch=2999_val_loss:val_loss=0.0008.ckpt`
