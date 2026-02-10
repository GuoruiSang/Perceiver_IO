# Transformer Baseline Architecture (Standard Diffusion)

Last updated: 2026-02-10

## Purpose

This document specifies the current `--backbone transformer` baseline used for standard diffusion comparisons against DPF/PerceiverIO.

## Entry Points

- Model class: `src/models/architectures.py` -> `TrajectoryTransformerDiffusion`
- Backbone switch: `src/models/trajectory_dpf.py` -> `--backbone {perceiverio, transformer}`

## Forward Architecture

For input tokens `x_tokens` with shape `[B, T, C_in]`:

1. `input_proj`: `Linear(C_in -> d_model)`
2. `input_norm`: `LayerNorm(d_model)`
3. `encoder`: `TransformerEncoder(num_layers)`
   - each layer is `TransformerEncoderLayer`
   - `nhead = 8`
   - `dim_feedforward = 4 * d_model`
   - `activation = GELU`
   - `norm_first = True` (pre-LN)
   - `dropout = 0.0`
4. `output_norm`: `LayerNorm(d_model)`
5. `output_proj`: `Linear(d_model -> state_dim)`

Output is full-sequence epsilon prediction:

- `eps_pred` shape: `[B, T, state_dim]`

## Token Construction

In transformer mode, token order is:

`x_tokens = concat(normalized_state, torque_norm, diffusion_enc, temporal_enc)`

where:

- `normalized_state`: `[B, T, state_dim]` (`state_dim = qpos_dim + mom_dim`)
- `torque_norm`: `[B, T, torque_dim]`
- `diffusion_enc`: Fourier embedding of current diffusion step, broadcast across `T`
- `temporal_enc`: absolute sinusoidal time embedding for trajectory index

## Noise Application

Only the state slice is noised during diffusion training/sampling:

- noised slice: first `state_dim` channels
- conditioning slices (`torque_norm`, `diffusion_enc`, `temporal_enc`) are not noised

## Training-Specific Notes

- `num_context` is sampled/logged for compatibility with shared training code.
- In transformer mode, `num_context` is **not used** for forward pass.
- There is no context/query split in transformer mode.

## Difference from PerceiverIO Backbone

- Transformer baseline:
  - no latent bottleneck
  - no Perceiver cross-attention decoder
  - predicts full sequence directly
- DPF/PerceiverIO:
  - latent array + cross-attention decoding
  - context/query pathway and torque-conditioned Perceiver blocks

## Default Hyperparameter Mapping (current)

When training with existing CLI defaults used in this project:

- `d_model = num_latent_channels` (typically 256)
- `num_layers = num_decoder_blocks` (typically 4)
- `nhead = 8`

This mapping keeps training scripts simple while preserving a single CLI surface.
