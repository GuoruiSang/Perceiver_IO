# Fixed-Length 3DoF Transformer Training

Date: 2026-03-11

This is the minimal launcher needed to reproduce the final 3DoF standard-diffusion checkpoint used by the active 7-method comparison (`alpha=1` one-step guidance setting).

## Final checkpoint target

- `checkpoints/3dof/transformer_diffusion/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&FixedTrajLength1000&UniformContext&EncoderNone&DecoderAttentions_backbone-transformer:epoch=2999_val_loss:val_loss=0.0008.ckpt`

## Required datasets

- `data/3dof/traj_40000-steps_4000.h5`
- `data/3dof/traj_2000-steps_4000.h5`

The 3DoF train/val pair must be regenerated together with one command so the
metadata stays matched.

If either file is missing or mismatched, regenerate both with:

```bash
bash scripts/train/generate_2dof_3dof_datasets.sh
```

## Launcher

```bash
SYSTEM=3dof bash scripts/train/train_transformer_diffusion_fixed_length.sh
```

## Direct command

```bash
python src/models/train_trajectory_dpf.py \
  --mode train \
  --h5_path data/3dof/traj_40000-steps_4000.h5 \
  --checkpoint_dir checkpoints/3dof/transformer_diffusion \
  --devices 0 1 \
  --epochs 3000 \
  --batch_size 128 \
  --lr 1e-4 \
  --num_decoder_blocks 4 \
  --num_latents 256 \
  --num_latent_channels 256 \
  --diffusion_steps 1000 \
  --fixed_trajectory_length 1000 \
  --backbone transformer \
  --wandb true \
  --wandb_project Trajectory-StdDiff-Transformer-3DOF \
  --resume_from_checkpoint ""
```
