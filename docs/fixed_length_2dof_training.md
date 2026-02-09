# Fixed-Length 2DoF Model Training

## Objective

Train a fixed-length (non-variable-length) diffusion model for the 2DoF hinge arm using the same `TrajectoryDPF` architecture used in 3DoF.

## Dataset

- Training file: `data/2dof/traj_40000-steps_4000.h5`
- Dataset metadata check:
  - `dt = 0.0001`
  - `data_dt = 0.0002`

## One-command launcher

```bash
bash scripts/train_2dof_fixed_length.sh
```

## Direct command (equivalent)

```bash
python src/models/trajectory_dpf.py \
  --mode train \
  --h5_path data/2dof/traj_40000-steps_4000.h5 \
  --checkpoint_dir checkpoints/2dof \
  --devices 0 1 2 \
  --epochs 3000 \
  --batch_size 128 \
  --lr 1e-4 \
  --num_decoder_blocks 4 \
  --num_latents 256 \
  --num_latent_channels 256 \
  --diffusion_steps 1000 \
  --fixed_trajectory_length 1000 \
  --wandb true \
  --wandb_project Trajectory-DPF-2DOF \
  --resume_from_checkpoint ""
```

## Notes

- `--fixed_trajectory_length 1000` is the key flag that switches from variable-length DPF training to fixed-length standard diffusion.
- Checkpoint names include `FixedTrajLength1000`, matching the 3DoF fixed-length naming convention.
- Log file (launcher script): `logs/dpf_2dof_fixed_length.log`.

## Useful overrides (launcher script)

```bash
DEVICES="0 1 2 3" FIXED_TRAJ_LENGTH=1000 EPOCHS=3000 bash scripts/train_2dof_fixed_length.sh
```
