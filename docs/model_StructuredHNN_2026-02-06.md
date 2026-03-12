# Model: StructuredHNN

Date: 2026-02-06
Updated: 2026-03-11

This document now tracks the active structured-HNN contract used by the active 7-method comparison (`alpha=1` one-step guidance setting).

## Final checkpoint targets

- 2DoF:
  - `checkpoints/2dof/hnn/StructuredHNN-2DOF-epoch-epoch=999.ckpt`
- 3DoF:
  - `checkpoints/3dof/hnn/StructuredHNN-dim256-traj40000-epoch-epoch=999.ckpt`

## Required datasets

- 2DoF train:
  - `data/2dof/traj_40000-steps_4000.h5`
- 2DoF val:
  - `data/2dof/traj_2000-steps_4000.h5`
- 3DoF train:
  - `data/3dof/traj_40000-steps_4000.h5`
- 3DoF val:
  - `data/3dof/traj_2000-steps_4000.h5`

For each DoF, the train and validation files must be generated together from the
same command/configuration. Do not mix an older train file with a newly
generated validation file.

Active generation contract:
- 2DoF:
  - `xml_path = configs/rigid_arm_hinge_2dof.xml`
  - `torque_policies = sinusoidal:1.0`
  - `num_trajectories = 40000`
  - `num_val = 2000`
  - `num_steps = 4000`
- 3DoF:
  - `xml_path = configs/rigid_arm_hinge.xml`
  - `torque_policies = sinusoidal:1.0`
  - `num_trajectories = 40000`
  - `num_val = 2000`
  - `num_steps = 4000`

## Launchers

```bash
bash scripts/train/train_2dof_dpf_hnn_models.sh
bash scripts/train/train_3dof_dpf_hnn_models.sh
bash scripts/train/generate_2dof_3dof_datasets.sh
```

The HNN portions are:

```bash
python src/models/HNN.py \
  --mode train \
  --train_file data/2dof/traj_40000-steps_4000.h5 \
  --test_file data/2dof/traj_2000-steps_4000.h5 \
  --checkpoint_dir checkpoints/2dof/hnn \
  --checkpoint_prefix StructuredHNN-2DOF \
  --wandb_name StructuredHNN-2DOF-Hinge \
  --xml_path configs/rigid_arm_hinge_2dof.xml \
  --model_type structured \
  --hidden_dim 256 \
  --num_layers 4
```

```bash
python src/models/HNN.py \
  --mode train \
  --train_file data/3dof/traj_40000-steps_4000.h5 \
  --test_file data/3dof/traj_2000-steps_4000.h5 \
  --checkpoint_dir checkpoints/3dof/hnn \
  --checkpoint_prefix StructuredHNN-dim256-traj40000 \
  --wandb_name StructuredHNN-3D-Hinge-traj40000 \
  --xml_path configs/rigid_arm_hinge.xml \
  --model_type structured \
  --hidden_dim 256 \
  --num_layers 4
```

## Architecture summary

- Kinetic energy is constrained to the quadratic form `0.5 * p^T M^{-1}(q) p`.
- `M^{-1}(q)` is parameterized via Cholesky factors, which guarantees positive definiteness.
- Potential energy is learned by a separate network on `[q, sin(q), cos(q)]`.
- The active 7-method comparison uses the same structured-HNN family for both 2DoF and 3DoF.
