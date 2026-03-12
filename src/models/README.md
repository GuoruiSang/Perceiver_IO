## Model Layer Roles

- `trajectory_dpf_model.py`
  - Main `TrajectoryDPF` LightningModule.
  - Owns model construction, normalization, and token building.

- `train_trajectory_dpf.py`
  - Train CLI entrypoint for `TrajectoryDPF`.

- `trajectory_dpf_training.py`
  - Training/validation logic used by `TrajectoryDPF`.
  - Owns optimizer setup, context/query view construction, denoising loss, and EMA-related training hooks.

- `trajectory_dpf_sampling.py`
  - Generic diffusion sampling logic used by `TrajectoryDPF`.
  - Owns `sample_trajectories()`, torque generation for sampling, and DDIM/DDPM update helpers.
  - Shifted-token checkpoints also use this generic path; only the state/torque pairing is shifted.

- `architectures.py`
  - Backbone architecture definitions used by `TrajectoryDPF`.

- `HNN.py`
  - Structured HNN training/inference entrypoint.

- `utils.py`
  - Shared model utilities such as EMA and physics/eval helpers.
