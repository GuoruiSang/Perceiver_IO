# Data Scripts

This folder contains dataset-generation and dataset-inspection utilities used by the trajectory models.

## Main Scripts

- `generate_dataset_forward.py`: older forward-simulation dataset generator.
- `generate_bidirectional_reacher_dataset.py`: bidirectional non-dissipative Reacher dataset generator.
- `plot_bidirectional_reacher_waypoints.py`: visualization tool for waypoint-centered bidirectional rollouts.
- `dataset.py`: HDF5 dataset loaders used by training.

## Bidirectional Reacher Dataset

The bidirectional generator is intended for the non-dissipative 2-DoF MuJoCo Reacher defined by:

- [`configs/reacher_non_diss_from_dataset.xml`](/home/gsang/Projects/hnn_guided_dpf/configs/reacher_non_diss_from_dataset.xml)

Generation idea:

- sample a waypoint in task space
- solve IK for the waypoint pose
- sample a waypoint velocity
- generate a helper prefix from the waypoint with negated velocity
- reverse the prefix torques and concatenate them with a forward suffix torque sequence
- replay the full torque sequence as one forward rollout
- keep the rollout only if the end effector hits the waypoint within tolerance and the loose `qvel` / `qacc` sanity checks pass

This keeps the saved dataset compatible with the existing HDF5 workflow while ensuring each accepted trajectory contains a designated waypoint.

## Reproducible Command

The command below reproduces the train/val datasets created for the current bidirectional Reacher setup:

```bash
/home/gsang/miniconda3/envs/perceiver/bin/python \
  /home/gsang/Projects/hnn_guided_dpf/scripts/data/generate_bidirectional_reacher_dataset.py \
  --xml_path /home/gsang/Projects/hnn_guided_dpf/configs/reacher_non_diss_from_dataset.xml \
  --output_dir /home/gsang/Projects/hnn_guided_dpf/data/reacher_bidirectional_dt0p001_len1000 \
  --train_trajectories 40000 \
  --val_trajectories 2000 \
  --trajectory_length 1000 \
  --dt 0.001 \
  --waypoint_radius 0.18 \
  --waypoint_qvel_scale 0.8 \
  --torque_scale 0.2 \
  --waypoint_tolerance 0.01 \
  --max_abs_qvel 200 \
  --max_abs_qacc 10000 \
  --num_workers 24 \
  --batch_size 256
```

Expected outputs:

- `traj_40000-steps_1000.h5`
- `traj_2000-steps_1000.h5`

inside:

- [`data/reacher_bidirectional_dt0p001_len1000`](/home/gsang/Projects/hnn_guided_dpf/data/reacher_bidirectional_dt0p001_len1000)

## Output Format

The generated HDF5 files are compatible with the loaders in [`dataset.py`](/home/gsang/Projects/hnn_guided_dpf/scripts/data/dataset.py).

Each trajectory group stores:

- `seq_qpos`
- `seq_qvel`
- `seq_qacc`
- `seq_mom`
- `seq_mom_dot`
- `seq_torque`
- `seq_energy`

Additional per-trajectory fields are also stored:

- `waypoint_xy`
- `seq_fingertip_xy`
- attrs: `waypoint_index`, `waypoint_error`

File-level attrs include:

- `xml`
- `num_steps`
- `num_trajectories`
- `dt`
- `data_dt`
- `skip_steps`
- `state_alignment=pre_step`
- `torque_alignment=interval_mean`
- `derivative_alignment=forward_difference`

## Important Considerations

### 1. `joint0` is unbounded

In the current XML, `joint0` is an unbounded hinge. That means `seq_qpos[:, 0]` can accumulate many turns and may fall far outside `[-pi, pi]`.

This is physically valid, but it can increase variance during training. It is usually better to handle `joint0` as a periodic variable in the model pipeline, for example with `sin(q0)` / `cos(q0)`, rather than rewriting the saved dataset.

For the current Trajectory DPF training workflow, this is now handled automatically for Reacher datasets:

- the HDF5 file stays unchanged on disk
- the DPF dataset loader exposes Reacher `qpos` as `[sin(q0), cos(q0), q1]`
- min-max normalization is treated as identity on `sin(q0)` / `cos(q0)`
- only `q1` is normalized/denormalized within the `qpos` block
- whenever generated trajectories are sent back to MuJoCo, `q0` is recovered with `atan2(sin(q0), cos(q0))`

### 2. Do not wrap raw `q0` in the saved trajectories unless you really mean to

Wrapping `q0` to `[-pi, pi]` creates a branch cut. If a trajectory crosses that cut, the time series gets an artificial jump. For sequence models, that can be worse than the large raw range.

### 3. Current rejection logic is intentionally light

The bidirectional generator does not reject by energy. It only rejects trajectories for:

- non-finite rollout state
- `qvel` beyond `max_abs_qvel`
- `qacc` beyond `max_abs_qacc`
- waypoint miss beyond `waypoint_tolerance`

For the current 40k/2k dataset run, the practical bottleneck was waypoint matching, not `qvel`/`qacc`.

### 3.1 Joint-limit reactions are not recorded in `seq_torque`

For the current non-dissipative Reacher XML, `joint1` still has a positional limit while `joint0` is unbounded. This matters for interpretation:

- `seq_torque` stores the actuator torque only
- if `joint1` hits or pushes against its limit, MuJoCo can introduce an additional hidden constraint reaction
- that reaction is part of the simulator's total generalized forcing, but it is not stored in `seq_torque`

So the saved trajectories are fully consistent with MuJoCo replay under the same XML, but they are not strictly consistent with a simple forced-Hamiltonian equation of the form:

- `p_dot = -dH/dq + tau`

whenever joint-limit reactions are active. In that regime the dynamics are closer to:

- `p_dot = -dH/dq + tau + J(q)^T lambda`

where `lambda` is the joint-limit reaction term.

Implication:

- this is usually acceptable for pure trajectory-generation training
- it is more important if the dataset will be used for explicit forced-Hamiltonian identification or HNN-style consistency objectives
- if strict `tau`-only forcing is desired, consider filtering out near-limit trajectories or using an unbounded `joint1`

### 4. `dt` matters a lot for bidirectional consistency

The qualitative bidirectional idea works at larger `dt`, but the replay consistency is much better at `dt=0.001` than at `dt=0.01`. If you change `dt`, re-check the waypoint-hit rate and rollout quality before generating a large dataset.

### 5. Smooth torque is clipped to actuator limits

Torques are generated as sums of sinusoids and then clipped to stay inside the actuator range. This keeps controls feasible for MuJoCo even when several sinusoidal components add constructively.

### 6. Benchmark comparability versus project-specific physics cleanliness

There is no single universal "standard Reacher XML" across the literature.

Two common benchmark families are:

- Gym/Gymnasium MuJoCo Reacher, whose default environment keeps the classic 2-joint reacher task and exposes torques on `joint0` and `joint1`
- DeepMind Control Suite Reacher, which is a different benchmark family with its own task definition and defaults

So benchmark papers often mean "the default Reacher of framework X", not one shared XML used across all papers.

Practical takeaway for this project:

- if you want comparability to Gym/Gymnasium Reacher, keeping the bounded `joint1` default is the closer choice
- if you want cleaner `tau`-only forced-Hamiltonian consistency, making `joint1` unbounded is a defensible project-specific modification
- that modification should be documented clearly because it makes the environment less directly comparable to the default Gym/Gymnasium Reacher benchmark

## Quick Visualization

To inspect the same configuration visually:

```bash
/home/gsang/miniconda3/envs/perceiver/bin/python \
  /home/gsang/Projects/hnn_guided_dpf/scripts/data/plot_bidirectional_reacher_waypoints.py \
  --plot_mode overlay_with_torque \
  --num_examples 1000 \
  --trajectory_length 1000 \
  --dt 0.001 \
  --waypoint_radius 0.18 \
  --waypoint_qvel_scale 0.8 \
  --torque_scale 0.2 \
  --show_stats
```

This is useful for checking:

- workspace coverage
- waypoint placement
- prefix/suffix balance
- torque smoothness

## Notes

- The saved dataset is compatible with the existing training and loading workflow.
- If you change the state representation for training, prefer doing that in the model/data pipeline rather than overwriting the raw HDF5 fields.
