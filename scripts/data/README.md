# Data Scripts

This folder contains dataset-generation and dataset-inspection utilities used by the trajectory models.

## Main Scripts

- `generate_dataset_forward.py`: older forward-simulation dataset generator.
- `generate_bidirectional_reacher_dataset.py`: bidirectional non-dissipative Reacher dataset generator.
- `plot_bidirectional_reacher_waypoints.py`: visualization tool for waypoint-centered bidirectional rollouts.
- `dataset.py`: HDF5 dataset loaders used by training.

## Bidirectional Reacher Dataset

The bidirectional generator is intended for the non-dissipative 2-DoF MuJoCo Reacher defined by:

- [`configs/reacher_non_diss_unbounded_j1.xml`](/home/gsang/Projects/hnn_guided_dpf/configs/reacher_non_diss_unbounded_j1.xml)

Generation idea:

- sample a waypoint in task space
- solve IK for the waypoint pose
- sample a waypoint velocity
- sample one smooth full-length torque sequence for the entire trajectory
- split that torque sequence at a random waypoint index
- run the helper prefix from the waypoint with negated velocity using the prefix torque segment in reversed time order
- replay the saved trajectory from timestep 0 using the original forward-time torque sequence
- keep the rollout only if the end effector hits the waypoint within tolerance and the loose `qvel` / `qacc` sanity checks pass

This keeps the saved dataset compatible with the existing HDF5 workflow while ensuring each accepted trajectory contains a designated waypoint.

## Reproducible Command

The command below reproduces the active bidirectional Reacher train/val datasets:

```bash
/home/gsang/miniconda3/envs/perceiver/bin/python \
  /home/gsang/Projects/hnn_guided_dpf/scripts/data/generate_bidirectional_reacher_dataset.py \
  --xml_path /home/gsang/Projects/hnn_guided_dpf/configs/reacher_non_diss_unbounded_j1.xml \
  --output_dir /home/gsang/Projects/hnn_guided_dpf/data/reacher_bidirectional_unbounded_j1_dt0p001_len1000 \
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

inside:

- [`data/reacher_bidirectional_unbounded_j1_dt0p001_len1000`](/home/gsang/Projects/hnn_guided_dpf/data/reacher_bidirectional_unbounded_j1_dt0p001_len1000)

The older bounded-`joint1` dataset at [`data/reacher_bidirectional_dt0p001_len1000`](/home/gsang/Projects/hnn_guided_dpf/data/reacher_bidirectional_dt0p001_len1000) is legacy/reference only and is no longer part of the supported Reacher DPF workflow.

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

### 1. Both arm joints are periodic in the active Reacher workflow

In the active XML, both `joint0` and `joint1` are unbounded hinges. That means both raw angle channels can accumulate many turns and may fall far outside `[-pi, pi]`.

This is physically valid, but it can increase variance during training. It is usually better to handle both joints as periodic variables in the model pipeline rather than rewriting the saved dataset.

For the current Trajectory DPF training workflow, this is now handled automatically for Reacher datasets:

- the HDF5 file stays unchanged on disk
- the DPF dataset loader exposes Reacher `qpos` as `[sin(q0), cos(q0), sin(q1), cos(q1)]`
- min-max normalization is treated as identity on all four Reacher `qpos` channels
- whenever generated trajectories are sent back to MuJoCo, both joints are recovered with `atan2`

### 2. Do not wrap raw joint angles in the saved trajectories unless you really mean to

Wrapping a raw joint angle to `[-pi, pi]` creates a branch cut. If a trajectory crosses that cut, the time series gets an artificial jump. For sequence models, that can be worse than the large raw range.

### 3. Current rejection logic is intentionally light

The bidirectional generator does not reject by energy. It only rejects trajectories for:

- non-finite rollout state
- `qvel` beyond `max_abs_qvel`
- `qacc` beyond `max_abs_qacc`
- waypoint miss beyond `waypoint_tolerance`

For the current 40k/2k dataset run, the practical bottleneck was waypoint matching, not `qvel`/`qacc`.

### 3.1 `seq_torque` is cleaner in the active unbounded-joint Reacher

Because both arm joints are unbounded in the active XML, MuJoCo does not need arm joint-limit reactions during rollout. That makes `seq_torque` a cleaner record of the explicit actuator forcing for the arm dynamics than it was in the old bounded-`joint1` variant.

The legacy bounded dataset can still be useful as a reference artifact, but it is no longer part of the supported Reacher DPF processing path.

### 4. `dt` matters a lot for bidirectional consistency

The qualitative bidirectional idea works at larger `dt`, but the replay consistency is much better at `dt=0.001` than at `dt=0.01`. If you change `dt`, re-check the waypoint-hit rate and rollout quality before generating a large dataset.

### 5. Smooth torque is clipped to actuator limits

Torques are generated as sums of sinusoids and then clipped to stay inside the actuator range. This keeps controls feasible for MuJoCo even when several sinusoidal components add constructively.

### 5.1 The active generator now uses one smooth torque sequence per trajectory

The current bidirectional construction no longer stitches together independently sampled prefix and suffix torques. Instead, it samples one smooth full-length torque sequence, splits it at the waypoint index, and only reverses the prefix segment for the internal helper rollout.

That means:

- the saved `seq_torque` is smooth across the waypoint in forward time
- the full saved trajectory can be replayed exactly from timestep 0 using only forward dynamics
- the replayed trajectory still passes through the designated waypoint at `waypoint_index`

### 6. Benchmark comparability versus project-specific physics cleanliness

There is no single universal "standard Reacher XML" across the literature.

Two common benchmark families are:

- Gym/Gymnasium MuJoCo Reacher, whose default environment keeps the classic 2-joint reacher task and exposes torques on `joint0` and `joint1`
- DeepMind Control Suite Reacher, which is a different benchmark family with its own task definition and defaults

So benchmark papers often mean "the default Reacher of framework X", not one shared XML used across all papers.

Practical takeaway for this project:

- the supported Reacher DPF workflow uses the unbounded-`joint1` XML because it is cleaner for `tau`-only forced-Hamiltonian interpretation
- this is a project-specific modification and is less directly comparable to the default Gym/Gymnasium Reacher benchmark

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
