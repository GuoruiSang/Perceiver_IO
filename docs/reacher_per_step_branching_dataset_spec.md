# Reacher Per-Step Branching Dataset Spec

## Goal

Collect a fixed-length Reacher dataset that supports this inference pattern directly:

- observe a clean prefix ending at timestep `t`
- generate multiple diverse suffixes from that exact prefix
- be able to do this for any timestep, not just one fixed branch point

The important idea is that **branchability should exist at every timestep**.

## Core Semantics

A single stored branch example means:

- one root rollout from `t = 0`
- one anchor timestep `a`
- one branch suffix that starts from the exact root state at `a`

Multiple stored examples with the same `(root_group_id, anchor_index)` share:

- the exact same prefix `0:a`
- different suffixes `a:T`

So branching is explicit at the anchor timestep.

Across the dataset, anchor timesteps should cover the whole trajectory, so the model sees branching behavior from early, middle, and late prefixes.

## Why Not Store a Full Tree?

A literal tree with branching at every timestep grows exponentially and is not practical to store.

Instead, the dataset should be **flat but branchable**:

- many root rollouts
- many anchor timesteps sampled from each root rollout
- many sibling suffixes from each anchor

This preserves the learning signal without storing an enormous explicit tree.

## Recommended Data Unit

Each stored sample is still a full fixed-length trajectory of length `T = 500`.

But each sample also carries metadata saying:

- which root rollout it came from
- which anchor timestep it branched from
- which sibling branch it is within that anchor set

That means training can still use the current fixed-length tensor format.

## Fixed Defaults

- total stored steps: `T = 500`
- timestep: `dt = 0.001`
- task type: no-contact, free-space Reacher
- state alignment: `pre_step`
- torque alignment: `interval_mean`
- default target radius for workspace-derived metadata: `0.2`

## Generator Structure

### 1. Root Rollout

For each root group:

- sample one source state
  - `q_source ~ Uniform([-pi, pi]^2)`
  - `qvel_source ~ Uniform([-0.2, 0.2]^2)`
- sample one smooth root torque program over the full horizon `T`
- replay it to produce one valid root rollout

The root rollout is only used to define anchor states and shared prefixes.

A practical v1 torque parameterization is:

- choose `m_root = 12` low-frequency control points per joint
- sample control values in `[-u_root, u_root]`
- spline-interpolate to length `T`

## 2. Anchor Timesteps

Conceptually, every timestep after a short warmup is branchable.

For a root rollout, define the valid anchor set:

- `a in {a_min, ..., T - suffix_min_len}`

with a practical default like:

- `a_min = 32`
- `suffix_min_len = 32`

In the full conceptual model, every valid `a` can branch.

For storage efficiency, the dataset does not need to branch at every timestep in every root group. Instead:

- choose a dense subset of anchors per root, e.g. every `8` steps with a random offset
- across many root groups, this gives effective coverage over all timesteps

So the design remains **per-step branchable**, even if one individual root group stores only a subset of anchor times.

## 3. Anchor-Local Branch Generation

At anchor timestep `a`, read from the root rollout:

- branch state `q_a, qvel_a`
- boundary torque `tau_a`
- boundary torque slope `dot_tau_a`

Then generate `K` smooth sibling suffixes from that exact anchor state.

Recommended v1 defaults:

- sibling count per anchor: `K = 4` or `8`
- suffix basis control points: `m_suffix = 16`
- suffix residual scale: moderate, e.g. `0.08 - 0.12`

Use the boundary-conditioned suffix parameterization:

```text
tau_suffix(s) = tau_a + dot_tau_a * (duration_suffix * s) + s^2 * r(s)
```

where:

- `s in [0, 1]`
- `r(s)` is a smooth residual function sampled from a spline basis

This guarantees:

- torque continuity at the anchor
- optional slope continuity at the anchor
- diversity through the residual coefficients

## 4. Residual Basis

Parameterize the residual `r(s)` in a smooth basis.

Recommended v1:

- cubic spline interpolation over `16` residual control points per joint
- sample coefficients in a bounded box
- use scrambled Sobol sampling for even coefficient coverage

This gives:

- smooth suffix torque
- diverse suffixes from the same prefix
- broader and more controlled coverage than raw per-step torque noise

## 5. Replay and Acceptance

Replay every candidate suffix from the anchor state.

Reject a candidate branch if:

- replay becomes non-finite
- `max |tau|` exceeds the limit
- `max |qvel|` exceeds the limit
- `max |qacc|` exceeds the limit
- the branch is too similar to already accepted siblings from the same anchor

Recommended sibling diversity checks:

- suffix end-effector RMSE from `a:` must exceed a small threshold
- final end-effector position gap must exceed a small threshold

These checks are local to the anchor set, not global across the whole dataset.

## 6. Stored Fields

Keep the standard rollout fields:

- `seq_qpos`
- `seq_qvel`
- `seq_qacc`
- `seq_mom`
- `seq_mom_dot`
- `seq_torque`
- `seq_energy`
- `seq_fingertip_xy`

Store root metadata:

- `source_qpos`
- `source_qvel`
- `source_mom`
- `root_group_id`

Store anchor metadata:

- `anchor_index`
- `anchor_state_qpos`
- `anchor_state_qvel`
- `anchor_state_mom`
- `boundary_tau`
- `boundary_tau_slope`

Store sibling metadata:

- `anchor_branch_group_id`
- `branch_index`

Optional compatibility metadata:

- `waypoint_xy`
- `waypoint_index`

For each stored branch, `waypoint_xy` can simply be chosen from the replayed suffix after the anchor.

## How Training Uses This

Training still sees fixed-length trajectories.

But when the model is trained on many samples that share the same `(root_group_id, anchor_index)` and differ only after that anchor, it learns the correct multimodal semantics:

- same prefix
- different valid futures

That is a much better match to the actual suffix-sampling problem than a single fixed branch point or a source-target-end construction.

## How Tree Visualization Uses This

The stored dataset is flat, but tree visualization becomes easy:

- choose one root rollout
- choose one anchor
- plot its sibling branches
- then recursively branch again from one or more child nodes using the same anchor-local suffix generator

So the same generator supports both:

- practical flat dataset storage
- recursive tree visualizations at evaluation time

## Recommended Pilot

Use this pilot first:

- `T = 500`
- `dt = 0.001`
- `K = 4`
- root torque from `12` spline control points per joint
- anchor stride `8`
- anchors from `32` to `468`
- suffix residual basis with `16` control points per joint
- suffix residual scale `0.10`
- clipped boundary slope cap `0.75`
- replay with the same torque, qvel, and qacc limits as the current branching pilot

This pilot keeps the design simple while making branchability effectively available across the whole trajectory.

## Why This Is the Recommended Next Design

This directly satisfies the real requirement:

- same prefix can lead to many suffixes
- this should be possible at every timestep
- torques should stay smooth at the branch boundary
- coverage should be broad but controlled

And it does so without trying to store an impossible explicit full branching tree.
