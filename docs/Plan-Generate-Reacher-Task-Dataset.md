# Plan-Generate-Reacher-Task-Dataset

## Goal

Generate a MuJoCo Reacher trajectory dataset whose pooled fingertip positions cover the task/workspace as evenly as possible, while keeping rollout starts realistic.

## Core Principle

We care about the distribution of all visited fingertip positions across the dataset, not about forcing every rollout to start in a sparse region or to exactly hit a sampled target.

So the generation process should be:

1. Randomly initialize the arm.
2. Pick a sparse workspace bin as a waypoint or target.
3. Apply a controller or policy that tends to move toward that region, with some noise for diversity.
4. Add all visited positions to the occupancy map.
5. Repeat until the occupancy map is fairly even.

## Important Constraint

With a fixed rollout horizon such as 1000 steps, plus bounded velocity/torque, some waypoint assignments will be infeasible. A rollout may not be able to fully reach a far sparse waypoint in time.

This should be handled as a feasibility-constrained waypoint-selection problem rather than by assuming every sparse waypoint is equally good.

## Preferred Strategy

Do not require every rollout to exactly reach its waypoint.

Instead:

- If the waypoint is reachable within the horizon, that is ideal.
- If the rollout does not fully reach the waypoint but still moves substantially toward it and adds useful occupancy in sparse regions, keep it.
- If the rollout barely progresses and mostly stays in already dense regions, treat that waypoint choice as poor and resample or reject it.

## Waypoint Selection Rules

To improve efficiency under timestep and velocity limits:

- Choose sparse bins that are also near the current state.
- Limit waypoint distance based on current position, velocity cap, and remaining horizon.
- Prefer a sequence of local sparse waypoints rather than one far waypoint.
- Score a rollout by coverage gained, not only by final waypoint error.

## Practical Implication

Before increasing velocity limits or rollout length, first make waypoint selection horizon-aware.

Useful levers are:

1. Increase horizon.
2. Allow faster motion.
3. Reject infeasible waypoints.
4. Preferably: sample reachable sparse waypoints in the first place.

## Alternative: Bidirectional Generation Around a Chosen Waypoint

For the non-dissipative Reacher setting, there is another direct option:

1. Pick a target fingertip position.
2. Convert it to a joint configuration with IK.
3. Sample a waypoint velocity or momentum.
4. Construct a trajectory around that waypoint state using bidirectional dynamics.

This is useful when we want the chosen target position to definitely appear in the trajectory.

### Important Detail

A target position alone is not enough to define a physically valid trajectory segment. The waypoint should be treated as a full state:

- joint position `q*` or fingertip position mapped to `q*` through IK
- joint velocity `qdot*` or momentum `p*`

### How Bidirectional Construction Works

For a nearly conservative, non-dissipative mechanical system:

- Generate the suffix by simulating forward from the waypoint state `(q*, p*)`.
- Generate the prefix using time-reversal symmetry:
  - start from `(q*, -p*)`
  - simulate forward
  - reverse the resulting rollout in time
  - flip the momentum signs back
- Concatenate `prefix + waypoint + suffix`

This gives a trajectory whose midpoint is guaranteed to pass through the chosen waypoint.

### Why This Fits the Current Reacher Setting

This idea is especially appropriate for the non-dissipative Reacher because the dynamics are close to time-reversible. It is much less reliable for dissipative or contact-heavy systems.

### Main Caveats

- MuJoCo is not literally integrating backward in time here; this uses time-reversal symmetry instead.
- The method is most valid when damping and friction are removed or negligible.
- If waypoint velocity is always near zero, trajectories may look artificial because many trajectories will "pause" at the waypoint.
- This guarantees waypoint inclusion, but it does not automatically preserve a realistic random-start distribution.

## Summary

The intended dataset generator should keep random initial states for realism, use coverage-aware waypoint selection during rollout, and accept trajectories based on how much new occupancy they add to under-covered workspace regions.

For the non-dissipative Reacher specifically, bidirectional generation around a sampled waypoint state is also a viable direct method when guaranteed waypoint inclusion is more important than preserving a realistic random-start distribution.
