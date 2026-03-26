#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

import h5py
import mujoco
import numpy as np
from tqdm import tqdm

from scripts.data.generate_reacher_per_step_branching_dataset import (
    L1,
    L2,
    TRAJ_KEYS,
    boundary_conditioned_suffix,
    branch_diversity_metrics,
    build_root_rollout,
    choose_anchor_indices,
    compute_source_mom,
    get_model_ids,
    sample_workspace_uniform_source_state,
    simulate_dataset_rollout,
)


DEFAULT_DISTANCE_BIN_EDGES = (
    0.000,
    0.005,
    0.010,
    0.015,
    0.020,
    0.030,
    0.040,
    0.050,
    0.065,
    0.080,
    0.100,
    0.125,
    0.150,
    0.180,
    0.210,
)
DEFAULT_MOMENTUM_BIN_EDGES = (
    0.00,
    0.02,
    0.05,
    0.10,
    0.20,
    0.40,
    0.80,
    1.60,
    3.20,
)
REACH_MIN = abs(L1 - L2)
REACH_MAX = L1 + L2


def actuator_hard_torque_limit(model: mujoco.MjModel) -> np.ndarray:
    if model.nu <= 0:
        return np.empty((0,), dtype=np.float64)

    limited = np.asarray(model.actuator_ctrllimited, dtype=bool).reshape(-1)
    ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float64).reshape(model.nu, 2)
    hard_limit = np.full(model.nu, np.inf, dtype=np.float64)
    if np.any(limited):
        hard_limit[limited] = np.max(np.abs(ctrlrange[limited]), axis=1)
    return hard_limit


def json_safe_limit(limit: np.ndarray) -> float | list[float | None]:
    if limit.ndim != 1:
        raise ValueError(f"Expected 1D limit array, got shape {limit.shape}")
    if len(limit) == 0:
        return []
    if np.all(np.isfinite(limit)) and np.allclose(limit, limit[0]):
        return float(limit[0])
    return [None if not np.isfinite(v) else float(v) for v in limit.tolist()]


def resolve_requested_torque_limit(
    model: mujoco.MjModel,
    requested_max_abs_tau: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    hard_limit = actuator_hard_torque_limit(model)
    if requested_max_abs_tau is None:
        return hard_limit.copy(), hard_limit

    requested = np.full(model.nu, float(requested_max_abs_tau), dtype=np.float64)
    effective = requested.copy()
    finite_mask = np.isfinite(hard_limit)
    effective[finite_mask] = np.minimum(effective[finite_mask], hard_limit[finite_mask])
    return effective, hard_limit


def forward_kinematics_2link(qpos: np.ndarray) -> np.ndarray:
    q = np.asarray(qpos, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[None, :]
    q1 = q[:, 0]
    q2 = q[:, 1]
    out = np.stack(
        [
            L1 * np.cos(q1) + L2 * np.cos(q1 + q2),
            L1 * np.sin(q1) + L2 * np.sin(q1 + q2),
        ],
        axis=1,
    )
    return out[0] if single else out


def wrap_angle_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def compute_distance_bin(distance: float, distance_bin_edges: np.ndarray) -> int:
    idx = int(np.searchsorted(distance_bin_edges, distance, side="right") - 1)
    return int(np.clip(idx, 0, len(distance_bin_edges) - 2))


def compute_angle_bin(angle: float, angle_bin_count: int) -> int:
    shifted = (wrap_angle_pi(angle) + math.pi) / (2.0 * math.pi)
    idx = int(math.floor(shifted * angle_bin_count))
    return int(np.clip(idx, 0, angle_bin_count - 1))


def compute_goal_bin_id(distance_bin: int, angle_bin: int, angle_bin_count: int) -> int:
    return int(distance_bin * angle_bin_count + angle_bin)


def compute_momentum_bin(momentum_norm: float, momentum_bin_edges: np.ndarray) -> int:
    idx = int(np.searchsorted(momentum_bin_edges, momentum_norm, side="right") - 1)
    return int(np.clip(idx, 0, len(momentum_bin_edges) - 2))


def sample_reachable_workspace_target(
    anchor_xy: np.ndarray,
    goal_bin_success_counts: np.ndarray,
    distance_bin_edges: np.ndarray,
    angle_bin_count: int,
    distance_bin_prior_power: float,
    rng: np.random.Generator,
    max_attempts: int = 128,
) -> tuple[bool, dict]:
    if angle_bin_count <= 0:
        raise ValueError(f"angle_bin_count must be positive, got {angle_bin_count}")
    weights = 1.0 / (goal_bin_success_counts.astype(np.float64) + 1.0)
    if distance_bin_prior_power > 0.0:
        distance_midpoints = 0.5 * (distance_bin_edges[:-1] + distance_bin_edges[1:])
        distance_prior = np.power(np.maximum(distance_midpoints, 1e-6), -float(distance_bin_prior_power))
        weights *= np.repeat(distance_prior.astype(np.float64), angle_bin_count)
    weights_sum = float(weights.sum())
    if not np.isfinite(weights_sum) or weights_sum <= 0.0:
        weights = np.ones_like(weights, dtype=np.float64)
        weights_sum = float(weights.sum())
    probs = weights / weights_sum

    for _ in range(max_attempts):
        flat_bin = int(rng.choice(len(probs), p=probs))
        distance_bin = flat_bin // angle_bin_count
        angle_bin = flat_bin % angle_bin_count

        lo = float(distance_bin_edges[distance_bin])
        hi = float(distance_bin_edges[distance_bin + 1])
        rho = float(rng.uniform(lo, hi))

        angle_lo = -math.pi + (2.0 * math.pi * angle_bin) / angle_bin_count
        angle_hi = -math.pi + (2.0 * math.pi * (angle_bin + 1)) / angle_bin_count
        theta = float(rng.uniform(angle_lo, angle_hi))

        delta = np.array([rho * math.cos(theta), rho * math.sin(theta)], dtype=np.float64)
        target_xy = anchor_xy.astype(np.float64) + delta
        radius = float(np.linalg.norm(target_xy))
        if radius < REACH_MIN or radius > REACH_MAX:
            continue
        return True, {
            "target_xy": target_xy,
            "goal_distance_from_anchor": rho,
            "goal_angle_from_anchor": wrap_angle_pi(theta),
            "target_distance_bin": distance_bin,
            "target_angle_bin": angle_bin,
            "target_bin_id": compute_goal_bin_id(distance_bin, angle_bin, angle_bin_count),
        }

    return False, {"target_reason": "target_sampling_failed"}


def write_h5_header(
    file: h5py.File,
    xml_path: str,
    num_steps: int,
    num_trajectories: int,
    dt: float,
    config: dict,
) -> None:
    file.attrs["xml"] = Path(xml_path).read_text()
    file.attrs["num_steps"] = int(num_steps)
    file.attrs["num_trajectories"] = int(num_trajectories)
    file.attrs["dt"] = float(dt)
    file.attrs["data_dt"] = float(dt)
    file.attrs["skip_steps"] = 1
    file.attrs["state_alignment"] = "pre_step"
    file.attrs["torque_alignment"] = "interval_mean"
    file.attrs["derivative_alignment"] = "forward_difference"
    file.attrs["generator"] = "reacher_goal_conditioned_branching"
    file.attrs["generator_config"] = json.dumps(config, sort_keys=True)


def finalize_h5_header(file: h5py.File, num_trajectories: int) -> None:
    file.attrs["num_trajectories"] = int(num_trajectories)


def write_trajectory_group(file: h5py.File, traj_index: int, result: dict) -> None:
    group = file.create_group(f"traj_{traj_index}")
    for key in TRAJ_KEYS:
        group.create_dataset(key, data=result[key], dtype="f4")
    group.create_dataset("seq_fingertip_xy", data=result["seq_fingertip_xy"], dtype="f4")
    group.create_dataset("waypoint_xy", data=result["target_xy"], dtype="f4")
    group.create_dataset("target_xy", data=result["target_xy"], dtype="f4")
    group.create_dataset("source_xy", data=result["source_xy"], dtype="f4")
    group.create_dataset("source_qpos", data=result["source_qpos"], dtype="f4")
    group.create_dataset("source_qvel", data=result["source_qvel"], dtype="f4")
    group.create_dataset("source_mom", data=result["source_mom"], dtype="f4")
    group.create_dataset("anchor_state_qpos", data=result["anchor_state_qpos"], dtype="f4")
    group.create_dataset("anchor_state_qvel", data=result["anchor_state_qvel"], dtype="f4")
    group.create_dataset("anchor_state_mom", data=result["anchor_state_mom"], dtype="f4")
    group.create_dataset("anchor_xy", data=result["anchor_xy"], dtype="f4")
    group.create_dataset("boundary_tau", data=result["boundary_tau"], dtype="f4")
    group.create_dataset("boundary_tau_slope", data=result["boundary_tau_slope"], dtype="f4")
    group.create_dataset("goal_distance_from_anchor", data=np.float32(result["goal_distance_from_anchor"]), dtype="f4")
    group.create_dataset("goal_angle_from_anchor", data=np.float32(result["goal_angle_from_anchor"]), dtype="f4")
    group.create_dataset("target_reached_min_dist", data=np.float32(result["target_reached_min_dist"]), dtype="f4")
    group.create_dataset("target_reached_step", data=np.int32(result["target_reached_step"]), dtype="i4")
    group.create_dataset("planner_score", data=np.float32(result["planner_score"]), dtype="f4")
    group.create_dataset("planner_success", data=np.int8(result["planner_success"]), dtype="i1")
    group.attrs["waypoint_index"] = int(result["target_reached_step"])
    group.attrs["target_index"] = int(result["target_reached_step"])
    group.attrs["source_group_id"] = int(result["source_group_id"])
    group.attrs["root_group_id"] = int(result["root_group_id"])
    group.attrs["root_index_within_source"] = int(result["root_index_within_source"])
    group.attrs["anchor_index"] = int(result["anchor_index"])
    group.attrs["anchor_branch_group_id"] = int(result["anchor_branch_group_id"])
    group.attrs["target_group_id"] = int(result["target_group_id"])
    group.attrs["branch_index"] = int(result["branch_index"])
    group.attrs["target_distance_bin"] = int(result["target_distance_bin"])
    group.attrs["target_angle_bin"] = int(result["target_angle_bin"])
    group.attrs["target_bin_id"] = int(result["target_bin_id"])
    group.attrs["momentum_bin"] = int(result["momentum_bin"])
    group.attrs["nearest_sibling_xy_rmse"] = float(result["nearest_sibling_xy_rmse"])
    group.attrs["nearest_sibling_final_xy_gap"] = float(result["nearest_sibling_final_xy_gap"])


def summarize_h5(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        num_trajectories = int(f.attrs["num_trajectories"])
        num_steps = int(f.attrs["num_steps"])
        generator_config = json.loads(f.attrs["generator_config"])
        source_group_ids: list[int] = []
        root_group_ids: list[int] = []
        anchor_group_ids: list[int] = []
        target_group_ids: list[int] = []
        root_indices_within_source: list[int] = []
        anchor_indices: list[int] = []
        goal_distances: list[float] = []
        goal_angles: list[float] = []
        min_dists: list[float] = []
        reached_steps: list[int] = []
        tau_absmax: list[float] = []
        qvel_absmax: list[float] = []
        qacc_absmax: list[float] = []
        source_xy_values: list[np.ndarray] = []
        target_xy_values: list[np.ndarray] = []
        planner_success_values: list[int] = []
        target_bin_counts: dict[int, int] = {}
        distance_bin_counts: dict[int, int] = {}
        angle_bin_counts: dict[int, int] = {}
        momentum_bin_counts: dict[int, int] = {}
        target_groups_to_traj: dict[int, list[int]] = {}
        roots_by_source: dict[int, set[int]] = {}

        for idx in range(num_trajectories):
            group = f[f"traj_{idx}"]
            source_group_id = int(group.attrs.get("source_group_id", -1))
            root_group_id = int(group.attrs.get("root_group_id", -1))
            anchor_group_id = int(group.attrs.get("anchor_branch_group_id", -1))
            target_group_id = int(group.attrs.get("target_group_id", -1))
            root_index_within_source = int(group.attrs.get("root_index_within_source", -1))
            target_bin_id = int(group.attrs.get("target_bin_id", -1))
            target_distance_bin = int(group.attrs.get("target_distance_bin", -1))
            target_angle_bin = int(group.attrs.get("target_angle_bin", -1))
            momentum_bin = int(group.attrs.get("momentum_bin", -1))

            source_group_ids.append(source_group_id)
            root_group_ids.append(root_group_id)
            anchor_group_ids.append(anchor_group_id)
            target_group_ids.append(target_group_id)
            root_indices_within_source.append(root_index_within_source)
            anchor_indices.append(int(group.attrs["anchor_index"]))
            goal_distances.append(float(group["goal_distance_from_anchor"][()]))
            goal_angles.append(float(group["goal_angle_from_anchor"][()]))
            min_dists.append(float(group["target_reached_min_dist"][()]))
            reached_steps.append(int(group["target_reached_step"][()]))
            tau_absmax.append(float(np.max(np.abs(group["seq_torque"][:]))))
            qvel_absmax.append(float(np.max(np.abs(group["seq_qvel"][:]))))
            qacc_absmax.append(float(np.max(np.abs(group["seq_qacc"][:]))))
            source_xy_values.append(group["source_xy"][:].astype(np.float64))
            target_xy_values.append(group["target_xy"][:].astype(np.float64))
            planner_success_values.append(int(group["planner_success"][()]))
            target_bin_counts[target_bin_id] = target_bin_counts.get(target_bin_id, 0) + 1
            distance_bin_counts[target_distance_bin] = distance_bin_counts.get(target_distance_bin, 0) + 1
            angle_bin_counts[target_angle_bin] = angle_bin_counts.get(target_angle_bin, 0) + 1
            momentum_bin_counts[momentum_bin] = momentum_bin_counts.get(momentum_bin, 0) + 1
            target_groups_to_traj.setdefault(target_group_id, []).append(idx)
            roots_by_source.setdefault(source_group_id, set()).add(root_group_id)

        pairwise_rmse: list[float] = []
        pairwise_final_gap: list[float] = []
        for _, indices in target_groups_to_traj.items():
            if len(indices) < 2:
                continue
            samples = [f[f"traj_{idx}"]["seq_fingertip_xy"][:] for idx in indices]
            anchor_index = int(f[f"traj_{indices[0]}"].attrs["anchor_index"])
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    diff = samples[i][anchor_index:] - samples[j][anchor_index:]
                    pairwise_rmse.append(float(np.sqrt(np.mean(np.square(diff), dtype=np.float64))))
                    pairwise_final_gap.append(
                        float(np.linalg.norm(samples[i][-1].astype(np.float64) - samples[j][-1].astype(np.float64)))
                    )

    goal_distance_arr = np.array(goal_distances, dtype=np.float64) if goal_distances else np.empty((0,), dtype=np.float64)
    goal_angle_arr = np.array(goal_angles, dtype=np.float64) if goal_angles else np.empty((0,), dtype=np.float64)
    min_dist_arr = np.array(min_dists, dtype=np.float64) if min_dists else np.empty((0,), dtype=np.float64)
    reached_step_arr = np.array(reached_steps, dtype=np.float64) if reached_steps else np.empty((0,), dtype=np.float64)
    tau_arr = np.array(tau_absmax, dtype=np.float64) if tau_absmax else np.empty((0,), dtype=np.float64)
    qvel_arr = np.array(qvel_absmax, dtype=np.float64) if qvel_absmax else np.empty((0,), dtype=np.float64)
    qacc_arr = np.array(qacc_absmax, dtype=np.float64) if qacc_absmax else np.empty((0,), dtype=np.float64)
    source_xy_arr = np.stack(source_xy_values, axis=0) if source_xy_values else np.empty((0, 2), dtype=np.float64)
    target_xy_arr = np.stack(target_xy_values, axis=0) if target_xy_values else np.empty((0, 2), dtype=np.float64)
    planner_success_arr = (
        np.array(planner_success_values, dtype=np.float64) if planner_success_values else np.empty((0,), dtype=np.float64)
    )

    summary = {
        "file": str(h5_path),
        "num_trajectories": int(num_trajectories),
        "num_steps": int(num_steps),
        "generator_config": generator_config,
        "source_groups": int(len(set(source_group_ids))),
        "root_groups": int(len(set(root_group_ids))),
        "anchor_groups": int(len(set(anchor_group_ids))),
        "target_groups": int(len(set(target_group_ids))),
        "planner_success_rate": float(planner_success_arr.mean()) if len(planner_success_arr) else 0.0,
        "accepted_root_groups_per_source": {str(k): int(len(v)) for k, v in sorted(roots_by_source.items())},
    }
    if len(goal_distance_arr):
        summary["goal_distance_from_anchor"] = {
            "min": float(goal_distance_arr.min()),
            "mean": float(goal_distance_arr.mean()),
            "p50": float(np.percentile(goal_distance_arr, 50)),
            "p90": float(np.percentile(goal_distance_arr, 90)),
            "max": float(goal_distance_arr.max()),
        }
    if len(goal_angle_arr):
        summary["goal_angle_from_anchor"] = {
            "mean": float(goal_angle_arr.mean()),
            "p05": float(np.percentile(goal_angle_arr, 5)),
            "p95": float(np.percentile(goal_angle_arr, 95)),
        }
    if len(min_dist_arr):
        summary["target_reached_min_dist"] = {
            "min": float(min_dist_arr.min()),
            "mean": float(min_dist_arr.mean()),
            "p50": float(np.percentile(min_dist_arr, 50)),
            "p90": float(np.percentile(min_dist_arr, 90)),
            "max": float(min_dist_arr.max()),
        }
    if len(reached_step_arr):
        summary["target_reached_step"] = {
            "min": int(reached_step_arr.min()),
            "mean": float(reached_step_arr.mean()),
            "max": int(reached_step_arr.max()),
        }
    if len(tau_arr):
        summary["max_abs_tau"] = {
            "mean": float(tau_arr.mean()),
            "p95": float(np.percentile(tau_arr, 95)),
            "max": float(tau_arr.max()),
        }
    if len(qvel_arr):
        summary["max_abs_qvel"] = {
            "mean": float(qvel_arr.mean()),
            "p95": float(np.percentile(qvel_arr, 95)),
            "max": float(qvel_arr.max()),
        }
    if len(qacc_arr):
        summary["max_abs_qacc"] = {
            "mean": float(qacc_arr.mean()),
            "p95": float(np.percentile(qacc_arr, 95)),
            "max": float(qacc_arr.max()),
        }
    if len(source_xy_arr):
        source_radii = np.linalg.norm(source_xy_arr, axis=1)
        summary["source_xy"] = {
            "x_min": float(source_xy_arr[:, 0].min()),
            "x_max": float(source_xy_arr[:, 0].max()),
            "y_min": float(source_xy_arr[:, 1].min()),
            "y_max": float(source_xy_arr[:, 1].max()),
            "radius_min": float(source_radii.min()),
            "radius_mean": float(source_radii.mean()),
            "radius_max": float(source_radii.max()),
        }
    if len(target_xy_arr):
        target_radii = np.linalg.norm(target_xy_arr, axis=1)
        summary["target_xy"] = {
            "x_min": float(target_xy_arr[:, 0].min()),
            "x_max": float(target_xy_arr[:, 0].max()),
            "y_min": float(target_xy_arr[:, 1].min()),
            "y_max": float(target_xy_arr[:, 1].max()),
            "radius_min": float(target_radii.min()),
            "radius_mean": float(target_radii.mean()),
            "radius_max": float(target_radii.max()),
        }
    valid_root_indices = [v for v in root_indices_within_source if v >= 0]
    if valid_root_indices:
        root_idx_arr = np.array(valid_root_indices, dtype=np.float64)
        summary["root_index_within_source"] = {
            "min": int(root_idx_arr.min()),
            "mean": float(root_idx_arr.mean()),
            "max": int(root_idx_arr.max()),
        }
    if pairwise_rmse:
        rmse_arr = np.array(pairwise_rmse, dtype=np.float64)
        final_gap_arr = np.array(pairwise_final_gap, dtype=np.float64)
        summary["pairwise_target_group_suffix_xy_rmse"] = {
            "mean": float(rmse_arr.mean()),
            "p05": float(np.percentile(rmse_arr, 5)),
            "min": float(rmse_arr.min()),
        }
        summary["pairwise_target_group_final_xy_gap"] = {
            "mean": float(final_gap_arr.mean()),
            "p05": float(np.percentile(final_gap_arr, 5)),
            "min": float(final_gap_arr.min()),
        }
    summary["target_bin_counts"] = {str(k): int(v) for k, v in sorted(target_bin_counts.items())}
    summary["target_distance_bin_counts"] = {str(k): int(v) for k, v in sorted(distance_bin_counts.items())}
    summary["target_angle_bin_counts"] = {str(k): int(v) for k, v in sorted(angle_bin_counts.items())}
    summary["momentum_bin_counts"] = {str(k): int(v) for k, v in sorted(momentum_bin_counts.items())}
    return summary


def score_rollout_against_target(
    replay: dict[str, np.ndarray],
    target_xy: np.ndarray,
    anchor_index: int,
    suffix_tau: np.ndarray,
    max_abs_qvel: float,
    max_abs_qacc: float,
) -> dict[str, float]:
    suffix_xy = replay["seq_fingertip_xy"][anchor_index:].astype(np.float64)
    dists = np.linalg.norm(suffix_xy - target_xy[None, :], axis=1)
    min_offset = int(np.argmin(dists))
    min_dist = float(dists[min_offset])
    final_dist = float(np.linalg.norm(replay["seq_fingertip_xy"][-1].astype(np.float64) - target_xy))

    if len(suffix_tau):
        control_energy = float(np.mean(np.sum(np.square(suffix_tau, dtype=np.float64), axis=1), dtype=np.float64))
    else:
        control_energy = 0.0

    if len(suffix_tau) >= 2:
        diff_tau = np.diff(suffix_tau, axis=0)
        torque_smoothness = float(np.mean(np.sum(np.square(diff_tau, dtype=np.float64), axis=1), dtype=np.float64))
    else:
        torque_smoothness = 0.0

    qvel_ratio = float(np.max(np.abs(replay["seq_qvel"])) / max_abs_qvel)
    qacc_ratio = float(np.max(np.abs(replay["seq_qacc"])) / max_abs_qacc)
    physics_penalty = qvel_ratio * qvel_ratio + qacc_ratio * qacc_ratio

    planner_score = (
        min_dist
        + 0.5 * final_dist
        + 0.01 * control_energy
        + 0.05 * torque_smoothness
        + 1.0 * physics_penalty
    )

    return {
        "target_reached_min_dist": min_dist,
        "target_reached_step": float(anchor_index + min_offset),
        "planner_score": planner_score,
    }


def build_anchor_target_group(
    model: mujoco.MjModel,
    ids: dict[str, int],
    root_info: dict,
    trajectory_length: int,
    dt: float,
    anchor_index: int,
    target_info: dict,
    branches_per_target_keep: int,
    suffix_control_points: int,
    max_abs_tau: np.ndarray,
    max_abs_qvel: float,
    max_abs_qacc: float,
    min_suffix_xy_rmse: float,
    min_final_xy_gap: float,
    boundary_margin_ratio: float,
    boundary_tau_slope_cap: float,
    success_threshold: float,
    candidates_per_iter: int,
    elite_count: int,
    cem_iters: int,
    initial_residual_std: float,
    std_floor: float,
    source_group_id: int,
    root_group_id: int,
    root_index_within_source: int,
    anchor_branch_group_id: int,
    target_group_id: int,
    rng: np.random.Generator,
) -> tuple[bool, dict]:
    dummy_waypoint = np.zeros(2, dtype=np.float64)
    suffix_steps = int(trajectory_length - anchor_index)
    if suffix_steps <= 0:
        return False, {"target_reason": "anchor_invalid", "candidate_evals": 0, "candidate_reject_counts": {}}

    root_tau = root_info["root_tau"]
    source_xy = root_info["source_xy"]
    source_qpos = root_info["source_qpos"]
    source_qvel = root_info["source_qvel"]
    source_mom = root_info["source_mom"]
    root_rollout = root_info["root_rollout"]

    prefix_tau = root_tau[:anchor_index].astype(np.float64)
    if len(prefix_tau) == 0:
        return False, {"target_reason": "anchor_no_prefix", "candidate_evals": 0, "candidate_reject_counts": {}}

    boundary_tau = prefix_tau[-1].astype(np.float64)
    boundary_tau_slope = np.zeros(model.nu, dtype=np.float64)
    if anchor_index >= 2:
        raw_slope = (prefix_tau[-1] - prefix_tau[-2]) / dt
        boundary_tau_slope = np.clip(raw_slope, -boundary_tau_slope_cap, boundary_tau_slope_cap)

    if np.any(np.abs(boundary_tau) > max_abs_tau * boundary_margin_ratio):
        return False, {
            "target_reason": "anchor_boundary_margin",
            "candidate_evals": 0,
            "candidate_reject_counts": {},
        }

    anchor_state_qpos = root_rollout["seq_qpos"][anchor_index].astype(np.float64)
    anchor_state_qvel = root_rollout["seq_qvel"][anchor_index].astype(np.float64)
    anchor_xy = root_rollout["seq_fingertip_xy"][anchor_index].astype(np.float64)
    anchor_state_mom = compute_source_mom(model, ids, anchor_state_qpos, anchor_state_qvel, dummy_waypoint)
    momentum_bin = compute_momentum_bin(float(np.linalg.norm(anchor_state_mom)), target_info["momentum_bin_edges"])

    dim = suffix_control_points * model.nu
    mean = np.zeros(dim, dtype=np.float64)
    std = np.full(dim, float(initial_residual_std), dtype=np.float64)
    sample_clip = 3.0 * float(initial_residual_std)

    candidate_evals = 0
    candidate_reject_counts: dict[str, int] = {}
    successful_candidates: list[dict] = []

    for _ in range(cem_iters):
        candidate_vectors = mean[None, :] + std[None, :] * rng.standard_normal((candidates_per_iter, dim))
        candidate_vectors = np.clip(candidate_vectors, -sample_clip, sample_clip)
        valid_candidates: list[dict] = []

        for vector in candidate_vectors:
            candidate_evals += 1
            residual_controls = vector.reshape(suffix_control_points, model.nu)
            ok, suffix_tau = boundary_conditioned_suffix(
                boundary_tau=boundary_tau,
                boundary_tau_slope=boundary_tau_slope,
                residual_controls=residual_controls,
                suffix_steps=suffix_steps,
                dt=dt,
                tau_limit=max_abs_tau,
            )
            if not ok or suffix_tau is None:
                candidate_reject_counts["suffix_tau_limit"] = candidate_reject_counts.get("suffix_tau_limit", 0) + 1
                continue

            full_tau = np.concatenate([prefix_tau, suffix_tau], axis=0)
            try:
                replay = simulate_dataset_rollout(
                    model=model,
                    start_arm_qpos=source_qpos,
                    start_arm_qvel=source_qvel,
                    waypoint_xy=dummy_waypoint,
                    torque_seq=full_tau,
                    ids=ids,
                )
            except FloatingPointError:
                candidate_reject_counts["replay_non_finite"] = candidate_reject_counts.get("replay_non_finite", 0) + 1
                continue

            if np.any(np.abs(replay["seq_qvel"]) > max_abs_qvel):
                candidate_reject_counts["qvel_limit"] = candidate_reject_counts.get("qvel_limit", 0) + 1
                continue
            if np.any(np.abs(replay["seq_qacc"]) > max_abs_qacc):
                candidate_reject_counts["qacc_limit"] = candidate_reject_counts.get("qacc_limit", 0) + 1
                continue

            score_info = score_rollout_against_target(
                replay=replay,
                target_xy=target_info["target_xy"],
                anchor_index=anchor_index,
                suffix_tau=suffix_tau,
                max_abs_qvel=max_abs_qvel,
                max_abs_qacc=max_abs_qacc,
            )
            candidate = {
                "vector": vector.astype(np.float64, copy=True),
                "replay": replay,
                "planner_score": float(score_info["planner_score"]),
                "target_reached_min_dist": float(score_info["target_reached_min_dist"]),
                "target_reached_step": int(score_info["target_reached_step"]),
                "suffix_tau": suffix_tau,
            }
            valid_candidates.append(candidate)
            if candidate["target_reached_min_dist"] <= success_threshold:
                successful_candidates.append(candidate)

        if valid_candidates:
            valid_candidates.sort(key=lambda item: item["planner_score"])
            elite = valid_candidates[: max(1, min(elite_count, len(valid_candidates)))]
            elite_vectors = np.stack([item["vector"] for item in elite], axis=0)
            mean = elite_vectors.mean(axis=0)
            std = np.maximum(elite_vectors.std(axis=0), std_floor)
        else:
            std = np.maximum(std * 0.5, std_floor)

    if not successful_candidates:
        return False, {
            "target_reason": "target_no_success",
            "candidate_evals": int(candidate_evals),
            "candidate_reject_counts": candidate_reject_counts,
        }

    successful_candidates.sort(key=lambda item: item["planner_score"])
    accepted_results: list[dict] = []
    for candidate in successful_candidates:
        nearest_rmse, nearest_final_gap = branch_diversity_metrics(
            candidate_xy=candidate["replay"]["seq_fingertip_xy"],
            sibling_results=accepted_results,
            branch_point=anchor_index,
        )
        if accepted_results and nearest_rmse < min_suffix_xy_rmse and nearest_final_gap < min_final_xy_gap:
            candidate_reject_counts["branch_too_similar"] = candidate_reject_counts.get("branch_too_similar", 0) + 1
            continue

        replay = dict(candidate["replay"])
        replay["target_xy"] = target_info["target_xy"].astype(np.float32)
        replay["waypoint_xy"] = target_info["target_xy"].astype(np.float32)
        replay["source_xy"] = source_xy.astype(np.float32)
        replay["source_qpos"] = source_qpos.astype(np.float32)
        replay["source_qvel"] = source_qvel.astype(np.float32)
        replay["source_mom"] = source_mom.astype(np.float32)
        replay["anchor_state_qpos"] = anchor_state_qpos.astype(np.float32)
        replay["anchor_state_qvel"] = anchor_state_qvel.astype(np.float32)
        replay["anchor_state_mom"] = anchor_state_mom.astype(np.float32)
        replay["anchor_xy"] = anchor_xy.astype(np.float32)
        replay["boundary_tau"] = boundary_tau.astype(np.float32)
        replay["boundary_tau_slope"] = boundary_tau_slope.astype(np.float32)
        replay["goal_distance_from_anchor"] = np.float32(target_info["goal_distance_from_anchor"])
        replay["goal_angle_from_anchor"] = np.float32(target_info["goal_angle_from_anchor"])
        replay["target_reached_min_dist"] = np.float32(candidate["target_reached_min_dist"])
        replay["target_reached_step"] = np.int32(candidate["target_reached_step"])
        replay["planner_score"] = np.float32(candidate["planner_score"])
        replay["planner_success"] = np.int8(1)
        replay["source_group_id"] = np.int32(source_group_id)
        replay["root_group_id"] = np.int32(root_group_id)
        replay["root_index_within_source"] = np.int32(root_index_within_source)
        replay["anchor_index"] = np.int32(anchor_index)
        replay["anchor_branch_group_id"] = np.int32(anchor_branch_group_id)
        replay["target_group_id"] = np.int32(target_group_id)
        replay["branch_index"] = np.int32(len(accepted_results))
        replay["target_distance_bin"] = np.int32(target_info["target_distance_bin"])
        replay["target_angle_bin"] = np.int32(target_info["target_angle_bin"])
        replay["target_bin_id"] = np.int32(target_info["target_bin_id"])
        replay["momentum_bin"] = np.int32(momentum_bin)
        replay["nearest_sibling_xy_rmse"] = np.float32(nearest_rmse if np.isfinite(nearest_rmse) else np.nan)
        replay["nearest_sibling_final_xy_gap"] = np.float32(
            nearest_final_gap if np.isfinite(nearest_final_gap) else np.nan
        )
        accepted_results.append(replay)
        if len(accepted_results) >= branches_per_target_keep:
            break

    if not accepted_results:
        return False, {
            "target_reason": "target_no_diverse_success",
            "candidate_evals": int(candidate_evals),
            "candidate_reject_counts": candidate_reject_counts,
        }

    return True, {
        "results": accepted_results,
        "candidate_evals": int(candidate_evals),
        "candidate_reject_counts": candidate_reject_counts,
        "target_bin_id": int(target_info["target_bin_id"]),
    }


def generate_split(
    output_path: Path,
    xml_path: str,
    num_trajectories: int | None,
    trajectory_length: int,
    dt: float,
    source_qvel_scale: float,
    root_control_points: int,
    root_tau_scale: float,
    anchor_min: int,
    suffix_min_len: int,
    anchor_stride: int,
    anchors_per_root: int,
    roots_per_source: int,
    targets_per_anchor: int,
    branches_per_target_keep: int,
    suffix_control_points: int,
    success_threshold: float,
    candidates_per_iter: int,
    elite_count: int,
    cem_iters: int,
    initial_residual_std: float,
    std_floor: float,
    distance_bin_edges: list[float],
    angle_bin_count: int,
    distance_bin_prior_power: float,
    momentum_bin_edges: list[float],
    max_target_sampling_attempts: int,
    max_abs_tau: float | None,
    max_abs_qvel: float,
    max_abs_qacc: float,
    min_suffix_xy_rmse: float,
    min_final_xy_gap: float,
    boundary_margin_ratio: float,
    boundary_tau_slope_cap: float,
    seed_offset: int,
    split_name: str,
    source_budget: int | None = None,
) -> dict:
    if num_trajectories is None and source_budget is None:
        raise ValueError("Either num_trajectories or source_budget must be provided")
    if num_trajectories is not None and num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive when provided")
    if source_budget is not None and source_budget <= 0:
        raise ValueError("source_budget must be positive when provided")
    if targets_per_anchor <= 0:
        raise ValueError(f"targets_per_anchor must be positive, got {targets_per_anchor}")
    if branches_per_target_keep <= 0:
        raise ValueError(f"branches_per_target_keep must be positive, got {branches_per_target_keep}")
    if elite_count <= 0 or elite_count > candidates_per_iter:
        raise ValueError(
            f"elite_count must be in [1, candidates_per_iter], got elite_count={elite_count}, "
            f"candidates_per_iter={candidates_per_iter}"
        )

    distance_bin_edges_arr = np.asarray(distance_bin_edges, dtype=np.float64)
    momentum_bin_edges_arr = np.asarray(momentum_bin_edges, dtype=np.float64)
    if distance_bin_edges_arr.ndim != 1 or len(distance_bin_edges_arr) < 2:
        raise ValueError("distance_bin_edges must have at least two entries")
    if momentum_bin_edges_arr.ndim != 1 or len(momentum_bin_edges_arr) < 2:
        raise ValueError("momentum_bin_edges must have at least two entries")
    if not np.all(np.diff(distance_bin_edges_arr) > 0):
        raise ValueError("distance_bin_edges must be strictly increasing")
    if not np.all(np.diff(momentum_bin_edges_arr) > 0):
        raise ValueError("momentum_bin_edges must be strictly increasing")

    requested_max_abs_tau = None if max_abs_tau is None else float(max_abs_tau)

    config = {
        "trajectory_length": int(trajectory_length),
        "dt": float(dt),
        "source_qvel_scale": float(source_qvel_scale),
        "root_control_points": int(root_control_points),
        "root_tau_scale": float(root_tau_scale),
        "anchor_min": int(anchor_min),
        "suffix_min_len": int(suffix_min_len),
        "anchor_stride": int(anchor_stride),
        "anchors_per_root": int(anchors_per_root),
        "roots_per_source": int(roots_per_source),
        "targets_per_anchor": int(targets_per_anchor),
        "branches_per_target_keep": int(branches_per_target_keep),
        "suffix_control_points": int(suffix_control_points),
        "success_threshold": float(success_threshold),
        "candidates_per_iter": int(candidates_per_iter),
        "elite_count": int(elite_count),
        "cem_iters": int(cem_iters),
        "initial_residual_std": float(initial_residual_std),
        "std_floor": float(std_floor),
        "distance_bin_edges": [float(v) for v in distance_bin_edges_arr.tolist()],
        "angle_bin_count": int(angle_bin_count),
        "distance_bin_prior_power": float(distance_bin_prior_power),
        "momentum_bin_edges": [float(v) for v in momentum_bin_edges_arr.tolist()],
        "max_target_sampling_attempts": int(max_target_sampling_attempts),
        "requested_max_abs_tau": requested_max_abs_tau,
        "max_abs_qvel": float(max_abs_qvel),
        "max_abs_qacc": float(max_abs_qacc),
        "min_suffix_xy_rmse": float(min_suffix_xy_rmse),
        "min_final_xy_gap": float(min_final_xy_gap),
        "boundary_margin_ratio": float(boundary_margin_ratio),
        "boundary_tau_slope_cap": float(boundary_tau_slope_cap),
        "split_name": split_name,
        "source_budget": None if source_budget is None else int(source_budget),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = float(dt)
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    ids = get_model_ids(model)
    rng = np.random.default_rng(seed_offset)
    effective_max_abs_tau, hard_max_abs_tau = resolve_requested_torque_limit(model, requested_max_abs_tau)
    config["max_abs_tau"] = json_safe_limit(effective_max_abs_tau)
    config["model_hard_max_abs_tau"] = json_safe_limit(hard_max_abs_tau)

    goal_bin_success_counts = np.zeros((len(distance_bin_edges_arr) - 1) * angle_bin_count, dtype=np.int64)

    accepted = 0
    source_attempts = 0
    source_successes = 0
    root_attempts = 0
    root_successes = 0
    anchor_group_attempts = 0
    anchor_group_successes = 0
    target_group_attempts = 0
    target_group_successes = 0
    candidate_evals = 0
    source_group_id = 0
    root_group_id = 0
    anchor_branch_group_id = 0
    target_group_id = 0
    root_reject_counts: dict[str, int] = {}
    anchor_group_reject_counts: dict[str, int] = {}
    target_group_reject_counts: dict[str, int] = {}
    candidate_reject_counts: dict[str, int] = {}

    pbar = tqdm(total=num_trajectories, desc=f"Generating {split_name}")

    with h5py.File(output_path, "w") as h5_file:
        write_h5_header(
            file=h5_file,
            xml_path=xml_path,
            num_steps=trajectory_length,
            num_trajectories=0 if num_trajectories is None else num_trajectories,
            dt=dt,
            config=config,
        )

        while True:
            if num_trajectories is not None and accepted >= num_trajectories:
                break
            if source_budget is not None and source_attempts >= source_budget:
                break

            source_attempts += 1
            current_source_group_id = source_group_id
            source_qpos, source_qvel, source_xy = sample_workspace_uniform_source_state(
                rng=rng,
                source_qvel_scale=source_qvel_scale,
            )

            source_had_success = False
            for root_index_within_source in range(roots_per_source):
                if num_trajectories is not None and accepted >= num_trajectories:
                    break

                root_attempts += 1
                root_ok, root_info = build_root_rollout(
                    model=model,
                    ids=ids,
                    rng=rng,
                    trajectory_length=trajectory_length,
                    dt=dt,
                    source_qvel_scale=source_qvel_scale,
                    root_control_points=root_control_points,
                    root_tau_scale=root_tau_scale,
                    max_abs_tau=effective_max_abs_tau,
                    max_abs_qvel=max_abs_qvel,
                    max_abs_qacc=max_abs_qacc,
                    source_qpos=source_qpos,
                    source_qvel=source_qvel,
                    source_xy=source_xy,
                )
                if not root_ok:
                    reason = str(root_info.get("root_reason", "root_reject"))
                    root_reject_counts[reason] = root_reject_counts.get(reason, 0) + 1
                    continue

                anchor_indices = choose_anchor_indices(
                    trajectory_length=trajectory_length,
                    anchor_min=anchor_min,
                    suffix_min_len=suffix_min_len,
                    anchor_stride=anchor_stride,
                    anchors_per_root=anchors_per_root,
                    rng=rng,
                )
                if not anchor_indices:
                    root_reject_counts["root_no_valid_anchors"] = root_reject_counts.get("root_no_valid_anchors", 0) + 1
                    continue

                current_root_group_id = root_group_id
                root_had_success = False
                for anchor_index in anchor_indices:
                    if num_trajectories is not None and accepted >= num_trajectories:
                        break

                    anchor_group_attempts += 1
                    current_anchor_group_id = anchor_branch_group_id
                    anchor_xy = root_info["root_rollout"]["seq_fingertip_xy"][anchor_index].astype(np.float64)
                    anchor_had_success = False

                    for _ in range(targets_per_anchor):
                        if num_trajectories is not None and accepted >= num_trajectories:
                            break
                        target_group_attempts += 1
                        sampled, target_info = sample_reachable_workspace_target(
                            anchor_xy=anchor_xy,
                            goal_bin_success_counts=goal_bin_success_counts,
                            distance_bin_edges=distance_bin_edges_arr,
                            angle_bin_count=angle_bin_count,
                            distance_bin_prior_power=distance_bin_prior_power,
                            rng=rng,
                            max_attempts=max_target_sampling_attempts,
                        )
                        if not sampled:
                            target_group_reject_counts["target_sampling_failed"] = (
                                target_group_reject_counts.get("target_sampling_failed", 0) + 1
                            )
                            continue
                        target_info["momentum_bin_edges"] = momentum_bin_edges_arr

                        ok, info = build_anchor_target_group(
                            model=model,
                            ids=ids,
                            root_info=root_info,
                            trajectory_length=trajectory_length,
                            dt=dt,
                            anchor_index=anchor_index,
                            target_info=target_info,
                            branches_per_target_keep=branches_per_target_keep,
                            suffix_control_points=suffix_control_points,
                            max_abs_tau=effective_max_abs_tau,
                            max_abs_qvel=max_abs_qvel,
                            max_abs_qacc=max_abs_qacc,
                            min_suffix_xy_rmse=min_suffix_xy_rmse,
                            min_final_xy_gap=min_final_xy_gap,
                            boundary_margin_ratio=boundary_margin_ratio,
                            boundary_tau_slope_cap=boundary_tau_slope_cap,
                            success_threshold=success_threshold,
                            candidates_per_iter=candidates_per_iter,
                            elite_count=elite_count,
                            cem_iters=cem_iters,
                            initial_residual_std=initial_residual_std,
                            std_floor=std_floor,
                            source_group_id=current_source_group_id,
                            root_group_id=current_root_group_id,
                            root_index_within_source=root_index_within_source,
                            anchor_branch_group_id=current_anchor_group_id,
                            target_group_id=target_group_id,
                            rng=rng,
                        )
                        candidate_evals += int(info.get("candidate_evals", 0))
                        for reason, count in info.get("candidate_reject_counts", {}).items():
                            candidate_reject_counts[reason] = candidate_reject_counts.get(reason, 0) + int(count)

                        if ok:
                            for result in info["results"]:
                                if num_trajectories is not None and accepted >= num_trajectories:
                                    break
                                write_trajectory_group(h5_file, accepted, result)
                                accepted += 1
                                pbar.update(1)
                            target_group_successes += 1
                            goal_bin_success_counts[int(info["target_bin_id"])] += 1
                            target_group_id += 1
                            anchor_had_success = True
                        else:
                            reason = str(info.get("target_reason", "target_reject"))
                            target_group_reject_counts[reason] = target_group_reject_counts.get(reason, 0) + 1

                    if anchor_had_success:
                        anchor_group_successes += 1
                        anchor_branch_group_id += 1
                        root_had_success = True
                    else:
                        anchor_group_reject_counts["anchor_no_successful_targets"] = (
                            anchor_group_reject_counts.get("anchor_no_successful_targets", 0) + 1
                        )

                if root_had_success:
                    root_successes += 1
                    root_group_id += 1
                    source_had_success = True
                else:
                    root_reject_counts["root_no_successful_anchor_groups"] = (
                        root_reject_counts.get("root_no_successful_anchor_groups", 0) + 1
                    )

            if source_had_success:
                source_successes += 1
                source_group_id += 1

            accept_rate = accepted / max(1, candidate_evals)
            pbar.set_postfix(
                sources=source_successes,
                roots=root_successes,
                anchors=anchor_group_successes,
                targets=target_group_successes,
                candidate_evals=candidate_evals,
                accept_rate=f"{accept_rate:.2%}",
            )

        finalize_h5_header(h5_file, accepted)
    pbar.close()

    summary = {
        "output_path": str(output_path),
        "num_trajectories": int(accepted),
        "requested_num_trajectories": None if num_trajectories is None else int(num_trajectories),
        "source_attempts": int(source_attempts),
        "source_successes": int(source_successes),
        "root_attempts": int(root_attempts),
        "root_successes": int(root_successes),
        "anchor_group_attempts": int(anchor_group_attempts),
        "anchor_group_successes": int(anchor_group_successes),
        "target_group_attempts": int(target_group_attempts),
        "target_group_successes": int(target_group_successes),
        "candidate_evals": int(candidate_evals),
        "accept_rate": float(accepted / max(1, candidate_evals)),
        "source_success_rate": float(source_successes / max(1, source_attempts)),
        "root_success_rate": float(root_successes / max(1, root_attempts)),
        "anchor_group_success_rate": float(anchor_group_successes / max(1, anchor_group_attempts)),
        "target_group_success_rate": float(target_group_successes / max(1, target_group_attempts)),
        "root_reject_counts": root_reject_counts,
        "anchor_group_reject_counts": anchor_group_reject_counts,
        "target_group_reject_counts": target_group_reject_counts,
        "candidate_reject_counts": candidate_reject_counts,
        "source_groups": int(source_successes),
        "root_groups": int(root_group_id),
        "anchor_groups": int(anchor_branch_group_id),
        "target_groups": int(target_group_id),
        "target_bin_success_counts": {str(i): int(v) for i, v in enumerate(goal_bin_success_counts.tolist())},
        "generator_config": config,
    }
    print(
        f"[{split_name}] saved {accepted} trajectories to {output_path} "
        f"(source attempts {source_attempts}, root attempts {root_attempts}, anchor groups {anchor_group_attempts}, "
        f"target groups {target_group_attempts}, candidate evals {candidate_evals}, "
        f"candidate accept rate {accepted / max(1, candidate_evals):.2%})",
        flush=True,
    )
    print(
        f"  - torque_limit_applied: requested={requested_max_abs_tau}, "
        f"effective={json_safe_limit(effective_max_abs_tau)}, "
        f"model_hard={json_safe_limit(hard_max_abs_tau)}",
        flush=True,
    )
    for reason, count in sorted(root_reject_counts.items()):
        print(f"  - root {reason}: {count}", flush=True)
    for reason, count in sorted(anchor_group_reject_counts.items()):
        print(f"  - anchor {reason}: {count}", flush=True)
    for reason, count in sorted(target_group_reject_counts.items()):
        print(f"  - target {reason}: {count}", flush=True)
    for reason, count in sorted(candidate_reject_counts.items()):
        print(f"  - candidate {reason}: {count}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a goal-conditioned per-step branching Reacher dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/data/reacher_goal_conditioned_branching_dt0p001_len500_pilot",
    )
    parser.add_argument(
        "--xml_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/configs/reacher_non_diss_unbounded_j1.xml",
    )
    parser.add_argument("--train_trajectories", type=int, default=64)
    parser.add_argument("--val_trajectories", type=int, default=16)
    parser.add_argument("--train_sources", type=int, default=0)
    parser.add_argument("--val_sources", type=int, default=0)
    parser.add_argument("--trajectory_length", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--source_qvel_scale", type=float, default=0.1)
    parser.add_argument("--root_control_points", type=int, default=12)
    parser.add_argument("--root_tau_scale", type=float, default=0.08)
    parser.add_argument("--anchor_min", type=int, default=8)
    parser.add_argument("--suffix_min_len", type=int, default=8)
    parser.add_argument("--anchor_stride", type=int, default=8)
    parser.add_argument("--anchors_per_root", type=int, default=2)
    parser.add_argument("--roots_per_source", type=int, default=2)
    parser.add_argument("--targets_per_anchor", type=int, default=3)
    parser.add_argument("--branches_per_target_keep", type=int, default=2)
    parser.add_argument("--suffix_control_points", type=int, default=16)
    parser.add_argument("--success_threshold", type=float, default=0.005)
    parser.add_argument("--candidates_per_iter", type=int, default=128)
    parser.add_argument("--elite_count", type=int, default=16)
    parser.add_argument("--cem_iters", type=int, default=6)
    parser.add_argument("--initial_residual_std", type=float, default=0.12)
    parser.add_argument("--std_floor", type=float, default=0.01)
    parser.add_argument("--distance_bin_edges", type=float, nargs="+", default=list(DEFAULT_DISTANCE_BIN_EDGES))
    parser.add_argument("--angle_bin_count", type=int, default=24)
    parser.add_argument("--distance_bin_prior_power", type=float, default=0.0)
    parser.add_argument("--momentum_bin_edges", type=float, nargs="+", default=list(DEFAULT_MOMENTUM_BIN_EDGES))
    parser.add_argument("--max_target_sampling_attempts", type=int, default=128)
    parser.add_argument("--max_abs_tau", type=float, default=None)
    parser.add_argument("--max_abs_qvel", type=float, default=4.0 * math.pi)
    parser.add_argument("--max_abs_qacc", type=float, default=100.0)
    parser.add_argument("--min_suffix_xy_rmse", type=float, default=0.003)
    parser.add_argument("--min_final_xy_gap", type=float, default=0.005)
    parser.add_argument("--boundary_margin_ratio", type=float, default=0.95)
    parser.add_argument("--boundary_tau_slope_cap", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / (
        f"train_traj_from_{args.train_sources}_sources_steps_{args.trajectory_length}.h5"
        if args.train_sources > 0
        else f"train_traj_{args.train_trajectories}-steps_{args.trajectory_length}.h5"
    )
    val_path = output_dir / (
        f"val_traj_from_{args.val_sources}_sources_steps_{args.trajectory_length}.h5"
        if args.val_sources > 0
        else f"val_traj_{args.val_trajectories}-steps_{args.trajectory_length}.h5"
    )

    print("Configuration:", flush=True)
    for key, value in vars(args).items():
        print(f"  {key}: {value}", flush=True)

    train_generation = generate_split(
        output_path=train_path,
        xml_path=args.xml_path,
        num_trajectories=None if args.train_sources > 0 else args.train_trajectories,
        trajectory_length=args.trajectory_length,
        dt=args.dt,
        source_qvel_scale=args.source_qvel_scale,
        root_control_points=args.root_control_points,
        root_tau_scale=args.root_tau_scale,
        anchor_min=args.anchor_min,
        suffix_min_len=args.suffix_min_len,
        anchor_stride=args.anchor_stride,
        anchors_per_root=args.anchors_per_root,
        roots_per_source=args.roots_per_source,
        targets_per_anchor=args.targets_per_anchor,
        branches_per_target_keep=args.branches_per_target_keep,
        suffix_control_points=args.suffix_control_points,
        success_threshold=args.success_threshold,
        candidates_per_iter=args.candidates_per_iter,
        elite_count=args.elite_count,
        cem_iters=args.cem_iters,
        initial_residual_std=args.initial_residual_std,
        std_floor=args.std_floor,
        distance_bin_edges=args.distance_bin_edges,
        angle_bin_count=args.angle_bin_count,
        distance_bin_prior_power=args.distance_bin_prior_power,
        momentum_bin_edges=args.momentum_bin_edges,
        max_target_sampling_attempts=args.max_target_sampling_attempts,
        max_abs_tau=args.max_abs_tau,
        max_abs_qvel=args.max_abs_qvel,
        max_abs_qacc=args.max_abs_qacc,
        min_suffix_xy_rmse=args.min_suffix_xy_rmse,
        min_final_xy_gap=args.min_final_xy_gap,
        boundary_margin_ratio=args.boundary_margin_ratio,
        boundary_tau_slope_cap=args.boundary_tau_slope_cap,
        seed_offset=0,
        split_name="train",
        source_budget=args.train_sources if args.train_sources > 0 else None,
    )

    val_generation = None
    if args.val_trajectories > 0 or args.val_sources > 0:
        val_generation = generate_split(
            output_path=val_path,
            xml_path=args.xml_path,
            num_trajectories=None if args.val_sources > 0 else args.val_trajectories,
            trajectory_length=args.trajectory_length,
            dt=args.dt,
            source_qvel_scale=args.source_qvel_scale,
            root_control_points=args.root_control_points,
            root_tau_scale=args.root_tau_scale,
            anchor_min=args.anchor_min,
            suffix_min_len=args.suffix_min_len,
            anchor_stride=args.anchor_stride,
            anchors_per_root=args.anchors_per_root,
            roots_per_source=args.roots_per_source,
            targets_per_anchor=args.targets_per_anchor,
            branches_per_target_keep=args.branches_per_target_keep,
            suffix_control_points=args.suffix_control_points,
            success_threshold=args.success_threshold,
            candidates_per_iter=args.candidates_per_iter,
            elite_count=args.elite_count,
            cem_iters=args.cem_iters,
            initial_residual_std=args.initial_residual_std,
            std_floor=args.std_floor,
            distance_bin_edges=args.distance_bin_edges,
            angle_bin_count=args.angle_bin_count,
            distance_bin_prior_power=args.distance_bin_prior_power,
            momentum_bin_edges=args.momentum_bin_edges,
            max_target_sampling_attempts=args.max_target_sampling_attempts,
            max_abs_tau=args.max_abs_tau,
            max_abs_qvel=args.max_abs_qvel,
            max_abs_qacc=args.max_abs_qacc,
            min_suffix_xy_rmse=args.min_suffix_xy_rmse,
            min_final_xy_gap=args.min_final_xy_gap,
            boundary_margin_ratio=args.boundary_margin_ratio,
            boundary_tau_slope_cap=args.boundary_tau_slope_cap,
            seed_offset=10_000_000,
            split_name="val",
            source_budget=args.val_sources if args.val_sources > 0 else None,
        )

    report = {
        "dataset_dir": str(output_dir),
        "xml_path": args.xml_path,
        "train_generation": train_generation,
        "train_summary": summarize_h5(train_path),
    }
    if val_generation is not None:
        report["val_generation"] = val_generation
        report["val_summary"] = summarize_h5(val_path)

    report_path = output_dir / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
