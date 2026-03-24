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

from scipy.interpolate import CubicSpline
from scipy.stats import qmc
from tqdm import tqdm

from scripts.data.generate_bidirectional_reacher_dataset import (
    L1,
    L2,
    TRAJ_KEYS,
    get_model_ids,
    ik_2link,
    set_reacher_state,
    simulate_dataset_rollout,
    simulate_helper_segment,
)


def spline_series(control_points: np.ndarray, num_steps: int) -> np.ndarray:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if num_steps == 1:
        return np.repeat(control_points[:1], 1, axis=0).astype(np.float64, copy=True)

    num_ctrl, num_dims = control_points.shape
    if num_ctrl < 2:
        return np.repeat(control_points[:1], num_steps, axis=0).astype(np.float64, copy=True)

    knots = np.linspace(0.0, 1.0, num_ctrl, dtype=np.float64)
    eval_t = np.linspace(0.0, 1.0, num_steps, dtype=np.float64)
    out = np.empty((num_steps, num_dims), dtype=np.float64)
    bc_type = "natural" if num_ctrl >= 3 else "not-a-knot"
    for dim in range(num_dims):
        spline = CubicSpline(knots, control_points[:, dim], bc_type=bc_type)
        out[:, dim] = spline(eval_t)
    return out


def sample_uniform_controls(
    rng: np.random.Generator,
    num_control_points: int,
    num_dims: int,
    scale: float,
) -> np.ndarray:
    return rng.uniform(-scale, scale, size=(num_control_points, num_dims)).astype(np.float64)


def sobol_candidates(
    dim: int,
    count: int,
    seed: int,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, dim), dtype=np.float64)
    m = int(math.ceil(math.log2(count)))
    engine = qmc.Sobol(d=dim, scramble=True, seed=seed)
    return engine.random_base2(m=m)


def boundary_conditioned_suffix(
    boundary_tau: np.ndarray,
    boundary_tau_slope: np.ndarray,
    residual_controls: np.ndarray,
    suffix_steps: int,
    dt: float,
    tau_limit: float,
) -> tuple[bool, np.ndarray | None]:
    if suffix_steps <= 0:
        return True, np.empty((0, boundary_tau.shape[0]), dtype=np.float64)

    residual_curve = spline_series(residual_controls, suffix_steps)
    s = np.linspace(0.0, 1.0, suffix_steps, dtype=np.float64)[:, None]
    duration = max((suffix_steps - 1) * dt, dt)

    base = boundary_tau[None, :] + boundary_tau_slope[None, :] * (duration * s)
    if np.any(np.abs(base) > tau_limit):
        return False, None

    residual = (s**2) * residual_curve
    tau = base + residual
    if np.any(np.abs(tau) > tau_limit + 1e-9):
        return False, None
    return True, tau


def compute_source_mom(
    model: mujoco.MjModel,
    ids: dict[str, int],
    qpos: np.ndarray,
    qvel: np.ndarray,
    waypoint_xy: np.ndarray,
) -> np.ndarray:
    data = mujoco.MjData(model)
    set_reacher_state(model, data, qpos, qvel, waypoint_xy, ids)
    M = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, M, data.qM)
    mom = M @ data.qvel
    return mom[[ids["joint0_dof"], ids["joint1_dof"]]]


def branch_diversity_metrics(
    candidate_xy: np.ndarray,
    sibling_results: list[dict],
    branch_point: int,
) -> tuple[float, float]:
    if not sibling_results:
        return float("inf"), float("inf")

    candidate_suffix = candidate_xy[branch_point:].astype(np.float64)
    candidate_final = candidate_xy[-1].astype(np.float64)

    min_rmse = float("inf")
    min_final_gap = float("inf")
    for sibling in sibling_results:
        sibling_xy = sibling["seq_fingertip_xy"].astype(np.float64)
        sibling_suffix = sibling_xy[branch_point:]
        rmse = float(np.sqrt(np.mean(np.square(candidate_suffix - sibling_suffix), dtype=np.float64)))
        final_gap = float(np.linalg.norm(candidate_final - sibling_xy[-1].astype(np.float64)))
        min_rmse = min(min_rmse, rmse)
        min_final_gap = min(min_final_gap, final_gap)
    return min_rmse, min_final_gap


def choose_anchor_indices(
    trajectory_length: int,
    anchor_min: int,
    suffix_min_len: int,
    anchor_stride: int,
    anchors_per_root: int,
    rng: np.random.Generator,
) -> list[int]:
    max_anchor = trajectory_length - suffix_min_len
    if max_anchor < anchor_min:
        return []

    offset = int(rng.integers(0, max(1, anchor_stride)))
    start = anchor_min + offset
    anchors = list(range(start, max_anchor + 1, anchor_stride))
    if not anchors:
        anchors = list(range(anchor_min, max_anchor + 1, anchor_stride))
    if not anchors:
        return []
    if anchors_per_root > 0 and len(anchors) > anchors_per_root:
        selected = rng.choice(np.array(anchors, dtype=np.int32), size=anchors_per_root, replace=False)
        anchors = sorted(int(v) for v in selected.tolist())
    return anchors


def sample_workspace_uniform_source_state(
    rng: np.random.Generator,
    source_qvel_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_min = abs(L1 - L2)
    r_max = L1 + L2
    u = float(rng.random())
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    radius = np.sqrt(u * (r_max * r_max - r_min * r_min) + r_min * r_min)
    source_xy = np.array([radius * np.cos(theta), radius * np.sin(theta)], dtype=np.float64)
    elbow_branch = 1 if rng.random() < 0.5 else -1
    source_qpos = ik_2link(source_xy, elbow_branch=elbow_branch).astype(np.float64)
    source_qvel = rng.uniform(-source_qvel_scale, source_qvel_scale, size=2).astype(np.float64)
    return source_qpos, source_qvel, source_xy


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
    file.attrs["generator"] = "reacher_per_step_branching"
    file.attrs["generator_config"] = json.dumps(config, sort_keys=True)


def finalize_h5_header(file: h5py.File, num_trajectories: int) -> None:
    file.attrs["num_trajectories"] = int(num_trajectories)


def write_trajectory_group(file: h5py.File, traj_index: int, result: dict) -> None:
    group = file.create_group(f"traj_{traj_index}")
    for key in TRAJ_KEYS:
        group.create_dataset(key, data=result[key], dtype="f4")
    group.create_dataset("seq_fingertip_xy", data=result["seq_fingertip_xy"], dtype="f4")
    group.create_dataset("waypoint_xy", data=result["waypoint_xy"], dtype="f4")
    group.create_dataset("target_xy", data=result["waypoint_xy"], dtype="f4")
    group.create_dataset("source_xy", data=result["source_xy"], dtype="f4")
    group.create_dataset("source_qpos", data=result["source_qpos"], dtype="f4")
    group.create_dataset("source_qvel", data=result["source_qvel"], dtype="f4")
    group.create_dataset("source_mom", data=result["source_mom"], dtype="f4")
    group.create_dataset("anchor_state_qpos", data=result["anchor_state_qpos"], dtype="f4")
    group.create_dataset("anchor_state_qvel", data=result["anchor_state_qvel"], dtype="f4")
    group.create_dataset("anchor_state_mom", data=result["anchor_state_mom"], dtype="f4")
    group.create_dataset("boundary_tau", data=result["boundary_tau"], dtype="f4")
    group.create_dataset("boundary_tau_slope", data=result["boundary_tau_slope"], dtype="f4")
    group.attrs["waypoint_index"] = int(result["waypoint_index"])
    group.attrs["target_index"] = int(result["waypoint_index"])
    group.attrs["source_group_id"] = int(result["source_group_id"])
    group.attrs["root_group_id"] = int(result["root_group_id"])
    group.attrs["root_index_within_source"] = int(result["root_index_within_source"])
    group.attrs["anchor_index"] = int(result["anchor_index"])
    group.attrs["anchor_branch_group_id"] = int(result["anchor_branch_group_id"])
    group.attrs["branch_index"] = int(result["branch_index"])
    group.attrs["nearest_sibling_xy_rmse"] = float(result["nearest_sibling_xy_rmse"])
    group.attrs["nearest_sibling_final_xy_gap"] = float(result["nearest_sibling_final_xy_gap"])


def summarize_h5(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        num_trajectories = int(f.attrs["num_trajectories"])
        num_steps = int(f.attrs["num_steps"])
        generator_config = json.loads(f.attrs["generator_config"])
        source_group_ids: list[int] = []
        root_group_ids: list[int] = []
        root_indices_within_source: list[int] = []
        anchor_group_ids: list[int] = []
        anchor_indices: list[int] = []
        waypoint_indices: list[int] = []
        tau_absmax: list[float] = []
        qvel_absmax: list[float] = []
        qacc_absmax: list[float] = []
        source_xy_values: list[np.ndarray] = []
        by_anchor_group: dict[int, list[int]] = {}
        roots_by_source: dict[int, set[int]] = {}

        for idx in range(num_trajectories):
            group = f[f"traj_{idx}"]
            source_group_id = int(group.attrs.get("source_group_id", -1))
            root_group_id = int(group.attrs["root_group_id"])
            root_index_within_source = int(group.attrs.get("root_index_within_source", -1))
            anchor_group_id = int(group.attrs["anchor_branch_group_id"])
            anchor_index = int(group.attrs["anchor_index"])
            source_group_ids.append(source_group_id)
            root_group_ids.append(root_group_id)
            root_indices_within_source.append(root_index_within_source)
            anchor_group_ids.append(anchor_group_id)
            anchor_indices.append(anchor_index)
            waypoint_indices.append(int(group.attrs["waypoint_index"]))
            tau_absmax.append(float(np.max(np.abs(group["seq_torque"][:] ))))
            qvel_absmax.append(float(np.max(np.abs(group["seq_qvel"][:] ))))
            qacc_absmax.append(float(np.max(np.abs(group["seq_qacc"][:] ))))
            source_xy_values.append(group["source_xy"][:].astype(np.float64))
            roots_by_source.setdefault(source_group_id, set()).add(root_group_id)
            by_anchor_group.setdefault(anchor_group_id, []).append(idx)

        pairwise_rmse: list[float] = []
        pairwise_final_gap: list[float] = []
        for _, indices in by_anchor_group.items():
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

    anchor_arr = np.array(anchor_indices, dtype=np.float64)
    waypoint_arr = np.array(waypoint_indices, dtype=np.float64)
    tau_arr = np.array(tau_absmax, dtype=np.float64)
    qvel_arr = np.array(qvel_absmax, dtype=np.float64)
    qacc_arr = np.array(qacc_absmax, dtype=np.float64)
    source_xy_arr = np.stack(source_xy_values, axis=0) if source_xy_values else np.empty((0, 2), dtype=np.float64)

    summary = {
        "file": str(h5_path),
        "num_trajectories": num_trajectories,
        "num_steps": num_steps,
        "generator_config": generator_config,
        "source_groups": int(len(set(source_group_ids))),
        "root_groups": int(len(set(root_group_ids))),
        "anchor_branch_groups": int(len(set(anchor_group_ids))),
        "anchor_index": {
            "min": int(anchor_arr.min()),
            "mean": float(anchor_arr.mean()),
            "max": int(anchor_arr.max()),
        },
        "waypoint_index": {
            "min": int(waypoint_arr.min()),
            "mean": float(waypoint_arr.mean()),
            "max": int(waypoint_arr.max()),
        },
        "max_abs_tau": {
            "mean": float(tau_arr.mean()),
            "p95": float(np.percentile(tau_arr, 95)),
            "max": float(tau_arr.max()),
        },
        "max_abs_qvel": {
            "mean": float(qvel_arr.mean()),
            "p95": float(np.percentile(qvel_arr, 95)),
            "max": float(qvel_arr.max()),
        },
        "max_abs_qacc": {
            "mean": float(qacc_arr.mean()),
            "p95": float(np.percentile(qacc_arr, 95)),
            "max": float(qacc_arr.max()),
        },
        "accepted_root_groups_per_source": {str(k): int(len(v)) for k, v in sorted(roots_by_source.items())},
    }
    if source_xy_values:
        radii = np.linalg.norm(source_xy_arr, axis=1)
        summary["source_xy"] = {
            "x_min": float(source_xy_arr[:, 0].min()),
            "x_max": float(source_xy_arr[:, 0].max()),
            "y_min": float(source_xy_arr[:, 1].min()),
            "y_max": float(source_xy_arr[:, 1].max()),
            "radius_min": float(radii.min()),
            "radius_mean": float(radii.mean()),
            "radius_max": float(radii.max()),
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
        summary["pairwise_suffix_xy_rmse"] = {
            "mean": float(rmse_arr.mean()),
            "p05": float(np.percentile(rmse_arr, 5)),
            "min": float(rmse_arr.min()),
        }
        summary["pairwise_final_xy_gap"] = {
            "mean": float(final_gap_arr.mean()),
            "p05": float(np.percentile(final_gap_arr, 5)),
            "min": float(final_gap_arr.min()),
        }
    return summary


def build_root_rollout(
    model: mujoco.MjModel,
    ids: dict[str, int],
    rng: np.random.Generator,
    trajectory_length: int,
    dt: float,
    source_qvel_scale: float,
    root_control_points: int,
    root_tau_scale: float,
    max_abs_tau: float,
    max_abs_qvel: float,
    max_abs_qacc: float,
    source_qpos: np.ndarray | None = None,
    source_qvel: np.ndarray | None = None,
    source_xy: np.ndarray | None = None,
) -> tuple[bool, dict]:
    dummy_waypoint = np.zeros(2, dtype=np.float64)

    if source_qpos is None or source_qvel is None or source_xy is None:
        source_qpos, source_qvel, source_xy = sample_workspace_uniform_source_state(
            rng=rng,
            source_qvel_scale=source_qvel_scale,
        )
    else:
        source_qpos = np.asarray(source_qpos, dtype=np.float64)
        source_qvel = np.asarray(source_qvel, dtype=np.float64)
        source_xy = np.asarray(source_xy, dtype=np.float64)

    root_controls = sample_uniform_controls(
        rng=rng,
        num_control_points=root_control_points,
        num_dims=model.nu,
        scale=root_tau_scale,
    )
    root_tau = spline_series(root_controls, trajectory_length)
    if np.any(np.abs(root_tau) > max_abs_tau):
        return False, {"root_reason": "root_tau_limit"}

    try:
        root_rollout = simulate_dataset_rollout(
            model=model,
            start_arm_qpos=source_qpos,
            start_arm_qvel=source_qvel,
            waypoint_xy=dummy_waypoint,
            torque_seq=root_tau,
            ids=ids,
        )
        root_qpos_hist, root_qvel_hist = simulate_helper_segment(
            model=model,
            arm_qpos=source_qpos,
            arm_qvel=source_qvel,
            waypoint_xy=dummy_waypoint,
            torque_seq=root_tau,
            ids=ids,
        )
    except FloatingPointError:
        return False, {"root_reason": "root_non_finite"}

    if np.any(np.abs(root_rollout["seq_qvel"]) > max_abs_qvel):
        return False, {"root_reason": "root_qvel_limit"}
    if np.any(np.abs(root_rollout["seq_qacc"]) > max_abs_qacc):
        return False, {"root_reason": "root_qacc_limit"}

    source_mom = compute_source_mom(model, ids, source_qpos, source_qvel, dummy_waypoint)
    return True, {
        "source_xy": source_xy,
        "source_qpos": source_qpos,
        "source_qvel": source_qvel,
        "source_mom": source_mom,
        "root_tau": root_tau,
        "root_rollout": root_rollout,
        "root_qpos_hist": root_qpos_hist,
        "root_qvel_hist": root_qvel_hist,
    }


def build_anchor_branch_group(
    model: mujoco.MjModel,
    ids: dict[str, int],
    root_info: dict,
    trajectory_length: int,
    dt: float,
    anchor_index: int,
    branches_per_anchor: int,
    suffix_control_points: int,
    suffix_residual_scale: float,
    waypoint_offset_min: int,
    max_branch_attempts: int,
    max_abs_tau: float,
    max_abs_qvel: float,
    max_abs_qacc: float,
    min_suffix_xy_rmse: float,
    min_final_xy_gap: float,
    boundary_margin_ratio: float,
    boundary_tau_slope_cap: float,
    source_group_id: int,
    root_group_id: int,
    root_index_within_source: int,
    anchor_branch_group_id: int,
    sobol_seed: int,
    rng: np.random.Generator,
) -> tuple[bool, dict]:
    dummy_waypoint = np.zeros(2, dtype=np.float64)
    suffix_steps = int(trajectory_length - anchor_index)
    if suffix_steps <= 0:
        return False, {"anchor_reason": "anchor_invalid", "branch_attempts": 0, "branch_reject_counts": {}}

    root_tau = root_info["root_tau"]
    source_xy = root_info["source_xy"]
    source_qpos = root_info["source_qpos"]
    source_qvel = root_info["source_qvel"]
    source_mom = root_info["source_mom"]
    root_qpos_hist = root_info["root_qpos_hist"]
    root_qvel_hist = root_info["root_qvel_hist"]

    prefix_tau = root_tau[:anchor_index].astype(np.float64)
    boundary_tau = prefix_tau[-1].astype(np.float64)
    boundary_tau_slope = np.zeros(model.nu, dtype=np.float64)
    if anchor_index >= 2:
        raw_slope = (prefix_tau[-1] - prefix_tau[-2]) / dt
        boundary_tau_slope = np.clip(raw_slope, -boundary_tau_slope_cap, boundary_tau_slope_cap)

    if np.max(np.abs(boundary_tau)) > max_abs_tau * boundary_margin_ratio:
        return False, {"anchor_reason": "anchor_boundary_margin", "branch_attempts": 0, "branch_reject_counts": {}}

    branch_state_qpos = root_qpos_hist[anchor_index - 1, [ids["joint0_qpos"], ids["joint1_qpos"]]].astype(np.float64)
    branch_state_qvel = root_qvel_hist[anchor_index - 1, [ids["joint0_dof"], ids["joint1_dof"]]].astype(np.float64)
    branch_state_mom = compute_source_mom(model, ids, branch_state_qpos, branch_state_qvel, dummy_waypoint)

    branch_reject_counts: dict[str, int] = {}
    branch_attempts = 0
    accepted_results: list[dict] = []

    candidate_vectors = sobol_candidates(dim=suffix_control_points * model.nu, count=max_branch_attempts, seed=sobol_seed)
    candidate_vectors = candidate_vectors[:max_branch_attempts]
    for candidate in candidate_vectors:
        if len(accepted_results) >= branches_per_anchor:
            break
        branch_attempts += 1

        residual_controls = (2.0 * candidate.reshape(suffix_control_points, model.nu) - 1.0) * suffix_residual_scale
        ok, suffix_tau = boundary_conditioned_suffix(
            boundary_tau=boundary_tau,
            boundary_tau_slope=boundary_tau_slope,
            residual_controls=residual_controls,
            suffix_steps=suffix_steps,
            dt=dt,
            tau_limit=max_abs_tau,
        )
        if not ok or suffix_tau is None:
            branch_reject_counts["suffix_tau_limit"] = branch_reject_counts.get("suffix_tau_limit", 0) + 1
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
            branch_reject_counts["replay_non_finite"] = branch_reject_counts.get("replay_non_finite", 0) + 1
            continue

        if np.any(np.abs(replay["seq_qvel"]) > max_abs_qvel):
            branch_reject_counts["qvel_limit"] = branch_reject_counts.get("qvel_limit", 0) + 1
            continue
        if np.any(np.abs(replay["seq_qacc"]) > max_abs_qacc):
            branch_reject_counts["qacc_limit"] = branch_reject_counts.get("qacc_limit", 0) + 1
            continue

        waypoint_index_min = min(anchor_index + waypoint_offset_min, trajectory_length - 1)
        waypoint_index = int(rng.integers(waypoint_index_min, trajectory_length))
        waypoint_xy = replay["seq_fingertip_xy"][waypoint_index].astype(np.float64)

        nearest_rmse, nearest_final_gap = branch_diversity_metrics(
            candidate_xy=replay["seq_fingertip_xy"],
            sibling_results=accepted_results,
            branch_point=anchor_index,
        )
        if accepted_results and nearest_rmse < min_suffix_xy_rmse and nearest_final_gap < min_final_xy_gap:
            branch_reject_counts["branch_too_similar"] = branch_reject_counts.get("branch_too_similar", 0) + 1
            continue

        replay["waypoint_xy"] = waypoint_xy.astype(np.float32)
        replay["source_xy"] = source_xy.astype(np.float32)
        replay["source_qpos"] = source_qpos.astype(np.float32)
        replay["source_qvel"] = source_qvel.astype(np.float32)
        replay["source_mom"] = source_mom.astype(np.float32)
        replay["anchor_state_qpos"] = branch_state_qpos.astype(np.float32)
        replay["anchor_state_qvel"] = branch_state_qvel.astype(np.float32)
        replay["anchor_state_mom"] = branch_state_mom.astype(np.float32)
        replay["boundary_tau"] = boundary_tau.astype(np.float32)
        replay["boundary_tau_slope"] = boundary_tau_slope.astype(np.float32)
        replay["waypoint_index"] = np.int32(waypoint_index)
        replay["source_group_id"] = np.int32(source_group_id)
        replay["root_group_id"] = np.int32(root_group_id)
        replay["root_index_within_source"] = np.int32(root_index_within_source)
        replay["anchor_index"] = np.int32(anchor_index)
        replay["anchor_branch_group_id"] = np.int32(anchor_branch_group_id)
        replay["branch_index"] = np.int32(len(accepted_results))
        replay["nearest_sibling_xy_rmse"] = np.float32(nearest_rmse if np.isfinite(nearest_rmse) else np.nan)
        replay["nearest_sibling_final_xy_gap"] = np.float32(
            nearest_final_gap if np.isfinite(nearest_final_gap) else np.nan
        )
        accepted_results.append(replay)

    if len(accepted_results) < branches_per_anchor:
        anchor_reason = "anchor_underfilled"
        if branch_attempts >= max_branch_attempts:
            anchor_reason = "anchor_branch_attempt_limit"
        return False, {
            "anchor_reason": anchor_reason,
            "branch_attempts": branch_attempts,
            "branch_reject_counts": branch_reject_counts,
        }

    return True, {
        "results": accepted_results,
        "branch_attempts": branch_attempts,
        "branch_reject_counts": branch_reject_counts,
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
    branches_per_anchor: int,
    suffix_control_points: int,
    suffix_residual_scale: float,
    waypoint_offset_min: int,
    max_branch_attempts: int,
    max_abs_tau: float,
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
    if num_trajectories is not None and num_trajectories % branches_per_anchor != 0:
        raise ValueError(
            f"num_trajectories={num_trajectories} must be divisible by branches_per_anchor={branches_per_anchor}"
        )

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
        "branches_per_anchor": int(branches_per_anchor),
        "suffix_control_points": int(suffix_control_points),
        "suffix_residual_scale": float(suffix_residual_scale),
        "waypoint_offset_min": int(waypoint_offset_min),
        "max_branch_attempts": int(max_branch_attempts),
        "max_abs_tau": float(max_abs_tau),
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

    accepted = 0
    source_attempts = 0
    source_successes = 0
    root_attempts = 0
    root_successes = 0
    anchor_group_attempts = 0
    anchor_group_successes = 0
    branch_attempts = 0
    source_group_id = 0
    root_group_id = 0
    anchor_branch_group_id = 0
    root_reject_counts: dict[str, int] = {}
    anchor_group_reject_counts: dict[str, int] = {}
    branch_reject_counts: dict[str, int] = {}
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
                    max_abs_tau=max_abs_tau,
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
                    ok, info = build_anchor_branch_group(
                        model=model,
                        ids=ids,
                        root_info=root_info,
                        trajectory_length=trajectory_length,
                        dt=dt,
                        anchor_index=anchor_index,
                        branches_per_anchor=branches_per_anchor,
                        suffix_control_points=suffix_control_points,
                        suffix_residual_scale=suffix_residual_scale,
                        waypoint_offset_min=waypoint_offset_min,
                        max_branch_attempts=max_branch_attempts,
                        max_abs_tau=max_abs_tau,
                        max_abs_qvel=max_abs_qvel,
                        max_abs_qacc=max_abs_qacc,
                        min_suffix_xy_rmse=min_suffix_xy_rmse,
                        min_final_xy_gap=min_final_xy_gap,
                        boundary_margin_ratio=boundary_margin_ratio,
                        boundary_tau_slope_cap=boundary_tau_slope_cap,
                        source_group_id=current_source_group_id,
                        root_group_id=current_root_group_id,
                        root_index_within_source=root_index_within_source,
                        anchor_branch_group_id=anchor_branch_group_id,
                        sobol_seed=seed_offset + 104729 * (anchor_group_attempts + 1),
                        rng=rng,
                    )
                    branch_attempts += int(info.get("branch_attempts", 0))
                    for reason, count in info.get("branch_reject_counts", {}).items():
                        branch_reject_counts[reason] = branch_reject_counts.get(reason, 0) + int(count)

                    if ok:
                        for result in info["results"]:
                            write_trajectory_group(h5_file, accepted, result)
                            accepted += 1
                            pbar.update(1)
                        anchor_group_successes += 1
                        anchor_branch_group_id += 1
                        root_had_success = True
                    else:
                        reason = str(info.get("anchor_reason", "anchor_reject"))
                        anchor_group_reject_counts[reason] = anchor_group_reject_counts.get(reason, 0) + 1

                    if num_trajectories is not None and accepted >= num_trajectories:
                        break

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

            pbar.set_postfix(
                sources=source_successes,
                roots=root_successes,
                anchors=anchor_group_successes,
                branch_attempts=branch_attempts,
                accept_rate=f"{accepted / max(1, branch_attempts):.1%}",
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
        "branch_attempts": int(branch_attempts),
        "accept_rate": float(accepted / max(1, branch_attempts)),
        "source_success_rate": float(source_successes / max(1, source_attempts)),
        "root_success_rate": float(root_successes / max(1, root_attempts)),
        "anchor_group_success_rate": float(anchor_group_successes / max(1, anchor_group_attempts)),
        "root_reject_counts": root_reject_counts,
        "anchor_group_reject_counts": anchor_group_reject_counts,
        "branch_reject_counts": branch_reject_counts,
        "source_groups": int(source_successes),
        "root_groups": int(root_group_id),
        "anchor_branch_groups": int(anchor_branch_group_id),
        "generator_config": config,
    }
    print(
        f"[{split_name}] saved {accepted} trajectories to {output_path} "
        f"(source attempts {source_attempts}, root attempts {root_attempts}, anchor groups {anchor_group_attempts}, "
        f"branch attempts {branch_attempts}, branch accept rate {accepted / max(1, branch_attempts):.1%})",
        flush=True,
    )
    for reason, count in sorted(root_reject_counts.items()):
        print(f"  - root {reason}: {count}", flush=True)
    for reason, count in sorted(anchor_group_reject_counts.items()):
        print(f"  - anchor {reason}: {count}", flush=True)
    for reason, count in sorted(branch_reject_counts.items()):
        print(f"  - branch {reason}: {count}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a pilot per-step branching Reacher dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/data/reacher_per_step_branching_dt0p001_len500_pilot",
    )
    parser.add_argument(
        "--xml_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/configs/reacher_non_diss_unbounded_j1.xml",
    )
    parser.add_argument("--train_trajectories", type=int, default=32)
    parser.add_argument("--val_trajectories", type=int, default=8)
    parser.add_argument("--train_sources", type=int, default=0)
    parser.add_argument("--val_sources", type=int, default=0)
    parser.add_argument("--trajectory_length", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--source_qvel_scale", type=float, default=0.1)
    parser.add_argument("--root_control_points", type=int, default=12)
    parser.add_argument("--root_tau_scale", type=float, default=0.06)
    parser.add_argument("--anchor_min", type=int, default=8)
    parser.add_argument("--suffix_min_len", type=int, default=8)
    parser.add_argument("--anchor_stride", type=int, default=8)
    parser.add_argument("--anchors_per_root", type=int, default=4)
    parser.add_argument("--roots_per_source", type=int, default=4)
    parser.add_argument("--branches_per_anchor", type=int, default=2)
    parser.add_argument("--suffix_control_points", type=int, default=16)
    parser.add_argument("--suffix_residual_scale", type=float, default=0.08)
    parser.add_argument("--waypoint_offset_min", type=int, default=16)
    parser.add_argument("--max_branch_attempts", type=int, default=256)
    parser.add_argument("--max_abs_tau", type=float, default=0.8)
    parser.add_argument("--max_abs_qvel", type=float, default=10.0)
    parser.add_argument("--max_abs_qacc", type=float, default=100.0)
    parser.add_argument("--min_suffix_xy_rmse", type=float, default=0.001)
    parser.add_argument("--min_final_xy_gap", type=float, default=0.001)
    parser.add_argument("--boundary_margin_ratio", type=float, default=0.95)
    parser.add_argument("--boundary_tau_slope_cap", type=float, default=0.75)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / (
        f"traj_from_{args.train_sources}_sources_steps_{args.trajectory_length}.h5"
        if args.train_sources > 0
        else f"traj_{args.train_trajectories}-steps_{args.trajectory_length}.h5"
    )
    val_path = output_dir / (
        f"traj_from_{args.val_sources}_sources_steps_{args.trajectory_length}.h5"
        if args.val_sources > 0
        else f"traj_{args.val_trajectories}-steps_{args.trajectory_length}.h5"
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
        branches_per_anchor=args.branches_per_anchor,
        suffix_control_points=args.suffix_control_points,
        suffix_residual_scale=args.suffix_residual_scale,
        waypoint_offset_min=args.waypoint_offset_min,
        max_branch_attempts=args.max_branch_attempts,
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
            branches_per_anchor=args.branches_per_anchor,
            suffix_control_points=args.suffix_control_points,
            suffix_residual_scale=args.suffix_residual_scale,
            waypoint_offset_min=args.waypoint_offset_min,
            max_branch_attempts=args.max_branch_attempts,
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

    report_path = output_dir / "pilot_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
