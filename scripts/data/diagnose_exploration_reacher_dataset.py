#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


L1 = 0.10
L2 = 0.11
REACH_MAX = L1 + L2


def summarize_array(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def threshold_label(threshold_m: float) -> str:
    cm = threshold_m * 100.0
    if abs(cm - round(cm)) < 1e-9:
        return f"progress_at_least_{int(round(cm))}cm"
    text = f"{cm:.1f}".rstrip("0").rstrip(".")
    return f"progress_at_least_{text.replace('.', 'p')}cm"


def state_features(qpos: np.ndarray, mom: np.ndarray) -> np.ndarray:
    return np.concatenate([np.sin(qpos), np.cos(qpos), mom], axis=1).astype(np.float64)


def sample_workspace_uniform_targets(count: int, rng: np.random.Generator, max_radius: float) -> np.ndarray:
    radius = max_radius * np.sqrt(rng.random(count))
    theta = rng.uniform(-math.pi, math.pi, size=count)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1).astype(np.float64)


def circular_spread(angles: np.ndarray) -> float:
    if angles.size == 0:
        return 0.0
    cos_mean = float(np.mean(np.cos(angles)))
    sin_mean = float(np.mean(np.sin(angles)))
    resultant = math.sqrt(cos_mean * cos_mean + sin_mean * sin_mean)
    return float(1.0 - resultant)


def compute_workspace_coverage(points_xy: np.ndarray, max_radius: float, bins: int) -> dict:
    hist, _, _ = np.histogram2d(
        points_xy[:, 0],
        points_xy[:, 1],
        bins=bins,
        range=[[-max_radius, max_radius], [-max_radius, max_radius]],
    )
    x_centers = np.linspace(-max_radius, max_radius, bins, endpoint=False) + (max_radius * 2.0 / bins) * 0.5
    y_centers = np.linspace(-max_radius, max_radius, bins, endpoint=False) + (max_radius * 2.0 / bins) * 0.5
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
    valid_mask = (xx * xx + yy * yy) <= max_radius * max_radius
    occupied = (hist > 0) & valid_mask
    return {
        "bins": int(bins),
        "valid_bins": int(np.sum(valid_mask)),
        "occupied_bins": int(np.sum(occupied)),
        "occupancy_ratio": float(np.sum(occupied) / max(1, np.sum(valid_mask))),
        "point_count": int(points_xy.shape[0]),
    }


def load_exploration_dataset(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        num_trajectories = int(f.attrs["num_trajectories"])
        if num_trajectories <= 0:
            raise ValueError(f"{h5_path} contains no trajectories")

        generator_config = json.loads(f.attrs["generator_config"])
        num_steps = int(f.attrs["num_steps"])

        seq_qpos = np.empty((num_trajectories, num_steps, 2), dtype=np.float32)
        seq_qvel = np.empty((num_trajectories, num_steps, 2), dtype=np.float32)
        seq_qacc = np.empty((num_trajectories, num_steps, 2), dtype=np.float32)
        seq_mom = np.empty((num_trajectories, num_steps, 2), dtype=np.float32)
        seq_torque = np.empty((num_trajectories, num_steps, 2), dtype=np.float32)
        seq_fingertip_xy = np.empty((num_trajectories, num_steps, 2), dtype=np.float32)
        start_qpos = np.empty((num_trajectories, 2), dtype=np.float32)
        start_qvel = np.empty((num_trajectories, 2), dtype=np.float32)
        source_xy = np.empty((num_trajectories, 2), dtype=np.float32)
        elbow_branch = np.empty((num_trajectories,), dtype=np.int32)

        for idx in tqdm(range(num_trajectories), desc=f"Loading {h5_path.name}"):
            group = f[f"traj_{idx}"]
            seq_qpos[idx] = group["seq_qpos"][:]
            seq_qvel[idx] = group["seq_qvel"][:]
            seq_qacc[idx] = group["seq_qacc"][:]
            seq_mom[idx] = group["seq_mom"][:]
            seq_torque[idx] = group["seq_torque"][:]
            seq_fingertip_xy[idx] = group["seq_fingertip_xy"][:]
            start_qpos[idx] = group["start_qpos"][:]
            start_qvel[idx] = group["start_qvel"][:]
            source_xy[idx] = group["source_xy"][:]
            elbow_branch[idx] = int(group.attrs["elbow_branch"])

    return {
        "path": str(h5_path),
        "num_trajectories": int(num_trajectories),
        "num_steps": int(num_steps),
        "generator_config": generator_config,
        "seq_qpos": seq_qpos.astype(np.float64),
        "seq_qvel": seq_qvel.astype(np.float64),
        "seq_qacc": seq_qacc.astype(np.float64),
        "seq_mom": seq_mom.astype(np.float64),
        "seq_torque": seq_torque.astype(np.float64),
        "seq_fingertip_xy": seq_fingertip_xy.astype(np.float64),
        "start_qpos": start_qpos.astype(np.float64),
        "start_qvel": start_qvel.astype(np.float64),
        "source_xy": source_xy.astype(np.float64),
        "elbow_branch": elbow_branch,
    }


def dataset_quality_summary(dataset: dict, workspace_bins: int) -> dict:
    seq_qpos = dataset["seq_qpos"]
    seq_qvel = dataset["seq_qvel"]
    seq_qacc = dataset["seq_qacc"]
    seq_mom = dataset["seq_mom"]
    seq_torque = dataset["seq_torque"]
    seq_xy = dataset["seq_fingertip_xy"]
    max_radius = float(dataset["generator_config"].get("source_radius", REACH_MAX))

    return {
        "path": dataset["path"],
        "num_trajectories": int(dataset["num_trajectories"]),
        "num_steps": int(dataset["num_steps"]),
        "generator_config": dataset["generator_config"],
        "workspace_coverage": compute_workspace_coverage(seq_xy.reshape(-1, 2), max_radius=max_radius, bins=workspace_bins),
        "source_radius": summarize_array(np.linalg.norm(dataset["source_xy"], axis=1)),
        "start_qpos": summarize_array(dataset["start_qpos"].reshape(-1)),
        "start_qvel": summarize_array(dataset["start_qvel"].reshape(-1)),
        "seq_qpos": summarize_array(seq_qpos.reshape(-1)),
        "seq_qvel": summarize_array(seq_qvel.reshape(-1)),
        "seq_qacc": summarize_array(seq_qacc.reshape(-1)),
        "seq_mom": summarize_array(seq_mom.reshape(-1)),
        "seq_torque": summarize_array(np.abs(seq_torque.reshape(-1))),
    }


def build_anchor_table(dataset: dict, horizon: int, stride: int, anchor_min: int) -> dict:
    seq_qpos = dataset["seq_qpos"]
    seq_mom = dataset["seq_mom"]
    seq_torque = dataset["seq_torque"]
    seq_xy = dataset["seq_fingertip_xy"]
    num_trajectories, num_steps, _ = seq_qpos.shape

    records_qpos = []
    records_mom = []
    records_anchor_xy = []
    records_first_tau = []
    records_end_xy = []
    records_best_xy = []
    records_traj_idx = []
    records_anchor_idx = []

    for traj_idx in range(num_trajectories):
        last_anchor = num_steps - horizon - 1
        if last_anchor < anchor_min:
            continue
        for anchor_idx in range(anchor_min, last_anchor + 1, stride):
            suffix_xy = seq_xy[traj_idx, anchor_idx + 1 : anchor_idx + horizon + 1]
            records_qpos.append(seq_qpos[traj_idx, anchor_idx])
            records_mom.append(seq_mom[traj_idx, anchor_idx])
            records_anchor_xy.append(seq_xy[traj_idx, anchor_idx])
            records_first_tau.append(seq_torque[traj_idx, anchor_idx])
            records_end_xy.append(suffix_xy[-1])
            records_best_xy.append(suffix_xy)
            records_traj_idx.append(traj_idx)
            records_anchor_idx.append(anchor_idx)

    if not records_qpos:
        raise ValueError("no valid anchors found; reduce horizon or anchor_min")

    anchor_qpos = np.asarray(records_qpos, dtype=np.float64)
    anchor_mom = np.asarray(records_mom, dtype=np.float64)
    anchor_xy = np.asarray(records_anchor_xy, dtype=np.float64)
    first_tau = np.asarray(records_first_tau, dtype=np.float64)
    end_xy = np.asarray(records_end_xy, dtype=np.float64)
    return {
        "anchor_qpos": anchor_qpos,
        "anchor_mom": anchor_mom,
        "anchor_xy": anchor_xy,
        "first_tau": first_tau,
        "end_xy": end_xy,
        "suffix_xy": records_best_xy,
        "traj_idx": np.asarray(records_traj_idx, dtype=np.int32),
        "anchor_idx": np.asarray(records_anchor_idx, dtype=np.int32),
        "features": state_features(anchor_qpos, anchor_mom),
    }


def knn_indices(train_features: np.ndarray, query_features: np.ndarray, neighbors: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    sigma = train_features.std(axis=0) + 1e-6
    train_std = train_features / sigma[None, :]
    train_sq = np.sum(np.square(train_std, dtype=np.float64), axis=1)
    query_std = query_features / sigma[None, :]

    k = max(1, min(neighbors, train_std.shape[0]))
    all_indices = np.empty((query_std.shape[0], k), dtype=np.int64)
    all_dists = np.empty((query_std.shape[0], k), dtype=np.float64)

    for start in range(0, query_std.shape[0], batch_size):
        stop = min(start + batch_size, query_std.shape[0])
        batch = query_std[start:stop]
        batch_sq = np.sum(np.square(batch, dtype=np.float64), axis=1)
        dist2 = batch_sq[:, None] + train_sq[None, :] - 2.0 * (batch @ train_std.T)
        dist2 = np.maximum(dist2, 0.0)
        idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        for row in range(stop - start):
            row_idx = idx[row]
            row_dist2 = dist2[row, row_idx]
            order = np.argsort(row_dist2)
            all_indices[start + row] = row_idx[order]
            all_dists[start + row] = np.sqrt(row_dist2[order])
    return all_indices, all_dists


def evaluate_anchor_diversity(train_anchors: dict, query_anchors: dict, neighbors: int, batch_size: int) -> dict:
    idxs, dists = knn_indices(
        train_features=train_anchors["features"],
        query_features=query_anchors["features"],
        neighbors=neighbors,
        batch_size=batch_size,
    )

    first_tau_spread = np.empty(query_anchors["features"].shape[0], dtype=np.float64)
    endpoint_spread = np.empty_like(first_tau_spread)
    direction_spread = np.empty_like(first_tau_spread)

    for row in range(query_anchors["features"].shape[0]):
        neigh = idxs[row]
        tau_neighbors = train_anchors["first_tau"][neigh]
        end_xy_neighbors = train_anchors["end_xy"][neigh]
        anchor_xy = query_anchors["anchor_xy"][row]
        displacement = end_xy_neighbors - anchor_xy[None, :]

        first_tau_spread[row] = float(np.mean(np.std(tau_neighbors, axis=0)))
        endpoint_spread[row] = float(np.mean(np.linalg.norm(end_xy_neighbors - np.mean(end_xy_neighbors, axis=0, keepdims=True), axis=1)))

        disp_norm = np.linalg.norm(displacement, axis=1)
        valid = disp_norm > 1e-6
        if np.any(valid):
            direction_spread[row] = circular_spread(np.arctan2(displacement[valid, 1], displacement[valid, 0]))
        else:
            direction_spread[row] = 0.0

    return {
        "num_queries": int(query_anchors["features"].shape[0]),
        "neighbors": int(idxs.shape[1]),
        "nearest_state_distance": summarize_array(dists[:, 0]),
        "first_tau_spread": summarize_array(first_tau_spread),
        "endpoint_spread": summarize_array(endpoint_spread),
        "direction_spread": summarize_array(direction_spread),
    }


def evaluate_random_target_progress(
    train_anchors: dict,
    query_anchors: dict,
    neighbors: int,
    batch_size: int,
    random_seed: int,
    target_radius: float,
    progress_thresholds: list[float],
) -> dict:
    idxs, dists = knn_indices(
        train_features=train_anchors["features"],
        query_features=query_anchors["features"],
        neighbors=neighbors,
        batch_size=batch_size,
    )
    rng = np.random.default_rng(random_seed)
    targets = sample_workspace_uniform_targets(query_anchors["features"].shape[0], rng=rng, max_radius=target_radius)

    initial_dist = np.linalg.norm(targets - query_anchors["anchor_xy"], axis=1)
    best_min_dist = np.empty_like(initial_dist)
    progress = np.empty_like(initial_dist)

    for row in range(query_anchors["features"].shape[0]):
        target_xy = targets[row]
        best = float("inf")
        for train_idx in idxs[row].tolist():
            suffix_xy = train_anchors["suffix_xy"][train_idx]
            candidate = float(np.min(np.linalg.norm(suffix_xy - target_xy[None, :], axis=1)))
            if candidate < best:
                best = candidate
        best_min_dist[row] = best
        progress[row] = initial_dist[row] - best

    threshold_report = {
        threshold_label(threshold): float(np.mean(progress >= threshold))
        for threshold in progress_thresholds
    }
    threshold_report["any_positive_progress"] = float(np.mean(progress > 0.0))

    return {
        "num_queries": int(query_anchors["features"].shape[0]),
        "neighbors": int(idxs.shape[1]),
        "nearest_state_distance": summarize_array(dists[:, 0]),
        "initial_target_distance": summarize_array(initial_dist),
        "best_min_target_distance": summarize_array(best_min_dist),
        "progress_toward_target": summarize_array(progress),
        "progress_rates": threshold_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose an exploratory Reacher dataset for unconditional DPF MVPs")
    parser.add_argument("--train_h5", type=str, required=True)
    parser.add_argument("--val_h5", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--anchor_horizon", type=int, default=32)
    parser.add_argument("--anchor_stride", type=int, default=16)
    parser.add_argument("--anchor_min", type=int, default=8)
    parser.add_argument("--num_anchor_queries", type=int, default=1000)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workspace_bins", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--progress_thresholds", type=float, nargs="*", default=[0.01, 0.02, 0.05])
    args = parser.parse_args()

    train_dataset = load_exploration_dataset(Path(args.train_h5))
    report = {
        "train_dataset": dataset_quality_summary(train_dataset, workspace_bins=args.workspace_bins),
    }

    train_anchors = build_anchor_table(
        dataset=train_dataset,
        horizon=args.anchor_horizon,
        stride=args.anchor_stride,
        anchor_min=args.anchor_min,
    )
    report["train_anchor_pool"] = {
        "count": int(train_anchors["features"].shape[0]),
        "anchor_horizon": int(args.anchor_horizon),
        "anchor_stride": int(args.anchor_stride),
        "anchor_min": int(args.anchor_min),
    }

    query_dataset = train_dataset
    query_label = "train"
    if args.val_h5:
        query_dataset = load_exploration_dataset(Path(args.val_h5))
        report["val_dataset"] = dataset_quality_summary(query_dataset, workspace_bins=args.workspace_bins)
        query_label = "val"

    query_anchors_full = build_anchor_table(
        dataset=query_dataset,
        horizon=args.anchor_horizon,
        stride=args.anchor_stride,
        anchor_min=args.anchor_min,
    )

    rng = np.random.default_rng(args.random_seed)
    total_queries = query_anchors_full["features"].shape[0]
    query_count = min(max(1, args.num_anchor_queries), total_queries)
    query_indices = rng.choice(total_queries, size=query_count, replace=False)
    query_anchors = {
        key: (value[query_indices] if isinstance(value, np.ndarray) else [value[idx] for idx in query_indices])
        for key, value in query_anchors_full.items()
    }

    report["anchor_diversity"] = {
        "query_split": query_label,
        **evaluate_anchor_diversity(
            train_anchors=train_anchors,
            query_anchors=query_anchors,
            neighbors=args.neighbors,
            batch_size=args.batch_size,
        ),
    }
    report["random_target_progress_oracle"] = {
        "query_split": query_label,
        **evaluate_random_target_progress(
            train_anchors=train_anchors,
            query_anchors=query_anchors,
            neighbors=args.neighbors,
            batch_size=args.batch_size,
            random_seed=args.random_seed,
            target_radius=float(train_dataset["generator_config"].get("source_radius", REACH_MAX)),
            progress_thresholds=[float(v) for v in args.progress_thresholds],
        ),
    }

    output_json = Path(args.output_json) if args.output_json else None
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"Wrote {output_json}", flush=True)
    else:
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
