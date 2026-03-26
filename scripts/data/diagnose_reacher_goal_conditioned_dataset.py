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
import numpy as np
from tqdm import tqdm

from scripts.data.generate_reacher_goal_conditioned_branching_dataset import (
    DEFAULT_DISTANCE_BIN_EDGES,
    REACH_MAX,
    REACH_MIN,
)


def threshold_label(threshold_m: float) -> str:
    cm = threshold_m * 100.0
    if abs(cm - round(cm)) < 1e-9:
        return f"within_{int(round(cm))}cm"
    text = f"{cm:.1f}".rstrip("0").rstrip(".")
    return f"within_{text.replace('.', 'p')}cm"


def summarize_array(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def summarize_thresholds(min_dists: np.ndarray, thresholds_m: list[float]) -> dict:
    if min_dists.size == 0:
        return {}
    return {
        threshold_label(threshold): float(np.mean(min_dists <= threshold))
        for threshold in thresholds_m
    }


def compute_distance_bin(distance: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges, distance, side="right") - 1
    return np.clip(bins, 0, len(edges) - 2).astype(np.int32)


def state_features(qpos: np.ndarray, mom: np.ndarray) -> np.ndarray:
    return np.concatenate([np.sin(qpos), np.cos(qpos), mom], axis=1).astype(np.float64)


def sample_workspace_uniform_targets(count: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(count)
    theta = rng.uniform(-math.pi, math.pi, size=count)
    radius = np.sqrt(u * (REACH_MAX * REACH_MAX - REACH_MIN * REACH_MIN) + REACH_MIN * REACH_MIN)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1).astype(np.float64)


def per_bin_stats(bin_ids: np.ndarray, min_dists: np.ndarray, thresholds_m: list[float]) -> dict:
    stats: dict[str, dict] = {}
    for bin_id in sorted(int(v) for v in np.unique(bin_ids)):
        mask = bin_ids == bin_id
        stats[str(bin_id)] = {
            "count": int(np.sum(mask)),
            **summarize_thresholds(min_dists[mask], thresholds_m),
            "min_dist": summarize_array(min_dists[mask]),
        }
    return stats


def load_goal_conditioned_dataset(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        num_trajectories = int(f.attrs["num_trajectories"])
        if num_trajectories <= 0:
            raise ValueError(f"{h5_path} contains no trajectories")

        anchor_state_qpos = []
        anchor_state_mom = []
        anchor_xy = []
        target_xy = []
        goal_distance_from_anchor = []
        goal_angle_from_anchor = []
        target_reached_min_dist = []
        planner_success = []
        anchor_index = []
        target_distance_bin = []
        target_angle_bin = []
        target_bin_id = []
        momentum_bin = []
        suffix_xy = []

        for idx in tqdm(range(num_trajectories), desc=f"Loading {h5_path.name}"):
            group = f[f"traj_{idx}"]
            anchor_state_qpos.append(group["anchor_state_qpos"][:].astype(np.float64))
            anchor_state_mom.append(group["anchor_state_mom"][:].astype(np.float64))
            anchor_xy.append(group["anchor_xy"][:].astype(np.float64))
            target_xy.append(group["target_xy"][:].astype(np.float64))
            goal_distance_from_anchor.append(float(group["goal_distance_from_anchor"][()]))
            goal_angle_from_anchor.append(float(group["goal_angle_from_anchor"][()]))
            target_reached_min_dist.append(float(group["target_reached_min_dist"][()]))
            planner_success.append(int(group["planner_success"][()]))

            traj_anchor_index = int(group.attrs["anchor_index"])
            anchor_index.append(traj_anchor_index)
            target_distance_bin.append(int(group.attrs.get("target_distance_bin", -1)))
            target_angle_bin.append(int(group.attrs.get("target_angle_bin", -1)))
            target_bin_id.append(int(group.attrs.get("target_bin_id", -1)))
            momentum_bin.append(int(group.attrs.get("momentum_bin", -1)))
            suffix_xy.append(group["seq_fingertip_xy"][traj_anchor_index:].astype(np.float64))

        anchor_state_qpos_arr = np.stack(anchor_state_qpos, axis=0)
        anchor_state_mom_arr = np.stack(anchor_state_mom, axis=0)
        anchor_xy_arr = np.stack(anchor_xy, axis=0)
        target_xy_arr = np.stack(target_xy, axis=0)
        goal_distance_arr = np.asarray(goal_distance_from_anchor, dtype=np.float64)
        goal_angle_arr = np.asarray(goal_angle_from_anchor, dtype=np.float64)
        min_dist_arr = np.asarray(target_reached_min_dist, dtype=np.float64)
        planner_success_arr = np.asarray(planner_success, dtype=np.int32)
        anchor_index_arr = np.asarray(anchor_index, dtype=np.int32)
        target_distance_bin_arr = np.asarray(target_distance_bin, dtype=np.int32)
        target_angle_bin_arr = np.asarray(target_angle_bin, dtype=np.int32)
        target_bin_id_arr = np.asarray(target_bin_id, dtype=np.int32)
        momentum_bin_arr = np.asarray(momentum_bin, dtype=np.int32)

        features = state_features(anchor_state_qpos_arr, anchor_state_mom_arr)
        generator_config = json.loads(f.attrs["generator_config"])

    return {
        "path": str(h5_path),
        "num_trajectories": num_trajectories,
        "generator_config": generator_config,
        "anchor_state_qpos": anchor_state_qpos_arr,
        "anchor_state_mom": anchor_state_mom_arr,
        "anchor_xy": anchor_xy_arr,
        "target_xy": target_xy_arr,
        "goal_distance_from_anchor": goal_distance_arr,
        "goal_angle_from_anchor": goal_angle_arr,
        "target_reached_min_dist": min_dist_arr,
        "planner_success": planner_success_arr,
        "anchor_index": anchor_index_arr,
        "target_distance_bin": target_distance_bin_arr,
        "target_angle_bin": target_angle_bin_arr,
        "target_bin_id": target_bin_id_arr,
        "momentum_bin": momentum_bin_arr,
        "suffix_xy": suffix_xy,
        "state_features": features,
    }


def dataset_quality_summary(dataset: dict, thresholds_m: list[float]) -> dict:
    return {
        "path": dataset["path"],
        "num_trajectories": int(dataset["num_trajectories"]),
        "stored_target_min_dist": summarize_array(dataset["target_reached_min_dist"]),
        "stored_target_success": summarize_thresholds(dataset["target_reached_min_dist"], thresholds_m),
        "goal_distance_from_anchor": summarize_array(dataset["goal_distance_from_anchor"]),
        "goal_angle_from_anchor": summarize_array(dataset["goal_angle_from_anchor"]),
        "by_distance_bin": per_bin_stats(
            dataset["target_distance_bin"],
            dataset["target_reached_min_dist"],
            thresholds_m,
        ),
        "by_angle_bin": per_bin_stats(
            dataset["target_angle_bin"],
            dataset["target_reached_min_dist"],
            thresholds_m,
        ),
        "by_momentum_bin": per_bin_stats(
            dataset["momentum_bin"],
            dataset["target_reached_min_dist"],
            thresholds_m,
        ),
    }


def evaluate_knn_oracle(
    train_dataset: dict,
    query_qpos: np.ndarray,
    query_mom: np.ndarray,
    query_anchor_xy: np.ndarray,
    query_target_xy: np.ndarray,
    neighbors: int,
    thresholds_m: list[float],
    distance_bin_edges: np.ndarray,
    batch_size: int = 64,
    desc: str = "oracle",
) -> dict:
    train_features = train_dataset["state_features"].astype(np.float64)
    sigma = train_features.std(axis=0) + 1e-6
    train_features_std = train_features / sigma[None, :]
    train_sq = np.sum(np.square(train_features_std, dtype=np.float64), axis=1)
    suffix_xy = train_dataset["suffix_xy"]

    query_features_std = state_features(query_qpos, query_mom) / sigma[None, :]
    query_count = query_features_std.shape[0]
    k = max(1, min(neighbors, train_features_std.shape[0]))

    best_min_dists = np.empty(query_count, dtype=np.float64)
    best_neighbor_state_dist = np.empty(query_count, dtype=np.float64)
    anchor_to_target_dist = np.linalg.norm(query_target_xy - query_anchor_xy, axis=1)

    for start in tqdm(range(0, query_count, batch_size), desc=desc):
        stop = min(start + batch_size, query_count)
        batch_feat = query_features_std[start:stop]
        batch_sq = np.sum(np.square(batch_feat, dtype=np.float64), axis=1)
        dist2 = (
            batch_sq[:, None]
            + train_sq[None, :]
            - 2.0 * (batch_feat @ train_features_std.T)
        )
        dist2 = np.maximum(dist2, 0.0)
        neighbor_idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]

        for row in range(stop - start):
            idxs = neighbor_idx[row]
            row_dist2 = dist2[row, idxs]
            order = np.argsort(row_dist2)
            idxs = idxs[order]
            row_dist2 = row_dist2[order]

            target_xy = query_target_xy[start + row]
            best_min = float("inf")
            for train_idx in idxs.tolist():
                candidate_min = float(
                    np.min(np.linalg.norm(suffix_xy[train_idx] - target_xy[None, :], axis=1))
                )
                if candidate_min < best_min:
                    best_min = candidate_min

            best_min_dists[start + row] = best_min
            best_neighbor_state_dist[start + row] = math.sqrt(float(row_dist2[0]))

    query_distance_bin = compute_distance_bin(anchor_to_target_dist, distance_bin_edges)
    return {
        "num_queries": int(query_count),
        "neighbors": int(k),
        "best_min_dist": summarize_array(best_min_dists),
        "success": summarize_thresholds(best_min_dists, thresholds_m),
        "nearest_state_distance": summarize_array(best_neighbor_state_dist),
        "anchor_to_target_distance": summarize_array(anchor_to_target_dist),
        "by_query_distance_bin": per_bin_stats(query_distance_bin, best_min_dists, thresholds_m),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a goal-conditioned Reacher branching dataset")
    parser.add_argument("--train_h5", type=str, required=True)
    parser.add_argument("--val_h5", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--num_queries", type=int, default=1000)
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--success_threshold", type=float, default=0.005)
    parser.add_argument("--relaxed_thresholds", type=float, nargs="*", default=[0.01, 0.02])
    parser.add_argument("--distance_bin_edges", type=float, nargs="+", default=list(DEFAULT_DISTANCE_BIN_EDGES))
    args = parser.parse_args()

    thresholds_m = [float(args.success_threshold)] + [float(v) for v in args.relaxed_thresholds]
    thresholds_m = sorted(set(thresholds_m))
    distance_bin_edges = np.asarray(args.distance_bin_edges, dtype=np.float64)

    train_h5 = Path(args.train_h5)
    val_h5 = Path(args.val_h5) if args.val_h5 else None

    train_dataset = load_goal_conditioned_dataset(train_h5)
    report = {
        "train_dataset": dataset_quality_summary(train_dataset, thresholds_m),
    }

    if val_h5 is not None:
        val_dataset = load_goal_conditioned_dataset(val_h5)
        report["val_dataset"] = dataset_quality_summary(val_dataset, thresholds_m)

        rng = np.random.default_rng(args.random_seed)
        total_val = int(val_dataset["num_trajectories"])
        query_count = min(max(1, args.num_queries), total_val)
        query_indices = rng.choice(total_val, size=query_count, replace=False)

        query_qpos = val_dataset["anchor_state_qpos"][query_indices]
        query_mom = val_dataset["anchor_state_mom"][query_indices]
        query_anchor_xy = val_dataset["anchor_xy"][query_indices]
        stored_targets = val_dataset["target_xy"][query_indices]
        random_targets = sample_workspace_uniform_targets(query_count, rng)

        report["oracle_eval"] = {
            "query_split": str(val_h5),
            "query_count": int(query_count),
            "stored_targets": evaluate_knn_oracle(
                train_dataset=train_dataset,
                query_qpos=query_qpos,
                query_mom=query_mom,
                query_anchor_xy=query_anchor_xy,
                query_target_xy=stored_targets,
                neighbors=args.neighbors,
                thresholds_m=thresholds_m,
                distance_bin_edges=distance_bin_edges,
                batch_size=args.batch_size,
                desc="Stored-target oracle",
            ),
            "random_workspace_targets": evaluate_knn_oracle(
                train_dataset=train_dataset,
                query_qpos=query_qpos,
                query_mom=query_mom,
                query_anchor_xy=query_anchor_xy,
                query_target_xy=random_targets,
                neighbors=args.neighbors,
                thresholds_m=thresholds_m,
                distance_bin_edges=distance_bin_edges,
                batch_size=args.batch_size,
                desc="Random-target oracle",
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
