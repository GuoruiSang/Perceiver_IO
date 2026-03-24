#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

import h5py
from tqdm import tqdm

from scripts.data.generate_reacher_per_step_branching_dataset import generate_split, summarize_h5


def _chunk_sizes(total: int, parts: int) -> list[int]:
    base = total // parts
    remainder = total % parts
    return [base + (1 if i < remainder else 0) for i in range(parts)]


def _worker_build(args: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    shard_path = Path(args["shard_path"])
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    summary = generate_split(
        output_path=shard_path,
        xml_path=args["xml_path"],
        num_trajectories=None,
        source_budget=args["source_budget"],
        trajectory_length=args["trajectory_length"],
        dt=args["dt"],
        source_qvel_scale=args["source_qvel_scale"],
        root_control_points=args["root_control_points"],
        root_tau_scale=args["root_tau_scale"],
        anchor_min=args["anchor_min"],
        suffix_min_len=args["suffix_min_len"],
        anchor_stride=args["anchor_stride"],
        anchors_per_root=args["anchors_per_root"],
        roots_per_source=args["roots_per_source"],
        branches_per_anchor=args["branches_per_anchor"],
        suffix_control_points=args["suffix_control_points"],
        suffix_residual_scale=args["suffix_residual_scale"],
        waypoint_offset_min=args["waypoint_offset_min"],
        max_branch_attempts=args["max_branch_attempts"],
        max_abs_tau=args["max_abs_tau"],
        max_abs_qvel=args["max_abs_qvel"],
        max_abs_qacc=args["max_abs_qacc"],
        min_suffix_xy_rmse=args["min_suffix_xy_rmse"],
        min_final_xy_gap=args["min_final_xy_gap"],
        boundary_margin_ratio=args["boundary_margin_ratio"],
        boundary_tau_slope_cap=args["boundary_tau_slope_cap"],
        seed_offset=args["seed_offset"],
        split_name=args["split_name"],
    )
    shard_summary = summarize_h5(shard_path)
    return {
        "shard_index": args["shard_index"],
        "shard_path": str(shard_path),
        "generation": summary,
        "summary": shard_summary,
    }


def _copy_group_with_offsets(src_group: h5py.Group, dst_group: h5py.Group, source_offset: int, root_offset: int, anchor_offset: int) -> None:
    for key, value in src_group.attrs.items():
        dst_group.attrs[key] = value
    if "source_group_id" in dst_group.attrs:
        dst_group.attrs["source_group_id"] = int(dst_group.attrs["source_group_id"]) + source_offset
    if "root_group_id" in dst_group.attrs:
        dst_group.attrs["root_group_id"] = int(dst_group.attrs["root_group_id"]) + root_offset
    if "anchor_branch_group_id" in dst_group.attrs:
        dst_group.attrs["anchor_branch_group_id"] = int(dst_group.attrs["anchor_branch_group_id"]) + anchor_offset


def merge_shards(shard_results: list[dict[str, Any]], output_path: Path, total_source_budget: int) -> dict[str, Any]:
    shard_results = sorted(shard_results, key=lambda item: int(item["shard_index"]))
    if not shard_results:
        raise ValueError("No shard results to merge")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_path = Path(shard_results[0]["shard_path"])
    with h5py.File(first_path, "r") as first_file, h5py.File(output_path, "w") as out_file:
        for key, value in first_file.attrs.items():
            out_file.attrs[key] = value
        cfg = json.loads(out_file.attrs["generator_config"])
        cfg["source_budget"] = int(total_source_budget)
        cfg["split_name"] = "train"
        cfg["merged_shards"] = len(shard_results)
        out_file.attrs["generator_config"] = json.dumps(cfg, sort_keys=True)
        out_file.attrs["num_trajectories"] = 0

        next_idx = 0
        source_offset = 0
        root_offset = 0
        anchor_offset = 0
        for shard in tqdm(shard_results, desc="Merging shards"):
            shard_path = Path(shard["shard_path"])
            num_traj = int(shard["summary"]["num_trajectories"])
            with h5py.File(shard_path, "r") as shard_file:
                for traj_idx in range(num_traj):
                    src_group = shard_file[f"traj_{traj_idx}"]
                    dst_group = out_file.create_group(f"traj_{next_idx}")
                    for ds_name, ds in src_group.items():
                        dst_group.create_dataset(ds_name, data=ds[...], dtype=ds.dtype)
                    _copy_group_with_offsets(src_group, dst_group, source_offset, root_offset, anchor_offset)
                    next_idx += 1
            source_offset += int(shard["generation"]["source_groups"])
            root_offset += int(shard["generation"]["root_groups"])
            anchor_offset += int(shard["generation"]["anchor_branch_groups"])
        out_file.attrs["num_trajectories"] = int(next_idx)

    return summarize_h5(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a large per-step branching Reacher dataset in parallel shards")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--xml_path", type=str, default="/home/gsang/Projects/hnn_guided_dpf/configs/reacher_non_diss_unbounded_j1.xml")
    parser.add_argument("--num_sources", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--trajectory_length", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--source_qvel_scale", type=float, default=0.1)
    parser.add_argument("--root_control_points", type=int, default=12)
    parser.add_argument("--root_tau_scale", type=float, default=0.06)
    parser.add_argument("--anchor_min", type=int, default=8)
    parser.add_argument("--suffix_min_len", type=int, default=8)
    parser.add_argument("--anchor_stride", type=int, default=8)
    parser.add_argument("--anchors_per_root", type=int, default=2)
    parser.add_argument("--roots_per_source", type=int, default=2)
    parser.add_argument("--branches_per_anchor", type=int, default=2)
    parser.add_argument("--suffix_control_points", type=int, default=16)
    parser.add_argument("--suffix_residual_scale", type=float, default=0.08)
    parser.add_argument("--waypoint_offset_min", type=int, default=16)
    parser.add_argument("--max_branch_attempts", type=int, default=64)
    parser.add_argument("--max_abs_tau", type=float, default=0.8)
    parser.add_argument("--max_abs_qvel", type=float, default=10.0)
    parser.add_argument("--max_abs_qacc", type=float, default=100.0)
    parser.add_argument("--min_suffix_xy_rmse", type=float, default=0.001)
    parser.add_argument("--min_final_xy_gap", type=float, default=0.001)
    parser.add_argument("--boundary_margin_ratio", type=float, default=0.95)
    parser.add_argument("--boundary_tau_slope_cap", type=float, default=0.75)
    parser.add_argument("--seed_base", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    num_workers = max(1, min(args.num_workers, args.num_sources))
    shard_sizes = [size for size in _chunk_sizes(args.num_sources, num_workers) if size > 0]
    final_path = output_dir / f"traj_from_{args.num_sources}_sources_steps_{args.trajectory_length}.h5"

    worker_jobs: list[dict[str, Any]] = []
    for shard_index, shard_sources in enumerate(shard_sizes):
        shard_dir = shards_dir / f"shard_{shard_index:03d}"
        shard_path = shard_dir / f"traj_from_{shard_sources}_sources_steps_{args.trajectory_length}.h5"
        worker_jobs.append(
            {
                "shard_index": shard_index,
                "shard_path": str(shard_path),
                "source_budget": int(shard_sources),
                "xml_path": args.xml_path,
                "trajectory_length": args.trajectory_length,
                "dt": args.dt,
                "source_qvel_scale": args.source_qvel_scale,
                "root_control_points": args.root_control_points,
                "root_tau_scale": args.root_tau_scale,
                "anchor_min": args.anchor_min,
                "suffix_min_len": args.suffix_min_len,
                "anchor_stride": args.anchor_stride,
                "anchors_per_root": args.anchors_per_root,
                "roots_per_source": args.roots_per_source,
                "branches_per_anchor": args.branches_per_anchor,
                "suffix_control_points": args.suffix_control_points,
                "suffix_residual_scale": args.suffix_residual_scale,
                "waypoint_offset_min": args.waypoint_offset_min,
                "max_branch_attempts": args.max_branch_attempts,
                "max_abs_tau": args.max_abs_tau,
                "max_abs_qvel": args.max_abs_qvel,
                "max_abs_qacc": args.max_abs_qacc,
                "min_suffix_xy_rmse": args.min_suffix_xy_rmse,
                "min_final_xy_gap": args.min_final_xy_gap,
                "boundary_margin_ratio": args.boundary_margin_ratio,
                "boundary_tau_slope_cap": args.boundary_tau_slope_cap,
                "seed_offset": args.seed_base + shard_index * 10_000_000,
                "split_name": f"train_shard_{shard_index:03d}",
            }
        )

    print("Configuration:", flush=True)
    for key, value in vars(args).items():
        print(f"  {key}: {value}", flush=True)
    print(f"  effective_workers: {num_workers}", flush=True)
    print(f"  shard_sizes: {shard_sizes}", flush=True)

    shard_results: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = [executor.submit(_worker_build, job) for job in worker_jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building shards"):
            shard_results.append(future.result())

    merged_summary = merge_shards(shard_results, final_path, total_source_budget=args.num_sources)
    report = {
        "dataset_dir": str(output_dir),
        "final_h5": str(final_path),
        "config": vars(args),
        "num_workers_effective": int(num_workers),
        "shards": sorted(shard_results, key=lambda item: int(item["shard_index"])),
        "merged_summary": merged_summary,
    }
    report_path = output_dir / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
