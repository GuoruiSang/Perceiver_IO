#!/usr/bin/env python3
"""Create a strict train/val split for the branching Reacher dataset by source_group_id."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import h5py


def _trajectory_sort_key(name: str) -> int:
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return 0


def _copy_selected_groups(src_path: Path, dst_path: Path, group_names: list[str]) -> None:
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["num_trajectories"] = len(group_names)
        for new_idx, name in enumerate(group_names):
            src.copy(name, dst, name=f"traj_{new_idx}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_h5", type=Path, required=True, help="Full branching dataset H5")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for split outputs")
    parser.add_argument("--train_fraction", type=float, default=0.9, help="Fraction of source groups for train")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for source-group split")
    args = parser.parse_args()

    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train_fraction must be in (0, 1)")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_to_groups: dict[int, list[str]] = defaultdict(list)
    with h5py.File(args.input_h5, "r") as src:
        for name in src.keys():
            source_group_id = int(src[name].attrs["source_group_id"])
            source_to_groups[source_group_id].append(name)

    source_ids = sorted(source_to_groups)
    rng = random.Random(args.seed)
    rng.shuffle(source_ids)
    num_train_sources = int(len(source_ids) * args.train_fraction)
    train_source_ids = set(source_ids[:num_train_sources])
    val_source_ids = set(source_ids[num_train_sources:])

    train_group_names: list[str] = []
    val_group_names: list[str] = []
    for source_group_id, names in source_to_groups.items():
        target = train_group_names if source_group_id in train_source_ids else val_group_names
        target.extend(names)

    train_group_names.sort(key=_trajectory_sort_key)
    val_group_names.sort(key=_trajectory_sort_key)

    train_path = args.output_dir / (
        f"{args.input_h5.stem}_train_sources{len(train_source_ids)}_steps_500.h5"
    )
    val_path = args.output_dir / (
        f"{args.input_h5.stem}_val_sources{len(val_source_ids)}_steps_500.h5"
    )

    _copy_selected_groups(args.input_h5, train_path, train_group_names)
    _copy_selected_groups(args.input_h5, val_path, val_group_names)

    train_counts = Counter()
    val_counts = Counter()
    for source_group_id, names in source_to_groups.items():
        if source_group_id in train_source_ids:
            train_counts[source_group_id] = len(names)
        else:
            val_counts[source_group_id] = len(names)

    report = {
        "input_h5": str(args.input_h5),
        "train_h5": str(train_path),
        "val_h5": str(val_path),
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "total_source_groups": len(source_ids),
        "train_source_groups": len(train_source_ids),
        "val_source_groups": len(val_source_ids),
        "train_trajectories": len(train_group_names),
        "val_trajectories": len(val_group_names),
        "train_traj_per_source": {
            "min": min(train_counts.values()) if train_counts else 0,
            "max": max(train_counts.values()) if train_counts else 0,
            "mean": (sum(train_counts.values()) / len(train_counts)) if train_counts else 0.0,
        },
        "val_traj_per_source": {
            "min": min(val_counts.values()) if val_counts else 0,
            "max": max(val_counts.values()) if val_counts else 0,
            "mean": (sum(val_counts.values()) / len(val_counts)) if val_counts else 0.0,
        },
    }
    report_path = args.output_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
