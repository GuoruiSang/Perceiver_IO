#!/home/gsang/miniconda3/envs/perceiver/bin/python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "data" / "run_reacher_goal_prefix_expansion.py"
PYTHON_BIN = Path("/home/gsang/miniconda3/envs/perceiver/bin/python")

DEFAULT_DPF_CHECKPOINT = (
    "/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/"
    "dpf_exploration_iid_uniform_len1000_v1/"
    "trajectory_dpf_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone"
    "&DecoderAttentions_cond-concat_torque_in_state_layout-shifted_tau_qpos-raw:epoch=2999_val_loss:val_loss=0.0983.ckpt"
)
DEFAULT_H5_PATH = (
    "/home/gsang/Projects/hnn_guided_dpf/data/"
    "reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/val_traj_4000-steps_1000.h5"
)
DEFAULT_OUTPUT_ROOT = (
    "/home/gsang/Projects/hnn_guided_dpf/plots/"
    f"reacher_unguided_candidate_sweep_sharded_{date.today().isoformat()}"
)
DEFAULT_TASK_MODE = "validation_random_source_random_target_across_trajs"
DEFAULT_TASK_IDS = list(range(10))
DEFAULT_CANDIDATE_COUNTS = [16, 32, 64, 128]
DEFAULT_SPLIT_CANDIDATE_COUNTS = [32, 128]
DEFAULT_GPU_IDS = [0, 1, 3]


@dataclass(frozen=True)
class ShardJob:
    name: str
    candidate_count: int
    shard_index: int
    task_ids: list[int]
    output_dir: Path
    command: list[str]
    estimated_cost: int


def parse_int_list(text: str) -> list[int]:
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_row(row: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"rollout_qpos_raw", "rollout_mom", "rollout_tau", "selection_trace"}
    }


def aggregate_metric(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return {}
    values_array = np.asarray([float(value) for value in values], dtype=np.float64)
    return {
        "mean": float(values_array.mean()),
        "median": float(np.median(values_array)),
        "min": float(values_array.min()),
        "max": float(values_array.max()),
    }


def summary_is_complete(summary: dict[str, object]) -> bool:
    if "is_complete" in summary:
        return bool(summary["is_complete"])
    task_ids = summary.get("task_ids") or []
    rows = summary.get("rows") or []
    return len(rows) >= len(task_ids) and len(task_ids) > 0


class SweepLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._lock = threading.Lock()

    def log(self, text: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}"
        with self._lock:
            print(line, flush=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run unguided Reacher candidate-count sweeps with task sharding. "
            "Heavy candidate counts can be split across task subsets, and merged progress "
            "files are refreshed while the run is still in flight."
        )
    )
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_DPF_CHECKPOINT)
    parser.add_argument("--runner_path", type=str, default=str(RUNNER_PATH))
    parser.add_argument("--h5_path", type=str, default=DEFAULT_H5_PATH)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task_mode", type=str, default=DEFAULT_TASK_MODE)
    parser.add_argument("--task_ids", type=str, default=",".join(str(x) for x in DEFAULT_TASK_IDS))
    parser.add_argument(
        "--candidate_counts",
        type=str,
        default=",".join(str(x) for x in DEFAULT_CANDIDATE_COUNTS),
    )
    parser.add_argument(
        "--split_candidate_counts",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SPLIT_CANDIDATE_COUNTS),
        help="Candidate counts that should be split into two interleaved task shards.",
    )
    parser.add_argument("--gpu_ids", type=str, default=",".join(str(x) for x in DEFAULT_GPU_IDS))
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help=(
            "Number of independent shard workers to run concurrently per listed GPU. "
            "Useful when each worker under-utilizes the GPU."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidate_scoring",
        type=str,
        default="predicted_suffix",
        choices=["predicted_suffix", "mujoco_rollout", "hnn_iid_random_shooting"],
        help="Candidate ranking mode forwarded to run_reacher_goal_prefix_expansion.py.",
    )
    parser.add_argument(
        "--torque_scale",
        type=float,
        default=0.2,
        help="Forwarded for HNN iid random shooting; ignored by DPF rollout runners.",
    )
    parser.add_argument("--num_diffusion_steps", type=int, default=20)
    parser.add_argument("--lookahead_steps", type=int, default=256)
    parser.add_argument("--recent_prefix_cap", type=int, default=64)
    parser.add_argument("--stall_patience_steps", type=int, default=500)
    parser.add_argument("--goal_tolerance", type=float, default=0.01)
    parser.add_argument("--random_future_target_min_initial_distance", type=float, default=0.1)
    parser.add_argument("--cross_traj_sampling_max_tries", type=int, default=128)
    parser.add_argument("--max_sampling_retries", type=int, default=3)
    parser.add_argument("--retry_improvement_margin", type=float, default=1e-3)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip shards whose final merged summary already exists and is complete.",
    )
    parser.add_argument(
        "--poll_interval_sec",
        type=float,
        default=20.0,
        help="Refresh merged progress files at this interval while shards are running.",
    )
    return parser


def base_command(args: argparse.Namespace, *, output_dir: Path, task_ids: list[int], candidate_count: int) -> list[str]:
    return [
        str(PYTHON_BIN),
        str(args.runner_path),
        "--checkpoint_path",
        str(args.checkpoint_path),
        "--h5_path",
        str(args.h5_path),
        "--output_dir",
        str(output_dir),
        "--task_mode",
        str(args.task_mode),
        "--task_ids",
        ",".join(str(task_id) for task_id in task_ids),
        "--num_candidates",
        str(int(candidate_count)),
        "--candidate_scoring",
        str(args.candidate_scoring),
        "--torque_scale",
        str(float(args.torque_scale)),
        "--num_diffusion_steps",
        str(int(args.num_diffusion_steps)),
        "--lookahead_steps",
        str(int(args.lookahead_steps)),
        "--recent_prefix_cap",
        str(int(args.recent_prefix_cap)),
        "--reset_window_time_indices",
        "--stall_patience_steps",
        str(int(args.stall_patience_steps)),
        "--goal_tolerance",
        str(float(args.goal_tolerance)),
        "--random_future_target_min_initial_distance",
        str(float(args.random_future_target_min_initial_distance)),
        "--cross_traj_sampling_max_tries",
        str(int(args.cross_traj_sampling_max_tries)),
        "--max_sampling_retries",
        str(int(args.max_sampling_retries)),
        "--retry_improvement_margin",
        str(float(args.retry_improvement_margin)),
        "--seed",
        str(int(args.seed)),
        "--device",
        "cuda:0",
    ]


def build_jobs(
    args: argparse.Namespace,
    *,
    task_ids: list[int],
    candidate_counts: list[int],
    split_candidate_counts: set[int],
    output_root: Path,
) -> tuple[list[ShardJob], dict[int, list[Path]]]:
    jobs: list[ShardJob] = []
    shard_dirs_by_candidate: dict[int, list[Path]] = {}

    for candidate_count in sorted(candidate_counts, reverse=True):
        candidate_dir = output_root / f"n{candidate_count}"
        if candidate_count in split_candidate_counts and len(task_ids) >= 2:
            shards = [task_ids[::2], task_ids[1::2]]
        else:
            shards = [task_ids]
        shard_dirs_by_candidate[candidate_count] = []
        for shard_index, shard_task_ids in enumerate(shards):
            if not shard_task_ids:
                continue
            output_dir = candidate_dir / "shards" / f"shard{shard_index:02d}"
            shard_dirs_by_candidate[candidate_count].append(output_dir)
            jobs.append(
                ShardJob(
                    name=f"n{candidate_count}_shard{shard_index:02d}",
                    candidate_count=int(candidate_count),
                    shard_index=int(shard_index),
                    task_ids=list(shard_task_ids),
                    output_dir=output_dir,
                    command=base_command(
                        args,
                        output_dir=output_dir,
                        task_ids=list(shard_task_ids),
                        candidate_count=int(candidate_count),
                    ),
                    estimated_cost=int(candidate_count) * len(shard_task_ids),
                )
            )
    return jobs, shard_dirs_by_candidate


def assign_jobs_to_gpus(jobs: list[ShardJob], gpu_ids: list[int]) -> dict[int, list[ShardJob]]:
    queues = {gpu_id: [] for gpu_id in gpu_ids}
    loads = {gpu_id: 0 for gpu_id in gpu_ids}
    for job in sorted(jobs, key=lambda item: item.estimated_cost, reverse=True):
        gpu_id = min(gpu_ids, key=lambda gpu: (loads[gpu], len(queues[gpu])))
        queues[gpu_id].append(job)
        loads[gpu_id] += int(job.estimated_cost)
    return queues


def assign_jobs_to_gpu_slots(jobs: list[ShardJob], gpu_ids: list[int], workers_per_gpu: int) -> dict[int, dict[str, object]]:
    slots: dict[int, dict[str, object]] = {}
    slot_ids: list[int] = []
    slot_id = 0
    for gpu_id in gpu_ids:
        for worker_index in range(max(1, int(workers_per_gpu))):
            slots[slot_id] = {
                "gpu_id": int(gpu_id),
                "worker_index": int(worker_index),
                "jobs": [],
                "load": 0,
            }
            slot_ids.append(slot_id)
            slot_id += 1

    for job in sorted(jobs, key=lambda item: item.estimated_cost, reverse=True):
        best_slot_id = min(
            slot_ids,
            key=lambda candidate: (
                int(slots[candidate]["load"]),
                len(slots[candidate]["jobs"]),
                int(slots[candidate]["gpu_id"]),
                int(slots[candidate]["worker_index"]),
            ),
        )
        slots[best_slot_id]["jobs"].append(job)
        slots[best_slot_id]["load"] = int(slots[best_slot_id]["load"]) + int(job.estimated_cost)
    return slots


def build_merged_candidate_payload(
    *,
    args: argparse.Namespace,
    candidate_count: int,
    global_task_ids: list[int],
    shard_dirs: list[Path],
) -> dict[str, object]:
    rows_by_task_id: dict[int, dict[str, object]] = {}
    shard_status: list[dict[str, object]] = []

    for shard_dir in shard_dirs:
        progress_path = shard_dir / "reacher_goal_prefix_expansion_progress.json"
        summary_path = shard_dir / "reacher_goal_prefix_expansion_summary.json"
        payload_path = summary_path if summary_path.exists() else progress_path
        if not payload_path.exists():
            shard_status.append(
                {
                    "shard_dir": str(shard_dir),
                    "exists": False,
                    "is_complete": False,
                    "num_completed_tasks": 0,
                    "num_total_tasks": None,
                }
            )
            continue

        payload = read_json(payload_path)
        compact_rows = [compact_row(row) for row in payload.get("rows", [])]
        for row in compact_rows:
            rows_by_task_id[int(row["task_id"])] = row
        shard_status.append(
            {
                "shard_dir": str(shard_dir),
                "exists": True,
                "is_complete": bool(summary_is_complete(payload)),
                "num_completed_tasks": int(payload.get("num_completed_tasks", len(compact_rows))),
                "num_total_tasks": payload.get("num_total_tasks", len(payload.get("task_ids", []))),
                "task_ids": payload.get("task_ids", []),
                "summary_path": str(payload_path),
            }
        )

    rows = [rows_by_task_id[task_id] for task_id in sorted(rows_by_task_id)]
    is_complete = bool(rows) and len(rows) == len(global_task_ids) and all(
        bool(shard["is_complete"]) for shard in shard_status if bool(shard["exists"])
    )
    return {
        "checkpoint_path": str(args.checkpoint_path),
        "runner_path": str(args.runner_path),
        "h5_path": str(args.h5_path),
        "task_mode": str(args.task_mode),
        "task_ids": list(global_task_ids),
        "num_total_tasks": int(len(global_task_ids)),
        "num_completed_tasks": int(len(rows)),
        "completed_task_ids": [int(row["task_id"]) for row in rows],
        "is_complete": bool(is_complete),
        "num_candidates": int(candidate_count),
        "candidate_scoring": str(args.candidate_scoring),
        "num_diffusion_steps": int(args.num_diffusion_steps),
        "lookahead_steps": int(args.lookahead_steps),
        "recent_prefix_cap": int(args.recent_prefix_cap),
        "reset_window_time_indices": True,
        "stall_patience_steps": int(args.stall_patience_steps),
        "goal_tolerance": float(args.goal_tolerance),
        "random_future_target_min_initial_distance": float(args.random_future_target_min_initial_distance),
        "cross_traj_sampling_max_tries": int(args.cross_traj_sampling_max_tries),
        "max_sampling_retries": int(args.max_sampling_retries),
        "retry_improvement_margin": float(args.retry_improvement_margin),
        "aggregate": {
            "success_rate": float(np.mean([float(row["reached_goal"]) for row in rows])) if rows else 0.0,
            "best_goal_distance": aggregate_metric(rows, "best_goal_distance"),
            "final_goal_distance": aggregate_metric(rows, "final_goal_distance"),
            "qpos_mse_to_replay": aggregate_metric(rows, "qpos_mse_to_replay"),
            "mom_mse_to_replay": aggregate_metric(rows, "mom_mse_to_replay"),
            "ee_xy_mse_to_replay": aggregate_metric(rows, "ee_xy_mse_to_replay"),
        },
        "rows": rows,
        "shards": shard_status,
        "is_compact_merged_summary": True,
    }


def write_candidate_progress(
    *,
    args: argparse.Namespace,
    candidate_count: int,
    global_task_ids: list[int],
    shard_dirs: list[Path],
    output_root: Path,
) -> dict[str, object]:
    candidate_dir = output_root / f"n{candidate_count}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    payload = build_merged_candidate_payload(
        args=args,
        candidate_count=candidate_count,
        global_task_ids=global_task_ids,
        shard_dirs=shard_dirs,
    )
    progress_path = candidate_dir / "reacher_goal_prefix_expansion_progress.json"
    write_json(progress_path, payload)

    aggregate = payload["aggregate"]
    method_label = str(args.candidate_scoring)
    if method_label == "predicted_suffix":
        method_label = "DPF unguided predicted-suffix selection"
    elif method_label == "mujoco_rollout":
        method_label = "DPF with MuJoCo oracle candidate selection"
    elif method_label == "hnn_iid_random_shooting":
        method_label = "HNN iid random shooting"

    lines = [
        f"# {method_label}: n={candidate_count}",
        "",
        f"- output_root = `{output_root}`",
        f"- checkpoint = `{args.checkpoint_path}`",
        f"- task_mode = `{args.task_mode}`",
        f"- candidate_scoring = `{args.candidate_scoring}`",
        f"- lookahead_steps = `{args.lookahead_steps}`",
        f"- recent_prefix_cap = `{args.recent_prefix_cap}`",
        f"- max_sampling_retries = `{args.max_sampling_retries}`",
        f"- status = `{'complete' if payload['is_complete'] else 'running'}`",
        f"- completed_tasks = `{payload['num_completed_tasks']}/{payload['num_total_tasks']}`",
        f"- progress_json = `{progress_path}`",
        "",
        "## Aggregate",
        "",
    ]
    if payload["num_completed_tasks"] > 0:
        lines.extend(
            [
                f"- success_rate = `{aggregate['success_rate']:.3f}`",
                f"- best_goal_mean = `{aggregate['best_goal_distance'].get('mean', float('nan')):.6f}`",
                f"- final_goal_mean = `{aggregate['final_goal_distance'].get('mean', float('nan')):.6f}`",
                "",
            ]
        )
    else:
        lines.extend(["No finished tasks yet.", ""])

    lines.extend(["## Shards", ""])
    for shard in payload["shards"]:
        if not shard["exists"]:
            lines.append(f"- `{shard['shard_dir']}`: not started")
        else:
            lines.append(
                f"- `{shard['shard_dir']}`: "
                f"`{shard['num_completed_tasks']}/{shard['num_total_tasks']}` tasks, "
                f"`{'complete' if shard['is_complete'] else 'running'}`"
            )
    lines.extend(["", "## Rows", ""])
    if payload["rows"]:
        lines.append("| task_id | reached_goal | best_goal_distance | final_goal_distance | workspace_gif |")
        lines.append("|---:|---:|---:|---:|---|")
        for row in payload["rows"]:
            lines.append(
                "| {task_id} | {reached_goal} | {best_goal_distance:.6f} | {final_goal_distance:.6f} | `{workspace_gif}` |".format(
                    task_id=int(row["task_id"]),
                    reached_goal=int(bool(row["reached_goal"])),
                    best_goal_distance=float(row["best_goal_distance"]),
                    final_goal_distance=float(row["final_goal_distance"]),
                    workspace_gif=str(row["workspace_gif"]),
                )
            )
    else:
        lines.append("No finished rows yet.")

    (candidate_dir / "reacher_goal_prefix_expansion_progress.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    if bool(payload["is_complete"]):
        write_json(candidate_dir / "reacher_goal_prefix_expansion_summary.json", payload)
    return payload


def write_root_progress(
    *,
    args: argparse.Namespace,
    candidate_counts: list[int],
    output_root: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for candidate_count in sorted(candidate_counts):
        progress_path = output_root / f"n{candidate_count}" / "reacher_goal_prefix_expansion_progress.json"
        if not progress_path.exists():
            continue
        payload = read_json(progress_path)
        rows.append(
            {
                "candidate_count": int(candidate_count),
                "is_complete": bool(payload.get("is_complete", False)),
                "num_completed_tasks": int(payload.get("num_completed_tasks", 0)),
                "num_total_tasks": int(payload.get("num_total_tasks", 0)),
                "aggregate": payload.get("aggregate", {}),
                "progress_path": str(progress_path),
            }
        )

    summary_payload = {
        "candidate_counts": list(sorted(candidate_counts)),
        "rows": rows,
    }
    write_json(output_root / "sweep_progress.json", summary_payload)

    method_label = str(args.candidate_scoring)
    if method_label == "predicted_suffix":
        method_label = "DPF Unguided Candidate Sweep"
    elif method_label == "mujoco_rollout":
        method_label = "DPF MuJoCo-Oracle Candidate Sweep"
    elif method_label == "hnn_iid_random_shooting":
        method_label = "HNN IID Random-Shooting Candidate Sweep"

    lines = [
        f"# {method_label} Progress",
        "",
        f"- output_root = `{output_root}`",
        f"- checkpoint = `{args.checkpoint_path}`",
        f"- h5_path = `{args.h5_path}`",
        f"- task_mode = `{args.task_mode}`",
        f"- candidate_scoring = `{args.candidate_scoring}`",
        f"- num_diffusion_steps = `{args.num_diffusion_steps}`",
        f"- lookahead_steps = `{args.lookahead_steps}`",
        f"- recent_prefix_cap = `{args.recent_prefix_cap}`",
        f"- stall_patience_steps = `{args.stall_patience_steps}`",
        f"- max_sampling_retries = `{args.max_sampling_retries}`",
        f"- retry_improvement_margin = `{args.retry_improvement_margin}`",
        "",
        "| candidates | status | tasks | success_rate | best_goal_mean | final_goal_mean | progress_json |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        aggregate = row["aggregate"]
        best_goal = aggregate.get("best_goal_distance", {}).get("mean")
        final_goal = aggregate.get("final_goal_distance", {}).get("mean")
        lines.append(
            "| {candidate_count} | {status} | {done}/{total} | {success:.3f} | {best} | {final} | `{path}` |".format(
                candidate_count=int(row["candidate_count"]),
                status="complete" if row["is_complete"] else "running",
                done=int(row["num_completed_tasks"]),
                total=int(row["num_total_tasks"]),
                success=float(aggregate.get("success_rate", 0.0)),
                best="-" if best_goal is None else f"{float(best_goal):.6f}",
                final="-" if final_goal is None else f"{float(final_goal):.6f}",
                path=str(row["progress_path"]),
            )
        )
    (output_root / "sweep_progress.md").write_text("\n".join(lines), encoding="utf-8")


def run_one_job(
    *,
    job: ShardJob,
    gpu_id: int,
    resume: bool,
    logger: SweepLogger,
) -> dict[str, object]:
    summary_path = job.output_dir / "reacher_goal_prefix_expansion_summary.json"
    progress_path = job.output_dir / "reacher_goal_prefix_expansion_progress.json"
    log_path = job.output_dir / "run.log"
    job.output_dir.mkdir(parents=True, exist_ok=True)

    if resume and summary_path.exists():
        summary = read_json(summary_path)
        if summary_is_complete(summary):
            logger.log(f"Skipping existing {job.name} on gpu{gpu_id}: {summary_path}")
            return {
                "job": job,
                "gpu_id": int(gpu_id),
                "summary_path": str(summary_path),
                "progress_path": str(progress_path),
                "duration_sec": 0.0,
                "skipped": True,
            }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    logger.log(
        f"Starting {job.name} on gpu{gpu_id} "
        f"(tasks={job.task_ids}, estimated_cost={job.estimated_cost})"
    )
    logger.log("Command: " + " ".join(job.command))
    start_time = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND:\n")
        handle.write(" ".join(job.command) + "\n\n")
        handle.flush()
        subprocess.run(
            job.command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    duration_sec = time.time() - start_time
    logger.log(f"Finished {job.name} on gpu{gpu_id} in {duration_sec / 60.0:.1f} min")
    return {
        "job": job,
        "gpu_id": int(gpu_id),
        "summary_path": str(summary_path),
        "progress_path": str(progress_path),
        "duration_sec": float(duration_sec),
        "skipped": False,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = SweepLogger(output_root / "sweep.log")

    task_ids = parse_int_list(args.task_ids)
    candidate_counts = sorted(parse_int_list(args.candidate_counts))
    split_candidate_counts = set(parse_int_list(args.split_candidate_counts)) if args.split_candidate_counts else set()
    gpu_ids = parse_int_list(args.gpu_ids)

    jobs, shard_dirs_by_candidate = build_jobs(
        args,
        task_ids=task_ids,
        candidate_counts=candidate_counts,
        split_candidate_counts=split_candidate_counts,
        output_root=output_root,
    )
    gpu_slots = assign_jobs_to_gpu_slots(jobs, gpu_ids, int(args.workers_per_gpu))

    config_payload = {
        "checkpoint_path": str(args.checkpoint_path),
        "h5_path": str(args.h5_path),
        "task_mode": str(args.task_mode),
        "task_ids": task_ids,
        "candidate_counts": candidate_counts,
        "split_candidate_counts": sorted(split_candidate_counts),
        "gpu_ids": gpu_ids,
        "workers_per_gpu": int(args.workers_per_gpu),
        "candidate_scoring": str(args.candidate_scoring),
        "torque_scale": float(args.torque_scale),
        "num_diffusion_steps": int(args.num_diffusion_steps),
        "lookahead_steps": int(args.lookahead_steps),
        "recent_prefix_cap": int(args.recent_prefix_cap),
        "stall_patience_steps": int(args.stall_patience_steps),
        "goal_tolerance": float(args.goal_tolerance),
        "random_future_target_min_initial_distance": float(args.random_future_target_min_initial_distance),
        "cross_traj_sampling_max_tries": int(args.cross_traj_sampling_max_tries),
        "max_sampling_retries": int(args.max_sampling_retries),
        "retry_improvement_margin": float(args.retry_improvement_margin),
        "resume": bool(args.resume),
        "poll_interval_sec": float(args.poll_interval_sec),
        "gpu_slots": {
            str(slot_id): {
                "gpu_id": int(slot["gpu_id"]),
                "worker_index": int(slot["worker_index"]),
                "load": int(slot["load"]),
                "jobs": [
                {
                    "name": job.name,
                    "candidate_count": int(job.candidate_count),
                    "shard_index": int(job.shard_index),
                    "task_ids": list(job.task_ids),
                    "estimated_cost": int(job.estimated_cost),
                    "output_dir": str(job.output_dir),
                }
                for job in slot["jobs"]
                ],
            }
            for slot_id, slot in gpu_slots.items()
        },
    }
    write_json(output_root / "sweep_config.json", config_payload)

    for candidate_count in candidate_counts:
        write_candidate_progress(
            args=args,
            candidate_count=int(candidate_count),
            global_task_ids=task_ids,
            shard_dirs=shard_dirs_by_candidate[int(candidate_count)],
            output_root=output_root,
        )
    write_root_progress(args=args, candidate_counts=candidate_counts, output_root=output_root)

    stop_event = threading.Event()
    result_lock = threading.Lock()
    results: dict[str, dict[str, object]] = {}

    def refresh_all_progress() -> None:
        for candidate_count in candidate_counts:
            write_candidate_progress(
                args=args,
                candidate_count=int(candidate_count),
                global_task_ids=task_ids,
                shard_dirs=shard_dirs_by_candidate[int(candidate_count)],
                output_root=output_root,
            )
        write_root_progress(args=args, candidate_counts=candidate_counts, output_root=output_root)

    def poller() -> None:
        while not stop_event.is_set():
            refresh_all_progress()
            stop_event.wait(float(args.poll_interval_sec))

    def worker(slot_id: int) -> None:
        slot = gpu_slots[slot_id]
        gpu_id = int(slot["gpu_id"])
        for job in slot["jobs"]:
            result = run_one_job(job=job, gpu_id=gpu_id, resume=bool(args.resume), logger=logger)
            with result_lock:
                results[job.name] = result
            refresh_all_progress()

    logger.log(f"Starting sharded unguided sweep in {output_root}")
    for slot_id, slot in gpu_slots.items():
        logger.log(
            f"Queue slot{slot_id} gpu{slot['gpu_id']} worker{slot['worker_index']}: "
            + ", ".join(job.name for job in slot["jobs"])
        )

    poll_thread = threading.Thread(target=poller, daemon=True)
    poll_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=len(gpu_slots)) as executor:
            futures = [executor.submit(worker, slot_id) for slot_id in gpu_slots]
            for future in futures:
                future.result()
    finally:
        stop_event.set()
        poll_thread.join(timeout=5.0)
        refresh_all_progress()

    logger.log(f"Finished sharded unguided sweep. Root progress: {output_root / 'sweep_progress.md'}")


if __name__ == "__main__":
    main()
