#!/home/gsang/miniconda3/envs/perceiver/bin/python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))


DEFAULT_TRAIN_H5 = (
    "/home/gsang/Projects/hnn_guided_dpf/data/"
    "reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/train_traj_40000-steps_1000.h5"
)
DEFAULT_EVAL_H5 = (
    "/home/gsang/Projects/hnn_guided_dpf/data/"
    "reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/val_traj_4000-steps_1000.h5"
)
DEFAULT_OUTPUT_ROOT = (
    "/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sb3_her_multiseed_suite_2026-04-17"
)


@dataclass
class JobSpec:
    name: str
    algorithm: str
    output_dir: Path
    seed: int
    use_her: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an automatic multi-seed SAC/TD3 vs SAC+HER/TD3+HER suite on the IID-uniform "
            "goal-conditioned Reacher benchmark."
        )
    )
    parser.add_argument("--algorithm", type=str, default="td3", choices=["sac", "td3"])
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu_ids", type=str, default="0,1,3")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--train_h5_path", type=str, default=DEFAULT_TRAIN_H5)
    parser.add_argument("--eval_h5_path", type=str, default=DEFAULT_EVAL_H5)
    parser.add_argument("--eval_task_ids", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--total_timesteps", type=int, default=300000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=500000)
    parser.add_argument("--eval_every_steps", type=int, default=10000)
    parser.add_argument("--save_every_steps", type=int, default=10000)
    parser.add_argument("--train_max_episode_steps", type=int, default=1000)
    parser.add_argument("--eval_max_episode_steps", type=int, default=5000)
    parser.add_argument("--stall_patience_steps", type=int, default=500)
    parser.add_argument("--goal_tolerance", type=float, default=0.01)
    parser.add_argument("--torque_scale", type=float, default=0.2)
    parser.add_argument("--random_future_target_min_initial_distance", type=float, default=0.1)
    parser.add_argument("--cross_traj_sampling_max_tries", type=int, default=128)
    parser.add_argument("--budget_steps", type=str, default="256,512,1000")
    parser.add_argument("--td3_action_noise_sigma", type=float, default=0.1)
    parser.add_argument("--progress_scale", type=float, default=10.0)
    parser.add_argument("--distance_scale", type=float, default=1.0)
    parser.add_argument("--action_l2_weight", type=float, default=0.01)
    parser.add_argument("--success_bonus", type=float, default=5.0)
    parser.add_argument("--her_n_sampled_goal", type=int, default=4)
    parser.add_argument("--her_goal_selection_strategy", type=str, default="future")
    parser.add_argument("--poll_interval_sec", type=int, default=20)
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def method_key(args: argparse.Namespace, use_her: bool) -> str:
    base = str(args.algorithm).lower()
    return f"{base}_her" if use_her else base


def method_label(args: argparse.Namespace, use_her: bool) -> str:
    base = str(args.algorithm).upper()
    return f"{base}+HER" if use_her else base


def suite_title(args: argparse.Namespace) -> str:
    base = str(args.algorithm).upper()
    return f"{base} vs {base}+HER Multi-Seed Suite"


def build_jobs(output_root: Path, seeds: list[int], args: argparse.Namespace) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    algorithm = str(args.algorithm).lower()
    for seed in seeds:
        jobs.append(
            JobSpec(
                name=f"{algorithm}_seed{seed}",
                algorithm=algorithm,
                output_dir=output_root / "runs" / f"{algorithm}_seed{seed}",
                seed=int(seed),
                use_her=False,
            )
        )
        jobs.append(
            JobSpec(
                name=f"{algorithm}_her_seed{seed}",
                algorithm=algorithm,
                output_dir=output_root / "runs" / f"{algorithm}_her_seed{seed}",
                seed=int(seed),
                use_her=True,
            )
        )
    return jobs


def train_cmd(job: JobSpec, args: argparse.Namespace, gpu_id: int) -> list[str]:
    cmd = [
        "/home/gsang/miniconda3/envs/perceiver/bin/python",
        "/home/gsang/Projects/hnn_guided_dpf/scripts/train/train_reacher_sb3.py",
        "--algorithm",
        str(job.algorithm),
        "--output_dir",
        str(job.output_dir),
        "--train_h5_path",
        str(args.train_h5_path),
        "--eval_h5_path",
        str(args.eval_h5_path),
        "--eval_task_ids",
        str(args.eval_task_ids),
        "--total_timesteps",
        str(int(args.total_timesteps)),
        "--learning_starts",
        str(int(args.learning_starts)),
        "--batch_size",
        str(int(args.batch_size)),
        "--buffer_size",
        str(int(args.buffer_size)),
        "--learning_rate",
        str(float(args.learning_rate)),
        "--eval_every_steps",
        str(int(args.eval_every_steps)),
        "--save_every_steps",
        str(int(args.save_every_steps)),
        "--budget_steps",
        str(args.budget_steps),
        "--train_max_episode_steps",
        str(int(args.train_max_episode_steps)),
        "--eval_max_episode_steps",
        str(int(args.eval_max_episode_steps)),
        "--stall_patience_steps",
        str(int(args.stall_patience_steps)),
        "--goal_tolerance",
        str(float(args.goal_tolerance)),
        "--torque_scale",
        str(float(args.torque_scale)),
        "--random_future_target_min_initial_distance",
        str(float(args.random_future_target_min_initial_distance)),
        "--cross_traj_sampling_max_tries",
        str(int(args.cross_traj_sampling_max_tries)),
        "--num_envs",
        str(int(args.num_envs)),
        "--seed",
        str(int(job.seed)),
        "--device",
        f"cuda:{int(gpu_id)}",
        "--progress_scale",
        str(float(args.progress_scale)),
        "--distance_scale",
        str(float(args.distance_scale)),
        "--action_l2_weight",
        str(float(args.action_l2_weight)),
        "--success_bonus",
        str(float(args.success_bonus)),
    ]
    if str(job.algorithm).lower() == "td3":
        cmd.extend(
            [
                "--td3_action_noise_sigma",
                str(float(args.td3_action_noise_sigma)),
            ]
        )
    if job.use_her:
        cmd.extend(
            [
                "--use_her",
                "--her_n_sampled_goal",
                str(int(args.her_n_sampled_goal)),
                "--her_goal_selection_strategy",
                str(args.her_goal_selection_strategy),
            ]
        )
    return cmd


def launch_job(job: JobSpec, args: argparse.Namespace, gpu_id: int, logs_dir: Path) -> subprocess.Popen:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{job.name}.log"
    cmd = train_cmd(job, args, gpu_id)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
        handle.write(f"START_UNIX={time.time():.6f}\n")
    handle = log_path.open("a", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        cwd="/home/gsang/Projects/hnn_guided_dpf",
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )


def collect_eval_rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for summary_path in sorted(run_dir.glob("evaluations/step_*/reacher_rl_policy_eval_summary.json")):
        payload = json.loads(summary_path.read_text())
        aggregate = payload["aggregate"]
        metadata = payload.get("metadata", {})
        rows.append(
            {
                "step": int(summary_path.parent.name.split("_")[1]),
                "summary_path": str(summary_path),
                "checkpoint_path": str(Path(payload["checkpoint_path"])),
                "success_rate": float(aggregate["success_rate"]),
                "best_goal_mean": float(aggregate["best_goal_distance"].get("mean", float("inf"))),
                "final_goal_mean": float(aggregate["final_goal_distance"].get("mean", float("inf"))),
                "best_goal_mean_at_256": float(
                    aggregate["budget_metrics"]["256"]["best_goal_distance"].get("mean", float("inf"))
                ),
                "wall_clock_to_eval_start_s": float(metadata.get("wall_clock_seconds", float("nan"))),
                "evaluation_wall_clock_s": float(metadata.get("evaluation_wall_clock_seconds", float("nan"))),
            }
        )
    return rows


def best_row(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No evaluation rows found.")
    return max(
        rows,
        key=lambda row: (
            float(row["success_rate"]),
            -float(row["best_goal_mean"]),
            -float(row["best_goal_mean_at_256"]),
            -int(row["step"]),
        ),
    )


def final_row(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No evaluation rows found.")
    return max(rows, key=lambda row: int(row["step"]))


def summarize_run(job: JobSpec, run_dir: Path, wall_clock_s: float | None) -> dict:
    rows = collect_eval_rows(run_dir)
    best = best_row(rows)
    final = final_row(rows)
    training_config_path = run_dir / "training_config.json"
    training_config = json.loads(training_config_path.read_text()) if training_config_path.exists() else {}
    payload = {
        "name": job.name,
        "algorithm": job.algorithm,
        "use_her": bool(job.use_her),
        "seed": int(job.seed),
        "run_dir": str(run_dir),
        "training_config": training_config,
        "best": best,
        "final": final,
        "run_wall_clock_s": None if wall_clock_s is None else float(wall_clock_s),
        "rows": rows,
    }
    write_json(run_dir / "seed_summary.json", payload)
    return payload


def aggregate_seed_summaries(seed_summaries: list[dict], section: str) -> dict:
    if not seed_summaries:
        return {}
    success = np.asarray([row[section]["success_rate"] for row in seed_summaries], dtype=np.float64)
    best_goal = np.asarray([row[section]["best_goal_mean"] for row in seed_summaries], dtype=np.float64)
    final_goal = np.asarray([row[section]["final_goal_mean"] for row in seed_summaries], dtype=np.float64)
    best_goal_256 = np.asarray([row[section]["best_goal_mean_at_256"] for row in seed_summaries], dtype=np.float64)
    return {
        "success_rate_mean": float(success.mean()),
        "success_rate_std": float(success.std()),
        "best_goal_mean_mean": float(best_goal.mean()),
        "best_goal_mean_std": float(best_goal.std()),
        "final_goal_mean_mean": float(final_goal.mean()),
        "final_goal_mean_std": float(final_goal.std()),
        "best_goal_mean_at_256_mean": float(best_goal_256.mean()),
        "best_goal_mean_at_256_std": float(best_goal_256.std()),
    }


def render_suite_summary(output_root: Path, seed_summaries: list[dict], suite_wall_clock_s: float) -> dict:
    algorithms = sorted({str(summary["algorithm"]).lower() for summary in seed_summaries})
    if len(algorithms) != 1:
        raise ValueError(f"Expected one algorithm in suite, got {algorithms}.")
    algorithm = algorithms[0]
    by_method = {
        algorithm: [summary for summary in seed_summaries if not summary["use_her"]],
        f"{algorithm}_her": [summary for summary in seed_summaries if summary["use_her"]],
    }
    payload = {
        "output_root": str(output_root),
        "suite_wall_clock_s": float(suite_wall_clock_s),
        "runs": seed_summaries,
        "aggregates": {
            method: {
                "best": aggregate_seed_summaries(rows, "best"),
                "final": aggregate_seed_summaries(rows, "final"),
            }
            for method, rows in by_method.items()
        },
    }
    write_json(output_root / "suite_summary.json", payload)

    lines = [
        f"# {algorithm.upper()} vs {algorithm.upper()}+HER Multi-Seed Suite",
        "",
        f"- output_root = `{output_root}`",
        f"- suite_wall_clock_s = `{suite_wall_clock_s:.1f}`",
        f"- suite_wall_clock_min = `{suite_wall_clock_s / 60.0:.2f}`",
        "",
        "## Per-Run Results",
        "",
        "| run | method | seed | best step | best success | best goal mean | final step | final success | final goal mean | run wall clock (s) | checkpoint |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in sorted(seed_summaries, key=lambda item: item["name"]):
        method = f"{algorithm.upper()}+HER" if summary["use_her"] else algorithm.upper()
        lines.append(
            "| {name} | {method} | {seed} | {best_step} | {best_success:.3f} | {best_goal:.6f} | "
            "{final_step} | {final_success:.3f} | {final_goal:.6f} | {runtime} | `{checkpoint}` |".format(
                name=summary["name"],
                method=method,
                seed=int(summary["seed"]),
                best_step=int(summary["best"]["step"]),
                best_success=float(summary["best"]["success_rate"]),
                best_goal=float(summary["best"]["best_goal_mean"]),
                final_step=int(summary["final"]["step"]),
                final_success=float(summary["final"]["success_rate"]),
                final_goal=float(summary["final"]["best_goal_mean"]),
                runtime=(
                    "n/a"
                    if summary["run_wall_clock_s"] is None
                    else f"{float(summary['run_wall_clock_s']):.1f}"
                ),
                checkpoint=summary["best"]["checkpoint_path"],
            )
        )

    lines.extend(["", "## Aggregates", ""])
    for method_key, label in (
        (algorithm, algorithm.upper()),
        (f"{algorithm}_her", f"{algorithm.upper()}+HER"),
    ):
        best = payload["aggregates"][method_key]["best"]
        final = payload["aggregates"][method_key]["final"]
        lines.extend(
            [
                f"### {label}",
                "",
                f"- best success_rate mean±std = `{best.get('success_rate_mean', float('nan')):.3f} ± {best.get('success_rate_std', float('nan')):.3f}`",
                f"- best goal mean mean±std = `{best.get('best_goal_mean_mean', float('nan')):.6f} ± {best.get('best_goal_mean_std', float('nan')):.6f}`",
                f"- final success_rate mean±std = `{final.get('success_rate_mean', float('nan')):.3f} ± {final.get('success_rate_std', float('nan')):.3f}`",
                f"- final goal mean mean±std = `{final.get('best_goal_mean_mean', float('nan')):.6f} ± {final.get('best_goal_mean_std', float('nan')):.6f}`",
                "",
            ]
        )

    write_text(output_root / "suite_summary.md", "\n".join(lines) + "\n")
    return payload


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    gpu_ids = parse_int_list(args.gpu_ids)
    seeds = parse_int_list(args.seeds)
    if not gpu_ids:
        raise ValueError("At least one GPU id is required.")
    if not seeds:
        raise ValueError("At least one seed is required.")

    suite_config = dict(vars(args))
    suite_config["gpu_ids"] = gpu_ids
    suite_config["seeds"] = seeds
    write_json(output_root / "suite_config.json", suite_config)

    jobs = build_jobs(output_root, seeds, args)
    logs_dir = output_root / "logs"
    progress_path = output_root / "suite_progress.json"
    progress_md_path = output_root / "suite_progress.md"

    queue = list(jobs)
    active: dict[int, tuple[JobSpec, subprocess.Popen, float]] = {}
    completed: list[dict] = []
    failed: list[dict] = []
    suite_start = time.time()

    def flush_progress() -> None:
        payload = {
            "output_root": str(output_root),
            "suite_started_at_unix": float(suite_start),
            "suite_wall_clock_s_so_far": float(time.time() - suite_start),
            "queue": [job.name for job in queue],
            "active": {
                str(gpu_id): {
                    "job_name": job.name,
                    "gpu_id": int(gpu_id),
                    "seed": int(job.seed),
                    "use_her": bool(job.use_her),
                    "started_at_unix": float(start_time),
                    "runtime_s_so_far": float(time.time() - start_time),
                }
                for gpu_id, (job, _proc, start_time) in active.items()
            },
            "completed": completed,
            "failed": failed,
        }
        write_json(progress_path, payload)

        lines = [
            f"# {suite_title(args)} Progress",
            "",
            f"- output_root = `{output_root}`",
            f"- suite_wall_clock_s_so_far = `{time.time() - suite_start:.1f}`",
            "",
            "## Active",
            "",
        ]
        if active:
            for gpu_id, (job, _proc, start_time) in sorted(active.items()):
                lines.append(
                    f"- gpu {gpu_id}: `{job.name}` (seed={job.seed}, use_her={job.use_her}) "
                    f"runtime={time.time() - start_time:.1f}s"
                )
        else:
            lines.append("- none")

        lines.extend(["", "## Queue", ""])
        if queue:
            for job in queue:
                lines.append(f"- `{job.name}`")
        else:
            lines.append("- empty")

        lines.extend(["", "## Completed", ""])
        if completed:
            for summary in completed:
                lines.append(
                    f"- `{summary['name']}`: best_success={summary['best']['success_rate']:.3f}, "
                    f"best_goal_mean={summary['best']['best_goal_mean']:.6f}, "
                    f"runtime={summary['run_wall_clock_s']:.1f}s"
                )
        else:
            lines.append("- none")

        if failed:
            lines.extend(["", "## Failed", ""])
            for item in failed:
                lines.append(f"- `{item['name']}` returncode={item['returncode']}")

        write_text(progress_md_path, "\n".join(lines) + "\n")

    flush_progress()

    poll_interval = max(5, int(args.poll_interval_sec))
    while queue or active:
        free_gpus = [gpu_id for gpu_id in gpu_ids if gpu_id not in active]
        while queue and free_gpus:
            gpu_id = free_gpus.pop(0)
            job = queue.pop(0)
            proc = launch_job(job, args, gpu_id, logs_dir)
            active[gpu_id] = (job, proc, time.time())
            flush_progress()

        time.sleep(poll_interval)

        for gpu_id, (job, proc, start_time) in list(active.items()):
            returncode = proc.poll()
            if returncode is None:
                continue
            active.pop(gpu_id)
            wall_clock_s = time.time() - start_time
            if returncode != 0:
                failed.append({"name": job.name, "returncode": int(returncode), "gpu_id": int(gpu_id)})
                flush_progress()
                raise RuntimeError(f"{job.name} failed with return code {returncode}.")
            completed.append(summarize_run(job, job.output_dir, wall_clock_s))
            flush_progress()

    suite_payload = render_suite_summary(output_root, completed, time.time() - suite_start)
    flush_progress()
    write_json(output_root / "pipeline_state.json", {"status": "completed", **suite_payload})


if __name__ == "__main__":
    main()
