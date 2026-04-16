#!/home/gsang/miniconda3/envs/perceiver/bin/python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time

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
    "/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/sb3_auto_pipeline_2026-04-15"
)


@dataclass
class JobSpec:
    name: str
    algorithm: str
    output_dir: Path
    total_timesteps: int
    learning_rate: float
    num_envs: int
    learning_starts: int
    reward_overrides: dict[str, float]
    extra_args: dict[str, str | int | float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatic overnight pipeline for SB3 SAC/TD3 tuning and full training "
            "on the IID-uniform Reacher benchmark."
        )
    )
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu_ids", type=str, default="0,1,3")
    parser.add_argument("--train_h5_path", type=str, default=DEFAULT_TRAIN_H5)
    parser.add_argument("--eval_h5_path", type=str, default=DEFAULT_EVAL_H5)
    parser.add_argument("--eval_task_ids", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--sac_tune_timesteps", type=int, default=30000)
    parser.add_argument("--td3_tune_timesteps", type=int, default=30000)
    parser.add_argument("--full_timesteps", type=int, default=300000)
    parser.add_argument("--tune_num_envs", type=int, default=4)
    parser.add_argument("--full_num_envs", type=int, default=8)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--existing_sac_tune_dir",
        type=str,
        default="",
        help="Optional existing SAC tune directory to reuse instead of rerunning SAC tune.",
    )
    parser.add_argument("--poll_interval_sec", type=int, default=30)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def parse_gpu_ids(text: str) -> list[int]:
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def base_train_cmd(
    *,
    algorithm: str,
    output_dir: Path,
    total_timesteps: int,
    learning_rate: float,
    num_envs: int,
    learning_starts: int,
    reward_overrides: dict[str, float],
    extra_args: dict[str, str | int | float],
    args: argparse.Namespace,
    gpu_id: int,
) -> list[str]:
    cmd = [
        "/home/gsang/miniconda3/envs/perceiver/bin/python",
        "/home/gsang/Projects/hnn_guided_dpf/scripts/train/train_reacher_sb3.py",
        "--algorithm",
        str(algorithm),
        "--output_dir",
        str(output_dir),
        "--train_h5_path",
        str(args.train_h5_path),
        "--eval_h5_path",
        str(args.eval_h5_path),
        "--eval_task_ids",
        str(args.eval_task_ids),
        "--total_timesteps",
        str(int(total_timesteps)),
        "--learning_starts",
        str(int(learning_starts)),
        "--batch_size",
        str(int(args.batch_size)),
        "--buffer_size",
        str(int(args.buffer_size)),
        "--learning_rate",
        str(float(learning_rate)),
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
        str(int(num_envs)),
        "--seed",
        str(int(args.seed)),
        "--device",
        f"cuda:{int(gpu_id)}",
    ]
    for key, value in reward_overrides.items():
        cmd.extend([f"--{key}", str(value)])
    for key, value in extra_args.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def launch_job(job: JobSpec, args: argparse.Namespace, gpu_id: int, stage_dir: Path) -> subprocess.Popen:
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / f"{job.name}.log"
    cmd = base_train_cmd(
        algorithm=job.algorithm,
        output_dir=job.output_dir,
        total_timesteps=job.total_timesteps,
        learning_rate=job.learning_rate,
        num_envs=job.num_envs,
        learning_starts=job.learning_starts,
        reward_overrides=job.reward_overrides,
        extra_args=job.extra_args,
        args=args,
        gpu_id=gpu_id,
    )
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
    handle = log_path.open("a", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        cwd="/home/gsang/Projects/hnn_guided_dpf",
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )


def run_parallel_jobs(
    *,
    jobs: list[JobSpec],
    gpu_ids: list[int],
    stage_dir: Path,
    args: argparse.Namespace,
) -> dict[str, dict]:
    pending = list(jobs)
    active: dict[int, tuple[JobSpec, subprocess.Popen]] = {}
    results: dict[str, dict] = {}
    poll_interval = max(5, int(args.poll_interval_sec))

    while pending or active:
        free_gpus = [gpu for gpu in gpu_ids if gpu not in active]
        while free_gpus and pending:
            gpu_id = free_gpus.pop(0)
            job = pending.pop(0)
            proc = launch_job(job, args, gpu_id, stage_dir)
            active[gpu_id] = (job, proc)
            results[job.name] = {
                "status": "running",
                "gpu_id": int(gpu_id),
                "output_dir": str(job.output_dir),
            }
            write_json(stage_dir / "stage_progress.json", results)

        time.sleep(poll_interval)

        for gpu_id, (job, proc) in list(active.items()):
            returncode = proc.poll()
            if returncode is None:
                continue
            results[job.name] = {
                "status": "completed" if returncode == 0 else "failed",
                "gpu_id": int(gpu_id),
                "returncode": int(returncode),
                "output_dir": str(job.output_dir),
            }
            active.pop(gpu_id)
            write_json(stage_dir / "stage_progress.json", results)
            if returncode != 0:
                raise RuntimeError(f"Job {job.name} failed with return code {returncode}.")

    return results


def collect_eval_summaries(tune_root: Path) -> list[dict]:
    summaries: list[dict] = []
    for summary_path in sorted(tune_root.glob("*/evaluations/step_*/reacher_rl_policy_eval_summary.json")):
        payload = json.loads(summary_path.read_text())
        aggregate = payload["aggregate"]
        metadata = payload.get("metadata", {})
        summaries.append(
            {
                "variant": summary_path.parents[2].name,
                "step": int(summary_path.parent.name.split("_")[1]),
                "summary_path": str(summary_path),
                "checkpoint_path": str(Path(payload["checkpoint_path"])),
                "success_rate": float(aggregate["success_rate"]),
                "best_goal_mean": float(aggregate["best_goal_distance"].get("mean", float("inf"))),
                "final_goal_mean": float(aggregate["final_goal_distance"].get("mean", float("inf"))),
                "best_goal_mean_at_256": float(
                    aggregate["budget_metrics"]["256"]["best_goal_distance"].get("mean", float("inf"))
                ),
                "best_goal_mean_at_1000": float(
                    aggregate["budget_metrics"]["1000"]["best_goal_distance"].get("mean", float("inf"))
                ),
                "wall_clock_seconds": float(metadata.get("wall_clock_seconds", float("nan"))),
            }
        )
    return summaries


def best_summary_row(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No evaluation summaries were found.")
    return max(
        rows,
        key=lambda row: (
            float(row["success_rate"]),
            -float(row["best_goal_mean"]),
            -float(row["best_goal_mean_at_256"]),
            -int(row["step"]),
        ),
    )


def summarize_tune_root(tune_root: Path, stage_name: str) -> dict:
    rows = collect_eval_summaries(tune_root)
    best_row = best_summary_row(rows)
    payload = {
        "stage_name": stage_name,
        "tune_root": str(tune_root),
        "rows": rows,
        "best": best_row,
    }
    write_json(tune_root / f"{stage_name}_summary.json", payload)

    lines = [
        f"# {stage_name}",
        "",
        f"- tune_root = `{tune_root}`",
        "",
        "| variant | step | success_rate | best_goal_mean | final_goal_mean | best_goal_mean@256 | wall_clock_to_eval_start_s | checkpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {step} | {success_rate:.3f} | {best_goal_mean:.6f} | "
            "{final_goal_mean:.6f} | {best_goal_mean_at_256:.6f} | {wall_clock_seconds:.3f} | "
            "`{checkpoint_path}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Best",
            "",
            f"- variant = `{best_row['variant']}`",
            f"- step = `{best_row['step']}`",
            f"- success_rate = `{best_row['success_rate']:.3f}`",
            f"- best_goal_mean = `{best_row['best_goal_mean']:.6f}`",
            f"- final_goal_mean = `{best_row['final_goal_mean']:.6f}`",
            f"- best_goal_mean@256 = `{best_row['best_goal_mean_at_256']:.6f}`",
            f"- wall_clock_to_eval_start_s = `{best_row['wall_clock_seconds']:.3f}`",
            f"- checkpoint = `{best_row['checkpoint_path']}`",
        ]
    )
    write_text(tune_root / f"{stage_name}_summary.md", "\n".join(lines) + "\n")
    return payload


def sac_tune_jobs(root: Path, args: argparse.Namespace) -> list[JobSpec]:
    return [
        JobSpec(
            name="base_lr3e4_ls1000_30k",
            algorithm="sac",
            output_dir=root / "base_lr3e4_ls1000_30k",
            total_timesteps=int(args.sac_tune_timesteps),
            learning_rate=3e-4,
            num_envs=int(args.tune_num_envs),
            learning_starts=int(args.learning_starts),
            reward_overrides={},
            extra_args={},
        ),
        JobSpec(
            name="lowlr1e4_ls1000_30k",
            algorithm="sac",
            output_dir=root / "lowlr1e4_ls1000_30k",
            total_timesteps=int(args.sac_tune_timesteps),
            learning_rate=1e-4,
            num_envs=int(args.tune_num_envs),
            learning_starts=int(args.learning_starts),
            reward_overrides={},
            extra_args={},
        ),
        JobSpec(
            name="reward20_bonus10_lr3e4_ls1000_30k",
            algorithm="sac",
            output_dir=root / "reward20_bonus10_lr3e4_ls1000_30k",
            total_timesteps=int(args.sac_tune_timesteps),
            learning_rate=3e-4,
            num_envs=int(args.tune_num_envs),
            learning_starts=int(args.learning_starts),
            reward_overrides={"progress_scale": 20, "success_bonus": 10},
            extra_args={},
        ),
    ]


def td3_tune_jobs(root: Path, args: argparse.Namespace) -> list[JobSpec]:
    return [
        JobSpec(
            name="base_lr3e4_noise0p1_30k",
            algorithm="td3",
            output_dir=root / "base_lr3e4_noise0p1_30k",
            total_timesteps=int(args.td3_tune_timesteps),
            learning_rate=3e-4,
            num_envs=int(args.tune_num_envs),
            learning_starts=int(args.learning_starts),
            reward_overrides={},
            extra_args={"td3_action_noise_sigma": 0.1},
        ),
        JobSpec(
            name="lowlr1e4_noise0p1_30k",
            algorithm="td3",
            output_dir=root / "lowlr1e4_noise0p1_30k",
            total_timesteps=int(args.td3_tune_timesteps),
            learning_rate=1e-4,
            num_envs=int(args.tune_num_envs),
            learning_starts=int(args.learning_starts),
            reward_overrides={},
            extra_args={"td3_action_noise_sigma": 0.1},
        ),
        JobSpec(
            name="base_lr3e4_noise0p2_30k",
            algorithm="td3",
            output_dir=root / "base_lr3e4_noise0p2_30k",
            total_timesteps=int(args.td3_tune_timesteps),
            learning_rate=3e-4,
            num_envs=int(args.tune_num_envs),
            learning_starts=int(args.learning_starts),
            reward_overrides={},
            extra_args={"td3_action_noise_sigma": 0.2},
        ),
    ]


def full_job_from_best(
    *,
    algorithm: str,
    best: dict,
    output_dir: Path,
    args: argparse.Namespace,
) -> JobSpec:
    training_config_path = Path(best["summary_path"]).parents[2] / "training_config.json"
    training_config = json.loads(training_config_path.read_text())
    reward_cfg = {}
    for key in ("progress_scale", "distance_scale", "action_l2_weight", "success_bonus"):
        reward_cfg[key] = float(training_config[key])
    extra_args = {}
    if algorithm == "td3":
        extra_args["td3_action_noise_sigma"] = float(training_config["td3_action_noise_sigma"])
    return JobSpec(
        name=f"{algorithm}_full",
        algorithm=algorithm,
        output_dir=output_dir,
        total_timesteps=int(args.full_timesteps),
        learning_rate=float(training_config["learning_rate"]),
        num_envs=int(args.full_num_envs),
        learning_starts=int(training_config["learning_starts"]),
        reward_overrides=reward_cfg,
        extra_args=extra_args,
    )


def run_single_job(job: JobSpec, gpu_id: int, args: argparse.Namespace, stage_dir: Path) -> None:
    proc = launch_job(job, args, gpu_id, stage_dir)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"{job.name} failed with return code {ret}.")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        raise ValueError("At least one GPU id is required.")

    pipeline_state = {
        "output_root": str(output_root),
        "gpu_ids": gpu_ids,
        "started_at_unix": time.time(),
        "stages": {},
    }
    write_json(output_root / "pipeline_state.json", pipeline_state)

    sac_tune_root = Path(args.existing_sac_tune_dir) if args.existing_sac_tune_dir else output_root / "sac_tune"
    if args.existing_sac_tune_dir:
        sac_summary = summarize_tune_root(sac_tune_root, "sac_tune")
    else:
        stage_dir = output_root / "logs" / "sac_tune"
        run_parallel_jobs(
            jobs=sac_tune_jobs(sac_tune_root, args),
            gpu_ids=gpu_ids,
            stage_dir=stage_dir,
            args=args,
        )
        sac_summary = summarize_tune_root(sac_tune_root, "sac_tune")
    pipeline_state["stages"]["sac_tune"] = sac_summary["best"]
    write_json(output_root / "pipeline_state.json", pipeline_state)

    sac_full_gpu = gpu_ids[0]
    sac_full_job = full_job_from_best(
        algorithm="sac",
        best=sac_summary["best"],
        output_dir=output_root / "sac_full",
        args=args,
    )
    sac_full_log_dir = output_root / "logs" / "sac_full"
    sac_full_proc = launch_job(sac_full_job, args, sac_full_gpu, sac_full_log_dir)

    remaining_gpus = [gpu for gpu in gpu_ids if gpu != sac_full_gpu]
    td3_tune_root = output_root / "td3_tune"
    td3_stage_dir = output_root / "logs" / "td3_tune"
    if remaining_gpus:
        run_parallel_jobs(
            jobs=td3_tune_jobs(td3_tune_root, args),
            gpu_ids=remaining_gpus,
            stage_dir=td3_stage_dir,
            args=args,
        )
    else:
        sac_ret = sac_full_proc.wait()
        if sac_ret != 0:
            raise RuntimeError(f"SAC full training failed with return code {sac_ret}.")
        run_parallel_jobs(
            jobs=td3_tune_jobs(td3_tune_root, args),
            gpu_ids=[sac_full_gpu],
            stage_dir=td3_stage_dir,
            args=args,
        )
    td3_summary = summarize_tune_root(td3_tune_root, "td3_tune")
    pipeline_state["stages"]["td3_tune"] = td3_summary["best"]
    write_json(output_root / "pipeline_state.json", pipeline_state)

    td3_full_gpu = remaining_gpus[0] if remaining_gpus else sac_full_gpu
    td3_full_job = full_job_from_best(
        algorithm="td3",
        best=td3_summary["best"],
        output_dir=output_root / "td3_full",
        args=args,
    )
    td3_full_log_dir = output_root / "logs" / "td3_full"
    td3_full_proc = launch_job(td3_full_job, args, td3_full_gpu, td3_full_log_dir)

    if remaining_gpus:
        sac_ret = sac_full_proc.wait()
        if sac_ret != 0:
            raise RuntimeError(f"SAC full training failed with return code {sac_ret}.")
    td3_ret = td3_full_proc.wait()
    if td3_ret != 0:
        raise RuntimeError(f"TD3 full training failed with return code {td3_ret}.")

    pipeline_state["finished_at_unix"] = time.time()
    pipeline_state["status"] = "completed"
    write_json(output_root / "pipeline_state.json", pipeline_state)


if __name__ == "__main__":
    main()
