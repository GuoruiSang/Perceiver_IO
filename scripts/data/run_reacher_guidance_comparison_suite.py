#!/home/gsang/miniconda3/envs/perceiver/bin/python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "data" / "run_reacher_goal_prefix_expansion.py"
PYTHON_BIN = Path("/home/gsang/miniconda3/envs/perceiver/bin/python")

DEFAULT_DPF_CHECKPOINT = (
    "/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/"
    "dpf_exploration_iid_uniform_len1000_v1/"
    "trajectory_dpf_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone"
    "&DecoderAttentions_cond-concat_torque_in_state_layout-shifted_tau_qpos-raw:epoch=2999_val_loss:val_loss=0.0983.ckpt"
)
DEFAULT_HNN_CHECKPOINT = (
    "/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/"
    "hnn_exploration_iid_uniform_len1000_v1/StructuredHNN-ReacherExploration-IID-epoch-epoch=999.ckpt"
)
DEFAULT_H5_PATH = (
    "/home/gsang/Projects/hnn_guided_dpf/data/"
    "reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/val_traj_4000-steps_1000.h5"
)
DEFAULT_PLOTS_ROOT = Path("/home/gsang/Projects/hnn_guided_dpf/plots")
DEFAULT_TASK_MODE = "validation_random_source_random_target_across_trajs"
DEFAULT_TASK_IDS = list(range(10))
DEFAULT_GPU_IDS = [0, 1, 3]
DEFAULT_TARGET_ALPHA_GRID = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2]
DEFAULT_TARGET_NORMS = ["l1", "l2", "linf"]
DEFAULT_CANDIDATE_COUNTS = [1, 8]


@dataclass(frozen=True)
class JobSpec:
    name: str
    output_dir: Path
    command: list[str]
    metadata: dict[str, object]


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def parse_str_list(text: str) -> list[str]:
    values: list[str] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(token)
    return values


def parse_target_alpha_map(text: str) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"Invalid fixed target-alpha token {token!r}. "
                "Use comma-separated norm=value entries such as l1=1e-4,l2=1e-4,linf=1e-4."
            )
        norm, value = token.split("=", 1)
        norm = norm.strip().lower()
        if norm == "l_infinite":
            norm = "linf"
        mapping[norm] = float(value.strip())
    return mapping


def format_alpha(alpha: float) -> str:
    sci = f"{alpha:.0e}"
    return sci.replace("+0", "").replace("+", "")


def display_norm(norm: str) -> str:
    if norm == "linf":
        return "l_infinite"
    return norm


def ranking_tuple(summary: dict[str, object]) -> tuple[float, float, float]:
    aggregate = summary["aggregate"]
    success_rate = float(aggregate["success_rate"])
    best_goal_mean = float(aggregate["best_goal_distance"]["mean"])
    final_goal_mean = float(aggregate["final_goal_distance"]["mean"])
    return (-success_rate, best_goal_mean, final_goal_mean)


def read_summary(summary_path: Path) -> dict[str, object]:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def summary_is_complete(summary: dict[str, object]) -> bool:
    if "is_complete" in summary:
        return bool(summary["is_complete"])
    task_ids = summary.get("task_ids") or []
    num_total_tasks = int(summary.get("num_total_tasks") or len(task_ids))
    rows = summary.get("rows") or []
    return num_total_tasks > 0 and len(rows) >= num_total_tasks


def compact_summary(summary: dict[str, object]) -> dict[str, object]:
    return {
        "aggregate": summary["aggregate"],
        "task_ids": summary.get("task_ids"),
        "task_mode": summary.get("task_mode"),
        "num_total_tasks": summary.get("num_total_tasks"),
        "num_completed_tasks": summary.get("num_completed_tasks"),
        "completed_task_ids": summary.get("completed_task_ids"),
        "is_complete": summary.get("is_complete"),
        "num_candidates": summary.get("num_candidates"),
        "lookahead_steps": summary.get("lookahead_steps"),
        "recent_prefix_cap": summary.get("recent_prefix_cap"),
        "reset_window_time_indices": summary.get("reset_window_time_indices"),
        "stall_patience_steps": summary.get("stall_patience_steps"),
        "target_guidance_alpha": summary.get("target_guidance_alpha"),
        "target_guidance_norm": summary.get("target_guidance_norm"),
        "target_guidance_use_time_weights": summary.get("target_guidance_use_time_weights"),
        "hnn_checkpoint_path": summary.get("hnn_checkpoint_path"),
        "guidance_method": summary.get("guidance_method"),
        "alpha_q": summary.get("alpha_q"),
        "alpha_p": summary.get("alpha_p"),
        "guidance_trust_lambda": summary.get("guidance_trust_lambda"),
        "guidance_normalize_grad": summary.get("guidance_normalize_grad"),
        "guidance_joint_update": summary.get("guidance_joint_update"),
        "guidance_order": summary.get("guidance_order"),
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def collect_sweep_entries(results: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for result in results.values():
        metadata = result["job"].metadata
        if metadata.get("stage") != "sweep":
            continue
        entries.append(
            {
                "target_norm": str(metadata["target_norm"]),
                "target_alpha": float(metadata["target_alpha"]),
                "output_dir": str(result["job"].output_dir),
                "summary": compact_summary(result["summary"]),
                "summary_path": result["summary_path"],
            }
        )
    entries.sort(key=lambda entry: (str(entry["target_norm"]), float(entry["target_alpha"])))
    return entries


def best_target_alphas_from_entries(
    sweep_entries: list[dict[str, object]],
    *,
    target_norms: list[str],
) -> dict[str, float]:
    best: dict[str, float] = {}
    for norm in target_norms:
        norm_entries = [entry for entry in sweep_entries if entry["target_norm"] == norm]
        if not norm_entries:
            continue
        norm_entries.sort(key=lambda entry: ranking_tuple({"aggregate": entry["summary"]["aggregate"]}))
        best[norm] = float(norm_entries[0]["target_alpha"])
    return best


def collect_final_rows(
    *,
    candidate_count: int,
    final_results: dict[str, dict[str, object]],
    target_norms: list[str],
    best_target_alpha_by_norm: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    key = f"cand{candidate_count}_unguided"
    if key in final_results:
        rows.append(
            {
                "label": "unguided",
                "summary": compact_summary(final_results[key]["summary"]),
                "output_dir": str(final_results[key]["job"].output_dir),
            }
        )

    for norm in target_norms:
        key = f"cand{candidate_count}_target_{norm}"
        if key in final_results:
            entry = {
                "label": f"target_guided_{display_norm(norm)}",
                "summary": compact_summary(final_results[key]["summary"]),
                "output_dir": str(final_results[key]["job"].output_dir),
            }
            if norm in best_target_alpha_by_norm:
                entry["selected_alpha"] = float(best_target_alpha_by_norm[norm])
            rows.append(entry)

    key = f"cand{candidate_count}_hnn"
    if key in final_results:
        rows.append(
            {
                "label": "hnn_guided",
                "summary": compact_summary(final_results[key]["summary"]),
                "output_dir": str(final_results[key]["job"].output_dir),
            }
        )

    for norm in target_norms:
        key = f"cand{candidate_count}_target_{norm}_then_hnn"
        if key in final_results:
            entry = {
                "label": f"target_guided_{display_norm(norm)}_plus_hnn",
                "summary": compact_summary(final_results[key]["summary"]),
                "output_dir": str(final_results[key]["job"].output_dir),
            }
            if norm in best_target_alpha_by_norm:
                entry["selected_alpha"] = float(best_target_alpha_by_norm[norm])
            rows.append(entry)

    rows.sort(key=lambda row: ranking_tuple({"aggregate": row["summary"]["aggregate"]}))
    return rows


def write_candidate_progress(
    *,
    candidate_dir: Path,
    candidate_count: int,
    target_norms: list[str],
    sweep_results: dict[str, dict[str, object]],
    final_results: dict[str, dict[str, object]],
    best_target_alpha_by_norm: dict[str, float],
    skip_sweep: bool,
) -> None:
    sweep_entries = collect_sweep_entries(sweep_results)
    derived_best_alphas = dict(best_target_alpha_by_norm)
    for norm, alpha in best_target_alphas_from_entries(sweep_entries, target_norms=target_norms).items():
        derived_best_alphas.setdefault(norm, alpha)
    final_rows = collect_final_rows(
        candidate_count=candidate_count,
        final_results=final_results,
        target_norms=target_norms,
        best_target_alpha_by_norm=derived_best_alphas,
    )

    payload = {
        "candidate_count": int(candidate_count),
        "is_complete": False,
        "best_target_alpha_by_norm": derived_best_alphas,
        "sweep_entries": sweep_entries,
        "final_rows": final_rows,
        "num_completed_sweep_jobs": int(len(sweep_entries)),
        "num_completed_final_jobs": int(len(final_rows)),
    }
    write_json(candidate_dir / "suite_progress.json", payload)

    lines = [
        f"# Candidate Count = {candidate_count} Progress",
        "",
        f"- completed_sweep_jobs = `{len(sweep_entries)}`",
        f"- completed_final_jobs = `{len(final_rows)}`",
        "",
        "## Target Alpha Selection Progress",
        "",
    ]
    if sweep_entries:
        lines.extend([make_sweep_table(sweep_entries), ""])
    elif skip_sweep:
        lines.extend(["Sweep skipped for this run.", ""])
    else:
        lines.extend(["No completed sweep jobs yet.", ""])

    if derived_best_alphas:
        lines.append("Current best target alphas:")
        for norm in target_norms:
            if norm in derived_best_alphas:
                lines.append(f"- `{display_norm(norm)}` -> `{derived_best_alphas[norm]:.0e}`")
        lines.append("")

    lines.extend(["## Final Comparison Progress", ""])
    if final_rows:
        lines.extend([make_final_table(final_rows), ""])
    else:
        lines.extend(["No completed final-comparison jobs yet.", ""])

    (candidate_dir / "suite_progress.md").write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fair overnight Reacher guidance comparison suite. "
            "For each candidate-count pass, sweep 5 target-guidance alphas per norm, "
            "select the best alpha on the fixed 10-task benchmark, then compare "
            "unguided vs target-guided vs HNN-guided vs combined target+HNN."
        )
    )
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_DPF_CHECKPOINT)
    parser.add_argument("--hnn_checkpoint_path", type=str, default=DEFAULT_HNN_CHECKPOINT)
    parser.add_argument("--h5_path", type=str, default=DEFAULT_H5_PATH)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DEFAULT_PLOTS_ROOT / f"reacher_guidance_compare_suite_{date.today().isoformat()}"),
    )
    parser.add_argument("--task_mode", type=str, default=DEFAULT_TASK_MODE)
    parser.add_argument("--task_ids", type=str, default=",".join(str(x) for x in DEFAULT_TASK_IDS))
    parser.add_argument("--gpu_ids", type=str, default=",".join(str(x) for x in DEFAULT_GPU_IDS))
    parser.add_argument(
        "--candidate_counts",
        type=str,
        default=",".join(str(x) for x in DEFAULT_CANDIDATE_COUNTS),
        help="Run the full suite sequentially for these candidate counts, e.g. 1,8.",
    )
    parser.add_argument(
        "--target_alpha_grid",
        type=str,
        default=",".join(format_alpha(x) for x in DEFAULT_TARGET_ALPHA_GRID),
        help="Comma-separated target-guidance alphas to sweep for each norm.",
    )
    parser.add_argument(
        "--target_norms",
        type=str,
        default=",".join(DEFAULT_TARGET_NORMS),
        help="Comma-separated target norms: l1,l2,linf.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_diffusion_steps", type=int, default=20)
    parser.add_argument("--lookahead_steps", type=int, default=256)
    parser.add_argument("--recent_prefix_cap", type=int, default=64)
    parser.add_argument("--stall_patience_steps", type=int, default=500)
    parser.add_argument("--goal_tolerance", type=float, default=0.01)
    parser.add_argument("--random_future_target_min_initial_distance", type=float, default=0.1)
    parser.add_argument("--cross_traj_sampling_max_tries", type=int, default=128)
    parser.add_argument("--max_sampling_retries", type=int, default=3)
    parser.add_argument("--retry_improvement_margin", type=float, default=1e-3)
    parser.add_argument("--target_guidance_time_power", type=float, default=2.0)
    parser.add_argument(
        "--target_guidance_normalize_grad",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--target_guidance_use_time_weights",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--guidance_method", type=str, default="strategy2", choices=["strategy1", "strategy2"])
    parser.add_argument("--alpha_q", type=float, default=1e-3)
    parser.add_argument("--alpha_p", type=float, default=1e-3)
    parser.add_argument("--guidance_trust_lambda", type=float, default=0.0)
    parser.add_argument(
        "--guidance_normalize_grad",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--guidance_joint_update",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--combined_guidance_order",
        type=str,
        default="target_then_hnn",
        choices=["target_then_hnn", "hnn_then_target"],
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip jobs whose summary JSON already exists.",
    )
    parser.add_argument(
        "--skip_sweep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the target-guidance sweep and use fixed target alphas directly.",
    )
    parser.add_argument(
        "--fixed_target_alphas",
        type=str,
        default="",
        help=(
            "Comma-separated norm=value mapping used when --skip_sweep is enabled, "
            "for example: l1=1e-4,l2=1e-4,linf=1e-4"
        ),
    )
    return parser


class SuiteLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._lock = threading.Lock()

    def log(self, text: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}"
        with self._lock:
            print(line, flush=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def common_base_command(args: argparse.Namespace, *, output_dir: Path, num_candidates: int) -> list[str]:
    command = [
        str(PYTHON_BIN),
        str(RUNNER_PATH),
        "--checkpoint_path",
        str(args.checkpoint_path),
        "--h5_path",
        str(args.h5_path),
        "--output_dir",
        str(output_dir),
        "--task_mode",
        str(args.task_mode),
        "--task_ids",
        str(args.task_ids),
        "--num_candidates",
        str(num_candidates),
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
    return command


def maybe_add_target_args(command: list[str], args: argparse.Namespace, *, alpha: float, norm: str) -> None:
    command.extend(
        [
            "--target_guidance_alpha",
            str(alpha),
            "--target_guidance_time_power",
            str(float(args.target_guidance_time_power)),
            "--target_guidance_norm",
            str(norm),
        ]
    )
    command.append(
        "--target_guidance_normalize_grad"
        if bool(args.target_guidance_normalize_grad)
        else "--no-target_guidance_normalize_grad"
    )
    command.append(
        "--target_guidance_use_time_weights"
        if bool(args.target_guidance_use_time_weights)
        else "--no-target_guidance_use_time_weights"
    )


def maybe_add_hnn_args(command: list[str], args: argparse.Namespace) -> None:
    command.extend(
        [
            "--hnn_checkpoint_path",
            str(args.hnn_checkpoint_path),
            "--guidance_method",
            str(args.guidance_method),
            "--alpha_q",
            str(float(args.alpha_q)),
            "--alpha_p",
            str(float(args.alpha_p)),
            "--guidance_trust_lambda",
            str(float(args.guidance_trust_lambda)),
        ]
    )
    command.append(
        "--guidance_normalize_grad"
        if bool(args.guidance_normalize_grad)
        else "--no-guidance_normalize_grad"
    )
    command.append(
        "--guidance_joint_update"
        if bool(args.guidance_joint_update)
        else "--no-guidance_joint_update"
    )


def build_sweep_jobs(args: argparse.Namespace, *, candidate_count: int, suite_dir: Path) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    sweep_root = suite_dir / "sweep"
    for norm in parse_str_list(args.target_norms):
        for alpha in parse_float_list(args.target_alpha_grid):
            alpha_tag = format_alpha(alpha).replace("-", "m")
            output_dir = sweep_root / f"target_{norm}_alpha_{alpha_tag}"
            command = common_base_command(args, output_dir=output_dir, num_candidates=candidate_count)
            maybe_add_target_args(command, args, alpha=alpha, norm=norm)
            jobs.append(
                JobSpec(
                    name=f"cand{candidate_count}_target_{norm}_alpha_{alpha_tag}",
                    output_dir=output_dir,
                    command=command,
                    metadata={
                        "stage": "sweep",
                        "candidate_count": int(candidate_count),
                        "target_norm": str(norm),
                        "target_alpha": float(alpha),
                        "uses_hnn": False,
                    },
                )
            )
    return jobs


def build_final_jobs(
    args: argparse.Namespace,
    *,
    candidate_count: int,
    suite_dir: Path,
    best_target_alpha_by_norm: dict[str, float],
) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    final_root = suite_dir / "final"

    unguided_dir = final_root / "unguided"
    jobs.append(
        JobSpec(
            name=f"cand{candidate_count}_unguided",
            output_dir=unguided_dir,
            command=common_base_command(args, output_dir=unguided_dir, num_candidates=candidate_count),
            metadata={
                "stage": "final",
                "candidate_count": int(candidate_count),
                "method_key": "unguided",
            },
        )
    )

    hnn_dir = final_root / "hnn"
    hnn_command = common_base_command(args, output_dir=hnn_dir, num_candidates=candidate_count)
    maybe_add_hnn_args(hnn_command, args)
    jobs.append(
        JobSpec(
            name=f"cand{candidate_count}_hnn",
            output_dir=hnn_dir,
            command=hnn_command,
            metadata={
                "stage": "final",
                "candidate_count": int(candidate_count),
                "method_key": "hnn_guided",
            },
        )
    )

    for norm, alpha in best_target_alpha_by_norm.items():
        target_dir = final_root / f"target_{norm}"
        target_command = common_base_command(args, output_dir=target_dir, num_candidates=candidate_count)
        maybe_add_target_args(target_command, args, alpha=alpha, norm=norm)
        jobs.append(
            JobSpec(
                name=f"cand{candidate_count}_target_{norm}",
                output_dir=target_dir,
                command=target_command,
                metadata={
                    "stage": "final",
                    "candidate_count": int(candidate_count),
                    "method_key": f"target_{norm}",
                    "target_norm": str(norm),
                    "target_alpha": float(alpha),
                },
            )
        )

        combined_dir = final_root / f"target_{norm}_then_hnn"
        combined_command = common_base_command(args, output_dir=combined_dir, num_candidates=candidate_count)
        maybe_add_target_args(combined_command, args, alpha=alpha, norm=norm)
        maybe_add_hnn_args(combined_command, args)
        combined_command.extend(["--guidance_order", str(args.combined_guidance_order)])
        jobs.append(
            JobSpec(
                name=f"cand{candidate_count}_target_{norm}_then_hnn",
                output_dir=combined_dir,
                command=combined_command,
                metadata={
                    "stage": "final",
                    "candidate_count": int(candidate_count),
                    "method_key": f"target_{norm}_plus_hnn",
                    "target_norm": str(norm),
                    "target_alpha": float(alpha),
                },
            )
        )

    return jobs


def run_one_job(
    *,
    job: JobSpec,
    gpu_id: int,
    resume: bool,
    logger: SuiteLogger,
) -> dict[str, object]:
    summary_path = job.output_dir / "reacher_goal_prefix_expansion_summary.json"
    log_path = job.output_dir / "suite_run.log"
    job.output_dir.mkdir(parents=True, exist_ok=True)

    if resume and summary_path.exists():
        summary = read_summary(summary_path)
        if summary_is_complete(summary):
            logger.log(f"Skipping existing {job.name} on gpu{gpu_id}: {summary_path}")
            return {
                "job": job,
                "gpu_id": int(gpu_id),
                "summary_path": str(summary_path),
                "summary": summary,
                "log_path": str(log_path),
                "skipped": True,
                "duration_sec": 0.0,
            }
        logger.log(
            f"Existing summary for {job.name} is incomplete "
            f"({summary.get('num_completed_tasks', 0)}/{summary.get('num_total_tasks', '?')}); rerunning."
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    logger.log(f"Starting {job.name} on gpu{gpu_id}")
    logger.log("Command: " + " ".join(job.command))
    start_time = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("COMMAND:\n")
        log_handle.write(" ".join(job.command) + "\n\n")
        log_handle.flush()
        subprocess.run(
            job.command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    duration_sec = time.time() - start_time
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary after {job.name}: {summary_path}")
    summary = read_summary(summary_path)
    logger.log(
        f"Finished {job.name} on gpu{gpu_id} in {duration_sec / 60.0:.1f} min "
        f"(success_rate={summary['aggregate']['success_rate']:.3f}, "
        f"best_goal_mean={summary['aggregate']['best_goal_distance']['mean']:.6f})"
    )
    return {
        "job": job,
        "gpu_id": int(gpu_id),
        "summary_path": str(summary_path),
        "summary": summary,
        "log_path": str(log_path),
        "skipped": False,
        "duration_sec": float(duration_sec),
    }


def run_stage(
    *,
    jobs: list[JobSpec],
    gpu_ids: list[int],
    resume: bool,
    logger: SuiteLogger,
    on_result=None,
) -> dict[str, dict[str, object]]:
    buckets: dict[int, list[JobSpec]] = {gpu_id: [] for gpu_id in gpu_ids}
    for index, job in enumerate(jobs):
        gpu_id = gpu_ids[index % len(gpu_ids)]
        buckets[gpu_id].append(job)

    results: dict[str, dict[str, object]] = {}
    result_lock = threading.Lock()

    def worker(gpu_id: int) -> None:
        for job in buckets[gpu_id]:
            result = run_one_job(job=job, gpu_id=gpu_id, resume=resume, logger=logger)
            with result_lock:
                results[job.name] = result
                snapshot = dict(results)
            if on_result is not None:
                on_result(result, snapshot)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(worker, gpu_id) for gpu_id in gpu_ids]
        for future in futures:
            future.result()

    return results


def make_sweep_table(entries: list[dict[str, object]]) -> str:
    header = "| norm | alpha | success_rate | best_goal_mean | final_goal_mean | output_dir |\n"
    header += "|---|---:|---:|---:|---:|---|\n"
    rows: list[str] = []
    for entry in entries:
        rows.append(
            "| {norm} | {alpha:.0e} | {success:.3f} | {best:.6f} | {final:.6f} | `{path}` |".format(
                norm=display_norm(str(entry["target_norm"])),
                alpha=float(entry["target_alpha"]),
                success=float(entry["summary"]["aggregate"]["success_rate"]),
                best=float(entry["summary"]["aggregate"]["best_goal_distance"]["mean"]),
                final=float(entry["summary"]["aggregate"]["final_goal_distance"]["mean"]),
                path=str(entry["output_dir"]),
            )
        )
    return header + "\n".join(rows)


def make_final_table(rows: list[dict[str, object]]) -> str:
    header = "| method | success_rate | best_goal_mean | final_goal_mean | qpos_mse_mean | mom_mse_mean | output_dir |\n"
    header += "|---|---:|---:|---:|---:|---:|---|\n"
    lines: list[str] = []
    for row in rows:
        aggregate = row["summary"]["aggregate"]
        qpos_mean = aggregate["qpos_mse_to_replay"]["mean"]
        mom_mean = aggregate["mom_mse_to_replay"]["mean"]
        lines.append(
            "| {method} | {success:.3f} | {best:.6f} | {final:.6f} | {qpos:.6f} | {mom:.6f} | `{path}` |".format(
                method=str(row["label"]),
                success=float(aggregate["success_rate"]),
                best=float(aggregate["best_goal_distance"]["mean"]),
                final=float(aggregate["final_goal_distance"]["mean"]),
                qpos=float(qpos_mean),
                mom=float(mom_mean),
                path=str(row["output_dir"]),
            )
        )
    return header + "\n".join(lines)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = SuiteLogger(output_root / "suite.log")

    gpu_ids = parse_int_list(args.gpu_ids)
    candidate_counts = parse_int_list(args.candidate_counts)
    target_norms = parse_str_list(args.target_norms)
    target_alpha_grid = parse_float_list(args.target_alpha_grid)
    fixed_target_alphas = parse_target_alpha_map(args.fixed_target_alphas)

    config_snapshot = {
        "checkpoint_path": str(args.checkpoint_path),
        "hnn_checkpoint_path": str(args.hnn_checkpoint_path),
        "h5_path": str(args.h5_path),
        "task_mode": str(args.task_mode),
        "task_ids": parse_int_list(args.task_ids),
        "gpu_ids": gpu_ids,
        "candidate_counts": candidate_counts,
        "target_norms": target_norms,
        "target_alpha_grid": target_alpha_grid,
        "skip_sweep": bool(args.skip_sweep),
        "fixed_target_alphas": fixed_target_alphas,
        "num_diffusion_steps": int(args.num_diffusion_steps),
        "lookahead_steps": int(args.lookahead_steps),
        "recent_prefix_cap": int(args.recent_prefix_cap),
        "stall_patience_steps": int(args.stall_patience_steps),
        "goal_tolerance": float(args.goal_tolerance),
        "random_future_target_min_initial_distance": float(args.random_future_target_min_initial_distance),
        "cross_traj_sampling_max_tries": int(args.cross_traj_sampling_max_tries),
        "max_sampling_retries": int(args.max_sampling_retries),
        "retry_improvement_margin": float(args.retry_improvement_margin),
        "target_guidance_time_power": float(args.target_guidance_time_power),
        "target_guidance_normalize_grad": bool(args.target_guidance_normalize_grad),
        "target_guidance_use_time_weights": bool(args.target_guidance_use_time_weights),
        "guidance_method": str(args.guidance_method),
        "alpha_q": float(args.alpha_q),
        "alpha_p": float(args.alpha_p),
        "guidance_trust_lambda": float(args.guidance_trust_lambda),
        "guidance_normalize_grad": bool(args.guidance_normalize_grad),
        "guidance_joint_update": bool(args.guidance_joint_update),
        "combined_guidance_order": str(args.combined_guidance_order),
        "resume": bool(args.resume),
    }
    write_json(output_root / "suite_config.json", config_snapshot)

    suite_report: dict[str, object] = {
        "config": config_snapshot,
        "candidate_runs": {},
    }

    markdown_sections: list[str] = [
        "# Reacher Guidance Comparison Suite",
        "",
        "This suite compares:",
        "- unguided",
        "- target guided with `l1`, `l2`, `l_infinite`",
        "- HNN guided",
        "- target guided with `l1`, `l2`, `l_infinite` plus HNN guidance",
        "",
        "Target-guidance alpha is swept over 5 configs per norm, then the best alpha is selected automatically on the fixed 10-task benchmark.",
        "",
        "Benchmark config:",
        f"- task_mode = `{args.task_mode}`",
        f"- task_ids = `{args.task_ids}`",
        f"- lookahead_steps = `{args.lookahead_steps}`",
        f"- recent_prefix_cap = `{args.recent_prefix_cap}`",
        f"- reset_window_time_indices = `true`",
        f"- stall_patience_steps = `{args.stall_patience_steps}`",
        f"- random_future_target_min_initial_distance = `{args.random_future_target_min_initial_distance}`",
        f"- cross_traj_sampling_max_tries = `{args.cross_traj_sampling_max_tries}`",
        "",
    ]

    for candidate_count in candidate_counts:
        logger.log(f"=== Starting candidate-count pass: {candidate_count} ===")
        candidate_dir = output_root / f"cand{candidate_count}"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        sweep_entries: list[dict[str, object]] = []
        best_target_alpha_by_norm: dict[str, float] = {}
        sweep_results: dict[str, dict[str, object]] = {}
        write_candidate_progress(
            candidate_dir=candidate_dir,
            candidate_count=candidate_count,
            target_norms=target_norms,
            sweep_results={},
            final_results={},
            best_target_alpha_by_norm={},
            skip_sweep=bool(args.skip_sweep),
        )
        if bool(args.skip_sweep):
            missing_norms = [norm for norm in target_norms if norm not in fixed_target_alphas]
            if missing_norms:
                raise ValueError(
                    "Missing fixed target alphas for norms: "
                    + ", ".join(missing_norms)
                    + ". Provide --fixed_target_alphas."
                )
            for norm in target_norms:
                best_target_alpha_by_norm[norm] = float(fixed_target_alphas[norm])
                logger.log(
                    f"Using fixed target alpha for cand{candidate_count} norm={norm}: "
                    f"{best_target_alpha_by_norm[norm]:.0e}"
                )
            write_candidate_progress(
                candidate_dir=candidate_dir,
                candidate_count=candidate_count,
                target_norms=target_norms,
                sweep_results={},
                final_results={},
                best_target_alpha_by_norm=best_target_alpha_by_norm,
                skip_sweep=bool(args.skip_sweep),
            )
        else:
            sweep_jobs = build_sweep_jobs(args, candidate_count=candidate_count, suite_dir=candidate_dir)
            sweep_results = run_stage(
                jobs=sweep_jobs,
                gpu_ids=gpu_ids,
                resume=bool(args.resume),
                logger=logger,
                on_result=lambda _result, snapshot: write_candidate_progress(
                    candidate_dir=candidate_dir,
                    candidate_count=candidate_count,
                    target_norms=target_norms,
                    sweep_results=snapshot,
                    final_results={},
                    best_target_alpha_by_norm={},
                    skip_sweep=bool(args.skip_sweep),
                ),
            )

            for norm in target_norms:
                norm_entries = [
                    {
                        "target_norm": norm,
                        "target_alpha": float(result["job"].metadata["target_alpha"]),
                        "output_dir": str(result["job"].output_dir),
                        "summary": compact_summary(result["summary"]),
                        "summary_path": result["summary_path"],
                    }
                    for result in sweep_results.values()
                    if result["job"].metadata["target_norm"] == norm
                ]
                norm_entries.sort(key=lambda entry: ranking_tuple({"aggregate": entry["summary"]["aggregate"]}))
                if not norm_entries:
                    raise RuntimeError(f"No sweep results found for norm={norm}")
                sweep_entries.extend(norm_entries)
                best_target_alpha_by_norm[norm] = float(norm_entries[0]["target_alpha"])
                logger.log(
                    f"Best target alpha for cand{candidate_count} norm={norm}: "
                    f"{norm_entries[0]['target_alpha']:.0e} "
                    f"(success_rate={norm_entries[0]['summary']['aggregate']['success_rate']:.3f}, "
                    f"best_goal_mean={norm_entries[0]['summary']['aggregate']['best_goal_distance']['mean']:.6f})"
                )
            write_candidate_progress(
                candidate_dir=candidate_dir,
                candidate_count=candidate_count,
                target_norms=target_norms,
                sweep_results=sweep_results,
                final_results={},
                best_target_alpha_by_norm=best_target_alpha_by_norm,
                skip_sweep=bool(args.skip_sweep),
            )

        final_jobs = build_final_jobs(
            args,
            candidate_count=candidate_count,
            suite_dir=candidate_dir,
            best_target_alpha_by_norm=best_target_alpha_by_norm,
        )
        final_results = run_stage(
            jobs=final_jobs,
            gpu_ids=gpu_ids,
            resume=bool(args.resume),
            logger=logger,
            on_result=lambda _result, snapshot: write_candidate_progress(
                candidate_dir=candidate_dir,
                candidate_count=candidate_count,
                target_norms=target_norms,
                sweep_results=sweep_results,
                final_results=snapshot,
                best_target_alpha_by_norm=best_target_alpha_by_norm,
                skip_sweep=bool(args.skip_sweep),
            ),
        )

        final_rows: list[dict[str, object]] = []
        final_rows.append(
            {
                "label": "unguided",
                "summary": compact_summary(final_results[f"cand{candidate_count}_unguided"]["summary"]),
                "output_dir": str(final_results[f"cand{candidate_count}_unguided"]["job"].output_dir),
            }
        )
        for norm in target_norms:
            final_rows.append(
                {
                    "label": f"target_guided_{display_norm(norm)}",
                    "summary": compact_summary(final_results[f"cand{candidate_count}_target_{norm}"]["summary"]),
                    "output_dir": str(final_results[f"cand{candidate_count}_target_{norm}"]["job"].output_dir),
                    "selected_alpha": float(best_target_alpha_by_norm[norm]),
                }
            )
        final_rows.append(
            {
                "label": "hnn_guided",
                "summary": compact_summary(final_results[f"cand{candidate_count}_hnn"]["summary"]),
                "output_dir": str(final_results[f"cand{candidate_count}_hnn"]["job"].output_dir),
            }
        )
        for norm in target_norms:
            combined_key = f"cand{candidate_count}_target_{norm}_then_hnn"
            final_rows.append(
                {
                    "label": f"target_guided_{display_norm(norm)}_plus_hnn",
                    "summary": compact_summary(final_results[combined_key]["summary"]),
                    "output_dir": str(final_results[combined_key]["job"].output_dir),
                    "selected_alpha": float(best_target_alpha_by_norm[norm]),
                }
            )

        final_rows.sort(key=lambda row: ranking_tuple({"aggregate": row["summary"]["aggregate"]}))

        candidate_report = {
            "candidate_count": int(candidate_count),
            "best_target_alpha_by_norm": best_target_alpha_by_norm,
            "sweep_entries": sweep_entries,
            "final_rows": final_rows,
        }
        suite_report["candidate_runs"][str(candidate_count)] = candidate_report
        write_json(candidate_dir / "suite_report.json", candidate_report)
        candidate_progress_payload = {
            "candidate_count": int(candidate_count),
            "is_complete": True,
            "best_target_alpha_by_norm": best_target_alpha_by_norm,
            "sweep_entries": sweep_entries,
            "final_rows": final_rows,
            "num_completed_sweep_jobs": int(len(sweep_entries)),
            "num_completed_final_jobs": int(len(final_rows)),
        }
        write_json(candidate_dir / "suite_progress.json", candidate_progress_payload)

        markdown_sections.extend(
            [
                f"## Candidate Count = {candidate_count}",
                "",
                "### Target Alpha Selection",
                "",
            ]
        )
        if sweep_entries:
            markdown_sections.extend(
                [
                    make_sweep_table(sweep_entries),
                    "",
                    "Selected target alphas:",
                ]
            )
        else:
            markdown_sections.extend(
                [
                    "Sweep skipped. Fixed target alphas were used:",
                ]
            )
        for norm in target_norms:
            markdown_sections.append(
                f"- `{display_norm(norm)}` -> `{best_target_alpha_by_norm[norm]:.0e}`"
            )
        markdown_sections.extend(
            [
                "",
                "### Final Comparison",
                "",
                make_final_table(final_rows),
                "",
            ]
        )
        (candidate_dir / "suite_progress.md").write_text(
            "\n".join(
                [
                    f"# Candidate Count = {candidate_count} Progress",
                    "",
                    "- status = `complete`",
                    "",
                    "## Target Alpha Selection",
                    "",
                    make_sweep_table(sweep_entries) if sweep_entries else "Sweep skipped. Fixed target alphas were used.",
                    "",
                    "Current best target alphas:",
                    *[
                        f"- `{display_norm(norm)}` -> `{best_target_alpha_by_norm[norm]:.0e}`"
                        for norm in target_norms
                    ],
                    "",
                    "## Final Comparison",
                    "",
                    make_final_table(final_rows),
                    "",
                ]
            ),
            encoding="utf-8",
        )

    write_json(output_root / "suite_report.json", suite_report)
    (output_root / "suite_report.md").write_text("\n".join(markdown_sections), encoding="utf-8")
    logger.log(f"Suite finished. Report: {output_root / 'suite_report.md'}")


if __name__ == "__main__":
    main()
