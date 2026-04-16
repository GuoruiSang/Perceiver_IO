#!/home/gsang/miniconda3/envs/perceiver/bin/python
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import sys

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

import h5py
import mujoco
import numpy as np
import torch

from scripts.data.generate_bidirectional_reacher_dataset import get_model_ids
from scripts.data.run_reacher_goal_prefix_expansion import (
    TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_FUTURE_TARGET,
    TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_TARGET_ACROSS_TRAJS,
    ReacherRolloutStepper,
    aggregate_metric,
    build_summary_payload,
    build_time_series_plot,
    build_validation_random_source_random_future_target_task,
    build_validation_random_source_random_target_across_trajs_task,
    build_workspace_gif,
    build_workspace_plot,
    fingertip_xy_from_qpos_raw,
    fingertip_xy_from_qpos_tensor,
    normalize_task_mode,
    parse_indices,
)
from src.models.HNN import HNNWrapper


DEFAULT_HNN_CHECKPOINT = (
    "/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/"
    "hnn_exploration_iid_uniform_len1000_v1/StructuredHNN-ReacherExploration-IID-epoch-epoch=999.ckpt"
)
DEFAULT_H5_PATH = (
    "/home/gsang/Projects/hnn_guided_dpf/data/"
    "reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/val_traj_4000-steps_1000.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Closed-loop Reacher task rollout with structured-HNN iid random shooting. "
            "At each MPC step, sample iid-uniform torque suffix candidates, roll them out "
            "through the trained HNN, choose the closest predicted candidate, and execute "
            "the first torque in MuJoCo."
        )
    )
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_HNN_CHECKPOINT)
    parser.add_argument("--h5_path", type=str, default=DEFAULT_H5_PATH)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/plots/reacher_hnn_iid_random_shooting",
    )
    parser.add_argument("--task_ids", "--sample_indices", dest="task_ids", type=str, default="0,1,2,3")
    parser.add_argument(
        "--task_mode",
        type=str,
        default=TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_TARGET_ACROSS_TRAJS,
    )
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument(
        "--candidate_scoring",
        type=str,
        default="hnn_iid_random_shooting",
        help="Recorded in summaries; accepted for compatibility with the sharded launcher.",
    )
    parser.add_argument(
        "--torque_scale",
        type=float,
        default=0.2,
        help="iid-uniform control torque is sampled from [-torque_scale, torque_scale].",
    )
    parser.add_argument(
        "--num_diffusion_steps",
        type=int,
        default=0,
        help="Compatibility no-op; this runner does not use diffusion sampling.",
    )
    parser.add_argument("--lookahead_steps", type=int, default=256)
    parser.add_argument("--recent_prefix_cap", type=int, default=64)
    parser.add_argument("--goal_tolerance", type=float, default=0.01)
    parser.add_argument("--gif_fps", type=int, default=18)
    parser.add_argument("--gif_max_frames", type=int, default=200)
    parser.add_argument("--reset_window_time_indices", action="store_true")
    parser.add_argument("--max_prefix_len", "--max_rollout_steps", dest="max_rollout_steps", type=int, default=0)
    parser.add_argument("--stall_patience_steps", type=int, default=500)
    parser.add_argument("--random_future_target_min_initial_distance", type=float, default=0.1)
    parser.add_argument("--cross_traj_sampling_max_tries", type=int, default=128)
    parser.add_argument("--max_sampling_retries", type=int, default=3)
    parser.add_argument("--retry_improvement_margin", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary_name", type=str, default="reacher_goal_prefix_expansion_summary.json")
    parser.add_argument(
        "--progress_summary_name",
        type=str,
        default="reacher_goal_prefix_expansion_progress.json",
    )
    return parser.parse_args()


def sample_iid_torque_suffix(
    *,
    num_steps: int,
    num_candidates: int,
    torque_dim: int,
    torque_scale: float,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    tau = torch.empty((num_steps, num_candidates, torque_dim), dtype=torch.float32, device=device)
    tau.uniform_(-float(torque_scale), float(torque_scale), generator=generator)
    return tau


@torch.inference_mode()
def rollout_hnn_candidates(
    *,
    hnn_model: HNNWrapper,
    q0: torch.Tensor,
    p0: torch.Tensor,
    tau_control_seq: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    num_steps, num_candidates, _ = tau_control_seq.shape
    q = q0.expand(num_candidates, -1).contiguous()
    p = p0.expand(num_candidates, -1).contiguous()
    tau_generalized_seq = hnn_model._scale_torque(tau_control_seq)
    q_steps = []
    for step_idx in range(num_steps):
        p, q = hnn_model.symplectic_euler_step(p, q, tau_generalized_seq[step_idx], float(dt))
        q_steps.append(q)
    return torch.stack(q_steps, dim=1)


def main() -> None:
    args = parse_args()
    args.task_mode = normalize_task_mode(args.task_mode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / args.summary_name
    progress_summary_path = output_dir / args.progress_summary_name

    # Fields expected by the shared summary writer.
    args.hnn_checkpoint_path = args.checkpoint_path
    args.guidance_method = "hnn_iid_random_shooting"
    args.alpha_q = 0.0
    args.alpha_p = 0.0
    args.guidance_trust_lambda = 0.0
    args.guidance_normalize_grad = False
    args.guidance_joint_update = False
    args.guidance_order = "none"
    args.target_guidance_alpha = 0.0
    args.target_guidance_time_power = 0.0
    args.target_guidance_normalize_grad = False
    args.target_guidance_norm = "none"
    args.target_guidance_use_time_weights = False

    task_ids = parse_indices(args.task_ids)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[HNNRandomShooting] device={device}")
    print(f"[HNNRandomShooting] output_dir={output_dir}")
    print(f"[HNNRandomShooting] task_mode={args.task_mode}")
    print(f"[HNNRandomShooting] task_ids={task_ids}")

    hnn_model = HNNWrapper.load_from_checkpoint(args.checkpoint_path, map_location=device)
    hnn_model = hnn_model.to(device)
    hnn_model.eval()
    coordinate_dim = int(hnn_model.hparams["coordinate_dim"])

    with h5py.File(args.h5_path, "r") as h5_file:
        xml_content = h5_file.attrs["xml"]
        if isinstance(xml_content, bytes):
            xml_content = xml_content.decode()
        trajectory_length_total = int(h5_file.attrs["num_steps"])
        data_dt = float(h5_file.attrs.get("data_dt", h5_file.attrs.get("dt", 0.001)))
        sim_dt = float(h5_file.attrs.get("dt", data_dt))
        rollout_limit = (
            int(trajectory_length_total)
            if int(args.max_rollout_steps) <= 0
            else max(2, min(int(args.max_rollout_steps), trajectory_length_total))
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / "model.xml"
            xml_path.write_text(xml_content, encoding="utf-8")
            mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
            get_model_ids(mj_model)
            stepper = ReacherRolloutStepper(mj_model, dt=sim_dt, data_dt=data_dt)
            summary_rows: list[dict] = []

            for task_id in task_ids:
                if args.task_mode == TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_FUTURE_TARGET:
                    task = build_validation_random_source_random_future_target_task(
                        h5_file=h5_file,
                        traj_index=int(task_id),
                        rollout_limit=rollout_limit,
                        task_seed=int(args.seed) + int(task_id),
                        min_initial_distance=float(args.random_future_target_min_initial_distance),
                    )
                else:
                    task = build_validation_random_source_random_target_across_trajs_task(
                        h5_file=h5_file,
                        task_seed=int(args.seed) + int(task_id),
                        rollout_limit=rollout_limit,
                        min_initial_distance=float(args.random_future_target_min_initial_distance),
                        max_tries=int(args.cross_traj_sampling_max_tries),
                    )

                goal_xy = np.asarray(task["goal_xy"], dtype=np.float64)
                initial_qpos_raw = np.asarray(task["initial_qpos_raw"], dtype=np.float64)[:coordinate_dim]
                initial_mom = np.asarray(task["initial_mom"], dtype=np.float64)[:coordinate_dim]
                reference_qpos_raw = task["reference_qpos_raw"]
                reference_mom = task["reference_mom"]
                task_rollout_limit = int(task.get("rollout_limit", rollout_limit))

                qpos_prefix_raw = [initial_qpos_raw.copy()]
                mom_prefix = [initial_mom.copy()]
                torque_prefix = []
                rollout_goal_distances = [
                    float(np.linalg.norm(fingertip_xy_from_qpos_raw(initial_qpos_raw[None, :])[0] - goal_xy))
                ]
                use_unbounded_reset_rollout = bool(args.reset_window_time_indices)
                best_goal_distance_so_far = float(rollout_goal_distances[-1])
                stall_steps_since_decrease = 0
                stopped_due_to_stall = False
                selection_trace: list[dict] = []
                reached_goal = rollout_goal_distances[-1] <= float(args.goal_tolerance)

                while not reached_goal:
                    if (not use_unbounded_reset_rollout) and len(qpos_prefix_raw) >= task_rollout_limit:
                        break
                    if use_unbounded_reset_rollout and stall_steps_since_decrease >= int(args.stall_patience_steps):
                        stopped_due_to_stall = True
                        break

                    prefix_len = len(qpos_prefix_raw)
                    future_steps = int(args.lookahead_steps) if int(args.lookahead_steps) > 0 else trajectory_length_total
                    current_goal_distance = float(rollout_goal_distances[-1])
                    required_goal_distance = current_goal_distance - float(args.retry_improvement_margin)
                    goal_xy_t = torch.as_tensor(goal_xy, dtype=torch.float32, device=device)
                    q0 = torch.as_tensor(qpos_prefix_raw[-1], dtype=torch.float32, device=device).view(1, -1)
                    p0 = torch.as_tensor(mom_prefix[-1], dtype=torch.float32, device=device).view(1, -1)

                    best_candidate_idx = -1
                    best_retry_idx = -1
                    best_retry_seed = -1
                    best_candidate_goal_distance = float("inf")
                    best_generated_tau = None
                    retry_trace: list[dict] = []

                    for retry_idx in range(max(0, int(args.max_sampling_retries)) + 1):
                        local_seed = int(args.seed) + int(task["task_id"]) * 1000 + prefix_len * 100 + retry_idx
                        tau_seq = sample_iid_torque_suffix(
                            num_steps=future_steps,
                            num_candidates=int(args.num_candidates),
                            torque_dim=coordinate_dim,
                            torque_scale=float(args.torque_scale),
                            device=device,
                            seed=local_seed,
                        )
                        q_candidates = rollout_hnn_candidates(
                            hnn_model=hnn_model,
                            q0=q0,
                            p0=p0,
                            tau_control_seq=tau_seq,
                            dt=data_dt,
                        )
                        candidate_suffix_xy = fingertip_xy_from_qpos_tensor(q_candidates)
                        candidate_min_goal_dist = torch.linalg.norm(
                            candidate_suffix_xy - goal_xy_t.view(1, 1, 2),
                            dim=-1,
                        ).amin(dim=1)
                        retry_best_idx = int(torch.argmin(candidate_min_goal_dist).item())
                        retry_best_dist = float(candidate_min_goal_dist[retry_best_idx].item())

                        retry_trace.append(
                            {
                                "retry_idx": int(retry_idx),
                                "sample_seed": int(local_seed),
                                "candidate_scoring": "hnn_iid_random_shooting",
                                "best_candidate_idx": int(retry_best_idx),
                                "best_scored_goal_distance": float(retry_best_dist),
                                "improved_over_current": bool(retry_best_dist < required_goal_distance),
                            }
                        )

                        if retry_best_dist < best_candidate_goal_distance:
                            best_candidate_goal_distance = retry_best_dist
                            best_candidate_idx = retry_best_idx
                            best_retry_idx = retry_idx
                            best_retry_seed = local_seed
                            best_generated_tau = tau_seq.detach().cpu().numpy()

                        if retry_best_dist < required_goal_distance:
                            break

                    if best_generated_tau is None or best_candidate_idx < 0:
                        raise RuntimeError("HNN random shooting failed to produce any candidate torque.")

                    applied_tau = best_generated_tau[0, best_candidate_idx].astype(np.float64, copy=False)
                    next_qpos_raw, next_mom = stepper.step(
                        qpos_raw=qpos_prefix_raw[-1],
                        mom=mom_prefix[-1],
                        torque=applied_tau,
                    )
                    next_qpos_raw = next_qpos_raw[:coordinate_dim]
                    next_mom = next_mom[:coordinate_dim]
                    next_goal_distance = float(
                        np.linalg.norm(fingertip_xy_from_qpos_raw(next_qpos_raw[None, :])[0] - goal_xy)
                    )

                    qpos_prefix_raw.append(next_qpos_raw)
                    mom_prefix.append(next_mom)
                    torque_prefix.append(applied_tau)
                    rollout_goal_distances.append(next_goal_distance)
                    reached_goal = next_goal_distance <= float(args.goal_tolerance)
                    if next_goal_distance + 1e-12 < best_goal_distance_so_far:
                        best_goal_distance_so_far = next_goal_distance
                    if next_goal_distance + 1e-12 < current_goal_distance:
                        stall_steps_since_decrease = 0
                    else:
                        stall_steps_since_decrease += 1

                    selection_trace.append(
                        {
                            "prefix_len_before_step": int(prefix_len),
                            "sample_horizon": int(future_steps),
                            "reset_window_time_indices": bool(args.reset_window_time_indices),
                            "sample_seed": int(best_retry_seed),
                            "current_goal_distance": float(current_goal_distance),
                            "required_goal_distance": float(required_goal_distance),
                            "num_retries_used": int(best_retry_idx),
                            "num_sampling_attempts": int(len(retry_trace)),
                            "total_candidates_evaluated": int(len(retry_trace) * int(args.num_candidates)),
                            "candidate_scoring": "hnn_iid_random_shooting",
                            "retry_trace": retry_trace,
                            "chosen_retry_idx": int(best_retry_idx),
                            "chosen_candidate_idx": int(best_candidate_idx),
                            "chosen_candidate_best_goal_dist": float(best_candidate_goal_distance),
                            "applied_tau": applied_tau.tolist(),
                            "result_goal_distance": float(next_goal_distance),
                            "best_goal_distance_so_far": float(best_goal_distance_so_far),
                            "stall_steps_since_decrease": int(stall_steps_since_decrease),
                        }
                    )

                rollout_qpos_raw = np.asarray(qpos_prefix_raw, dtype=np.float64)
                rollout_mom = np.asarray(mom_prefix, dtype=np.float64)
                rollout_tau = (
                    np.asarray(torque_prefix, dtype=np.float64)
                    if torque_prefix
                    else np.zeros((0, coordinate_dim), dtype=np.float64)
                )
                rollout_goal_dist = np.asarray(rollout_goal_distances, dtype=np.float64)
                replay_goal_dist = None
                qpos_mse = None
                mom_mse = None
                ee_mse = None
                if reference_qpos_raw is not None and reference_mom is not None:
                    reference_qpos_raw = np.asarray(reference_qpos_raw, dtype=np.float64)[:, :coordinate_dim]
                    reference_mom = np.asarray(reference_mom, dtype=np.float64)[:, :coordinate_dim]
                    replay_goal_dist = np.linalg.norm(
                        fingertip_xy_from_qpos_raw(reference_qpos_raw) - goal_xy[None, :],
                        axis=-1,
                    )
                    common_horizon = min(len(rollout_qpos_raw), len(reference_qpos_raw))
                    qpos_mse = float(
                        np.mean((rollout_qpos_raw[:common_horizon] - reference_qpos_raw[:common_horizon]) ** 2)
                    )
                    mom_mse = float(np.mean((rollout_mom[:common_horizon] - reference_mom[:common_horizon]) ** 2))
                    ee_mse = float(
                        np.mean(
                            (
                                fingertip_xy_from_qpos_raw(rollout_qpos_raw[:common_horizon])
                                - fingertip_xy_from_qpos_raw(reference_qpos_raw[:common_horizon])
                            )
                            ** 2
                        )
                    )

                workspace_plot = output_dir / f"{task['task_label']}_workspace.png"
                workspace_gif = output_dir / f"{task['task_label']}_workspace.gif"
                timeseries_plot = output_dir / f"{task['task_label']}_timeseries.png"
                build_workspace_plot(
                    rollout_qpos_raw=rollout_qpos_raw,
                    goal_xy=goal_xy,
                    goal_tolerance=float(args.goal_tolerance),
                    best_goal_distance=float(rollout_goal_dist.min()),
                    final_goal_distance=float(rollout_goal_dist[-1]),
                    title=(
                        f"{task['task_label']}  HNN iid shooting  steps={len(rollout_qpos_raw)}  "
                        f"best_goal={float(rollout_goal_dist.min()):.4f}  "
                        f"final_goal={float(rollout_goal_dist[-1]):.4f}"
                    ),
                    save_path=workspace_plot,
                )
                build_workspace_gif(
                    rollout_qpos_raw=rollout_qpos_raw,
                    goal_xy=goal_xy,
                    goal_tolerance=float(args.goal_tolerance),
                    goal_distance=rollout_goal_dist,
                    title=task["task_label"],
                    save_path=workspace_gif,
                    fps=int(args.gif_fps),
                    max_frames=int(args.gif_max_frames),
                )
                build_time_series_plot(
                    rollout_qpos_raw=rollout_qpos_raw,
                    rollout_mom=rollout_mom,
                    goal_xy=goal_xy,
                    goal_tolerance=float(args.goal_tolerance),
                    goal_distance=rollout_goal_dist,
                    best_goal_distance=float(rollout_goal_dist.min()),
                    final_goal_distance=float(rollout_goal_dist[-1]),
                    save_path=timeseries_plot,
                )

                row = {
                    "task_mode": str(task["task_mode"]),
                    "task_id": int(task["task_id"]),
                    "task_label": str(task["task_label"]),
                    "traj_index": int(task["metadata"]["traj_index"]) if "traj_index" in task["metadata"] else None,
                    "goal_xy": goal_xy.tolist(),
                    "goal_tolerance": float(args.goal_tolerance),
                    "hnn_checkpoint_path": str(args.checkpoint_path),
                    "guidance_method": "hnn_iid_random_shooting",
                    "alpha_q": 0.0,
                    "alpha_p": 0.0,
                    "guidance_trust_lambda": 0.0,
                    "guidance_normalize_grad": False,
                    "guidance_joint_update": False,
                    "guidance_order": "none",
                    "target_guidance_alpha": 0.0,
                    "target_guidance_time_power": 0.0,
                    "target_guidance_normalize_grad": False,
                    "target_guidance_norm": "none",
                    "target_guidance_use_time_weights": False,
                    "num_candidates": int(args.num_candidates),
                    "candidate_scoring": "hnn_iid_random_shooting",
                    "torque_mode": "iid_uniform",
                    "torque_scale": float(args.torque_scale),
                    "num_diffusion_steps": 0,
                    "lookahead_steps": int(args.lookahead_steps),
                    "recent_prefix_cap": int(args.recent_prefix_cap),
                    "max_rollout_steps": None if use_unbounded_reset_rollout else int(task_rollout_limit),
                    "steps_taken": int(len(rollout_qpos_raw)),
                    "reached_goal": bool(reached_goal),
                    "stopped_due_to_stall": bool(stopped_due_to_stall),
                    "best_goal_distance": float(rollout_goal_dist.min()),
                    "final_goal_distance": float(rollout_goal_dist[-1]),
                    "replay_best_goal_distance": None if replay_goal_dist is None else float(replay_goal_dist.min()),
                    "qpos_mse_to_replay": qpos_mse,
                    "mom_mse_to_replay": mom_mse,
                    "ee_xy_mse_to_replay": ee_mse,
                    "workspace_plot": str(workspace_plot),
                    "workspace_gif": str(workspace_gif),
                    "timeseries_plot": str(timeseries_plot),
                    "reference_waypoint_index": task["reference_waypoint_index"],
                    "task_metadata": task["metadata"],
                    "rollout_qpos_raw": rollout_qpos_raw.tolist(),
                    "rollout_mom": rollout_mom.tolist(),
                    "rollout_tau": rollout_tau.tolist(),
                    "selection_trace": selection_trace,
                }
                summary_rows.append(row)
                progress_summary = build_summary_payload(
                    args=args,
                    device=device,
                    task_ids=task_ids,
                    rows=summary_rows,
                    hnn_model=hnn_model,
                    is_complete=False,
                    compact_rows_only=True,
                )
                progress_summary["torque_mode"] = "iid_uniform"
                progress_summary["torque_scale"] = float(args.torque_scale)
                progress_summary_path.write_text(json.dumps(progress_summary, indent=2), encoding="utf-8")
                print(
                    f"[HNNRandomShooting] progress={len(summary_rows)}/{len(task_ids)} "
                    f"progress_summary={progress_summary_path}"
                )
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in row
                            if k
                            not in {
                                "rollout_qpos_raw",
                                "rollout_mom",
                                "rollout_tau",
                                "selection_trace",
                            }
                        },
                        indent=2,
                    )
                )

    summary = build_summary_payload(
        args=args,
        device=device,
        task_ids=task_ids,
        rows=summary_rows,
        hnn_model=hnn_model,
        is_complete=True,
        compact_rows_only=False,
    )
    summary["torque_mode"] = "iid_uniform"
    summary["torque_scale"] = float(args.torque_scale)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    progress_summary = build_summary_payload(
        args=args,
        device=device,
        task_ids=task_ids,
        rows=summary_rows,
        hnn_model=hnn_model,
        is_complete=True,
        compact_rows_only=True,
    )
    progress_summary["torque_mode"] = "iid_uniform"
    progress_summary["torque_scale"] = float(args.torque_scale)
    progress_summary_path.write_text(json.dumps(progress_summary, indent=2), encoding="utf-8")
    print(f"[HNNRandomShooting] summary={summary_path}")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
