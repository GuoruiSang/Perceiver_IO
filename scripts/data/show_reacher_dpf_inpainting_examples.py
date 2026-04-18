#!/home/gsang/miniconda3/envs/perceiver/bin/python
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

import h5py
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.data.run_reacher_goal_prefix_expansion import (
    ARM_COLOR,
    EE_PATH_COLOR,
    END_COLOR,
    GOAL_COLOR,
    START_COLOR,
    TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_FUTURE_TARGET,
    apply_ema_once,
    arm_points_from_qpos_raw,
    build_validation_random_source_random_future_target_task,
    fingertip_xy_from_qpos_raw,
    setup_workspace_axis,
)
from src.models.HNN import HNNWrapper
from src.models.trajectory_dpf_model import TrajectoryDPF
from src.qpos_representation import decode_qpos_array, encode_qpos_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show direct DPF inpainting examples for source-to-target Reacher bridges. "
            "Each example conditions on the source state and the target terminal state, "
            "then generates the interior bridge."
        )
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/dpf_exploration_iid_uniform_len1000_v1/"
        "trajectory_dpf_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone"
        "&DecoderAttentions_cond-concat_torque_in_state_layout-shifted_tau_qpos-raw:epoch=2999_val_loss:val_loss=0.0983.ckpt",
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/data/reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/"
        "val_traj_4000-steps_1000.h5",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/plots/reacher_dpf_inpainting_examples_iid",
    )
    parser.add_argument(
        "--task_ids",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        help="Comma-separated validation trajectory indices used to sample future target tasks.",
    )
    parser.add_argument("--max_examples", type=int, default=6)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif_max_arms", type=int, default=18)
    parser.add_argument("--min_initial_distance", type=float, default=0.1)
    parser.add_argument(
        "--save_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save workspace/timeseries plots for each example. Disable for faster large sweeps.",
    )
    parser.add_argument(
        "--hnn_checkpoint_path",
        type=str,
        default="",
        help="Optional structured HNN checkpoint for guidance during endpoint-conditioned inpainting.",
    )
    parser.add_argument(
        "--guidance_method",
        type=str,
        default="strategy2",
        choices=["strategy1", "strategy2"],
    )
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
        "--summary_name",
        type=str,
        default="reacher_dpf_inpainting_examples_summary.json",
    )
    return parser.parse_args()


def parse_indices(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("task_ids produced an empty set")
    return values


def build_inpainting_observations(
    *,
    model: TrajectoryDPF,
    source_qpos_raw: np.ndarray,
    source_mom: np.ndarray,
    target_qpos_raw: np.ndarray,
    target_mom: np.ndarray,
    horizon: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observed_qpos = torch.zeros((1, horizon, model.qpos_dim), dtype=torch.float32, device=device)
    observed_mom = torch.zeros((1, horizon, model.mom_dim), dtype=torch.float32, device=device)
    observed_tau = torch.zeros((1, horizon, model.torque_dim), dtype=torch.float32, device=device)
    observed_state_mask = torch.zeros((1, horizon, model.state_dim), dtype=torch.bool, device=device)

    source_qpos_model = encode_qpos_array(
        np.asarray(source_qpos_raw, dtype=np.float32)[None, :],
        model.qpos_representation,
    )[0].astype(np.float32, copy=False)
    target_qpos_model = encode_qpos_array(
        np.asarray(target_qpos_raw, dtype=np.float32)[None, :],
        model.qpos_representation,
    )[0].astype(np.float32, copy=False)
    source_mom = np.asarray(source_mom, dtype=np.float32)
    target_mom = np.asarray(target_mom, dtype=np.float32)

    observed_qpos[0, 0] = torch.from_numpy(source_qpos_model).to(device=device)
    observed_mom[0, 0] = torch.from_numpy(source_mom).to(device=device)
    observed_state_mask[0, 0, :] = True

    observed_qpos[0, horizon - 1] = torch.from_numpy(target_qpos_model).to(device=device)
    observed_mom[0, horizon - 1] = torch.from_numpy(target_mom).to(device=device)
    observed_state_mask[0, horizon - 1, : model.qpos_dim + model.mom_dim] = True
    return observed_qpos, observed_mom, observed_tau, observed_state_mask


def plot_workspace_comparison(
    *,
    replay_qpos_raw: np.ndarray,
    generated_qpos_raw: np.ndarray,
    goal_xy: np.ndarray,
    save_path: Path,
    goal_tolerance: float,
    title: str,
    max_arms: int,
) -> None:
    replay_xy = fingertip_xy_from_qpos_raw(replay_qpos_raw)
    generated_xy = fingertip_xy_from_qpos_raw(generated_qpos_raw)

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 6.8), dpi=220)
    setup_workspace_axis(ax, goal_xy, goal_tolerance)

    ax.plot(
        replay_xy[:, 0],
        replay_xy[:, 1],
        color="#d62828",
        linewidth=1.8,
        alpha=0.9,
        label="replay path",
    )
    ax.plot(
        generated_xy[:, 0],
        generated_xy[:, 1],
        color=EE_PATH_COLOR,
        linewidth=1.8,
        alpha=0.9,
        label="generated path",
    )

    draw_idx = np.unique(
        np.linspace(0, len(generated_qpos_raw) - 1, num=min(max_arms, len(generated_qpos_raw)), dtype=int)
    )
    for order, idx in enumerate(draw_idx):
        pts = arm_points_from_qpos_raw(generated_qpos_raw[idx])
        alpha = 0.14 + 0.76 * (order + 1) / max(1, len(draw_idx))
        ax.plot(pts[:, 0], pts[:, 1], color=ARM_COLOR, linewidth=1.4, alpha=alpha)

    ax.scatter(replay_xy[0, 0], replay_xy[0, 1], color=START_COLOR, s=72, marker="o", label="source", zorder=6)
    ax.scatter(replay_xy[-1, 0], replay_xy[-1, 1], color=END_COLOR, s=72, marker="X", label="target", zorder=7)

    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries_comparison(
    *,
    replay_qpos_raw: np.ndarray,
    replay_mom: np.ndarray,
    generated_qpos_raw: np.ndarray,
    generated_mom: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    horizon = int(replay_qpos_raw.shape[0])
    t = np.arange(horizon, dtype=np.int64)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), dpi=220, sharex=True)
    series = [
        ("q0", replay_qpos_raw[:, 0], generated_qpos_raw[:, 0]),
        ("q1", replay_qpos_raw[:, 1], generated_qpos_raw[:, 1]),
        ("p0", replay_mom[:, 0], generated_mom[:, 0]),
        ("p1", replay_mom[:, 1], generated_mom[:, 1]),
    ]
    for ax, (name, replay, generated) in zip(axes.flat, series):
        ax.plot(t, replay, color="#d62828", linewidth=1.4, label="replay")
        ax.plot(t, generated, color="#2a6f97", linewidth=1.4, label="generated")
        ax.axvline(0, color="gray", linestyle=":", linewidth=1.0)
        ax.axvline(horizon - 1, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(name)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].set_xlabel("bridge timestep")
    axes[1, 1].set_xlabel("bridge timestep")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def summarize(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    task_ids = parse_indices(args.task_ids)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / args.summary_name

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[InpaintExamples] device={device}")
    print(f"[InpaintExamples] output_dir={output_dir}")
    print(f"[InpaintExamples] task_ids={task_ids}")

    model = TrajectoryDPF.load_from_checkpoint(args.checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    apply_ema_once(model)
    hnn_model = None
    if args.hnn_checkpoint_path:
        hnn_model = HNNWrapper.load_from_checkpoint(args.hnn_checkpoint_path, map_location=device)
        hnn_model = hnn_model.to(device)
        hnn_model.eval()

    rows: list[dict] = []
    skipped: list[dict] = []
    with h5py.File(args.h5_path, "r") as h5_file:
        rollout_limit = int(h5_file.attrs["num_steps"])
        for task_id in task_ids:
            try:
                task = build_validation_random_source_random_future_target_task(
                    h5_file=h5_file,
                    traj_index=int(task_id),
                    rollout_limit=rollout_limit,
                    task_seed=int(args.seed) + int(task_id),
                    min_initial_distance=float(args.min_initial_distance),
                )
            except ValueError as exc:
                skipped.append({"task_id": int(task_id), "reason": str(exc)})
                print(f"[InpaintExamples] skip task_id={int(task_id)} reason={exc}")
                continue
            source_index = int(task["metadata"]["source_index"])
            target_index = int(task["metadata"]["target_index"])
            bridge_len = int(task["reference_waypoint_index"]) + 1
            replay_qpos_raw = np.asarray(task["reference_qpos_raw"][:bridge_len], dtype=np.float64)
            replay_mom = np.asarray(task["reference_mom"][:bridge_len], dtype=np.float64)
            target_qpos_raw = np.asarray(task["target_qpos_raw"], dtype=np.float64)
            target_mom = np.asarray(task["target_mom"], dtype=np.float64)
            goal_xy = np.asarray(task["goal_xy"], dtype=np.float64)

            observed_qpos, observed_mom, observed_tau, observed_state_mask = build_inpainting_observations(
                model=model,
                source_qpos_raw=replay_qpos_raw[0],
                source_mom=replay_mom[0],
                target_qpos_raw=target_qpos_raw,
                target_mom=target_mom,
                horizon=bridge_len,
                device=device,
            )
            time_indices = torch.arange(source_index, target_index + 1, dtype=torch.long, device=device)

            need_grad_guidance = hnn_model is not None
            sample_context = contextlib.nullcontext() if need_grad_guidance else torch.no_grad()
            with sample_context:
                generated_state, generated_tau = model.sample_trajectories(
                    num_samples=1,
                    trajectory_length=bridge_len,
                    num_diffusion_steps=int(args.num_diffusion_steps),
                    prefix_len=1,
                    observed_qpos=observed_qpos,
                    observed_mom=observed_mom,
                    observed_torque=observed_tau,
                    observed_state_mask=observed_state_mask,
                    time_indices=time_indices,
                    use_ema=False,
                    sampler="ddim",
                    hnn=hnn_model,
                    guidance_method=str(args.guidance_method),
                    alpha_q=float(args.alpha_q),
                    alpha_p=float(args.alpha_p),
                    guidance_trust_lambda=float(args.guidance_trust_lambda),
                    guidance_normalize_grad=bool(args.guidance_normalize_grad),
                    guidance_joint_update=bool(args.guidance_joint_update),
                )

            generated_state_np = generated_state[0].detach().cpu().numpy()
            generated_tau_np = generated_tau[0].detach().cpu().numpy()
            generated_qpos_raw = decode_qpos_array(
                generated_state_np[:, : model.qpos_dim],
                model.qpos_representation,
            ).astype(np.float64, copy=False)
            generated_mom = generated_state_np[:, model.qpos_dim : model.qpos_dim + model.mom_dim].astype(
                np.float64,
                copy=False,
            )
            generated_ee_xy = fingertip_xy_from_qpos_raw(generated_qpos_raw)
            replay_ee_xy = fingertip_xy_from_qpos_raw(replay_qpos_raw)

            qpos_mse = float(np.mean((generated_qpos_raw - replay_qpos_raw) ** 2))
            mom_mse = float(np.mean((generated_mom - replay_mom) ** 2))
            ee_mse = float(np.mean((generated_ee_xy - replay_ee_xy) ** 2))
            target_qpos_error = float(np.linalg.norm(generated_qpos_raw[-1] - target_qpos_raw))
            target_mom_error = float(np.linalg.norm(generated_mom[-1] - target_mom))
            target_ee_error = float(np.linalg.norm(generated_ee_xy[-1] - goal_xy))

            stem = f"traj_{int(task_id):04d}_src{source_index:04d}_tgt{target_index:04d}"
            workspace_path = output_dir / f"{stem}_workspace.png"
            timeseries_path = output_dir / f"{stem}_timeseries.png"

            if bool(args.save_plots):
                plot_workspace_comparison(
                    replay_qpos_raw=replay_qpos_raw,
                    generated_qpos_raw=generated_qpos_raw,
                    goal_xy=goal_xy,
                    save_path=workspace_path,
                    goal_tolerance=0.01,
                    max_arms=int(args.gif_max_arms),
                    title=(
                        f"{stem}\n"
                        f"bridge_len={bridge_len}  init_goal={float(task['metadata']['initial_goal_distance']):.4f}  "
                        f"ee_mse={ee_mse:.6f}"
                    ),
                )
                plot_timeseries_comparison(
                    replay_qpos_raw=replay_qpos_raw,
                    replay_mom=replay_mom,
                    generated_qpos_raw=generated_qpos_raw,
                    generated_mom=generated_mom,
                    save_path=timeseries_path,
                    title=(
                        f"{stem}  qpos_mse={qpos_mse:.6f}  mom_mse={mom_mse:.6f}  "
                        f"target_ee_err={target_ee_error:.6e}"
                    ),
                )

            row = {
                "task_id": int(task_id),
                "task_label": str(task["task_label"]),
                "source_index": source_index,
                "target_index": target_index,
                "bridge_len": bridge_len,
                "initial_goal_distance": float(task["metadata"]["initial_goal_distance"]),
                "qpos_mse": qpos_mse,
                "mom_mse": mom_mse,
                "ee_mse": ee_mse,
                "target_qpos_error": target_qpos_error,
                "target_mom_error": target_mom_error,
                "target_ee_error": target_ee_error,
                "workspace_plot": (str(workspace_path) if bool(args.save_plots) else None),
                "timeseries_plot": (str(timeseries_path) if bool(args.save_plots) else None),
                "generated_torque_l2_mean": float(np.sqrt(np.mean(generated_tau_np ** 2))),
            }
            rows.append(row)
            print(
                "[InpaintExamples] task_id={} bridge_len={} qpos_mse={:.6f} mom_mse={:.6f} ee_mse={:.6f} target_ee_err={:.6e}".format(
                    int(task_id),
                    bridge_len,
                    qpos_mse,
                    mom_mse,
                    ee_mse,
                    target_ee_error,
                )
            )
            if len(rows) >= int(args.max_examples):
                break

    if not rows:
        raise RuntimeError("No valid inpainting examples were generated from the requested task_ids.")
    rows_sorted = sorted(rows, key=lambda row: (row["ee_mse"], row["qpos_mse"], row["mom_mse"]))
    summary = {
        "checkpoint_path": args.checkpoint_path,
        "hnn_checkpoint_path": (str(args.hnn_checkpoint_path) if args.hnn_checkpoint_path else None),
        "h5_path": args.h5_path,
        "device": str(device),
        "seed": int(args.seed),
        "task_ids": task_ids,
        "max_examples": int(args.max_examples),
        "num_diffusion_steps": int(args.num_diffusion_steps),
        "min_initial_distance": float(args.min_initial_distance),
        "save_plots": bool(args.save_plots),
        "guidance_method": (str(args.guidance_method) if hnn_model is not None else None),
        "alpha_q": (float(args.alpha_q) if hnn_model is not None else None),
        "alpha_p": (float(args.alpha_p) if hnn_model is not None else None),
        "guidance_trust_lambda": (float(args.guidance_trust_lambda) if hnn_model is not None else None),
        "guidance_normalize_grad": (bool(args.guidance_normalize_grad) if hnn_model is not None else None),
        "guidance_joint_update": (bool(args.guidance_joint_update) if hnn_model is not None else None),
        "num_generated_examples": int(len(rows)),
        "skipped": skipped,
        "aggregate": {
            "qpos_mse": summarize(rows, "qpos_mse"),
            "mom_mse": summarize(rows, "mom_mse"),
            "ee_mse": summarize(rows, "ee_mse"),
            "target_qpos_error": summarize(rows, "target_qpos_error"),
            "target_mom_error": summarize(rows, "target_mom_error"),
            "target_ee_error": summarize(rows, "target_ee_error"),
        },
        "rows": rows,
        "rows_sorted_by_ee_mse": rows_sorted,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[InpaintExamples] summary={summary_path}")


if __name__ == "__main__":
    main()
