#!/home/gsang/miniconda3/envs/perceiver/bin/python
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
import sys

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

import h5py
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from src.models.trajectory_dpf_model import TrajectoryDPF
from src.qpos_representation import decode_qpos_tensor, encode_qpos_array


L1 = 0.10
L2 = 0.11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Closed-loop Reacher task rollout by iteratively expanding the prefix. "
            "At each step, sample multiple suffixes, score them by best end-effector distance "
            "to the goal, and apply the first torque from the best suffix."
        )
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/dpf/"
        "trajectory_dpf_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone"
        "&DecoderAttentions_cond-concat_torque_in_state_layout-shifted_tau_qpos-raw:epoch=2999_val_loss:val_loss=0.0012.ckpt",
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/data/reacher_bidirectional_unbounded_j1_dt0p001_len1000/"
        "traj_2000-steps_1000.h5",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/plots/reacher_goal_prefix_expansion",
    )
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="313,267,787,338",
        help="Comma-separated trajectory indices to evaluate.",
    )
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--goal_tolerance", type=float, default=0.01)
    parser.add_argument(
        "--lookahead_steps",
        type=int,
        default=0,
        help=(
            "Sample only prefix_len + lookahead_steps timesteps for candidate scoring. "
            "Use 0 or a negative value to keep full-horizon sampling."
        ),
    )
    parser.add_argument(
        "--max_prefix_len",
        type=int,
        default=0,
        help=(
            "Stop when prefix reaches this length if the goal has not been reached. "
            "Use 0 or a negative value to roll out the full trajectory horizon."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--summary_name",
        type=str,
        default="reacher_goal_prefix_expansion_summary.json",
    )
    return parser.parse_args()


def parse_indices(text: str) -> list[int]:
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def fingertip_xy_from_qpos_raw(qpos_raw: np.ndarray) -> np.ndarray:
    q0 = qpos_raw[..., 0]
    q1 = qpos_raw[..., 1]
    xy = np.empty(qpos_raw.shape[:-1] + (2,), dtype=np.float64)
    xy[..., 0] = L1 * np.cos(q0) + L2 * np.cos(q0 + q1)
    xy[..., 1] = L1 * np.sin(q0) + L2 * np.sin(q0 + q1)
    return xy


def fingertip_xy_from_qpos_tensor(qpos_raw: torch.Tensor) -> torch.Tensor:
    q0 = qpos_raw[..., 0]
    q1 = qpos_raw[..., 1]
    return torch.stack(
        [
            L1 * torch.cos(q0) + L2 * torch.cos(q0 + q1),
            L1 * torch.sin(q0) + L2 * torch.sin(q0 + q1),
        ],
        dim=-1,
    )


def arm_points_from_qpos_raw(qpos_raw: np.ndarray) -> np.ndarray:
    q0 = float(qpos_raw[0])
    q1 = float(qpos_raw[1])
    elbow = np.array([L1 * np.cos(q0), L1 * np.sin(q0)], dtype=np.float64)
    tip = np.array(
        [L1 * np.cos(q0) + L2 * np.cos(q0 + q1), L1 * np.sin(q0) + L2 * np.sin(q0 + q1)],
        dtype=np.float64,
    )
    return np.stack([np.zeros(2, dtype=np.float64), elbow, tip], axis=0)


def compute_initial_qvel(model: mujoco.MjModel, qpos_raw: np.ndarray, mom: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    qpos_dim = int(qpos_raw.shape[-1])
    qvel_dim = int(mom.shape[-1])
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:qpos_dim] = qpos_raw
    mujoco.mj_forward(model, data)
    mass = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, mass, data.qM)
    return np.linalg.solve(mass[:qvel_dim, :qvel_dim], mom).astype(np.float64, copy=False)


class ReacherRolloutStepper:
    def __init__(self, model: mujoco.MjModel, dt: float, data_dt: float) -> None:
        self.model = model
        self.model.opt.timestep = float(dt)
        self.skip_steps = max(1, int(round(float(data_dt) / float(dt))))
        self.data = mujoco.MjData(model)
        self.mass = np.zeros((model.nv, model.nv), dtype=np.float64)

    def step(
        self,
        qpos_raw: np.ndarray,
        mom: np.ndarray,
        torque: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        data = self.data
        qpos_dim = int(qpos_raw.shape[-1])
        qvel_dim = int(mom.shape[-1])
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.qpos[:qpos_dim] = qpos_raw
        mujoco.mj_forward(self.model, data)
        mujoco.mj_fullM(self.model, self.mass, data.qM)
        data.qvel[:qvel_dim] = np.linalg.solve(self.mass[:qvel_dim, :qvel_dim], mom).astype(
            np.float64,
            copy=False,
        )
        mujoco.mj_forward(self.model, data)

        for _ in range(self.skip_steps):
            data.ctrl[:] = torque
            mujoco.mj_step(self.model, data)

        mujoco.mj_fullM(self.model, self.mass, data.qM)
        next_qpos_raw = data.qpos[:qpos_dim].copy()
        next_mom = (self.mass @ data.qvel)[:qvel_dim].copy()
        return next_qpos_raw, next_mom


def apply_ema_once(model: TrajectoryDPF) -> None:
    if model.ema is None:
        return
    device = next(model.model.parameters()).device
    for name, param in model.model.named_parameters():
        if param.requires_grad and name in model.ema.shadow:
            model.ema.shadow[name] = model.ema.shadow[name].to(device=device, dtype=param.dtype)
    model.ema.store(model.model)


def effective_sampling_horizon(prefix_len: int, rollout_limit: int, lookahead_steps: int) -> int:
    if int(lookahead_steps) <= 0:
        return int(rollout_limit)
    return int(min(rollout_limit, max(prefix_len + 1, prefix_len + int(lookahead_steps))))


def build_workspace_plot(
    *,
    rollout_qpos_raw: np.ndarray,
    goal_xy: np.ndarray,
    goal_tolerance: float,
    best_goal_distance: float,
    final_goal_distance: float,
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 6.6), dpi=220)
    ax.add_patch(plt.Circle((0.0, 0.0), L1 + L2, color="#bbbbbb", fill=False, linestyle="--", linewidth=1.0))
    ax.add_patch(plt.Circle((0.0, 0.0), abs(L1 - L2), color="#dddddd", fill=False, linestyle=":", linewidth=1.0))

    ee_xy = fingertip_xy_from_qpos_raw(rollout_qpos_raw)
    ax.plot(ee_xy[:, 0], ee_xy[:, 1], color="#2a6f97", linewidth=1.4, alpha=0.85, label="end effector path")

    num_draw = min(18, len(rollout_qpos_raw))
    draw_idx = np.linspace(0, len(rollout_qpos_raw) - 1, num=num_draw, dtype=int)
    for order, idx in enumerate(draw_idx):
        pts = arm_points_from_qpos_raw(rollout_qpos_raw[idx])
        alpha = 0.18 + 0.72 * (order + 1) / max(1, num_draw)
        ax.plot(pts[:, 0], pts[:, 1], color="#2a6f97", linewidth=1.6, alpha=alpha)
        ax.scatter(pts[-1, 0], pts[-1, 1], color="#2a6f97", s=8, alpha=alpha)

    ax.scatter(ee_xy[0, 0], ee_xy[0, 1], color="#2a9d8f", s=70, marker="o", label="start", zorder=6)
    ax.scatter(ee_xy[-1, 0], ee_xy[-1, 1], color="#264653", s=70, marker="X", label="end", zorder=7)
    ax.scatter(goal_xy[0], goal_xy[1], color="#d62828", s=180, marker="*", label="goal", zorder=8)
    ax.add_patch(
        plt.Circle(
            (float(goal_xy[0]), float(goal_xy[1])),
            radius=float(goal_tolerance),
            color="#d62828",
            fill=False,
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
        )
    )
    ax.set_title(
        "Closed-loop rollout\nbest_dist={:.4f}  final_dist={:.4f}".format(
            float(best_goal_distance),
            float(final_goal_distance),
        ),
        fontsize=11,
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def build_time_series_plot(
    *,
    rollout_qpos_raw: np.ndarray,
    rollout_mom: np.ndarray,
    goal_xy: np.ndarray,
    goal_tolerance: float,
    goal_distance: np.ndarray,
    best_goal_distance: float,
    final_goal_distance: float,
    save_path: Path,
) -> None:
    horizon = len(rollout_qpos_raw)
    t = np.arange(horizon)
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 9.0), dpi=220, sharex=True)
    series = [
        ("q0", rollout_qpos_raw[:horizon, 0]),
        ("q1", rollout_qpos_raw[:horizon, 1]),
        ("p0", rollout_mom[:horizon, 0]),
        ("p1", rollout_mom[:horizon, 1]),
    ]
    for ax, (name, rollout) in zip(axes.flat[:4], series):
        ax.plot(t, rollout, color="#2a6f97", linewidth=1.35, label="closed-loop")
        ax.set_title(name)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper right", fontsize=8)

    ax_goal = axes[2, 0]
    ax_goal.plot(t, goal_distance[:horizon], color="#d62828", linewidth=1.5)
    ax_goal.axhline(float(goal_tolerance), color="#6a994e", linestyle="--", linewidth=1.2, label="goal tolerance")
    ax_goal.set_title("goal distance")
    ax_goal.set_xlabel("step")
    ax_goal.grid(alpha=0.25)
    ax_goal.legend(loc="upper right", fontsize=8)

    axes[2, 1].axis("off")
    axes[1, 0].set_xlabel("step")
    axes[1, 1].set_xlabel("step")
    fig.suptitle(
        "goal=({:.3f}, {:.3f})  best_dist={:.4f}  final_dist={:.4f}".format(
            float(goal_xy[0]),
            float(goal_xy[1]),
            float(best_goal_distance),
            float(final_goal_distance),
        ),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def aggregate_metric(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = parse_indices(args.sample_indices)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[GoalExpand] device={device}")
    print(f"[GoalExpand] output_dir={output_dir}")
    print(f"[GoalExpand] sample_indices={sample_indices}")

    model = TrajectoryDPF.load_from_checkpoint(args.checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    apply_ema_once(model)

    with h5py.File(args.h5_path, "r") as h5_file:
        xml_content = h5_file.attrs["xml"]
        if isinstance(xml_content, bytes):
            xml_content = xml_content.decode()
        trajectory_length_total = int(h5_file.attrs["num_steps"])
        rollout_limit = (
            int(trajectory_length_total)
            if int(args.max_prefix_len) <= 0
            else max(2, min(int(args.max_prefix_len), trajectory_length_total))
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / "model.xml"
            xml_path.write_text(xml_content, encoding="utf-8")
            mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
            stepper = ReacherRolloutStepper(mj_model, dt=float(model.dt), data_dt=float(model.data_dt))

            summary_rows: list[dict] = []
            for traj_index in sample_indices:
                traj = h5_file[f"traj_{traj_index}"]
                replay_qpos_raw = traj["seq_qpos"][:rollout_limit].astype(np.float64)
                replay_mom = traj["seq_mom"][:rollout_limit].astype(np.float64)
                goal_xy = traj["waypoint_xy"][:].astype(np.float64)

                replay_qpos_model = encode_qpos_array(
                    replay_qpos_raw.astype(np.float32),
                    model.qpos_representation,
                ).astype(np.float32, copy=False)
                replay_mom_f32 = replay_mom.astype(np.float32, copy=False)

                qpos_prefix_raw = [replay_qpos_raw[0].copy()]
                mom_prefix = [replay_mom[0].copy()]
                torque_prefix = []
                rollout_goal_distances = [float(np.linalg.norm(fingertip_xy_from_qpos_raw(replay_qpos_raw[:1])[0] - goal_xy))]
                selection_trace: list[dict] = []
                reached_goal = rollout_goal_distances[-1] <= float(args.goal_tolerance)
                observed_qpos = torch.zeros(
                    (1, rollout_limit, replay_qpos_model.shape[-1]),
                    dtype=torch.float32,
                    device=device,
                )
                observed_mom = torch.zeros(
                    (1, rollout_limit, replay_mom_f32.shape[-1]),
                    dtype=torch.float32,
                    device=device,
                )
                observed_tau = torch.zeros(
                    (1, rollout_limit, model.torque_dim),
                    dtype=torch.float32,
                    device=device,
                )
                observed_qpos[0, 0] = torch.from_numpy(replay_qpos_model[0]).to(device=device)
                observed_mom[0, 0] = torch.from_numpy(replay_mom_f32[0]).to(device=device)

                while len(qpos_prefix_raw) < rollout_limit and not reached_goal:
                    prefix_len = len(qpos_prefix_raw)
                    sample_horizon = effective_sampling_horizon(
                        prefix_len=prefix_len,
                        rollout_limit=rollout_limit,
                        lookahead_steps=int(args.lookahead_steps),
                    )

                    local_seed = int(args.seed) + traj_index * 1000 + prefix_len
                    torch.manual_seed(local_seed)
                    np.random.seed(local_seed)

                    with torch.no_grad():
                        generated_state, generated_tau = model.sample_trajectories(
                            num_samples=int(args.num_candidates),
                            trajectory_length=sample_horizon,
                            num_diffusion_steps=int(args.num_diffusion_steps),
                            sample_mode="observed_prefix_completion",
                            prefix_len=prefix_len,
                            observed_qpos=observed_qpos[:, :sample_horizon, :],
                            observed_mom=observed_mom[:, :sample_horizon, :],
                            observed_torque=observed_tau[:, :sample_horizon, :],
                            use_ema=False,
                            sampler="ddim",
                        )

                    candidate_qpos_model = generated_state[:, prefix_len:, : model.qpos_dim]
                    candidate_qpos_raw = decode_qpos_tensor(candidate_qpos_model, model.qpos_representation)
                    candidate_suffix_xy = fingertip_xy_from_qpos_tensor(candidate_qpos_raw)
                    goal_xy_t = torch.as_tensor(goal_xy, dtype=candidate_suffix_xy.dtype, device=device)
                    candidate_min_goal_dist = torch.linalg.norm(
                        candidate_suffix_xy - goal_xy_t.view(1, 1, 2),
                        dim=-1,
                    ).amin(dim=1)
                    best_candidate_idx = int(torch.argmin(candidate_min_goal_dist).item())

                    applied_tau = (
                        generated_tau[best_candidate_idx, prefix_len - 1]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64, copy=False)
                    )
                    next_qpos_raw, next_mom = stepper.step(
                        qpos_raw=qpos_prefix_raw[-1],
                        mom=mom_prefix[-1],
                        torque=applied_tau,
                    )
                    next_goal_distance = float(
                        np.linalg.norm(fingertip_xy_from_qpos_raw(next_qpos_raw[None, :])[0] - goal_xy)
                    )

                    qpos_prefix_raw.append(next_qpos_raw)
                    mom_prefix.append(next_mom)
                    torque_prefix.append(applied_tau)
                    rollout_goal_distances.append(next_goal_distance)
                    reached_goal = next_goal_distance <= float(args.goal_tolerance)
                    observed_qpos[0, prefix_len] = torch.from_numpy(
                        encode_qpos_array(next_qpos_raw[None, :].astype(np.float32), model.qpos_representation)[0]
                    ).to(device=device)
                    observed_mom[0, prefix_len] = torch.from_numpy(next_mom.astype(np.float32, copy=False)).to(device=device)
                    observed_tau[0, prefix_len - 1] = torch.from_numpy(applied_tau.astype(np.float32, copy=False)).to(
                        device=device
                    )

                    selection_trace.append(
                        {
                            "prefix_len_before_step": prefix_len,
                            "sample_horizon": int(sample_horizon),
                            "sample_seed": local_seed,
                            "chosen_candidate_idx": best_candidate_idx,
                            "chosen_candidate_best_goal_dist": float(candidate_min_goal_dist[best_candidate_idx].item()),
                            "applied_tau": applied_tau.tolist(),
                            "result_goal_distance": next_goal_distance,
                        }
                    )

                rollout_qpos_raw = np.asarray(qpos_prefix_raw, dtype=np.float64)
                rollout_mom = np.asarray(mom_prefix, dtype=np.float64)
                rollout_tau = (
                    np.asarray(torque_prefix, dtype=np.float64)
                    if torque_prefix
                    else np.zeros((0, model.torque_dim), dtype=np.float64)
                )
                replay_goal_dist = np.linalg.norm(
                    fingertip_xy_from_qpos_raw(replay_qpos_raw) - goal_xy[None, :],
                    axis=-1,
                )
                rollout_goal_dist = np.asarray(rollout_goal_distances, dtype=np.float64)
                common_horizon = min(len(rollout_qpos_raw), len(replay_qpos_raw))
                qpos_mse = float(np.mean((rollout_qpos_raw[:common_horizon] - replay_qpos_raw[:common_horizon]) ** 2))
                mom_mse = float(np.mean((rollout_mom[:common_horizon] - replay_mom[:common_horizon]) ** 2))
                ee_mse = float(
                    np.mean(
                        (
                            fingertip_xy_from_qpos_raw(rollout_qpos_raw[:common_horizon])
                            - fingertip_xy_from_qpos_raw(replay_qpos_raw[:common_horizon])
                        )
                        ** 2
                    )
                )

                workspace_plot = output_dir / f"traj_{traj_index:04d}_workspace.png"
                timeseries_plot = output_dir / f"traj_{traj_index:04d}_timeseries.png"
                build_workspace_plot(
                    rollout_qpos_raw=rollout_qpos_raw,
                    goal_xy=goal_xy,
                    goal_tolerance=float(args.goal_tolerance),
                    best_goal_distance=float(rollout_goal_dist.min()),
                    final_goal_distance=float(rollout_goal_dist[-1]),
                    title=(
                        f"traj={traj_index}  steps={len(rollout_qpos_raw)}  "
                        f"best_goal={float(rollout_goal_dist.min()):.4f}  "
                        f"final_goal={float(rollout_goal_dist[-1]):.4f}"
                    ),
                    save_path=workspace_plot,
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
                    "traj_index": int(traj_index),
                    "goal_xy": goal_xy.tolist(),
                    "goal_tolerance": float(args.goal_tolerance),
                    "num_candidates": int(args.num_candidates),
                    "num_diffusion_steps": int(args.num_diffusion_steps),
                    "max_prefix_len": int(rollout_limit),
                    "steps_taken": int(len(rollout_qpos_raw)),
                    "reached_goal": bool(reached_goal),
                    "best_goal_distance": float(rollout_goal_dist.min()),
                    "final_goal_distance": float(rollout_goal_dist[-1]),
                    "replay_best_goal_distance": float(replay_goal_dist.min()),
                    "qpos_mse_to_replay": qpos_mse,
                    "mom_mse_to_replay": mom_mse,
                    "ee_xy_mse_to_replay": ee_mse,
                    "workspace_plot": str(workspace_plot),
                    "timeseries_plot": str(timeseries_plot),
                    "rollout_qpos_raw": rollout_qpos_raw.tolist(),
                    "rollout_mom": rollout_mom.tolist(),
                    "rollout_tau": rollout_tau.tolist(),
                    "selection_trace": selection_trace,
                }
                summary_rows.append(row)
                print(json.dumps({k: row[k] for k in row if k not in {"rollout_qpos_raw", "rollout_mom", "rollout_tau", "selection_trace"}}, indent=2))

    summary = {
        "checkpoint_path": args.checkpoint_path,
        "h5_path": args.h5_path,
        "device": str(device),
        "seed": int(args.seed),
        "sample_indices": sample_indices,
        "num_candidates": int(args.num_candidates),
        "num_diffusion_steps": int(args.num_diffusion_steps),
        "lookahead_steps": int(args.lookahead_steps),
        "goal_tolerance": float(args.goal_tolerance),
        "max_prefix_len": int(args.max_prefix_len),
        "aggregate": {
            "success_rate": float(np.mean([float(row["reached_goal"]) for row in summary_rows])),
            "best_goal_distance": aggregate_metric(summary_rows, "best_goal_distance"),
            "final_goal_distance": aggregate_metric(summary_rows, "final_goal_distance"),
            "qpos_mse_to_replay": aggregate_metric(summary_rows, "qpos_mse_to_replay"),
            "mom_mse_to_replay": aggregate_metric(summary_rows, "mom_mse_to_replay"),
            "ee_xy_mse_to_replay": aggregate_metric(summary_rows, "ee_xy_mse_to_replay"),
        },
        "rows": summary_rows,
    }
    summary_path = output_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[GoalExpand] summary={summary_path}")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
