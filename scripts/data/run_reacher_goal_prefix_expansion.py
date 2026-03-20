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
from matplotlib import animation
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from scripts.data.generate_bidirectional_reacher_dataset import (
    generate_smooth_torque,
    get_model_ids,
    ik_2link,
    sample_waypoint_in_disk,
    set_reacher_state,
    simulate_helper_segment,
)
from src.models.trajectory_dpf_model import TrajectoryDPF
from src.qpos_representation import decode_qpos_tensor, encode_qpos_array


L1 = 0.10
L2 = 0.11
ARM_COLOR = "#2a6f97"
EE_PATH_COLOR = "#f4a261"
GOAL_COLOR = "#d62828"
START_COLOR = "#2a9d8f"
END_COLOR = "#264653"
TASK_MODE_DATASET = "goal&initial_state_from_dataset"
TASK_MODE_SAME_METHOD = "goal&initial_state_from_same_dataset_method"
TASK_MODE_INDEPENDENT = "goal&initial_state_from_independent_sampling"


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
        "--task_ids",
        "--sample_indices",
        dest="task_ids",
        type=str,
        default="313,267,787,338",
        help=(
            "Comma-separated task identifiers. "
            f"For task_mode={TASK_MODE_DATASET}, these are dataset trajectory indices. "
            f"For task_mode={TASK_MODE_SAME_METHOD} or {TASK_MODE_INDEPENDENT}, these are RNG seeds."
        ),
    )
    parser.add_argument(
        "--task_mode",
        type=str,
        default=TASK_MODE_DATASET,
        help=(
            "How to choose the goal and initial prefix state. "
            f"Canonical modes: {TASK_MODE_DATASET}, {TASK_MODE_SAME_METHOD}, {TASK_MODE_INDEPENDENT}. "
            "Older aliases are still accepted."
        ),
    )
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument(
        "--max_sampling_retries",
        type=int,
        default=3,
        help=(
            "If the best predicted suffix does not improve on the current goal distance, "
            "sample another candidate batch up to this many retries."
        ),
    )
    parser.add_argument(
        "--retry_improvement_margin",
        type=float,
        default=1e-3,
        help="Required predicted improvement margin to stop retrying candidate sampling.",
    )
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--goal_tolerance", type=float, default=0.01)
    parser.add_argument(
        "--waypoint_radius",
        type=float,
        default=None,
        help=(
            f"Goal sampling radius for {TASK_MODE_SAME_METHOD} and {TASK_MODE_INDEPENDENT}. "
            "Default: from dataset generator config or 0.18."
        ),
    )
    parser.add_argument(
        "--waypoint_qvel_scale",
        type=float,
        default=None,
        help=(
            f"Waypoint-velocity scale for {TASK_MODE_SAME_METHOD}. "
            "Default: from dataset generator config or 0.8."
        ),
    )
    parser.add_argument(
        "--torque_scale",
        type=float,
        default=None,
        help=(
            f"Torque scale for {TASK_MODE_SAME_METHOD}. "
            "Default: from dataset generator config or 0.2."
        ),
    )
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
        "--ood_qpos_quantile_low",
        type=float,
        default=0.01,
        help=f"Lower quantile for independent qpos sampling in {TASK_MODE_INDEPENDENT} mode.",
    )
    parser.add_argument(
        "--ood_qpos_quantile_high",
        type=float,
        default=0.99,
        help=f"Upper quantile for independent qpos sampling in {TASK_MODE_INDEPENDENT} mode.",
    )
    parser.add_argument(
        "--ood_mom_quantile_low",
        type=float,
        default=0.01,
        help=f"Lower quantile for independent momentum sampling in {TASK_MODE_INDEPENDENT} mode.",
    )
    parser.add_argument(
        "--ood_mom_quantile_high",
        type=float,
        default=0.99,
        help=f"Upper quantile for independent momentum sampling in {TASK_MODE_INDEPENDENT} mode.",
    )
    parser.add_argument(
        "--ood_stats_trajectories",
        type=int,
        default=2000,
        help="Maximum number of dataset trajectories to use when estimating OOD sampling bounds.",
    )
    parser.add_argument("--gif_fps", type=int, default=18)
    parser.add_argument("--gif_max_frames", type=int, default=200)
    parser.add_argument(
        "--recent_prefix_cap",
        type=int,
        default=256,
        help=(
            "When sampling candidate suffixes, condition only on the most recent "
            "min(current_prefix_len, recent_prefix_cap) clean steps. "
            "Use 0 or a negative value to keep the full prefix."
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


def normalize_task_mode(task_mode: str) -> str:
    valid_modes = [TASK_MODE_DATASET, TASK_MODE_SAME_METHOD, TASK_MODE_INDEPENDENT]
    if task_mode not in valid_modes:
        valid = ", ".join(valid_modes)
        raise ValueError(f"Unsupported task_mode={task_mode!r}. Supported values: {valid}")
    return task_mode


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


def resolve_generator_defaults(h5_file: h5py.File, args: argparse.Namespace) -> dict[str, float]:
    config_raw = h5_file.attrs.get("generator_config", "")
    config = json.loads(config_raw) if config_raw else {}
    return {
        "waypoint_radius": float(
            args.waypoint_radius if args.waypoint_radius is not None else config.get("waypoint_radius", 0.18)
        ),
        "waypoint_qvel_scale": float(
            args.waypoint_qvel_scale
            if args.waypoint_qvel_scale is not None
            else config.get("waypoint_qvel_scale", 0.8)
        ),
        "torque_scale": float(args.torque_scale if args.torque_scale is not None else config.get("torque_scale", 0.2)),
    }


def compute_arm_momentum_from_qvel(
    model: mujoco.MjModel,
    ids: dict[str, int],
    arm_qpos: np.ndarray,
    arm_qvel: np.ndarray,
    waypoint_xy: np.ndarray,
) -> np.ndarray:
    data = mujoco.MjData(model)
    set_reacher_state(model, data, arm_qpos, arm_qvel, waypoint_xy, ids)
    mass = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, mass, data.qM)
    mom = mass @ data.qvel
    return mom[[ids["joint0_dof"], ids["joint1_dof"]]].astype(np.float64, copy=False)


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


def effective_recent_prefix_len(prefix_len: int, recent_prefix_cap: int) -> int:
    if int(recent_prefix_cap) <= 0:
        return int(prefix_len)
    return int(min(int(prefix_len), int(recent_prefix_cap)))


def setup_workspace_axis(ax: plt.Axes, goal_xy: np.ndarray, goal_tolerance: float) -> None:
    ax.add_patch(plt.Circle((0.0, 0.0), L1 + L2, color="#bbbbbb", fill=False, linestyle="--", linewidth=1.0))
    ax.add_patch(plt.Circle((0.0, 0.0), abs(L1 - L2), color="#dddddd", fill=False, linestyle=":", linewidth=1.0))
    ax.scatter(goal_xy[0], goal_xy[1], color=GOAL_COLOR, s=180, marker="*", label="goal", zorder=8)
    ax.add_patch(
        plt.Circle(
            (float(goal_xy[0]), float(goal_xy[1])),
            radius=float(goal_tolerance),
            color=GOAL_COLOR,
            fill=False,
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
        )
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


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
    setup_workspace_axis(ax, goal_xy, goal_tolerance)

    ee_xy = fingertip_xy_from_qpos_raw(rollout_qpos_raw)
    ax.plot(ee_xy[:, 0], ee_xy[:, 1], color=EE_PATH_COLOR, linewidth=1.6, alpha=0.95, label="end effector path")

    num_draw = min(18, len(rollout_qpos_raw))
    draw_idx = np.linspace(0, len(rollout_qpos_raw) - 1, num=num_draw, dtype=int)
    for order, idx in enumerate(draw_idx):
        pts = arm_points_from_qpos_raw(rollout_qpos_raw[idx])
        alpha = 0.18 + 0.72 * (order + 1) / max(1, num_draw)
        ax.plot(pts[:, 0], pts[:, 1], color=ARM_COLOR, linewidth=1.6, alpha=alpha)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=ARM_COLOR, s=8, alpha=alpha)

    ax.scatter(ee_xy[0, 0], ee_xy[0, 1], color=START_COLOR, s=70, marker="o", label="start", zorder=6)
    ax.scatter(ee_xy[-1, 0], ee_xy[-1, 1], color=END_COLOR, s=70, marker="X", label="end", zorder=7)
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


def build_workspace_gif(
    *,
    rollout_qpos_raw: np.ndarray,
    goal_xy: np.ndarray,
    goal_tolerance: float,
    goal_distance: np.ndarray,
    title: str,
    save_path: Path,
    fps: int,
    max_frames: int,
) -> None:
    frame_idx = np.linspace(
        0,
        len(rollout_qpos_raw) - 1,
        num=min(max_frames, len(rollout_qpos_raw)),
        dtype=int,
    )
    frame_idx = np.unique(frame_idx)
    ee_xy = fingertip_xy_from_qpos_raw(rollout_qpos_raw)

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 6.6), dpi=140)
    setup_workspace_axis(ax, goal_xy, goal_tolerance)
    path_line, = ax.plot([], [], color=EE_PATH_COLOR, linewidth=1.8, alpha=0.95, label="end effector path")
    arm_line, = ax.plot([], [], color=ARM_COLOR, linewidth=2.2, alpha=0.95)
    start_marker = ax.scatter([], [], color=START_COLOR, s=70, marker="o", zorder=6)
    end_marker = ax.scatter([], [], color=END_COLOR, s=70, marker="X", zorder=7)
    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=13)

    def _set_scatter(scatter_obj, xy: np.ndarray) -> None:
        scatter_obj.set_offsets(np.asarray(xy, dtype=np.float64).reshape(1, 2))

    def init() -> tuple:
        path_line.set_data([], [])
        arm_line.set_data([], [])
        _set_scatter(start_marker, ee_xy[0])
        _set_scatter(end_marker, ee_xy[0])
        status_text.set_text("")
        return path_line, arm_line, start_marker, end_marker, status_text

    def update(frame_pos: int) -> tuple:
        step_idx = int(frame_idx[frame_pos])
        path_line.set_data(ee_xy[: step_idx + 1, 0], ee_xy[: step_idx + 1, 1])
        arm_pts = arm_points_from_qpos_raw(rollout_qpos_raw[step_idx])
        arm_line.set_data(arm_pts[:, 0], arm_pts[:, 1])
        _set_scatter(start_marker, ee_xy[0])
        _set_scatter(end_marker, ee_xy[step_idx])
        status_text.set_text(f"step={step_idx}  goal_dist={float(goal_distance[step_idx]):.4f}")
        return path_line, arm_line, start_marker, end_marker, status_text

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frame_idx),
        interval=max(1, int(round(1000.0 / max(1, fps)))),
        blit=False,
    )
    writer = animation.PillowWriter(fps=max(1, fps))
    ani.save(save_path, writer=writer)
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
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return {}
    values = np.asarray([float(value) for value in values], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def estimate_ood_state_bounds(
    h5_file: h5py.File,
    max_trajectories: int,
    qpos_quantiles: tuple[float, float],
    mom_quantiles: tuple[float, float],
) -> dict[str, np.ndarray]:
    traj_names = sorted(
        (name for name in h5_file.keys() if name.startswith("traj_")),
        key=lambda name: int(name.split("_")[1]),
    )[: max(1, int(max_trajectories))]
    qpos_samples = []
    mom_samples = []
    for name in traj_names:
        traj = h5_file[name]
        qpos_samples.append(traj["seq_qpos"][:].astype(np.float64, copy=False))
        mom_samples.append(traj["seq_mom"][:].astype(np.float64, copy=False))
    qpos_flat = np.concatenate(qpos_samples, axis=0)
    mom_flat = np.concatenate(mom_samples, axis=0)
    return {
        "qpos_low": np.quantile(qpos_flat, qpos_quantiles[0], axis=0),
        "qpos_high": np.quantile(qpos_flat, qpos_quantiles[1], axis=0),
        "mom_low": np.quantile(mom_flat, mom_quantiles[0], axis=0),
        "mom_high": np.quantile(mom_flat, mom_quantiles[1], axis=0),
        "num_trajectories_used": np.array([len(traj_names)], dtype=np.int64),
    }


def build_dataset_start_task(
    *,
    h5_file: h5py.File,
    traj_index: int,
    rollout_limit: int,
    model: TrajectoryDPF,
) -> dict:
    traj = h5_file[f"traj_{traj_index}"]
    replay_qpos_raw = traj["seq_qpos"][:rollout_limit].astype(np.float64)
    replay_mom = traj["seq_mom"][:rollout_limit].astype(np.float64)
    goal_xy = traj["waypoint_xy"][:].astype(np.float64)
    return {
        "task_id": int(traj_index),
        "task_label": f"traj_{traj_index:04d}",
        "task_mode": TASK_MODE_DATASET,
        "goal_xy": goal_xy,
        "initial_qpos_raw": replay_qpos_raw[0].copy(),
        "initial_mom": replay_mom[0].copy(),
        "reference_qpos_raw": replay_qpos_raw,
        "reference_mom": replay_mom,
        "reference_waypoint_index": int(traj.attrs.get("waypoint_index", -1)),
        "metadata": {
            "traj_index": int(traj_index),
            "waypoint_index": int(traj.attrs.get("waypoint_index", -1)),
            "waypoint_error": float(traj.attrs.get("waypoint_error", np.nan)),
        },
    }


def build_generator_id_task(
    *,
    task_seed: int,
    rollout_limit: int,
    mj_model: mujoco.MjModel,
    ids: dict[str, int],
    generator_defaults: dict[str, float],
) -> dict:
    rng = np.random.default_rng(int(task_seed))
    waypoint_xy = sample_waypoint_in_disk(rng, max_radius=float(generator_defaults["waypoint_radius"]))
    elbow_branch = 1 if rng.random() < 0.5 else -1
    arm_qpos = ik_2link(waypoint_xy, elbow_branch=elbow_branch)
    arm_qvel = rng.uniform(
        -float(generator_defaults["waypoint_qvel_scale"]),
        float(generator_defaults["waypoint_qvel_scale"]),
        size=2,
    )
    prefix_steps = int(rng.integers(1, rollout_limit - 1))
    full_tau = generate_smooth_torque(
        rng,
        rollout_limit,
        float(mj_model.opt.timestep),
        mj_model.nu,
        float(generator_defaults["torque_scale"]),
    )
    prefix_helper_tau = full_tau[:prefix_steps][::-1].copy()
    prefix_qpos_helper, prefix_qvel_helper = simulate_helper_segment(
        model=mj_model,
        arm_qpos=arm_qpos,
        arm_qvel=-arm_qvel,
        waypoint_xy=waypoint_xy,
        torque_seq=prefix_helper_tau,
        ids=ids,
    )
    start_arm_qpos = prefix_qpos_helper[-1, [ids["joint0_qpos"], ids["joint1_qpos"]]]
    start_arm_qvel = -prefix_qvel_helper[-1, [ids["joint0_dof"], ids["joint1_dof"]]]
    start_arm_mom = compute_arm_momentum_from_qvel(
        mj_model,
        ids,
        arm_qpos=start_arm_qpos,
        arm_qvel=start_arm_qvel,
        waypoint_xy=waypoint_xy,
    )
    return {
        "task_id": int(task_seed),
        "task_label": f"id_seed_{task_seed:06d}",
        "task_mode": TASK_MODE_SAME_METHOD,
        "goal_xy": waypoint_xy.astype(np.float64),
        "initial_qpos_raw": start_arm_qpos.astype(np.float64, copy=False),
        "initial_mom": start_arm_mom.astype(np.float64, copy=False),
        "reference_qpos_raw": None,
        "reference_mom": None,
        "reference_waypoint_index": prefix_steps,
        "metadata": {
            "task_seed": int(task_seed),
            "waypoint_index": int(prefix_steps),
            "elbow_branch": int(elbow_branch),
            "waypoint_qvel": arm_qvel.astype(np.float64).tolist(),
        },
    }


def build_ood_random_state_task(
    *,
    task_seed: int,
    mj_model: mujoco.MjModel,
    generator_defaults: dict[str, float],
    qpos_bounds: tuple[np.ndarray, np.ndarray],
    mom_bounds: tuple[np.ndarray, np.ndarray],
) -> dict:
    rng = np.random.default_rng(int(task_seed))
    goal_xy = sample_waypoint_in_disk(rng, max_radius=float(generator_defaults["waypoint_radius"]))
    qpos_low, qpos_high = qpos_bounds
    mom_low, mom_high = mom_bounds
    initial_qpos_raw = rng.uniform(qpos_low, qpos_high).astype(np.float64)
    initial_mom = rng.uniform(mom_low, mom_high).astype(np.float64)
    return {
        "task_id": int(task_seed),
        "task_label": f"ood_seed_{task_seed:06d}",
        "task_mode": TASK_MODE_INDEPENDENT,
        "goal_xy": goal_xy.astype(np.float64),
        "initial_qpos_raw": initial_qpos_raw,
        "initial_mom": initial_mom,
        "reference_qpos_raw": None,
        "reference_mom": None,
        "reference_waypoint_index": None,
        "metadata": {
            "task_seed": int(task_seed),
            "qpos_range_low": np.asarray(qpos_low, dtype=np.float64).tolist(),
            "qpos_range_high": np.asarray(qpos_high, dtype=np.float64).tolist(),
            "mom_range_low": np.asarray(mom_low, dtype=np.float64).tolist(),
            "mom_range_high": np.asarray(mom_high, dtype=np.float64).tolist(),
            "nu": int(mj_model.nu),
        },
    }


def main() -> None:
    args = parse_args()
    args.task_mode = normalize_task_mode(args.task_mode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_ids = parse_indices(args.task_ids)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[GoalExpand] device={device}")
    print(f"[GoalExpand] output_dir={output_dir}")
    print(f"[GoalExpand] task_mode={args.task_mode}")
    print(f"[GoalExpand] task_ids={task_ids}")

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
            ids = get_model_ids(mj_model)
            stepper = ReacherRolloutStepper(mj_model, dt=float(model.dt), data_dt=float(model.data_dt))
            generator_defaults = resolve_generator_defaults(h5_file, args)
            ood_bounds = None
            if args.task_mode == TASK_MODE_INDEPENDENT:
                bounds = estimate_ood_state_bounds(
                    h5_file,
                    max_trajectories=int(args.ood_stats_trajectories),
                    qpos_quantiles=(float(args.ood_qpos_quantile_low), float(args.ood_qpos_quantile_high)),
                    mom_quantiles=(float(args.ood_mom_quantile_low), float(args.ood_mom_quantile_high)),
                )
                ood_bounds = (
                    (bounds["qpos_low"], bounds["qpos_high"]),
                    (bounds["mom_low"], bounds["mom_high"]),
                )
                print(
                    "[GoalExpand] OOD bounds from {} trajectories: qpos_low={} qpos_high={} mom_low={} mom_high={}".format(
                        int(bounds["num_trajectories_used"][0]),
                        np.asarray(bounds["qpos_low"]).tolist(),
                        np.asarray(bounds["qpos_high"]).tolist(),
                        np.asarray(bounds["mom_low"]).tolist(),
                        np.asarray(bounds["mom_high"]).tolist(),
                    )
                )

            summary_rows: list[dict] = []
            for task_id in task_ids:
                if args.task_mode == TASK_MODE_DATASET:
                    task = build_dataset_start_task(
                        h5_file=h5_file,
                        traj_index=int(task_id),
                        rollout_limit=rollout_limit,
                        model=model,
                    )
                elif args.task_mode == TASK_MODE_SAME_METHOD:
                    task = build_generator_id_task(
                        task_seed=int(args.seed) + int(task_id),
                        rollout_limit=rollout_limit,
                        mj_model=mj_model,
                        ids=ids,
                        generator_defaults=generator_defaults,
                    )
                else:
                    assert ood_bounds is not None
                    task = build_ood_random_state_task(
                        task_seed=int(args.seed) + int(task_id),
                        mj_model=mj_model,
                        generator_defaults=generator_defaults,
                        qpos_bounds=ood_bounds[0],
                        mom_bounds=ood_bounds[1],
                    )

                goal_xy = np.asarray(task["goal_xy"], dtype=np.float64)
                initial_qpos_raw = np.asarray(task["initial_qpos_raw"], dtype=np.float64)
                initial_mom = np.asarray(task["initial_mom"], dtype=np.float64)
                reference_qpos_raw = task["reference_qpos_raw"]
                reference_mom = task["reference_mom"]

                initial_qpos_model = encode_qpos_array(
                    initial_qpos_raw[None, :].astype(np.float32),
                    model.qpos_representation,
                )[0].astype(np.float32, copy=False)
                initial_mom_f32 = initial_mom.astype(np.float32, copy=False)

                qpos_prefix_raw = [initial_qpos_raw.copy()]
                mom_prefix = [initial_mom.copy()]
                torque_prefix = []
                rollout_goal_distances = [
                    float(np.linalg.norm(fingertip_xy_from_qpos_raw(initial_qpos_raw[None, :])[0] - goal_xy))
                ]
                selection_trace: list[dict] = []
                reached_goal = rollout_goal_distances[-1] <= float(args.goal_tolerance)
                observed_qpos = torch.zeros(
                    (1, rollout_limit, model.qpos_dim),
                    dtype=torch.float32,
                    device=device,
                )
                observed_mom = torch.zeros(
                    (1, rollout_limit, model.mom_dim),
                    dtype=torch.float32,
                    device=device,
                )
                observed_tau = torch.zeros(
                    (1, rollout_limit, model.torque_dim),
                    dtype=torch.float32,
                    device=device,
                )
                observed_qpos[0, 0] = torch.from_numpy(initial_qpos_model).to(device=device)
                observed_mom[0, 0] = torch.from_numpy(initial_mom_f32).to(device=device)

                while len(qpos_prefix_raw) < rollout_limit and not reached_goal:
                    prefix_len = len(qpos_prefix_raw)
                    sample_horizon_abs = effective_sampling_horizon(
                        prefix_len=prefix_len,
                        rollout_limit=rollout_limit,
                        lookahead_steps=int(args.lookahead_steps),
                    )
                    conditioning_prefix_len = effective_recent_prefix_len(
                        prefix_len=prefix_len,
                        recent_prefix_cap=int(args.recent_prefix_cap),
                    )
                    crop_start = prefix_len - conditioning_prefix_len
                    sample_horizon = sample_horizon_abs - crop_start
                    time_indices = torch.arange(
                        crop_start,
                        sample_horizon_abs,
                        dtype=torch.long,
                        device=device,
                    )
                    current_goal_distance = float(rollout_goal_distances[-1])
                    required_goal_distance = current_goal_distance - float(args.retry_improvement_margin)
                    goal_xy_t = torch.as_tensor(goal_xy, dtype=torch.float32, device=device)

                    best_candidate_idx = -1
                    best_retry_idx = -1
                    best_retry_seed = -1
                    best_candidate_goal_distance = float("inf")
                    best_generated_tau = None
                    retry_trace: list[dict] = []

                    for retry_idx in range(max(0, int(args.max_sampling_retries)) + 1):
                        local_seed = (
                            int(args.seed)
                            + int(task["task_id"]) * 1000
                            + prefix_len * 100
                            + retry_idx
                        )
                        torch.manual_seed(local_seed)
                        np.random.seed(local_seed)

                        with torch.no_grad():
                            generated_state, generated_tau = model.sample_trajectories(
                                num_samples=int(args.num_candidates),
                                trajectory_length=sample_horizon,
                                num_diffusion_steps=int(args.num_diffusion_steps),
                                sample_mode="observed_prefix_completion",
                                prefix_len=conditioning_prefix_len,
                                observed_qpos=observed_qpos[:, crop_start:sample_horizon_abs, :],
                                observed_mom=observed_mom[:, crop_start:sample_horizon_abs, :],
                                observed_torque=observed_tau[:, crop_start:sample_horizon_abs, :],
                                time_indices=time_indices,
                                use_ema=False,
                                sampler="ddim",
                            )

                        candidate_qpos_model = generated_state[:, conditioning_prefix_len:, : model.qpos_dim]
                        candidate_qpos_raw = decode_qpos_tensor(candidate_qpos_model, model.qpos_representation)
                        candidate_suffix_xy = fingertip_xy_from_qpos_tensor(candidate_qpos_raw)
                        candidate_min_goal_dist = torch.linalg.norm(
                            candidate_suffix_xy - goal_xy_t.view(1, 1, 2),
                            dim=-1,
                        ).amin(dim=1)
                        retry_best_idx = int(torch.argmin(candidate_min_goal_dist).item())
                        retry_best_dist = float(candidate_min_goal_dist[retry_best_idx].item())
                        retry_trace.append(
                            {
                                "retry_idx": retry_idx,
                                "sample_seed": local_seed,
                                "best_candidate_idx": retry_best_idx,
                                "best_predicted_goal_distance": retry_best_dist,
                                "improved_over_current": bool(retry_best_dist < required_goal_distance),
                            }
                        )

                        if retry_best_dist < best_candidate_goal_distance:
                            best_candidate_goal_distance = retry_best_dist
                            best_candidate_idx = retry_best_idx
                            best_retry_idx = retry_idx
                            best_retry_seed = local_seed
                            best_generated_tau = generated_tau.detach().cpu().numpy()

                        if retry_best_dist < required_goal_distance:
                            break

                    if best_generated_tau is None or best_candidate_idx < 0:
                        raise RuntimeError("Adaptive retry loop failed to produce any candidate torque.")

                    applied_tau = best_generated_tau[best_candidate_idx, conditioning_prefix_len - 1].astype(
                        np.float64,
                        copy=False,
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
                            "conditioning_prefix_len": int(conditioning_prefix_len),
                            "conditioning_crop_start": int(crop_start),
                            "sample_horizon": int(sample_horizon),
                            "sample_horizon_absolute": int(sample_horizon_abs),
                            "sample_seed": best_retry_seed,
                            "current_goal_distance": current_goal_distance,
                            "required_goal_distance": required_goal_distance,
                            "num_retries_used": best_retry_idx,
                            "num_sampling_attempts": len(retry_trace),
                            "total_candidates_evaluated": int(len(retry_trace) * int(args.num_candidates)),
                            "retry_trace": retry_trace,
                            "chosen_retry_idx": best_retry_idx,
                            "chosen_candidate_idx": best_candidate_idx,
                            "chosen_candidate_best_goal_dist": best_candidate_goal_distance,
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
                rollout_goal_dist = np.asarray(rollout_goal_distances, dtype=np.float64)
                replay_goal_dist = None
                qpos_mse = None
                mom_mse = None
                ee_mse = None
                if reference_qpos_raw is not None and reference_mom is not None:
                    replay_goal_dist = np.linalg.norm(
                        fingertip_xy_from_qpos_raw(reference_qpos_raw) - goal_xy[None, :],
                        axis=-1,
                    )
                    common_horizon = min(len(rollout_qpos_raw), len(reference_qpos_raw))
                    qpos_mse = float(
                        np.mean((rollout_qpos_raw[:common_horizon] - reference_qpos_raw[:common_horizon]) ** 2)
                    )
                    mom_mse = float(
                        np.mean((rollout_mom[:common_horizon] - reference_mom[:common_horizon]) ** 2)
                    )
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
                        f"{task['task_label']}  steps={len(rollout_qpos_raw)}  "
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
                    "num_candidates": int(args.num_candidates),
                    "num_diffusion_steps": int(args.num_diffusion_steps),
                    "lookahead_steps": int(args.lookahead_steps),
                    "recent_prefix_cap": int(args.recent_prefix_cap),
                    "max_prefix_len": int(rollout_limit),
                    "steps_taken": int(len(rollout_qpos_raw)),
                    "reached_goal": bool(reached_goal),
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

    summary = {
        "checkpoint_path": args.checkpoint_path,
        "h5_path": args.h5_path,
        "device": str(device),
        "seed": int(args.seed),
        "task_mode": str(args.task_mode),
        "task_ids": task_ids,
        "num_candidates": int(args.num_candidates),
        "max_sampling_retries": int(args.max_sampling_retries),
        "retry_improvement_margin": float(args.retry_improvement_margin),
        "num_diffusion_steps": int(args.num_diffusion_steps),
        "lookahead_steps": int(args.lookahead_steps),
        "recent_prefix_cap": int(args.recent_prefix_cap),
        "goal_tolerance": float(args.goal_tolerance),
        "max_prefix_len": int(args.max_prefix_len),
        "generator_defaults": generator_defaults,
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
