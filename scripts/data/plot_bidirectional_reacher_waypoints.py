#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


L1 = 0.10
L2 = 0.11


def sample_waypoint_in_disk(rng: np.random.Generator, max_radius: float) -> np.ndarray:
    while True:
        xy = rng.uniform(-max_radius, max_radius, size=2)
        if np.linalg.norm(xy) <= max_radius:
            return xy


def ik_2link(xy: np.ndarray, elbow_branch: int) -> np.ndarray:
    x, y = float(xy[0]), float(xy[1])
    r2 = x * x + y * y
    cos_q2 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_q2 = float(np.clip(cos_q2, -1.0, 1.0))
    base = math.acos(cos_q2)
    q2 = base if elbow_branch >= 0 else -base
    q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    return np.array([q1, q2], dtype=np.float64)


def generate_smooth_torque(
    rng: np.random.Generator,
    num_steps: int,
    dt: float,
    nu: int,
    scale: float,
    num_sin: int = 4,
) -> np.ndarray:
    t = np.arange(num_steps, dtype=np.float64) * dt
    torque = np.zeros((num_steps, nu), dtype=np.float64)
    for dim in range(nu):
        for _ in range(num_sin):
            amp = rng.uniform(0.15 * scale, scale)
            freq = rng.uniform(0.4, 2.6)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            torque[:, dim] += amp * np.sin(2.0 * math.pi * freq * t + phase)
    return np.clip(torque, -0.9, 0.9)


def set_reacher_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_qpos: np.ndarray,
    arm_qvel: np.ndarray,
    waypoint_xy: np.ndarray,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    joint0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint0")
    joint1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    target_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_x")
    target_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_y")

    qpos_ids = [
        int(model.jnt_qposadr[joint0]),
        int(model.jnt_qposadr[joint1]),
        int(model.jnt_qposadr[target_x]),
        int(model.jnt_qposadr[target_y]),
    ]
    qvel_ids = [
        int(model.jnt_dofadr[joint0]),
        int(model.jnt_dofadr[joint1]),
        int(model.jnt_dofadr[target_x]),
        int(model.jnt_dofadr[target_y]),
    ]

    data.qpos[qpos_ids[0]] = arm_qpos[0]
    data.qpos[qpos_ids[1]] = arm_qpos[1]
    data.qpos[qpos_ids[2]] = waypoint_xy[0]
    data.qpos[qpos_ids[3]] = waypoint_xy[1]
    data.qvel[qvel_ids[0]] = arm_qvel[0]
    data.qvel[qvel_ids[1]] = arm_qvel[1]
    data.qvel[qvel_ids[2]] = 0.0
    data.qvel[qvel_ids[3]] = 0.0
    mujoco.mj_forward(model, data)


def simulate_segment(
    model: mujoco.MjModel,
    arm_qpos: np.ndarray,
    arm_qvel: np.ndarray,
    waypoint_xy: np.ndarray,
    torque_seq: np.ndarray,
    fingertip_body_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    set_reacher_state(model, data, arm_qpos, arm_qvel, waypoint_xy)

    qpos_hist = np.empty((torque_seq.shape[0] + 1, model.nq), dtype=np.float64)
    qvel_hist = np.empty((torque_seq.shape[0] + 1, model.nv), dtype=np.float64)
    xy_hist = np.empty((torque_seq.shape[0] + 1, 2), dtype=np.float64)

    qpos_hist[0] = data.qpos.copy()
    qvel_hist[0] = data.qvel.copy()
    xy_hist[0] = data.xpos[fingertip_body_id, :2]

    for step, tau in enumerate(torque_seq, start=1):
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        qpos_hist[step] = data.qpos.copy()
        qvel_hist[step] = data.qvel.copy()
        xy_hist[step] = data.xpos[fingertip_body_id, :2]

    return qpos_hist, qvel_hist, xy_hist


def build_bidirectional_example(
    model: mujoco.MjModel,
    rng: np.random.Generator,
    trajectory_length: int,
    waypoint_radius: float,
    waypoint_qvel_scale: float,
    torque_scale: float,
    fingertip_body_id: int,
) -> dict:
    if trajectory_length < 3:
        raise ValueError("trajectory_length must be at least 3 so prefix, waypoint, suffix all exist")

    waypoint_xy = sample_waypoint_in_disk(rng, max_radius=waypoint_radius)
    elbow_branch = 1 if rng.random() < 0.5 else -1
    arm_qpos = ik_2link(waypoint_xy, elbow_branch=elbow_branch)
    arm_qvel = rng.uniform(-waypoint_qvel_scale, waypoint_qvel_scale, size=2)
    prefix_steps = int(rng.integers(1, trajectory_length - 1))
    suffix_steps = int(trajectory_length - 1 - prefix_steps)

    prefix_helper_tau = generate_smooth_torque(rng, prefix_steps, model.opt.timestep, model.nu, torque_scale)
    suffix_tau = generate_smooth_torque(rng, suffix_steps, model.opt.timestep, model.nu, torque_scale)

    prefix_qpos_helper, prefix_qvel_helper, prefix_xy_helper = simulate_segment(
        model=model,
        arm_qpos=arm_qpos,
        arm_qvel=-arm_qvel,
        waypoint_xy=waypoint_xy,
        torque_seq=prefix_helper_tau,
        fingertip_body_id=fingertip_body_id,
    )
    suffix_qpos, suffix_qvel, suffix_xy = simulate_segment(
        model=model,
        arm_qpos=arm_qpos,
        arm_qvel=arm_qvel,
        waypoint_xy=waypoint_xy,
        torque_seq=suffix_tau,
        fingertip_body_id=fingertip_body_id,
    )

    prefix_qpos = prefix_qpos_helper[::-1].copy()
    prefix_qvel = -prefix_qvel_helper[::-1].copy()
    prefix_xy = prefix_xy_helper[::-1].copy()

    full_qpos = np.concatenate([prefix_qpos[:-1], suffix_qpos], axis=0)
    full_qvel = np.concatenate([prefix_qvel[:-1], suffix_qvel], axis=0)
    full_xy = np.concatenate([prefix_xy[:-1], suffix_xy], axis=0)
    waypoint_index = prefix_xy.shape[0] - 1

    return {
        "waypoint_xy": waypoint_xy,
        "arm_qpos": arm_qpos,
        "arm_qvel": arm_qvel,
        "prefix_xy": prefix_xy,
        "suffix_xy": suffix_xy,
        "full_xy": full_xy,
        "full_qpos": full_qpos,
        "full_qvel": full_qvel,
        "waypoint_index": waypoint_index,
        "prefix_steps": prefix_steps,
        "suffix_steps": suffix_steps,
        "prefix_tau": prefix_helper_tau,
        "suffix_tau": suffix_tau,
        "start_xy": full_xy[0],
        "end_xy": full_xy[-1],
    }


def _draw_workspace_guides(ax: plt.Axes) -> None:
    ax.add_patch(plt.Circle((0.0, 0.0), L1 + L2, color="#bbbbbb", fill=False, linestyle="--", linewidth=1.0))
    ax.add_patch(plt.Circle((0.0, 0.0), abs(L1 - L2), color="#dddddd", fill=False, linestyle=":", linewidth=1.0))
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_examples(examples: list[dict], save_path: Path) -> None:
    n = len(examples)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 6.5 * nrows), dpi=180)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, example in zip(axes.flat, examples):
        _draw_workspace_guides(ax)

        prefix_xy = example["prefix_xy"]
        suffix_xy = example["suffix_xy"]
        waypoint_xy = example["waypoint_xy"]

        ax.plot(prefix_xy[:, 0], prefix_xy[:, 1], color="#2a6f97", linewidth=2.0, label="prefix")
        ax.plot(suffix_xy[:, 0], suffix_xy[:, 1], color="#ee6c4d", linewidth=2.0, label="suffix")
        ax.scatter(waypoint_xy[0], waypoint_xy[1], color="#d62828", s=140, marker="*", label="waypoint", zorder=5)

        ax.set_title(
            "waypoint=({:.3f}, {:.3f})".format(
                waypoint_xy[0],
                waypoint_xy[1],
            ),
            fontsize=9.5,
        )
        ax.legend(loc="upper right", fontsize=8)

    for ax in axes.flat[len(examples):]:
        ax.axis("off")

    fig.suptitle("Non-dissipative Reacher bidirectional trajectories around chosen waypoints", fontsize=14)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(examples: list[dict], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 8.2), dpi=220)
    _draw_workspace_guides(ax)

    all_points = np.concatenate([example["full_xy"] for example in examples], axis=0)
    all_waypoints = np.array([example["waypoint_xy"] for example in examples], dtype=np.float64)

    for example in examples:
        xy = example["full_xy"]
        ax.plot(xy[:, 0], xy[:, 1], color="#2a6f97", alpha=0.14, linewidth=0.9)

    ax.scatter(all_waypoints[:, 0], all_waypoints[:, 1], s=18, color="#d62828", alpha=0.65, marker="*", label="waypoints")
    ax.set_title(
        "Bidirectional non-dissipative Reacher overlay: {} trajectories".format(len(examples)),
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_with_torque(examples: list[dict], save_path: Path) -> None:
    fig = plt.figure(figsize=(14.0, 8.0), dpi=220)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0], wspace=0.18, hspace=0.18)
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_tau0 = fig.add_subplot(gs[0, 1])
    ax_tau1 = fig.add_subplot(gs[1, 1], sharex=ax_tau0)

    _draw_workspace_guides(ax_xy)
    all_waypoints = np.array([example["waypoint_xy"] for example in examples], dtype=np.float64)

    prefix_color = "#2a6f97"
    suffix_color = "#ee6c4d"

    for example in examples:
        prefix_xy = example["prefix_xy"]
        suffix_xy = example["suffix_xy"]
        ax_xy.plot(prefix_xy[:, 0], prefix_xy[:, 1], color=prefix_color, alpha=0.05, linewidth=0.7)
        ax_xy.plot(suffix_xy[:, 0], suffix_xy[:, 1], color=suffix_color, alpha=0.05, linewidth=0.7)

        prefix_steps = example["prefix_steps"]
        suffix_steps = example["suffix_steps"]
        prefix_t = np.arange(prefix_steps, dtype=np.int32)
        suffix_t = np.arange(prefix_steps, prefix_steps + suffix_steps, dtype=np.int32)
        prefix_tau = example["prefix_tau"]
        suffix_tau = example["suffix_tau"]

        ax_tau0.plot(prefix_t, prefix_tau[:, 0], color=prefix_color, alpha=0.025, linewidth=0.6)
        ax_tau0.plot(suffix_t, suffix_tau[:, 0], color=suffix_color, alpha=0.025, linewidth=0.6)
        ax_tau1.plot(prefix_t, prefix_tau[:, 1], color=prefix_color, alpha=0.025, linewidth=0.6)
        ax_tau1.plot(suffix_t, suffix_tau[:, 1], color=suffix_color, alpha=0.025, linewidth=0.6)

    ax_xy.scatter(all_waypoints[:, 0], all_waypoints[:, 1], s=8, color="#d62828", alpha=0.30, marker="*", label="waypoints")
    ax_xy.set_title(f"Task-Space Overlay: {len(examples)} trajectories", fontsize=13)
    ax_xy.legend(loc="upper right", fontsize=9)

    ax_tau0.set_title("Torque dim 0", fontsize=12)
    ax_tau1.set_title("Torque dim 1", fontsize=12)
    ax_tau0.set_ylabel("tau[0]")
    ax_tau1.set_ylabel("tau[1]")
    ax_tau1.set_xlabel("timestep")
    ax_tau0.grid(alpha=0.25)
    ax_tau1.grid(alpha=0.25)
    ax_tau0.plot([], [], color=prefix_color, label="prefix")
    ax_tau0.plot([], [], color=suffix_color, label="suffix")
    ax_tau0.legend(loc="upper right", fontsize=9)

    fig.suptitle("Bidirectional non-dissipative Reacher: positions and torques", fontsize=15)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def print_trajectory_stats(examples: list[dict], dt: float) -> None:
    waypoint_xy = np.array([example["waypoint_xy"] for example in examples], dtype=np.float64)
    waypoint_radius = np.linalg.norm(waypoint_xy, axis=1)
    prefix_steps = np.array([example["prefix_steps"] for example in examples], dtype=np.int32)
    suffix_steps = np.array([example["suffix_steps"] for example in examples], dtype=np.int32)
    waypoint_index = np.array([example["waypoint_index"] for example in examples], dtype=np.int32)

    full_xy = np.concatenate([example["full_xy"] for example in examples], axis=0)
    full_radius = np.linalg.norm(full_xy, axis=1)
    all_torque = np.concatenate(
        [
            np.concatenate([example["prefix_tau"], example["suffix_tau"]], axis=0)
            for example in examples
        ],
        axis=0,
    )

    print("---- Trajectory Statistics ----", flush=True)
    print(f"num_examples: {len(examples)}", flush=True)
    print(f"dt: {dt:.6f}", flush=True)
    print(f"trajectory_length: {examples[0]['full_xy'].shape[0]}", flush=True)
    print(
        "prefix_steps: min={} mean={:.2f} max={}".format(
            int(prefix_steps.min()), float(prefix_steps.mean()), int(prefix_steps.max())
        ),
        flush=True,
    )
    print(
        "suffix_steps: min={} mean={:.2f} max={}".format(
            int(suffix_steps.min()), float(suffix_steps.mean()), int(suffix_steps.max())
        ),
        flush=True,
    )
    print(
        "waypoint_index: min={} mean={:.2f} max={}".format(
            int(waypoint_index.min()), float(waypoint_index.mean()), int(waypoint_index.max())
        ),
        flush=True,
    )
    print(
        "waypoint_radius: min={:.4f} mean={:.4f} max={:.4f}".format(
            float(waypoint_radius.min()), float(waypoint_radius.mean()), float(waypoint_radius.max())
        ),
        flush=True,
    )
    print(
        "visited_radius: min={:.4f} mean={:.4f} max={:.4f}".format(
            float(full_radius.min()), float(full_radius.mean()), float(full_radius.max())
        ),
        flush=True,
    )
    print(
        "x_range: [{:.4f}, {:.4f}]".format(float(full_xy[:, 0].min()), float(full_xy[:, 0].max())),
        flush=True,
    )
    print(
        "y_range: [{:.4f}, {:.4f}]".format(float(full_xy[:, 1].min()), float(full_xy[:, 1].max())),
        flush=True,
    )
    for dim in range(all_torque.shape[1]):
        tau = all_torque[:, dim]
        print(
            "tau[{}]: min={:.4f} mean={:.4f} std={:.4f} max={:.4f}".format(
                dim,
                float(tau.min()),
                float(tau.mean()),
                float(tau.std()),
                float(tau.max()),
            ),
            flush=True,
        )
    print("-------------------------------", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot bidirectional non-dissipative Reacher waypoint examples")
    parser.add_argument(
        "--xml_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/configs/reacher_non_diss_from_dataset.xml",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/plots/bidirectional_reacher_waypoint_examples.png",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/plots/bidirectional_reacher_waypoint_examples.json",
    )
    parser.add_argument("--num_examples", type=int, default=1000)
    parser.add_argument("--plot_mode", type=str, choices=("grid", "overlay", "overlay_with_torque"), default="overlay_with_torque")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--show_stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory_length", type=int, default=1000)
    parser.add_argument("--waypoint_radius", type=float, default=0.18)
    parser.add_argument("--waypoint_qvel_scale", type=float, default=0.8)
    parser.add_argument("--torque_scale", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    if args.dt is not None:
        model.opt.timestep = float(args.dt)
    fingertip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
    rng = np.random.default_rng(args.seed)

    examples = [
        build_bidirectional_example(
            model=model,
            rng=rng,
            trajectory_length=args.trajectory_length,
            waypoint_radius=args.waypoint_radius,
            waypoint_qvel_scale=args.waypoint_qvel_scale,
            torque_scale=args.torque_scale,
            fingertip_body_id=fingertip_body_id,
        )
        for _ in range(args.num_examples)
    ]

    save_path = Path(args.save_path)
    summary_path = Path(args.summary_path)
    if args.plot_mode == "overlay":
        plot_overlay(examples, save_path)
    elif args.plot_mode == "overlay_with_torque":
        plot_overlay_with_torque(examples, save_path)
    else:
        plot_examples(examples, save_path)

    if args.show_stats:
        print_trajectory_stats(examples, dt=float(model.opt.timestep))

    summary = {
        "xml_path": args.xml_path,
        "seed": args.seed,
        "dt": float(model.opt.timestep),
        "num_examples": args.num_examples,
        "trajectory_length": args.trajectory_length,
        "waypoint_radius": args.waypoint_radius,
        "waypoint_qvel_scale": args.waypoint_qvel_scale,
        "torque_scale": args.torque_scale,
        "examples": [
            {
                "waypoint_xy": ex["waypoint_xy"].tolist(),
                "arm_qpos": ex["arm_qpos"].tolist(),
                "arm_qvel": ex["arm_qvel"].tolist(),
                "start_xy": ex["start_xy"].tolist(),
                "end_xy": ex["end_xy"].tolist(),
                "prefix_steps": int(ex["prefix_steps"]),
                "suffix_steps": int(ex["suffix_steps"]),
                "num_points": int(ex["full_xy"].shape[0]),
                "waypoint_index": int(ex["waypoint_index"]),
            }
            for ex in examples
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
