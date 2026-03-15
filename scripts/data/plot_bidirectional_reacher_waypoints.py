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
    prefix_steps: int,
    suffix_steps: int,
    waypoint_radius: float,
    waypoint_qvel_scale: float,
    torque_scale: float,
    fingertip_body_id: int,
) -> dict:
    waypoint_xy = sample_waypoint_in_disk(rng, max_radius=waypoint_radius)
    elbow_branch = 1 if rng.random() < 0.5 else -1
    arm_qpos = ik_2link(waypoint_xy, elbow_branch=elbow_branch)
    arm_qvel = rng.uniform(-waypoint_qvel_scale, waypoint_qvel_scale, size=2)

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
        "start_xy": full_xy[0],
        "end_xy": full_xy[-1],
    }


def plot_examples(examples: list[dict], save_path: Path) -> None:
    n = len(examples)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 6.5 * nrows), dpi=180)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    outer = plt.Circle((0.0, 0.0), L1 + L2, color="#bbbbbb", fill=False, linestyle="--", linewidth=1.0)
    inner = plt.Circle((0.0, 0.0), abs(L1 - L2), color="#dddddd", fill=False, linestyle=":", linewidth=1.0)
    target_disk = plt.Circle((0.0, 0.0), 0.20, color="#f2d6a2", fill=False, linestyle="-.", linewidth=1.0)

    for ax, example in zip(axes.flat, examples):
        ax.add_patch(plt.Circle((0.0, 0.0), L1 + L2, color="#bbbbbb", fill=False, linestyle="--", linewidth=1.0))
        ax.add_patch(plt.Circle((0.0, 0.0), abs(L1 - L2), color="#dddddd", fill=False, linestyle=":", linewidth=1.0))
        ax.add_patch(plt.Circle((0.0, 0.0), 0.20, color="#f2d6a2", fill=False, linestyle="-.", linewidth=1.0))

        prefix_xy = example["prefix_xy"]
        suffix_xy = example["suffix_xy"]
        waypoint_xy = example["waypoint_xy"]
        start_xy = example["start_xy"]
        end_xy = example["end_xy"]

        ax.plot(prefix_xy[:, 0], prefix_xy[:, 1], color="#2a6f97", linewidth=2.0, label="prefix")
        ax.plot(suffix_xy[:, 0], suffix_xy[:, 1], color="#ee6c4d", linewidth=2.0, label="suffix")
        ax.scatter(start_xy[0], start_xy[1], color="#1d3557", s=55, marker="o", label="start")
        ax.scatter(end_xy[0], end_xy[1], color="#6d597a", s=55, marker="s", label="end")
        ax.scatter(waypoint_xy[0], waypoint_xy[1], color="#d62828", s=140, marker="*", label="waypoint", zorder=5)

        ax.set_title(
            "waypoint=({:.3f}, {:.3f})  start=({:.3f}, {:.3f})  end=({:.3f}, {:.3f})".format(
                waypoint_xy[0],
                waypoint_xy[1],
                start_xy[0],
                start_xy[1],
                end_xy[0],
                end_xy[1],
            ),
            fontsize=9.5,
        )
        ax.set_aspect("equal")
        ax.set_xlim(-0.23, 0.23)
        ax.set_ylim(-0.23, 0.23)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    for ax in axes.flat[len(examples):]:
        ax.axis("off")

    fig.suptitle("Non-dissipative Reacher bidirectional trajectories around chosen waypoints", fontsize=14)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument("--num_examples", type=int, default=4)
    parser.add_argument("--prefix_steps", type=int, default=180)
    parser.add_argument("--suffix_steps", type=int, default=180)
    parser.add_argument("--waypoint_radius", type=float, default=0.18)
    parser.add_argument("--waypoint_qvel_scale", type=float, default=0.8)
    parser.add_argument("--torque_scale", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    fingertip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
    rng = np.random.default_rng(args.seed)

    examples = [
        build_bidirectional_example(
            model=model,
            rng=rng,
            prefix_steps=args.prefix_steps,
            suffix_steps=args.suffix_steps,
            waypoint_radius=args.waypoint_radius,
            waypoint_qvel_scale=args.waypoint_qvel_scale,
            torque_scale=args.torque_scale,
            fingertip_body_id=fingertip_body_id,
        )
        for _ in range(args.num_examples)
    ]

    save_path = Path(args.save_path)
    summary_path = Path(args.summary_path)
    plot_examples(examples, save_path)

    summary = {
        "xml_path": args.xml_path,
        "seed": args.seed,
        "num_examples": args.num_examples,
        "prefix_steps": args.prefix_steps,
        "suffix_steps": args.suffix_steps,
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
