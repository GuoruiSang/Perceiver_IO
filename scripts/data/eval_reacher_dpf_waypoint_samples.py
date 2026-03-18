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
from src.models.utils import compare_generated_with_reconstructed


L1 = 0.10
L2 = 0.11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Reacher waypoint-conditioned trajectories from a DPF checkpoint and compare them to MuJoCo replay."
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
        default="/home/gsang/Projects/hnn_guided_dpf/plots/reacher_dpf_waypoint_samples",
    )
    parser.add_argument("--num_trajectories", type=int, default=1000)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_plot_examples", type=int, default=4)
    parser.add_argument(
        "--figure_name",
        type=str,
        default="reacher_dpf_waypoint_examples.png",
    )
    parser.add_argument(
        "--plot_mode",
        type=str,
        choices=("grid_examples", "overlay_with_torque"),
        default="grid_examples",
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default="reacher_dpf_waypoint_report.json",
    )
    return parser.parse_args()


def fingertip_xy_from_qpos(qpos: np.ndarray) -> np.ndarray:
    q0 = qpos[:, 0]
    q1 = qpos[:, 1]
    xy = np.empty((qpos.shape[0], 2), dtype=np.float64)
    xy[:, 0] = L1 * np.cos(q0) + L2 * np.cos(q0 + q1)
    xy[:, 1] = L1 * np.sin(q0) + L2 * np.sin(q0 + q1)
    return xy


def _draw_workspace_guides(ax: plt.Axes) -> None:
    ax.add_patch(plt.Circle((0.0, 0.0), L1 + L2, color="#bbbbbb", fill=False, linestyle="--", linewidth=1.0))
    ax.add_patch(plt.Circle((0.0, 0.0), abs(L1 - L2), color="#dddddd", fill=False, linestyle=":", linewidth=1.0))
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def build_selected_examples(rows: list[dict], num_examples: int) -> list[dict]:
    if num_examples <= 0:
        return []
    if not rows:
        return []
    num_examples = max(1, min(int(num_examples), len(rows)))
    sorted_rows = sorted(rows, key=lambda row: row["suffix_mse_total"])
    if num_examples == 1:
        chosen = [sorted_rows[len(sorted_rows) // 2]]
    else:
        chosen = []
        for i in range(num_examples):
            idx = round(i * (len(sorted_rows) - 1) / (num_examples - 1))
            chosen.append(sorted_rows[idx])
    return chosen


def plot_waypoint_examples(examples: list[dict], save_path: Path) -> None:
    if not examples:
        return
    n = len(examples)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 6.5 * nrows), dpi=180)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, example in zip(axes.flat, examples):
        _draw_workspace_guides(ax)
        prefix_xy = np.asarray(example["prefix_xy"], dtype=np.float64)
        suffix_xy = np.asarray(example["suffix_xy"], dtype=np.float64)
        waypoint_xy = np.asarray(example["waypoint_xy"], dtype=np.float64)
        traj_index = int(example["traj_index"])

        ax.plot(prefix_xy[:, 0], prefix_xy[:, 1], color="#2a6f97", linewidth=2.0, label="prefix")
        ax.plot(suffix_xy[:, 0], suffix_xy[:, 1], color="#ee6c4d", linewidth=2.0, label="suffix")
        ax.scatter(waypoint_xy[0], waypoint_xy[1], color="#d62828", s=140, marker="*", label="waypoint", zorder=5)
        ax.set_title(
            "traj={} waypoint=({:.3f}, {:.3f}) suffix_mse={:.4e}".format(
                traj_index,
                waypoint_xy[0],
                waypoint_xy[1],
                float(example["suffix_mse_total"]),
            ),
            fontsize=9.5,
        )
        ax.legend(loc="upper right", fontsize=8)

    for ax in axes.flat[len(examples):]:
        ax.axis("off")

    fig.suptitle("Reacher DPF waypoint-conditioned prefix/suffix samples", fontsize=14)
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
        prefix_xy = np.asarray(example["prefix_xy"], dtype=np.float64)
        suffix_xy = np.asarray(example["suffix_xy"], dtype=np.float64)
        prefix_tau = np.asarray(example["prefix_tau"], dtype=np.float64)
        suffix_tau = np.asarray(example["suffix_tau"], dtype=np.float64)
        prefix_steps = int(example["prefix_steps"])
        suffix_steps = int(example["suffix_steps"])

        ax_xy.plot(prefix_xy[:, 0], prefix_xy[:, 1], color=prefix_color, alpha=0.05, linewidth=0.7)
        ax_xy.plot(suffix_xy[:, 0], suffix_xy[:, 1], color=suffix_color, alpha=0.05, linewidth=0.7)

        prefix_t = np.arange(prefix_steps, dtype=np.int32)
        suffix_t = np.arange(prefix_steps, prefix_steps + suffix_steps, dtype=np.int32)
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

    fig.suptitle("Reacher DPF waypoint-conditioned trajectories: positions and torques", fontsize=15)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def aggregate_metrics(rows: list[dict], keys: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Eval] device={device}")
    print(f"[Eval] loading checkpoint: {args.checkpoint_path}")
    model = TrajectoryDPF.load_from_checkpoint(args.checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()

    figure_path = output_dir / args.figure_name
    report_path = output_dir / args.report_name

    rows: list[dict] = []
    print(f"[Eval] reading dataset: {args.h5_path}")
    with h5py.File(args.h5_path, "r") as h5_file:
        xml_content = h5_file.attrs["xml"]
        if isinstance(xml_content, bytes):
            xml_content = xml_content.decode()

        total_available = int(h5_file.attrs["num_trajectories"])
        start_index = max(0, int(args.start_index))
        if start_index >= total_available:
            raise ValueError(f"start_index={start_index} is out of range for dataset size {total_available}")
        num_trajectories = min(int(args.num_trajectories), total_available - start_index)
        trajectory_length = int(h5_file.attrs["num_steps"])

        print(
            f"[Eval] generating {num_trajectories} trajectories with batch_size={args.batch_size}, "
            f"trajectory_length={trajectory_length}, diffusion_steps={args.num_diffusion_steps}"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / "model.xml"
            xml_path.write_text(xml_content, encoding="utf-8")

            for local_start in range(0, num_trajectories, args.batch_size):
                local_end = min(local_start + args.batch_size, num_trajectories)
                batch_count = local_end - local_start
                global_start = start_index + local_start
                global_end = start_index + local_end
                print(f"[Eval] batch {global_start}:{global_end}")

                observed_qpos = np.empty((batch_count, trajectory_length, model.qpos_dim), dtype=np.float32)
                observed_mom = np.empty((batch_count, trajectory_length, model.mom_dim), dtype=np.float32)
                observed_torque = np.empty((batch_count, trajectory_length, model.torque_dim), dtype=np.float32)
                prefix_lens: list[int] = []
                waypoint_xy_list: list[np.ndarray] = []
                gt_prefix_xy: list[np.ndarray] = []

                for batch_offset, traj_index in enumerate(range(global_start, global_end)):
                    traj = h5_file[f"traj_{traj_index}"]
                    observed_qpos[batch_offset] = traj["seq_qpos"][:].astype(np.float32)
                    observed_mom[batch_offset] = traj["seq_mom"][:].astype(np.float32)
                    observed_torque[batch_offset] = traj["seq_torque"][:].astype(np.float32)
                    prefix_len = int(traj.attrs["waypoint_index"])
                    prefix_len = max(1, min(prefix_len, trajectory_length - 1))
                    prefix_lens.append(prefix_len)
                    waypoint_xy = traj["waypoint_xy"][:].astype(np.float64)
                    waypoint_xy_list.append(waypoint_xy)
                    gt_prefix_xy.append(traj["seq_fingertip_xy"][:prefix_len + 1].astype(np.float64))

                observed_qpos_t = torch.from_numpy(observed_qpos).to(device=device, dtype=torch.float32)
                observed_mom_t = torch.from_numpy(observed_mom).to(device=device, dtype=torch.float32)
                observed_torque_t = torch.from_numpy(observed_torque).to(device=device, dtype=torch.float32)

                for batch_offset, traj_index in enumerate(range(global_start, global_end)):
                    prefix_len = prefix_lens[batch_offset]
                    with torch.no_grad():
                        generated_state, generated_tau = model.sample_trajectories(
                            num_samples=1,
                            trajectory_length=trajectory_length,
                            num_diffusion_steps=args.num_diffusion_steps,
                            sample_mode="observed_prefix_completion",
                            prefix_len=prefix_len,
                            observed_qpos=observed_qpos_t[batch_offset : batch_offset + 1],
                            observed_mom=observed_mom_t[batch_offset : batch_offset + 1],
                            observed_torque=observed_torque_t[batch_offset : batch_offset + 1],
                            use_ema=True,
                            sampler="ddim",
                        )

                    generated_state_np = generated_state[0].detach().cpu().numpy()
                    generated_tau_np = generated_tau[0].detach().cpu().numpy()
                    generated_qpos = generated_state_np[:, : model.qpos_dim]
                    generated_mom = generated_state_np[:, model.qpos_dim : model.qpos_dim + model.mom_dim]

                    compare = compare_generated_with_reconstructed(
                        generated={
                            "seq_qpos": generated_qpos,
                            "seq_mom": generated_mom,
                            "seq_torque": generated_tau_np,
                        },
                        mujoco_model_path=str(xml_path),
                        save_path=str(output_dir),
                        dt=float(model.dt),
                        data_dt=float(model.data_dt),
                        name=None,
                        prefix_len=prefix_len,
                        qpos_representation=model.qpos_representation,
                        return_series=True,
                    )

                    gen_qpos = compare["generated_qpos"]
                    gen_mom = compare["generated_mom"]
                    gen_tau = compare["generated_torque"]
                    replay_qpos = compare["reconstructed_qpos"]
                    replay_mom = compare["reconstructed_mom"]
                    replay_tau = compare["reconstructed_torque"]

                    generated_xy = fingertip_xy_from_qpos(gen_qpos)
                    replay_xy = fingertip_xy_from_qpos(replay_qpos)

                    prefix_slice = slice(0, prefix_len + 1)
                    suffix_slice = slice(prefix_len, trajectory_length)

                    suffix_mse_qpos = float(np.mean((gen_qpos[suffix_slice] - replay_qpos[suffix_slice]) ** 2))
                    suffix_mse_mom = float(np.mean((gen_mom[suffix_slice] - replay_mom[suffix_slice]) ** 2))
                    suffix_mse_tau = float(np.mean((gen_tau[prefix_len:] - replay_tau[prefix_len:]) ** 2))
                    suffix_mse_xy = float(np.mean((generated_xy[suffix_slice] - replay_xy[suffix_slice]) ** 2))

                    row = {
                        "traj_index": int(traj_index),
                        "prefix_len": int(prefix_len),
                        "trajectory_length": int(trajectory_length),
                        "waypoint_xy": waypoint_xy_list[batch_offset].tolist(),
                        "full_mse_qpos": float(compare["mse_qpos"]),
                        "full_mse_mom": float(compare["mse_mom"]),
                        "full_mse_total": float(compare["mse_total"]),
                        "full_mse_xy": float(np.mean((generated_xy - replay_xy) ** 2)),
                        "suffix_mse_qpos": suffix_mse_qpos,
                        "suffix_mse_mom": suffix_mse_mom,
                        "suffix_mse_tau": suffix_mse_tau,
                        "suffix_mse_total": suffix_mse_qpos + suffix_mse_mom,
                        "suffix_mse_xy": suffix_mse_xy,
                        "prefix_xy": gt_prefix_xy[batch_offset][prefix_slice].tolist(),
                        "suffix_xy": generated_xy[suffix_slice].tolist(),
                        "prefix_tau": gen_tau[:prefix_len].tolist(),
                        "suffix_tau": gen_tau[prefix_len:].tolist(),
                        "prefix_steps": int(prefix_len),
                        "suffix_steps": int(trajectory_length - prefix_len),
                    }
                    rows.append(row)

    metric_keys = [
        "full_mse_qpos",
        "full_mse_mom",
        "full_mse_total",
        "full_mse_xy",
        "suffix_mse_qpos",
        "suffix_mse_mom",
        "suffix_mse_tau",
        "suffix_mse_total",
        "suffix_mse_xy",
    ]
    aggregate = aggregate_metrics(rows, metric_keys)
    selected_examples = build_selected_examples(rows, num_examples=args.num_plot_examples)
    if args.plot_mode == "overlay_with_torque":
        plot_overlay_with_torque(rows, figure_path)
    else:
        plot_waypoint_examples(selected_examples, figure_path)

    report = {
        "checkpoint_path": args.checkpoint_path,
        "h5_path": args.h5_path,
        "device": str(device),
        "seed": int(args.seed),
        "start_index": int(args.start_index),
        "num_trajectories": int(len(rows)),
        "num_diffusion_steps": int(args.num_diffusion_steps),
        "batch_size": int(args.batch_size),
        "aggregate": aggregate,
        "selected_examples": selected_examples,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[Eval] figure: {figure_path}")
    print(f"[Eval] report: {report_path}")
    print("[Eval] aggregate metrics:")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
