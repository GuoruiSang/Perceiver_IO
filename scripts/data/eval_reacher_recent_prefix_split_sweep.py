#!/home/gsang/miniconda3/envs/perceiver/bin/python
from __future__ import annotations

import argparse
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

from src.models.trajectory_dpf_model import TrajectoryDPF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep multiple replay split points and multiple recent clean prefix lengths, "
            "then report direct generated-vs-replay suffix MSE."
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
        default="/home/gsang/Projects/hnn_guided_dpf/plots/reacher_recent_prefix_split_sweep",
    )
    parser.add_argument("--num_trajectories", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split_prefix_lens",
        type=str,
        default="99,199,299,399,499,599,699,799,899,999",
    )
    parser.add_argument(
        "--recent_window_lens",
        type=str,
        default="",
        help=(
            "Optional comma-separated global recent clean prefix lengths. "
            "If empty, use a default log-spaced grid per split: 1,2,4,8,16,32,64,128,256,512,split."
        ),
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default="recent_prefix_split_sweep_report.json",
    )
    return parser.parse_args()


def parse_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value > 0:
            values.append(value)
    if not values:
        raise ValueError("Expected at least one positive integer")
    return values


def build_prefix_grid(split_prefix_len: int, requested: list[int] | None) -> list[int]:
    if requested:
        return sorted(set(min(int(split_prefix_len), int(v)) for v in requested if int(v) > 0))
    base = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, int(split_prefix_len)]
    return sorted(set(v for v in base if v <= int(split_prefix_len)))


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def plot_split_boxplots(split_row: dict, save_path: Path) -> None:
    prefix_lens = [row["recent_window_len"] for row in split_row["rows"]]
    qpos_values = [row["suffix_mse_qpos_values"] for row in split_row["rows"]]
    mom_values = [row["suffix_mse_mom_values"] for row in split_row["rows"]]
    total_values = [row["suffix_mse_total_values"] for row in split_row["rows"]]
    labels = [str(v) for v in prefix_lens]

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.8), dpi=180, constrained_layout=True)
    panels = [
        (axes[0], qpos_values, "Suffix qpos MSE", "suffix qpos MSE"),
        (axes[1], mom_values, "Suffix mom MSE", "suffix mom MSE"),
        (axes[2], total_values, "Suffix total MSE", "suffix total MSE"),
    ]
    for ax, values, title, ylabel in panels:
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.set_title(title)
        ax.set_xlabel("recent clean prefix length")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"Direct generated-vs-replay suffix MSE at split={int(split_row['split_prefix_len'])}",
        fontsize=14,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / args.report_name

    requested_splits = parse_int_csv(args.split_prefix_lens)
    requested_windows = parse_int_csv(args.recent_window_lens) if args.recent_window_lens.strip() else None

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[SplitSweep] device={device}")
    model = TrajectoryDPF.load_from_checkpoint(args.checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()

    split_rows: list[dict] = []

    with h5py.File(args.h5_path, "r") as h5_file:
        total_available = int(h5_file.attrs["num_trajectories"])
        trajectory_length = int(h5_file.attrs["num_steps"])
        split_prefix_lens = sorted(
            set(max(1, min(int(v), trajectory_length - 1)) for v in requested_splits)
        )

        start_index = max(0, int(args.start_index))
        if start_index >= total_available:
            raise ValueError(f"start_index={start_index} is out of range for dataset size {total_available}")
        num_trajectories = min(int(args.num_trajectories), total_available - start_index)

        print(
            f"[SplitSweep] split_prefix_lens={split_prefix_lens}, "
            f"num_trajectories={num_trajectories}, diffusion_steps={args.num_diffusion_steps}"
        )

        for split_prefix_len in split_prefix_lens:
            prefix_grid = build_prefix_grid(split_prefix_len=split_prefix_len, requested=requested_windows)
            print(f"[SplitSweep] split={split_prefix_len} prefix_grid={prefix_grid}")

            split_row = {
                "split_prefix_len": int(split_prefix_len),
                "rows": [],
            }

            for recent_window_len in prefix_grid:
                crop_start = int(split_prefix_len) - int(recent_window_len)
                crop_length = trajectory_length - crop_start
                absolute_time_indices = torch.arange(
                    crop_start,
                    crop_start + crop_length,
                    dtype=torch.long,
                    device=device,
                )

                suffix_mse_qpos_values: list[float] = []
                suffix_mse_mom_values: list[float] = []
                suffix_mse_total_values: list[float] = []

                print(
                    f"[SplitSweep] split={split_prefix_len} window={recent_window_len} "
                    f"crop_start={crop_start} crop_length={crop_length}"
                )

                for local_start in range(0, num_trajectories, int(args.batch_size)):
                    local_end = min(local_start + int(args.batch_size), num_trajectories)
                    batch_count = local_end - local_start
                    global_start = start_index + local_start
                    global_end = start_index + local_end
                    print(
                        f"[SplitSweep] split={split_prefix_len} window={recent_window_len} "
                        f"batch {global_start}:{global_end}"
                    )

                    observed_qpos = np.empty((batch_count, crop_length, model.qpos_dim), dtype=np.float32)
                    observed_mom = np.empty((batch_count, crop_length, model.mom_dim), dtype=np.float32)
                    observed_torque = np.empty((batch_count, crop_length, model.torque_dim), dtype=np.float32)

                    for batch_offset, traj_index in enumerate(range(global_start, global_end)):
                        traj = h5_file[f"traj_{traj_index}"]
                        observed_qpos[batch_offset] = traj["seq_qpos"][crop_start:].astype(np.float32)
                        observed_mom[batch_offset] = traj["seq_mom"][crop_start:].astype(np.float32)
                        observed_torque[batch_offset] = traj["seq_torque"][crop_start:].astype(np.float32)

                    observed_qpos_t = torch.from_numpy(observed_qpos).to(device=device, dtype=torch.float32)
                    observed_mom_t = torch.from_numpy(observed_mom).to(device=device, dtype=torch.float32)
                    observed_torque_t = torch.from_numpy(observed_torque).to(device=device, dtype=torch.float32)

                    with torch.no_grad():
                        generated_state, _generated_tau = model.sample_trajectories(
                            num_samples=batch_count,
                            trajectory_length=crop_length,
                            num_diffusion_steps=int(args.num_diffusion_steps),
                            sample_mode="observed_prefix_completion",
                            prefix_len=int(recent_window_len),
                            observed_qpos=observed_qpos_t,
                            observed_mom=observed_mom_t,
                            observed_torque=observed_torque_t,
                            time_indices=absolute_time_indices,
                            use_ema=True,
                            sampler="ddim",
                        )

                    generated_state_np = generated_state.detach().cpu().numpy()
                    generated_qpos = generated_state_np[:, :, : model.qpos_dim]
                    generated_mom = generated_state_np[:, :, model.qpos_dim : model.qpos_dim + model.mom_dim]

                    suffix_slice = slice(int(recent_window_len), crop_length)
                    batch_qpos_mse = np.mean(
                        (generated_qpos[:, suffix_slice, :] - observed_qpos[:, suffix_slice, :]) ** 2,
                        axis=(1, 2),
                    )
                    batch_mom_mse = np.mean(
                        (generated_mom[:, suffix_slice, :] - observed_mom[:, suffix_slice, :]) ** 2,
                        axis=(1, 2),
                    )

                    suffix_mse_qpos_values.extend(batch_qpos_mse.astype(np.float64).tolist())
                    suffix_mse_mom_values.extend(batch_mom_mse.astype(np.float64).tolist())
                    suffix_mse_total_values.extend((batch_qpos_mse + batch_mom_mse).astype(np.float64).tolist())

                split_row["rows"].append(
                    {
                        "recent_window_len": int(recent_window_len),
                        "crop_start": int(crop_start),
                        "crop_length": int(crop_length),
                        "num_trajectories": int(num_trajectories),
                        "suffix_mse_qpos": summarize(suffix_mse_qpos_values),
                        "suffix_mse_mom": summarize(suffix_mse_mom_values),
                        "suffix_mse_total": summarize(suffix_mse_total_values),
                        "suffix_mse_qpos_values": [float(v) for v in suffix_mse_qpos_values],
                        "suffix_mse_mom_values": [float(v) for v in suffix_mse_mom_values],
                        "suffix_mse_total_values": [float(v) for v in suffix_mse_total_values],
                    }
                )

            figure_path = output_dir / f"split_{int(split_prefix_len):04d}_suffix_mse_boxplots.png"
            plot_split_boxplots(split_row, figure_path)
            split_row["figure_path"] = str(figure_path)
            split_rows.append(split_row)

            print(f"[SplitSweep] figure: {figure_path}")
            for row in split_row["rows"]:
                print(
                    "[SplitSweep] split={} window={} total_mean={:.6f} total_median={:.6f}".format(
                        int(split_prefix_len),
                        int(row["recent_window_len"]),
                        float(row["suffix_mse_total"]["mean"]),
                        float(row["suffix_mse_total"]["median"]),
                    )
                )

    report = {
        "checkpoint_path": args.checkpoint_path,
        "h5_path": args.h5_path,
        "device": str(device),
        "seed": int(args.seed),
        "start_index": int(args.start_index),
        "num_trajectories": int(num_trajectories),
        "batch_size": int(args.batch_size),
        "num_diffusion_steps": int(args.num_diffusion_steps),
        "split_prefix_lens": [int(v) for v in split_prefix_lens],
        "requested_recent_window_lens": None if requested_windows is None else [int(v) for v in requested_windows],
        "split_rows": split_rows,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[SplitSweep] report: {report_path}")


if __name__ == "__main__":
    main()
