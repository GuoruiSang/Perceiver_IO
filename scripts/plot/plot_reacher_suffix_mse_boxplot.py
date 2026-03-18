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
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.trajectory_dpf_model import TrajectoryDPF
from src.models.utils import compare_generated_with_reconstructed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute suffix qpos/mom MSE distributions from a Reacher DPF eval report and save a box plot."
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/plots/reacher_dpf_waypoint_samples/"
        "reacher_dpf_overlay_with_torque_1000traj_1000steps_dt_0p001.report.json",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fixed_prefix_len", type=int, default=None)
    return parser.parse_args()


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_path)
    report = json.loads(report_path.read_text())

    checkpoint_path = str(report["checkpoint_path"])
    h5_path = str(report["h5_path"])
    seed = int(report["seed"])
    start_index = int(report.get("start_index", 0))
    num_trajectories = int(report["num_trajectories"])
    num_diffusion_steps = int(report["num_diffusion_steps"])
    fixed_prefix_len = (
        int(args.fixed_prefix_len)
        if args.fixed_prefix_len is not None
        else (
            None if report.get("fixed_prefix_len") is None else int(report["fixed_prefix_len"])
        )
    )

    if "suffix_mse_qpos_values" in report and "suffix_mse_mom_values" in report:
        print(f"[Boxplot] report={report_path}")
        print("[Boxplot] using suffix MSE values embedded in report")
        suffix_qpos_arr = np.asarray(report["suffix_mse_qpos_values"], dtype=np.float64)
        suffix_mom_arr = np.asarray(report["suffix_mse_mom_values"], dtype=np.float64)
        device = "report_only"
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        print(f"[Boxplot] report={report_path}")
        print(f"[Boxplot] device={device}")
        print(f"[Boxplot] loading checkpoint: {checkpoint_path}")
        model = TrajectoryDPF.load_from_checkpoint(checkpoint_path, map_location=device)
        model = model.to(device)
        model.eval()

        suffix_qpos_mse: list[float] = []
        suffix_mom_mse: list[float] = []

        with h5py.File(h5_path, "r") as h5_file:
            trajectory_length = int(h5_file.attrs["num_steps"])
            xml_content = h5_file.attrs["xml"]
            if isinstance(xml_content, bytes):
                xml_content = xml_content.decode()

            with tempfile.TemporaryDirectory() as tmp_dir:
                xml_path = Path(tmp_dir) / "model.xml"
                xml_path.write_text(xml_content, encoding="utf-8")

                for traj_index in range(start_index, start_index + num_trajectories):
                    if (traj_index - start_index) % 100 == 0:
                        print(f"[Boxplot] trajectory {traj_index - start_index + 1}/{num_trajectories}")

                    traj = h5_file[f"traj_{traj_index}"]
                    observed_qpos = torch.from_numpy(traj["seq_qpos"][:].astype(np.float32)).unsqueeze(0).to(device=device)
                    observed_mom = torch.from_numpy(traj["seq_mom"][:].astype(np.float32)).unsqueeze(0).to(device=device)
                    observed_torque = torch.from_numpy(traj["seq_torque"][:].astype(np.float32)).unsqueeze(0).to(device=device)
                    prefix_len = fixed_prefix_len if fixed_prefix_len is not None else int(traj.attrs["waypoint_index"])
                    prefix_len = max(1, min(prefix_len, trajectory_length - 1))

                    with torch.no_grad():
                        generated_state, generated_tau = model.sample_trajectories(
                            num_samples=1,
                            trajectory_length=trajectory_length,
                            num_diffusion_steps=num_diffusion_steps,
                            sample_mode="observed_prefix_completion",
                            prefix_len=prefix_len,
                            observed_qpos=observed_qpos,
                            observed_mom=observed_mom,
                            observed_torque=observed_torque,
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
                        save_path=str(report_path.parent),
                        dt=float(model.dt),
                        data_dt=float(model.data_dt),
                        name=None,
                        prefix_len=prefix_len,
                        qpos_representation=model.qpos_representation,
                        return_series=True,
                    )

                    gen_qpos = compare["generated_qpos"]
                    gen_mom = compare["generated_mom"]
                    replay_qpos = compare["reconstructed_qpos"]
                    replay_mom = compare["reconstructed_mom"]
                    suffix_slice = slice(prefix_len, trajectory_length)

                    suffix_qpos_mse.append(float(np.mean((gen_qpos[suffix_slice] - replay_qpos[suffix_slice]) ** 2)))
                    suffix_mom_mse.append(float(np.mean((gen_mom[suffix_slice] - replay_mom[suffix_slice]) ** 2)))

        suffix_qpos_arr = np.asarray(suffix_qpos_mse, dtype=np.float64)
        suffix_mom_arr = np.asarray(suffix_mom_mse, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=220)
    bp = ax.boxplot(
        [suffix_qpos_arr, suffix_mom_arr],
        tick_labels=["Suffix qpos MSE", "Suffix mom MSE"],
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )
    for patch, color in zip(bp["boxes"], ["#4c78a8", "#f58518"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.6)

    ax.set_yscale("log")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("Reacher DPF suffix MSE distribution across 1000 trajectories")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()

    output_png = report_path.with_name(report_path.stem.replace(".report", "") + "_suffix_mse_boxplot.png")
    output_json = report_path.with_name(report_path.stem.replace(".report", "") + "_suffix_mse_boxplot_summary.json")
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "report_path": str(report_path),
        "checkpoint_path": checkpoint_path,
        "h5_path": h5_path,
        "device": str(device),
        "num_trajectories": int(num_trajectories),
        "num_diffusion_steps": int(num_diffusion_steps),
        "fixed_prefix_len": fixed_prefix_len,
        "suffix_mse_qpos": summarize(suffix_qpos_arr),
        "suffix_mse_mom": summarize(suffix_mom_arr),
        "output_png": str(output_png),
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[Boxplot] figure: {output_png}")
    print(f"[Boxplot] summary: {output_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
