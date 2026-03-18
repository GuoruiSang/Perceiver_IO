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
import numpy as np
import torch

from src.models.trajectory_dpf_model import TrajectoryDPF
from src.models.utils import compare_generated_with_reconstructed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render W&B-style comparison plots for selected Reacher DPF samples."
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/plots/reacher_dpf_prefix1_initialstate/"
        "reacher_dpf_prefix1_overlay_with_torque_1000traj_1000steps_dt_0p001.report.json",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_diffusion_steps", type=int, default=None)
    parser.add_argument(
        "--traj_indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit trajectory indices. Defaults to selected_examples from the report.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional output directory. Defaults to <report_dir>/wandb_style_samples.",
    )
    return parser.parse_args()


def choose_examples(report: dict, explicit_indices: list[int] | None) -> list[dict]:
    if explicit_indices:
        return [
            {
                "traj_index": int(traj_index),
                "label": f"traj{int(traj_index):04d}",
            }
            for traj_index in explicit_indices
        ]

    selected = list(report.get("selected_examples") or [])
    if not selected:
        raise ValueError("Report has no selected_examples and no explicit traj_indices were provided.")

    labels = ["best", "low", "high", "worst"]
    examples: list[dict] = []
    for i, row in enumerate(selected):
        label = labels[i] if i < len(labels) else f"rank{i:02d}"
        examples.append(
            {
                "traj_index": int(row["traj_index"]),
                "label": label,
                "report_metrics": {
                    "suffix_mse_qpos": float(row["suffix_mse_qpos"]),
                    "suffix_mse_mom": float(row["suffix_mse_mom"]),
                    "suffix_mse_total": float(row["suffix_mse_total"]),
                },
            }
        )
    return examples


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_path)
    report = json.loads(report_path.read_text())

    checkpoint_path = str(report["checkpoint_path"])
    h5_path = str(report["h5_path"])
    seed = int(report.get("seed", 0))
    fixed_prefix_len = report.get("fixed_prefix_len")
    num_diffusion_steps = (
        int(args.num_diffusion_steps)
        if args.num_diffusion_steps is not None
        else int(report["num_diffusion_steps"])
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else report_path.parent / "wandb_style_samples"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = choose_examples(report, args.traj_indices)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[Render] report={report_path}")
    print(f"[Render] device={device}")
    print(f"[Render] output_dir={output_dir}")
    print(f"[Render] loading checkpoint: {checkpoint_path}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TrajectoryDPF.load_from_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()

    summary_rows: list[dict] = []

    with h5py.File(h5_path, "r") as h5_file:
        trajectory_length = int(h5_file.attrs["num_steps"])
        xml_content = h5_file.attrs["xml"]
        if isinstance(xml_content, bytes):
            xml_content = xml_content.decode()

        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / "model.xml"
            xml_path.write_text(xml_content, encoding="utf-8")

            for example_idx, example in enumerate(examples):
                traj_index = int(example["traj_index"])
                traj = h5_file[f"traj_{traj_index}"]
                prefix_len = (
                    int(fixed_prefix_len)
                    if fixed_prefix_len is not None
                    else int(traj.attrs["waypoint_index"])
                )
                prefix_len = max(1, min(prefix_len, trajectory_length - 1))

                observed_qpos = torch.from_numpy(traj["seq_qpos"][:].astype(np.float32)).unsqueeze(0).to(device=device)
                observed_mom = torch.from_numpy(traj["seq_mom"][:].astype(np.float32)).unsqueeze(0).to(device=device)
                observed_torque = torch.from_numpy(traj["seq_torque"][:].astype(np.float32)).unsqueeze(0).to(device=device)

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

                name = f"comparison_{example_idx:02d}_{example['label']}_traj{traj_index:04d}"
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
                    name=name,
                    prefix_len=prefix_len,
                    qpos_representation=model.qpos_representation,
                    return_series=True,
                )

                gen_qpos = compare["generated_qpos"]
                gen_mom = compare["generated_mom"]
                replay_qpos = compare["reconstructed_qpos"]
                replay_mom = compare["reconstructed_mom"]
                suffix_slice = slice(prefix_len, trajectory_length)

                suffix_mse_qpos = float(np.mean((gen_qpos[suffix_slice] - replay_qpos[suffix_slice]) ** 2))
                suffix_mse_mom = float(np.mean((gen_mom[suffix_slice] - replay_mom[suffix_slice]) ** 2))
                suffix_mse_total = suffix_mse_qpos + suffix_mse_mom

                row = {
                    "label": str(example["label"]),
                    "traj_index": traj_index,
                    "prefix_len": prefix_len,
                    "num_diffusion_steps": num_diffusion_steps,
                    "output_path": str(output_dir / f"{name}.jpg"),
                    "full_mse_qpos": float(compare["mse_qpos"]),
                    "full_mse_mom": float(compare["mse_mom"]),
                    "full_mse_total": float(compare["mse_total"]),
                    "suffix_mse_qpos": suffix_mse_qpos,
                    "suffix_mse_mom": suffix_mse_mom,
                    "suffix_mse_total": suffix_mse_total,
                }
                if "report_metrics" in example:
                    row["report_metrics"] = example["report_metrics"]
                summary_rows.append(row)
                print(json.dumps(row, indent=2))

    summary = {
        "report_path": str(report_path),
        "checkpoint_path": checkpoint_path,
        "h5_path": h5_path,
        "device": str(device),
        "fixed_prefix_len": fixed_prefix_len,
        "num_diffusion_steps": num_diffusion_steps,
        "examples": summary_rows,
    }
    summary_path = output_dir / "render_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Render] summary={summary_path}")


if __name__ == "__main__":
    main()
