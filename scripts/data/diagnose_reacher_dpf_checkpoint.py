#!/home/gsang/miniconda3/envs/perceiver/bin/python
"""Diagnose Reacher DPF prefix-completion failures on a saved checkpoint."""

from __future__ import annotations

import argparse
import json
import os
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
from src.models.utils import compare_generated_with_reconstructed, reconstruct_traj_with_momentum
from src.qpos_representation import decode_qpos_array, encode_qpos_array, infer_qpos_representation_from_xml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/dpf/"
        "trajectory_dpf_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone"
        "&DecoderAttentions_cond-concat_torque_in_state_layout-shifted_tau:epoch=359_val_loss:val_loss=0.0091.ckpt",
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
        default="/home/gsang/Projects/hnn_guided_dpf/plots/reacher_dpf_checkpoint_diagnostics",
    )
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated validation trajectory indices to diagnose.",
    )
    parser.add_argument("--prefix_len", type=int, default=500)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot_count", type=int, default=3)
    parser.add_argument("--boundary_window", type=int, default=120)
    return parser.parse_args()


def compute_initial_qvel(model: mujoco.MjModel, qpos: np.ndarray, mom: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    qpos_dim = int(qpos.shape[-1])
    qvel_dim = int(mom.shape[-1])
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:qpos_dim] = qpos
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    qvel = np.linalg.solve(M[:qvel_dim, :qvel_dim], mom).astype(np.float64, copy=False)
    return qvel


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def summarize_sincos_norms(encoded_qpos: np.ndarray, prefix_len: int) -> dict[str, float]:
    suffix = encoded_qpos[prefix_len:]
    q0_norm = np.sqrt(np.square(suffix[:, 0]) + np.square(suffix[:, 1]))
    q1_norm = np.sqrt(np.square(suffix[:, 2]) + np.square(suffix[:, 3]))
    q0_dev = np.abs(q0_norm - 1.0)
    q1_dev = np.abs(q1_norm - 1.0)
    return {
        "q0_norm_mean_abs_dev": float(q0_dev.mean()),
        "q0_norm_max_abs_dev": float(q0_dev.max()),
        "q1_norm_mean_abs_dev": float(q1_dev.mean()),
        "q1_norm_max_abs_dev": float(q1_dev.max()),
    }


def plot_boundary_window(
    *,
    gt_qpos_raw: np.ndarray,
    gen_qpos_raw: np.ndarray,
    recon_qpos_raw: np.ndarray,
    gt_mom: np.ndarray,
    gen_mom: np.ndarray,
    recon_mom: np.ndarray,
    gt_tau: np.ndarray,
    gen_tau: np.ndarray,
    prefix_len: int,
    window: int,
    save_path: str,
) -> None:
    start = max(0, prefix_len - window)
    end = min(len(gt_qpos_raw), prefix_len + window)
    t = np.arange(start, end)

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(f"Boundary window around prefix_len={prefix_len}", fontsize=14)

    panels = [
        ("qpos[0]", gt_qpos_raw[:, 0], gen_qpos_raw[:, 0], recon_qpos_raw[:, 0]),
        ("qpos[1]", gt_qpos_raw[:, 1], gen_qpos_raw[:, 1], recon_qpos_raw[:, 1]),
        ("mom[0]", gt_mom[:, 0], gen_mom[:, 0], recon_mom[:, 0]),
        ("mom[1]", gt_mom[:, 1], gen_mom[:, 1], recon_mom[:, 1]),
        ("torque[0]", gt_tau[:, 0], gen_tau[:, 0], None),
        ("torque[1]", gt_tau[:, 1], gen_tau[:, 1], None),
    ]

    for ax, (title, gt, gen, recon) in zip(axes.flat, panels):
        ax.plot(t, gt[start:end], label="Ground truth", color="black", linewidth=1.4)
        ax.plot(t, gen[start:end], label="Generated", color="royalblue", linewidth=1.1)
        if recon is not None:
            ax.plot(t, recon[start:end], label="Reconstructed", color="crimson", linewidth=1.0, alpha=0.9)
        ax.axvline(prefix_len - 1, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = [int(x) for x in args.sample_indices.split(",") if x.strip()]
    if not sample_indices:
        raise ValueError("sample_indices must contain at least one index")

    print(f"[Diag] Loading checkpoint from {args.checkpoint_path}")
    model = TrajectoryDPF.load_from_checkpoint(args.checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()

    with h5py.File(args.h5_path, "r") as h5_file:
        xml_content = h5_file.attrs["xml"]
        if isinstance(xml_content, bytes):
            xml_content = xml_content.decode()
        qpos_representation = infer_qpos_representation_from_xml(xml_content)
        data_dt = float(h5_file.attrs.get("data_dt", h5_file.attrs.get("dt", model.data_dt)))

        mj_model = mujoco.MjModel.from_xml_string(xml_content)

        sample_summaries = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = os.path.join(tmp_dir, "model.xml")
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(xml_content)

            for rank, sample_idx in enumerate(sample_indices):
                traj = h5_file[f"traj_{sample_idx}"]
                gt_qpos_raw = traj["seq_qpos"][:]
                gt_qpos = encode_qpos_array(gt_qpos_raw, qpos_representation).astype(np.float32)
                gt_mom = traj["seq_mom"][:].astype(np.float32)
                gt_tau = traj["seq_torque"][:].astype(np.float32)

                T = int(gt_qpos.shape[0])
                prefix_len = max(1, min(args.prefix_len, T - 1))

                observed_qpos = torch.from_numpy(gt_qpos).unsqueeze(0).to(device)
                observed_mom = torch.from_numpy(gt_mom).unsqueeze(0).to(device)
                observed_tau = torch.from_numpy(gt_tau).unsqueeze(0).to(device)

                with torch.no_grad():
                    generated_state, generated_tau = model.sample_trajectories(
                        num_samples=1,
                        trajectory_length=T,
                        num_diffusion_steps=args.num_diffusion_steps,
                        sample_mode="observed_prefix_completion",
                        prefix_len=prefix_len,
                        observed_qpos=observed_qpos,
                        observed_mom=observed_mom,
                        observed_torque=observed_tau,
                        use_ema=True,
                        sampler="ddim",
                    )

                generated_state_np = generated_state[0].detach().cpu().numpy()
                generated_tau_np = generated_tau[0].detach().cpu().numpy()

                gen_qpos = generated_state_np[:, : model.qpos_dim]
                gen_mom = generated_state_np[:, model.qpos_dim : model.qpos_dim + model.mom_dim]
                gen_qpos_raw = decode_qpos_array(gen_qpos, qpos_representation)

                # Full-trajectory physics consistency from t=0 using the generated torque.
                compare_metrics = compare_generated_with_reconstructed(
                    {
                        "seq_qpos": gen_qpos,
                        "seq_mom": gen_mom,
                        "seq_torque": generated_tau_np,
                    },
                    xml_path,
                    str(output_dir),
                    dt=float(model.dt),
                    data_dt=float(model.data_dt),
                    name=f"sample_{sample_idx}_generated_vs_reconstructed",
                    prefix_len=prefix_len,
                    qpos_representation=qpos_representation,
                    return_series=False,
                )

                # Boundary-local reconstruction from the last observed state using the generated suffix torque.
                boundary_state_idx = prefix_len - 1
                suffix_steps = T - boundary_state_idx
                boundary_qpos = gen_qpos_raw[boundary_state_idx]
                boundary_mom = gen_mom[boundary_state_idx]
                boundary_qvel = compute_initial_qvel(mj_model, boundary_qpos, boundary_mom)
                recon_suffix = reconstruct_traj_with_momentum(
                    mj_model,
                    suffix_steps,
                    float(model.dt),
                    boundary_qpos,
                    boundary_qvel,
                    generated_tau_np[boundary_state_idx:],
                    data_dt=data_dt,
                    trajectory_alignment="pre_step",
                )
                recon_suffix_qpos_raw = recon_suffix["seq_qpos"]
                recon_suffix_mom = recon_suffix["seq_mom"][:, : model.mom_dim]

                # Stitch reconstructed suffix onto the observed prefix for visualization.
                stitched_recon_qpos_raw = np.concatenate(
                    [gt_qpos_raw[:boundary_state_idx], recon_suffix_qpos_raw], axis=0
                )
                stitched_recon_mom = np.concatenate(
                    [gt_mom[:boundary_state_idx], recon_suffix_mom], axis=0
                )

                boundary_plot_path = output_dir / f"sample_{sample_idx}_boundary_window.jpg"
                if rank < args.plot_count:
                    plot_boundary_window(
                        gt_qpos_raw=gt_qpos_raw,
                        gen_qpos_raw=gen_qpos_raw,
                        recon_qpos_raw=stitched_recon_qpos_raw,
                        gt_mom=gt_mom,
                        gen_mom=gen_mom,
                        recon_mom=stitched_recon_mom,
                        gt_tau=gt_tau,
                        gen_tau=generated_tau_np,
                        prefix_len=prefix_len,
                        window=args.boundary_window,
                        save_path=str(boundary_plot_path),
                    )

                sample_summary = {
                    "sample_idx": sample_idx,
                    "prefix_len": prefix_len,
                    "trajectory_length": T,
                    "gt_suffix_mse_qpos_encoded": mse(gen_qpos[prefix_len:], gt_qpos[prefix_len:]),
                    "gt_suffix_mse_mom": mse(gen_mom[prefix_len:], gt_mom[prefix_len:]),
                    "gt_suffix_mse_torque": mse(generated_tau_np[prefix_len - 1 :], gt_tau[prefix_len - 1 :]),
                    "gt_boundary_mae_qpos_encoded": mae(gen_qpos[prefix_len], gt_qpos[prefix_len]),
                    "gt_boundary_mae_mom": mae(gen_mom[prefix_len], gt_mom[prefix_len]),
                    "gt_boundary_mae_torque": mae(generated_tau_np[prefix_len - 1], gt_tau[prefix_len - 1]),
                    "gt_boundary_step_mae_qpos_encoded": mae(
                        gen_qpos[prefix_len] - gen_qpos[prefix_len - 1],
                        gt_qpos[prefix_len] - gt_qpos[prefix_len - 1],
                    ),
                    "gt_boundary_step_mae_mom": mae(
                        gen_mom[prefix_len] - gen_mom[prefix_len - 1],
                        gt_mom[prefix_len] - gt_mom[prefix_len - 1],
                    ),
                    "physics_suffix_mse_qpos_raw_from_boundary": mse(
                        recon_suffix_qpos_raw[1:], gen_qpos_raw[boundary_state_idx + 1 :]
                    ),
                    "physics_suffix_mse_mom_from_boundary": mse(
                        recon_suffix_mom[1:], gen_mom[boundary_state_idx + 1 :]
                    ),
                    **summarize_sincos_norms(gen_qpos, prefix_len),
                    **compare_metrics,
                }
                sample_summaries.append(sample_summary)
                print(json.dumps(sample_summary, indent=2))

    aggregate = {}
    metric_keys = [k for k in sample_summaries[0].keys() if k not in {"sample_idx", "prefix_len", "trajectory_length"}]
    for key in metric_keys:
        values = np.array([row[key] for row in sample_summaries], dtype=np.float64)
        aggregate[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "max": float(values.max()),
            "min": float(values.min()),
        }

    report = {
        "checkpoint_path": args.checkpoint_path,
        "h5_path": args.h5_path,
        "device": str(device),
        "sample_indices": sample_indices,
        "sample_summaries": sample_summaries,
        "aggregate": aggregate,
    }
    report_path = output_dir / "diagnostic_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Diag] Wrote summary to {report_path}")


if __name__ == "__main__":
    main()
