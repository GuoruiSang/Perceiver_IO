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
import numpy as np
import torch

from src.models.HNN import HNNWrapper
from src.models.trajectory_dpf_model import TrajectoryDPF
from src.qpos_representation import decode_qpos_array, encode_qpos_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare unguided vs HNN-guided Reacher DPF prefix completions on replay MSE. "
            "For each validation trajectory, sample one random prefix, generate an unguided and guided "
            "completion with the same initial diffusion noise, and compare the full generated trajectory "
            "against the replay trajectory."
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
        "--hnn_checkpoint_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher/hnn_exploration_iid_uniform_len1000_v1/"
        "StructuredHNN-ReacherExploration-IID-epoch-epoch=999.ckpt",
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
        default="/home/gsang/Projects/hnn_guided_dpf/plots/reacher_sampling_hnn_guidance_compare",
    )
    parser.add_argument("--num_trajectories", type=int, default=1000)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_diffusion_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--prefix_min", type=int, default=1)
    parser.add_argument("--prefix_max", type=int, default=0, help="0 means T-1.")
    parser.add_argument("--guidance_method", type=str, default="strategy2", choices=["strategy1", "strategy2"])
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
    parser.add_argument("--report_name", type=str, default="reacher_sampling_hnn_guidance_compare.json")
    return parser.parse_args()


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def aggregate_metric(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def run_completion(
    *,
    model: TrajectoryDPF,
    hnn_model: HNNWrapper | None,
    observed_qpos: torch.Tensor,
    observed_mom: torch.Tensor,
    observed_torque: torch.Tensor,
    prefix_len: int,
    trajectory_length: int,
    num_diffusion_steps: int,
    initial_noise: torch.Tensor,
    guidance_method: str,
    alpha_q: float,
    alpha_p: float,
    guidance_trust_lambda: float,
    guidance_normalize_grad: bool,
    guidance_joint_update: bool,
) -> tuple[np.ndarray, np.ndarray]:
    sample_context = contextlib.nullcontext() if hnn_model is not None else torch.no_grad()
    with sample_context:
        generated_state, generated_tau = model.sample_trajectories(
            num_samples=1,
            trajectory_length=trajectory_length,
            num_diffusion_steps=num_diffusion_steps,
            sample_mode="observed_prefix_completion",
            prefix_len=prefix_len,
            observed_qpos=observed_qpos,
            observed_mom=observed_mom,
            observed_torque=observed_torque,
            initial_noise=initial_noise,
            use_ema=True,
            sampler="ddim",
            hnn=hnn_model,
            guidance_method=guidance_method,
            alpha_q=alpha_q,
            alpha_p=alpha_p,
            guidance_trust_lambda=guidance_trust_lambda,
            guidance_normalize_grad=guidance_normalize_grad,
            guidance_joint_update=guidance_joint_update,
        )

    return (
        generated_state[0].detach().cpu().numpy(),
        generated_tau[0].detach().cpu().numpy(),
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Compare] device={device}")
    print(f"[Compare] checkpoint={args.checkpoint_path}")
    print(f"[Compare] hnn_checkpoint={args.hnn_checkpoint_path}")
    print(f"[Compare] h5_path={args.h5_path}")

    model = TrajectoryDPF.load_from_checkpoint(args.checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()

    hnn_model = HNNWrapper.load_from_checkpoint(args.hnn_checkpoint_path, map_location=device)
    hnn_model = hnn_model.to(device)
    hnn_model.eval()

    rows: list[dict] = []
    rng = np.random.default_rng(int(args.seed))

    with h5py.File(args.h5_path, "r") as h5_file:
        total_available = int(h5_file.attrs["num_trajectories"])
        start_index = max(0, int(args.start_index))
        if start_index >= total_available:
            raise ValueError(f"start_index={start_index} is out of range for dataset size {total_available}")
        num_trajectories = min(int(args.num_trajectories), total_available - start_index)
        trajectory_length = int(h5_file.attrs["num_steps"])
        prefix_max = trajectory_length - 1 if int(args.prefix_max) <= 0 else min(int(args.prefix_max), trajectory_length - 1)
        prefix_min = max(1, min(int(args.prefix_min), prefix_max))

        print(
            f"[Compare] num_trajectories={num_trajectories} trajectory_length={trajectory_length} "
            f"prefix_range=[{prefix_min}, {prefix_max}] diffusion_steps={args.num_diffusion_steps}"
        )

        for local_idx, traj_index in enumerate(range(start_index, start_index + num_trajectories)):
            traj = h5_file[f"traj_{traj_index}"]
            gt_qpos_raw = traj["seq_qpos"][:].astype(np.float32)
            gt_mom = traj["seq_mom"][:].astype(np.float32)
            gt_tau = traj["seq_torque"][:].astype(np.float32)
            gt_qpos_model = encode_qpos_array(gt_qpos_raw, model.qpos_representation).astype(np.float32, copy=False)

            prefix_len = int(rng.integers(prefix_min, prefix_max + 1))
            observed_qpos = torch.from_numpy(gt_qpos_model).unsqueeze(0).to(device=device, dtype=torch.float32)
            observed_mom = torch.from_numpy(gt_mom).unsqueeze(0).to(device=device, dtype=torch.float32)
            observed_torque = torch.from_numpy(gt_tau).unsqueeze(0).to(device=device, dtype=torch.float32)

            noise_seed = int(args.seed) + int(traj_index) * 1009 + int(prefix_len) * 17
            noise_generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
            noise_generator.manual_seed(noise_seed)
            initial_noise = torch.randn(
                1,
                trajectory_length,
                model.state_dim,
                generator=noise_generator,
                device=device,
                dtype=torch.float32,
            )

            unguided_state, _ = run_completion(
                model=model,
                hnn_model=None,
                observed_qpos=observed_qpos,
                observed_mom=observed_mom,
                observed_torque=observed_torque,
                prefix_len=prefix_len,
                trajectory_length=trajectory_length,
                num_diffusion_steps=int(args.num_diffusion_steps),
                initial_noise=initial_noise,
                guidance_method=str(args.guidance_method),
                alpha_q=float(args.alpha_q),
                alpha_p=float(args.alpha_p),
                guidance_trust_lambda=float(args.guidance_trust_lambda),
                guidance_normalize_grad=bool(args.guidance_normalize_grad),
                guidance_joint_update=bool(args.guidance_joint_update),
            )
            guided_state, _ = run_completion(
                model=model,
                hnn_model=hnn_model,
                observed_qpos=observed_qpos,
                observed_mom=observed_mom,
                observed_torque=observed_torque,
                prefix_len=prefix_len,
                trajectory_length=trajectory_length,
                num_diffusion_steps=int(args.num_diffusion_steps),
                initial_noise=initial_noise.clone(),
                guidance_method=str(args.guidance_method),
                alpha_q=float(args.alpha_q),
                alpha_p=float(args.alpha_p),
                guidance_trust_lambda=float(args.guidance_trust_lambda),
                guidance_normalize_grad=bool(args.guidance_normalize_grad),
                guidance_joint_update=bool(args.guidance_joint_update),
            )

            unguided_qpos_raw = decode_qpos_array(
                unguided_state[:, : model.qpos_dim],
                model.qpos_representation,
            ).astype(np.float32, copy=False)
            guided_qpos_raw = decode_qpos_array(
                guided_state[:, : model.qpos_dim],
                model.qpos_representation,
            ).astype(np.float32, copy=False)
            unguided_mom = unguided_state[:, model.qpos_dim : model.qpos_dim + model.mom_dim].astype(np.float32, copy=False)
            guided_mom = guided_state[:, model.qpos_dim : model.qpos_dim + model.mom_dim].astype(np.float32, copy=False)

            row = {
                "traj_index": int(traj_index),
                "prefix_len": int(prefix_len),
                "noise_seed": int(noise_seed),
                "unguided_full_mse_qpos": mse(unguided_qpos_raw, gt_qpos_raw),
                "unguided_full_mse_mom": mse(unguided_mom, gt_mom),
                "guided_full_mse_qpos": mse(guided_qpos_raw, gt_qpos_raw),
                "guided_full_mse_mom": mse(guided_mom, gt_mom),
            }
            row["unguided_full_mse_total"] = float(row["unguided_full_mse_qpos"] + row["unguided_full_mse_mom"])
            row["guided_full_mse_total"] = float(row["guided_full_mse_qpos"] + row["guided_full_mse_mom"])
            row["delta_full_mse_total"] = float(row["guided_full_mse_total"] - row["unguided_full_mse_total"])
            rows.append(row)

            if (local_idx + 1) % 50 == 0 or (local_idx + 1) == num_trajectories:
                print(
                    f"[Compare] processed {local_idx + 1}/{num_trajectories} "
                    f"mean_unguided={np.mean([r['unguided_full_mse_total'] for r in rows]):.6f} "
                    f"mean_guided={np.mean([r['guided_full_mse_total'] for r in rows]):.6f}"
                )

    report = {
        "checkpoint_path": args.checkpoint_path,
        "hnn_checkpoint_path": args.hnn_checkpoint_path,
        "h5_path": args.h5_path,
        "device": str(device),
        "seed": int(args.seed),
        "num_trajectories": int(len(rows)),
        "start_index": int(args.start_index),
        "num_diffusion_steps": int(args.num_diffusion_steps),
        "prefix_min": int(prefix_min),
        "prefix_max": int(prefix_max),
        "guidance_method": str(args.guidance_method),
        "alpha_q": float(args.alpha_q),
        "alpha_p": float(args.alpha_p),
        "guidance_trust_lambda": float(args.guidance_trust_lambda),
        "guidance_normalize_grad": bool(args.guidance_normalize_grad),
        "guidance_joint_update": bool(args.guidance_joint_update),
        "aggregate": {
            "unguided_full_mse_qpos": aggregate_metric([row["unguided_full_mse_qpos"] for row in rows]),
            "unguided_full_mse_mom": aggregate_metric([row["unguided_full_mse_mom"] for row in rows]),
            "unguided_full_mse_total": aggregate_metric([row["unguided_full_mse_total"] for row in rows]),
            "guided_full_mse_qpos": aggregate_metric([row["guided_full_mse_qpos"] for row in rows]),
            "guided_full_mse_mom": aggregate_metric([row["guided_full_mse_mom"] for row in rows]),
            "guided_full_mse_total": aggregate_metric([row["guided_full_mse_total"] for row in rows]),
            "delta_full_mse_total": aggregate_metric([row["delta_full_mse_total"] for row in rows]),
            "guided_better_fraction": float(np.mean([row["guided_full_mse_total"] < row["unguided_full_mse_total"] for row in rows])),
        },
        "rows": rows,
    }

    report_path = output_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Compare] report={report_path}")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
