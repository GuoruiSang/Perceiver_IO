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

from scripts.data.run_reacher_goal_prefix_expansion import (
    TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_TARGET_ACROSS_TRAJS,
    apply_ema_once,
    build_observed_windows,
    build_validation_random_source_random_target_across_trajs_task,
    fingertip_xy_from_qpos_raw,
)
from src.models.trajectory_dpf_model import TrajectoryDPF
from src.models.utils import compare_generated_with_reconstructed
from src.qpos_representation import decode_qpos_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate unguided endpoint-conditioned DPF inpainting quality on the actual "
            "across-trajectory Reacher task setup, sweeping num_diffusion_steps."
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
        "--h5_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/data/reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/"
        "val_traj_4000-steps_1000.h5",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/plots/reacher_dpf_inpainting_quality_sweep",
    )
    parser.add_argument("--task_ids", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--num_diffusion_steps_list", type=str, default="20,50,100")
    parser.add_argument(
        "--conditioning_prefix_len_list",
        type=str,
        default="1",
        help=(
            "Observed source-prefix lengths to test. Uses the real source trajectory states and torques "
            "for those prefix steps before endpoint-conditioned generation starts."
        ),
    )
    parser.add_argument(
        "--goal_condition_mode",
        type=str,
        default="qpos_soft_block",
        choices=("qpos_mom_last", "qpos_last", "qpos_soft_block"),
        help=(
            "Terminal conditioning mode. qpos_soft_block with block_len=1 is equivalent to "
            "terminal qpos-only conditioning."
        ),
    )
    parser.add_argument(
        "--goal_condition_block_len_list",
        type=str,
        default="1",
        help=(
            "Terminal block lengths to test. For qpos_soft_block, the final state stays exact and the "
            "preceding terminal qpos states are Gaussian-softened toward the goal."
        ),
    )
    parser.add_argument(
        "--goal_condition_qpos_noise_std",
        type=float,
        default=0.05,
        help="Base raw-qpos Gaussian std for qpos_soft_block terminal conditioning.",
    )
    parser.add_argument(
        "--goal_condition_noise_decay",
        type=str,
        default="linear",
        choices=("constant", "linear"),
        help="How terminal qpos soft-block noise decays toward the final exact goal state.",
    )
    parser.add_argument("--lookahead_steps", type=int, default=256)
    parser.add_argument("--min_initial_distance", type=float, default=0.1)
    parser.add_argument("--cross_traj_sampling_max_tries", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot_first_n", type=int, default=6)
    parser.add_argument(
        "--plot_dpi",
        type=int,
        default=360,
        help="DPI for saved wandb-style generated-vs-reconstructed comparison plots.",
    )
    parser.add_argument(
        "--summary_name",
        type=str,
        default="inpainting_quality_sweep_summary.json",
    )
    return parser.parse_args()


def parse_indices(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer index.")
    return values


def summarize(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def compute_transition_metrics(
    ee_xy: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
) -> dict[str, float]:
    ee_xy = np.asarray(ee_xy, dtype=np.float64)
    start_xy = np.asarray(start_xy, dtype=np.float64)
    goal_xy = np.asarray(goal_xy, dtype=np.float64)

    eps = 1e-12
    step_vecs = ee_xy[1:] - ee_xy[:-1]
    step_lengths = np.linalg.norm(step_vecs, axis=1)
    path_length = float(step_lengths.sum())

    chord_vec = goal_xy - start_xy
    chord_len = float(np.linalg.norm(chord_vec))
    if chord_len > eps:
        chord_dir = chord_vec / chord_len
        signed_progress = step_vecs @ chord_dir
        progress_step_fraction = float(np.mean(signed_progress > 0.0))
        regress_step_fraction = float(np.mean(signed_progress < 0.0))
        path_efficiency = float(chord_len / max(path_length, eps))

        scalar_proj = (ee_xy - start_xy[None, :]) @ chord_dir
        scalar_proj = np.clip(scalar_proj, 0.0, chord_len)
        closest_on_chord = start_xy[None, :] + scalar_proj[:, None] * chord_dir[None, :]
        chord_deviation = np.linalg.norm(ee_xy - closest_on_chord, axis=1)
    else:
        progress_step_fraction = 0.0
        regress_step_fraction = 0.0
        path_efficiency = 1.0
        chord_deviation = np.linalg.norm(ee_xy - start_xy[None, :], axis=1)

    interior_deviation = chord_deviation[1:-1] if chord_deviation.shape[0] > 2 else chord_deviation[:0]
    if interior_deviation.size == 0:
        mean_chord_deviation = 0.0
        max_chord_deviation = 0.0
    else:
        mean_chord_deviation = float(np.mean(interior_deviation))
        max_chord_deviation = float(np.max(interior_deviation))

    goal_dists = np.linalg.norm(ee_xy - goal_xy[None, :], axis=1)
    if goal_dists.shape[0] > 1:
        goal_distance_decrease_fraction = float(np.mean(goal_dists[1:] <= goal_dists[:-1] + 1e-12))
    else:
        goal_distance_decrease_fraction = 1.0

    return {
        "transition_path_length": path_length,
        "transition_path_efficiency": path_efficiency,
        "transition_progress_step_fraction": progress_step_fraction,
        "transition_regress_step_fraction": regress_step_fraction,
        "transition_mean_chord_deviation": mean_chord_deviation,
        "transition_max_chord_deviation": max_chord_deviation,
        "transition_goal_distance_decrease_fraction": goal_distance_decrease_fraction,
    }


def compute_boundary_discontinuity_metrics(
    generated_qpos_raw: np.ndarray,
    generated_mom: np.ndarray,
    generated_ee_xy: np.ndarray,
    conditioning_prefix_len: int,
) -> dict[str, float]:
    qpos = np.asarray(generated_qpos_raw, dtype=np.float64)
    mom = np.asarray(generated_mom, dtype=np.float64)
    ee_xy = np.asarray(generated_ee_xy, dtype=np.float64)

    if qpos.shape[0] < 2:
        return {
            "start_qpos_jump": 0.0,
            "end_qpos_jump": 0.0,
            "start_mom_jump": 0.0,
            "end_mom_jump": 0.0,
            "start_ee_jump": 0.0,
            "end_ee_jump": 0.0,
            "mean_qpos_step": 0.0,
            "mean_mom_step": 0.0,
            "mean_ee_step": 0.0,
            "start_qpos_jump_over_mean_step": 0.0,
            "end_qpos_jump_over_mean_step": 0.0,
            "start_mom_jump_over_mean_step": 0.0,
            "end_mom_jump_over_mean_step": 0.0,
            "start_ee_jump_over_mean_step": 0.0,
            "end_ee_jump_over_mean_step": 0.0,
        }

    eps = 1e-12
    qpos_step = np.linalg.norm(qpos[1:] - qpos[:-1], axis=1)
    mom_step = np.linalg.norm(mom[1:] - mom[:-1], axis=1)
    ee_step = np.linalg.norm(ee_xy[1:] - ee_xy[:-1], axis=1)
    boundary_idx = int(max(0, min(int(conditioning_prefix_len) - 1, qpos_step.shape[0] - 1)))

    # Use purely generated interior-to-interior steps for normalization when available.
    interior_start = int(min(boundary_idx + 1, qpos_step.shape[0]))
    interior_end = int(max(interior_start, qpos_step.shape[0] - 1))
    interior_qpos_step = qpos_step[interior_start:interior_end]
    interior_mom_step = mom_step[interior_start:interior_end]
    interior_ee_step = ee_step[interior_start:interior_end]
    if interior_qpos_step.size == 0:
        interior_qpos_step = qpos_step
    if interior_mom_step.size == 0:
        interior_mom_step = mom_step
    if interior_ee_step.size == 0:
        interior_ee_step = ee_step

    mean_qpos_step = float(np.mean(interior_qpos_step))
    mean_mom_step = float(np.mean(interior_mom_step))
    mean_ee_step = float(np.mean(interior_ee_step))

    start_qpos_jump = float(qpos_step[boundary_idx])
    end_qpos_jump = float(qpos_step[-1])
    start_mom_jump = float(mom_step[boundary_idx])
    end_mom_jump = float(mom_step[-1])
    start_ee_jump = float(ee_step[boundary_idx])
    end_ee_jump = float(ee_step[-1])

    return {
        "start_qpos_jump": start_qpos_jump,
        "end_qpos_jump": end_qpos_jump,
        "start_mom_jump": start_mom_jump,
        "end_mom_jump": end_mom_jump,
        "start_ee_jump": start_ee_jump,
        "end_ee_jump": end_ee_jump,
        "mean_qpos_step": mean_qpos_step,
        "mean_mom_step": mean_mom_step,
        "mean_ee_step": mean_ee_step,
        "start_qpos_jump_over_mean_step": float(start_qpos_jump / max(mean_qpos_step, eps)),
        "end_qpos_jump_over_mean_step": float(end_qpos_jump / max(mean_qpos_step, eps)),
        "start_mom_jump_over_mean_step": float(start_mom_jump / max(mean_mom_step, eps)),
        "end_mom_jump_over_mean_step": float(end_mom_jump / max(mean_mom_step, eps)),
        "start_ee_jump_over_mean_step": float(start_ee_jump / max(mean_ee_step, eps)),
        "end_ee_jump_over_mean_step": float(end_ee_jump / max(mean_ee_step, eps)),
    }


def main() -> None:
    args = parse_args()
    task_ids = parse_indices(args.task_ids)
    diffusion_steps_list = parse_indices(args.num_diffusion_steps_list)
    prefix_len_list = parse_indices(args.conditioning_prefix_len_list)
    goal_block_len_list = parse_indices(args.goal_condition_block_len_list)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / args.summary_name

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[InpaintQuality] device={device}")
    print(f"[InpaintQuality] output_dir={output_dir}")
    print(f"[InpaintQuality] task_ids={task_ids}")
    print(f"[InpaintQuality] diffusion_steps_list={diffusion_steps_list}")
    print(f"[InpaintQuality] prefix_len_list={prefix_len_list}")
    print(f"[InpaintQuality] goal_condition_mode={args.goal_condition_mode}")
    print(f"[InpaintQuality] goal_block_len_list={goal_block_len_list}")

    model = TrajectoryDPF.load_from_checkpoint(args.checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    apply_ema_once(model)

    all_setting_rows: list[dict] = []
    with h5py.File(args.h5_path, "r") as h5_file:
        xml_content = h5_file.attrs["xml"]
        if isinstance(xml_content, bytes):
            xml_content = xml_content.decode()
        rollout_limit = int(h5_file.attrs["num_steps"])
        tasks = [
            build_validation_random_source_random_target_across_trajs_task(
                h5_file=h5_file,
                task_seed=int(args.seed) + int(task_id),
                rollout_limit=rollout_limit,
                min_initial_distance=float(args.min_initial_distance),
                max_tries=int(args.cross_traj_sampling_max_tries),
            )
            for task_id in task_ids
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / "model.xml"
            xml_path.write_text(xml_content, encoding="utf-8")

            for conditioning_prefix_len in prefix_len_list:
                for goal_block_len in goal_block_len_list:
                    for num_diffusion_steps in diffusion_steps_list:
                        setting_dir = (
                            output_dir
                            / f"prefix{int(conditioning_prefix_len)}_goalK{int(goal_block_len)}_ndd{int(num_diffusion_steps)}"
                        )
                        setting_dir.mkdir(parents=True, exist_ok=True)
                        rows: list[dict] = []
                        skipped: list[dict] = []
                        for order, task in enumerate(tasks):
                            source_traj_index = int(task["metadata"]["source_traj_index"])
                            source_index = int(task["metadata"]["source_index"])
                            source_traj = h5_file[f"traj_{source_traj_index}"]
                            source_torque_full = source_traj["seq_torque"][:].astype(np.float64)
                            reference_qpos_raw = np.asarray(task["reference_qpos_raw"], dtype=np.float64)
                            reference_mom = np.asarray(task["reference_mom"], dtype=np.float64)
                            max_prefix_available = int(reference_qpos_raw.shape[0])
                            prefix_len = int(conditioning_prefix_len)
                            if prefix_len >= max_prefix_available:
                                skipped.append(
                                    {
                                        "task_id": int(task["task_id"]),
                                        "reason": (
                                            f"requested prefix_len={prefix_len} but only "
                                            f"{max_prefix_available} source states available"
                                        ),
                                    }
                                )
                                continue
                            qpos_prefix_raw = [step.copy() for step in reference_qpos_raw[:prefix_len]]
                            mom_prefix = [step.copy() for step in reference_mom[:prefix_len]]
                            torque_prefix_np = source_torque_full[source_index : source_index + prefix_len - 1]
                            torque_prefix = [step.astype(np.float32, copy=False) for step in torque_prefix_np]
                            sample_horizon = prefix_len + int(args.lookahead_steps)

                            observed_qpos, observed_mom, observed_tau, observed_state_mask = build_observed_windows(
                                qpos_prefix_raw=qpos_prefix_raw,
                                mom_prefix=mom_prefix,
                                torque_prefix=torque_prefix,
                                crop_start=0,
                                conditioning_prefix_len=prefix_len,
                                sample_horizon=sample_horizon,
                                model=model,
                                device=device,
                                condition_suffix_last_state_on_goal=False,
                                goal_condition_mode=str(args.goal_condition_mode),
                                goal_condition_block_len=int(goal_block_len),
                                goal_condition_qpos_noise_std=float(args.goal_condition_qpos_noise_std),
                                goal_condition_noise_decay=str(args.goal_condition_noise_decay),
                                goal_condition_seed=(
                                    int(args.seed) * 1000003
                                    + int(task["task_id"]) * 10007
                                    + int(prefix_len) * 101
                                    + int(goal_block_len) * 37
                                    + int(num_diffusion_steps) * 17
                                ),
                                target_qpos_raw=np.asarray(task["target_qpos_raw"], dtype=np.float64),
                                target_mom=np.asarray(task["target_mom"], dtype=np.float64),
                            )
                            time_indices = torch.arange(sample_horizon, dtype=torch.long, device=device)

                            local_seed = (
                                int(args.seed)
                                + int(task["task_id"]) * 1000
                                + int(num_diffusion_steps) * 10
                                + prefix_len
                                + int(goal_block_len) * 100
                            )
                            torch.manual_seed(local_seed)
                            np.random.seed(local_seed)
                            with torch.no_grad():
                                generated_state, generated_tau = model.sample_trajectories(
                                    num_samples=1,
                                    trajectory_length=sample_horizon,
                                    num_diffusion_steps=int(num_diffusion_steps),
                                    prefix_len=prefix_len,
                                    observed_qpos=observed_qpos,
                                    observed_mom=observed_mom,
                                    observed_torque=observed_tau,
                                    observed_state_mask=observed_state_mask,
                                    time_indices=time_indices,
                                    use_ema=False,
                                    sampler="ddim",
                                )

                            generated_state_np = generated_state[0].detach().cpu().numpy()
                            generated_tau_np = generated_tau[0].detach().cpu().numpy()
                            generated_qpos_model = generated_state_np[:, : model.qpos_dim]
                            generated_mom = generated_state_np[:, model.qpos_dim : model.qpos_dim + model.mom_dim]
                            generated_qpos_raw = decode_qpos_array(
                                generated_qpos_model,
                                model.qpos_representation,
                            ).astype(np.float64, copy=False)
                            generated_ee_xy = fingertip_xy_from_qpos_raw(generated_qpos_raw)

                            start_ee_xy = fingertip_xy_from_qpos_raw(
                                np.asarray(task["initial_qpos_raw"], dtype=np.float64)[None, :]
                            )[0]
                            goal_xy = np.asarray(task["goal_xy"], dtype=np.float64)
                            target_qpos_raw = np.asarray(task["target_qpos_raw"], dtype=np.float64)
                            target_mom = np.asarray(task["target_mom"], dtype=np.float64)
                            transition_metrics = compute_transition_metrics(
                                generated_ee_xy,
                                start_ee_xy,
                                goal_xy,
                            )
                            boundary_metrics = compute_boundary_discontinuity_metrics(
                                generated_qpos_raw,
                                generated_mom,
                                generated_ee_xy,
                                conditioning_prefix_len=prefix_len,
                            )

                            plot_name = None
                            if order < int(args.plot_first_n):
                                plot_name = (
                                    f"task_{int(task['task_id']):04d}_"
                                    f"src{int(task['metadata']['source_index']):04d}_"
                                    f"goal{int(task['metadata']['target_index']):04d}"
                                )
                            compare_row = compare_generated_with_reconstructed(
                                generated={
                                    "seq_qpos": generated_qpos_model,
                                    "seq_mom": generated_mom,
                                    "seq_torque": generated_tau_np,
                                },
                                mujoco_model_path=str(xml_path),
                                save_path=str(setting_dir),
                                dt=float(model.dt),
                                data_dt=float(model.data_dt),
                                name=plot_name,
                                trajectory_alignment="pre_step",
                                prefix_len=prefix_len,
                                qpos_representation=model.qpos_representation,
                                plot_dpi=int(args.plot_dpi),
                            )

                            row = {
                                "task_id": int(task["task_id"]),
                                "task_label": str(task["task_label"]),
                                "conditioning_prefix_len": int(prefix_len),
                                "goal_condition_mode": str(args.goal_condition_mode),
                                "goal_condition_block_len": int(goal_block_len),
                                "goal_condition_qpos_noise_std": float(args.goal_condition_qpos_noise_std),
                                "goal_condition_noise_decay": str(args.goal_condition_noise_decay),
                                "num_diffusion_steps": int(num_diffusion_steps),
                                "source_index": int(task["metadata"]["source_index"]),
                                "target_index": int(task["metadata"]["target_index"]),
                                "initial_goal_distance": float(task["metadata"]["initial_goal_distance"]),
                                "mse_qpos": float(compare_row["mse_qpos"]),
                                "mse_mom": float(compare_row["mse_mom"]),
                                "mse_total": float(compare_row["mse_total"]),
                                "source_qpos_endpoint_error": float(
                                    np.linalg.norm(
                                        generated_qpos_raw[0] - np.asarray(task["initial_qpos_raw"], dtype=np.float64)
                                    )
                                ),
                                "source_mom_endpoint_error": float(
                                    np.linalg.norm(generated_mom[0] - np.asarray(task["initial_mom"], dtype=np.float64))
                                ),
                                "target_qpos_endpoint_error": float(
                                    np.linalg.norm(generated_qpos_raw[-1] - target_qpos_raw)
                                ),
                                "target_mom_endpoint_error": float(np.linalg.norm(generated_mom[-1] - target_mom)),
                                "max_goal_distance_generated": float(
                                    np.linalg.norm(generated_ee_xy - goal_xy[None, :], axis=1).max()
                                ),
                                "mean_goal_distance_generated": float(
                                    np.linalg.norm(generated_ee_xy - goal_xy[None, :], axis=1).mean()
                                ),
                                "start_ee_distance_generated": float(np.linalg.norm(generated_ee_xy[0] - start_ee_xy)),
                                "plot_path": (str(setting_dir / f"{plot_name}.jpg") if plot_name is not None else None),
                            }
                            row.update(transition_metrics)
                            row.update(boundary_metrics)
                            rows.append(row)
                            print(
                                "[InpaintQuality] prefix={} goalK={} ndd={} task_id={} start_ee_jump={:.4f} end_ee_jump={:.4f} "
                                "start_jump_ratio={:.2f} end_jump_ratio={:.2f}".format(
                                    prefix_len,
                                    int(goal_block_len),
                                    int(num_diffusion_steps),
                                    int(task["task_id"]),
                                    row["start_ee_jump"],
                                    row["end_ee_jump"],
                                    row["start_ee_jump_over_mean_step"],
                                    row["end_ee_jump_over_mean_step"],
                                )
                            )

                        setting_summary = {
                            "conditioning_prefix_len": int(conditioning_prefix_len),
                            "goal_condition_mode": str(args.goal_condition_mode),
                            "goal_condition_block_len": int(goal_block_len),
                            "goal_condition_qpos_noise_std": float(args.goal_condition_qpos_noise_std),
                            "goal_condition_noise_decay": str(args.goal_condition_noise_decay),
                            "num_diffusion_steps": int(num_diffusion_steps),
                            "num_tasks": int(len(rows)),
                            "skipped": skipped,
                            "aggregate": {
                                "mse_total": summarize(rows, "mse_total"),
                                "mse_qpos": summarize(rows, "mse_qpos"),
                                "mse_mom": summarize(rows, "mse_mom"),
                                "source_qpos_endpoint_error": summarize(rows, "source_qpos_endpoint_error"),
                                "source_mom_endpoint_error": summarize(rows, "source_mom_endpoint_error"),
                                "target_qpos_endpoint_error": summarize(rows, "target_qpos_endpoint_error"),
                                "target_mom_endpoint_error": summarize(rows, "target_mom_endpoint_error"),
                                "mean_goal_distance_generated": summarize(rows, "mean_goal_distance_generated"),
                                "max_goal_distance_generated": summarize(rows, "max_goal_distance_generated"),
                                "transition_path_length": summarize(rows, "transition_path_length"),
                                "transition_path_efficiency": summarize(rows, "transition_path_efficiency"),
                                "transition_progress_step_fraction": summarize(rows, "transition_progress_step_fraction"),
                                "transition_regress_step_fraction": summarize(rows, "transition_regress_step_fraction"),
                                "transition_mean_chord_deviation": summarize(rows, "transition_mean_chord_deviation"),
                                "transition_max_chord_deviation": summarize(rows, "transition_max_chord_deviation"),
                                "transition_goal_distance_decrease_fraction": summarize(
                                    rows, "transition_goal_distance_decrease_fraction"
                                ),
                                "start_qpos_jump": summarize(rows, "start_qpos_jump"),
                                "end_qpos_jump": summarize(rows, "end_qpos_jump"),
                                "start_mom_jump": summarize(rows, "start_mom_jump"),
                                "end_mom_jump": summarize(rows, "end_mom_jump"),
                                "start_ee_jump": summarize(rows, "start_ee_jump"),
                                "end_ee_jump": summarize(rows, "end_ee_jump"),
                                "mean_qpos_step": summarize(rows, "mean_qpos_step"),
                                "mean_mom_step": summarize(rows, "mean_mom_step"),
                                "mean_ee_step": summarize(rows, "mean_ee_step"),
                                "start_qpos_jump_over_mean_step": summarize(rows, "start_qpos_jump_over_mean_step"),
                                "end_qpos_jump_over_mean_step": summarize(rows, "end_qpos_jump_over_mean_step"),
                                "start_mom_jump_over_mean_step": summarize(rows, "start_mom_jump_over_mean_step"),
                                "end_mom_jump_over_mean_step": summarize(rows, "end_mom_jump_over_mean_step"),
                                "start_ee_jump_over_mean_step": summarize(rows, "start_ee_jump_over_mean_step"),
                                "end_ee_jump_over_mean_step": summarize(rows, "end_ee_jump_over_mean_step"),
                            },
                            "rows": rows,
                        }
                        (setting_dir / "summary.json").write_text(json.dumps(setting_summary, indent=2), encoding="utf-8")
                        all_setting_rows.append(setting_summary)

    master_summary = {
        "checkpoint_path": args.checkpoint_path,
        "h5_path": args.h5_path,
        "device": str(device),
        "seed": int(args.seed),
        "task_mode": TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_TARGET_ACROSS_TRAJS,
        "task_ids": task_ids,
        "num_diffusion_steps_list": diffusion_steps_list,
        "conditioning_prefix_len_list": prefix_len_list,
        "goal_condition_mode": str(args.goal_condition_mode),
        "goal_condition_block_len_list": goal_block_len_list,
        "goal_condition_qpos_noise_std": float(args.goal_condition_qpos_noise_std),
        "goal_condition_noise_decay": str(args.goal_condition_noise_decay),
        "lookahead_steps": int(args.lookahead_steps),
        "min_initial_distance": float(args.min_initial_distance),
        "cross_traj_sampling_max_tries": int(args.cross_traj_sampling_max_tries),
        "plot_first_n": int(args.plot_first_n),
        "plot_dpi": int(args.plot_dpi),
        "settings": all_setting_rows,
    }
    summary_path.write_text(json.dumps(master_summary, indent=2), encoding="utf-8")
    print(f"[InpaintQuality] summary={summary_path}")


if __name__ == "__main__":
    main()
