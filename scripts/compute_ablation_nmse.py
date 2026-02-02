"""
Compute Relative L2 Error (RL2) for saved ablation trajectories.

RL2 = ||gen - recon||_2 / ||recon||_2  per DOF, then averaged across DOFs.
Scale-invariant, stable (no blow-up for short/constant signals).
0 = perfect match, 0.05 = 5% relative error.

Optimized v3: single-pass workers, persistent pool, per-worker caches,
bulk-submit all groups at once, subsample for speed.

Usage:
    python scripts/compute_ablation_nmse.py --num_workers 48 --num_samples 200
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import h5py
import numpy as np
import pandas as pd
import os
import time
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.models.utils import reconstruct_traj_with_momentum
from src.models.HNN import HNNWrapper

import mujoco

XML_PATH = str(project_root / 'configs' / 'rigid_arm_hinge.xml')

# Per-worker caches (persist across tasks in the same worker process)
_worker_hnn = None
_worker_hnn_path = None
_worker_mj_model = None


def _get_mj_model():
    global _worker_mj_model
    if _worker_mj_model is None:
        _worker_mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    return _worker_mj_model


def _get_hnn(hnn_path, device):
    global _worker_hnn, _worker_hnn_path
    if _worker_hnn is None or _worker_hnn_path != hnn_path:
        _worker_hnn = HNNWrapper.load_from_checkpoint(hnn_path, map_location=device)
        _worker_hnn = _worker_hnn.to(device)
        _worker_hnn.eval()
        _worker_hnn_path = hnn_path
    return _worker_hnn


def compute_single_rl2_all(args_tuple):
    """Single-pass worker: one MuJoCo recon → traj RL2 + energy RL2.

    Returns (group_key, sample_idx, rl2_qpos, rl2_mom, rl2_energy).
    """
    group_key, sample_idx, state_np, torque_np, qpos_dim, dt, data_dt, hnn_path, hnn_device = args_tuple

    qpos_gen = state_np[:, :qpos_dim]
    mom_gen = state_np[:, qpos_dim:]

    # --- MuJoCo reconstruction (done ONCE) ---
    model = _get_mj_model()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_gen[0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    initial_qvel = np.linalg.solve(M, mom_gen[0])

    recon = reconstruct_traj_with_momentum(
        model, len(qpos_gen), dt,
        qpos_gen[0], initial_qvel, torque_np,
        data_dt=data_dt,
    )
    qpos_r = recon['seq_qpos']
    mom_r = recon['seq_mom']

    # --- Trajectory RL2 ---
    eps = 1e-10
    rl2_qpos = np.sqrt(np.sum((qpos_gen[1:] - qpos_r) ** 2, axis=0)) / \
               np.maximum(np.sqrt(np.sum(qpos_r ** 2, axis=0)), eps)
    rl2_mom = np.sqrt(np.sum((mom_gen[1:] - mom_r) ** 2, axis=0)) / \
              np.maximum(np.sqrt(np.sum(mom_r ** 2, axis=0)), eps)

    # --- Energy RL2 (reuse same reconstruction) ---
    qpos_recon_full = np.concatenate([qpos_gen[:1], qpos_r], axis=0)
    mom_recon_full = np.concatenate([mom_gen[:1], mom_r], axis=0)

    hnn = _get_hnn(hnn_path, hnn_device)
    with torch.no_grad():
        q_gen_t = torch.tensor(qpos_gen, dtype=torch.float32, device=hnn_device)
        p_gen_t = torch.tensor(mom_gen, dtype=torch.float32, device=hnn_device)
        H_gen = hnn(p_gen_t, q_gen_t).cpu().numpy().squeeze(-1)

        q_rec_t = torch.tensor(qpos_recon_full, dtype=torch.float32, device=hnn_device)
        p_rec_t = torch.tensor(mom_recon_full, dtype=torch.float32, device=hnn_device)
        H_recon = hnn(p_rec_t, q_rec_t).cpu().numpy().squeeze(-1)

    rl2_energy = np.sqrt(np.sum((H_gen - H_recon) ** 2)) / \
                 max(np.sqrt(np.sum(H_recon ** 2)), eps)

    return group_key, sample_idx, rl2_qpos, rl2_mom, rl2_energy


def aggregate_group_results(results_dict):
    """Aggregate per-sample results into group metrics."""
    rl2_qpos_arr = np.stack([r[0] for r in results_dict.values()], axis=0)
    rl2_mom_arr = np.stack([r[1] for r in results_dict.values()], axis=0)
    rl2_energy_arr = np.array([r[2] for r in results_dict.values()])

    rl2_qpos_per_sample = np.mean(rl2_qpos_arr, axis=1)
    rl2_mom_per_sample = np.mean(rl2_mom_arr, axis=1)
    rl2_total_per_sample = (rl2_qpos_per_sample + rl2_mom_per_sample) / 2

    return {
        'rl2_qpos_mean': np.mean(rl2_qpos_per_sample),
        'rl2_qpos_std': np.std(rl2_qpos_per_sample),
        'rl2_mom_mean': np.mean(rl2_mom_per_sample),
        'rl2_mom_std': np.std(rl2_mom_per_sample),
        'rl2_total_mean': np.mean(rl2_total_per_sample),
        'rl2_total_std': np.std(rl2_total_per_sample),
        'rl2_energy_mean': np.mean(rl2_energy_arr),
        'rl2_energy_std': np.std(rl2_energy_arr),
    }


def process_exp_a_file(executor, h5_path, output_csv, qpos_dim, dt, data_dt,
                       hnn_path, hnn_device, num_samples):
    """Process one Experiment A H5 file → RL2 CSV. Bulk-submit all groups."""
    print(f"Processing: {h5_path}", flush=True)
    file_start = time.time()

    with h5py.File(h5_path, 'r') as f:
        all_groups = []
        for mode in ['unguided', 'guided']:
            if mode in f:
                for length_key in sorted(f[mode].keys(), key=lambda x: int(x[1:])):
                    all_groups.append(f'{mode}/{length_key}')

        # Submit ALL groups' samples at once
        all_futures = {}
        group_sample_counts = {}
        for group_name in all_groups:
            grp = f[group_name]
            states = grp['state'][:]
            torques = grp['torque'][:]
            total = states.shape[0]
            n = min(num_samples, total)
            indices = np.random.choice(total, n, replace=False) if n < total else np.arange(total)
            group_sample_counts[group_name] = n

            for i, idx in enumerate(indices):
                args = (group_name, i, states[idx], torques[idx],
                        qpos_dim, dt, data_dt, hnn_path, hnn_device)
                fut = executor.submit(compute_single_rl2_all, args)
                all_futures[fut] = (group_name, i)

    total_tasks = len(all_futures)
    print(f"  Submitted {total_tasks} tasks across {len(all_groups)} groups "
          f"({num_samples} samples each)", flush=True)

    # Collect results grouped by group_name
    group_results = {g: {} for g in all_groups}
    done_count = 0
    for future in as_completed(all_futures):
        gk, si, rq, rm, re = future.result()
        group_results[gk][si] = (rq, rm, re)
        done_count += 1
        if done_count % 500 == 0:
            print(f"    {done_count}/{total_tasks} tasks done...", flush=True)

    # Aggregate and build CSV
    lengths = sorted(set(int(g.split('/')[1][1:]) for g in all_groups))
    results = []
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    for traj_length in lengths:
        row = {'trajectory_length': traj_length}
        for mode in ['unguided', 'guided']:
            group_name = f'{mode}/L{traj_length}'
            if group_name not in group_results or not group_results[group_name]:
                continue
            metrics = aggregate_group_results(group_results[group_name])
            prefix = f'{mode}_'
            for key, val in metrics.items():
                row[f'{prefix}{key}'] = val
            print(f"  L{traj_length} {mode}: "
                  f"RL2_total={metrics['rl2_total_mean']:.6f} "
                  f"RL2_energy={metrics['rl2_energy_mean']:.6f}", flush=True)
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    elapsed = time.time() - file_start
    print(f"  Saved: {output_csv} ({elapsed:.1f}s total)", flush=True)
    return df


def process_exp_b_file(executor, h5_path, output_csv, qpos_dim, dt, data_dt,
                       hnn_path, hnn_device, num_samples):
    """Process Experiment B H5 file → RL2 CSV. Bulk-submit all groups."""
    print(f"Processing: {h5_path}", flush=True)
    file_start = time.time()

    with h5py.File(h5_path, 'r') as f:
        all_groups = []
        for mode in ['unguided', 'guided']:
            if mode in f:
                for cf_key in sorted(f[mode].keys(), key=lambda x: float(x[3:])):
                    all_groups.append(f'{mode}/{cf_key}')

        all_futures = {}
        for group_name in all_groups:
            grp = f[group_name]
            states = grp['state'][:]
            torques = grp['torque'][:]
            total = states.shape[0]
            n = min(num_samples, total)
            indices = np.random.choice(total, n, replace=False) if n < total else np.arange(total)

            for i, idx in enumerate(indices):
                args = (group_name, i, states[idx], torques[idx],
                        qpos_dim, dt, data_dt, hnn_path, hnn_device)
                fut = executor.submit(compute_single_rl2_all, args)
                all_futures[fut] = (group_name, i)

    total_tasks = len(all_futures)
    print(f"  Submitted {total_tasks} tasks across {len(all_groups)} groups "
          f"({num_samples} samples each)", flush=True)

    group_results = {g: {} for g in all_groups}
    done_count = 0
    for future in as_completed(all_futures):
        gk, si, rq, rm, re = future.result()
        group_results[gk][si] = (rq, rm, re)
        done_count += 1
        if done_count % 500 == 0:
            print(f"    {done_count}/{total_tasks} tasks done...", flush=True)

    fractions = sorted(set(float(g.split('/')[1][3:]) for g in all_groups))
    results = []
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    for ctx_frac in fractions:
        row = {'context_fraction': ctx_frac}
        for mode in ['unguided', 'guided']:
            group_name = f'{mode}/cf_{ctx_frac:.2f}'
            if group_name not in group_results or not group_results[group_name]:
                continue
            metrics = aggregate_group_results(group_results[group_name])
            prefix = f'{mode}_'
            for key, val in metrics.items():
                row[f'{prefix}{key}'] = val
            print(f"  cf={ctx_frac:.2f} {mode}: "
                  f"RL2_total={metrics['rl2_total_mean']:.6f} "
                  f"RL2_energy={metrics['rl2_energy_mean']:.6f}", flush=True)
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    elapsed = time.time() - file_start
    print(f"  Saved: {output_csv} ({elapsed:.1f}s total)", flush=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compute Relative L2 Error for ablation trajectories")
    parser.add_argument("--input_dir", type=str, default="output_ablation/trajectories")
    parser.add_argument("--output_dir", type=str, default="output_ablation/results")
    parser.add_argument("--num_workers", type=int, default=48)
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Samples per group (default 200, use 0 for all)")
    parser.add_argument("--qpos_dim", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.0001)
    parser.add_argument("--data_dt", type=float, default=0.0002)
    parser.add_argument("--model_names", type=str, nargs="+",
                        default=["original", "global_cond", "torque_concat"])
    parser.add_argument("--hnn_checkpoint", type=str,
                        default="checkpoints/SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt")
    parser.add_argument("--hnn_device", type=str, default="cpu")
    parser.add_argument("--torque_conditions", type=str, default=None,
                        help="Comma-separated torque conditions (e.g., 'training,sinusoidal')")
    parser.add_argument("--skip_exp_b", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    input_root = project_root / args.input_dir
    output_root = project_root / args.output_dir
    hnn_path = str(project_root / args.hnn_checkpoint)

    num_samples = args.num_samples if args.num_samples > 0 else 999999

    print(f"Input: {input_root}")
    print(f"Output: {output_root}")
    print(f"Workers: {args.num_workers}")
    print(f"Samples per group: {args.num_samples if args.num_samples > 0 else 'all'}")
    print(f"HNN: {hnn_path}")

    total_start = time.time()

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for model_name in args.model_names:
            model_dir = input_root / model_name
            if not model_dir.exists():
                print(f"\nSkipping {model_name} (directory not found)")
                continue

            print(f"\n{'='*60}")
            print(f"  MODEL: {model_name}")
            print(f"{'='*60}", flush=True)

            torque_filter = None
            if args.torque_conditions:
                torque_filter = set(t.strip() for t in args.torque_conditions.split(','))

            for h5_file in sorted(model_dir.glob('exp_a_*.h5')):
                torque_label = h5_file.stem.replace('exp_a_', '')
                if torque_filter and torque_label not in torque_filter:
                    print(f"\n  Skipping exp_a_{torque_label} (not in filter)")
                    continue
                output_csv = output_root / model_name / f'exp_a_{torque_label}_rl2.csv'
                process_exp_a_file(
                    executor, str(h5_file), str(output_csv),
                    args.qpos_dim, args.dt, args.data_dt,
                    hnn_path, args.hnn_device, num_samples,
                )

            exp_b_file = model_dir / 'exp_b_context_fractions.h5'
            if args.skip_exp_b:
                print(f"\n  Skipping Experiment B (--skip_exp_b)")
            elif exp_b_file.exists():
                output_csv = output_root / model_name / 'exp_b_context_fractions_rl2.csv'
                process_exp_b_file(
                    executor, str(exp_b_file), str(output_csv),
                    args.qpos_dim, args.dt, args.data_dt,
                    hnn_path, args.hnn_device, num_samples,
                )

    total_elapsed = time.time() - total_start
    print(f"\nAll computation complete in {total_elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
