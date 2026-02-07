"""
Compute MSE and NMSE metrics for 2-DoF ablation trajectories.

Loads generated trajectories from H5, runs MuJoCo physics reconstruction,
computes MSE/NMSE metrics.

Usage:
    python scripts/compute_ablation_mse_2dof.py --num_workers 24
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
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.models.utils import reconstruct_traj_with_momentum

import mujoco


# 2-DoF specific paths
XML_PATH = str(project_root / 'configs' / 'rigid_arm_hinge_2dof.xml')
INPUT_DIR = project_root / 'output_ablation' / 'trajectories' / '2dof'
OUTPUT_DIR = project_root / 'output_ablation' / 'results' / '2dof'

# Dimensions for 2-DoF system
QPOS_DIM = 2
DT = 0.0001
DATA_DT = 0.0002


def compute_single_mse(args_tuple):
    """Compute MSE and NMSE for a single trajectory."""
    state_np, torque_np, qpos_dim, dt, data_dt = args_tuple

    qpos_gen = state_np[:, :qpos_dim]
    mom_gen = state_np[:, qpos_dim:]

    # MuJoCo reconstruction
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    model.opt.timestep = dt  # Set correct simulation timestep
    data = mujoco.MjData(model)

    # Compute initial velocity from initial momentum: v = M^{-1} @ p
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

    # MSE: generated[1:] vs reconstructed (both length T-1)
    mse_qpos = np.mean((qpos_gen[1:] - recon['seq_qpos']) ** 2)
    mse_mom = np.mean((mom_gen[1:] - recon['seq_mom']) ** 2)

    # NMSE: MSE / var(generated)
    var_qpos = np.var(qpos_gen[1:])
    var_mom = np.var(mom_gen[1:])

    nmse_qpos = mse_qpos / var_qpos if var_qpos > 1e-12 else 0.0
    nmse_mom = mse_mom / var_mom if var_mom > 1e-12 else 0.0

    return mse_qpos, mse_mom, nmse_qpos, nmse_mom


def compute_metrics_for_group(h5_path, group_name, num_workers):
    """Compute MSE and NMSE metrics for all trajectories in an H5 group."""
    with h5py.File(h5_path, 'r') as f:
        grp = f[group_name]
        states = grp['state'][:]      # [N, T, state_dim]
        torques = grp['torque'][:]    # [N, T, torque_dim]

    num_samples = states.shape[0]

    # Prepare arguments for parallel MuJoCo reconstruction
    args_list = [
        (states[i], torques[i], QPOS_DIM, DT, DATA_DT)
        for i in range(num_samples)
    ]

    mse_qpos_list = []
    mse_mom_list = []
    nmse_qpos_list = []
    nmse_mom_list = []

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(compute_single_mse, args): i
                       for i, args in enumerate(args_list)}
            results_by_idx = {}
            for future in as_completed(futures):
                idx = futures[future]
                results_by_idx[idx] = future.result()

            for i in range(num_samples):
                mse_q, mse_m, nmse_q, nmse_m = results_by_idx[i]
                mse_qpos_list.append(mse_q)
                mse_mom_list.append(mse_m)
                nmse_qpos_list.append(nmse_q)
                nmse_mom_list.append(nmse_m)
    else:
        for args in args_list:
            mse_q, mse_m, nmse_q, nmse_m = compute_single_mse(args)
            mse_qpos_list.append(mse_q)
            mse_mom_list.append(mse_m)
            nmse_qpos_list.append(nmse_q)
            nmse_mom_list.append(nmse_m)

    metrics = {
        'mse_qpos_mean': np.mean(mse_qpos_list),
        'mse_qpos_std': np.std(mse_qpos_list),
        'mse_mom_mean': np.mean(mse_mom_list),
        'mse_mom_std': np.std(mse_mom_list),
        'nmse_q_mean': np.mean(nmse_qpos_list),
        'nmse_q_std': np.std(nmse_qpos_list),
        'nmse_p_mean': np.mean(nmse_mom_list),
        'nmse_p_std': np.std(nmse_mom_list),
    }

    return metrics


def process_exp_a_file(h5_path, output_csv, num_workers):
    """Process one Experiment A H5 file → CSV."""
    print(f"Processing: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        all_groups = []
        for mode in ['unguided', 'guided']:
            if mode in f:
                for length_key in sorted(f[mode].keys(), key=lambda x: int(x[1:])):
                    all_groups.append(f'{mode}/{length_key}')

    lengths = sorted(set(
        int(g.split('/')[1][1:]) for g in all_groups
    ))

    results = []
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    for traj_length in lengths:
        row = {'trajectory_length': traj_length}

        for mode in ['unguided', 'guided']:
            group_name = f'{mode}/L{traj_length}'
            if group_name not in all_groups:
                continue

            start = time.time()
            metrics = compute_metrics_for_group(h5_path, group_name, num_workers)
            elapsed = time.time() - start

            prefix = f'{mode}_'
            for key, val in metrics.items():
                row[f'{prefix}{key}'] = val

            print(f"  L{traj_length} {mode}: "
                  f"NMSE_q={metrics['nmse_q_mean']:.6f} "
                  f"NMSE_p={metrics['nmse_p_mean']:.6f} "
                  f"({elapsed:.1f}s)")

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compute MSE + NMSE for 2-DoF ablation trajectories")
    parser.add_argument("--num_workers", type=int, default=24,
                        help="Number of parallel workers for MuJoCo reconstruction")
    parser.add_argument("--torque_conditions", type=str, default=None,
                        help="Comma-separated torque conditions to process "
                             "(e.g., 'sinusoidal,gp'). Default: all found.")
    args = parser.parse_args()

    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Workers: {args.num_workers}")

    total_start = time.time()

    # Process Experiment A files
    torque_filter = None
    if args.torque_conditions:
        torque_filter = set(t.strip() for t in args.torque_conditions.split(','))

    for h5_file in sorted(INPUT_DIR.glob('exp_a_*.h5')):
        torque_label = h5_file.stem.replace('exp_a_', '')
        if torque_filter and torque_label not in torque_filter:
            print(f"\nSkipping exp_a_{torque_label} (not in filter)")
            continue

        output_csv = OUTPUT_DIR / f'exp_a_{torque_label}.csv'
        process_exp_a_file(str(h5_file), str(output_csv), args.num_workers)

    total_elapsed = time.time() - total_start
    print(f"\nAll computation complete in {total_elapsed / 3600:.1f} hours")


if __name__ == "__main__":
    main()
