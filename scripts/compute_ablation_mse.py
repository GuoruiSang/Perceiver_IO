"""
Compute MSE for saved ablation trajectories.

Loads generated trajectories from H5, runs MuJoCo physics reconstruction,
and computes MSE metrics. Uses multiprocessing for CPU parallelism.

Usage:
    python scripts/compute_ablation_mse.py \
        --input_dir output_ablation/trajectories \
        --output_dir output_ablation/results \
        --num_workers 24
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

from src.models.utils import compare_generated_with_reconstructed


# These must be module-level for multiprocessing pickling
XML_PATH = str(project_root / 'configs' / 'rigid_arm_hinge.xml')
PLOTS_PATH = str(project_root / 'plots')


def compute_single_mse(args_tuple):
    """Compute MSE for a single trajectory. Worker function for multiprocessing."""
    state_np, torque_np, qpos_dim, dt, data_dt = args_tuple

    generated = {
        'seq_qpos': state_np[:, :qpos_dim],
        'seq_mom': state_np[:, qpos_dim:],
        'seq_torque': torque_np,
    }
    mse_dict = compare_generated_with_reconstructed(
        generated, XML_PATH, PLOTS_PATH,
        dt=dt, data_dt=data_dt, name=None,
    )
    return mse_dict['mse_qpos'], mse_dict['mse_mom']


def compute_mse_for_group(h5_path, group_name, qpos_dim, dt, data_dt, num_workers):
    """Compute MSE for all trajectories in an H5 group."""
    with h5py.File(h5_path, 'r') as f:
        grp = f[group_name]
        states = grp['state'][:]      # [N, T, state_dim]
        torques = grp['torque'][:]    # [N, T, torque_dim]

    num_samples = states.shape[0]

    # Prepare arguments for parallel processing
    args_list = [
        (states[i], torques[i], qpos_dim, dt, data_dt)
        for i in range(num_samples)
    ]

    mse_qpos_list = []
    mse_mom_list = []

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(compute_single_mse, args): i
                       for i, args in enumerate(args_list)}
            for future in as_completed(futures):
                mse_q, mse_m = future.result()
                mse_qpos_list.append(mse_q)
                mse_mom_list.append(mse_m)
    else:
        for args in args_list:
            mse_q, mse_m = compute_single_mse(args)
            mse_qpos_list.append(mse_q)
            mse_mom_list.append(mse_m)

    return {
        'mse_qpos': np.mean(mse_qpos_list),
        'mse_mom': np.mean(mse_mom_list),
        'mse_total': np.mean(mse_qpos_list) + np.mean(mse_mom_list),
        'mse_qpos_std': np.std(mse_qpos_list),
        'mse_mom_std': np.std(mse_mom_list),
    }


def process_exp_a_file(h5_path, output_csv, qpos_dim, dt, data_dt, num_workers):
    """Process one Experiment A H5 file → CSV."""
    print(f"Processing: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        # Get all groups: unguided/L50, unguided/L100, ..., guided/L50, ...
        all_groups = []
        for mode in ['unguided', 'guided']:
            if mode in f:
                for length_key in sorted(f[mode].keys(), key=lambda x: int(x[1:])):
                    all_groups.append(f'{mode}/{length_key}')

    # Extract trajectory lengths
    lengths = sorted(set(
        int(g.split('/')[1][1:]) for g in all_groups
    ))

    results = []
    for traj_length in lengths:
        row = {'trajectory_length': traj_length}

        for mode in ['unguided', 'guided']:
            group_name = f'{mode}/L{traj_length}'
            if group_name not in all_groups:
                continue

            start = time.time()
            mse = compute_mse_for_group(h5_path, group_name, qpos_dim, dt, data_dt, num_workers)
            elapsed = time.time() - start

            prefix = f'{mode}_'
            row[f'{prefix}mse_qpos'] = mse['mse_qpos']
            row[f'{prefix}mse_mom'] = mse['mse_mom']
            row[f'{prefix}mse_total'] = mse['mse_total']
            row[f'{prefix}mse_qpos_std'] = mse['mse_qpos_std']
            row[f'{prefix}mse_mom_std'] = mse['mse_mom_std']

            print(f"  L{traj_length} {mode}: MSE={mse['mse_total']:.6f} ({elapsed:.1f}s)")

        results.append(row)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")
    return df


def process_exp_b_file(h5_path, output_csv, qpos_dim, dt, data_dt, num_workers):
    """Process Experiment B H5 file → CSV."""
    print(f"Processing: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        all_groups = []
        for mode in ['unguided', 'guided']:
            if mode in f:
                for cf_key in sorted(f[mode].keys(), key=lambda x: float(x[3:])):
                    all_groups.append(f'{mode}/{cf_key}')

    fractions = sorted(set(
        float(g.split('/')[1][3:]) for g in all_groups
    ))

    results = []
    for ctx_frac in fractions:
        row = {'context_fraction': ctx_frac}

        for mode in ['unguided', 'guided']:
            group_name = f'{mode}/cf_{ctx_frac:.2f}'
            if group_name not in all_groups:
                continue

            start = time.time()
            mse = compute_mse_for_group(h5_path, group_name, qpos_dim, dt, data_dt, num_workers)
            elapsed = time.time() - start

            prefix = f'{mode}_'
            row[f'{prefix}mse_qpos'] = mse['mse_qpos']
            row[f'{prefix}mse_mom'] = mse['mse_mom']
            row[f'{prefix}mse_total'] = mse['mse_total']
            row[f'{prefix}mse_qpos_std'] = mse['mse_qpos_std']
            row[f'{prefix}mse_mom_std'] = mse['mse_mom_std']

            print(f"  cf={ctx_frac:.2f} {mode}: MSE={mse['mse_total']:.6f} ({elapsed:.1f}s)")

        results.append(row)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Compute MSE for ablation trajectories")
    parser.add_argument("--input_dir", type=str, default="output_ablation/trajectories")
    parser.add_argument("--output_dir", type=str, default="output_ablation/results")
    parser.add_argument("--num_workers", type=int, default=24,
                        help="Number of parallel workers for MuJoCo reconstruction")
    parser.add_argument("--qpos_dim", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.0001)
    parser.add_argument("--data_dt", type=float, default=0.00025)
    parser.add_argument("--model_names", type=str, nargs="+",
                        default=["original", "global_cond", "torque_concat"])
    args = parser.parse_args()

    input_root = project_root / args.input_dir
    output_root = project_root / args.output_dir

    print(f"Input: {input_root}")
    print(f"Output: {output_root}")
    print(f"Workers: {args.num_workers}")

    total_start = time.time()

    for model_name in args.model_names:
        model_dir = input_root / model_name
        if not model_dir.exists():
            print(f"\nSkipping {model_name} (directory not found)")
            continue

        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*60}")

        # Process Experiment A files
        for h5_file in sorted(model_dir.glob('exp_a_*.h5')):
            torque_label = h5_file.stem.replace('exp_a_', '')
            output_csv = output_root / model_name / f'exp_a_{torque_label}.csv'
            process_exp_a_file(
                str(h5_file), str(output_csv),
                args.qpos_dim, args.dt, args.data_dt, args.num_workers,
            )

        # Process Experiment B file
        exp_b_file = model_dir / 'exp_b_context_fractions.h5'
        if exp_b_file.exists():
            output_csv = output_root / model_name / 'exp_b_context_fractions.csv'
            process_exp_b_file(
                str(exp_b_file), str(output_csv),
                args.qpos_dim, args.dt, args.data_dt, args.num_workers,
            )

    total_elapsed = time.time() - total_start
    print(f"\nAll MSE computation complete in {total_elapsed / 3600:.1f} hours")


if __name__ == "__main__":
    main()
