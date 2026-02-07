"""
Compute MSE and Hamiltonian energy metrics for saved ablation trajectories.

Loads generated trajectories from H5, runs MuJoCo physics reconstruction,
computes MSE metrics, and compares Hamiltonian energy H(t) between generated
and reconstructed trajectories using the pre-trained HNN.

Uses multiprocessing for CPU parallelism on MuJoCo reconstruction.

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
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.models.utils import reconstruct_traj_with_momentum, central_difference
from src.models.HNN import HNNWrapper

import mujoco


# Module-level constants for multiprocessing pickling
XML_PATH = str(project_root / 'configs' / 'rigid_arm_hinge.xml')


def compute_single_mse(args_tuple):
    """Compute MSE for a single trajectory and return reconstructed trajectory.

    Returns:
        (mse_qpos, mse_mom, mse_total, qpos_recon, mom_recon)
        where qpos_recon/mom_recon are [T, dim] arrays (full length, including
        initial state prepended).
    """
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
    mse_total = mse_qpos + mse_mom

    # Prepend initial state to reconstructed trajectory for energy comparison
    qpos_recon = np.concatenate([qpos_gen[:1], recon['seq_qpos']], axis=0)
    mom_recon = np.concatenate([mom_gen[:1], recon['seq_mom']], axis=0)

    return mse_qpos, mse_mom, mse_total, qpos_recon, mom_recon


def compute_hnn_energy_batch(hnn, qpos, mom, device='cpu', batch_size=10000):
    """Compute H(t) for a batch of trajectories using HNN.

    Args:
        hnn: HNNWrapper model
        qpos: [N, T, qpos_dim] numpy array
        mom: [N, T, mom_dim] numpy array
        device: torch device
        batch_size: max points per forward pass

    Returns:
        H: [N, T] numpy array of Hamiltonian values
    """
    N, T, _ = qpos.shape
    total = N * T

    q_flat = torch.tensor(qpos.reshape(total, -1), dtype=torch.float32, device=device)
    p_flat = torch.tensor(mom.reshape(total, -1), dtype=torch.float32, device=device)

    H_list = []
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            H_batch = hnn(p_flat[start:end], q_flat[start:end])  # [batch, 1]
            H_list.append(H_batch.cpu().numpy().squeeze(-1))

    H_flat = np.concatenate(H_list)
    return H_flat.reshape(N, T)


def compute_physics_energy_per_sample(states, torques, qpos_dim, data_dt,
                                      hnn, device, batch_size=100):
    """Compute per-sample physics consistency energy (e1 + e2) from generated trajectories.

    Measures how well the generated trajectory satisfies Hamilton's equations:
        e1 = MSE(dq/dt, dH/dp)
        e2 = MSE(dp/dt, -dH/dq + torque)

    Args:
        states: [N, T, state_dim] numpy array (qpos | mom)
        torques: [N, T, torque_dim] numpy array
        qpos_dim: int
        data_dt: float, timestep between data points
        hnn: HNNWrapper model
        device: torch device string
        batch_size: samples per batch (controls GPU memory)

    Returns:
        energies: [N] numpy array of per-sample physics energy values
    """
    N, T, state_dim = states.shape
    mom_dim = state_dim - qpos_dim

    all_energies = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        seq_qpos = torch.tensor(
            states[start:end, :, :qpos_dim], dtype=torch.float32, device=device)
        seq_mom = torch.tensor(
            states[start:end, :, qpos_dim:], dtype=torch.float32, device=device)
        seq_torque = torch.tensor(
            torques[start:end], dtype=torch.float32, device=device)

        # Time derivatives via central difference
        dot_qpos = central_difference(seq_qpos, data_dt)  # [B, T, qpos_dim]
        dot_mom = central_difference(seq_mom, data_dt)     # [B, T, mom_dim]

        # HNN predictions with autograd
        B, T_len, _ = seq_qpos.shape
        q_flat = seq_qpos.reshape(-1, qpos_dim).detach().requires_grad_(True)
        p_flat = seq_mom.reshape(-1, mom_dim).detach().requires_grad_(True)

        H = hnn(p_flat, q_flat)  # [B*T, 1]
        dH_dp, dH_dq = torch.autograd.grad(H.sum(), (p_flat, q_flat))

        dH_dp = dH_dp.reshape(B, T_len, mom_dim)
        dH_dq = dH_dq.reshape(B, T_len, qpos_dim)

        dot_qpos_pred = dH_dp
        dot_mom_pred = -dH_dq + seq_torque

        # Per-sample MSE over time and dimensions
        e1 = torch.mean((dot_qpos - dot_qpos_pred) ** 2, dim=(1, 2))  # [B]
        e2 = torch.mean((dot_mom - dot_mom_pred) ** 2, dim=(1, 2))    # [B]

        all_energies.append((e1 + e2).detach().cpu().numpy())

    return np.concatenate(all_energies)


def compute_metrics_for_group(h5_path, group_name, qpos_dim, dt, data_dt,
                              num_workers, hnn, hnn_device):
    """Compute MSE and energy metrics for all trajectories in an H5 group.

    Returns:
        metrics: dict with mean/std for mse_qpos, mse_mom, mse_total, mse_energy
        H_gen: [N, T] energy profiles for generated trajectories
        H_recon: [N, T] energy profiles for reconstructed trajectories
    """
    with h5py.File(h5_path, 'r') as f:
        grp = f[group_name]
        states = grp['state'][:]      # [N, T, state_dim]
        torques = grp['torque'][:]    # [N, T, torque_dim]

    num_samples = states.shape[0]

    # Prepare arguments for parallel MuJoCo reconstruction
    args_list = [
        (states[i], torques[i], qpos_dim, dt, data_dt)
        for i in range(num_samples)
    ]

    mse_qpos_list = []
    mse_mom_list = []
    mse_total_list = []
    recon_qpos_all = []
    recon_mom_all = []

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(compute_single_mse, args): i
                       for i, args in enumerate(args_list)}
            # Collect results preserving order
            results_by_idx = {}
            for future in as_completed(futures):
                idx = futures[future]
                mse_q, mse_m, mse_t, qpos_r, mom_r = future.result()
                results_by_idx[idx] = (mse_q, mse_m, mse_t, qpos_r, mom_r)

            for i in range(num_samples):
                mse_q, mse_m, mse_t, qpos_r, mom_r = results_by_idx[i]
                mse_qpos_list.append(mse_q)
                mse_mom_list.append(mse_m)
                mse_total_list.append(mse_t)
                recon_qpos_all.append(qpos_r)
                recon_mom_all.append(mom_r)
    else:
        for args in args_list:
            mse_q, mse_m, mse_t, qpos_r, mom_r = compute_single_mse(args)
            mse_qpos_list.append(mse_q)
            mse_mom_list.append(mse_m)
            mse_total_list.append(mse_t)
            recon_qpos_all.append(qpos_r)
            recon_mom_all.append(mom_r)

    mse_qpos_arr = np.array(mse_qpos_list)
    mse_mom_arr = np.array(mse_mom_list)
    mse_total_arr = np.array(mse_total_list)

    # Compute HNN energy for generated and reconstructed trajectories
    gen_qpos = states[:, :, :qpos_dim]                # [N, T, qpos_dim]
    gen_mom = states[:, :, qpos_dim:]                  # [N, T, mom_dim]
    recon_qpos = np.stack(recon_qpos_all, axis=0)      # [N, T, qpos_dim]
    recon_mom = np.stack(recon_mom_all, axis=0)         # [N, T, mom_dim]

    H_gen = compute_hnn_energy_batch(hnn, gen_qpos, gen_mom, device=hnn_device)
    H_recon = compute_hnn_energy_batch(hnn, recon_qpos, recon_mom, device=hnn_device)

    # Energy MSE: per-sample mean over time of (H_gen - H_recon)^2
    mse_energy_per_sample = np.mean((H_gen - H_recon) ** 2, axis=1)  # [N]

    # Physics consistency energy (final energy from sampling)
    final_energy_per_sample = compute_physics_energy_per_sample(
        states, torques, qpos_dim, data_dt, hnn, hnn_device)

    metrics = {
        'mse_qpos_mean': np.mean(mse_qpos_arr),
        'mse_qpos_std': np.std(mse_qpos_arr),
        'mse_mom_mean': np.mean(mse_mom_arr),
        'mse_mom_std': np.std(mse_mom_arr),
        'mse_total_mean': np.mean(mse_total_arr),
        'mse_total_std': np.std(mse_total_arr),
        'mse_energy_mean': np.mean(mse_energy_per_sample),
        'mse_energy_std': np.std(mse_energy_per_sample),
        'final_energy_mean': np.mean(final_energy_per_sample),
        'final_energy_std': np.std(final_energy_per_sample),
    }

    return metrics, H_gen, H_recon


def process_exp_a_file(h5_path, output_csv, energy_h5_path,
                       qpos_dim, dt, data_dt, num_workers, hnn, hnn_device):
    """Process one Experiment A H5 file → CSV + energy H5."""
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

    with h5py.File(energy_h5_path, 'w') as energy_h5:
        for traj_length in lengths:
            row = {'trajectory_length': traj_length}

            for mode in ['unguided', 'guided']:
                group_name = f'{mode}/L{traj_length}'
                if group_name not in all_groups:
                    continue

                start = time.time()
                metrics, H_gen, H_recon = compute_metrics_for_group(
                    h5_path, group_name, qpos_dim, dt, data_dt,
                    num_workers, hnn, hnn_device,
                )
                elapsed = time.time() - start

                prefix = f'{mode}_'
                for key, val in metrics.items():
                    row[f'{prefix}{key}'] = val

                # Save energy profiles
                eg = energy_h5.create_group(group_name)
                eg.create_dataset('H_gen', data=H_gen, dtype='f4')
                eg.create_dataset('H_recon', data=H_recon, dtype='f4')
                energy_h5.flush()

                print(f"  L{traj_length} {mode}: "
                      f"MSE_total={metrics['mse_total_mean']:.6f} "
                      f"MSE_energy={metrics['mse_energy_mean']:.6f} "
                      f"final_energy={metrics['final_energy_mean']:.6f} "
                      f"({elapsed:.1f}s)")

            results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")
    print(f"  Saved: {energy_h5_path}")
    return df


def process_exp_a_final_energy_only(h5_path, existing_csv, qpos_dim, data_dt,
                                     hnn, hnn_device):
    """Compute only physics energy and merge into existing CSV (no MuJoCo)."""
    print(f"Processing (final energy only): {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        all_groups = []
        for mode in ['unguided', 'guided']:
            if mode in f:
                for length_key in sorted(f[mode].keys(), key=lambda x: int(x[1:])):
                    all_groups.append(f'{mode}/{length_key}')

    lengths = sorted(set(
        int(g.split('/')[1][1:]) for g in all_groups
    ))

    # Load existing CSV to merge into
    if os.path.exists(existing_csv):
        df = pd.read_csv(existing_csv)
    else:
        df = pd.DataFrame({'trajectory_length': lengths})

    # Compute physics energy for each length/mode
    for traj_length in lengths:
        row_mask = df['trajectory_length'] == traj_length

        for mode in ['unguided', 'guided']:
            group_name = f'{mode}/L{traj_length}'
            if group_name not in all_groups:
                continue

            with h5py.File(h5_path, 'r') as f:
                grp = f[group_name]
                states = grp['state'][:]    # [N, T, state_dim]
                torques = grp['torque'][:]  # [N, T, torque_dim]

            start = time.time()
            fe = compute_physics_energy_per_sample(
                states, torques, qpos_dim, data_dt, hnn, hnn_device)
            elapsed = time.time() - start

            prefix = f'{mode}_'
            df.loc[row_mask, f'{prefix}final_energy_mean'] = np.mean(fe)
            df.loc[row_mask, f'{prefix}final_energy_std'] = np.std(fe)

            print(f"  L{traj_length} {mode}: "
                  f"final_energy={np.mean(fe):.6f} ({elapsed:.1f}s)")

    df.to_csv(existing_csv, index=False)
    print(f"  Saved: {existing_csv}")
    return df


def process_exp_b_file(h5_path, output_csv, energy_h5_path,
                       qpos_dim, dt, data_dt, num_workers, hnn, hnn_device):
    """Process Experiment B H5 file → CSV + energy H5."""
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
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with h5py.File(energy_h5_path, 'w') as energy_h5:
        for ctx_frac in fractions:
            row = {'context_fraction': ctx_frac}

            for mode in ['unguided', 'guided']:
                group_name = f'{mode}/cf_{ctx_frac:.2f}'
                if group_name not in all_groups:
                    continue

                start = time.time()
                metrics, H_gen, H_recon = compute_metrics_for_group(
                    h5_path, group_name, qpos_dim, dt, data_dt,
                    num_workers, hnn, hnn_device,
                )
                elapsed = time.time() - start

                prefix = f'{mode}_'
                for key, val in metrics.items():
                    row[f'{prefix}{key}'] = val

                # Save energy profiles
                eg = energy_h5.create_group(group_name)
                eg.create_dataset('H_gen', data=H_gen, dtype='f4')
                eg.create_dataset('H_recon', data=H_recon, dtype='f4')
                energy_h5.flush()

                print(f"  cf={ctx_frac:.2f} {mode}: "
                      f"MSE_total={metrics['mse_total_mean']:.6f} "
                      f"MSE_energy={metrics['mse_energy_mean']:.6f} "
                      f"({elapsed:.1f}s)")

            results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")
    print(f"  Saved: {energy_h5_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compute MSE + Hamiltonian energy for ablation trajectories")
    parser.add_argument("--input_dir", type=str, default="output_ablation/trajectories")
    parser.add_argument("--output_dir", type=str, default="output_ablation/results")
    parser.add_argument("--num_workers", type=int, default=24,
                        help="Number of parallel workers for MuJoCo reconstruction")
    parser.add_argument("--qpos_dim", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.0001)
    parser.add_argument("--data_dt", type=float, default=0.0002)
    parser.add_argument("--model_names", type=str, nargs="+",
                        default=["original", "global_cond", "torque_concat"])
    parser.add_argument("--hnn_checkpoint", type=str,
                        default="checkpoints/SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt")
    parser.add_argument("--hnn_device", type=str, default="cpu",
                        help="Device for HNN energy computation (cpu or cuda:X)")
    parser.add_argument("--torque_conditions", type=str, default=None,
                        help="Comma-separated torque conditions to process for Exp A "
                             "(e.g., 'training,sinusoidal,gp'). Default: all found.")
    parser.add_argument("--skip_exp_b", action="store_true",
                        help="Skip Experiment B processing")
    parser.add_argument("--final_energy_only", action="store_true",
                        help="Only compute physics energy (no MuJoCo reconstruction). "
                             "Merges final_energy columns into existing CSVs.")
    args = parser.parse_args()

    input_root = project_root / args.input_dir
    output_root = project_root / args.output_dir

    print(f"Input: {input_root}")
    print(f"Output: {output_root}")
    print(f"Workers: {args.num_workers}")

    # Load HNN for energy computation
    hnn_path = str(project_root / args.hnn_checkpoint)
    print(f"Loading HNN from: {hnn_path}")
    hnn = HNNWrapper.load_from_checkpoint(hnn_path, map_location=args.hnn_device)
    hnn = hnn.to(args.hnn_device)
    hnn.eval()
    print(f"  HNN loaded on {args.hnn_device}")

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
        torque_filter = None
        if args.torque_conditions:
            torque_filter = set(t.strip() for t in args.torque_conditions.split(','))

        for h5_file in sorted(model_dir.glob('exp_a_*.h5')):
            torque_label = h5_file.stem.replace('exp_a_', '')
            if torque_filter and torque_label not in torque_filter:
                print(f"\n  Skipping exp_a_{torque_label} (not in filter)")
                continue
            output_csv = output_root / model_name / f'exp_a_{torque_label}.csv'

            if args.final_energy_only:
                process_exp_a_final_energy_only(
                    str(h5_file), str(output_csv),
                    args.qpos_dim, args.data_dt, hnn, args.hnn_device,
                )
            else:
                energy_h5 = output_root / model_name / f'exp_a_{torque_label}_energy.h5'
                process_exp_a_file(
                    str(h5_file), str(output_csv), str(energy_h5),
                    args.qpos_dim, args.dt, args.data_dt, args.num_workers,
                    hnn, args.hnn_device,
                )

        # Process Experiment B file
        exp_b_file = model_dir / 'exp_b_context_fractions.h5'
        if args.skip_exp_b:
            print(f"\n  Skipping Experiment B (--skip_exp_b)")
        elif exp_b_file.exists():
            output_csv = output_root / model_name / 'exp_b_context_fractions.csv'
            energy_h5 = output_root / model_name / 'exp_b_context_fractions_energy.h5'
            process_exp_b_file(
                str(exp_b_file), str(output_csv), str(energy_h5),
                args.qpos_dim, args.dt, args.data_dt, args.num_workers,
                hnn, args.hnn_device,
            )

    total_elapsed = time.time() - total_start
    print(f"\nAll computation complete in {total_elapsed / 3600:.1f} hours")


if __name__ == "__main__":
    main()
