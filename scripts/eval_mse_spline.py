"""
Evaluate MSE (position, momentum, Hamiltonian energy) on the validation dataset
using ONLY cubic-spline torque trajectories.

Compares the original DPF model before and after HNN guidance.

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/eval_mse_spline.py \
        --num_workers 48 --num_samples 200 --seed 42
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

from src.models.trajectory_dpf import TrajectoryDPF, EMA
from src.models.utils import reconstruct_traj_with_momentum
from src.models.HNN import HNNWrapper

import mujoco

XML_PATH = str(project_root / 'configs' / 'rigid_arm_hinge.xml')

# Per-worker caches
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


def compute_single_mse_all(args_tuple):
    """Single-pass worker: MuJoCo recon -> MSE for qpos, mom, energy.

    Returns (group_key, sample_idx, mse_qpos, mse_mom, mse_energy).
    """
    group_key, sample_idx, state_np, torque_np, qpos_dim, dt, data_dt, hnn_path, hnn_device = args_tuple

    qpos_gen = state_np[:, :qpos_dim]
    mom_gen = state_np[:, qpos_dim:]

    # --- MuJoCo reconstruction ---
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
    qpos_r = recon['seq_qpos']  # (T-1, qpos_dim)
    mom_r = recon['seq_mom']    # (T-1, mom_dim)

    # --- MSE for qpos and momentum (skip first timestep of generated) ---
    mse_qpos = np.mean((qpos_gen[1:] - qpos_r) ** 2)
    mse_mom = np.mean((mom_gen[1:] - mom_r) ** 2)

    # --- Energy MSE via HNN ---
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

    mse_energy = np.mean((H_gen - H_recon) ** 2)

    return group_key, sample_idx, mse_qpos, mse_mom, mse_energy


def aggregate_group_results(results_dict):
    """Aggregate per-sample MSE results into group metrics."""
    mse_qpos_arr = np.array([r[0] for r in results_dict.values()])
    mse_mom_arr = np.array([r[1] for r in results_dict.values()])
    mse_energy_arr = np.array([r[2] for r in results_dict.values()])

    return {
        'mse_qpos_mean': np.mean(mse_qpos_arr),
        'mse_qpos_std': np.std(mse_qpos_arr),
        'mse_mom_mean': np.mean(mse_mom_arr),
        'mse_mom_std': np.std(mse_mom_arr),
        'mse_energy_mean': np.mean(mse_energy_arr),
        'mse_energy_std': np.std(mse_energy_arr),
    }


def load_validation_torques_by_policy(val_path, num_samples, torque_policy='spline'):
    """Load torque sequences from validation HDF5, filtered by torque policy."""
    torques = []
    with h5py.File(val_path, 'r') as f:
        num_available = f.attrs['num_trajectories']
        for i in range(num_available):
            grp = f[f'traj_{i}']
            policy = grp.attrs.get('torque_policy', b'sinusoidal')
            if isinstance(policy, bytes):
                policy = policy.decode()
            if policy != torque_policy:
                continue
            torques.append(grp['seq_torque'][:])
            if len(torques) >= num_samples:
                break
    return np.stack(torques, axis=0)


def load_dpf_with_ema(dpf_path, device):
    """Load DPF model and restore EMA shadow parameters."""
    model = TrajectoryDPF.load_from_checkpoint(dpf_path, map_location=device, strict=False)
    model = model.to(device)

    checkpoint = torch.load(dpf_path, map_location=device, weights_only=False)
    ema_shadow = checkpoint.get('ema_shadow', None)
    ema_decay = checkpoint.get('ema_decay', 0.9995)
    if ema_shadow is not None:
        model.ema = EMA(model.model, decay=ema_decay)
        for name, tensor in ema_shadow.items():
            if name in model.ema.shadow:
                model.ema.shadow[name] = tensor.to(device=device, dtype=model.ema.shadow[name].dtype)
        print(f"  EMA loaded ({len(ema_shadow)} params)")
    del checkpoint

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MSE on spline-torque validation trajectories")
    parser.add_argument("--dpf_checkpoint", type=str,
                        default="checkpoints/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt")
    parser.add_argument("--hnn_checkpoint", type=str,
                        default="checkpoints/SeperableHNN(dim1024)-CELU-epoch-epoch=849.ckpt")
    parser.add_argument("--val_file", type=str, default="data/traj_2000-steps_4000.h5")
    parser.add_argument("--output_csv", type=str,
                        default="output_ablation/results/latest/mse_spline_torques.csv")
    parser.add_argument("--output_table", type=str,
                        default="output_ablation/tables/mse_spline_torques.md")
    parser.add_argument("--torque_policy", type=str, default="spline",
                        help="Torque policy to filter for")
    parser.add_argument("--num_workers", type=int, default=48)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--context_fraction", type=float, default=0.2)
    parser.add_argument("--guidance_method", type=str, default="adam")
    parser.add_argument("--guidance_steps", type=int, default=3)
    parser.add_argument("--guidance_lr", type=float, default=0.01)
    parser.add_argument("--guidance_after_steps", type=int, default=45)
    parser.add_argument("--lambda_init", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--sampler", type=str, default="ddim")
    parser.add_argument("--hnn_rl2_device", type=str, default="cpu",
                        help="Device for HNN in MSE workers (cpu recommended for parallel)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dpf_path = str(project_root / args.dpf_checkpoint) if not os.path.isabs(args.dpf_checkpoint) else args.dpf_checkpoint
    hnn_path = str(project_root / args.hnn_checkpoint) if not os.path.isabs(args.hnn_checkpoint) else args.hnn_checkpoint
    val_path = str(project_root / args.val_file) if not os.path.isabs(args.val_file) else args.val_file
    output_csv = str(project_root / args.output_csv) if not os.path.isabs(args.output_csv) else args.output_csv
    output_table = str(project_root / args.output_table) if not os.path.isabs(args.output_table) else args.output_table

    print("=" * 60)
    print("  MSE Evaluation — Spline Torques (Validation Dataset)")
    print("=" * 60)
    print(f"DPF: {dpf_path}")
    print(f"HNN: {hnn_path}")
    print(f"Validation: {val_path}")
    print(f"Torque policy filter: {args.torque_policy}")
    print(f"Samples: {args.num_samples}, Workers: {args.num_workers}")
    print(f"Diffusion: steps={args.num_diffusion_steps}, sampler={args.sampler}")
    print(f"Context fraction: {args.context_fraction}")
    print(f"Guidance: method={args.guidance_method}, steps={args.guidance_steps}, "
          f"lr={args.guidance_lr}, after_steps={args.guidance_after_steps}, "
          f"lambda_init={args.lambda_init}")
    print()

    # Load DPF with EMA
    print("Loading DPF model...")
    model = load_dpf_with_ema(dpf_path, device)
    qpos_dim = model.qpos_dim
    dt = float(model.dt)
    data_dt = float(model.data_dt)
    print(f"  qpos_dim={qpos_dim}, dt={dt}, data_dt={data_dt}")

    # Load HNN (for guidance during generation)
    print("Loading HNN for guidance...")
    hnn = HNNWrapper.load_from_checkpoint(hnn_path, map_location=device)
    hnn = hnn.to(device)
    hnn.eval()
    print(f"  HNN loaded")

    # Load validation torques filtered by policy
    print(f"Loading validation torques (policy={args.torque_policy})...")
    all_torques = load_validation_torques_by_policy(val_path, args.num_samples,
                                                     torque_policy=args.torque_policy)
    actual_samples = all_torques.shape[0]
    print(f"  Loaded: {all_torques.shape} ({actual_samples} {args.torque_policy} trajectories)")
    if actual_samples < args.num_samples:
        print(f"  NOTE: Only {actual_samples} available (requested {args.num_samples})")

    trajectory_lengths = list(range(50, 1501, 50))
    results = []
    total_start = time.time()

    hnn_rl2_path = hnn_path

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for traj_len in trajectory_lengths:
            len_start = time.time()
            print(f"\n{'='*60}")
            print(f"  Length {traj_len}")
            print(f"{'='*60}")

            # Truncate torques
            torque_truncated = all_torques[:, :traj_len, :]
            torque_tensor = torch.tensor(torque_truncated, dtype=torch.float32, device=device)

            # --- Generate UNGUIDED ---
            print(f"  [Unguided] Generating {actual_samples} samples...")
            all_states_base = []
            all_torques_base = []
            for batch_start in range(0, actual_samples, args.batch_size):
                batch_end = min(batch_start + args.batch_size, actual_samples)
                batch_torque = torque_tensor[batch_start:batch_end]
                state, torque_out = model.sample_trajectories(
                    num_samples=batch_torque.shape[0],
                    trajectory_length=traj_len,
                    num_diffusion_steps=args.num_diffusion_steps,
                    context_fraction=args.context_fraction,
                    use_ema=True,
                    sampler=args.sampler,
                    guidance_scale=args.guidance_scale,
                    hnn=None,
                    guidance_steps=0,
                    torque=batch_torque,
                )
                all_states_base.append(state.cpu())
                all_torques_base.append(torque_out.cpu())
            states_base = torch.cat(all_states_base, dim=0).numpy()
            torques_base = torch.cat(all_torques_base, dim=0).numpy()

            # --- Generate GUIDED ---
            print(f"  [Guided] Generating {actual_samples} samples with HNN guidance...")
            all_states_guid = []
            all_torques_guid = []
            for batch_start in range(0, actual_samples, args.batch_size):
                batch_end = min(batch_start + args.batch_size, actual_samples)
                batch_torque = torque_tensor[batch_start:batch_end]
                state, torque_out = model.sample_trajectories(
                    num_samples=batch_torque.shape[0],
                    trajectory_length=traj_len,
                    num_diffusion_steps=args.num_diffusion_steps,
                    context_fraction=args.context_fraction,
                    use_ema=True,
                    sampler=args.sampler,
                    guidance_scale=args.guidance_scale,
                    hnn=hnn,
                    guidance_method=args.guidance_method,
                    guidance_after_steps=args.guidance_after_steps,
                    guidance_steps=args.guidance_steps,
                    guidance_lr=args.guidance_lr,
                    lambda_init=args.lambda_init,
                    torque=batch_torque,
                )
                all_states_guid.append(state.cpu())
                all_torques_guid.append(torque_out.cpu())
            states_guid = torch.cat(all_states_guid, dim=0).numpy()
            torques_guid = torch.cat(all_torques_guid, dim=0).numpy()

            # --- Submit MSE computation tasks ---
            print(f"  Computing MSE ({actual_samples * 2} tasks)...")
            all_futures = {}
            group_results = {'unguided': {}, 'guided': {}}

            for i in range(actual_samples):
                args_u = ('unguided', i, states_base[i], torques_base[i],
                          qpos_dim, dt, data_dt, hnn_rl2_path, args.hnn_rl2_device)
                fut = executor.submit(compute_single_mse_all, args_u)
                all_futures[fut] = ('unguided', i)

                args_g = ('guided', i, states_guid[i], torques_guid[i],
                          qpos_dim, dt, data_dt, hnn_rl2_path, args.hnn_rl2_device)
                fut = executor.submit(compute_single_mse_all, args_g)
                all_futures[fut] = ('guided', i)

            # Collect results
            done_count = 0
            for future in as_completed(all_futures):
                gk, si, mq, mm, me = future.result()
                group_results[gk][si] = (mq, mm, me)
                done_count += 1
                if done_count % 100 == 0:
                    print(f"    {done_count}/{len(all_futures)} tasks done...", flush=True)

            # Aggregate
            row = {'trajectory_length': traj_len}
            for mode in ['unguided', 'guided']:
                metrics = aggregate_group_results(group_results[mode])
                for key, val in metrics.items():
                    row[f'{mode}_{key}'] = val
                print(f"  L{traj_len} {mode}: "
                      f"MSE_qpos={metrics['mse_qpos_mean']:.6f} "
                      f"MSE_mom={metrics['mse_mom_mean']:.6f} "
                      f"MSE_energy={metrics['mse_energy_mean']:.6f}")
            results.append(row)

            elapsed = time.time() - len_start
            print(f"  ({elapsed:.1f}s)", flush=True)

    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved CSV: {output_csv}")

    # Generate markdown table
    os.makedirs(os.path.dirname(output_table), exist_ok=True)
    generate_markdown_table(df, output_table)
    print(f"Saved table: {output_table}")

    total_elapsed = time.time() - total_start
    print(f"\nAll done in {total_elapsed / 60:.1f} minutes")


def generate_markdown_table(df, output_path):
    """Generate a markdown table from the results DataFrame."""
    lines = []
    lines.append("# MSE Evaluation — Original DPF on Spline Torques (Validation Dataset)")
    lines.append("")
    lines.append("| Length | Unguided MSE Pos | Unguided MSE Mom | Unguided MSE Energy | Guided MSE Pos | Guided MSE Mom | Guided MSE Energy |")
    lines.append("|--------|-----------------|-----------------|---------------------|----------------|----------------|-------------------|")

    for _, row in df.iterrows():
        tl = int(row['trajectory_length'])
        vals = []
        for mode in ['unguided', 'guided']:
            for metric in ['mse_qpos', 'mse_mom', 'mse_energy']:
                mean = row[f'{mode}_{metric}_mean']
                std = row[f'{mode}_{metric}_std']
                vals.append(f"{mean:.6f} ± {std:.6f}")
        lines.append(f"| {tl} | " + " | ".join(vals) + " |")

    lines.append("")
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
