"""
Quick DPF evaluation: generate trajectories at various lengths using validation torques.
Compares unguided (baseline) vs HNN-guided results.

Produces:
  - MSE table (position, momentum, Hamiltonian energy)
  - Per-length 5-row comparison figures:
      Row 1: Torque (3 dims)
      Row 2: Unguided state (qpos+mom) vs MuJoCo
      Row 3: Guided state (qpos+mom) vs MuJoCo
      Row 4: Unguided H(t) vs MuJoCo H(t)
      Row 5: Guided H(t) vs MuJoCo H(t)

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/quick_eval_dpf.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
import time
import h5py
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from concurrent.futures import ProcessPoolExecutor, as_completed

import mujoco

from src.models.trajectory_dpf import TrajectoryDPF
from src.models.utils import reconstruct_traj_with_momentum
from src.models.HNN import HNNWrapper
from src.models.trajectory_dpf import EMA

# Module-level for multiprocessing pickling
XML_PATH = str(project_root / 'configs' / 'rigid_arm_hinge.xml')


def compute_single_mse(args_tuple):
    """Compute MSE for a single trajectory. Returns (mse_qpos, mse_mom, qpos_recon, mom_recon)."""
    state_np, torque_np, qpos_dim, dt, data_dt = args_tuple

    qpos_gen = state_np[:, :qpos_dim]
    mom_gen = state_np[:, qpos_dim:]

    model = mujoco.MjModel.from_xml_path(XML_PATH)
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

    mse_qpos = np.mean((qpos_gen[1:] - recon['seq_qpos']) ** 2)
    mse_mom = np.mean((mom_gen[1:] - recon['seq_mom']) ** 2)

    qpos_recon = np.concatenate([qpos_gen[:1], recon['seq_qpos']], axis=0)
    mom_recon = np.concatenate([mom_gen[:1], recon['seq_mom']], axis=0)

    return mse_qpos, mse_mom, qpos_recon, mom_recon


def compute_hnn_energy_batch(hnn, qpos, mom, device='cpu', batch_size=10000):
    """Compute H(q, p) for trajectories. Returns [N, T] array."""
    N, T, _ = qpos.shape
    total = N * T

    q_flat = torch.tensor(qpos.reshape(total, -1), dtype=torch.float32, device=device)
    p_flat = torch.tensor(mom.reshape(total, -1), dtype=torch.float32, device=device)

    H_list = []
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            H_batch = hnn(p_flat[start:end], q_flat[start:end])
            H_list.append(H_batch.cpu().numpy().squeeze(-1))

    return np.concatenate(H_list).reshape(N, T)


def compute_mse_for_batch(state_np, torque_np, qpos_dim, dt, data_dt, hnn, device, num_workers):
    """Compute all MSE metrics for a batch. Returns dict + per-sample recon arrays."""
    num_samples = state_np.shape[0]
    task_args = [
        (state_np[i], torque_np[i], qpos_dim, dt, data_dt)
        for i in range(num_samples)
    ]

    mse_qpos_list, mse_mom_list = [], []
    recon_qpos_all, recon_mom_all = [], []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(compute_single_mse, a): i for i, a in enumerate(task_args)}
        results_by_idx = {}
        for future in as_completed(futures):
            idx = futures[future]
            results_by_idx[idx] = future.result()

        for i in range(num_samples):
            mse_q, mse_m, qr, mr = results_by_idx[i]
            mse_qpos_list.append(mse_q)
            mse_mom_list.append(mse_m)
            recon_qpos_all.append(qr)
            recon_mom_all.append(mr)

    mse_qpos_arr = np.array(mse_qpos_list)
    mse_mom_arr = np.array(mse_mom_list)

    gen_qpos = state_np[:, :, :qpos_dim]
    gen_mom = state_np[:, :, qpos_dim:]
    recon_qpos = np.stack(recon_qpos_all, axis=0)
    recon_mom = np.stack(recon_mom_all, axis=0)

    H_gen = compute_hnn_energy_batch(hnn, gen_qpos, gen_mom, device=device)
    H_recon = compute_hnn_energy_batch(hnn, recon_qpos, recon_mom, device=device)
    mse_energy_arr = np.mean((H_gen - H_recon) ** 2, axis=1)

    return {
        'mse_qpos': mse_qpos_arr,
        'mse_mom': mse_mom_arr,
        'mse_energy': mse_energy_arr,
        'recon_qpos': recon_qpos,
        'recon_mom': recon_mom,
        'H_gen': H_gen,
        'H_recon': H_recon,
    }


def load_validation_data(val_path, num_samples, torque_policy=None):
    """Load torque sequences from validation HDF5 file, optionally filtering by policy."""
    torques = []
    with h5py.File(val_path, 'r') as f:
        num_available = f.attrs['num_trajectories']
        for i in range(num_available):
            grp = f[f'traj_{i}']
            if torque_policy is not None:
                policy = grp.attrs.get('torque_policy', b'sinusoidal')
                if isinstance(policy, bytes):
                    policy = policy.decode()
                if policy != torque_policy:
                    continue
            torques.append(grp['seq_torque'][:])
            if len(torques) >= num_samples:
                break

    torques = np.stack(torques, axis=0)   # [N, 4000, 3]
    return torques


def make_comparison_figure(traj_len, torque_np, state_base_np, state_guid_np,
                           recon_base, recon_guid, qpos_dim, data_dt,
                           hnn, device, save_path):
    """Create 5-row comparison figure for ONE sample at a given length.

    Matches the plotting style of compare_generated_with_reconstructed() from
    src/models/utils.py: scatter plots, timestep x-axis, default y-axis.

    Row 1: Torque (3 dims)
    Row 2: Unguided (qpos+mom) vs Reconstructed
    Row 3: Guided (qpos+mom) vs Reconstructed
    Row 4: Unguided H(t) vs Reconstructed H(t)
    Row 5: Guided H(t) vs Reconstructed H(t)
    """
    # Extract state components — generated[1:] vs reconstructed[1:] (aligned, skip initial)
    qpos_base = state_base_np[1:, :qpos_dim]
    mom_base = state_base_np[1:, qpos_dim:]
    qpos_guid = state_guid_np[1:, :qpos_dim]
    mom_guid = state_guid_np[1:, qpos_dim:]
    qpos_recon_base = recon_base['qpos'][1:]
    mom_recon_base = recon_base['mom'][1:]
    qpos_recon_guid = recon_guid['qpos'][1:]
    mom_recon_guid = recon_guid['mom'][1:]

    T = len(qpos_base)
    t = np.arange(T)

    # Compute H(t) for this single sample
    def get_H(qpos, mom):
        q = torch.tensor(qpos[None], dtype=torch.float32, device=device)
        p = torch.tensor(mom[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            H = hnn(p.reshape(-1, qpos_dim), q.reshape(-1, qpos_dim))
        return H.cpu().numpy().squeeze()

    H_base_gen = get_H(qpos_base, mom_base)
    H_base_recon = get_H(qpos_recon_base, mom_recon_base)
    H_guid_gen = get_H(qpos_guid, mom_guid)
    H_guid_recon = get_H(qpos_recon_guid, mom_recon_guid)

    # 5 rows x 6 cols (qpos 3 + mom 3); torque & H(t) use first 3 cols only
    fig, axes = plt.subplots(5, 6, figsize=(30, 25))

    # Row 0: Torque
    for d in range(3):
        ax = axes[0, d]
        ax.scatter(np.arange(traj_len), torque_np[:, d], s=1, c='black', alpha=0.7)
        ax.set_title(f'seq_torque[{d}]')
    for d in range(3, 6):
        axes[0, d].set_visible(False)

    # Row 1: Unguided qpos + mom vs Reconstructed
    for d in range(3):
        ax = axes[1, d]
        ax.scatter(t, qpos_base[:, d], s=1, c='blue', label='Generated', alpha=0.7)
        ax.scatter(t, qpos_recon_base[:, d], s=1, c='red', label='MuJoCo', alpha=0.7)
        ax.set_title(f'seq_qpos[{d}] (unguided)')
        ax.legend(markerscale=5)
    for d in range(3):
        ax = axes[1, 3 + d]
        ax.scatter(t, mom_base[:, d], s=1, c='blue', label='Generated', alpha=0.7)
        ax.scatter(t, mom_recon_base[:, d], s=1, c='red', label='MuJoCo', alpha=0.7)
        ax.set_title(f'seq_mom[{d}] (unguided)')
        ax.legend(markerscale=5)

    # Row 2: Guided qpos + mom vs Reconstructed
    for d in range(3):
        ax = axes[2, d]
        ax.scatter(t, qpos_guid[:, d], s=1, c='blue', label='Generated', alpha=0.7)
        ax.scatter(t, qpos_recon_guid[:, d], s=1, c='red', label='MuJoCo', alpha=0.7)
        ax.set_title(f'seq_qpos[{d}] (guided)')
        ax.legend(markerscale=5)
    for d in range(3):
        ax = axes[2, 3 + d]
        ax.scatter(t, mom_guid[:, d], s=1, c='blue', label='Generated', alpha=0.7)
        ax.scatter(t, mom_recon_guid[:, d], s=1, c='red', label='MuJoCo', alpha=0.7)
        ax.set_title(f'seq_mom[{d}] (guided)')
        ax.legend(markerscale=5)

    # Row 3: Unguided H(t)
    axes[3, 0].scatter(t, H_base_gen, s=1, c='blue', label='Generated', alpha=0.7)
    axes[3, 0].scatter(t, H_base_recon, s=1, c='red', label='MuJoCo', alpha=0.7)
    axes[3, 0].set_title('H(t) (unguided)')
    axes[3, 0].legend(markerscale=5)
    for d in range(1, 6):
        axes[3, d].set_visible(False)

    # Row 4: Guided H(t)
    axes[4, 0].scatter(t, H_guid_gen, s=1, c='blue', label='Generated', alpha=0.7)
    axes[4, 0].scatter(t, H_guid_recon, s=1, c='red', label='MuJoCo', alpha=0.7)
    axes[4, 0].set_title('H(t) (guided)')
    axes[4, 0].legend(markerscale=5)
    for d in range(1, 6):
        axes[4, d].set_visible(False)

    fig.suptitle(f'Trajectory Comparison — Length {traj_len}', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Quick DPF evaluation: unguided vs HNN-guided")
    parser.add_argument("--dpf_checkpoint", type=str, default=None)
    parser.add_argument("--hnn_checkpoint", type=str, default=None,
                        help="HNN checkpoint path (default: latest in checkpoints/)")
    parser.add_argument("--val_file", type=str,
                        default="data/traj_2000-steps_4000.h5")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--lengths", type=str,
                        default="50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500")
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_dir", type=str, default="output/eval_plots")
    # Sampling settings
    parser.add_argument("--sampler", type=str, default="ddim")
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--context_fraction", type=float, default=0.2)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    # HNN guidance settings
    parser.add_argument("--guidance_method", type=str, default="adam")
    parser.add_argument("--guidance_steps", type=int, default=25)
    parser.add_argument("--guidance_lr", type=float, default=0.01)
    parser.add_argument("--guidance_after_steps", type=int, default=45)
    parser.add_argument("--lambda_init", type=float, default=0.0)
    parser.add_argument("--torque_policy", type=str, default=None,
                        help="Filter validation torques by policy (e.g. 'sinusoidal'). None=all.")
    args = parser.parse_args()

    device = torch.device('cuda:0')

    # Find DPF checkpoint
    if args.dpf_checkpoint is None:
        ckpt_dir = project_root / 'checkpoints'
        dpf_ckpts = sorted(
            ckpt_dir.glob('trajectory_dpf_StateOnlyAdaLN*AbsoluteTimeEncoding&VariableTrajLength*.ckpt'),
            key=lambda p: p.stat().st_mtime
        )
        if not dpf_ckpts:
            print("ERROR: No DPF checkpoint found!")
            return
        args.dpf_checkpoint = str(dpf_ckpts[-1])

    print(f"DPF checkpoint: {args.dpf_checkpoint}")
    print(f"HNN checkpoint: {args.hnn_checkpoint}")
    print(f"Validation file: {args.val_file}")
    print(f"\nSettings:")
    print(f"  context_fraction={args.context_fraction}")
    print(f"  sampler={args.sampler}, diffusion_steps={args.num_diffusion_steps}")
    print(f"  guidance_scale={args.guidance_scale}")
    print(f"  HNN guidance: method={args.guidance_method}, steps={args.guidance_steps}, "
          f"lr={args.guidance_lr}, after_steps={args.guidance_after_steps}, lambda_init={args.lambda_init}")

    # Load DPF
    print("\nLoading DPF model...")
    model = TrajectoryDPF.load_from_checkpoint(args.dpf_checkpoint, map_location=device)
    model = model.to(device)

    checkpoint = torch.load(args.dpf_checkpoint, map_location=device, weights_only=False)
    ema_shadow = checkpoint.get('ema_shadow', None)
    ema_decay = checkpoint.get('ema_decay', 0.9995)
    if ema_shadow is not None:
        model.ema = EMA(model.model, decay=ema_decay)
        for name, tensor in ema_shadow.items():
            if name in model.ema.shadow:
                model.ema.shadow[name] = tensor.to(device=device, dtype=model.ema.shadow[name].dtype)
        print(f"  EMA loaded ({len(ema_shadow)} params)")
    del checkpoint

    print(f"  qpos_dim={model.qpos_dim}, mom_dim={model.mom_dim}, torque_dim={model.torque_dim}")
    print(f"  dt={float(model.dt)}, data_dt={float(model.data_dt)}")

    # Find HNN checkpoint
    if args.hnn_checkpoint is None:
        hnn_ckpts = sorted(
            (project_root / 'checkpoints').glob('SeperableHNN*.ckpt'),
            key=lambda p: p.stat().st_mtime
        )
        if not hnn_ckpts:
            print("ERROR: No HNN checkpoint found!")
            return
        args.hnn_checkpoint = str(hnn_ckpts[-1])

    # Load HNN
    hnn_path = str(project_root / args.hnn_checkpoint) if not os.path.isabs(args.hnn_checkpoint) else args.hnn_checkpoint
    print(f"Loading HNN from {hnn_path}...")
    hnn = HNNWrapper.load_from_checkpoint(hnn_path, map_location=device)
    hnn = hnn.to(device)
    hnn.eval()
    print(f"  HNN loaded (q_std={hnn.q_std.mean().item():.4f}, p_std={hnn.p_std.mean().item():.4f})")

    # Load validation torques
    val_path = str(project_root / args.val_file)
    print(f"\nLoading {args.num_samples} torque sequences from validation file...")
    all_val_torques = load_validation_data(val_path, args.num_samples, torque_policy=args.torque_policy)
    print(f"  Loaded shape: {all_val_torques.shape} (policy filter: {args.torque_policy or 'all'})")

    # Parse lengths
    lengths = [int(x) for x in args.lengths.split(',')]
    print(f"\nEvaluating {len(lengths)} lengths, {args.num_samples} samples each")

    dt = float(model.dt)
    data_dt = float(model.data_dt)
    qpos_dim = model.qpos_dim

    # Create plot directory
    plot_dir = str(project_root / args.plot_dir)
    os.makedirs(plot_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = []

    for traj_len in lengths:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  Length {traj_len}")
        print(f"{'='*60}")

        # Truncate validation torques to current length
        torque_truncated = all_val_torques[:, :traj_len, :]  # [N, traj_len, 3]
        torque_tensor = torch.tensor(torque_truncated, dtype=torch.float32, device=device)

        # Step 1: Generate UNGUIDED baseline (no HNN), using validation torques
        print(f"  [Unguided] Generating {args.num_samples} samples...")
        state_base, torque_base = model.sample_trajectories(
            num_samples=args.num_samples,
            trajectory_length=traj_len,
            num_diffusion_steps=args.num_diffusion_steps,
            context_fraction=args.context_fraction,
            use_ema=True,
            sampler=args.sampler,
            guidance_scale=args.guidance_scale,
            hnn=None,
            guidance_steps=0,
            torque=torque_tensor,
        )

        # Step 2: Generate GUIDED with HNN (same torques)
        print(f"  [Guided] Generating {args.num_samples} samples with HNN guidance...")
        state_guid, torque_guid = model.sample_trajectories(
            num_samples=args.num_samples,
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
            torque=torque_tensor,
        )

        # Step 3: Compute MSE for both
        state_base_np = state_base.cpu().numpy()
        torque_base_np = torque_base.cpu().numpy()
        state_guid_np = state_guid.cpu().numpy()
        torque_guid_np = torque_guid.cpu().numpy()

        print(f"  [Unguided] Computing MSE...")
        mse_base = compute_mse_for_batch(
            state_base_np, torque_base_np, qpos_dim, dt, data_dt, hnn, device, args.num_workers)

        print(f"  [Guided] Computing MSE...")
        mse_guid = compute_mse_for_batch(
            state_guid_np, torque_guid_np, qpos_dim, dt, data_dt, hnn, device, args.num_workers)

        elapsed = time.time() - t0

        row = {
            'length': traj_len,
            'base_qpos_mean': np.mean(mse_base['mse_qpos']),
            'base_qpos_std': np.std(mse_base['mse_qpos']),
            'base_mom_mean': np.mean(mse_base['mse_mom']),
            'base_mom_std': np.std(mse_base['mse_mom']),
            'base_energy_mean': np.mean(mse_base['mse_energy']),
            'base_energy_std': np.std(mse_base['mse_energy']),
            'guid_qpos_mean': np.mean(mse_guid['mse_qpos']),
            'guid_qpos_std': np.std(mse_guid['mse_qpos']),
            'guid_mom_mean': np.mean(mse_guid['mse_mom']),
            'guid_mom_std': np.std(mse_guid['mse_mom']),
            'guid_energy_mean': np.mean(mse_guid['mse_energy']),
            'guid_energy_std': np.std(mse_guid['mse_energy']),
        }
        results.append(row)

        print(f"\n  {'Metric':<14} {'Unguided':>22} {'Guided':>22}")
        print(f"  {'-'*58}")
        print(f"  {'MSE qpos':<14} {row['base_qpos_mean']:>10.6f} ± {row['base_qpos_std']:<10.6f} {row['guid_qpos_mean']:>10.6f} ± {row['guid_qpos_std']:<10.6f}")
        print(f"  {'MSE mom':<14} {row['base_mom_mean']:>10.6f} ± {row['base_mom_std']:<10.6f} {row['guid_mom_mean']:>10.6f} ± {row['guid_mom_std']:<10.6f}")
        print(f"  {'MSE energy':<14} {row['base_energy_mean']:>10.6f} ± {row['base_energy_std']:<10.6f} {row['guid_energy_mean']:>10.6f} ± {row['guid_energy_std']:<10.6f}")
        print(f"  ({elapsed:.1f}s)")

        # Step 4: Create comparison figure for first sample
        print(f"  Creating comparison plot...")
        recon_base_0 = {
            'qpos': mse_base['recon_qpos'][0],
            'mom': mse_base['recon_mom'][0],
        }
        recon_guid_0 = {
            'qpos': mse_guid['recon_qpos'][0],
            'mom': mse_guid['recon_mom'][0],
        }
        fig_path = os.path.join(plot_dir, f'comparison_L{traj_len:04d}.png')
        make_comparison_figure(
            traj_len, torque_base_np[0], state_base_np[0], state_guid_np[0],
            recon_base_0, recon_guid_0, qpos_dim, data_dt,
            hnn, device, fig_path,
        )
        print(f"  Saved: {fig_path}")

    # Print markdown tables
    print("\n\n" + "=" * 80)
    print("RESULTS TABLES (Markdown)")
    print("=" * 80)

    ckpt_name = Path(args.dpf_checkpoint).stem
    epoch_str = "unknown"
    if "epoch=" in ckpt_name:
        epoch_str = ckpt_name.split("epoch=")[1].split("_")[0]

    print(f"\n# DPF Evaluation (Epoch {epoch_str}, {args.num_samples} samples/length, context_fraction={args.context_fraction})")

    print("\n## MSE Position\n")
    print("| Length | Unguided | Guided |")
    print("| --- | --- | --- |")
    for r in results:
        print(f"| {r['length']} | {r['base_qpos_mean']:.6g} ± {r['base_qpos_std']:.6g} | {r['guid_qpos_mean']:.6g} ± {r['guid_qpos_std']:.6g} |")

    print("\n## MSE Momentum\n")
    print("| Length | Unguided | Guided |")
    print("| --- | --- | --- |")
    for r in results:
        print(f"| {r['length']} | {r['base_mom_mean']:.6g} ± {r['base_mom_std']:.6g} | {r['guid_mom_mean']:.6g} ± {r['guid_mom_std']:.6g} |")

    print("\n## MSE Energy\n")
    print("| Length | Unguided | Guided |")
    print("| --- | --- | --- |")
    for r in results:
        print(f"| {r['length']} | {r['base_energy_mean']:.6g} ± {r['base_energy_std']:.6g} | {r['guid_energy_mean']:.6g} ± {r['guid_energy_std']:.6g} |")


if __name__ == "__main__":
    main()
