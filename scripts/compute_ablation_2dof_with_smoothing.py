"""
Compute NMSE and HamRes for 2DoF/3DoF ablation with smoothing sigma sweep.
Generates trajectories on-the-fly with smoothing and computes both metrics.

Usage:
    # Single sigma (legacy)
    python scripts/compute_ablation_2dof_with_smoothing.py --system 2dof --policy sinusoidal --smooth_sigma 3.0

    # Sigma sweep
    python scripts/compute_ablation_2dof_with_smoothing.py --system 2dof --policy sinusoidal \
        --sigmas 0,0.5,1,2,3,5,7,10,15,20,30,50 --num_samples 200 --lengths 100,200,500,1000

    python scripts/compute_ablation_2dof_with_smoothing.py --system 3dof --policy sinusoidal \
        --sigmas 0,0.5,1,2,3,5,7,10,15,20,30,50 --num_samples 200 --lengths 100,200,500,1000
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import pandas as pd
import torch
import h5py
import mujoco
from tqdm import tqdm

from src.models.trajectory_dpf import TrajectoryDPF
from src.models.HNN import HNNWrapper
from src.models.utils import EMA, reconstruct_traj_with_momentum


# Shared constants
DT = 0.0002       # Data timestep (both 2DoF and 3DoF)
SIM_DT = 0.0001   # MuJoCo simulation timestep
BATCH_SIZE_UNGUIDED = 200
BATCH_SIZE_GUIDED = 50

# Shared torque files (3DoF native; 2DoF slices first 2 dims)
TORQUE_PATHS = {
    'sinusoidal': project_root / 'data' / 'sinusoidal_torques_1000_L1500.h5',
    'gp': project_root / 'data' / 'gp_torques_1000_L1500.h5',
    'spline': project_root / 'data' / 'spline_torques_1000_L1500.h5',
    'zero': None,
}

SYSTEM_CONFIGS = {
    '2dof': {
        'dpf_ckpt': project_root / 'checkpoints' / '2dof' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0008.ckpt',
        'hnn_ckpt': project_root / 'checkpoints' / '2dof' / 'SeperableHNN-2DOF-epoch-epoch=999.ckpt',
        'xml_path': str(project_root / 'configs' / 'rigid_arm_hinge_2dof.xml'),
        'qpos_dim': 2,
        'torque_dim': 2,
        'guidance_steps': 10,
        'guidance_lr': 0.0001,
        'guidance_after_steps': 45,
    },
    '3dof': {
        'dpf_ckpt': project_root / 'checkpoints' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt',
        'hnn_ckpt': project_root / 'checkpoints' / 'StructuredHNN-dim256-epoch-epoch=749.ckpt',
        'xml_path': str(project_root / 'configs' / 'rigid_arm_hinge.xml'),
        'qpos_dim': 3,
        'torque_dim': 3,
        'guidance_steps': 25,
        'guidance_lr': 0.01,
        'guidance_after_steps': 45,
    },
}


def load_torques(policy, num_samples, max_length, device, cfg):
    torque_dim = cfg['torque_dim']
    if policy == 'zero':
        return torch.zeros(num_samples, max_length, torque_dim, device=device)
    path = TORQUE_PATHS[policy]
    with h5py.File(path, 'r') as f:
        torques = f['torques'][:num_samples, :max_length, :torque_dim]
    return torch.tensor(torques, dtype=torch.float32, device=device)


def compute_hamres(qpos, mom, torque, hnn, var_dq, var_dp, dt=DT):
    T = qpos.shape[0]
    if T < 3:
        return float('nan')
    eps = 1e-8
    qdot = (qpos[2:] - qpos[:-2]) / (2 * dt)
    pdot = (mom[2:] - mom[:-2]) / (2 * dt)
    q_mid, p_mid, tau_mid = qpos[1:-1], mom[1:-1], torque[1:-1]
    with torch.inference_mode(False):
        with torch.enable_grad():
            p_grad = p_mid.detach().clone().requires_grad_(True)
            q_grad = q_mid.detach().clone().requires_grad_(True)
            H = hnn(p_grad, q_grad)
            dH_dp, dH_dq = torch.autograd.grad(H.sum(), (p_grad, q_grad), create_graph=False)
    r_q = qdot - dH_dp
    r_p = pdot - (-dH_dq + tau_mid)
    mse_q = (r_q ** 2).mean()
    mse_p = (r_p ** 2).mean()
    return (mse_q / (var_dq + eps) + mse_p / (var_dp + eps)).item()


def compute_nmse(state, torque, mj_model, qpos_dim, dt=DT, sim_dt=SIM_DT):
    qpos = state[:, :qpos_dim].cpu().numpy()
    mom = state[:, qpos_dim:].cpu().numpy()
    torque_np = torque.cpu().numpy()
    T = qpos.shape[0]

    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos[0]
    data.qvel[:] = 0
    mujoco.mj_forward(mj_model, data)
    M = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, M, data.qM)
    initial_qvel = np.linalg.solve(M, mom[0])

    recon = reconstruct_traj_with_momentum(
        mj_model, T, sim_dt, qpos[0], initial_qvel, torque_np, data_dt=dt,
    )
    gt_qpos = recon['seq_qpos']
    gt_mom = recon['seq_mom']

    T_min = min(len(qpos) - 1, len(gt_qpos))
    # Per-dimension MSE (mean over time), then average across dimensions
    mse_q_per_dim = ((qpos[1:T_min+1] - gt_qpos[:T_min]) ** 2).mean(axis=0)
    mse_p_per_dim = ((mom[1:T_min+1] - gt_mom[:T_min]) ** 2).mean(axis=0)
    var_q_per_dim = np.var(gt_qpos[:T_min], axis=0)
    var_p_per_dim = np.var(gt_mom[:T_min], axis=0)
    nmse_q_per_dim = np.where(var_q_per_dim > 1e-12, mse_q_per_dim / var_q_per_dim, 0.0)
    nmse_p_per_dim = np.where(var_p_per_dim > 1e-12, mse_p_per_dim / var_p_per_dim, 0.0)
    return float(nmse_q_per_dim.mean()), float(nmse_p_per_dim.mean()), nmse_q_per_dim, nmse_p_per_dim


def stats_dict(prefix, values):
    """Compute mean, std, p25, median, p95, p99 for a list of values."""
    if not values:
        return {f'{prefix}_{k}': np.nan for k in ['mean', 'std', 'p25', 'median', 'p95', 'p99']}
    arr = np.array(values)
    return {
        f'{prefix}_mean': np.mean(arr),
        f'{prefix}_std': np.std(arr),
        f'{prefix}_p25': np.percentile(arr, 25),
        f'{prefix}_median': np.median(arr),
        f'{prefix}_p95': np.percentile(arr, 95),
        f'{prefix}_p99': np.percentile(arr, 99),
    }


def generate_batch(dpf, num_samples, L, batch_torques, batch_size, guidance_kwargs,
                    initial_noise=None):
    """Generate trajectories in batches, optionally with shared initial noise."""
    all_states, all_torques = [], []
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        bt = batch_torques[batch_start:batch_end]
        bn = initial_noise[batch_start:batch_end] if initial_noise is not None else None
        state, torque_out = dpf.sample_trajectories(
            num_samples=bt.shape[0],
            trajectory_length=L,
            context_fraction=0.5,
            use_ema=True,
            torque=bt,
            initial_noise=bn,
            **guidance_kwargs,
        )
        all_states.append(state)
        all_torques.append(torque_out)
    return torch.cat(all_states, dim=0), torch.cat(all_torques, dim=0)


def compute_metrics_for_samples(states, torques_out, num_samples, mj_model, hnn, var_dq, var_dp, qpos_dim, desc=""):
    """Compute per-sample NMSE and HamRes, return raw lists.

    Returns: nmse_q_list, nmse_p_list, hamres_list, nmse_q_per_dim_list, nmse_p_per_dim_list
        where *_per_dim_list[i] is a 1D array of shape (num_dims,).
    """
    nmse_q_list, nmse_p_list, hamres_list = [], [], []
    nmse_q_per_dim_list, nmse_p_per_dim_list = [], []
    for i in tqdm(range(num_samples), desc=desc):
        state = states[i]
        tau = torques_out[i]
        nmse_q, nmse_p, nmse_q_pd, nmse_p_pd = compute_nmse(state, tau, mj_model, qpos_dim)
        qpos = state[:, :qpos_dim]
        mom = state[:, qpos_dim:]
        hr = compute_hamres(qpos, mom, tau, hnn, var_dq, var_dp)
        nmse_q_list.append(nmse_q)
        nmse_p_list.append(nmse_p)
        nmse_q_per_dim_list.append(nmse_q_pd)
        nmse_p_per_dim_list.append(nmse_p_pd)
        if not np.isnan(hr):
            hamres_list.append(hr)
    return nmse_q_list, nmse_p_list, hamres_list, nmse_q_per_dim_list, nmse_p_per_dim_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, default='2dof', choices=['2dof', '3dof'])
    parser.add_argument('--policy', type=str, required=True,
                        choices=['sinusoidal', 'gp', 'zero', 'spline'])
    parser.add_argument('--smooth_sigma', type=float, default=None,
                        help='Single sigma (legacy mode)')
    parser.add_argument('--sigmas', type=str, default=None,
                        help='Comma-separated sigma values for sweep mode')
    parser.add_argument('--lengths', type=str, default=None,
                        help='Comma-separated trajectory lengths (default: 50-1500 step 50)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_samples', type=int, default=1000)
    args = parser.parse_args()

    # Determine sigma list
    if args.sigmas:
        sigmas = [float(s) for s in args.sigmas.split(',')]
        sweep_mode = True
    elif args.smooth_sigma is not None:
        sigmas = [args.smooth_sigma]
        sweep_mode = False
    else:
        sigmas = [0.0]
        sweep_mode = False

    # Determine trajectory lengths
    if args.lengths:
        traj_lengths = [int(x) for x in args.lengths.split(',')]
    else:
        traj_lengths = list(range(50, 1501, 50))

    cfg = SYSTEM_CONFIGS[args.system]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"System: {args.system}, Policy: {args.policy}, Device: {device}")
    print(f"Sigmas: {sigmas}")
    print(f"Lengths: {traj_lengths}")
    print(f"Samples: {args.num_samples}")

    # Load DPF
    print("Loading DPF model...")
    dpf = TrajectoryDPF.load_from_checkpoint(str(cfg['dpf_ckpt']), map_location=device, strict=False)
    dpf.eval()
    dpf.to(device)

    checkpoint = torch.load(str(cfg['dpf_ckpt']), map_location=device, weights_only=False)
    if 'ema_shadow' in checkpoint:
        dpf.ema = EMA(dpf.model, decay=0.9995)
        for name, tensor in checkpoint['ema_shadow'].items():
            if name in dpf.ema.shadow:
                dpf.ema.shadow[name] = tensor.to(device=device, dtype=dpf.ema.shadow[name].dtype)
        print("  EMA loaded")
    del checkpoint

    # Load HNN
    print("Loading HNN model...")
    hnn = HNNWrapper.load_from_checkpoint(str(cfg['hnn_ckpt']), map_location=device)
    hnn.eval()
    hnn.to(device)
    var_dq = hnn.qvel_var.mean().to(device)
    var_dp = hnn.mom_dot_var.mean().to(device)
    print(f"  var_dq={var_dq.item():.4f}, var_dp={var_dp.item():.4f}")

    # Load MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path(cfg['xml_path'])
    mj_model.opt.timestep = SIM_DT

    # Load torques
    max_length = max(traj_lengths)
    print(f"Loading torques for policy '{args.policy}'...")
    torques = load_torques(args.policy, args.num_samples, max_length, device, cfg)

    qpos_dim = cfg['qpos_dim']
    results = []

    for L in traj_lengths:
        print(f"\n{'='*60}")
        print(f"Length {L}")
        print(f"{'='*60}")

        batch_torques = torques[:args.num_samples, :L]

        for sigma in sigmas:
            print(f"\n  --- sigma={sigma} ---")

            # Shared initial noise for paired comparison
            noise = torch.randn(args.num_samples, L, dpf.state_dim, device=device)

            # Generate unguided with this sigma
            print("  Generating unguided...")
            ung_kwargs = dict(hnn=None, guidance_steps=0, smooth_sigma=sigma)
            ung_states, ung_torques = generate_batch(
                dpf, args.num_samples, L, batch_torques, BATCH_SIZE_UNGUIDED, ung_kwargs,
                initial_noise=noise)
            print("  Computing unguided metrics...")
            ung_nq, ung_np, ung_hr, _, _ = compute_metrics_for_samples(
                ung_states, ung_torques, args.num_samples, mj_model, hnn, var_dq, var_dp, qpos_dim,
                desc=f"unguided L={L} s={sigma}")
            del ung_states, ung_torques

            # Generate guided with this sigma (same initial noise)
            gui_kwargs = dict(
                hnn=hnn,
                guidance_method='adam',
                guidance_steps=cfg['guidance_steps'],
                guidance_lr=cfg['guidance_lr'],
                guidance_after_steps=cfg['guidance_after_steps'],
                smooth_sigma=sigma,
            )
            gui_states, gui_torques = generate_batch(
                dpf, args.num_samples, L, batch_torques, BATCH_SIZE_GUIDED, gui_kwargs,
                initial_noise=noise)
            gui_nq, gui_np, gui_hr, _, _ = compute_metrics_for_samples(
                gui_states, gui_torques, args.num_samples, mj_model, hnn, var_dq, var_dp, qpos_dim,
                desc=f"guided L={L} s={sigma}")

            # Build row
            row = {'sigma': sigma, 'trajectory_length': L}
            row.update(stats_dict('unguided_nmse_q', ung_nq))
            row.update(stats_dict('unguided_nmse_p', ung_np))
            row.update(stats_dict('unguided_hamres', ung_hr))
            row.update(stats_dict('guided_nmse_q', gui_nq))
            row.update(stats_dict('guided_nmse_p', gui_np))
            row.update(stats_dict('guided_hamres', gui_hr))
            results.append(row)

            print(f"    Ung: NMSE_q={np.mean(ung_nq):.6f}, HamRes={np.mean(ung_hr):.4f}")
            print(f"    Gui: NMSE_q={np.mean(gui_nq):.6f}, HamRes={np.mean(gui_hr):.4f}")

    # Save results
    df = pd.DataFrame(results)
    if sweep_mode:
        out_dir = project_root / 'output_ablation' / 'results' / f'{args.system}_smoothing_sweep'
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f'sweep_{args.policy}.csv'
    else:
        out_dir = project_root / 'output_ablation' / 'results' / f'{args.system}_smoothed'
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f'metrics_{args.policy}_sigma{sigmas[0]}.csv'

    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Print summary if sweep mode
    if sweep_mode and len(sigmas) > 1:
        print(f"\n{'='*80}")
        print(f"SUMMARY: {args.system} {args.policy}")
        print(f"{'='*80}")
        print(f"{'sigma':>6} | {'gui_nmse_q_med':>14} | {'gui_hamres_med':>14} | {'gui_nmse_q_mean':>15} | {'gui_hamres_mean':>15}")
        print("-" * 75)
        for sigma in sigmas:
            sub = df[df['sigma'] == sigma]
            print(f"{sigma:>6} | {sub['guided_nmse_q_median'].mean():>14.6f} | {sub['guided_hamres_median'].mean():>14.4f} | {sub['guided_nmse_q_mean'].mean():>15.6f} | {sub['guided_hamres_mean'].mean():>15.4f}")

        # Find best
        avg = df.groupby('sigma').agg({
            'guided_nmse_q_median': 'mean',
            'guided_hamres_median': 'mean',
        })
        best_nmse_sigma = avg['guided_nmse_q_median'].idxmin()
        best_hamres_sigma = avg['guided_hamres_median'].idxmin()
        print(f"\nBest NMSE sigma: {best_nmse_sigma}")
        print(f"Best HamRes sigma: {best_hamres_sigma}")


if __name__ == '__main__':
    main()
