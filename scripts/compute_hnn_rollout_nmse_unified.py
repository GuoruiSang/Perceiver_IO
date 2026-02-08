"""
Compute NMSE for HNN forward rollout trajectories (unified 2DoF/3DoF).

Uses initial states from saved DPF trajectory H5 files and torques from
shared torque files (3DoF native; 2DoF slices first 2 dims).

Usage:
    python scripts/compute_hnn_rollout_nmse_unified.py --system 2dof --policy sinusoidal
    python scripts/compute_hnn_rollout_nmse_unified.py --system 3dof --policy gp
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
from tqdm import tqdm
import mujoco

from src.models.HNN import HNNWrapper
from src.models.utils import reconstruct_traj_with_momentum

# Constants
SIM_DT = 0.0001
DATA_DT = 0.0002
TRAJECTORY_LENGTHS = list(range(50, 1501, 50))
MAX_SAMPLES = 100

# Shared torque files (3DoF native; 2DoF slices first 2 dims)
TORQUE_PATHS = {
    'sinusoidal': project_root / 'data' / 'sinusoidal_torques_1000_L1500.h5',
    'gp': project_root / 'data' / 'gp_torques_1000_L1500.h5',
    'spline': project_root / 'data' / 'spline_torques_1000_L1500.h5',
    'zero': None,
}

SYSTEM_CONFIGS = {
    '2dof': {
        'hnn_ckpt': project_root / 'checkpoints' / '2dof' / 'SeperableHNN-2DOF-epoch-epoch=999.ckpt',
        'xml_path': str(project_root / 'configs' / 'rigid_arm_hinge_2dof.xml'),
        'traj_dir': project_root / 'output_ablation' / 'trajectories' / '2dof',
        'output_dir': project_root / 'output_ablation' / 'results' / '2dof_smoothed',
        'qpos_dim': 2,
        'torque_dim': 2,
    },
    '3dof': {
        'hnn_ckpt': project_root / 'checkpoints' / 'StructuredHNN-dim256-epoch-epoch=749.ckpt',
        'xml_path': str(project_root / 'configs' / 'rigid_arm_hinge.xml'),
        'traj_dir': project_root / 'output_ablation' / 'trajectories' / 'original',
        'output_dir': project_root / 'output_ablation' / 'results' / '3dof_smoothed',
        'qpos_dim': 3,
        'torque_dim': 3,
    },
}


def compute_nmse(generated, reconstructed):
    """NMSE = MSE / var(reconstructed)."""
    mse = np.mean((generated - reconstructed) ** 2)
    var = np.var(reconstructed)
    return mse / var if var > 1e-12 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, required=True, choices=['2dof', '3dof'])
    parser.add_argument('--policy', type=str, required=True,
                        choices=['sinusoidal', 'gp', 'zero', 'spline'])
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    cfg = SYSTEM_CONFIGS[args.system]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    qpos_dim = cfg['qpos_dim']
    torque_dim = cfg['torque_dim']

    print(f"System: {args.system}, Policy: {args.policy}, Device: {device}")

    # Load HNN
    print("Loading HNN model...")
    hnn = HNNWrapper.load_from_checkpoint(str(cfg['hnn_ckpt']), map_location=device)
    hnn.eval()
    hnn.to(device)

    # Load MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path(cfg['xml_path'])
    mj_model.opt.timestep = SIM_DT

    # Load initial states from trajectory H5 file
    # Use any available policy's H5 (fallback for 3DoF spline which has no H5)
    h5_policy = args.policy
    h5_path = cfg['traj_dir'] / f'exp_a_{h5_policy}.h5'
    if not h5_path.exists():
        h5_policy = 'zero'
        h5_path = cfg['traj_dir'] / f'exp_a_{h5_policy}.h5'
        print(f"  No H5 for {args.policy}, using {h5_policy} for initial states")

    with h5py.File(h5_path, 'r') as f:
        # Use L=50 entry for initial states (always available, initial state is length-independent)
        states_init = f['unguided/L50/state'][:MAX_SAMPLES]  # [N, 50, state_dim]
    initial_qpos = states_init[:, 0, :qpos_dim]  # [N, qpos_dim]
    initial_mom = states_init[:, 0, qpos_dim:]    # [N, qpos_dim]
    num_samples = initial_qpos.shape[0]
    print(f"Loaded {num_samples} initial states from {h5_path.name} (L=50)")

    # Load torques from shared files
    max_length = max(TRAJECTORY_LENGTHS)
    if args.policy == 'zero':
        all_torques = np.zeros((num_samples, max_length, torque_dim))
    else:
        with h5py.File(TORQUE_PATHS[args.policy], 'r') as f:
            all_torques = f['torques'][:num_samples, :max_length, :torque_dim]
    print(f"Loaded torques: {all_torques.shape}")

    # Convert initial states to torch
    p0 = torch.tensor(initial_mom, dtype=torch.float32, device=device)
    q0 = torch.tensor(initial_qpos, dtype=torch.float32, device=device)

    results = []

    for L in tqdm(TRAJECTORY_LENGTHS, desc=f'{args.system} {args.policy}'):
        torques_L = all_torques[:, :L]  # [N, L, torque_dim]
        tau_seq = torch.tensor(torques_L, dtype=torch.float32, device=device)
        tau_seq = tau_seq.permute(1, 0, 2)  # [T, N, torque_dim]

        # HNN forward rollout
        with torch.no_grad():
            p_traj, q_traj, _ = hnn.integrate_trajectory(
                p0=p0, q0=q0,
                tau_seq=tau_seq,
                dt=DATA_DT,
                num_steps=L - 1
            )

        # [T, N, dim] -> [N, T, dim]
        hnn_qpos_all = q_traj.permute(1, 0, 2).cpu().numpy()
        hnn_mom_all = p_traj.permute(1, 0, 2).cpu().numpy()

        nmse_q_list = []
        nmse_p_list = []

        for i in range(num_samples):
            hnn_qpos = hnn_qpos_all[i]  # [T, dim]
            hnn_mom = hnn_mom_all[i]
            torque_np = torques_L[i]

            # MuJoCo reconstruction from HNN initial state
            data = mujoco.MjData(mj_model)
            data.qpos[:] = hnn_qpos[0]
            data.qvel[:] = 0
            mujoco.mj_forward(mj_model, data)
            M = np.zeros((mj_model.nv, mj_model.nv))
            mujoco.mj_fullM(mj_model, M, data.qM)
            init_qvel = np.linalg.solve(M, hnn_mom[0])

            recon = reconstruct_traj_with_momentum(
                mj_model, len(hnn_qpos), SIM_DT,
                hnn_qpos[0], init_qvel, torque_np,
                data_dt=DATA_DT,
            )

            nmse_q = compute_nmse(hnn_qpos[1:], recon['seq_qpos'])
            nmse_p = compute_nmse(hnn_mom[1:], recon['seq_mom'])
            nmse_q_list.append(nmse_q)
            nmse_p_list.append(nmse_p)

        results.append({
            'trajectory_length': L,
            'hnn_nmse_q_mean': np.mean(nmse_q_list),
            'hnn_nmse_q_std': np.std(nmse_q_list),
            'hnn_nmse_p_mean': np.mean(nmse_p_list),
            'hnn_nmse_p_std': np.std(nmse_p_list),
        })

        print(f"  L{L}: NMSE_q={results[-1]['hnn_nmse_q_mean']:.6f}, "
              f"NMSE_p={results[-1]['hnn_nmse_p_mean']:.6f}", flush=True)

    # Save results
    cfg['output_dir'].mkdir(parents=True, exist_ok=True)
    output_path = cfg['output_dir'] / f'hnn_rollout_nmse_{args.policy}.csv'
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
