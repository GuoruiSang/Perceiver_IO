"""
Plot trajectory showcase: DPF-generated vs MuJoCo-reconstructed trajectories.

Creates a figure showing qpos and momentum for both unguided and guided samples.

Usage:
    python scripts/plot_trajectory_showcase.py --policy gp --length 1000 --dof 3
    python scripts/plot_trajectory_showcase.py --policy sinusoidal --length 1000 --dof 3
    python scripts/plot_trajectory_showcase.py --policy gp --length 500 --dof 2
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
import mujoco
import torch

from src.models.HNN import HNNWrapper


# ICLR-friendly style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Paths (will be set based on DOF)
CONFIGS = {
    3: {
        'mujoco_xml': project_root / 'configs' / 'rigid_arm_hinge.xml',
        'hnn_checkpoint': project_root / 'checkpoints' / 'SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt',
        'h5_dir': project_root / 'output_ablation' / 'trajectories' / 'original',
        'dt': 0.0001,
        'data_dt': 0.0002,
        'qpos_dim': 3,
        'mom_dim': 3,
    },
    2: {
        'mujoco_xml': project_root / 'configs' / 'rigid_arm_hinge_2dof.xml',
        'hnn_checkpoint': project_root / 'checkpoints' / '2dof' / 'SeperableHNN-2DOF-epoch-epoch=999.ckpt',
        'h5_dir': project_root / 'output_ablation' / 'trajectories' / '2dof',
        'dt': 0.0001,
        'data_dt': 0.0002,
        'qpos_dim': 2,
        'mom_dim': 2,
    },
}

POLICY_LABELS = {
    'sinusoidal': 'Sinusoidal Torque Policy',
    'gp': 'GP Torque Policy',
    'zero': 'Zero Torque Policy',
    'spline': 'Cubic Spline Torque Policy',
}

PLOTS_DIR = project_root / 'plots'


def reconstruct_trajectory(qpos_init, mom_init, torque, mj_model, dt, data_dt):
    """Reconstruct trajectory using MuJoCo forward simulation."""
    qpos_dim = len(qpos_init)
    T = len(torque)

    # Compute initial velocity from momentum
    M = np.zeros((mj_model.nv, mj_model.nv))
    data = mujoco.MjData(mj_model)
    data.qpos[:qpos_dim] = qpos_init
    data.qvel[:] = 0
    mujoco.mj_forward(mj_model, data)
    mujoco.mj_fullM(mj_model, M, data.qM)
    initial_qvel = np.linalg.solve(M[:qpos_dim, :qpos_dim], mom_init)

    # Forward simulation
    substeps = int(data_dt / dt)
    qpos_rec = [qpos_init.copy()]
    mom_rec = [mom_init.copy()]

    data.qpos[:qpos_dim] = qpos_init
    data.qvel[:qpos_dim] = initial_qvel
    mujoco.mj_forward(mj_model, data)

    for t in range(T - 1):
        data.ctrl[:qpos_dim] = torque[t]
        for _ in range(substeps):
            mujoco.mj_step(mj_model, data)

        qpos_rec.append(data.qpos[:qpos_dim].copy())
        # Compute momentum
        mujoco.mj_fullM(mj_model, M, data.qM)
        mom = M[:qpos_dim, :qpos_dim] @ data.qvel[:qpos_dim]
        mom_rec.append(mom.copy())

    return np.array(qpos_rec), np.array(mom_rec)


def find_good_sample(h5_file, length, qpos_dim, min_range=0.1):
    """Find a sample with good trajectory variation."""
    key = f'unguided/L{length}'
    if key not in h5_file:
        # Try alternative key format
        for k in h5_file.keys():
            if f'L{length}' in k:
                key = k
                break

    states = h5_file[f'{key}/state'][:]
    N = len(states)

    best_idx = 0
    best_range = 0

    for i in range(min(N, 100)):  # Check first 100 samples
        state = states[i]
        qpos = state[:, :qpos_dim]
        ranges = np.ptp(qpos, axis=0)
        min_r = ranges.min()
        if min_r > best_range:
            best_range = min_r
            best_idx = i

    return best_idx


def plot_showcase(policy, length, dof, sample_idx=None):
    """Create trajectory showcase plot."""
    cfg = CONFIGS[dof]
    qpos_dim = cfg['qpos_dim']
    mom_dim = cfg['mom_dim']
    state_dim = qpos_dim + mom_dim

    # Load MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path(str(cfg['mujoco_xml']))

    # Load H5 data
    h5_path = cfg['h5_dir'] / f'exp_a_{policy}.h5'
    if not h5_path.exists():
        print(f"Error: {h5_path} not found")
        return

    with h5py.File(h5_path, 'r') as f:
        # Find good sample if not specified
        if sample_idx is None:
            sample_idx = find_good_sample(f, length, qpos_dim)

        # Load unguided and guided data
        ung_state = f[f'unguided/L{length}/state'][sample_idx]
        ung_torque = f[f'unguided/L{length}/torque'][sample_idx]
        gui_state = f[f'guided/L{length}/state'][sample_idx]
        gui_torque = f[f'guided/L{length}/torque'][sample_idx]

    # Extract qpos and momentum
    ung_qpos = ung_state[:, :qpos_dim]
    ung_mom = ung_state[:, qpos_dim:]
    gui_qpos = gui_state[:, :qpos_dim]
    gui_mom = gui_state[:, qpos_dim:]

    # Reconstruct trajectories
    print(f"Reconstructing unguided trajectory...")
    ung_qpos_rec, ung_mom_rec = reconstruct_trajectory(
        ung_qpos[0], ung_mom[0], ung_torque, mj_model, cfg['dt'], cfg['data_dt']
    )
    print(f"Reconstructing guided trajectory...")
    gui_qpos_rec, gui_mom_rec = reconstruct_trajectory(
        gui_qpos[0], gui_mom[0], gui_torque, mj_model, cfg['dt'], cfg['data_dt']
    )

    # Create figure: 2 rows (qpos, mom) x 2 cols (unguided, guided) x dof subplots
    fig, axes = plt.subplots(2 * dof, 2, figsize=(12, 3 * dof))
    if dof == 1:
        axes = axes.reshape(2, 2)

    time = np.arange(length) * cfg['data_dt'] * 1000  # Convert to ms
    time_rec = np.arange(len(ung_qpos_rec)) * cfg['data_dt'] * 1000

    colors = {'gen': '#1f77b4', 'rec': '#ff7f0e'}

    for dim in range(qpos_dim):
        # Unguided qpos
        ax = axes[dim, 0]
        ax.plot(time, ung_qpos[:, dim], color=colors['gen'], label='DPF', linewidth=1.5)
        ax.plot(time_rec, ung_qpos_rec[:, dim], color=colors['rec'], linestyle='--', label='MuJoCo', linewidth=1.5)
        ax.set_ylabel(f'$q_{dim+1}$ (rad)')
        if dim == 0:
            ax.set_title('Unguided')
            ax.legend(loc='upper right')

        # Guided qpos
        ax = axes[dim, 1]
        ax.plot(time, gui_qpos[:, dim], color=colors['gen'], label='DPF', linewidth=1.5)
        ax.plot(time_rec, gui_qpos_rec[:, dim], color=colors['rec'], linestyle='--', label='MuJoCo', linewidth=1.5)
        if dim == 0:
            ax.set_title('Guided (HNN)')

    for dim in range(mom_dim):
        row = qpos_dim + dim
        # Unguided momentum
        ax = axes[row, 0]
        ax.plot(time, ung_mom[:, dim], color=colors['gen'], linewidth=1.5)
        ax.plot(time_rec, ung_mom_rec[:, dim], color=colors['rec'], linestyle='--', linewidth=1.5)
        ax.set_ylabel(f'$p_{dim+1}$')
        if dim == mom_dim - 1:
            ax.set_xlabel('Time (ms)')

        # Guided momentum
        ax = axes[row, 1]
        ax.plot(time, gui_mom[:, dim], color=colors['gen'], linewidth=1.5)
        ax.plot(time_rec, gui_mom_rec[:, dim], color=colors['rec'], linestyle='--', linewidth=1.5)
        if dim == mom_dim - 1:
            ax.set_xlabel('Time (ms)')

    plt.suptitle(f'{dof}DoF - {POLICY_LABELS[policy]} (Length={length})', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / f'{dof}DoF-{POLICY_LABELS[policy].replace(" ", "_")}_Length_{length}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='gp',
                        choices=['sinusoidal', 'gp', 'zero', 'spline'])
    parser.add_argument('--length', type=int, default=1000)
    parser.add_argument('--dof', type=int, default=3, choices=[2, 3])
    parser.add_argument('--sample_idx', type=int, default=None)
    args = parser.parse_args()

    plot_showcase(args.policy, args.length, args.dof, args.sample_idx)


if __name__ == '__main__':
    main()
