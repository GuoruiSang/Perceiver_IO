"""
Plot saved trajectory samples: generated vs MuJoCo reconstruction.

For each experiment file, randomly picks samples and runs MuJoCo forward
simulation to get the physics-based reconstruction. Produces a 6-row figure:
  Row 0: Unguided torque
  Row 1: Unguided position (generated vs reconstructed)
  Row 2: Unguided momentum
  Row 3: Guided torque
  Row 4: Guided position
  Row 5: Guided momentum

This lets you visually compare physical consistency with and without guidance.

Usage:
    python scripts/plot_saved_trajectory_comparison.py \
        --model_name torque_concat \
        --traj_length 500
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import h5py
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.utils import reconstruct_traj_with_momentum


# ICLR-friendly style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

COLOR_GEN = '#1f77b4'    # blue - generated
COLOR_RECON = '#d62728'  # red - reconstructed
COLOR_TORQUE = '#2ca02c' # green - torque

XML_PATH = str(project_root / 'configs' / 'rigid_arm_hinge.xml')

EXPERIMENT_DISPLAY = {
    'exp_a_training': 'Training Torques',
    'exp_a_sinusoidal': 'Sinusoidal Torques',
    'exp_a_gp': 'GP Torques',
    'exp_a_zero': 'Zero Torques',
    'exp_b_context_fractions': 'Context Fractions',
}


def reconstruct_single(state_np, torque_np, qpos_dim=3, dt=0.0001, data_dt=0.00025):
    """Run MuJoCo reconstruction for a single trajectory.

    Returns:
        recon_qpos: [T, qpos_dim] (full length, initial state prepended)
        recon_mom:  [T, mom_dim]
        mse_qpos, mse_mom: scalar MSE values
    """
    qpos_gen = state_np[:, :qpos_dim]
    mom_gen = state_np[:, qpos_dim:]

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Initial velocity from initial momentum: v = M^{-1} p
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

    # MSE: generated[1:] vs reconstructed
    mse_qpos = np.mean((qpos_gen[1:] - recon['seq_qpos']) ** 2)
    mse_mom = np.mean((mom_gen[1:] - recon['seq_mom']) ** 2)

    # Prepend initial state for full-length alignment
    recon_qpos = np.concatenate([qpos_gen[:1], recon['seq_qpos']], axis=0)
    recon_mom = np.concatenate([mom_gen[:1], recon['seq_mom']], axis=0)

    return recon_qpos, recon_mom, mse_qpos, mse_mom


def plot_gen_vs_recon(
    ung_state, ung_torque, guid_state, guid_torque,
    title, output_path, qpos_dim=3, dt_sim=0.0001, data_dt=0.00025,
):
    """Plot generated vs reconstructed for both unguided and guided.

    6 rows x 3 cols:
      Row 0: Unguided torque (tau1, tau2, tau3)
      Row 1: Unguided position (q1, q2, q3)
      Row 2: Unguided momentum (p1, p2, p3)
      Row 3: Guided torque
      Row 4: Guided position
      Row 5: Guided momentum
    """
    # Reconstruct both
    ung_recon_q, ung_recon_p, ung_mse_q, ung_mse_p = reconstruct_single(
        ung_state, ung_torque, qpos_dim, dt_sim, data_dt)
    guid_recon_q, guid_recon_p, guid_mse_q, guid_mse_p = reconstruct_single(
        guid_state, guid_torque, qpos_dim, dt_sim, data_dt)

    T = ung_state.shape[0]
    t_axis = np.arange(T) * data_dt

    fig, axes = plt.subplots(6, 3, figsize=(10, 13), sharex=True)

    labels_tau = [r'$\tau_1$', r'$\tau_2$', r'$\tau_3$']
    labels_q = [r'$q_1$', r'$q_2$', r'$q_3$']
    labels_p = [r'$p_1$', r'$p_2$', r'$p_3$']

    def _plot_state_row(row, gen_data, recon_data, dim_labels):
        for j in range(3):
            ax = axes[row, j]
            ax.plot(t_axis, gen_data[:, j], color=COLOR_GEN,
                    linewidth=0.8, alpha=0.85, label='Generated')
            ax.plot(t_axis, recon_data[:, j], color=COLOR_RECON,
                    linewidth=0.8, alpha=0.85, linestyle='--', label='Reconstructed')
            ax.set_ylabel(dim_labels[j])
            ax.grid(True)

    def _plot_torque_row(row, torque_data, dim_labels):
        for j in range(3):
            ax = axes[row, j]
            ax.plot(t_axis, torque_data[:, j], color=COLOR_TORQUE,
                    linewidth=0.8, alpha=0.85)
            ax.set_ylabel(dim_labels[j])
            ax.grid(True)

    # Row 0: Unguided torque
    _plot_torque_row(0, ung_torque, labels_tau)
    # Row 1: Unguided position
    _plot_state_row(1, ung_state[:, :qpos_dim], ung_recon_q, labels_q)
    # Row 2: Unguided momentum
    _plot_state_row(2, ung_state[:, qpos_dim:], ung_recon_p, labels_p)
    # Row 3: Guided torque
    _plot_torque_row(3, guid_torque, labels_tau)
    # Row 4: Guided position
    _plot_state_row(4, guid_state[:, :qpos_dim], guid_recon_q, labels_q)
    # Row 5: Guided momentum
    _plot_state_row(5, guid_state[:, qpos_dim:], guid_recon_p, labels_p)

    # x-axis labels on bottom row only
    for j in range(3):
        axes[5, j].set_xlabel('Time (s)')

    # Row annotations
    ung_mse_total = ung_mse_q + ung_mse_p
    guid_mse_total = guid_mse_q + guid_mse_p

    axes[0, 0].set_title(
        'Unguided Torque',
        fontweight='bold', loc='left', fontsize=10)
    axes[1, 0].set_title(
        f'Unguided Position  (MSE q={ung_mse_q:.2e})',
        fontweight='bold', loc='left', fontsize=10)
    axes[2, 0].set_title(
        f'Unguided Momentum  (MSE p={ung_mse_p:.2e}, total={ung_mse_total:.2e})',
        fontweight='bold', loc='left', fontsize=10)
    axes[3, 0].set_title(
        'Guided Torque',
        fontweight='bold', loc='left', fontsize=10)
    axes[4, 0].set_title(
        f'Guided Position  (MSE q={guid_mse_q:.2e})',
        fontweight='bold', loc='left', fontsize=10)
    axes[5, 0].set_title(
        f'Guided Momentum  (MSE p={guid_mse_p:.2e}, total={guid_mse_total:.2e})',
        fontweight='bold', loc='left', fontsize=10)

    # Legend for state rows
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.01), framealpha=0.9)

    fig.suptitle(title, y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    print(f'  Saved: {output_path}  '
          f'[ung MSE={ung_mse_total:.2e}, guid MSE={guid_mse_total:.2e}]')


def main():
    parser = argparse.ArgumentParser(
        description='Plot generated vs MuJoCo reconstruction (unguided & guided)')
    parser.add_argument('--input_dir', type=str,
                        default='output_ablation/trajectories')
    parser.add_argument('--output_dir', type=str,
                        default='output_ablation/plots')
    parser.add_argument('--model_name', type=str, default='torque_concat')
    parser.add_argument('--traj_length', type=int, default=500,
                        help='Trajectory length for exp_a files')
    parser.add_argument('--context_fraction', type=float, default=0.50,
                        help='Context fraction for exp_b file')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of random samples to plot per file')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--qpos_dim', type=int, default=3)
    parser.add_argument('--dt', type=float, default=0.0001,
                        help='MuJoCo simulation timestep')
    parser.add_argument('--data_dt', type=float, default=0.00025,
                        help='Data collection timestep')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    input_dir = project_root / args.input_dir / args.model_name
    output_dir = project_root / args.output_dir / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Model: {args.model_name}')
    print(f'Input: {input_dir}')
    print(f'Output: {output_dir}')

    all_h5 = sorted(input_dir.glob('*.h5'))
    print(f'Found {len(all_h5)} H5 files: {[f.name for f in all_h5]}')

    for h5_path in all_h5:
        exp_name = h5_path.stem
        is_exp_b = 'exp_b' in exp_name

        if is_exp_b:
            group_key = f'cf_{args.context_fraction:.2f}'
        else:
            group_key = f'L{args.traj_length}'

        display = EXPERIMENT_DISPLAY.get(exp_name, exp_name)
        print(f'\n{exp_name} ({group_key}):')

        with h5py.File(str(h5_path), 'r') as f:
            ung_state_key = f'unguided/{group_key}/state'
            ung_torque_key = f'unguided/{group_key}/torque'
            guid_state_key = f'guided/{group_key}/state'
            guid_torque_key = f'guided/{group_key}/torque'

            for key in [ung_state_key, ung_torque_key,
                        guid_state_key, guid_torque_key]:
                if key not in f:
                    print(f'  Skipping: {key} not found')
                    break
            else:
                n_available = f[ung_state_key].shape[0]
                indices = rng.choice(n_available, size=args.num_samples,
                                     replace=False)

                for sample_idx in indices:
                    ung_state = f[ung_state_key][sample_idx]
                    ung_torque = f[ung_torque_key][sample_idx]
                    guid_state = f[guid_state_key][sample_idx]
                    guid_torque = f[guid_torque_key][sample_idx]

                    title = f'{display} — sample #{sample_idx} ({group_key})'
                    out_path = (output_dir /
                                f'{exp_name}_{group_key}_sample{sample_idx}.png')

                    plot_gen_vs_recon(
                        ung_state, ung_torque,
                        guid_state, guid_torque,
                        title, str(out_path),
                        qpos_dim=args.qpos_dim,
                        dt_sim=args.dt,
                        data_dt=args.data_dt,
                    )

    print('\nDone.')


if __name__ == '__main__':
    main()
