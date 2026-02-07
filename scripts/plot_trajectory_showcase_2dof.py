"""
Plot 2DoF trajectory showcase: DPF vs MuJoCo reconstruction.
Format: Torque row + alternating Unguided/Guided rows with gray/white backgrounds.

Usage:
    python scripts/plot_trajectory_showcase_2dof.py --policy sinusoidal
    python scripts/plot_trajectory_showcase_2dof.py --all
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

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'axes.grid': False,
})

# Config
MUJOCO_XML = project_root / 'configs' / 'rigid_arm_hinge_2dof.xml'
H5_DIR = project_root / 'output_ablation' / 'results' / 'trajectories' / '2dof_smoothed'
PLOTS_DIR = project_root / 'plots'
DT = 0.0001
DATA_DT = 0.0002
QPOS_DIM = 2
LENGTH = 1000
NUM_SAMPLES = 2

POLICY_LABELS = {
    'sinusoidal': 'Sinusoidal',
    'gp': 'GP',
    'zero': 'Zero',
    'spline': 'Cubic Spline',
}


def reconstruct_trajectory(qpos_init, mom_init, torque, mj_model):
    """Reconstruct trajectory using MuJoCo forward simulation."""
    T = len(torque)
    mj_model.opt.timestep = DT  # Must set explicitly (XML default is 0.002)
    M = np.zeros((mj_model.nv, mj_model.nv))
    data = mujoco.MjData(mj_model)
    data.qpos[:QPOS_DIM] = qpos_init
    data.qvel[:] = 0
    mujoco.mj_forward(mj_model, data)
    mujoco.mj_fullM(mj_model, M, data.qM)
    initial_qvel = np.linalg.solve(M[:QPOS_DIM, :QPOS_DIM], mom_init)

    substeps = int(DATA_DT / DT)
    qpos_rec = [qpos_init.copy()]
    mom_rec = [mom_init.copy()]

    data.qpos[:QPOS_DIM] = qpos_init
    data.qvel[:QPOS_DIM] = initial_qvel
    mujoco.mj_forward(mj_model, data)

    for t in range(T - 1):
        data.ctrl[:QPOS_DIM] = torque[t]
        for _ in range(substeps):
            mujoco.mj_step(mj_model, data)
        qpos_rec.append(data.qpos[:QPOS_DIM].copy())
        mujoco.mj_fullM(mj_model, M, data.qM)
        mom = M[:QPOS_DIM, :QPOS_DIM] @ data.qvel[:QPOS_DIM]
        mom_rec.append(mom.copy())

    return np.array(qpos_rec), np.array(mom_rec)


def compute_mse(gen, rec):
    """Compute MSE between generated and reconstructed."""
    T = min(len(gen), len(rec))
    return np.mean((gen[:T] - rec[:T]) ** 2)


def find_best_samples(h5_file, length, mj_model, num_samples=2):
    """Find samples where unguided is bad and guided is good."""
    ung_states = h5_file[f'unguided/L{length}/state'][:]
    gui_states = h5_file[f'guided/L{length}/state'][:]
    ung_torques = h5_file[f'unguided/L{length}/torque'][:]
    gui_torques = h5_file[f'guided/L{length}/torque'][:]
    N = len(ung_states)

    candidates = []
    for i in range(min(N, 50)):  # Check first 50
        # Unguided MSE
        ung_qpos = ung_states[i][:, :QPOS_DIM]
        ung_mom = ung_states[i][:, QPOS_DIM:]
        ung_qpos_rec, ung_mom_rec = reconstruct_trajectory(
            ung_qpos[0], ung_mom[0], ung_torques[i], mj_model)
        ung_mse = compute_mse(ung_qpos, ung_qpos_rec) + compute_mse(ung_mom, ung_mom_rec)

        # Guided MSE
        gui_qpos = gui_states[i][:, :QPOS_DIM]
        gui_mom = gui_states[i][:, QPOS_DIM:]
        gui_qpos_rec, gui_mom_rec = reconstruct_trajectory(
            gui_qpos[0], gui_mom[0], gui_torques[i], mj_model)
        gui_mse = compute_mse(gui_qpos, gui_qpos_rec) + compute_mse(gui_mom, gui_mom_rec)

        candidates.append((i, ung_mse, gui_mse))

    # Sort by improvement ratio (ung_mse / gui_mse), highest first
    # This finds samples where guided improves over unguided the most
    candidates_with_ratio = [(i, u, g, u/g if g > 1e-10 else 1e6) for i, u, g in candidates]
    candidates_with_ratio.sort(key=lambda x: x[3], reverse=True)

    selected = candidates_with_ratio[:num_samples]

    for idx, ung, gui, ratio in selected:
        print(f"    Sample {idx}: ung_mse={ung:.6f}, gui_mse={gui:.6f}, ratio={ratio:.1f}x")

    return [x[0] for x in selected]


def plot_showcase(policy):
    """Create 2DoF trajectory showcase plot."""
    print(f"Generating showcase for {policy}...")

    mj_model = mujoco.MjModel.from_xml_path(str(MUJOCO_XML))

    h5_path = H5_DIR / f'exp_a_{policy}.h5'
    if not h5_path.exists():
        print(f"Error: {h5_path} not found")
        return

    with h5py.File(h5_path, 'r') as f:
        sample_indices = find_best_samples(f, LENGTH, mj_model, NUM_SAMPLES)
        print(f"  Selected samples: {sample_indices}")

        samples_data = []
        for idx in sample_indices:
            samples_data.append({
                'ung_state': f[f'unguided/L{LENGTH}/state'][idx],
                'ung_torque': f[f'unguided/L{LENGTH}/torque'][idx],
                'gui_state': f[f'guided/L{LENGTH}/state'][idx],
                'gui_torque': f[f'guided/L{LENGTH}/torque'][idx],
            })

    # Create figure: 5 rows x 4 cols
    # Row 0: Torque (2 subplots, each spanning 2 cols)
    # Rows 1-4: q₀, q₁, p₀, p₁
    fig = plt.figure(figsize=(14, 12))

    # Use GridSpec for flexible layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(5, 4, figure=fig, hspace=0.35, wspace=0.3)

    time = np.arange(LENGTH)
    colors = {'dpf': '#1f77b4', 'mujoco': '#d62728'}

    # Row 0: Torque (τ₀ spans cols 0-1, τ₁ spans cols 2-3)
    torque = samples_data[0]['gui_torque']
    for dim in range(QPOS_DIM):
        ax = fig.add_subplot(gs[0, dim*2:(dim+1)*2])
        ax.plot(time, torque[:, dim], color='black', linewidth=1.2)
        ax.set_title(rf'$\tau_{dim}$', fontsize=14)
        if dim == 0:
            ax.set_ylabel('Torque', fontsize=12)
        ax.set_xlim(0, LENGTH)

    # Sample rows
    row_configs = [
        (1, 0, 'ung', 'Sample 1\nUnguided', True),   # gray
        (2, 0, 'gui', 'Sample 1\nGuided', False),    # white
        (3, 1, 'ung', 'Sample 2\nUnguided', True),   # gray
        (4, 1, 'gui', 'Sample 2\nGuided', False),    # white
    ]

    for row_idx, sample_idx, mode, ylabel, gray_bg in row_configs:
        sample = samples_data[sample_idx]
        state = sample[f'{mode}_state']
        torque = sample[f'{mode}_torque']

        qpos = state[:, :QPOS_DIM]
        mom = state[:, QPOS_DIM:]
        qpos_rec, mom_rec = reconstruct_trajectory(qpos[0], mom[0], torque, mj_model)
        time_rec = np.arange(len(qpos_rec))

        # Plot q₀, q₁, p₀, p₁
        labels = [rf'$q_0$', rf'$q_1$', rf'$p_0$', rf'$p_1$']
        data_pairs = [
            (qpos[:, 0], qpos_rec[:, 0]),
            (qpos[:, 1], qpos_rec[:, 1]),
            (mom[:, 0], mom_rec[:, 0]),
            (mom[:, 1], mom_rec[:, 1]),
        ]

        for col_idx, (gen, rec) in enumerate(data_pairs):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            if gray_bg:
                ax.set_facecolor('#E8E8E8')

            label_dpf = 'Unguided' if mode == 'ung' else 'Guided'
            ax.plot(time[:len(gen)], gen, color=colors['dpf'], linewidth=1,
                    label=label_dpf if col_idx == 0 else None, marker='.', markersize=1)
            ax.plot(time_rec, rec, color=colors['mujoco'], linewidth=1,
                    label='MuJoCo' if col_idx == 0 else None)

            if row_idx == 1:
                ax.set_title(labels[col_idx], fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=11)
            if row_idx == 4:
                ax.set_xlabel('Trajectory Length', fontsize=10)

            ax.set_xlim(0, LENGTH)
            if col_idx == 0:
                ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    plt.suptitle(f'2DoF-{POLICY_LABELS[policy]} Torque Policy (Length={LENGTH})', fontsize=16, y=0.98)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / f'traj_showcase_2dof_{policy}.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='sinusoidal',
                        choices=['sinusoidal', 'gp', 'zero', 'spline'])
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        for policy in ['sinusoidal', 'gp', 'zero', 'spline']:
            plot_showcase(policy)
    else:
        plot_showcase(args.policy)


if __name__ == '__main__':
    main()
