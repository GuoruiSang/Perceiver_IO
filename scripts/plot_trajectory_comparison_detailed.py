"""
Plot DPF-generated vs MuJoCo-reconstructed trajectories.

Creates 8x6 figure (4 lengths × 2 guidance modes × 6 state dimensions).

Usage:
    python scripts/plot_trajectory_comparison_detailed.py --policy sinusoidal
    python scripts/plot_trajectory_comparison_detailed.py --policy gp
    python scripts/plot_trajectory_comparison_detailed.py --policy zero
    python scripts/plot_trajectory_comparison_detailed.py --policy spline
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

from src.models.utils import reconstruct_traj_with_momentum
from src.models.HNN import HNNWrapper


# Configuration
DT = 0.0001  # Fine simulation timestep
DATA_DT = 0.0002  # Data collection timestep
LENGTHS = [250, 500, 750, 1000]
QPOS_DIM = 3
MOM_DIM = 3
STATE_DIM = QPOS_DIM + MOM_DIM

# Paths
MUJOCO_XML = project_root / 'configs' / 'rigid_arm_hinge.xml'
HNN_CHECKPOINT = project_root / 'checkpoints' / 'SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt'
H5_DIR = project_root / 'output_ablation' / 'trajectories' / 'original'
PLOTS_DIR = project_root / 'plots'

# Policy to H5 file mapping
POLICY_TO_H5 = {
    'sinusoidal': 'exp_a_sinusoidal.h5',
    'gp': 'exp_a_gp.h5',
    'zero': 'exp_a_zero.h5',
    'spline': 'exp_a_training.h5',  # Training data has spline torques
}

# ICLR-friendly style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
})


def compute_mse(generated: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute MSE between generated and reconstructed trajectories."""
    # Align lengths (reconstructed is 1 shorter due to integration)
    min_len = min(len(generated) - 1, len(reconstructed))
    gen = generated[1:min_len+1]
    rec = reconstructed[:min_len]
    return np.mean((gen - rec) ** 2)


def compute_hamres(qpos: np.ndarray, mom: np.ndarray, torque: np.ndarray,
                   hnn: HNNWrapper, var_dq: float, var_dp: float,
                   dt: float = DATA_DT) -> float:
    """Compute HamRes for a trajectory."""
    T = len(qpos)
    if T < 3:
        return float('nan')

    eps = 1e-8
    device = next(hnn.parameters()).device

    # Finite difference for velocities
    qdot = (qpos[2:] - qpos[:-2]) / (2 * dt)
    pdot = (mom[2:] - mom[:-2]) / (2 * dt)

    q_mid = qpos[1:-1]
    p_mid = mom[1:-1]
    tau_mid = torque[1:-1]

    # Convert to tensors
    q_t = torch.tensor(q_mid, dtype=torch.float32, device=device).requires_grad_(True)
    p_t = torch.tensor(p_mid, dtype=torch.float32, device=device).requires_grad_(True)

    # Compute HNN gradients
    with torch.enable_grad():
        H = hnn(p_t, q_t)
        dH_dp, dH_dq = torch.autograd.grad(H.sum(), (p_t, q_t))

    dH_dp = dH_dp.detach().cpu().numpy()
    dH_dq = dH_dq.detach().cpu().numpy()

    # Residuals
    r_q = qdot - dH_dp
    r_p = pdot - (-dH_dq + tau_mid)

    mse_q = np.mean(r_q ** 2)
    mse_p = np.mean(r_p ** 2)

    hamres = mse_q / (var_dq + eps) + mse_p / (var_dp + eps)
    return hamres


def find_p10_sample_idx(h5_file, length: int, mj_model, min_y_range: float = 0.1) -> int:
    """Find sample index with unguided MSE closest to 10th percentile (top 10%).

    Only considers samples where trajectory variation > min_y_range.
    """
    key = f'unguided/L{length}'
    states = h5_file[f'{key}/state'][:]  # [N, L, 6]
    torques = h5_file[f'{key}/torque'][:]  # [N, L, 3]

    N = len(states)
    mses = []
    y_ranges = []  # Track trajectory variation for each sample

    # Compute MSE and y-range for all unguided samples
    for i in range(N):
        state = states[i]
        torque = torques[i]
        qpos_gen = state[:, :QPOS_DIM]
        mom_gen = state[:, QPOS_DIM:]

        # Compute trajectory y-range (min range across all dimensions)
        qpos_ranges = np.ptp(qpos_gen, axis=0)  # peak-to-peak for each dim
        mom_ranges = np.ptp(mom_gen, axis=0)
        min_range = min(qpos_ranges.min(), mom_ranges.min())
        y_ranges.append(min_range)

        # Compute initial velocity from momentum
        M = np.zeros((mj_model.nv, mj_model.nv))
        data = mujoco.MjData(mj_model)
        data.qpos[:] = qpos_gen[0]
        data.qvel[:] = 0
        mujoco.mj_forward(mj_model, data)
        mujoco.mj_fullM(mj_model, M, data.qM)
        initial_qvel = np.linalg.solve(M, mom_gen[0])

        # Reconstruct
        recon = reconstruct_traj_with_momentum(
            mj_model, length, DT, qpos_gen[0], initial_qvel, torque, data_dt=DATA_DT
        )

        # Compute MSE
        mse_qpos = compute_mse(qpos_gen, recon['seq_qpos'])
        mse_mom = compute_mse(mom_gen, recon['seq_mom'])
        mses.append(mse_qpos + mse_mom)

    mses = np.array(mses)
    y_ranges = np.array(y_ranges)

    # Filter samples with sufficient trajectory variation
    valid_mask = y_ranges >= min_y_range
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        print(f"    Warning: No samples with y_range >= {min_y_range}, using all samples")
        valid_indices = np.arange(N)

    # Among valid samples, find P10
    valid_mses = mses[valid_indices]
    p10_mse = np.percentile(valid_mses, 10)  # Top 10%

    # Return index closest to 10th percentile among valid samples
    best_valid_idx = np.argmin(np.abs(valid_mses - p10_mse))
    return valid_indices[best_valid_idx]


def get_sample_by_idx(h5_file, guidance: str, length: int, idx: int) -> tuple:
    """Get state and torque for a specific sample index."""
    key = f'{guidance}/L{length}'
    state = h5_file[f'{key}/state'][idx]
    torque = h5_file[f'{key}/torque'][idx]
    return state, torque


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True,
                        choices=['sinusoidal', 'gp', 'zero', 'spline'])
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Creating comparison plots for policy: {args.policy}")

    # Load MuJoCo model
    print("Loading MuJoCo model...")
    mj_model = mujoco.MjModel.from_xml_path(str(MUJOCO_XML))

    # Load HNN for HamRes computation
    print("Loading HNN model...")
    hnn = HNNWrapper.load_from_checkpoint(str(HNN_CHECKPOINT), map_location=device)
    hnn.eval()
    hnn.to(device)
    var_dq = hnn.qvel_var.mean().item()
    var_dp = hnn.mom_dot_var.mean().item()

    # Load H5 file
    h5_path = H5_DIR / POLICY_TO_H5[args.policy]
    print(f"Loading trajectories from: {h5_path}")

    # Create figure: 8 rows × 6 columns
    fig, axes = plt.subplots(8, 6, figsize=(18, 16))

    col_labels = ['$q_0$', '$q_1$', '$q_2$', '$p_0$', '$p_1$', '$p_2$']

    with h5py.File(h5_path, 'r') as f:
        row_idx = 0
        for length in LENGTHS:
            # Find P10 sample index based on UNGUIDED MSE (top 10%)
            print(f"\nFinding P10 sample for L={length}...")
            sample_idx = find_p10_sample_idx(f, length, mj_model)
            print(f"  Selected sample index: {sample_idx}")

            # Store data for y-limit synchronization
            row_data = {}

            for guidance in ['unguided', 'guided']:
                print(f"  Processing {guidance}...")

                # Use the SAME sample index for both unguided and guided
                state, torque = get_sample_by_idx(f, guidance, length, sample_idx)

                qpos_gen = state[:, :QPOS_DIM]
                mom_gen = state[:, QPOS_DIM:]

                # Reconstruct via MuJoCo
                M = np.zeros((mj_model.nv, mj_model.nv))
                data = mujoco.MjData(mj_model)
                data.qpos[:] = qpos_gen[0]
                data.qvel[:] = 0
                mujoco.mj_forward(mj_model, data)
                mujoco.mj_fullM(mj_model, M, data.qM)
                initial_qvel = np.linalg.solve(M, mom_gen[0])

                recon = reconstruct_traj_with_momentum(
                    mj_model, length, DT, qpos_gen[0], initial_qvel, torque, data_dt=DATA_DT
                )

                qpos_rec = recon['seq_qpos']
                mom_rec = recon['seq_mom']

                # Compute metrics
                nmse_qpos = compute_mse(qpos_gen, qpos_rec) / np.var(qpos_gen)
                nmse_mom = compute_mse(mom_gen, mom_rec) / np.var(mom_gen)
                nmse = (nmse_qpos + nmse_mom) / 2

                hamres = compute_hamres(qpos_gen, mom_gen, torque, hnn, var_dq, var_dp)

                # Set row background
                bg_color = '#f0f0f0' if guidance == 'unguided' else 'white'

                # Time axis
                t_gen = np.arange(len(qpos_gen)) * DATA_DT
                t_rec = np.arange(len(qpos_rec)) * DATA_DT

                # Store for y-limit sync
                row_data[guidance] = {
                    'qpos_gen': qpos_gen, 'mom_gen': mom_gen,
                    'qpos_rec': qpos_rec, 'mom_rec': mom_rec,
                    't_gen': t_gen, 't_rec': t_rec,
                    'nmse': nmse, 'hamres': hamres,
                    'bg_color': bg_color, 'row_idx': row_idx
                }

                # Plot each dimension
                for col_idx in range(6):
                    ax = axes[row_idx, col_idx]
                    ax.set_facecolor(bg_color)

                    if col_idx < QPOS_DIM:
                        gen_data = qpos_gen[:, col_idx]
                        rec_data = qpos_rec[:, col_idx]
                    else:
                        gen_data = mom_gen[:, col_idx - QPOS_DIM]
                        rec_data = mom_rec[:, col_idx - QPOS_DIM]

                    # Plot with dots instead of lines
                    label_gen = 'Guided' if guidance == 'guided' else 'Unguided'
                    ax.scatter(t_gen, gen_data, c='blue', s=1, label=label_gen, alpha=0.7)
                    ax.scatter(t_rec, rec_data, c='red', s=1, label='MuJoCo', alpha=0.7)

                    # Styling
                    if row_idx == 0:
                        ax.set_title(col_labels[col_idx])
                    if col_idx == 0:
                        ax.set_ylabel(f'L={length}\n{guidance.capitalize()}', fontsize=9)
                    if row_idx == 7:
                        ax.set_xlabel('Time (s)')

                    # Legend only on first subplot of each row
                    if col_idx == 0:
                        ax.legend(loc='upper right', fontsize=6)

                    ax.tick_params(labelsize=7)

                # Add metrics annotation on the right side
                ax_right = axes[row_idx, -1]
                ax_right.annotate(
                    f'NMSE={nmse:.4f}\nHamRes={hamres:.2f}',
                    xy=(1.02, 0.5), xycoords='axes fraction',
                    fontsize=8, va='center', ha='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

                row_idx += 1

            # Synchronize y-limits for unguided/guided pair (same length)
            unguided_row = row_data['unguided']['row_idx']
            guided_row = row_data['guided']['row_idx']
            for col_idx in range(6):
                ax_ung = axes[unguided_row, col_idx]
                ax_gui = axes[guided_row, col_idx]
                ymin = min(ax_ung.get_ylim()[0], ax_gui.get_ylim()[0])
                ymax = max(ax_ung.get_ylim()[1], ax_gui.get_ylim()[1])
                ax_ung.set_ylim(ymin, ymax)
                ax_gui.set_ylim(ymin, ymax)

    # Overall title
    policy_labels = {
        'sinusoidal': 'Sinusoidal Torque',
        'gp': 'GP Torque',
        'zero': 'Zero Torque',
        'spline': 'Cubic Spline Torque'
    }
    fig.suptitle(f'DPF vs MuJoCo Reconstruction - {policy_labels[args.policy]}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 0.95, 0.96])

    # Save
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / f'traj_comparison_{args.policy}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
