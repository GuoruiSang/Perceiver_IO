"""
Plot DPF-generated vs MuJoCo-reconstructed trajectories with LIVE generation.

Uses specified guidance_lr instead of pre-generated data.

Usage:
    python scripts/plot_trajectory_comparison_live.py --policy sinusoidal --guidance_lr 0.001
    python scripts/plot_trajectory_comparison_live.py --policy gp --guidance_lr 0.001
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

from src.models.utils import reconstruct_traj_with_momentum, EMA
from src.models.HNN import HNNWrapper
from src.models.trajectory_dpf import TrajectoryDPF


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
DPF_CHECKPOINT = project_root / 'checkpoints' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt'
PLOTS_DIR = project_root / 'plots'

# Torque data paths
TORQUE_PATHS = {
    'sinusoidal': project_root / 'data' / 'sinusoidal_torques_1000_L1500.h5',
    'gp': project_root / 'data' / 'gp_torques_1000_L1500.h5',
    'zero': None,  # Will create zero torques
    'spline': project_root / 'data' / 'training_torques_1000_L1500.h5',
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

    qdot = (qpos[2:] - qpos[:-2]) / (2 * dt)
    pdot = (mom[2:] - mom[:-2]) / (2 * dt)

    q_mid = qpos[1:-1]
    p_mid = mom[1:-1]
    tau_mid = torque[1:-1]

    q_t = torch.tensor(q_mid, dtype=torch.float32, device=device).requires_grad_(True)
    p_t = torch.tensor(p_mid, dtype=torch.float32, device=device).requires_grad_(True)

    with torch.enable_grad():
        H = hnn(p_t, q_t)
        dH_dp, dH_dq = torch.autograd.grad(H.sum(), (p_t, q_t))

    dH_dp = dH_dp.detach().cpu().numpy()
    dH_dq = dH_dq.detach().cpu().numpy()

    r_q = qdot - dH_dp
    r_p = pdot - (-dH_dq + tau_mid)

    mse_q = np.mean(r_q ** 2)
    mse_p = np.mean(r_p ** 2)

    hamres = mse_q / (var_dq + eps) + mse_p / (var_dp + eps)
    return hamres


def load_model(checkpoint_path, hnn_checkpoint_path, device):
    """Load DPF model with EMA + HNN."""
    print(f"Loading DPF model from: {checkpoint_path}")
    model = TrajectoryDPF.load_from_checkpoint(str(checkpoint_path), map_location=device, strict=False)
    model = model.to(device)
    model.eval()

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    ema_shadow = checkpoint.get('ema_shadow', None)
    ema_decay = checkpoint.get('ema_decay', 0.9995)
    if ema_shadow is not None:
        model.ema = EMA(model.model, decay=ema_decay)
        for name, tensor in ema_shadow.items():
            if name in model.ema.shadow:
                model.ema.shadow[name] = tensor.to(device=device, dtype=model.ema.shadow[name].dtype)
        model._ema_loaded = True
        print("  EMA shadow loaded")

    print(f"Loading HNN from: {hnn_checkpoint_path}")
    hnn = HNNWrapper.load_from_checkpoint(str(hnn_checkpoint_path), map_location=device)
    hnn = hnn.to(device)
    hnn.eval()

    return model, hnn


def load_torques(policy: str, device: torch.device, num_samples: int = 100, max_length: int = 1500):
    """Load torque data for the given policy."""
    torque_path = TORQUE_PATHS[policy]

    if torque_path is None:
        # Zero torques
        print(f"Creating zero torques: [{num_samples}, {max_length}, 3]")
        return torch.zeros(num_samples, max_length, 3, device=device)

    print(f"Loading torques from: {torque_path}")
    with h5py.File(torque_path, 'r') as f:
        torques = torch.tensor(f['torques'][:num_samples], dtype=torch.float32, device=device)
        print(f"  Shape: {torques.shape}")
    return torques


def generate_trajectories(model, hnn, torques, device,
                          trajectory_length, num_samples, seed,
                          guided, guidance_lr):
    """Generate trajectories with specified guidance_lr."""
    traj_torques = torques[:num_samples, :trajectory_length, :]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if guided and hnn is not None:
        guidance_kwargs = dict(
            hnn=hnn,
            guidance_method='adam',
            guidance_steps=25,
            guidance_lr=guidance_lr,  # Use specified LR
            guidance_after_steps=45,
            lambda_init=0.0,
        )
    else:
        guidance_kwargs = dict(hnn=None, guidance_steps=0)

    state, torque_out = model.sample_trajectories(
        num_samples=num_samples,
        trajectory_length=trajectory_length,
        num_diffusion_steps=50,
        context_fraction=0.2,
        use_ema=False,
        sampler='ddim',
        guidance_scale=1.0,
        torque=traj_torques,
        **guidance_kwargs,
    )

    return state.cpu().numpy(), torque_out.cpu().numpy()


def find_p10_sample_idx_live(model, hnn, torques, device, length, mj_model,
                              guidance_lr, num_samples=50, min_y_range=0.1, seed=228):
    """Find sample index with unguided MSE closest to P10 among samples with sufficient variation."""
    print(f"    Generating {num_samples} unguided samples for P10 selection...")

    # Generate unguided trajectories
    states, torques_out = generate_trajectories(
        model, hnn, torques, device,
        length, num_samples, seed,
        guided=False, guidance_lr=guidance_lr
    )

    mses = []
    y_ranges = []

    for i in range(num_samples):
        state = states[i]
        torque = torques_out[i]
        qpos_gen = state[:, :QPOS_DIM]
        mom_gen = state[:, QPOS_DIM:]

        # Compute trajectory y-range
        qpos_ranges = np.ptp(qpos_gen, axis=0)
        mom_ranges = np.ptp(mom_gen, axis=0)
        min_range = min(qpos_ranges.min(), mom_ranges.min())
        y_ranges.append(min_range)

        # Compute MSE via MuJoCo reconstruction
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

        mse_qpos = compute_mse(qpos_gen, recon['seq_qpos'])
        mse_mom = compute_mse(mom_gen, recon['seq_mom'])
        mses.append(mse_qpos + mse_mom)

    mses = np.array(mses)
    y_ranges = np.array(y_ranges)

    # Filter samples with sufficient variation
    valid_mask = y_ranges >= min_y_range
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        print(f"    Warning: No samples with y_range >= {min_y_range}, using all")
        valid_indices = np.arange(num_samples)

    valid_mses = mses[valid_indices]
    p10_mse = np.percentile(valid_mses, 10)
    best_valid_idx = np.argmin(np.abs(valid_mses - p10_mse))
    selected_idx = valid_indices[best_valid_idx]

    print(f"    Selected sample {selected_idx} (y_range={y_ranges[selected_idx]:.3f}, MSE={mses[selected_idx]:.6f})")
    return selected_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True,
                        choices=['sinusoidal', 'gp', 'zero', 'spline'])
    parser.add_argument('--guidance_lr', type=float, default=0.001,
                        help='Guidance learning rate (default: 0.001)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_samples_per_length', type=int, default=50,
                        help='Number of samples to generate per length for P10 selection')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Creating comparison plots for policy: {args.policy}")
    print(f"Guidance LR: {args.guidance_lr}")

    # Load MuJoCo model
    print("Loading MuJoCo model...")
    mj_model = mujoco.MjModel.from_xml_path(str(MUJOCO_XML))

    # Load DPF and HNN models
    model, hnn = load_model(DPF_CHECKPOINT, HNN_CHECKPOINT, device)
    var_dq = hnn.qvel_var.mean().item()
    var_dp = hnn.mom_dot_var.mean().item()

    # Load torques
    torques = load_torques(args.policy, device, num_samples=100)

    # Create figure: 8 rows × 6 columns
    fig, axes = plt.subplots(8, 6, figsize=(18, 16))
    col_labels = ['$q_0$', '$q_1$', '$q_2$', '$p_0$', '$p_1$', '$p_2$']

    row_idx = 0
    for length in LENGTHS:
        print(f"\n=== Processing L={length} ===")

        # Find P10 sample based on unguided MSE
        sample_idx = find_p10_sample_idx_live(
            model, hnn, torques, device, length, mj_model,
            args.guidance_lr, num_samples=args.num_samples_per_length
        )

        # Store data for y-limit sync
        row_data = {}

        for guidance in ['unguided', 'guided']:
            print(f"  Generating {guidance} trajectory for sample {sample_idx}...")

            is_guided = (guidance == 'guided')

            # Generate single trajectory with fixed seed for this sample
            seed = 228 + sample_idx  # Use sample_idx as part of seed for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            traj_torque = torques[sample_idx:sample_idx+1, :length, :]

            if is_guided:
                guidance_kwargs = dict(
                    hnn=hnn,
                    guidance_method='adam',
                    guidance_steps=25,
                    guidance_lr=args.guidance_lr,
                    guidance_after_steps=45,
                    lambda_init=0.0,
                )
            else:
                guidance_kwargs = dict(hnn=None, guidance_steps=0)

            state, torque_out = model.sample_trajectories(
                num_samples=1,
                trajectory_length=length,
                num_diffusion_steps=50,
                context_fraction=0.2,
                use_ema=False,
                sampler='ddim',
                guidance_scale=1.0,
                torque=traj_torque,
                **guidance_kwargs,
            )

            state = state[0].cpu().numpy()
            torque = torque_out[0].cpu().numpy()

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
            print(f"    NMSE={nmse:.4f}, HamRes={hamres:.2f}")

            bg_color = '#f0f0f0' if guidance == 'unguided' else 'white'
            t_gen = np.arange(len(qpos_gen)) * DATA_DT
            t_rec = np.arange(len(qpos_rec)) * DATA_DT

            row_data[guidance] = {
                'qpos_gen': qpos_gen, 'mom_gen': mom_gen,
                'qpos_rec': qpos_rec, 'mom_rec': mom_rec,
                't_gen': t_gen, 't_rec': t_rec,
                'nmse': nmse, 'hamres': hamres,
                'bg_color': bg_color, 'row_idx': row_idx
            }

            # Plot
            for col_idx in range(6):
                ax = axes[row_idx, col_idx]
                ax.set_facecolor(bg_color)

                if col_idx < QPOS_DIM:
                    gen_data = qpos_gen[:, col_idx]
                    rec_data = qpos_rec[:, col_idx]
                else:
                    gen_data = mom_gen[:, col_idx - QPOS_DIM]
                    rec_data = mom_rec[:, col_idx - QPOS_DIM]

                label_gen = 'Guided' if is_guided else 'Unguided'
                ax.scatter(t_gen, gen_data, c='blue', s=1, label=label_gen, alpha=0.7)
                ax.scatter(t_rec, rec_data, c='red', s=1, label='MuJoCo', alpha=0.7)

                if row_idx == 0:
                    ax.set_title(col_labels[col_idx])
                if col_idx == 0:
                    ax.set_ylabel(f'L={length}\n{guidance.capitalize()}', fontsize=9)
                if row_idx == 7:
                    ax.set_xlabel('Time (s)')

                if col_idx == 0:
                    ax.legend(loc='upper right', fontsize=6)

                ax.tick_params(labelsize=7)

            ax_right = axes[row_idx, -1]
            ax_right.annotate(
                f'NMSE={nmse:.4f}\nHamRes={hamres:.2f}',
                xy=(1.02, 0.5), xycoords='axes fraction',
                fontsize=8, va='center', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            row_idx += 1

        # Sync y-limits for unguided/guided pair
        unguided_row = row_data['unguided']['row_idx']
        guided_row = row_data['guided']['row_idx']
        for col_idx in range(6):
            ax_ung = axes[unguided_row, col_idx]
            ax_gui = axes[guided_row, col_idx]
            ymin = min(ax_ung.get_ylim()[0], ax_gui.get_ylim()[0])
            ymax = max(ax_ung.get_ylim()[1], ax_gui.get_ylim()[1])
            ax_ung.set_ylim(ymin, ymax)
            ax_gui.set_ylim(ymin, ymax)

    # Title
    policy_labels = {
        'sinusoidal': 'Sinusoidal Torque',
        'gp': 'GP Torque',
        'zero': 'Zero Torque',
        'spline': 'Cubic Spline Torque'
    }
    fig.suptitle(f'DPF vs MuJoCo - {policy_labels[args.policy]} (LR={args.guidance_lr})',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 0.95, 0.96])

    # Save
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / f'traj_comparison_{args.policy}_lr{args.guidance_lr}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
