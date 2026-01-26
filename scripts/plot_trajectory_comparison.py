"""
Trajectory Comparison Visualization

Creates a 10x6 subplot figure comparing trajectories with and without guidance:
- Rows 1-5: 5 trajectories WITHOUT guidance (generated vs reconstructed)
- Rows 6-10: 5 trajectories WITH guidance (same samples)
- Columns: 6 state dimensions (qpos[0-2] + mom[0-2])

Usage:
    python scripts/plot_trajectory_comparison.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from src.models.trajectory_dpf import TrajectoryDPF
from src.models.utils import EMA, compare_generated_with_reconstructed


# ICLR-friendly style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.3,
})


def load_model_and_hnn(checkpoint_path, hnn_checkpoint_path, device):
    """Load the trajectory model and HNN from checkpoints."""
    print(f"Loading model from: {checkpoint_path}")
    model = TrajectoryDPF.load_from_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()

    # Load EMA shadow if present
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ema_shadow = checkpoint.get('ema_shadow', None)
    ema_decay = checkpoint.get('ema_decay', 0.9995)
    if ema_shadow is not None:
        model.ema = EMA(model.model, decay=ema_decay)
        for name, tensor in ema_shadow.items():
            if name in model.ema.shadow:
                model.ema.shadow[name] = tensor.to(device=device, dtype=model.ema.shadow[name].dtype)
        model._ema_loaded = True
        print("  EMA shadow loaded")

    # Load HNN
    hnn = None
    if hnn_checkpoint_path:
        from src.models.HNN import HNNWrapper
        print(f"Loading HNN from: {hnn_checkpoint_path}")
        hnn = HNNWrapper.load_from_checkpoint(hnn_checkpoint_path, map_location=device)
        hnn = hnn.to(device)
        hnn.eval()

    return model, hnn


def load_test_torques(torque_file, device):
    """Load pre-generated test torques from HDF5."""
    print(f"Loading test torques from: {torque_file}")
    with h5py.File(torque_file, 'r') as f:
        torques = torch.tensor(f['torques'][:], dtype=torch.float32, device=device)
        print(f"  Loaded {torques.shape[0]} torque sequences, shape: {torques.shape}")
    return torques


def compute_mse_and_reconstruction(model, state, torque):
    """Compute MSE and get reconstructed trajectory for a single sample."""
    import mujoco

    state_np = state.cpu().numpy()
    torque_np = torque.cpu().numpy()

    generated = {
        'seq_qpos': state_np[:, :model.qpos_dim],
        'seq_mom': state_np[:, model.qpos_dim:],
        'seq_torque': torque_np,
    }

    # Load MuJoCo model for reconstruction
    mujoco_model_path = str(project_root / 'configs/rigid_arm_hinge.xml')
    mj_model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    mj_data = mujoco.MjData(mj_model)

    # Compute initial velocity from initial momentum
    mj_data.qpos[:] = generated['seq_qpos'][0]
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    M = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, M, mj_data.qM)
    initial_qvel = np.linalg.solve(M, generated['seq_mom'][0])

    # Reconstruct using MuJoCo physics
    from src.models.utils import reconstruct_traj_with_momentum
    recon = reconstruct_traj_with_momentum(
        mj_model, len(generated['seq_qpos']), float(model.dt),
        generated['seq_qpos'][0], initial_qvel, generated['seq_torque'],
        data_dt=float(model.data_dt)
    )

    # Compute MSE (align: generated[1:] vs reconstructed)
    mse_qpos = np.mean((generated['seq_qpos'][1:] - recon['seq_qpos']) ** 2)
    mse_mom = np.mean((generated['seq_mom'][1:] - recon['seq_mom']) ** 2)
    mse_total = mse_qpos + mse_mom

    return {
        'mse_total': mse_total,
        'mse_qpos': mse_qpos,
        'mse_mom': mse_mom,
        'generated': generated,
        'reconstructed': recon,
    }


def select_best_samples(baseline_results, guided_results, n=5):
    """
    Select n samples that best demonstrate guidance improvement.

    Priority:
    1. Guided MSE should be very low (best overlap)
    2. Among those, baseline should be decent but not perfect (visible difference)
    """
    baseline_mses = np.array([r['mse_total'] for r in baseline_results])
    guided_mses = np.array([r['mse_total'] for r in guided_results])

    # Compute percentiles
    guided_p25 = np.percentile(guided_mses, 25)
    baseline_p25 = np.percentile(baseline_mses, 25)
    baseline_p75 = np.percentile(baseline_mses, 75)

    # Compute improvement ratio
    improvements = (baseline_mses - guided_mses) / (baseline_mses + 1e-8)

    # Score: PRIORITY is low guided MSE, then baseline should be visible but not too bad
    scores = []
    for i in range(len(baseline_mses)):
        # Primary: guided MSE should be very low (top 25%)
        if guided_mses[i] <= guided_p25:
            guided_score = 1.0 - guided_mses[i] / (guided_p25 + 1e-8) * 0.5
        else:
            guided_score = 0.3  # Penalize high guided MSE

        # Secondary: baseline should be decent but show room for improvement
        # Prefer baseline between 25th-75th percentile (not too good, not too bad)
        if baseline_p25 <= baseline_mses[i] <= baseline_p75:
            baseline_score = 0.8
        elif baseline_mses[i] < baseline_p25:
            # Too good baseline - less visible improvement
            baseline_score = 0.5
        else:
            # High baseline MSE - still ok if guided is good
            baseline_score = 0.6

        # Require significant improvement
        if improvements[i] > 0.7:
            improvement_score = 1.0
        elif improvements[i] > 0.5:
            improvement_score = 0.7
        else:
            improvement_score = 0.2

        # Combined score - guided quality is most important
        score = guided_score * 0.5 + baseline_score * 0.2 + improvement_score * 0.3
        scores.append((i, score, baseline_mses[i], guided_mses[i], improvements[i]))

    # Sort by score (descending) and select top n
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [s[0] for s in scores[:n]]

    print(f"\nSelected samples:")
    for idx in selected:
        print(f"  Sample {idx}: baseline MSE={baseline_mses[idx]:.6f}, "
              f"guided MSE={guided_mses[idx]:.6f}, "
              f"improvement={improvements[idx]*100:.1f}%")

    return selected


def create_comparison_figure(baseline_results, guided_results, selected_indices, output_path):
    """Create the 10x6 comparison figure with alternating baseline/guided rows."""

    n_samples = len(selected_indices)
    n_rows = n_samples * 2  # baseline + guided (alternating)
    n_cols = 6  # qpos[0-2] + mom[0-2]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 2.2 * n_rows))

    dim_names = ['qpos[0]', 'qpos[1]', 'qpos[2]', 'mom[0]', 'mom[1]', 'mom[2]']

    for sample_idx, data_idx in enumerate(selected_indices):
        baseline = baseline_results[data_idx]
        guided = guided_results[data_idx]

        # Alternating rows: baseline, guided, baseline, guided, ...
        row_baseline = sample_idx * 2      # rows 0, 2, 4, 6, 8
        row_guided = sample_idx * 2 + 1    # rows 1, 3, 5, 7, 9

        for col in range(n_cols):
            # Determine which data to plot
            if col < 3:
                # qpos dimensions
                gen_baseline = baseline['generated']['seq_qpos'][1:, col]
                rec_baseline = baseline['reconstructed']['seq_qpos'][:, col]
                gen_guided = guided['generated']['seq_qpos'][1:, col]
                rec_guided = guided['reconstructed']['seq_qpos'][:, col]
            else:
                # mom dimensions
                mom_col = col - 3
                gen_baseline = baseline['generated']['seq_mom'][1:, mom_col]
                rec_baseline = baseline['reconstructed']['seq_mom'][:, mom_col]
                gen_guided = guided['generated']['seq_mom'][1:, mom_col]
                rec_guided = guided['reconstructed']['seq_mom'][:, mom_col]

            t = np.arange(len(gen_baseline))

            # Compute shared y-axis limits for baseline/guided pair
            all_values = np.concatenate([gen_baseline, rec_baseline, gen_guided, rec_guided])
            y_min, y_max = all_values.min(), all_values.max()
            y_margin = (y_max - y_min) * 0.05
            y_lim = (y_min - y_margin, y_max + y_margin)

            # Plot baseline row
            ax_base = axes[row_baseline, col]
            ax_base.scatter(t, gen_baseline, s=0.3, c='#1f77b4', alpha=0.7, label='Generated')
            ax_base.scatter(t, rec_baseline, s=0.3, c='#d62728', alpha=0.7, label='Reconstructed')
            ax_base.set_ylim(y_lim)

            # Plot guided row
            ax_guided = axes[row_guided, col]
            ax_guided.scatter(t, gen_guided, s=0.3, c='#1f77b4', alpha=0.7, label='Generated')
            ax_guided.scatter(t, rec_guided, s=0.3, c='#d62728', alpha=0.7, label='Reconstructed')
            ax_guided.set_ylim(y_lim)

            # Column titles (only on first row)
            if sample_idx == 0:
                ax_base.set_title(dim_names[col])

            # Remove x-axis labels except bottom row
            if row_baseline != n_rows - 2:
                ax_base.set_xticklabels([])
            if row_guided != n_rows - 1:
                ax_guided.set_xticklabels([])

        # Row labels with MSE
        axes[row_baseline, 0].set_ylabel(
            f'Baseline #{sample_idx+1}\nMSE={baseline["mse_total"]:.4f}',
            fontsize=8
        )
        axes[row_guided, 0].set_ylabel(
            f'Guided #{sample_idx+1}\nMSE={guided["mse_total"]:.4f}',
            fontsize=8
        )

    # Add legend to first subplot
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=5, label='Generated'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=5, label='Reconstructed'),
    ]
    axes[0, n_cols-1].legend(handles=handles, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"\nFigure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Trajectory Comparison Visualization")
    parser.add_argument("--test_torques", type=str, default="data/test_torques_2000.h5",
                        help="Path to test torques HDF5 file")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Batch size for sampling")
    parser.add_argument("--trajectory_length", type=int, default=1000,
                        help="Trajectory length")
    parser.add_argument("--num_diffusion_steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/gsang/Projects/Perceiver_IO/checkpoints/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt",
                        help="Path to model checkpoint")
    parser.add_argument("--hnn_checkpoint", type=str,
                        default="/home/gsang/Projects/Perceiver_IO/checkpoints/SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt",
                        help="Path to HNN checkpoint")
    parser.add_argument("--output_dir", type=str, default="plots",
                        help="Output directory for figure")
    parser.add_argument("--n_display", type=int, default=5,
                        help="Number of trajectories to display")

    # Guidance config
    parser.add_argument("--guidance_steps", type=int, default=25)
    parser.add_argument("--guidance_lr", type=float, default=0.01)
    parser.add_argument("--guidance_after_steps", type=int, default=45)

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and HNN
    model, hnn = load_model_and_hnn(args.checkpoint, args.hnn_checkpoint, device)

    # Load test torques
    torque_path = project_root / args.test_torques
    torques = load_test_torques(str(torque_path), device)

    # Truncate torques to trajectory length
    traj_torques = torques[:args.num_samples, :args.trajectory_length, :]

    seed = 228

    # Generate baseline trajectories (no guidance)
    print("\nGenerating baseline trajectories (no guidance)...")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    baseline_states = []
    baseline_torques_out = []

    for batch_start in tqdm(range(0, args.num_samples, args.batch_size), desc="Baseline"):
        batch_end = min(batch_start + args.batch_size, args.num_samples)
        batch_torques = traj_torques[batch_start:batch_end]

        state, torque_out = model.sample_trajectories(
            num_samples=batch_torques.shape[0],
            trajectory_length=args.trajectory_length,
            num_diffusion_steps=args.num_diffusion_steps,
            context_fraction=0.2,
            use_ema=False,
            sampler='ddim',
            guidance_scale=1.0,
            hnn=None,
            guidance_steps=0,
            torque=batch_torques,
        )
        baseline_states.append(state)
        baseline_torques_out.append(torque_out)

    baseline_state = torch.cat(baseline_states, dim=0)
    baseline_torque = torch.cat(baseline_torques_out, dim=0)

    # Generate guided trajectories (same seed, same torque)
    print("\nGenerating guided trajectories...")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    guided_states = []
    guided_torques_out = []

    for batch_start in tqdm(range(0, args.num_samples, args.batch_size), desc="Guided"):
        batch_end = min(batch_start + args.batch_size, args.num_samples)
        batch_torques = traj_torques[batch_start:batch_end]

        state, torque_out = model.sample_trajectories(
            num_samples=batch_torques.shape[0],
            trajectory_length=args.trajectory_length,
            num_diffusion_steps=args.num_diffusion_steps,
            context_fraction=0.2,
            use_ema=False,
            sampler='ddim',
            guidance_scale=1.0,
            hnn=hnn,
            guidance_method='adam',
            guidance_after_steps=args.guidance_after_steps,
            guidance_steps=args.guidance_steps,
            guidance_lr=args.guidance_lr,
            lambda_init=0.0,
            torque=batch_torques,
        )
        guided_states.append(state)
        guided_torques_out.append(torque_out)

    guided_state = torch.cat(guided_states, dim=0)
    guided_torque = torch.cat(guided_torques_out, dim=0)

    # Compute MSE and reconstructions for all samples
    print("\nComputing MSE and reconstructions...")
    baseline_results = []
    guided_results = []

    for i in tqdm(range(args.num_samples), desc="Computing MSE"):
        baseline_results.append(
            compute_mse_and_reconstruction(model, baseline_state[i], baseline_torque[i])
        )
        guided_results.append(
            compute_mse_and_reconstruction(model, guided_state[i], guided_torque[i])
        )

    # Select best samples
    selected_indices = select_best_samples(baseline_results, guided_results, n=args.n_display)

    # Create comparison figure
    output_path = os.path.join(args.output_dir, 'trajectory_comparison_guidance.png')
    create_comparison_figure(baseline_results, guided_results, selected_indices, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
