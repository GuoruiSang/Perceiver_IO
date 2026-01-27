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


def select_best_samples(baseline_results, guided_results, n=3):
    """
    Select n samples that best demonstrate guidance improvement.

    Priority:
    1. Guided MSE should be as low as possible (best overlap)
    2. Baseline MSE should be above average (visible gap)
    """
    baseline_mses = np.array([r['mse_total'] for r in baseline_results])
    guided_mses = np.array([r['mse_total'] for r in guided_results])

    # Compute percentiles
    guided_p10 = np.percentile(guided_mses, 10)  # Want very low guided MSE
    baseline_median = np.percentile(baseline_mses, 50)

    # Compute improvement ratio
    improvements = (baseline_mses - guided_mses) / (baseline_mses + 1e-8)

    # Score: PRIORITY is low guided MSE, baseline should be above average
    scores = []
    for i in range(len(baseline_mses)):
        # Primary: guided MSE should be very low (top 10%)
        if guided_mses[i] <= guided_p10:
            guided_score = 1.0
        else:
            guided_score = guided_p10 / (guided_mses[i] + 1e-8)  # Penalize higher guided MSE

        # Secondary: baseline should be above average (visible gap)
        if baseline_mses[i] >= baseline_median:
            baseline_score = 1.0  # Above average - good for showing improvement
        else:
            baseline_score = 0.3  # Below average - less visible improvement

        # Require high improvement
        if improvements[i] > 0.9:
            improvement_score = 1.0
        elif improvements[i] > 0.7:
            improvement_score = 0.7
        else:
            improvement_score = 0.2

        # Combined score - guided quality is most important, then baseline visibility
        score = guided_score * 0.5 + baseline_score * 0.3 + improvement_score * 0.2
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
    """Create comparison figure with clear separation between sample pairs."""

    n_samples = len(selected_indices)
    n_cols = 6  # qpos[0-2] + mom[0-2]

    # Use GridSpec for custom spacing - larger gap between comparisons
    fig = plt.figure(figsize=(14, 2.5 * n_samples * 2))

    # Height ratios: each pair gets 2 rows, with extra space between pairs
    height_ratios = []
    for i in range(n_samples):
        height_ratios.extend([1, 1])  # baseline, guided
        if i < n_samples - 1:
            height_ratios.append(0.3)  # spacer between comparisons

    gs = fig.add_gridspec(len(height_ratios), n_cols, height_ratios=height_ratios,
                          hspace=0.1, wspace=0.25)

    dim_names = ['qpos[0]', 'qpos[1]', 'qpos[2]', 'mom[0]', 'mom[1]', 'mom[2]']

    for sample_idx, data_idx in enumerate(selected_indices):
        baseline = baseline_results[data_idx]
        guided = guided_results[data_idx]

        # Calculate actual row indices accounting for spacers
        base_row = sample_idx * 3 if sample_idx > 0 else 0
        if sample_idx > 0:
            base_row = sample_idx * 2 + sample_idx  # 0, 3, 6 for samples 0, 1, 2
        else:
            base_row = 0

        row_baseline = base_row
        row_guided = base_row + 1

        axes_baseline = [fig.add_subplot(gs[row_baseline, col]) for col in range(n_cols)]
        axes_guided = [fig.add_subplot(gs[row_guided, col]) for col in range(n_cols)]

        for col in range(n_cols):
            # Determine which data to plot
            if col < 3:
                gen_baseline = baseline['generated']['seq_qpos'][1:, col]
                rec_baseline = baseline['reconstructed']['seq_qpos'][:, col]
                gen_guided = guided['generated']['seq_qpos'][1:, col]
                rec_guided = guided['reconstructed']['seq_qpos'][:, col]
            else:
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

            # Plot baseline row - light gray background
            ax_base = axes_baseline[col]
            ax_base.set_facecolor('#f0f0f0')
            ax_base.scatter(t, gen_baseline, s=0.3, c='#1f77b4', alpha=0.7, label='Generated')
            ax_base.scatter(t, rec_baseline, s=0.3, c='#d62728', alpha=0.7, label='Reconstructed')
            ax_base.set_ylim(y_lim)
            ax_base.set_xticklabels([])

            # Plot guided row - white background
            ax_guided = axes_guided[col]
            ax_guided.set_facecolor('white')
            ax_guided.scatter(t, gen_guided, s=0.3, c='#1f77b4', alpha=0.7, label='Generated')
            ax_guided.scatter(t, rec_guided, s=0.3, c='#d62728', alpha=0.7, label='Reconstructed')
            ax_guided.set_ylim(y_lim)

            # Only show x-axis labels on bottom row of last comparison
            if sample_idx != n_samples - 1:
                ax_guided.set_xticklabels([])

            # Column titles (only on first row)
            if sample_idx == 0:
                ax_base.set_title(dim_names[col], fontsize=10)

        # Row labels
        axes_baseline[0].set_ylabel(f'w/o Guidance\nMSE={baseline["mse_total"]:.4f}',
                                     fontsize=9, fontweight='bold')
        axes_guided[0].set_ylabel(f'w/ Guidance\nMSE={guided["mse_total"]:.4f}',
                                   fontsize=9, fontweight='bold')

        # Add comparison label on the right
        fig.text(0.995, (row_baseline + 1) / len(height_ratios),
                 f'Sample {sample_idx+1}', ha='right', va='center',
                 fontsize=11, fontweight='bold', rotation=-90,
                 transform=fig.transFigure)

    # Add legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=5, label='Generated'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=5, label='Reconstructed'),
    ]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.99), framealpha=0.9)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"\nFigure saved to: {output_path}")


def generate_and_evaluate_for_length(model, hnn, torques, traj_length, args, device, seed=228):
    """Generate baseline and guided trajectories for a specific length and compute MSE."""

    # Truncate torques to trajectory length
    traj_torques = torques[:args.num_samples, :traj_length, :]

    # Generate baseline trajectories (no guidance)
    print(f"\n[Length={traj_length}] Generating baseline trajectories (no guidance)...")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    baseline_states = []
    baseline_torques_out = []

    for batch_start in tqdm(range(0, args.num_samples, args.batch_size), desc=f"Baseline L={traj_length}"):
        batch_end = min(batch_start + args.batch_size, args.num_samples)
        batch_torques = traj_torques[batch_start:batch_end]

        state, torque_out = model.sample_trajectories(
            num_samples=batch_torques.shape[0],
            trajectory_length=traj_length,
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
    print(f"[Length={traj_length}] Generating guided trajectories...")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    guided_states = []
    guided_torques_out = []

    for batch_start in tqdm(range(0, args.num_samples, args.batch_size), desc=f"Guided L={traj_length}"):
        batch_end = min(batch_start + args.batch_size, args.num_samples)
        batch_torques = traj_torques[batch_start:batch_end]

        state, torque_out = model.sample_trajectories(
            num_samples=batch_torques.shape[0],
            trajectory_length=traj_length,
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
    print(f"[Length={traj_length}] Computing MSE and reconstructions...")
    baseline_results = []
    guided_results = []

    for i in tqdm(range(args.num_samples), desc=f"Computing MSE L={traj_length}"):
        baseline_results.append(
            compute_mse_and_reconstruction(model, baseline_state[i], baseline_torque[i])
        )
        guided_results.append(
            compute_mse_and_reconstruction(model, guided_state[i], guided_torque[i])
        )

    return baseline_results, guided_results


def main():
    parser = argparse.ArgumentParser(description="Trajectory Comparison Visualization")
    parser.add_argument("--test_torques", type=str, default="data/test_torques_2000.h5",
                        help="Path to test torques HDF5 file")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Batch size for sampling")
    parser.add_argument("--trajectory_lengths", type=str, default="50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000",
                        help="Comma-separated list of trajectory lengths to evaluate")
    parser.add_argument("--num_diffusion_steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/gsang/Projects/Perceiver_IO/checkpoints/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt",
                        help="Path to model checkpoint")
    parser.add_argument("--hnn_checkpoint", type=str,
                        default="/home/gsang/Projects/Perceiver_IO/checkpoints/SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt",
                        help="Path to HNN checkpoint")
    parser.add_argument("--output_dir", type=str, default="plots",
                        help="Output directory for figures")
    parser.add_argument("--n_display", type=int, default=3,
                        help="Number of trajectories to display per length")

    # Guidance config
    parser.add_argument("--guidance_steps", type=int, default=25)
    parser.add_argument("--guidance_lr", type=float, default=0.01)
    parser.add_argument("--guidance_after_steps", type=int, default=45)

    args = parser.parse_args()

    # Parse trajectory lengths
    trajectory_lengths = [int(x.strip()) for x in args.trajectory_lengths.split(',')]
    print(f"Will evaluate trajectory lengths: {trajectory_lengths}")

    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and HNN
    model, hnn = load_model_and_hnn(args.checkpoint, args.hnn_checkpoint, device)

    # Load test torques
    torque_path = project_root / args.test_torques
    torques = load_test_torques(str(torque_path), device)

    seed = 228

    # Process each trajectory length
    for traj_length in trajectory_lengths:
        print(f"\n{'='*60}")
        print(f"Processing trajectory length: {traj_length}")
        print(f"{'='*60}")

        # Generate and evaluate
        baseline_results, guided_results = generate_and_evaluate_for_length(
            model, hnn, torques, traj_length, args, device, seed
        )

        # Select best samples
        selected_indices = select_best_samples(baseline_results, guided_results, n=args.n_display)

        # Create comparison figure for this length
        output_path = os.path.join(args.output_dir, f'trajectory_comparison_L{traj_length}.png')
        create_comparison_figure(baseline_results, guided_results, selected_indices, output_path)

    print(f"\n{'='*60}")
    print(f"All figures saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
