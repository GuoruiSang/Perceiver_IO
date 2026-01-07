"""
Generate trajectories with the SAME torque sequence to check variance.
This tests whether the model properly conditions on torques.

Also includes diagnostics for:
- Per-step causality check (torque pulse affects t+1)
- MuJoCo reconstruction error
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import mujoco

from src.models.trajectory_dpf import TrajectoryDPF
from src.models.utils import EMA, compare_generated_with_reconstructed, reconstruct_traj_with_momentum
from scripts.dataset import TrajectoryDPFCached
from src import config


def plot_variance_comparison(all_states, all_torques, qpos_dim, mom_dim, save_path, 
                             guidance_scale, num_diffusion_steps, name='variance_comparison'):
    """
    Plot multiple trajectories generated from the same torque to visualize variance.
    
    Args:
        all_states: list of [T, state_dim] arrays
        all_torques: [T, torque_dim] array (same for all)
        qpos_dim: dimension of position
        mom_dim: dimension of momentum
        save_path: directory to save plot
        guidance_scale: CFG guidance scale used
        num_diffusion_steps: number of diffusion steps used
    """
    num_samples = len(all_states)
    T = all_states[0].shape[0]
    
    # Convert to numpy arrays
    states = np.stack([s if isinstance(s, np.ndarray) else s.cpu().numpy() for s in all_states])  # [N, T, state_dim]
    torque = all_torques if isinstance(all_torques, np.ndarray) else all_torques.cpu().numpy()  # [T, torque_dim]
    
    # Split into components
    qpos_all = states[:, :, :qpos_dim]  # [N, T, qpos_dim]
    mom_all = states[:, :, qpos_dim:]   # [N, T, mom_dim]
    
    # Compute statistics
    qpos_mean = qpos_all.mean(axis=0)  # [T, qpos_dim]
    qpos_std = qpos_all.std(axis=0)
    mom_mean = mom_all.mean(axis=0)
    mom_std = mom_all.std(axis=0)
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    time = np.arange(T)
    colors = plt.cm.tab10(np.linspace(0, 1, num_samples))
    
    # Row 0: Position (qpos) with all samples
    for j in range(min(3, qpos_dim)):
        ax = axes[0, j]
        for i in range(num_samples):
            ax.plot(time, qpos_all[i, :, j], alpha=0.5, color=colors[i], linewidth=0.8)
        ax.fill_between(time, qpos_mean[:, j] - 2*qpos_std[:, j], 
                        qpos_mean[:, j] + 2*qpos_std[:, j], alpha=0.3, color='blue')
        ax.plot(time, qpos_mean[:, j], 'b-', linewidth=2, label='Mean')
        ax.set_title(f'Position q[{j}] (n={num_samples} samples)')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Row 1: Momentum with all samples
    for j in range(min(3, mom_dim)):
        ax = axes[1, j]
        for i in range(num_samples):
            ax.plot(time, mom_all[i, :, j], alpha=0.5, color=colors[i], linewidth=0.8)
        ax.fill_between(time, mom_mean[:, j] - 2*mom_std[:, j], 
                        mom_mean[:, j] + 2*mom_std[:, j], alpha=0.3, color='green')
        ax.plot(time, mom_mean[:, j], 'g-', linewidth=2, label='Mean')
        ax.set_title(f'Momentum p[{j}] (n={num_samples} samples)')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Momentum')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Row 2: Torque (conditioning signal - same for all)
    torque_dim = torque.shape[-1]
    for j in range(min(3, torque_dim)):
        ax = axes[2, j]
        ax.plot(time, torque[:, j], 'r-', linewidth=2)
        ax.set_title(f'Torque τ[{j}] (SAME for all samples)')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Torque')
        ax.grid(True, alpha=0.3)
    
    # Add overall statistics
    avg_qpos_std = qpos_std.mean()
    avg_mom_std = mom_std.mean()
    fig.suptitle(f'Variance Analysis: {num_samples} samples with SAME torque | '
                 f'CFG Scale: {guidance_scale} | Steps: {num_diffusion_steps}\n'
                 f'Avg Position Std: {avg_qpos_std:.4f} | Avg Momentum Std: {avg_mom_std:.4f}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f'{name}.jpg')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Variance Analysis] Saved to {out_path}")
    print(f"[Variance Analysis] CFG Scale: {guidance_scale}")
    print(f"[Variance Analysis] Avg Position Std: {avg_qpos_std:.4f}")
    print(f"[Variance Analysis] Avg Momentum Std: {avg_mom_std:.4f}")
    
    return avg_qpos_std, avg_mom_std


def run_variance_analysis(model, fixed_torque, guidance_scale, num_samples_per_torque, 
                          trajectory_length, num_diffusion_steps, context_fraction, 
                          use_ema, sampler, output_dir):
    """Run variance analysis for a specific guidance scale."""
    
    print(f"\n{'='*60}")
    print(f"Running variance analysis with CFG scale = {guidance_scale}")
    print(f"{'='*60}")
    
    all_states = []
    for i in range(num_samples_per_torque):
        # Use different random seed for each sample's noise
        torch.manual_seed(i * 1000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(i * 1000)
        
        print(f"  Sample {i+1}/{num_samples_per_torque}...", end=" ", flush=True)
        state, _ = model.sample_trajectories(
            num_samples=1,
            trajectory_length=trajectory_length,
            num_diffusion_steps=num_diffusion_steps,
            context_fraction=context_fraction,
            use_ema=use_ema,
            sampler=sampler,
            guidance_scale=guidance_scale,
            torque=fixed_torque,  # SAME torque for all
            hnn=None,
            guidance_steps=0,
        )
        all_states.append(state[0].cpu().numpy())
        print("done")
    
    # Plot variance comparison
    avg_qpos_std, avg_mom_std = plot_variance_comparison(
        all_states,
        fixed_torque[0].cpu().numpy(),
        model.qpos_dim,
        model.mom_dim,
        output_dir,
        guidance_scale=guidance_scale,
        num_diffusion_steps=num_diffusion_steps,
        name=f'variance_cfg{guidance_scale}_steps{num_diffusion_steps}'
    )
    
    return avg_qpos_std, avg_mom_std


def compute_reconstruction_error(generated_state, generated_torque, model, mujoco_xml_path):
    """
    Compute per-timestep reconstruction error between generated trajectory
    and MuJoCo-reconstructed trajectory.
    
    Args:
        generated_state: [T, state_dim] numpy array
        generated_torque: [T, torque_dim] numpy array
        model: TrajectoryDPF model (for dimensions and dt)
        mujoco_xml_path: path to MuJoCo XML model
        
    Returns:
        mse_qpos: per-timestep MSE for qpos [T-1,]
        mse_mom: per-timestep MSE for mom [T-1,]
        total_mse: scalar total MSE
    """
    qpos_dim = model.qpos_dim
    mom_dim = model.mom_dim
    dt = float(model.dt)
    data_dt = float(model.data_dt)
    
    # Split generated state
    gen_qpos = generated_state[:, :qpos_dim]
    gen_mom = generated_state[:, qpos_dim:]
    
    # Load MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path(mujoco_xml_path)
    mj_data = mujoco.MjData(mj_model)
    
    # Compute initial velocity from initial momentum: v = M^{-1} @ p
    mj_data.qpos[:] = gen_qpos[0]
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)
    
    M = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, M, mj_data.qM)
    initial_qvel = np.linalg.solve(M, gen_mom[0])
    
    # Reconstruct using MuJoCo
    T = len(gen_qpos)
    recon = reconstruct_traj_with_momentum(
        mj_model, T, dt,
        gen_qpos[0], initial_qvel, generated_torque,
        data_dt=data_dt
    )
    
    # Compute per-timestep MSE (recon has T-1 points, compare with gen[1:])
    recon_qpos = recon['seq_qpos']  # [T-1, qpos_dim]
    recon_mom = recon['seq_mom']    # [T-1, mom_dim]
    
    mse_qpos = np.mean((gen_qpos[1:] - recon_qpos) ** 2, axis=1)  # [T-1,]
    mse_mom = np.mean((gen_mom[1:] - recon_mom) ** 2, axis=1)     # [T-1,]
    
    total_mse = np.mean(mse_qpos) + np.mean(mse_mom)
    
    return mse_qpos, mse_mom, total_mse


def run_causality_test(model, base_torque, pulse_time, pulse_scale, trajectory_length,
                       num_diffusion_steps, context_fraction, use_ema, sampler, 
                       guidance_scale, output_dir):
    """
    Test per-step causality by comparing trajectories with/without a torque pulse.
    
    The key test: applying a torque pulse at time t should affect state from t+1 onwards,
    but NOT state at time t (which is before the pulse takes effect).
    
    Args:
        model: TrajectoryDPF model
        base_torque: [1, T, torque_dim] base torque sequence
        pulse_time: timestep at which to apply the pulse
        pulse_scale: scale of the pulse (multiplier on torque at that timestep)
        trajectory_length: length of trajectory
        num_diffusion_steps: number of diffusion steps
        context_fraction: context fraction for sampling
        use_ema: whether to use EMA weights
        sampler: sampling method
        guidance_scale: CFG guidance scale
        output_dir: directory to save results
    """
    print(f"\n{'='*60}")
    print(f"Running causality test: pulse at t={pulse_time}, scale={pulse_scale}")
    print(f"{'='*60}")
    
    device = base_torque.device
    
    # Create pulsed torque (scale up torque at pulse_time)
    pulsed_torque = base_torque.clone()
    pulsed_torque[:, pulse_time, :] *= pulse_scale
    
    # Generate with base torque
    torch.manual_seed(12345)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12345)
    
    print("Generating with base torque...")
    state_base, _ = model.sample_trajectories(
        num_samples=1,
        trajectory_length=trajectory_length,
        num_diffusion_steps=num_diffusion_steps,
        context_fraction=context_fraction,
        use_ema=use_ema,
        sampler=sampler,
        guidance_scale=guidance_scale,
        torque=base_torque,
        hnn=None,
        guidance_steps=0,
    )
    
    # Generate with pulsed torque (SAME initial noise)
    torch.manual_seed(12345)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12345)
    
    print("Generating with pulsed torque...")
    state_pulsed, _ = model.sample_trajectories(
        num_samples=1,
        trajectory_length=trajectory_length,
        num_diffusion_steps=num_diffusion_steps,
        context_fraction=context_fraction,
        use_ema=use_ema,
        sampler=sampler,
        guidance_scale=guidance_scale,
        torque=pulsed_torque,
        hnn=None,
        guidance_steps=0,
    )
    
    # Compute difference
    state_base_np = state_base[0].cpu().numpy()
    state_pulsed_np = state_pulsed[0].cpu().numpy()
    diff = np.abs(state_pulsed_np - state_base_np)  # [T, state_dim]
    diff_per_t = diff.mean(axis=1)  # [T,]
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    time = np.arange(trajectory_length)
    
    # Plot 1: Difference per timestep
    ax = axes[0]
    ax.plot(time, diff_per_t, 'b-', linewidth=1.5)
    ax.axvline(x=pulse_time, color='r', linestyle='--', linewidth=2, label=f'Pulse at t={pulse_time}')
    ax.axvline(x=pulse_time+1, color='g', linestyle=':', linewidth=2, label=f't+1 (expected effect start)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Mean absolute difference')
    ax.set_title(f'Trajectory difference: pulse at t={pulse_time} (scale={pulse_scale})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Zoom around pulse
    ax = axes[1]
    window = 20
    t_start = max(0, pulse_time - window)
    t_end = min(trajectory_length, pulse_time + 2*window)
    ax.plot(time[t_start:t_end], diff_per_t[t_start:t_end], 'b-', linewidth=1.5, marker='o', markersize=3)
    ax.axvline(x=pulse_time, color='r', linestyle='--', linewidth=2, label=f'Pulse at t={pulse_time}')
    ax.axvline(x=pulse_time+1, color='g', linestyle=':', linewidth=2, label=f't+1')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Mean absolute difference')
    ax.set_title(f'Zoomed view around pulse')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Check causality: difference before t+1 should be ~0
    diff_before = diff_per_t[:pulse_time+1].mean()
    diff_after = diff_per_t[pulse_time+1:].mean()
    
    fig.suptitle(f'Causality Test: torque_t affects state_{{t+1}}\n'
                 f'Diff before t+1: {diff_before:.6f} | Diff after t+1: {diff_after:.6f} | '
                 f'Ratio: {diff_after/(diff_before + 1e-8):.1f}x',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'causality_test_t{pulse_time}.jpg')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Causality Test] Saved to {out_path}")
    print(f"[Causality Test] Diff before t+1: {diff_before:.6f}")
    print(f"[Causality Test] Diff after t+1: {diff_after:.6f}")
    print(f"[Causality Test] Ratio (after/before): {diff_after/(diff_before + 1e-8):.1f}x")
    
    # Ideally: diff_before should be ~0, diff_after should be large
    causality_ok = diff_after > diff_before * 10  # At least 10x larger
    print(f"[Causality Test] Result: {'PASS' if causality_ok else 'MARGINAL'}")
    
    return diff_before, diff_after


def main():
    # Configuration
    checkpoint_path = "/home/gsang/Projects/Perceiver_IO/checkpoints/trajectory_dpf_TrainAlignsInfer:epoch=509_val_loss:val_loss=0.0261.ckpt"
    output_dir = "/home/gsang/Projects/Perceiver_IO/plots/variance_analysis"
    
    num_samples_per_torque = 10  # Number of samples with same torque
    trajectory_length = 1000
    
    # Sampling parameters
    sampler = "ddim"
    num_diffusion_steps = 200
    context_fraction = 0.5
    use_ema = True
    
    # Torque selection: random or from dataset
    use_dataset_torque = True  # Set to True to use a "trained" torque sequence
    dataset_index = 0          # Which trajectory from the dataset to use
    torque_seed = 42           # Seed for generating torque (if use_dataset_torque=False)
    
    # Guidance scales to test
    guidance_scales = [4, 8, 16]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model from checkpoint (EMA is handled by on_load_checkpoint)
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = TrajectoryDPF.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    model = model.to(device)
    print(f"Model loaded. qpos_dim={model.qpos_dim}, mom_dim={model.mom_dim}, torque_dim={model.torque_dim}")
    
    # Select fixed torque sequence
    if use_dataset_torque:
        print(f"\nLoading torque sequence from dataset (index {dataset_index})...")
        # Load dataset to extract a real torque sequence
        dataset = TrajectoryDPFCached(config.DEFAULT_H5_PATH, trajectory_length=trajectory_length)
        sample = dataset[dataset_index]
        fixed_torque = sample['seq_torque'].unsqueeze(0).to(device)  # [1, T, torque_dim]
        print(f"✓ Loaded torque from {config.DEFAULT_H5_PATH}")
    else:
        print(f"\nGenerating random torque sequence (seed={torque_seed})...")
        # Set seed for reproducible torque generation
        np.random.seed(torque_seed)
        fixed_torque = model._generate_random_torque(
            num_samples=1, 
            trajectory_length=trajectory_length,
            dt=model.data_dt
        )  # [1, T, torque_dim]
    
    # Run variance analysis for each guidance scale
    results = {}
    for guidance_scale in guidance_scales:
        avg_qpos_std, avg_mom_std = run_variance_analysis(
            model=model,
            fixed_torque=fixed_torque,
            guidance_scale=guidance_scale,
            num_samples_per_torque=num_samples_per_torque,
            trajectory_length=trajectory_length,
            num_diffusion_steps=num_diffusion_steps,
            context_fraction=context_fraction,
            use_ema=use_ema,
            sampler=sampler,
            output_dir=output_dir
        )
        results[guidance_scale] = (avg_qpos_std, avg_mom_std)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: Variance vs Guidance Scale")
    print(f"{'='*60}")
    print(f"{'CFG Scale':<12} {'Avg Position Std':<20} {'Avg Momentum Std':<20}")
    print("-" * 52)
    for scale in guidance_scales:
        qpos_std, mom_std = results[scale]
        print(f"{scale:<12} {qpos_std:<20.4f} {mom_std:<20.4f}")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    
    # =========================================================================
    # CAUSALITY TEST: Check that torque_t affects state_{t+1}
    # =========================================================================
    print("\n" + "="*60)
    print("CAUSALITY TEST: Verify per-step control (torque_t -> state_{t+1})")
    print("="*60)
    
    run_causality_test(
        model=model,
        base_torque=fixed_torque,
        pulse_time=trajectory_length // 2,  # Pulse in the middle
        pulse_scale=5.0,  # 5x torque at that timestep
        trajectory_length=trajectory_length,
        num_diffusion_steps=num_diffusion_steps,
        context_fraction=context_fraction,
        use_ema=use_ema,
        sampler=sampler,
        guidance_scale=guidance_scales[-1],  # Use highest CFG scale
        output_dir=output_dir
    )
    
    # =========================================================================
    # RECONSTRUCTION ERROR: Compare with MuJoCo physics
    # =========================================================================
    print("\n" + "="*60)
    print("RECONSTRUCTION ERROR: Compare with MuJoCo physics")
    print("="*60)
    
    mujoco_xml_path = "/home/gsang/Projects/Perceiver_IO/configs/rigid_arm_hinge.xml"
    
    # Generate one trajectory for reconstruction comparison
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    state_for_recon, torque_for_recon = model.sample_trajectories(
        num_samples=1,
        trajectory_length=trajectory_length,
        num_diffusion_steps=num_diffusion_steps,
        context_fraction=context_fraction,
        use_ema=use_ema,
        sampler=sampler,
        guidance_scale=guidance_scales[-1],
        torque=fixed_torque,
        hnn=None,
        guidance_steps=0,
    )
    
    state_np = state_for_recon[0].cpu().numpy()
    torque_np = torque_for_recon[0].cpu().numpy()
    
    mse_qpos, mse_mom, total_mse = compute_reconstruction_error(
        state_np, torque_np, model, mujoco_xml_path
    )
    
    print(f"[Reconstruction] Total MSE: {total_mse:.6f}")
    print(f"[Reconstruction] Avg MSE qpos: {mse_qpos.mean():.6f}")
    print(f"[Reconstruction] Avg MSE mom: {mse_mom.mean():.6f}")
    print(f"[Reconstruction] Max MSE qpos: {mse_qpos.max():.6f} at t={mse_qpos.argmax()}")
    print(f"[Reconstruction] Max MSE mom: {mse_mom.max():.6f} at t={mse_mom.argmax()}")
    
    # Plot reconstruction error over time
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    time = np.arange(len(mse_qpos))
    
    axes[0].plot(time, mse_qpos, 'b-', linewidth=1)
    axes[0].set_xlabel('Time step')
    axes[0].set_ylabel('MSE')
    axes[0].set_title(f'Position (qpos) reconstruction error over time | Avg: {mse_qpos.mean():.6f}')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time, mse_mom, 'g-', linewidth=1)
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('MSE')
    axes[1].set_title(f'Momentum (mom) reconstruction error over time | Avg: {mse_mom.mean():.6f}')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'MuJoCo Reconstruction Error | Total MSE: {total_mse:.6f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    recon_plot_path = os.path.join(output_dir, 'reconstruction_error.jpg')
    plt.savefig(recon_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Reconstruction] Saved error plot to {recon_plot_path}")
    
    # Also generate the comparison plot
    generated = {
        'seq_qpos': state_np[:, :model.qpos_dim],
        'seq_mom': state_np[:, model.qpos_dim:],
        'seq_torque': torque_np,
    }
    compare_generated_with_reconstructed(
        generated, mujoco_xml_path, output_dir,
        dt=float(model.dt), data_dt=float(model.data_dt),
        name='comparison_final'
    )
    
    print("\n" + "="*60)
    print("DIAGNOSTICS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
