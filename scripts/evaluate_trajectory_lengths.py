"""
Trajectory Length Experiments

Evaluate how trajectory length affects the quality of energy-guided sampling by comparing:
1. MSE: Physics reconstruction error (generated vs MuJoCo simulation)
2. Energy: Average HNN Hamiltonian energy compared to ground-truth training trajectories

Usage:
    python scripts/evaluate_trajectory_lengths.py \
        --test_torques data/test_torques_2000.h5 \
        --num_samples 100 \
        --output_dir output_traj_lengths
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
import pandas as pd
from tqdm import tqdm
import os

from src.models.trajectory_dpf import TrajectoryDPF
from src.models.utils import EMA, compare_generated_with_reconstructed
from scripts.dataset import TrajectoryDPFCached


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


def compute_avg_hnn_energy(hnn, state_tensor, qpos_dim):
    """
    Compute average Hamiltonian energy over trajectories.
    
    Args:
        hnn: HNNWrapper model
        state_tensor: [batch, timesteps, state_dim] tensor
        qpos_dim: dimension of qpos (to split state into qpos and momentum)
    
    Returns:
        mean_energy: Average energy across all samples and timesteps
        std_energy: Standard deviation of energy
    """
    batch_size, num_timesteps, state_dim = state_tensor.shape
    
    # Split state into qpos and momentum
    qpos = state_tensor[:, :, :qpos_dim]  # [batch, timesteps, qpos_dim]
    mom = state_tensor[:, :, qpos_dim:]   # [batch, timesteps, mom_dim]
    
    # Reshape for HNN: [batch * timesteps, dim]
    qpos_flat = qpos.reshape(-1, qpos_dim)
    mom_flat = mom.reshape(-1, mom.shape[-1])
    
    # Compute energy
    with torch.no_grad():
        energy = hnn(mom_flat, qpos_flat)  # HNN takes (p, q) order
    
    # Reshape back and compute mean
    energy = energy.reshape(batch_size, num_timesteps)
    
    mean_energy = energy.mean().item()
    std_energy = energy.std().item()
    
    return mean_energy, std_energy


def compute_training_energy(dataset, hnn, num_samples, trajectory_length, device):
    """
    Compute ground-truth average energy from training trajectories.
    
    Args:
        dataset: TrajectoryDPFCached dataset
        hnn: HNNWrapper model
        num_samples: Number of trajectories to sample
        trajectory_length: Length of each trajectory
        device: torch device
    
    Returns:
        mean_energy: Average energy across samples
        std_energy: Standard deviation
    """
    import random
    
    # Sample random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    energies = []
    
    for idx in tqdm(indices, desc="Computing training energy", leave=False):
        sample = dataset[idx]
        qpos = sample['seq_qpos'][:trajectory_length].to(device)  # [T, qpos_dim]
        mom = sample['seq_mom'][:trajectory_length].to(device)    # [T, mom_dim]
        
        # Compute energy for each timestep
        with torch.no_grad():
            energy = hnn(mom, qpos)  # [T, 1]
        
        energies.append(energy.mean().item())
    
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    
    return mean_energy, std_energy


def compute_mse_batch(model, state_tensor, torque_tensor):
    """Compute MSE for a batch of trajectories (no plotting)."""
    mse_qpos_list = []
    mse_mom_list = []
    
    for i in range(state_tensor.shape[0]):
        state_np = state_tensor[i].cpu().numpy()
        torque_np = torque_tensor[i].cpu().numpy()
        generated = {
            'seq_qpos': state_np[:, :model.qpos_dim],
            'seq_mom': state_np[:, model.qpos_dim:],
            'seq_torque': torque_np,
        }
        mse_dict = compare_generated_with_reconstructed(
            generated, 
            str(project_root / 'configs/rigid_arm_hinge.xml'),
            str(project_root / 'plots'),
            dt=float(model.dt),
            data_dt=float(model.data_dt),
            name=None  # No plotting
        )
        mse_qpos_list.append(mse_dict['mse_qpos'])
        mse_mom_list.append(mse_dict['mse_mom'])
    
    return {
        'qpos': np.mean(mse_qpos_list),
        'mom': np.mean(mse_mom_list),
        'total': np.mean(mse_qpos_list) + np.mean(mse_mom_list),
        'qpos_std': np.std(mse_qpos_list),
        'mom_std': np.std(mse_mom_list),
    }


def run_trajectory_length_experiment(
    model, hnn, torques, dataset, device, 
    trajectory_lengths, num_samples, batch_size, num_diffusion_steps,
    guidance_steps, guidance_lr, guidance_after_steps, output_dir
):
    """Run the trajectory length experiment."""
    
    print("\n" + "="*80)
    print("  TRAJECTORY LENGTH EXPERIMENT")
    print("="*80)
    print(f"  Trajectory lengths: {trajectory_lengths}")
    print(f"  Num samples: {num_samples}")
    print(f"  Guidance config: adam, steps={guidance_steps}, lr={guidance_lr}, after={guidance_after_steps}")
    print("="*80)
    
    seed = 228
    results = []
    
    for traj_length in trajectory_lengths:
        print(f"\n{'='*60}")
        print(f"  Trajectory Length: {traj_length}")
        print(f"{'='*60}")
        
        # Truncate torques to desired length
        traj_torques = torques[:num_samples, :traj_length, :]
        
        # 1. Compute ground-truth training energy for this trajectory length
        print("  Computing ground-truth training energy...")
        training_energy_mean, training_energy_std = compute_training_energy(
            dataset, hnn, num_samples=100, trajectory_length=traj_length, device=device
        )
        print(f"    Training energy: {training_energy_mean:.4f} ± {training_energy_std:.4f}")
        
        # 2. Generate baseline (no guidance)
        print("  Generating baseline trajectories (no guidance)...")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        baseline_states = []
        baseline_torques_out = []
        
        for batch_start in tqdm(range(0, num_samples, batch_size), desc="  Baseline", leave=False):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_torques = traj_torques[batch_start:batch_end]
            
            state, torque_out = model.sample_trajectories(
                num_samples=batch_torques.shape[0],
                trajectory_length=traj_length,
                num_diffusion_steps=num_diffusion_steps,
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
        
        # 3. Generate energy-guided trajectories (same seed, same torque)
        print("  Generating energy-guided trajectories...")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        guided_states = []
        guided_torques_out = []
        
        # Scale guidance_after_steps proportionally to trajectory length
        scaled_after_steps = int(guidance_after_steps * num_diffusion_steps / 50)
        
        for batch_start in tqdm(range(0, num_samples, batch_size), desc="  Guided", leave=False):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_torques = traj_torques[batch_start:batch_end]
            
            state, torque_out = model.sample_trajectories(
                num_samples=batch_torques.shape[0],
                trajectory_length=traj_length,
                num_diffusion_steps=num_diffusion_steps,
                context_fraction=0.2,
                use_ema=False,
                sampler='ddim',
                guidance_scale=1.0,
                hnn=hnn,
                guidance_method='adam',
                guidance_after_steps=scaled_after_steps,
                guidance_steps=guidance_steps,
                guidance_lr=guidance_lr,
                lambda_init=0.0,  # Disable e3 regularization term
                torque=batch_torques,
            )
            guided_states.append(state)
            guided_torques_out.append(torque_out)
        
        guided_state = torch.cat(guided_states, dim=0)
        guided_torque = torch.cat(guided_torques_out, dim=0)
        
        # 4. Compute MSE vs physics reconstruction
        print("  Computing MSE (physics reconstruction)...")
        baseline_mse = compute_mse_batch(model, baseline_state, baseline_torque)
        guided_mse = compute_mse_batch(model, guided_state, guided_torque)
        
        mse_improvement = (baseline_mse['total'] - guided_mse['total']) / baseline_mse['total'] * 100
        
        print(f"    Baseline MSE: {baseline_mse['total']:.6f}")
        print(f"    Guided MSE:   {guided_mse['total']:.6f}")
        print(f"    Improvement:  {mse_improvement:+.1f}%")
        
        # 5. Compute average HNN energy
        print("  Computing average HNN energy...")
        baseline_energy_mean, baseline_energy_std = compute_avg_hnn_energy(
            hnn, baseline_state, model.qpos_dim
        )
        guided_energy_mean, guided_energy_std = compute_avg_hnn_energy(
            hnn, guided_state, model.qpos_dim
        )
        
        print(f"    Training energy: {training_energy_mean:.4f}")
        print(f"    Baseline energy: {baseline_energy_mean:.4f}")
        print(f"    Guided energy:   {guided_energy_mean:.4f}")
        
        # Store results
        results.append({
            'trajectory_length': traj_length,
            'baseline_mse_qpos': baseline_mse['qpos'],
            'baseline_mse_mom': baseline_mse['mom'],
            'baseline_mse_total': baseline_mse['total'],
            'guided_mse_qpos': guided_mse['qpos'],
            'guided_mse_mom': guided_mse['mom'],
            'guided_mse_total': guided_mse['total'],
            'mse_improvement_pct': mse_improvement,
            'training_energy_mean': training_energy_mean,
            'training_energy_std': training_energy_std,
            'baseline_energy_mean': baseline_energy_mean,
            'baseline_energy_std': baseline_energy_std,
            'guided_energy_mean': guided_energy_mean,
            'guided_energy_std': guided_energy_std,
        })
    
    # Save results
    df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, 'trajectory_length_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary table
    print("\n" + "="*100)
    print("  RESULTS SUMMARY")
    print("="*100)
    print(f"{'Traj Len':>10} | {'Baseline MSE':>12} | {'Guided MSE':>12} | {'Improv %':>10} | "
          f"{'Train E':>10} | {'Base E':>10} | {'Guided E':>10}")
    print("-"*100)
    
    for _, row in df.iterrows():
        print(f"{int(row['trajectory_length']):>10} | "
              f"{row['baseline_mse_total']:>12.6f} | "
              f"{row['guided_mse_total']:>12.6f} | "
              f"{row['mse_improvement_pct']:>+10.1f} | "
              f"{row['training_energy_mean']:>10.2f} | "
              f"{row['baseline_energy_mean']:>10.2f} | "
              f"{row['guided_energy_mean']:>10.2f}")
    
    print("="*100)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Trajectory Length Experiments")
    parser.add_argument("--test_torques", type=str, default="data/test_torques_2000.h5",
                        help="Path to test torques HDF5 file")
    parser.add_argument("--training_data", type=str, 
                        default="/home/gsang/Projects/Perceiver_IO/data/traj_40000-steps_4000.h5",
                        help="Path to training dataset for ground-truth energy")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of samples to use from test set")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for sampling")
    parser.add_argument("--num_diffusion_steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--checkpoint", type=str, 
                        default="/home/gsang/Projects/Perceiver_IO/checkpoints/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt",
                        help="Path to model checkpoint")
    parser.add_argument("--hnn_checkpoint", type=str,
                        default="/home/gsang/Projects/Perceiver_IO/checkpoints/SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt",
                        help="Path to HNN checkpoint")
    parser.add_argument("--output_dir", type=str, default="output_traj_lengths",
                        help="Output directory for results")
    
    # Best guidance config from comprehensive sweep
    parser.add_argument("--guidance_steps", type=int, default=25,
                        help="Number of Adam optimization steps for guidance")
    parser.add_argument("--guidance_lr", type=float, default=0.01,
                        help="Learning rate for guidance optimization")
    parser.add_argument("--guidance_after_steps", type=int, default=45,
                        help="Start guidance after this many diffusion steps (for 50 total steps)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (e.g. cuda:0, cuda:2)")
    parser.add_argument("--trajectory_lengths", type=int, nargs="+", default=None,
                        help="Trajectory lengths to test (e.g. 1050 1100 1150)")

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and HNN
    model, hnn = load_model_and_hnn(args.checkpoint, args.hnn_checkpoint, device)
    
    # Load test torques
    torque_path = project_root / args.test_torques
    torques = load_test_torques(str(torque_path), device)
    
    # Load training dataset for ground-truth energy
    print(f"Loading training dataset from: {args.training_data}")
    dataset = TrajectoryDPFCached(args.training_data, trajectory_length=1000)
    print(f"  Loaded {len(dataset)} training trajectories")
    
    # Trajectory lengths to test (can be overridden via CLI)
    # Trained lengths [100, 200, 300, ..., 1000]
    trajectory_lengths = args.trajectory_lengths if args.trajectory_lengths else [1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500]
    
    # Run experiment
    results = run_trajectory_length_experiment(
        model=model,
        hnn=hnn,
        torques=torques,
        dataset=dataset,
        device=device,
        trajectory_lengths=trajectory_lengths,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_diffusion_steps=args.num_diffusion_steps,
        guidance_steps=args.guidance_steps,
        guidance_lr=args.guidance_lr,
        guidance_after_steps=args.guidance_after_steps,
        output_dir=args.output_dir,
    )
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
