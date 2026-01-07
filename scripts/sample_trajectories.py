"""
Script to generate trajectories from a trained Trajectory DPF model.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
from src.models.trajectory_dpf import TrajectoryDPF
import os


def main():
    parser = argparse.ArgumentParser(description="Generate trajectories using trained Trajectory DPF")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output_path", type=str, 
                        default="/home/gsang/Projects/Perceiver_IO/output/generated_trajectories.h5",
                        help="Path to save generated trajectories")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of trajectories to generate")
    parser.add_argument("--trajectory_length", type=int, default=1000,
                        help="Length of each trajectory")
    parser.add_argument("--num_diffusion_steps", type=int, default=50,
                        help="Number of DDIM denoising steps")
    parser.add_argument("--context_fraction", type=float, default=0.5,
                        help="Fraction of timesteps to use as context")
    parser.add_argument("--resample_context", action="store_true",
                        help="Resample context/query split at each denoising step")
    parser.add_argument("--use_ema", action="store_true", default=True,
                        help="Use EMA weights for sampling")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of trajectories to generate in each batch")
    # Guidance parameters
    parser.add_argument("--guidance_method", type=str, choices=["adam", "langevin"], default="adam",
                        help="Guidance method: 'adam' or 'langevin'")
    parser.add_argument("--guidance_after_steps", type=int, default=0,
                        help="Start guidance after this many diffusion steps")
    parser.add_argument("--guidance_steps", type=int, default=0,
                        help="Number of optimization steps per diffusion step (0 = disabled)")
    args = parser.parse_args()
    
    # Load model from checkpoint
    print(f"Loading model from {args.checkpoint}...")
    model = TrajectoryDPF.load_from_checkpoint(args.checkpoint)
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model configuration:")
    print(f"  qpos_dim: {model.qpos_dim}")
    print(f"  mom_dim: {model.mom_dim}")
    print(f"  torque_dim: {model.torque_dim}")
    print(f"  diffusion_steps: {model.diffusion_steps}")
    print(f"  max_timesteps: {model.max_timesteps}")
    
    # Generate trajectories in batches
    all_states = []
    all_torques = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for i in range(num_batches):
        batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
        print(f"\nGenerating batch {i+1}/{num_batches} ({batch_size} samples)...")
        
        state, torque = model.sample_trajectories(
            num_samples=batch_size,
            trajectory_length=args.trajectory_length,
            num_diffusion_steps=args.num_diffusion_steps,
            context_fraction=args.context_fraction,
            use_ema=args.use_ema,
            resample_context_every_step=args.resample_context,
            guidance_method=args.guidance_method,
            guidance_after_steps=args.guidance_after_steps,
            guidance_steps=args.guidance_steps,
        )
        
        all_states.append(state.cpu())
        all_torques.append(torque.cpu())
    
    # Concatenate all batches
    all_states = torch.cat(all_states, dim=0)
    all_torques = torch.cat(all_torques, dim=0)
    
    # Save to h5 file
    import h5py
    import numpy as np
    
    qpos = all_states[:, :, :model.qpos_dim].numpy()
    mom = all_states[:, :, model.qpos_dim:].numpy()
    torque = all_torques.numpy()
    
    print(f"\nSaving to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    
    with h5py.File(args.output_path, 'w') as f:
        f.attrs['num_trajectories'] = args.num_samples
        f.attrs['num_steps'] = args.trajectory_length
        
        for i in range(args.num_samples):
            traj_group = f.create_group(f'traj_{i}')
            traj_group.create_dataset('seq_qpos', data=qpos[i], dtype='f8')
            traj_group.create_dataset('seq_mom', data=mom[i], dtype='f8')
            traj_group.create_dataset('seq_torque', data=torque[i], dtype='f8')
        
        # Store metadata in root
        f.attrs['description'] = 'Generated trajectories from Trajectory DPF'
        f.attrs['checkpoint'] = args.checkpoint
    
    print(f"✓ Saved {args.num_samples} trajectories to {args.output_path}")
    print(f"\nDataset shape:")
    print(f"  qpos: {qpos.shape}")
    print(f"  mom: {mom.shape}")
    print(f"  torque: {torque.shape}")


if __name__ == "__main__":
    main()

