"""
Context Fraction Experiments

Test how different context fractions affect the MSE of generated trajectories (without guidance).
Fixed trajectory length = 1000.

Usage:
    python scripts/evaluate_context_fractions.py --num_samples 2000 --output_dir output_context_fractions
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


def load_model(checkpoint_path, device):
    """Load the trajectory model from checkpoint."""
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
    
    return model


def load_test_torques(torque_file, device):
    """Load pre-generated test torques from HDF5."""
    print(f"Loading test torques from: {torque_file}")
    with h5py.File(torque_file, 'r') as f:
        torques = torch.tensor(f['torques'][:], dtype=torch.float32, device=device)
        print(f"  Loaded {torques.shape[0]} torque sequences, shape: {torques.shape}")
    return torques


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


def run_context_fraction_experiment(
    model, torques, device, 
    context_fractions, trajectory_length, num_samples, batch_size, num_diffusion_steps,
    output_dir
):
    """Run the context fraction experiment."""
    
    print("\n" + "="*80)
    print("  CONTEXT FRACTION EXPERIMENT (No Guidance)")
    print("="*80)
    print(f"  Trajectory length: {trajectory_length}")
    print(f"  Context fractions: {context_fractions}")
    print(f"  Num samples: {num_samples}")
    print(f"  Diffusion steps: {num_diffusion_steps}")
    print("="*80)
    
    seed = 228
    results = []
    
    # Truncate torques to desired length
    traj_torques = torques[:num_samples, :trajectory_length, :]
    
    for ctx_frac in context_fractions:
        print(f"\n{'='*60}")
        print(f"  Context Fraction: {ctx_frac}")
        print(f"{'='*60}")
        
        # Reset seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        all_states = []
        all_torques = []
        
        # Generate trajectories without guidance
        for batch_start in tqdm(range(0, num_samples, batch_size), desc=f"  ctx={ctx_frac}", leave=False):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_torques = traj_torques[batch_start:batch_end]
            
            state, torque_out = model.sample_trajectories(
                num_samples=batch_torques.shape[0],
                trajectory_length=trajectory_length,
                num_diffusion_steps=num_diffusion_steps,
                context_fraction=ctx_frac,
                use_ema=False,
                sampler='ddim',
                guidance_scale=1.0,
                hnn=None,  # No guidance
                guidance_steps=0,
                torque=batch_torques,
            )
            all_states.append(state)
            all_torques.append(torque_out)
        
        all_state = torch.cat(all_states, dim=0)
        all_torque = torch.cat(all_torques, dim=0)
        
        # Compute MSE vs physics reconstruction
        print("  Computing MSE (physics reconstruction)...")
        mse = compute_mse_batch(model, all_state, all_torque)
        
        print(f"    MSE qpos: {mse['qpos']:.6f}")
        print(f"    MSE mom:  {mse['mom']:.6f}")
        print(f"    MSE total: {mse['total']:.6f}")
        
        # Store results
        results.append({
            'context_fraction': ctx_frac,
            'mse_qpos': mse['qpos'],
            'mse_mom': mse['mom'],
            'mse_total': mse['total'],
            'mse_qpos_std': mse['qpos_std'],
            'mse_mom_std': mse['mom_std'],
        })
    
    # Save results
    df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, 'context_fraction_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("  RESULTS SUMMARY")
    print("="*80)
    print(f"{'Context Frac':>12} | {'MSE qpos':>12} | {'MSE mom':>12} | {'MSE total':>12}")
    print("-"*80)
    
    for _, row in df.iterrows():
        print(f"{row['context_fraction']:>12.2f} | "
              f"{row['mse_qpos']:>12.6f} | "
              f"{row['mse_mom']:>12.6f} | "
              f"{row['mse_total']:>12.6f}")
    
    print("="*80)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Context Fraction Experiments")
    parser.add_argument("--test_torques", type=str, default="data/test_torques_2000.h5",
                        help="Path to test torques HDF5 file")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of samples to use from test set")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Batch size for sampling")
    parser.add_argument("--num_diffusion_steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--trajectory_length", type=int, default=1000,
                        help="Fixed trajectory length")
    parser.add_argument("--checkpoint", type=str, 
                        default="/home/gsang/Projects/Perceiver_IO/checkpoints/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt",
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="output_context_fractions",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load test torques
    torque_path = project_root / args.test_torques
    torques = load_test_torques(str(torque_path), device)
    
    # Context fractions to test
    context_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Run experiment
    results = run_context_fraction_experiment(
        model=model,
        torques=torques,
        device=device,
        context_fractions=context_fractions,
        trajectory_length=args.trajectory_length,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_diffusion_steps=args.num_diffusion_steps,
        output_dir=args.output_dir,
    )
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
