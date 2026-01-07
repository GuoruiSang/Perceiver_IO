"""
Test script to verify HNN physics energy on training data.

If the HNN correctly models the physics, the energy on ground truth trajectories
should be near zero (the trajectories already satisfy Hamilton's equations).

Energy = MSE(dot_qpos, dH/dp) + MSE(dot_mom, -dH/dq + torque)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from scripts.dataset import TrajectoryDPFCached
from src.models.HNN import HNNWrapper
from src.models.utils import compute_hnn_physics_energy, central_difference


def main():
    # Configuration
    hnn_checkpoint_path = '/home/gsang/Projects/Perceiver_IO/Projects/Perceiver_IO/checkpoints/SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt'
    data_path = '/home/gsang/Projects/Perceiver_IO/data/traj_4000-steps_4000.h5'
    trajectory_length = 1000
    num_test_trajectories = 10
    data_dt = 0.00025  # Data collection timestep
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load HNN using HNNWrapper (which includes input scaling)
    print(f"\nLoading HNN from: {hnn_checkpoint_path}")
    
    # Load the full HNNWrapper from checkpoint (preserves q_std, p_std scaling)
    hnn = HNNWrapper.load_from_checkpoint(hnn_checkpoint_path, map_location=device)
    hnn = hnn.to(device)
    hnn.eval()
    
    qpos_dim = hnn.hparams.get('coordinate_dim', 3)
    mom_dim = hnn.hparams.get('momenta_dim', 3)
    
    print(f"Detected dimensions: qpos_dim={qpos_dim}, mom_dim={mom_dim}")
    print(f"Scaling stats: q_std={hnn.q_std.mean().item():.4f}, p_std={hnn.p_std.mean().item():.4f}")
    print("✓ HNNWrapper loaded successfully (with scaling)")
    
    # Load dataset
    print(f"\nLoading dataset from: {data_path}")
    dataset = TrajectoryDPFCached(data_path, trajectory_length=trajectory_length)
    print(f"Dataset size: {len(dataset)} trajectories")
    
    # Test energy on training trajectories
    print(f"\n{'='*70}")
    print(f"Testing HNN Physics Energy on {num_test_trajectories} Training Trajectories")
    print(f"{'='*70}")
    print(f"dt = {data_dt}")
    print(f"Trajectory length = {trajectory_length}")
    print()
    
    energies = []
    e1_list = []  # Position derivative consistency
    e2_list = []  # Momentum derivative consistency
    
    for i in range(min(num_test_trajectories, len(dataset))):
        sample = dataset[i]
        seq_qpos = sample['seq_qpos'].unsqueeze(0).to(device)  # [1, T, qpos_dim]
        seq_mom = sample['seq_mom'].unsqueeze(0).to(device)    # [1, T, mom_dim]
        seq_torque = sample['seq_torque'].unsqueeze(0).to(device)  # [1, T, torque_dim]
        
        # Compute HNN physics energy (no regularization term since this is ground truth)
        with torch.enable_grad():
            seq_qpos_grad = seq_qpos.clone().requires_grad_(True)
            seq_mom_grad = seq_mom.clone().requires_grad_(True)
            
            energy = compute_hnn_physics_energy(
                seq_qpos_grad, 
                seq_mom_grad, 
                seq_torque, 
                hnn, 
                dt=data_dt,
                lambda_init=0.0,  # No regularization for ground truth test
            )
        
        # Also compute individual components for diagnostics
        with torch.no_grad():
            # Compute derivatives from trajectory
            dot_qpos = central_difference(seq_qpos, data_dt)
            dot_mom = central_difference(seq_mom, data_dt)
            
            # Compute HNN predictions
            seq_mom_g = seq_mom.clone().requires_grad_(True)
            seq_qpos_g = seq_qpos.clone().requires_grad_(True)
            
            with torch.enable_grad():
                H = hnn(seq_mom_g, seq_qpos_g)
                dH_dp, dH_dq = torch.autograd.grad(
                    H.sum(), (seq_mom_g, seq_qpos_g), create_graph=False
                )
            
            dot_qpos_pred = dH_dp
            dot_mom_pred = -dH_dq + seq_torque
            
            e1 = nn.functional.mse_loss(dot_qpos, dot_qpos_pred).item()
            e2 = nn.functional.mse_loss(dot_mom, dot_mom_pred).item()
        
        energies.append(energy.item())
        e1_list.append(e1)
        e2_list.append(e2)
        
        print(f"Trajectory {i:3d}: Total Energy = {energy.item():.6e}  "
              f"(e1_qvel={e1:.4e}, e2_mom={e2:.4e})")
    
    # Summary statistics
    energies = np.array(energies)
    e1_arr = np.array(e1_list)
    e2_arr = np.array(e2_list)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total Energy (e1 + e2):")
    print(f"  Mean:   {energies.mean():.6e}")
    print(f"  Std:    {energies.std():.6e}")
    print(f"  Min:    {energies.min():.6e}")
    print(f"  Max:    {energies.max():.6e}")
    print()
    print(f"e1 = MSE(dot_qpos, dH/dp):")
    print(f"  Mean:   {e1_arr.mean():.6e}")
    print(f"  Std:    {e1_arr.std():.6e}")
    print()
    print(f"e2 = MSE(dot_mom, -dH/dq + torque):")
    print(f"  Mean:   {e2_arr.mean():.6e}")
    print(f"  Std:    {e2_arr.std():.6e}")
    print()
    
    # Interpretation
    print(f"{'='*70}")
    print(f"INTERPRETATION")
    print(f"{'='*70}")
    if energies.mean() < 1.0:
        print("✓ Energy is reasonably low - HNN captures the physics well")
    elif energies.mean() < 10.0:
        print("⚠ Energy is moderate - HNN has some modeling error")
    else:
        print("✗ Energy is high - significant mismatch between HNN and data")
    
    print()
    print("Note: The energy should NOT be exactly 0 because:")
    print("  1. HNN training has finite error (nMSE ~0.2% from TEST 1)")
    print("  2. mom_dot in dataset is computed via finite differences, not exact")
    print("  3. The energy MSE amplifies small per-timestep errors over the trajectory")
    print()
    print("Expected range based on HNN test results:")
    print("  - e1 (qvel): ~0.03 (from MSE(qvel, dH/dp) in HNN test)")
    print("  - e2 (mom_dot): ~0.8 (from MSE(mom_dot, -dH/dq + tau) in HNN test)")
    print("  - Total: ~0.8-1.0")


if __name__ == '__main__':
    main()

