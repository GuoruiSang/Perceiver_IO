"""
Utility functions for Trajectory DPF training.

This module contains reusable utility functions:
- compute_normalization_stats(): Compute min-max normalization statistics
"""

import torch
from tqdm import tqdm


def compute_normalization_stats(dataloader, qpos_dim, qvel_dim, qacc_dim, torque_dim, max_timesteps):
    """
    Compute min and max for min-max normalization to [-1, 1].
    
    Normalization is computed per dimension (global across time and trajectories).
    Returns statistics with shape [dim] for qpos, qvel, qacc, and torque.
    """
    print("Computing min-max normalization statistics (per dimension, global across time)...")
    
    # Initialize with extreme values for all components
    qpos_min = torch.full((qpos_dim,), float('inf'))
    qpos_max = torch.full((qpos_dim,), float('-inf'))
    qvel_min = torch.full((qvel_dim,), float('inf'))
    qvel_max = torch.full((qvel_dim,), float('-inf'))
    qacc_min = torch.full((qacc_dim,), float('inf'))
    qacc_max = torch.full((qacc_dim,), float('-inf'))
    torque_min = torch.full((torque_dim,), float('inf'))
    torque_max = torch.full((torque_dim,), float('-inf'))
    count = 0
    
    for batch in tqdm(dataloader, desc="Computing stats"):
        # Get batch data (using seq_ prefix from TrajectoryDPFCached)
        qpos = batch['seq_qpos']  # [B, T, qpos_dim]
        qvel = batch['seq_qvel']  # [B, T, qvel_dim]
        qacc = batch['seq_qacc']  # [B, T, qacc_dim]
        torque = batch['seq_torque']  # [B, T, torque_dim]
        
        B = qpos.shape[0]
        
        # Update min/max across both batch and time dimensions
        # Take min/max over dimensions 0 (batch) and 1 (time), leaving only dimension
        qpos_min = torch.min(qpos_min, qpos.min(dim=0)[0].min(dim=0)[0])
        qpos_max = torch.max(qpos_max, qpos.max(dim=0)[0].max(dim=0)[0])
        qvel_min = torch.min(qvel_min, qvel.min(dim=0)[0].min(dim=0)[0])
        qvel_max = torch.max(qvel_max, qvel.max(dim=0)[0].max(dim=0)[0])
        qacc_min = torch.min(qacc_min, qacc.min(dim=0)[0].min(dim=0)[0])
        qacc_max = torch.max(qacc_max, qacc.max(dim=0)[0].max(dim=0)[0])
        torque_min = torch.min(torque_min, torque.min(dim=0)[0].min(dim=0)[0])
        torque_max = torch.max(torque_max, torque.max(dim=0)[0].max(dim=0)[0])
        count += B
    
    # Compute ranges
    qpos_range = qpos_max - qpos_min
    qvel_range = qvel_max - qvel_min
    qacc_range = qacc_max - qacc_min
    torque_range = torque_max - torque_min

    # Ensure minimum range for stability
    from src import config
    range_epsilon = config.DEFAULT_NORMALIZATION_RANGE_EPSILON
    qpos_max = torch.where(qpos_range < range_epsilon, qpos_min + range_epsilon, qpos_max)
    qvel_max = torch.where(qvel_range < range_epsilon, qvel_min + range_epsilon, qvel_max)
    qacc_max = torch.where(qacc_range < range_epsilon, qacc_min + range_epsilon, qacc_max)
    torque_max = torch.where(torque_range < range_epsilon, torque_min + range_epsilon, torque_max)
    
    # Recompute ranges after fix
    qpos_range = qpos_max - qpos_min
    qvel_range = qvel_max - qvel_min
    qacc_range = qacc_max - qacc_min
    torque_range = torque_max - torque_min
    
    print(f"\n{'='*80}")
    print(f"MIN-MAX NORMALIZATION STATISTICS")
    print(f"{'='*80}")
    print(f"Computed stats for {count} trajectories, {max_timesteps} timesteps each")
    print(f"\nSTATISTICS:")
    print(f"  qpos_min:   {qpos_min.tolist()}")
    print(f"  qpos_max:   {qpos_max.tolist()}")
    print(f"  qpos_range: min={qpos_range.min():.6f}, max={qpos_range.max():.6f}, mean={qpos_range.mean():.6f}")
    print(f"  qvel_min:   {qvel_min.tolist()}")
    print(f"  qvel_max:   {qvel_max.tolist()}")
    print(f"  qvel_range: min={qvel_range.min():.6f}, max={qvel_range.max():.6f}, mean={qvel_range.mean():.6f}")
    print(f"  qacc_min:   {qacc_min.tolist()}")
    print(f"  qacc_max:   {qacc_max.tolist()}")
    print(f"  qacc_range: min={qacc_range.min():.6f}, max={qacc_range.max():.6f}, mean={qacc_range.mean():.6f}")
    print(f"  torque_min:   {torque_min.tolist()}")
    print(f"  torque_max:   {torque_max.tolist()}")
    print(f"  torque_range: min={torque_range.min():.6f}, max={torque_range.max():.6f}, mean={torque_range.mean():.6f}")
    
    # Check for dimensions with very small ranges (essentially constant)
    print(f"\n{'='*80}")
    print(f"DIMENSIONS WITH SMALL RANGE (range < 0.01):")
    print(f"{'='*80}")
    
    small_range_count = 0
    for dim in range(qpos_dim):
        dim_range = qpos_range[dim].item()
        if dim_range < 0.01:
            print(f"  qpos[{dim}]: range={dim_range:.6f} (nearly constant!)")
            print(f"              values: [{qpos_min[dim]:.6f}, {qpos_max[dim]:.6f}]")
            small_range_count += 1
    
    for dim in range(qvel_dim):
        dim_range = qvel_range[dim].item()
        if dim_range < 0.01:
            print(f"  qvel[{dim}]: range={dim_range:.6f} (nearly constant!)")
            print(f"              values: [{qvel_min[dim]:.6f}, {qvel_max[dim]:.6f}]")
            small_range_count += 1
    
    for dim in range(qacc_dim):
        dim_range = qacc_range[dim].item()
        if dim_range < 0.01:
            print(f"  qacc[{dim}]: range={dim_range:.6f} (nearly constant!)")
            print(f"              values: [{qacc_min[dim]:.6f}, {qacc_max[dim]:.6f}]")
            small_range_count += 1
    
    for dim in range(torque_dim):
        dim_range = torque_range[dim].item()
        if dim_range < 0.01:
            print(f"  torque[{dim}]: range={dim_range:.6f} (nearly constant!)")
            print(f"                values: [{torque_min[dim]:.6f}, {torque_max[dim]:.6f}]")
            small_range_count += 1
    
    if small_range_count == 0:
        print("  None found! All dimensions have sufficient variation.")
    
    print(f"{'='*80}\n")
    
    return qpos_min, qpos_max, qvel_min, qvel_max, qacc_min, qacc_max, torque_min, torque_max
