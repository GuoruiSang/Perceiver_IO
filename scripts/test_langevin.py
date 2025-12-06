"""
Test script for Langevin dynamics on generated trajectories.
Tests kinematic consistency (qpos -> qvel -> qacc) without torque predictor.
Runs until convergence.
"""

import torch
import torch.nn as nn
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paths import DEFAULT_GENERATED_TRAJECTORIES_PATH
from src.models.utils import visualize_trajectory


def compute_kinematic_consistency_energy(qpos: torch.Tensor, qvel: torch.Tensor, qacc: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute energy measuring kinematic inconsistency (qpos -> qvel -> qacc)."""
    # Central difference for qpos_dot (should equal qvel)
    qpos_dot = torch.zeros_like(qvel)
    qpos_dot[:, 1:-1] = (qpos[:, 2:] - qpos[:, :-2]) / (2 * dt)
    qpos_dot[:, 0] = (-3*qpos[:, 0] + 4*qpos[:, 1] - qpos[:, 2]) / (2*dt)
    qpos_dot[:, -1] = (3*qpos[:, -1] - 4*qpos[:, -2] + qpos[:, -3]) / (2*dt)

    # Central difference for qvel_dot (should equal qacc)
    qvel_dot = torch.zeros_like(qacc)
    qvel_dot[:, 1:-1] = (qvel[:, 2:] - qvel[:, :-2]) / (2 * dt)
    qvel_dot[:, 0] = (-3*qvel[:, 0] + 4*qvel[:, 1] - qvel[:, 2]) / (2*dt)
    qvel_dot[:, -1] = (3*qvel[:, -1] - 4*qvel[:, -2] + qvel[:, -3]) / (2*dt)

    e1 = nn.functional.mse_loss(qpos_dot, qvel)
    e2 = nn.functional.mse_loss(qvel_dot, qacc)

    return e1 + e2


def run_adam_optimization(
    qpos: torch.Tensor, 
    qvel: torch.Tensor, 
    qacc: torch.Tensor,
    torque: torch.Tensor,
    dt: float,
    lr: float = 0.01,
    max_steps: int = 50000,
    patience: int = 1000,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """
    Optimize kinematic consistency using Adam optimizer.
    Much faster convergence than vanilla gradient descent.
    """
    # Create parameters
    qpos_param = nn.Parameter(qpos.clone())
    qvel_param = nn.Parameter(qvel.clone())
    qacc_param = nn.Parameter(qacc.clone())
    
    optimizer = torch.optim.Adam([qpos_param, qvel_param, qacc_param], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, verbose=True
    )
    
    energy_history = []
    best_energy = float('inf')
    best_state = None
    steps_without_improvement = 0
    
    for i in range(max_steps):
        optimizer.zero_grad()
        
        e = compute_kinematic_consistency_energy(qpos_param, qvel_param, qacc_param, dt)
        current_energy = e.item()
        energy_history.append(current_energy)
        
        # Track best
        if current_energy < best_energy - tol:
            best_energy = current_energy
            best_state = (qpos_param.data.clone(), qvel_param.data.clone(), qacc_param.data.clone())
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
        
        if steps_without_improvement >= patience:
            print(f"\nConverged at step {i}: No improvement for {patience} steps")
            break
        
        e.backward()
        optimizer.step()
        scheduler.step(current_energy)
        
        if i % 500 == 0:
            grad_norm = sum(p.grad.norm().item() for p in [qpos_param, qvel_param, qacc_param])
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {i}: Energy = {current_energy:.6f}, Grad norm = {grad_norm:.6f}, LR = {current_lr:.2e}")
    
    if i == max_steps - 1:
        print(f"\nReached max steps ({max_steps})")
    
    # Return best state
    if best_state is not None:
        qpos_out, qvel_out, qacc_out = best_state
    else:
        qpos_out = qpos_param.data
        qvel_out = qvel_param.data
        qacc_out = qacc_param.data
    
    return qpos_out, qvel_out, qacc_out, torque, energy_history


def load_trajectory(h5_path: str, traj_idx: int = 0):
    """Load a single trajectory from h5 file."""
    with h5py.File(h5_path, 'r') as f:
        traj = f[f'traj_{traj_idx}']
        qpos = torch.from_numpy(traj['seq_qpos'][:]).float()
        qvel = torch.from_numpy(traj['seq_qvel'][:]).float()
        qacc = torch.from_numpy(traj['seq_qacc'][:]).float()
        torque = torch.from_numpy(traj['seq_torque'][:]).float()
    return qpos, qvel, qacc, torque


def plot_energy_curve(energy_history, save_path: str):
    """Plot energy convergence curve."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(energy_history)
    plt.xlabel('Langevin Step')
    plt.ylabel('Kinematic Consistency Energy')
    plt.title('Energy During Langevin Dynamics')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(energy_history)
    plt.xlabel('Langevin Step')
    plt.ylabel('Kinematic Consistency Energy (log scale)')
    plt.title('Energy During Langevin Dynamics')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved energy plot to {save_path}")


def main():
    # Configuration
    h5_path = str(DEFAULT_GENERATED_TRAJECTORIES_PATH)
    dt = 0.002  # MuJoCo default timestep
    lr = 0.01  # Adam learning rate
    max_steps = 50000
    patience = 2000  # Stop if no improvement for this many steps
    tol = 1e-7  # Convergence tolerance
    
    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading trajectory from {h5_path}...")
    qpos, qvel, qacc, torque = load_trajectory(h5_path, traj_idx=0)
    
    print(f"Trajectory shape: qpos={qpos.shape}, qvel={qvel.shape}, qacc={qacc.shape}, torque={torque.shape}")
    
    # Save original trajectory visualization
    print("\nSaving original trajectory visualization...")
    original_traj = {
        'seq_qpos': qpos,
        'seq_qvel': qvel,
        'seq_qacc': qacc,
        'seq_torque': torque,
    }
    visualize_trajectory(original_traj, str(output_dir))
    import os
    os.rename(str(output_dir / 'trajectory.jpg'), str(output_dir / 'langevin_before.jpg'))
    print(f"Saved to {output_dir / 'langevin_before.jpg'}")
    
    # Add batch dimension
    qpos = qpos.unsqueeze(0)  # [1, T, dim]
    qvel = qvel.unsqueeze(0)
    qacc = qacc.unsqueeze(0)
    torque = torque.unsqueeze(0)
    
    # Compute initial energy
    initial_energy = compute_kinematic_consistency_energy(qpos, qvel, qacc, dt)
    print(f"\nInitial kinematic consistency energy: {initial_energy.item():.6f}")
    
    # Run Adam optimization until convergence
    print(f"\nRunning Adam optimization (max {max_steps} steps, patience={patience})...")
    print(f"Learning rate: {lr}, Convergence tolerance: {tol}")
    print("-" * 60)
    
    qpos_refined, qvel_refined, qacc_refined, torque_out, energy_history = run_adam_optimization(
        qpos, qvel, qacc, torque, dt, 
        lr=lr, 
        max_steps=max_steps,
        patience=patience,
        tol=tol,
    )
    
    # Compute final energy
    final_energy = compute_kinematic_consistency_energy(qpos_refined, qvel_refined, qacc_refined, dt)
    print("-" * 60)
    print(f"\nFinal kinematic consistency energy: {final_energy.item():.6f}")
    print(f"Energy reduction: {(1 - final_energy.item() / initial_energy.item()) * 100:.2f}%")
    print(f"Total steps: {len(energy_history)}")
    
    # Save refined trajectory visualization
    print("\nSaving refined trajectory visualization...")
    refined_traj = {
        'seq_qpos': qpos_refined.squeeze(0),
        'seq_qvel': qvel_refined.squeeze(0),
        'seq_qacc': qacc_refined.squeeze(0),
        'seq_torque': torque_out.squeeze(0),
    }
    visualize_trajectory(refined_traj, str(output_dir))
    os.rename(str(output_dir / 'trajectory.jpg'), str(output_dir / 'langevin_after.jpg'))
    print(f"Saved to {output_dir / 'langevin_after.jpg'}")
    
    # Plot energy curve
    plot_energy_curve(energy_history, str(output_dir / 'langevin_energy.png'))
    
    print("\nDone!")


if __name__ == "__main__":
    main()
