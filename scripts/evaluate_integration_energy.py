import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from src.models.HNN import HNNWrapper
from scripts.dataset import TrajectoryDPFCached
from src.models.utils import compute_chunked_integration_energy

def evaluate_integration_energy():
    # Configuration
    hnn_checkpoint = "/home/gsang/Projects/Perceiver_IO/checkpoints/SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt"
    data_path = "/home/gsang/Projects/Perceiver_IO/data/traj_40000-steps_4000.h5"
    chunk_lengths = [1, 5, 10, 20, 50]
    batch_size = 64
    num_batches = 10  # Evaluate on ~640 trajectories
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load HNN Model
    print(f"Loading HNN from {hnn_checkpoint}...")
    hnn = HNNWrapper.load_from_checkpoint(hnn_checkpoint, map_location=device)
    hnn.to(device)
    hnn.eval()
    print("HNN loaded successfully.")
    
    # Load Dataset
    print(f"Loading dataset from {data_path}...")
    # Using TrajectoryDPFCached to get full sequences
    dataset = TrajectoryDPFCached(data_path, trajectory_length=1000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset loaded. Total trajectories: {len(dataset)}")
    
    # Get simulation timestep from dataset
    dt = dataset.dt
    print(f"Simulation timestep (dt): {dt}")
    
    results = {}
    
    # Evaluation Loop
    print("\nStarting evaluation...")
    
    for length in chunk_lengths:
        print(f"\nEvaluating chunk length: {length}")
        energies = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc=f"Chunks len={length}", total=num_batches)):
                if i >= num_batches:
                    break
                
                seq_qpos = batch['seq_qpos'].to(device).float()
                seq_mom = batch['seq_mom'].to(device).float()
                seq_torque = batch['seq_torque'].to(device).float()
                
                energy = compute_chunked_integration_energy(
                    seq_qpos, seq_mom, seq_torque, hnn, dt, length
                )
                energies.append(energy.item())
        
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        results[length] = {'mean': mean_energy, 'std': std_energy}
        
        print(f"Chunk Length {length}: Mean Energy (MSE) = {mean_energy:.6e} ± {std_energy:.6e}")

    # Plotting Results
    print("\nPlotting results...")
    os.makedirs("plots", exist_ok=True)
    
    lengths = sorted(results.keys())
    means = [results[l]['mean'] for l in lengths]
    stds = [results[l]['std'] for l in lengths]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(lengths, means, yerr=stds, fmt='-o', capsize=5)
    plt.xlabel("Integration Horizon (steps)")
    plt.ylabel("Mean Integration Energy (MSE)")
    plt.title("HNN Integration Drift Analysis")
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to see small errors better
    
    plot_path = "plots/hnn_drift_analysis.jpg"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Recommendation
    print("\n--- Summary ---")
    for l in lengths:
        print(f"Horizon {l}: {results[l]['mean']:.6e}")
        
    print("\nRecommendation:")
    # Simple heuristic: find largest horizon where error is still 'acceptable' (e.g. < 1e-4 or inflection point)
    # This is qualitative, but helps interpretation.
    acceptable_threshold = 1e-3 
    valid_lengths = [l for l in lengths if results[l]['mean'] < acceptable_threshold]
    if valid_lengths:
        print(f"Horizons up to {max(valid_lengths)} steps have error < {acceptable_threshold}.")
    else:
        print(f"All tested horizons have error > {acceptable_threshold}. HNN might be too inaccurate for long-term refinement.")

if __name__ == "__main__":
    evaluate_integration_energy()
