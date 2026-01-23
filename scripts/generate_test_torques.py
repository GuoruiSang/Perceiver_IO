"""
Generate a test set of random torque sequences for hyperparameter sweep evaluation.

Uses the same torque generation method as trajectory_dpf.py for consistency.
Saves to HDF5 with metadata for reproducibility.

Usage:
    python scripts/generate_test_torques.py --output data/test_torques_2000.h5 --num_samples 2000
"""

import argparse
import numpy as np
import h5py
import math
import os


def generate_random_torque(
    num_samples: int,
    trajectory_length: int,
    torque_dim: int,
    dt: float,
    seed: int,
    num_sin: int = 5,
    lim_amplitude: float = 0.5,
    lim_frequency: float = 6 * math.pi,
    lim_phase: float = 2 * math.pi
) -> np.ndarray:
    """
    Generate random smooth torque sequences using sum of sinusoids.
    
    This matches the _generate_random_torque method in trajectory_dpf.py.
    
    Args:
        num_samples: Number of torque sequences to generate
        trajectory_length: Length of each sequence
        torque_dim: Dimension of torque (typically 3 for 3-joint arm)
        dt: Timestep between samples
        seed: Random seed for reproducibility
        num_sin: Number of sinusoids to sum
        lim_amplitude: Maximum amplitude for each sinusoid
        lim_frequency: Maximum frequency for each sinusoid
        lim_phase: Maximum phase for each sinusoid
        
    Returns:
        torques: [num_samples, trajectory_length, torque_dim]
    """
    np.random.seed(seed)
    
    all_torques = []
    for _ in range(num_samples):
        amplitudes = np.random.uniform(0, lim_amplitude, (torque_dim, num_sin, 1))
        frequencies = np.random.uniform(0, lim_frequency, (torque_dim, num_sin, 1))
        phases = np.random.uniform(0, lim_phase, (torque_dim, num_sin, 1))
        
        steps = np.arange(trajectory_length) * dt
        steps = steps[None, None, :]
        steps = np.tile(steps, (torque_dim, num_sin, 1))
        
        torque = np.sum(amplitudes * np.sin(frequencies * steps + phases), axis=1).T
        all_torques.append(torque)
    
    return np.array(all_torques, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate test torque sequences")
    parser.add_argument("--output", type=str, default="data/test_torques_2000.h5",
                        help="Output HDF5 file path")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of torque sequences to generate")
    parser.add_argument("--trajectory_length", type=int, default=1000,
                        help="Length of each trajectory")
    parser.add_argument("--torque_dim", type=int, default=3,
                        help="Dimension of torque (default: 3 for 3-joint arm)")
    parser.add_argument("--dt", type=float, default=0.00025,
                        help="Timestep between samples (data_dt, default: 0.00025)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Sinusoid generation parameters (matching trajectory_dpf.py defaults)
    parser.add_argument("--num_sin", type=int, default=5,
                        help="Number of sinusoids to sum")
    parser.add_argument("--lim_amplitude", type=float, default=0.5,
                        help="Maximum amplitude for each sinusoid")
    parser.add_argument("--lim_frequency", type=float, default=6*math.pi,
                        help="Maximum frequency for each sinusoid")
    parser.add_argument("--lim_phase", type=float, default=2*math.pi,
                        help="Maximum phase for each sinusoid")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {args.num_samples} torque sequences...")
    print(f"  Trajectory length: {args.trajectory_length}")
    print(f"  Torque dimension: {args.torque_dim}")
    print(f"  Timestep (dt): {args.dt}")
    print(f"  Seed: {args.seed}")
    
    # Generate torques
    torques = generate_random_torque(
        num_samples=args.num_samples,
        trajectory_length=args.trajectory_length,
        torque_dim=args.torque_dim,
        dt=args.dt,
        seed=args.seed,
        num_sin=args.num_sin,
        lim_amplitude=args.lim_amplitude,
        lim_frequency=args.lim_frequency,
        lim_phase=args.lim_phase
    )
    
    print(f"  Generated shape: {torques.shape}")
    
    # Save to HDF5
    print(f"Saving to {args.output}...")
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('torques', data=torques, compression='gzip')
        
        # Save metadata
        f.attrs['seed'] = args.seed
        f.attrs['num_samples'] = args.num_samples
        f.attrs['trajectory_length'] = args.trajectory_length
        f.attrs['torque_dim'] = args.torque_dim
        f.attrs['dt'] = args.dt
        f.attrs['num_sin'] = args.num_sin
        f.attrs['lim_amplitude'] = args.lim_amplitude
        f.attrs['lim_frequency'] = args.lim_frequency
        f.attrs['lim_phase'] = args.lim_phase
    
    print(f"Done! Saved {args.num_samples} torque sequences to {args.output}")
    print(f"  File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
