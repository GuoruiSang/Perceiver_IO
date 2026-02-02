"""
Generate Gaussian-process-style torque sequences with amplitude matched to sinusoidal torques.

Uses Gaussian-filtered white noise to approximate an RBF kernel GP.
Amplitude is matched to the sinusoidal torque distribution by:
1. Generating a reference batch of sinusoidal torques with training-identical parameters
2. Computing per-dimension std of the reference
3. Rescaling GP samples to match that std

Usage:
    python scripts/generate_gp_torques.py \
        --output data/gp_torques_1000_L1500.h5 \
        --num_samples 1000 \
        --trajectory_length 1500 \
        --length_scale 100 \
        --seed 123
"""

import argparse
import numpy as np
import h5py
import math
import os
from scipy.ndimage import gaussian_filter1d


def generate_sinusoidal_reference(
    num_samples: int,
    trajectory_length: int,
    torque_dim: int,
    dt: float,
    num_sin: int = 5,
    lim_amplitude: float = 0.5,
    lim_frequency: float = 6 * math.pi,
    lim_phase: float = 2 * math.pi,
) -> np.ndarray:
    """Generate sinusoidal torques for computing reference statistics."""
    rng = np.random.RandomState(0)  # Fixed seed for reference
    all_torques = []
    for _ in range(num_samples):
        amplitudes = rng.uniform(0, lim_amplitude, (torque_dim, num_sin, 1))
        frequencies = rng.uniform(0, lim_frequency, (torque_dim, num_sin, 1))
        phases = rng.uniform(0, lim_phase, (torque_dim, num_sin, 1))
        steps = np.arange(trajectory_length) * dt
        steps = steps[None, None, :]
        steps = np.tile(steps, (torque_dim, num_sin, 1))
        torque = np.sum(amplitudes * np.sin(frequencies * steps + phases), axis=1).T
        all_torques.append(torque)
    return np.array(all_torques, dtype=np.float32)


def generate_gp_torques(
    num_samples: int,
    trajectory_length: int,
    torque_dim: int,
    length_scale: float,
    target_std_per_dim: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Generate GP-style torques using Gaussian-filtered white noise.
    Rescaled so per-dimension std matches the sinusoidal reference.
    """
    rng = np.random.RandomState(seed)

    # Generate white noise
    noise = rng.randn(num_samples, trajectory_length, torque_dim).astype(np.float32)

    # Smooth along time axis (axis=1)
    smoothed = gaussian_filter1d(noise, sigma=length_scale, axis=1)

    # Rescale per-dimension to match target std
    for d in range(torque_dim):
        current_std = smoothed[:, :, d].std()
        if current_std > 0:
            smoothed[:, :, d] *= target_std_per_dim[d] / current_std

    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Generate GP-style torques (amplitude-matched)")
    parser.add_argument("--output", type=str, default="data/gp_torques_1000_L1500.h5")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--trajectory_length", type=int, default=1500)
    parser.add_argument("--torque_dim", type=int, default=3)
    parser.add_argument("--length_scale", type=float, default=100.0,
                        help="Gaussian filter sigma in timesteps (controls smoothness)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--reference_samples", type=int, default=10000,
                        help="Number of sinusoidal samples for computing reference statistics")
    parser.add_argument("--dt", type=float, default=0.0002,
                        help="Timestep for sinusoidal reference generation (skip_steps * dt = 2 * 0.0001)")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate sinusoidal reference and compute statistics
    print(f"Generating {args.reference_samples} sinusoidal reference samples for amplitude matching...")
    ref_torques = generate_sinusoidal_reference(
        num_samples=args.reference_samples,
        trajectory_length=args.trajectory_length,
        torque_dim=args.torque_dim,
        dt=args.dt,
    )
    ref_std_per_dim = ref_torques.std(axis=(0, 1))  # [torque_dim]
    ref_range_per_dim = np.abs(ref_torques).max(axis=(0, 1))
    print(f"  Reference std per dim: {ref_std_per_dim}")
    print(f"  Reference max|value| per dim: {ref_range_per_dim}")

    # Step 2: Generate GP torques
    print(f"\nGenerating {args.num_samples} GP torque sequences...")
    print(f"  length_scale={args.length_scale}, seed={args.seed}")
    torques = generate_gp_torques(
        num_samples=args.num_samples,
        trajectory_length=args.trajectory_length,
        torque_dim=args.torque_dim,
        length_scale=args.length_scale,
        target_std_per_dim=ref_std_per_dim,
        seed=args.seed,
    )
    print(f"  Generated shape: {torques.shape}")
    print(f"  Value range: [{torques.min():.4f}, {torques.max():.4f}]")
    print(f"  Std per dim: {torques.std(axis=(0,1))}")
    print(f"  Max|value| per dim: {np.abs(torques).max(axis=(0,1))}")

    # Step 3: Save
    print(f"\nSaving to {args.output}...")
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('torques', data=torques, compression='gzip')
        f.attrs['seed'] = args.seed
        f.attrs['num_samples'] = args.num_samples
        f.attrs['trajectory_length'] = args.trajectory_length
        f.attrs['torque_dim'] = args.torque_dim
        f.attrs['length_scale'] = args.length_scale
        f.attrs['generation_method'] = 'gaussian_filtered_noise'
        f.attrs['reference_std_per_dim'] = ref_std_per_dim
        f.attrs['reference_samples'] = args.reference_samples

    file_size = os.path.getsize(args.output) / 1024 / 1024
    print(f"Done! File size: {file_size:.2f} MB")


if __name__ == "__main__":
    main()
