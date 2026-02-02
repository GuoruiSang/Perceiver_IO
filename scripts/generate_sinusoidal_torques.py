"""
Generate sinusoidal torque sequences with the EXACT same parameters as training data.

Training data parameters (from generate_dataset_forward.py → generate() defaults):
  num_sin=5, lim_amplitude=0.5, lim_frequency=6π, lim_phase=2π, dt=0.0002

Usage:
    python scripts/generate_sinusoidal_torques.py \
        --output data/sinusoidal_torques_1000_L1500.h5 \
        --num_samples 1000 \
        --trajectory_length 1500 \
        --seed 99
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
    lim_phase: float = 2 * math.pi,
) -> np.ndarray:
    """
    Generate random smooth torque sequences using sum of sinusoids.
    Matches _generate_random_torque in trajectory_dpf.py and generate() in generate_dataset_forward.py.
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
    parser = argparse.ArgumentParser(description="Generate sinusoidal torques (same params as training)")
    parser.add_argument("--output", type=str, default="data/sinusoidal_torques_1000_L1500.h5")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--trajectory_length", type=int, default=1500)
    parser.add_argument("--torque_dim", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.0002,
                        help="Timestep (data_dt from training: skip_steps * dt = 2 * 0.0001)")
    parser.add_argument("--seed", type=int, default=99)
    # Exact same parameters as training data generation
    parser.add_argument("--num_sin", type=int, default=5)
    parser.add_argument("--lim_amplitude", type=float, default=0.5)
    parser.add_argument("--lim_frequency", type=float, default=6 * math.pi)
    parser.add_argument("--lim_phase", type=float, default=2 * math.pi)
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {args.num_samples} sinusoidal torque sequences...")
    print(f"  Parameters: num_sin={args.num_sin}, lim_amplitude={args.lim_amplitude}, "
          f"lim_frequency={args.lim_frequency:.4f}, lim_phase={args.lim_phase:.4f}")
    print(f"  dt={args.dt}, trajectory_length={args.trajectory_length}, seed={args.seed}")

    torques = generate_random_torque(
        num_samples=args.num_samples,
        trajectory_length=args.trajectory_length,
        torque_dim=args.torque_dim,
        dt=args.dt,
        seed=args.seed,
        num_sin=args.num_sin,
        lim_amplitude=args.lim_amplitude,
        lim_frequency=args.lim_frequency,
        lim_phase=args.lim_phase,
    )

    print(f"  Generated shape: {torques.shape}")
    print(f"  Value range: [{torques.min():.4f}, {torques.max():.4f}]")
    print(f"  Std per dim: {torques.std(axis=(0,1))}")

    print(f"Saving to {args.output}...")
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('torques', data=torques, compression='gzip')
        f.attrs['seed'] = args.seed
        f.attrs['num_samples'] = args.num_samples
        f.attrs['trajectory_length'] = args.trajectory_length
        f.attrs['torque_dim'] = args.torque_dim
        f.attrs['dt'] = args.dt
        f.attrs['num_sin'] = args.num_sin
        f.attrs['lim_amplitude'] = args.lim_amplitude
        f.attrs['lim_frequency'] = args.lim_frequency
        f.attrs['lim_phase'] = args.lim_phase
        f.attrs['generation_method'] = 'sum_of_sinusoids'

    file_size = os.path.getsize(args.output) / 1024 / 1024
    print(f"Done! File size: {file_size:.2f} MB")


if __name__ == "__main__":
    main()
