"""
Extract torque sequences from the training dataset for ablation evaluation.

Randomly samples trajectories from the training H5 file and extracts their
torque sequences, saving them in the same format as generate_test_torques.py.

Usage:
    python scripts/extract_training_torques.py \
        --dataset data/traj_40000-steps_4000.h5 \
        --output data/training_torques_1000_L1500.h5 \
        --num_samples 1000 \
        --max_length 1500 \
        --seed 42
"""

import argparse
import h5py
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser(description="Extract training torques")
    parser.add_argument("--dataset", type=str,
                        default="/home/gsang/Projects/Perceiver_IO/data/traj_40000-steps_4000.h5",
                        help="Path to training dataset")
    parser.add_argument("--output", type=str,
                        default="data/training_torques_1000_L1500.h5",
                        help="Output HDF5 file path")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of torque sequences to extract")
    parser.add_argument("--max_length", type=int, default=1500,
                        help="Maximum trajectory length to extract")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for trajectory selection")
    args = parser.parse_args()

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Loading training dataset from: {args.dataset}")
    with h5py.File(args.dataset, 'r') as f:
        num_trajectories = f.attrs['num_trajectories']
        num_steps = f.attrs['num_steps']
        print(f"  Dataset: {num_trajectories} trajectories, {num_steps} steps each")

        # Randomly select trajectory indices
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(num_trajectories, size=args.num_samples, replace=False)
        indices.sort()

        # Extract torques
        print(f"Extracting {args.num_samples} torque sequences (length {args.max_length})...")
        torques = []
        for idx in indices:
            torque = f[f'traj_{idx}']['seq_torque'][:args.max_length]
            torques.append(torque)

        torques = np.array(torques, dtype=np.float32)
        print(f"  Extracted shape: {torques.shape}")

    # Save
    print(f"Saving to {args.output}...")
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('torques', data=torques, compression='gzip')
        f.attrs['seed'] = args.seed
        f.attrs['num_samples'] = args.num_samples
        f.attrs['trajectory_length'] = args.max_length
        f.attrs['torque_dim'] = torques.shape[-1]
        f.attrs['source'] = args.dataset
        f.attrs['generation_method'] = 'extracted_from_training'

    file_size = os.path.getsize(args.output) / 1024 / 1024
    print(f"Done! Saved {args.num_samples} torque sequences to {args.output}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Value range: [{torques.min():.4f}, {torques.max():.4f}]")
    print(f"  Std per dim: {torques.std(axis=(0,1))}")


if __name__ == "__main__":
    main()
