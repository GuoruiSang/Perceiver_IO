#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def _plot_sequence_panel(ax, seq_array: np.ndarray, title: str, ylabel: str, color0: str = '#d14b41', color1: str = '#2b6cb0') -> None:
    if len(seq_array) == 0:
        ax.set_title(title)
        ax.set_xlabel('timestep')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        return
    timesteps = np.arange(seq_array.shape[1], dtype=np.int32)
    for seq in seq_array:
        ax.plot(timesteps, seq[:, 0], color=color0, alpha=0.04, lw=0.8)
        ax.plot(timesteps, seq[:, 1], color=color1, alpha=0.04, lw=0.8)
    mean0 = seq_array[:, :, 0].mean(axis=0)
    mean1 = seq_array[:, :, 1].mean(axis=0)
    p05_0 = np.percentile(seq_array[:, :, 0], 5, axis=0)
    p95_0 = np.percentile(seq_array[:, :, 0], 95, axis=0)
    p05_1 = np.percentile(seq_array[:, :, 1], 5, axis=0)
    p95_1 = np.percentile(seq_array[:, :, 1], 95, axis=0)
    ax.fill_between(timesteps, p05_0, p95_0, color=color0, alpha=0.12)
    ax.fill_between(timesteps, p05_1, p95_1, color=color1, alpha=0.12)
    ax.plot(timesteps, mean0, color=color0, lw=2.0, label=f'{ylabel}0 mean')
    ax.plot(timesteps, mean1, color=color1, lw=2.0, label=f'{ylabel}1 mean')
    ax.set_title(title)
    ax.set_xlabel('timestep')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8, frameon=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Render coverage plots directly from a saved Reacher H5 dataset')
    parser.add_argument('--h5_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--workspace_limit', type=float, default=0.22)
    parser.add_argument('--max_paths_plot', type=int, default=4000)
    parser.add_argument('--max_density_paths', type=int, default=12000)
    parser.add_argument('--max_source_target_plot', type=int, default=5000)
    parser.add_argument('--max_sequence_plot', type=int, default=600)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = Path(args.h5_path)
    rng = np.random.default_rng(args.seed)

    with h5py.File(h5_path, 'r') as f:
        num_trajectories = int(f.attrs['num_trajectories'])
        all_indices = np.arange(num_trajectories, dtype=np.int32)
        path_indices = set(rng.choice(all_indices, size=min(args.max_paths_plot, num_trajectories), replace=False).tolist())
        density_indices = set(rng.choice(all_indices, size=min(args.max_density_paths, num_trajectories), replace=False).tolist())
        source_target_indices = set(rng.choice(all_indices, size=min(args.max_source_target_plot, num_trajectories), replace=False).tolist())
        sequence_indices = set(rng.choice(all_indices, size=min(args.max_sequence_plot, num_trajectories), replace=False).tolist())

        sampled_paths: list[np.ndarray] = []
        density_paths: list[np.ndarray] = []
        source_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        qpos_sequences: list[np.ndarray] = []
        mom_sequences: list[np.ndarray] = []
        torque_sequences: list[np.ndarray] = []
        source_xy_all: list[np.ndarray] = []
        target_xy_all: list[np.ndarray] = []
        final_xy_all: list[np.ndarray] = []

        for traj_idx in tqdm(range(num_trajectories), desc='Reading H5 trajectories'):
            group = f[f'traj_{traj_idx}']
            source_xy = group['source_xy'][:].astype(np.float32)
            target_xy = group['target_xy'][:].astype(np.float32)
            final_xy = group['seq_fingertip_xy'][-1].astype(np.float32)
            source_xy_all.append(source_xy)
            target_xy_all.append(target_xy)
            final_xy_all.append(final_xy)

            if traj_idx in path_indices:
                sampled_paths.append(group['seq_fingertip_xy'][:].astype(np.float32))
            if traj_idx in density_indices:
                density_paths.append(group['seq_fingertip_xy'][:].astype(np.float32))
            if traj_idx in source_target_indices:
                source_pairs.append((source_xy, target_xy))
            if traj_idx in sequence_indices:
                qpos_sequences.append(group['seq_qpos'][:].astype(np.float32))
                mom_sequences.append(group['seq_mom'][:].astype(np.float32))
                torque_sequences.append(group['seq_torque'][:].astype(np.float32))

    source_xy_arr = np.stack(source_xy_all, axis=0) if source_xy_all else np.empty((0, 2), dtype=np.float32)
    target_xy_arr = np.stack(target_xy_all, axis=0) if target_xy_all else np.empty((0, 2), dtype=np.float32)
    final_xy_arr = np.stack(final_xy_all, axis=0) if final_xy_all else np.empty((0, 2), dtype=np.float32)
    qpos_seq_arr = np.stack(qpos_sequences, axis=0) if qpos_sequences else np.empty((0, 0, 2), dtype=np.float32)
    mom_seq_arr = np.stack(mom_sequences, axis=0) if mom_sequences else np.empty((0, 0, 2), dtype=np.float32)
    torque_seq_arr = np.stack(torque_sequences, axis=0) if torque_sequences else np.empty((0, 0, 2), dtype=np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)

    ax = axes[0, 0]
    for path in sampled_paths:
        ax.plot(path[:, 0], path[:, 1], color='#d14b41', alpha=0.03, lw=0.8)
        ax.scatter(path[0, 0], path[0, 1], s=4, color='black', alpha=0.06)
    ax.set_title(f'Workspace Path Coverage\n{len(sampled_paths)} sampled trajectories from dataset')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim(-args.workspace_limit, args.workspace_limit)
    ax.set_ylim(-args.workspace_limit, args.workspace_limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    if density_paths:
        density_points = np.concatenate(density_paths, axis=0)
        hb = ax.hexbin(
            density_points[:, 0], density_points[:, 1], gridsize=90, mincnt=1, cmap='YlOrRd', bins='log'
        )
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02, label='log10 path density')
    ax.set_title(f'Workspace Path Density\n{len(density_paths)} sampled trajectories')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim(-args.workspace_limit, args.workspace_limit)
    ax.set_ylim(-args.workspace_limit, args.workspace_limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)

    ax = axes[0, 2]
    for source_xy, target_xy in source_pairs:
        ax.plot([source_xy[0], target_xy[0]], [source_xy[1], target_xy[1]], color='#d14b41', alpha=0.02, lw=0.7)
    if len(source_xy_arr):
        ax.scatter(source_xy_arr[:, 0], source_xy_arr[:, 1], s=5, color='black', alpha=0.08, label='source')
    if len(target_xy_arr):
        ax.scatter(target_xy_arr[:, 0], target_xy_arr[:, 1], s=5, color='#2b6cb0', alpha=0.08, label='target')
    ax.set_title(f'Source-Target Coverage\n{len(source_pairs)} sampled source-target pairs')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim(-args.workspace_limit, args.workspace_limit)
    ax.set_ylim(-args.workspace_limit, args.workspace_limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8, frameon=True)

    _plot_sequence_panel(axes[1, 0], qpos_seq_arr, 'Joint Position Sequences', 'q')
    _plot_sequence_panel(axes[1, 1], mom_seq_arr, 'Momentum Sequences', 'p')
    _plot_sequence_panel(axes[1, 2], torque_seq_arr, 'Torque Sequences', 'tau')

    figure_path = output_dir / 'dataset_coverage.png'
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    summary = {
        'h5_path': str(h5_path),
        'figure_path': str(figure_path),
        'num_trajectories': int(num_trajectories),
        'sampled_paths_plotted': int(len(sampled_paths)),
        'density_paths_plotted': int(len(density_paths)),
        'source_target_pairs_plotted': int(len(source_pairs)),
        'sequence_paths_plotted': int(len(qpos_sequences)),
        'workspace_limit': float(args.workspace_limit),
        'source_xy_bbox': {
            'x_min': float(source_xy_arr[:, 0].min()),
            'x_max': float(source_xy_arr[:, 0].max()),
            'y_min': float(source_xy_arr[:, 1].min()),
            'y_max': float(source_xy_arr[:, 1].max()),
        },
        'target_xy_bbox': {
            'x_min': float(target_xy_arr[:, 0].min()),
            'x_max': float(target_xy_arr[:, 0].max()),
            'y_min': float(target_xy_arr[:, 1].min()),
            'y_max': float(target_xy_arr[:, 1].max()),
        },
        'final_xy_bbox': {
            'x_min': float(final_xy_arr[:, 0].min()),
            'x_max': float(final_xy_arr[:, 0].max()),
            'y_min': float(final_xy_arr[:, 1].min()),
            'y_max': float(final_xy_arr[:, 1].max()),
        },
    }
    summary_path = output_dir / 'dataset_coverage_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(summary_path, flush=True)


if __name__ == '__main__':
    main()
