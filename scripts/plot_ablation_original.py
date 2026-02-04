"""
Plot MSE metrics (Position, Momentum, Energy) for the Original model
across ablation experiments: sinusoidal, gp, zero torques + context fractions.

Exp A: single merged 2x3 figure (top=mean, bottom=std) with all torque types.
Exp B: separate 2x3 figure for context fractions.

Usage:
    python scripts/plot_ablation_original.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ICLR-friendly style (matching plot.py)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

METRICS = [
    ('mse_qpos', 'MSE Position'),
    ('mse_mom', 'MSE Momentum'),
    ('mse_energy', 'MSE Energy'),
]

# Per-torque color scheme: dark = unguided, light = guided
TORQUE_STYLES = {
    'sinusoidal': {
        'label': 'Sin.',
        'guid_color': '#1f77b4',  # dark blue
        'ung_color': '#6baed6',   # medium blue
    },
    'gp': {
        'label': 'GP',
        'guid_color': '#d62728',  # dark red
        'ung_color': '#e6756b',   # medium red
    },
    'zero': {
        'label': 'Zero',
        'guid_color': '#2ca02c',  # dark green
        'ung_color': '#74c476',   # medium green
    },
    'spline': {
        'label': 'Cubic Spline',
        'guid_color': '#9467bd',  # dark purple
        'ung_color': '#b09fd0',   # medium purple
    },
}

GUID_MARKER = 's'   # solid/dark lines
UNG_MARKER = 'o'    # light lines

RESULTS_DIR = project_root / 'output_ablation' / 'results' / 'original'
PLOTS_DIR = project_root / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_exp_a_merged(torque_data, output_name):
    """Plot merged 2x3 figure for all torque types.

    Args:
        torque_data: dict of torque_key -> DataFrame (filtered to <= 1000).
        output_name: output filename stem.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for col, (metric_key, metric_label) in enumerate(METRICS):
        ax_mean = axes[0, col]
        ax_std = axes[1, col]

        for torque_key, df in torque_data.items():
            style = TORQUE_STYLES[torque_key]
            lengths = df['trajectory_length'].values
            label = style['label']

            ung_mean = df[f'unguided_{metric_key}_mean'].values
            ung_std = df[f'unguided_{metric_key}_std'].values
            guid_mean = df[f'guided_{metric_key}_mean'].values
            guid_std = df[f'guided_{metric_key}_std'].values

            # Top row: mean
            ax_mean.plot(lengths, ung_mean,
                         marker=UNG_MARKER, linestyle='--',
                         color=style['ung_color'], alpha=0.8,
                         label=f'{label} (Unguided)', markersize=4, linewidth=1.5)
            ax_mean.plot(lengths, guid_mean,
                         marker=GUID_MARKER, linestyle='-',
                         color=style['guid_color'],
                         label=f'{label} (Guided)', markersize=5, linewidth=2)

            # Bottom row: std
            ax_std.plot(lengths, ung_std,
                        marker=UNG_MARKER, linestyle='--',
                        color=style['ung_color'], alpha=0.5,
                        label=f'{label} (Unguided)', markersize=4, linewidth=1.5)
            ax_std.plot(lengths, guid_std,
                        marker=GUID_MARKER, linestyle='-',
                        color=style['guid_color'],
                        label=f'{label} (Guided)', markersize=5, linewidth=2)

        ax_mean.set_ylabel(metric_label)

        ax_std.set_xlabel('Trajectory Length')
        ax_std.set_ylabel(r'$\sigma$(' + metric_label + ')')

    # Single shared legend at top of figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=14,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    out_path = PLOTS_DIR / f'{output_name}.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_exp_b(csv_path, output_name):
    """Plot 2x3 figure for Experiment B (context fraction on x-axis).

    Top row: mean, bottom row: std.
    """
    df = pd.read_csv(csv_path)
    fractions = df['context_fraction'].values

    ung_color = '#aec7e8'   # light blue (unguided)
    guid_color = '#1f77b4'  # dark blue (guided)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for col, (metric_key, metric_label) in enumerate(METRICS):
        ung_mean = df[f'unguided_{metric_key}_mean'].values
        ung_std = df[f'unguided_{metric_key}_std'].values
        guid_mean = df[f'guided_{metric_key}_mean'].values
        guid_std = df[f'guided_{metric_key}_std'].values

        # Top row: mean
        ax_mean = axes[0, col]
        ax_mean.plot(fractions, ung_mean,
                     marker=UNG_MARKER, linestyle='-', color=ung_color,
                     label='Unguided', markersize=5, linewidth=2)
        ax_mean.plot(fractions, guid_mean,
                     marker=GUID_MARKER, linestyle='--', color=guid_color,
                     label='Guided', markersize=5, linewidth=2)
        ax_mean.set_ylabel(metric_label)

        # Bottom row: std
        ax_std = axes[1, col]
        ax_std.plot(fractions, ung_std,
                    marker=UNG_MARKER, linestyle='-', color=ung_color,
                    label='Unguided', markersize=5, linewidth=2)
        ax_std.plot(fractions, guid_std,
                    marker=GUID_MARKER, linestyle='--', color=guid_color,
                    label='Guided', markersize=5, linewidth=2)
        ax_std.set_xlabel('Context Fraction')
        ax_std.set_ylabel(f'{metric_label} (Std)')

    # Single shared legend at top of figure (same as torques plot)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=18,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = PLOTS_DIR / f'{output_name}.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    # Load all Exp A CSVs
    exp_a_files = {
        'sinusoidal': RESULTS_DIR / 'exp_a_sinusoidal.csv',
        'gp': RESULTS_DIR / 'exp_a_gp.csv',
        'zero': RESULTS_DIR / 'exp_a_zero.csv',
        'spline': project_root / 'output_ablation' / 'results' / 'latest' / 'mse_spline_torques.csv',
    }

    torque_data = {}
    for torque_key, csv_path in exp_a_files.items():
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = df[df['trajectory_length'] <= 1000]
            torque_data[torque_key] = df
        else:
            print(f'Warning: {csv_path} not found, skipping')

    if torque_data:
        plot_exp_a_merged(torque_data, 'ablation_original_torques')

    # Experiment B plot (context fractions)
    csv_path = RESULTS_DIR / 'exp_b_context_fractions.csv'
    if csv_path.exists():
        plot_exp_b(csv_path, 'ablation_original_context_fractions')
    else:
        print(f'Warning: {csv_path} not found, skipping')


if __name__ == '__main__':
    main()
