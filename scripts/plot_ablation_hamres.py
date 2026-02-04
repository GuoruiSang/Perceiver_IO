"""
Plot HamRes percentiles (P25, Median, P95, P99) for ablation experiments.

Usage:
    python scripts/plot_ablation_hamres.py
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ICLR-friendly style (matching plot_ablation_original.py)
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

# Identical to plot_ablation_original.py
TORQUE_STYLES = {
    'sinusoidal': {
        'label': 'Sin.',
        'guid_color': '#1f77b4',   # dark blue
        'ung_color': '#6baed6',    # medium blue
    },
    'gp': {
        'label': 'GP',
        'guid_color': '#d62728',   # dark red
        'ung_color': '#e6756b',    # medium red
    },
    'zero': {
        'label': 'Zero',
        'guid_color': '#2ca02c',   # dark green
        'ung_color': '#74c476',    # medium green
    },
    'spline': {
        'label': 'Cubic Spline',
        'guid_color': '#9467bd',   # dark purple
        'ung_color': '#b09fd0',    # medium purple
    },
}

GUID_MARKER = 's'
UNG_MARKER = 'o'
GT_COLOR = '#888888'

RESULTS_DIR = project_root / 'output_ablation' / 'results' / 'original'
PLOTS_DIR = project_root / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

POLICIES = ['sinusoidal', 'gp', 'zero', 'spline']
# 2x2 layout: [[P25, Median], [P95, P99]]
STATS = [
    [('p25', 'P25'), ('median', 'Median')],
    [('p95', 'P95'), ('p99', 'P99')],
]


def main():
    # Load DPF data (skip missing)
    dpf_data = {}
    for pol in POLICIES:
        path = RESULTS_DIR / f'hamres_pct_{pol}.csv'
        if path.exists():
            dpf_data[pol] = pd.read_csv(path)

    # Load GT data (skip missing)
    gt_data = {}
    for pol in POLICIES:
        path = RESULTS_DIR / f'gt_hamres_pct_{pol}.csv'
        if path.exists():
            gt_data[pol] = pd.read_csv(path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row_idx, row_stats in enumerate(STATS):
        for col_idx, (stat_key, stat_label) in enumerate(row_stats):
            ax = axes[row_idx, col_idx]

            # Check if this stat exists in data
            sample_df = list(dpf_data.values())[0] if dpf_data else None
            has_stat = sample_df is not None and f'unguided_{stat_key}' in sample_df.columns

            if not has_stat:
                ax.text(0.5, 0.5, f'{stat_label}\n(data not available)',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=16, color='gray')
                ax.set_title(stat_label)
                continue

            # Plot ALL Unguided first (to match NMSE legend order)
            for pol in POLICIES:
                if pol not in dpf_data:
                    continue
                style = TORQUE_STYLES[pol]
                df = dpf_data[pol]
                L = df['trajectory_length'].values
                ax.plot(L, df[f'unguided_{stat_key}'].values,
                        marker=UNG_MARKER, linestyle='--',
                        color=style['ung_color'], alpha=0.8,
                        label=f'{style["label"]} (Unguided)', markersize=4, linewidth=1.5)

            # Plot ALL Guided second
            for pol in POLICIES:
                if pol not in dpf_data:
                    continue
                style = TORQUE_STYLES[pol]
                df = dpf_data[pol]
                L = df['trajectory_length'].values
                ax.plot(L, df[f'guided_{stat_key}'].values,
                        marker=GUID_MARKER, linestyle='-',
                        color=style['guid_color'],
                        label=f'{style["label"]} (Guided)', markersize=5, linewidth=2)

            # Plot GT last
            if gt_data:
                gt_col = f'gt_{stat_key}'
                if gt_col in list(gt_data.values())[0].columns:
                    gt_vals = np.stack([gt_data[pol][gt_col].values for pol in gt_data])
                    gt_avg = gt_vals.mean(axis=0)
                    lengths = list(gt_data.values())[0]['trajectory_length'].values
                    ax.plot(lengths, gt_avg,
                            marker='D', linestyle='-', color=GT_COLOR, linewidth=2, markersize=5,
                            label='GT', zorder=10)

            ax.set_xlabel('Trajectory Length')
            ax.set_title(stat_label)
            if col_idx == 0:
                ax.set_ylabel('HamRes')

    # Single shared legend at top with ncol=5 (GT in 5th column, 1st row)
    # Get handles from a subplot that has data (Median at [0,1])
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        # Reorder for column-major display with ncol=5
        # Original: [Sin(U), GP(U), Zero(U), Spline(U), Sin(G), GP(G), Zero(G), Spline(G), GT]
        # Want Row 1: Sin(U), GP(U), Zero(U), Spline(U), GT
        # Want Row 2: Sin(G), GP(G), Zero(G), Spline(G)
        reorder = [0, 4, 1, 5, 2, 6, 3, 7, 8]
        handles = [handles[i] for i in reorder]
        labels = [labels[i] for i in reorder]
        fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=14,
                   framealpha=0.9, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = PLOTS_DIR / 'ablation_hamres.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
