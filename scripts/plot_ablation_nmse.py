"""
Plot NMSE_q and NMSE_p for ablation experiments.

Usage:
    python scripts/plot_ablation_nmse.py
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import pandas as pd

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

TORQUE_STYLES = {
    'sinusoidal': {'label': 'Sin.', 'guid_color': '#1f77b4', 'ung_color': '#6baed6'},
    'gp':         {'label': 'GP',   'guid_color': '#d62728', 'ung_color': '#e6756b'},
    'zero':       {'label': 'Zero', 'guid_color': '#2ca02c', 'ung_color': '#74c476'},
    'spline':     {'label': 'Cubic Spline', 'guid_color': '#9467bd', 'ung_color': '#b09fd0'},
}

RESULTS_DIR = project_root / 'output_ablation' / 'results' / 'original'
PLOTS_DIR = project_root / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MSE_FILES = {
    'sinusoidal': RESULTS_DIR / 'exp_a_sinusoidal.csv',
    'gp': RESULTS_DIR / 'exp_a_gp.csv',
    'zero': RESULTS_DIR / 'exp_a_zero.csv',
    'spline': project_root / 'output_ablation' / 'results' / 'latest' / 'mse_spline_torques.csv',
}

METRICS = [
    ('nmse_q', r'NMSE$_q$'),
    ('nmse_p', r'NMSE$_p$'),
]


def main():
    torque_data = {}
    for key, path in MSE_FILES.items():
        if path.exists():
            df = pd.read_csv(path)
            torque_data[key] = df[df['trajectory_length'] <= 1000]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

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

            ax_mean.plot(lengths, ung_mean,
                         marker='o', linestyle='--',
                         color=style['ung_color'], alpha=0.8,
                         label=f'{label} (Unguided)', markersize=4, linewidth=1.5)
            ax_mean.plot(lengths, guid_mean,
                         marker='s', linestyle='-',
                         color=style['guid_color'],
                         label=f'{label} (Guided)', markersize=5, linewidth=2)

            ax_std.plot(lengths, ung_std,
                        marker='o', linestyle='--',
                        color=style['ung_color'], alpha=0.5,
                        label=f'{label} (Unguided)', markersize=4, linewidth=1.5)
            ax_std.plot(lengths, guid_std,
                        marker='s', linestyle='-',
                        color=style['guid_color'],
                        label=f'{label} (Guided)', markersize=5, linewidth=2)

        ax_mean.set_ylabel(metric_label)
        ax_std.set_xlabel('Trajectory Length')
        ax_std.set_ylabel(r'$\sigma$(' + metric_label + ')')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=14,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    out_path = PLOTS_DIR / 'ablation_nmse.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
