"""Plot non-zero-policy NRMSE(range) views: unguided and guidance improvement."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


project_root = Path('/home/gsang/Projects/Perceiver_IO')
metrics_csv = project_root / 'output_ablation' / 'default_full_sweep' / 'metrics' / 'nrmse_range_summary_Lle1000.csv'
plots_root = project_root / 'plots'
plots_root.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 1000
POLICIES = ['sinusoidal', 'gp', 'spline']  # exclude zero
FILL_ALPHA = 0.12

TORQUE_STYLES = {
    'sinusoidal': {'label': 'Sin.', 'color': '#1f77b4'},
    'gp': {'label': 'GP', 'color': '#d62728'},
    'spline': {'label': 'Cubic Spline', 'color': '#8b00ff'},
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 12,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def _plot_subplot(ax, df_all, system, suffix, mode, title):
    for pol in POLICIES:
        df = df_all[(df_all['system'] == system) & (df_all['policy'] == pol)].copy()
        if df.empty:
            continue
        df = df[df['length'] <= MAX_LENGTH].sort_values('length')
        style = TORQUE_STYLES[pol]
        L = df['length'].to_numpy()

        if mode == 'ung':
            mean = df[f'ung_nrmse_range_{suffix}_mean'].to_numpy()
            std = df[f'ung_nrmse_range_{suffix}_std'].to_numpy()
        elif mode == 'improve':
            # Improvement = unguided - guided = -(guided - unguided)
            mean = -df[f'd_nrmse_range_{suffix}_mean'].to_numpy()
            std = df[f'd_nrmse_range_{suffix}_std'].to_numpy()
        else:
            raise ValueError(f'Unknown mode: {mode}')

        ax.plot(L, mean, marker='o', linestyle='-', color=style['color'],
                label=style['label'], markersize=3, linewidth=2)
        ax.fill_between(L, mean - std, mean + std, color=style['color'],
                        alpha=FILL_ALPHA, edgecolor=style['color'], linewidth=1.2)

    ax.set_xlabel('Trajectory Length')
    if mode == 'ung':
        ax.set_ylabel('NRMSE(range)')
    else:
        ax.set_ylabel('Improvement (unguided - guided)')
        ax.axhline(0.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.set_title(title)


def _make_figure(df, mode, out_name):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _plot_subplot(axes[0, 0], df, '2dof', 'q', mode, r'2DoF - $q$')
    _plot_subplot(axes[0, 1], df, '2dof', 'p', mode, r'2DoF - $p$')
    _plot_subplot(axes[1, 0], df, '3dof', 'q', mode, r'3DoF - $q$')
    _plot_subplot(axes[1, 1], df, '3dof', 'p', mode, r'3DoF - $p$')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=14,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out = plots_root / out_name
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'Saved: {out}')


def main():
    df = pd.read_csv(metrics_csv)
    print(f'Loaded: {metrics_csv} ({len(df)} rows)')

    _make_figure(df, mode='ung', out_name='ablation_nrmse_range_unguided_nonzero_meanstd.png')
    _make_figure(df, mode='improve', out_name='ablation_nrmse_range_improvement_nonzero_meanstd.png')


if __name__ == '__main__':
    main()
