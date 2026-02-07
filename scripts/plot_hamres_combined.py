"""
Combine 2DoF and 3DoF HamRes plots.
1 row x 2 cols: mean lines + fill_between std bands.
Both systems use σ=5 smoothed data from metrics CSVs.
Linear scale.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

project_root = Path('/home/gsang/Projects/Perceiver_IO')

# Style
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

TORQUE_STYLES = {
    'sinusoidal': {'label': 'Sin.', 'guid_color': '#1f77b4', 'ung_color': '#6baed6'},
    'gp': {'label': 'GP', 'guid_color': '#d62728', 'ung_color': '#e6756b'},
    'zero': {'label': 'Zero', 'guid_color': '#2ca02c', 'ung_color': '#74c476'},
    'spline': {'label': 'Cubic Spline', 'guid_color': '#8b00ff', 'ung_color': '#c77dff'},
}

policies = ['sinusoidal', 'gp', 'zero', 'spline']
FILL_ALPHA = 0.12

# Load data for both systems from unified metrics CSVs (σ=5 smoothed)
data_2dof = {}
results_dir_2dof = project_root / 'output_ablation' / 'results' / '2dof_smoothed'
for pol in policies:
    path = results_dir_2dof / f'metrics_{pol}_sigma5.0.csv'
    if path.exists():
        df = pd.read_csv(path)
        df.rename(columns={
            'unguided_hamres_mean': 'unguided_mean', 'unguided_hamres_std': 'unguided_std',
            'guided_hamres_mean': 'guided_mean', 'guided_hamres_std': 'guided_std',
        }, inplace=True)
        data_2dof[pol] = df
        print(f"Loaded 2DoF {pol}: {len(df)} rows")

data_3dof_hamres = {}
results_dir_3dof = project_root / 'output_ablation' / 'results' / '3dof_smoothed'
for pol in policies:
    path = results_dir_3dof / f'metrics_{pol}_sigma5.0.csv'
    if path.exists():
        df = pd.read_csv(path)
        df.rename(columns={
            'unguided_hamres_mean': 'unguided_mean', 'unguided_hamres_std': 'unguided_std',
            'guided_hamres_mean': 'guided_mean', 'guided_hamres_std': 'guided_std',
        }, inplace=True)
        data_3dof_hamres[pol] = df
        print(f"Loaded 3DoF HamRes {pol}: {len(df)} rows")


def make_plot(out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    MAX_LENGTH = 1100

    # ── Left: 2DoF (mean + std bands) ──
    ax = axes[0]
    for pol in policies:
        if pol not in data_2dof:
            continue
        df = data_2dof[pol]
        df = df[df['trajectory_length'] <= MAX_LENGTH]
        style = TORQUE_STYLES[pol]
        L = df['trajectory_length'].values

        mean_u = df['unguided_mean'].values
        std_u = df['unguided_std'].values
        ax.plot(L, mean_u, marker='o', linestyle='--', color=style['ung_color'],
                alpha=0.8, label=f'{style["label"]} (Unguided)', markersize=2, linewidth=1.5)
        fill_u = ax.fill_between(L, np.maximum(mean_u - std_u, 0), mean_u + std_u,
                        color=style['ung_color'], alpha=FILL_ALPHA,
                        edgecolor=style['ung_color'], linewidth=1.8)
        fill_u.set_linestyle('--')
        fill_u.set_edgecolor(style['guid_color'])

        mean_g = df['guided_mean'].values
        std_g = df['guided_std'].values
        ax.plot(L, mean_g, marker='s', linestyle='-', color=style['guid_color'],
                label=f'{style["label"]} (Guided)', markersize=3, linewidth=2)
        ax.fill_between(L, np.maximum(mean_g - std_g, 0), mean_g + std_g,
                        color=style['guid_color'], alpha=FILL_ALPHA,
                        edgecolor=style['guid_color'], linewidth=1.5)
    ax.set_xlabel('Trajectory Length')
    ax.set_ylabel('HamRes')
    ax.set_title('2DoF')

    # ── Right: 3DoF (mean + std bands) ──
    ax = axes[1]
    for pol in policies:
        if pol not in data_3dof_hamres:
            continue
        df = data_3dof_hamres[pol]
        df = df[df['trajectory_length'] <= MAX_LENGTH]
        style = TORQUE_STYLES[pol]
        L = df['trajectory_length'].values

        mean_u = df['unguided_mean'].values
        std_u = df['unguided_std'].values if 'unguided_std' in df.columns else np.zeros_like(mean_u)
        ax.plot(L, mean_u, marker='o', linestyle='--', color=style['ung_color'],
                alpha=0.8, label=f'{style["label"]} (Unguided)', markersize=2, linewidth=1.5)
        fill_u = ax.fill_between(L, np.maximum(mean_u - std_u, 0), mean_u + std_u,
                        color=style['ung_color'], alpha=FILL_ALPHA,
                        edgecolor=style['ung_color'], linewidth=1.8)
        fill_u.set_linestyle('--')
        fill_u.set_edgecolor(style['guid_color'])

        mean_g = df['guided_mean'].values
        std_g = df['guided_std'].values if 'guided_std' in df.columns else np.zeros_like(mean_g)
        ax.plot(L, mean_g, marker='s', linestyle='-', color=style['guid_color'],
                label=f'{style["label"]} (Guided)', markersize=3, linewidth=2)
        ax.fill_between(L, np.maximum(mean_g - std_g, 0), mean_g + std_g,
                        color=style['guid_color'], alpha=FILL_ALPHA,
                        edgecolor=style['guid_color'], linewidth=1.5)
    ax.set_xlabel('Trajectory Length')
    ax.set_ylabel('HamRes')
    ax.set_title('3DoF')

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    n = len([p for p in policies if p in data_2dof])
    if len(handles) >= 2 * n:
        new_handles, new_labels = [], []
        for i in range(n):
            new_handles.append(handles[2 * i])
            new_handles.append(handles[2 * i + 1])
            new_labels.append(labels[2 * i])
            new_labels.append(labels[2 * i + 1])
        handles, labels = new_handles, new_labels
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=12,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'Saved: {out_path}')


plots_dir = project_root / 'plots'
make_plot(out_path=plots_dir / 'ablation_hamres_combined.png')
