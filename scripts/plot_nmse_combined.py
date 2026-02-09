"""Combine 2DoF and 3DoF NMSE plots from default full sweep."""
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
MAX_LENGTH = 1000
FILL_ALPHA = 0.12

metrics_path = project_root / 'output_ablation' / 'default_full_sweep' / 'metrics' / 'metrics_summary.csv'
df_all = pd.read_csv(metrics_path)
print(f'Loaded metrics: {metrics_path} ({len(df_all)} rows)')


def plot_nmse_subplot_mean_std(ax, system, col_suffix, title):
    """Plot mean ± std bands for one subplot."""
    mean_ung_col = f'ung_nmse_{col_suffix}_mean'
    std_ung_col = f'ung_nmse_{col_suffix}_std'
    mean_gui_col = f'gui_nmse_{col_suffix}_mean'
    std_gui_col = f'gui_nmse_{col_suffix}_std'

    for pol in policies:
        df = df_all[(df_all['system'] == system) & (df_all['policy'] == pol)].copy()
        if df.empty:
            continue
        df = df[df['length'] <= MAX_LENGTH].sort_values('length')
        style = TORQUE_STYLES[pol]
        L = df['length'].values

        mean_u = df[mean_ung_col].values
        std_u = df[std_ung_col].values
        ax.plot(
            L, mean_u, marker='o', linestyle='--', color=style['ung_color'],
            alpha=0.8, label=f'{style["label"]} (Unguided)', markersize=2, linewidth=1.5
        )
        fill_u = ax.fill_between(
            L, np.maximum(mean_u - std_u, 0), mean_u + std_u,
            color=style['ung_color'], alpha=FILL_ALPHA,
            edgecolor=style['ung_color'], linewidth=1.8
        )
        fill_u.set_linestyle('--')
        fill_u.set_edgecolor(style['guid_color'])

        mean_g = df[mean_gui_col].values
        std_g = df[std_gui_col].values
        ax.plot(
            L, mean_g, marker='s', linestyle='-', color=style['guid_color'],
            label=f'{style["label"]} (Guided)', markersize=3, linewidth=2
        )
        ax.fill_between(
            L, np.maximum(mean_g - std_g, 0), mean_g + std_g,
            color=style['guid_color'], alpha=FILL_ALPHA,
            edgecolor=style['guid_color'], linewidth=1.5
        )

    ax.set_xlabel('Trajectory Length')
    ax.set_ylabel('NMSE')
    ax.set_title(title)


def plot_nmse_subplot_median_iqr(ax, system, col_suffix, title):
    """Plot median ± IQR bands for one subplot."""
    med_ung_col = f'ung_nmse_{col_suffix}_median'
    p25_ung_col = f'ung_nmse_{col_suffix}_p25'
    p75_ung_col = f'ung_nmse_{col_suffix}_p75'
    med_gui_col = f'gui_nmse_{col_suffix}_median'
    p25_gui_col = f'gui_nmse_{col_suffix}_p25'
    p75_gui_col = f'gui_nmse_{col_suffix}_p75'

    for pol in policies:
        df = df_all[(df_all['system'] == system) & (df_all['policy'] == pol)].copy()
        if df.empty:
            continue
        df = df[df['length'] <= MAX_LENGTH].sort_values('length')
        style = TORQUE_STYLES[pol]
        L = df['length'].values

        med_u = df[med_ung_col].values
        p25_u = df[p25_ung_col].values
        p75_u = df[p75_ung_col].values
        ax.plot(
            L, med_u, marker='o', linestyle='--', color=style['ung_color'],
            alpha=0.8, label=f'{style["label"]} (Unguided)', markersize=2, linewidth=1.5
        )
        fill_u = ax.fill_between(
            L, np.maximum(p25_u, 0), p75_u,
            color=style['ung_color'], alpha=FILL_ALPHA,
            edgecolor=style['ung_color'], linewidth=1.8
        )
        fill_u.set_linestyle('--')
        fill_u.set_edgecolor(style['guid_color'])

        med_g = df[med_gui_col].values
        p25_g = df[p25_gui_col].values
        p75_g = df[p75_gui_col].values
        ax.plot(
            L, med_g, marker='s', linestyle='-', color=style['guid_color'],
            label=f'{style["label"]} (Guided)', markersize=3, linewidth=2
        )
        ax.fill_between(
            L, np.maximum(p25_g, 0), p75_g,
            color=style['guid_color'], alpha=FILL_ALPHA,
            edgecolor=style['guid_color'], linewidth=1.5
        )

    ax.set_xlabel('Trajectory Length')
    ax.set_ylabel('NMSE')
    ax.set_title(title)


def add_interleaved_legend(fig, axes):
    handles, labels = axes[0, 0].get_legend_handles_labels()
    n = len(policies)
    if len(handles) >= 2 * n:
        new_handles, new_labels = [], []
        for i in range(n):
            new_handles.append(handles[2 * i])
            new_handles.append(handles[2 * i + 1])
            new_labels.append(labels[2 * i])
            new_labels.append(labels[2 * i + 1])
        handles, labels = new_handles, new_labels

    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=14,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.02))


# Mean ± std figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col_idx, (col_suffix, stat_label) in enumerate([('q', r'$\mathrm{NMSE}_q$'), ('p', r'$\mathrm{NMSE}_p$')]):
    plot_nmse_subplot_mean_std(axes[0, col_idx], '2dof', col_suffix, f'2DoF - {stat_label}')
    plot_nmse_subplot_mean_std(axes[1, col_idx], '3dof', col_suffix, f'3DoF - {stat_label}')

add_interleaved_legend(fig, axes)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = project_root / 'plots' / 'ablation_nmse_combined.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print(f'Saved: {out_path}')

# Median ± IQR figure (IQR band shown as p25 to p75)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col_idx, (col_suffix, stat_label) in enumerate([('q', r'$\mathrm{NMSE}_q$'), ('p', r'$\mathrm{NMSE}_p$')]):
    plot_nmse_subplot_median_iqr(axes[0, col_idx], '2dof', col_suffix, f'2DoF - {stat_label}')
    plot_nmse_subplot_median_iqr(axes[1, col_idx], '3dof', col_suffix, f'3DoF - {stat_label}')

add_interleaved_legend(fig, axes)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = project_root / 'plots' / 'ablation_nmse_combined_median_iqr.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print(f'Saved: {out_path}')
