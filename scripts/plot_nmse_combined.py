"""
Combine 2DoF and 3DoF NMSE plots.
Mean lines + std fill_between bands. Linear scale.
Both systems use σ=5 smoothed data + separate HNN rollout baselines.
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
    'spline': {'label': 'Cubic Spline', 'guid_color': '#9467bd', 'ung_color': '#b09fd0'},
}

policies = ['sinusoidal', 'gp', 'zero', 'spline']
MAX_LENGTH = 1100
FILL_ALPHA = 0.12

# Load 2DoF data (σ=5 smoothed)
data_2dof = {}
results_dir_2dof = project_root / 'output_ablation' / 'results' / '2dof_smoothed'
for pol in policies:
    path = results_dir_2dof / f'metrics_{pol}_sigma5.0.csv'
    if path.exists():
        df = pd.read_csv(path)
        data_2dof[pol] = df[df['trajectory_length'] <= MAX_LENGTH]
        print(f"Loaded 2DoF {pol}: {len(data_2dof[pol])} rows")

# Load 3DoF data (σ=5 smoothed)
data_3dof = {}
results_dir_3dof = project_root / 'output_ablation' / 'results' / '3dof_smoothed'
for pol in policies:
    path = results_dir_3dof / f'metrics_{pol}_sigma5.0.csv'
    if path.exists():
        df = pd.read_csv(path)
        data_3dof[pol] = df[df['trajectory_length'] <= MAX_LENGTH]
        print(f"Loaded 3DoF {pol}: {len(data_3dof[pol])} rows")

# Load HNN rollout data (separate files for both systems)
data_2dof_hnn = {}
for pol in policies:
    path = results_dir_2dof / f'hnn_rollout_nmse_{pol}.csv'
    if path.exists():
        df = pd.read_csv(path)
        data_2dof_hnn[pol] = df[df['trajectory_length'] <= MAX_LENGTH]

data_3dof_hnn = {}
for pol in policies:
    path = results_dir_3dof / f'hnn_rollout_nmse_{pol}.csv'
    if path.exists():
        df = pd.read_csv(path)
        data_3dof_hnn[pol] = df[df['trajectory_length'] <= MAX_LENGTH]


def plot_nmse_subplot(ax, data, data_hnn, col_suffix, title):
    """Plot mean + std bands for one subplot."""
    mean_ung_col = f'unguided_nmse_{col_suffix}_mean'
    std_ung_col = f'unguided_nmse_{col_suffix}_std'
    mean_gui_col = f'guided_nmse_{col_suffix}_mean'
    std_gui_col = f'guided_nmse_{col_suffix}_std'

    for pol in policies:
        if pol not in data:
            continue
        df = data[pol]
        style = TORQUE_STYLES[pol]
        L = df['trajectory_length'].values

        # Unguided: mean + std band
        if mean_ung_col in df.columns:
            mean_u = df[mean_ung_col].values
            std_u = df[std_ung_col].values if std_ung_col in df.columns else np.zeros_like(mean_u)
            ax.plot(L, mean_u, marker='o', linestyle='--', color=style['ung_color'],
                    alpha=0.8, label=f'{style["label"]} (Unguided)', markersize=2, linewidth=1.5)
            fill_u = ax.fill_between(L, np.maximum(mean_u - std_u, 0), mean_u + std_u,
                            color=style['ung_color'], alpha=FILL_ALPHA,
                            edgecolor=style['ung_color'], linewidth=1.8)
            fill_u.set_linestyle('--')
            fill_u.set_edgecolor(style['guid_color'])

        # Guided: mean + std band
        if mean_gui_col in df.columns:
            mean_g = df[mean_gui_col].values
            std_g = df[std_gui_col].values if std_gui_col in df.columns else np.zeros_like(mean_g)
            ax.plot(L, mean_g, marker='s', linestyle='-', color=style['guid_color'],
                    label=f'{style["label"]} (Guided)', markersize=3, linewidth=2)
            ax.fill_between(L, np.maximum(mean_g - std_g, 0), mean_g + std_g,
                            color=style['guid_color'], alpha=FILL_ALPHA,
                            edgecolor=style['guid_color'], linewidth=1.5)

    # HNN Rollout: mean + std band
    hnn_mean_col = f'hnn_nmse_{col_suffix}_mean'
    hnn_std_col = f'hnn_nmse_{col_suffix}_std'
    hnn_mean_vals, hnn_std_vals = [], []
    for pol in policies:
        if pol in data_hnn and hnn_mean_col in data_hnn[pol].columns:
            hnn_mean_vals.append(data_hnn[pol][hnn_mean_col].values)
        if pol in data_hnn and hnn_std_col in data_hnn[pol].columns:
            hnn_std_vals.append(data_hnn[pol][hnn_std_col].values)
    if hnn_mean_vals:
        hnn_avg = np.nanmean(hnn_mean_vals, axis=0)
        first_pol = next(iter(data_hnn.keys()))
        L = data_hnn[first_pol]['trajectory_length'].values
        ax.plot(L, hnn_avg, marker='D', linestyle='-', color='#ff7f0e', linewidth=2,
                markersize=3, label='HNN Rollout')
        if hnn_std_vals:
            hnn_std_avg = np.nanmean(hnn_std_vals, axis=0)
            ax.fill_between(L, np.maximum(hnn_avg - hnn_std_avg, 0), hnn_avg + hnn_std_avg,
                            color='#ff7f0e', alpha=FILL_ALPHA,
                            edgecolor='#ff7f0e', linewidth=1.5)

    ax.set_xlabel('Trajectory Length')
    ax.set_ylabel('NMSE')
    ax.set_title(title)


# Create combined figure: 2 rows x 2 cols
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col_idx, (col_suffix, stat_label) in enumerate([('q', r'$\mathrm{NMSE}_q$'), ('p', r'$\mathrm{NMSE}_p$')]):
    plot_nmse_subplot(axes[0, col_idx], data_2dof, data_2dof_hnn, col_suffix, f'2DoF - {stat_label}')
    plot_nmse_subplot(axes[1, col_idx], data_3dof, data_3dof_hnn, col_suffix, f'3DoF - {stat_label}')

# Create legend (top center) — interleave unguided/guided per policy
handles, labels = axes[0, 0].get_legend_handles_labels()
n = len([p for p in policies if p in data_2dof])
if len(handles) >= 2 * n:
    new_handles, new_labels = [], []
    for i in range(n):
        new_handles.append(handles[2 * i])
        new_handles.append(handles[2 * i + 1])
        new_labels.append(labels[2 * i])
        new_labels.append(labels[2 * i + 1])
    if len(handles) > 2 * n:
        new_handles.append(handles[-1])
        new_labels.append(labels[-1])
    handles, labels = new_handles, new_labels

fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=14,
           framealpha=0.9, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = project_root / 'plots' / 'ablation_nmse_combined.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print(f'Saved: {out_path}')
