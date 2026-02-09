"""Plot combined 2DoF/3DoF HamRes from default full sweep (linear scale)."""
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

metrics_path = project_root / 'output_ablation' / 'default_full_sweep' / 'metrics' / 'metrics_summary.csv'
df_all = pd.read_csv(metrics_path)
print(f'Loaded metrics: {metrics_path} ({len(df_all)} rows)')


def make_plot_mean_std(out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    MAX_LENGTH = 1000

    # ── Left: 2DoF (mean ± std bands) ──
    ax = axes[0]
    for pol in policies:
        df = df_all[(df_all['system'] == '2dof') & (df_all['policy'] == pol)].copy()
        if df.empty:
            continue
        df = df[df['length'] <= MAX_LENGTH].sort_values('length')
        style = TORQUE_STYLES[pol]
        L = df['length'].values

        mean_u = df['ung_hamres_mean'].values
        std_u = df['ung_hamres_std'].values
        ax.plot(L, mean_u, marker='o', linestyle='--', color=style['ung_color'],
                alpha=0.8, label=f'{style["label"]} (Unguided)', markersize=2, linewidth=1.5)
        fill_u = ax.fill_between(L, np.maximum(mean_u - std_u, 0), mean_u + std_u,
                                 color=style['ung_color'], alpha=FILL_ALPHA,
                                 edgecolor=style['ung_color'], linewidth=1.8)
        fill_u.set_linestyle('--')
        fill_u.set_edgecolor(style['guid_color'])

        mean_g = df['gui_hamres_mean'].values
        std_g = df['gui_hamres_std'].values
        ax.plot(L, mean_g, marker='s', linestyle='-', color=style['guid_color'],
                label=f'{style["label"]} (Guided)', markersize=3, linewidth=2)
        ax.fill_between(L, np.maximum(mean_g - std_g, 0), mean_g + std_g,
                        color=style['guid_color'], alpha=FILL_ALPHA,
                        edgecolor=style['guid_color'], linewidth=1.5)
    ax.set_xlabel('Trajectory Length')
    ax.set_ylabel('HamRes')
    ax.set_title('2DoF')

    # ── Right: 3DoF (mean ± std bands) ──
    ax = axes[1]
    for pol in policies:
        df = df_all[(df_all['system'] == '3dof') & (df_all['policy'] == pol)].copy()
        if df.empty:
            continue
        df = df[df['length'] <= MAX_LENGTH].sort_values('length')
        style = TORQUE_STYLES[pol]
        L = df['length'].values

        mean_u = df['ung_hamres_mean'].values
        std_u = df['ung_hamres_std'].values
        ax.plot(L, mean_u, marker='o', linestyle='--', color=style['ung_color'],
                alpha=0.8, label=f'{style["label"]} (Unguided)', markersize=2, linewidth=1.5)
        fill_u = ax.fill_between(L, np.maximum(mean_u - std_u, 0), mean_u + std_u,
                        color=style['ung_color'], alpha=FILL_ALPHA,
                        edgecolor=style['ung_color'], linewidth=1.8)
        fill_u.set_linestyle('--')
        fill_u.set_edgecolor(style['guid_color'])

        mean_g = df['gui_hamres_mean'].values
        std_g = df['gui_hamres_std'].values
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
    n = len(policies)
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


def make_plot_median_iqr(out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    MAX_LENGTH = 1000

    # ── Left: 2DoF (median ± IQR bands) ──
    ax = axes[0]
    for pol in policies:
        df = df_all[(df_all['system'] == '2dof') & (df_all['policy'] == pol)].copy()
        if df.empty:
            continue
        df = df[df['length'] <= MAX_LENGTH].sort_values('length')
        style = TORQUE_STYLES[pol]
        L = df['length'].values

        med_u = df['ung_hamres_median'].values
        q1_u = df['ung_hamres_p25'].values
        q3_u = df['ung_hamres_p75'].values
        ax.plot(L, med_u, marker='o', linestyle='--', color=style['ung_color'],
                alpha=0.8, label=f'{style["label"]} (Unguided)', markersize=2, linewidth=1.5)
        fill_u = ax.fill_between(L, np.maximum(q1_u, 0), q3_u,
                                 color=style['ung_color'], alpha=FILL_ALPHA,
                                 edgecolor=style['ung_color'], linewidth=1.8)
        fill_u.set_linestyle('--')
        fill_u.set_edgecolor(style['guid_color'])

        med_g = df['gui_hamres_median'].values
        q1_g = df['gui_hamres_p25'].values
        q3_g = df['gui_hamres_p75'].values
        ax.plot(L, med_g, marker='s', linestyle='-', color=style['guid_color'],
                label=f'{style["label"]} (Guided)', markersize=3, linewidth=2)
        ax.fill_between(L, np.maximum(q1_g, 0), q3_g,
                        color=style['guid_color'], alpha=FILL_ALPHA,
                        edgecolor=style['guid_color'], linewidth=1.5)
    ax.set_xlabel('Trajectory Length')
    ax.set_ylabel('HamRes')
    ax.set_title('2DoF')

    # ── Right: 3DoF (median ± IQR bands) ──
    ax = axes[1]
    for pol in policies:
        df = df_all[(df_all['system'] == '3dof') & (df_all['policy'] == pol)].copy()
        if df.empty:
            continue
        df = df[df['length'] <= MAX_LENGTH].sort_values('length')
        style = TORQUE_STYLES[pol]
        L = df['length'].values

        med_u = df['ung_hamres_median'].values
        q1_u = df['ung_hamres_p25'].values
        q3_u = df['ung_hamres_p75'].values
        ax.plot(L, med_u, marker='o', linestyle='--', color=style['ung_color'],
                alpha=0.8, label=f'{style["label"]} (Unguided)', markersize=2, linewidth=1.5)
        fill_u = ax.fill_between(L, np.maximum(q1_u, 0), q3_u,
                                 color=style['ung_color'], alpha=FILL_ALPHA,
                                 edgecolor=style['ung_color'], linewidth=1.8)
        fill_u.set_linestyle('--')
        fill_u.set_edgecolor(style['guid_color'])

        med_g = df['gui_hamres_median'].values
        q1_g = df['gui_hamres_p25'].values
        q3_g = df['gui_hamres_p75'].values
        ax.plot(L, med_g, marker='s', linestyle='-', color=style['guid_color'],
                label=f'{style["label"]} (Guided)', markersize=3, linewidth=2)
        ax.fill_between(L, np.maximum(q1_g, 0), q3_g,
                        color=style['guid_color'], alpha=FILL_ALPHA,
                        edgecolor=style['guid_color'], linewidth=1.5)
    ax.set_xlabel('Trajectory Length')
    ax.set_ylabel('HamRes')
    ax.set_title('3DoF')

    handles, labels = axes[0].get_legend_handles_labels()
    n = len(policies)
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
make_plot_mean_std(out_path=plots_dir / 'ablation_hamres_combined_linear.png')
make_plot_median_iqr(out_path=plots_dir / 'ablation_hamres_combined_linear_median_iqr.png')
