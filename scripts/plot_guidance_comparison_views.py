"""Create robust comparison views for guidance quality from per-sample metrics.

Outputs (with zero policy included):
1) ratio curves: (guided + eps) / (unguided + eps), mean ± std
2) improvement probability curves: P(guided < unguided)
3) all-3-metrics win-rate heatmaps: P(ΔNMSE_q<0 & ΔNMSE_p<0 & ΔHamRes<0)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


project_root = Path('/home/gsang/Projects/Perceiver_IO')
metrics_csv = project_root / 'output_ablation' / 'default_full_sweep' / 'metrics' / 'metrics_per_sample.csv'
summary_out = project_root / 'output_ablation' / 'default_full_sweep' / 'metrics' / 'guidance_compare_views_summary_Lle1000.csv'
plots_dir = project_root / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 1000
EPS = 1e-8

SYSTEMS = ['2dof', '3dof']
POLICIES = ['sinusoidal', 'gp', 'zero', 'spline']
METRICS = ['nmse_q', 'nmse_p', 'hamres']

FILL_ALPHA = 0.12

TORQUE_STYLES = {
    'sinusoidal': {'label': 'Sin.', 'color': '#1f77b4'},
    'gp': {'label': 'GP', 'color': '#d62728'},
    'zero': {'label': 'Zero', 'color': '#2ca02c'},
    'spline': {'label': 'Cubic Spline', 'color': '#8b00ff'},
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Ratio view: ratio < 1 means guidance improved.
    for m in METRICS:
        df[f'ratio_{m}'] = (df[f'gui_{m}'] + EPS) / (df[f'ung_{m}'] + EPS)
        df[f'imp_{m}'] = (df[f'gui_{m}'] < df[f'ung_{m}']).astype(float)

    df['all3_win'] = (
        (df['gui_nmse_q'] < df['ung_nmse_q']) &
        (df['gui_nmse_p'] < df['ung_nmse_p']) &
        (df['gui_hamres'] < df['ung_hamres'])
    ).astype(float)

    agg = {
        'sample_idx': 'count',
        'all3_win': 'mean',
    }
    for m in METRICS:
        agg[f'ratio_{m}'] = ['mean', 'std']
        agg[f'imp_{m}'] = ['mean', 'std']

    grouped = (
        df.groupby(['system', 'policy', 'length'], as_index=False)
        .agg(agg)
    )

    # Flatten multi-index columns
    flat_cols = []
    for c in grouped.columns:
        if isinstance(c, tuple):
            if c[1] == '':
                flat_cols.append(c[0])
            else:
                flat_cols.append(f'{c[0]}_{c[1]}')
        else:
            flat_cols.append(c)
    grouped.columns = flat_cols
    grouped = grouped.rename(columns={
        'sample_idx_count': 'n_samples',
        'all3_win_mean': 'all3_win_rate',
    })

    grouped.to_csv(summary_out, index=False)
    print(f'Saved summary: {summary_out} ({len(grouped)} rows)')
    return grouped


def _interleaved_legend(fig, axes):
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=12,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.02))


def plot_ratio(summary: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metric_labels = {
        'nmse_q': r'$(\mathrm{gui}+\epsilon)/(\mathrm{ung}+\epsilon)$ for $\mathrm{NMSE}_q$',
        'nmse_p': r'$(\mathrm{gui}+\epsilon)/(\mathrm{ung}+\epsilon)$ for $\mathrm{NMSE}_p$',
        'hamres': r'$(\mathrm{gui}+\epsilon)/(\mathrm{ung}+\epsilon)$ for HamRes',
    }

    for r, system in enumerate(SYSTEMS):
        for c, m in enumerate(METRICS):
            ax = axes[r, c]
            for pol in POLICIES:
                sub = summary[(summary['system'] == system) & (summary['policy'] == pol)].sort_values('length')
                if sub.empty:
                    continue
                L = sub['length'].to_numpy()
                mu = sub[f'ratio_{m}_mean'].to_numpy()
                sd = sub[f'ratio_{m}_std'].fillna(0.0).to_numpy()
                style = TORQUE_STYLES[pol]
                ax.plot(L, mu, marker='o', color=style['color'], linewidth=2, markersize=3, label=style['label'])
                ax.fill_between(L, np.maximum(mu - sd, 0.0), mu + sd, color=style['color'], alpha=FILL_ALPHA, linewidth=0)
            ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.9)
            ax.set_title(f"{system.upper()} - {m}")
            ax.set_xlabel('Trajectory Length')
            ax.set_ylabel(metric_labels[m])

    _interleaved_legend(fig, axes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = plots_dir / 'ablation_guidance_ratio_meanstd.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'Saved: {out}')


def plot_prob_improve(summary: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metric_labels = {
        'nmse_q': r'$P(\mathrm{guided}<\mathrm{unguided})$ for $\mathrm{NMSE}_q$',
        'nmse_p': r'$P(\mathrm{guided}<\mathrm{unguided})$ for $\mathrm{NMSE}_p$',
        'hamres': r'$P(\mathrm{guided}<\mathrm{unguided})$ for HamRes',
    }

    for r, system in enumerate(SYSTEMS):
        for c, m in enumerate(METRICS):
            ax = axes[r, c]
            for pol in POLICIES:
                sub = summary[(summary['system'] == system) & (summary['policy'] == pol)].sort_values('length')
                if sub.empty:
                    continue
                L = sub['length'].to_numpy()
                p = sub[f'imp_{m}_mean'].to_numpy()
                n = sub['n_samples'].to_numpy()
                # Binomial SE for reliability shading
                se = np.sqrt(np.clip(p * (1.0 - p) / np.maximum(n, 1), 0.0, None))
                lo = np.clip(p - 1.96 * se, 0.0, 1.0)
                hi = np.clip(p + 1.96 * se, 0.0, 1.0)
                style = TORQUE_STYLES[pol]
                ax.plot(L, p, marker='o', color=style['color'], linewidth=2, markersize=3, label=style['label'])
                ax.fill_between(L, lo, hi, color=style['color'], alpha=FILL_ALPHA, linewidth=0)
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.9)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{system.upper()} - {m}")
            ax.set_xlabel('Trajectory Length')
            ax.set_ylabel(metric_labels[m])

    _interleaved_legend(fig, axes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = plots_dir / 'ablation_guidance_p_improve.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'Saved: {out}')


def plot_all3_heatmap(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True, constrained_layout=True)
    lengths = sorted(summary['length'].unique().tolist())

    for i, system in enumerate(SYSTEMS):
        ax = axes[i]
        sub = summary[summary['system'] == system]
        mat = np.full((len(POLICIES), len(lengths)), np.nan, dtype=float)
        for r, pol in enumerate(POLICIES):
            sp = sub[sub['policy'] == pol]
            for _, row in sp.iterrows():
                c = lengths.index(int(row['length']))
                mat[r, c] = float(row['all3_win_rate'])

        im = ax.imshow(mat, aspect='auto', vmin=0.0, vmax=1.0, cmap='viridis')
        ax.set_title(f'{system.upper()} - All3 Win Rate')
        ax.set_xlabel('Trajectory Length')
        ax.set_xticks(np.arange(len(lengths)))
        # Reduce crowded ticks
        show_every = max(1, len(lengths) // 10)
        xticklabels = [str(lengths[k]) if k % show_every == 0 else '' for k in range(len(lengths))]
        ax.set_xticklabels(xticklabels, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(POLICIES)))
        ax.set_yticklabels([TORQUE_STYLES[p]['label'] for p in POLICIES])

    cbar = fig.colorbar(im, ax=axes, location='right', fraction=0.035, pad=0.02)
    cbar.set_label('P(Δq<0 & Δp<0 & Δhamres<0)')
    out = plots_dir / 'ablation_guidance_all3_winrate_heatmap.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'Saved: {out}')


def main():
    df = pd.read_csv(metrics_csv)
    df = df[df['length'] <= MAX_LENGTH].copy()
    print(f'Loaded: {metrics_csv} ({len(df)} rows after length<={MAX_LENGTH})')

    summary = build_summary(df)
    plot_ratio(summary)
    plot_prob_improve(summary)
    plot_all3_heatmap(summary)


if __name__ == '__main__':
    main()
