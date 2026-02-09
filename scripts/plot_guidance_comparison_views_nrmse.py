"""Create guidance comparison views using NRMSE(range) + HamRes.

This script:
1) Builds sample-level NRMSE(range) metrics from saved trajectories (cached to CSV).
2) Merges with sample-level HamRes from metrics_per_sample.csv.
3) Plots:
   - ratio mean±std: (guided+eps)/(unguided+eps)
   - P(improve): P(guided < unguided)
   - all3 win-rate heatmap: P(Δnrmse_q<0 & Δnrmse_p<0 & Δhamres<0)
"""
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

from compute_ablation_2dof_with_smoothing import SYSTEM_CONFIGS, DT, SIM_DT
from src.models.utils import reconstruct_traj_with_momentum


project_root = Path('/home/gsang/Projects/Perceiver_IO')
traj_root = project_root / 'output_ablation' / 'default_full_sweep' / 'trajectories'
metrics_root = project_root / 'output_ablation' / 'default_full_sweep' / 'metrics'
plots_dir = project_root / 'plots'
metrics_root.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 1000
MIN_SCALE_Q = 1e-3
MIN_SCALE_P = 1e-3
EPS = 1e-8

NMRSE_PER_SAMPLE_CSV = metrics_root / 'nrmse_range_per_sample_Lle1000.csv'
SUMMARY_OUT = metrics_root / 'guidance_compare_views_nrmse_hamres_summary_Lle1000.csv'
HAMRES_PER_SAMPLE_CSV = metrics_root / 'metrics_per_sample.csv'

SYSTEMS = ['2dof', '3dof']
POLICIES = ['sinusoidal', 'gp', 'zero', 'spline']

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


def _parse_combo_from_filename(path: Path):
    m = re.match(r'^(2dof|3dof)_(sinusoidal|gp|zero|spline)_L(\d+)_N(\d+)_seed(\d+)\.pt$', path.name)
    if not m:
        return None
    return {
        'combo': path.stem,
        'system': m.group(1),
        'policy': m.group(2),
        'length': int(m.group(3)),
        'num_samples': int(m.group(4)),
        'seed': int(m.group(5)),
    }


def _compute_nrmse_range_for_sample(state, torque, mj_model, qpos_dim):
    qpos = state[:, :qpos_dim].cpu().numpy()
    mom = state[:, qpos_dim:].cpu().numpy()
    torque_np = torque.cpu().numpy()
    T = qpos.shape[0]

    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos[0]
    data.qvel[:] = 0
    mujoco.mj_forward(mj_model, data)
    M = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, M, data.qM)
    initial_qvel = np.linalg.solve(M, mom[0])

    recon = reconstruct_traj_with_momentum(
        mj_model, T, SIM_DT, qpos[0], initial_qvel, torque_np, data_dt=DT
    )
    gt_qpos = recon['seq_qpos']
    gt_mom = recon['seq_mom']

    T_min = min(len(qpos) - 1, len(gt_qpos))
    if T_min <= 0:
        return np.nan, np.nan

    e_q = qpos[1:T_min + 1] - gt_qpos[:T_min]
    e_p = mom[1:T_min + 1] - gt_mom[:T_min]
    rmse_q = np.sqrt((e_q ** 2).mean(axis=0))
    rmse_p = np.sqrt((e_p ** 2).mean(axis=0))

    range_q = gt_qpos[:T_min].max(axis=0) - gt_qpos[:T_min].min(axis=0)
    range_p = gt_mom[:T_min].max(axis=0) - gt_mom[:T_min].min(axis=0)
    scale_q = np.maximum(range_q, MIN_SCALE_Q)
    scale_p = np.maximum(range_p, MIN_SCALE_P)

    nrmse_q = rmse_q / scale_q
    nrmse_p = rmse_p / scale_p
    return float(nrmse_q.mean()), float(nrmse_p.mean())


def build_nrmse_per_sample():
    if NMRSE_PER_SAMPLE_CSV.exists():
        df = pd.read_csv(NMRSE_PER_SAMPLE_CSV)
        print(f'Loaded cached NRMSE per-sample: {NMRSE_PER_SAMPLE_CSV} ({len(df)} rows)')
        return df

    rows = []
    all_pt = sorted(traj_root.glob('*/*/*.pt'))
    all_pt = [p for p in all_pt if _parse_combo_from_filename(p) is not None]
    print(f'Found trajectory files: {len(all_pt)}')

    for p in tqdm(all_pt, desc='NRMSE combos'):
        meta = _parse_combo_from_filename(p)
        if meta is None or meta['length'] > MAX_LENGTH:
            continue

        data = torch.load(p, map_location='cpu')
        ung_states = data['unguided_states']
        ung_torques = data['unguided_torques']
        gui_states = data['guided_states']
        gui_torques = data['guided_torques']

        cfg = SYSTEM_CONFIGS[meta['system']]
        qpos_dim = cfg['qpos_dim']
        mj_model = mujoco.MjModel.from_xml_path(cfg['xml_path'])
        mj_model.opt.timestep = SIM_DT

        n = min(ung_states.shape[0], gui_states.shape[0])
        for i in range(n):
            ung_q, ung_p = _compute_nrmse_range_for_sample(ung_states[i], ung_torques[i], mj_model, qpos_dim)
            gui_q, gui_p = _compute_nrmse_range_for_sample(gui_states[i], gui_torques[i], mj_model, qpos_dim)
            rows.append({
                'combo': meta['combo'],
                'system': meta['system'],
                'policy': meta['policy'],
                'length': meta['length'],
                'seed': meta['seed'],
                'sample_idx': i,
                'ung_nrmse_q': ung_q,
                'gui_nrmse_q': gui_q,
                'd_nrmse_q': gui_q - ung_q,
                'ung_nrmse_p': ung_p,
                'gui_nrmse_p': gui_p,
                'd_nrmse_p': gui_p - ung_p,
            })

    df = pd.DataFrame(rows).sort_values(['system', 'policy', 'length', 'combo', 'sample_idx']).reset_index(drop=True)
    df.to_csv(NMRSE_PER_SAMPLE_CSV, index=False)
    print(f'Saved NRMSE per-sample: {NMRSE_PER_SAMPLE_CSV} ({len(df)} rows)')
    return df


def build_merged_summary(nrmse_df: pd.DataFrame):
    hamres_df = pd.read_csv(HAMRES_PER_SAMPLE_CSV)
    hamres_df = hamres_df[hamres_df['length'] <= MAX_LENGTH].copy()
    keep = [
        'combo', 'system', 'policy', 'length', 'seed', 'sample_idx',
        'ung_hamres', 'gui_hamres', 'd_hamres',
    ]
    hamres_df = hamres_df[keep]

    merged = pd.merge(
        nrmse_df,
        hamres_df,
        on=['combo', 'system', 'policy', 'length', 'seed', 'sample_idx'],
        how='inner',
        validate='one_to_one',
    )
    print(f'Merged sample table rows: {len(merged)}')

    for m in ['nrmse_q', 'nrmse_p', 'hamres']:
        merged[f'ratio_{m}'] = (merged[f'gui_{m}'] + EPS) / (merged[f'ung_{m}'] + EPS)
        merged[f'imp_{m}'] = (merged[f'gui_{m}'] < merged[f'ung_{m}']).astype(float)

    merged['all3_win'] = (
        (merged['gui_nrmse_q'] < merged['ung_nrmse_q']) &
        (merged['gui_nrmse_p'] < merged['ung_nrmse_p']) &
        (merged['gui_hamres'] < merged['ung_hamres'])
    ).astype(float)

    agg = {'sample_idx': 'count', 'all3_win': 'mean'}
    for m in ['nrmse_q', 'nrmse_p', 'hamres']:
        agg[f'ratio_{m}'] = ['mean', 'std']
        agg[f'imp_{m}'] = ['mean', 'std']

    summary = merged.groupby(['system', 'policy', 'length'], as_index=False).agg(agg)
    summary.columns = [
        f'{a}_{b}' if isinstance(a, str) and b else a
        for a, b in [c if isinstance(c, tuple) else (c, '') for c in summary.columns]
    ]
    summary = summary.rename(columns={
        'sample_idx_count': 'n_samples',
        'all3_win_mean': 'all3_win_rate',
    })
    summary.to_csv(SUMMARY_OUT, index=False)
    print(f'Saved summary: {SUMMARY_OUT} ({len(summary)} rows)')
    return summary


def _add_shared_legend(fig, axes):
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=12,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.02))


def plot_ratio(summary: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ['nrmse_q', 'nrmse_p', 'hamres']

    for r, system in enumerate(SYSTEMS):
        for c, m in enumerate(metrics):
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
                ax.fill_between(L, np.maximum(mu - sd, 0.0), mu + sd, color=style['color'], alpha=FILL_ALPHA)
            ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.9)
            ax.set_title(f'{system.upper()} - {m}')
            ax.set_xlabel('Trajectory Length')
            ax.set_ylabel('(guided+eps)/(unguided+eps)')

    _add_shared_legend(fig, axes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = plots_dir / 'ablation_guidance_ratio_meanstd_nrmse_hamres.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'Saved: {out}')


def plot_p_improve(summary: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ['nrmse_q', 'nrmse_p', 'hamres']

    for r, system in enumerate(SYSTEMS):
        for c, m in enumerate(metrics):
            ax = axes[r, c]
            for pol in POLICIES:
                sub = summary[(summary['system'] == system) & (summary['policy'] == pol)].sort_values('length')
                if sub.empty:
                    continue
                L = sub['length'].to_numpy()
                p = sub[f'imp_{m}_mean'].to_numpy()
                n = sub['n_samples'].to_numpy()
                se = np.sqrt(np.clip(p * (1.0 - p) / np.maximum(n, 1), 0.0, None))
                lo = np.clip(p - 1.96 * se, 0.0, 1.0)
                hi = np.clip(p + 1.96 * se, 0.0, 1.0)
                style = TORQUE_STYLES[pol]
                ax.plot(L, p, marker='o', color=style['color'], linewidth=2, markersize=3, label=style['label'])
                ax.fill_between(L, lo, hi, color=style['color'], alpha=FILL_ALPHA)
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.9)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f'{system.upper()} - {m}')
            ax.set_xlabel('Trajectory Length')
            ax.set_ylabel('P(guided < unguided)')

    _add_shared_legend(fig, axes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = plots_dir / 'ablation_guidance_p_improve_nrmse_hamres.png'
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
        show_every = max(1, len(lengths) // 10)
        ax.set_xticklabels([str(lengths[k]) if k % show_every == 0 else '' for k in range(len(lengths))],
                           rotation=45, ha='right')
        ax.set_yticks(np.arange(len(POLICIES)))
        ax.set_yticklabels([TORQUE_STYLES[p]['label'] for p in POLICIES])

    cbar = fig.colorbar(im, ax=axes, location='right', fraction=0.035, pad=0.02)
    cbar.set_label('P(Δnrmse_q<0 & Δnrmse_p<0 & Δhamres<0)')
    out = plots_dir / 'ablation_guidance_all3_winrate_heatmap_nrmse_hamres.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'Saved: {out}')


def main():
    nrmse_df = build_nrmse_per_sample()
    summary = build_merged_summary(nrmse_df)
    plot_ratio(summary)
    plot_p_improve(summary)
    plot_all3_heatmap(summary)


if __name__ == '__main__':
    main()
