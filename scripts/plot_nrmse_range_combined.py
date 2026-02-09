"""Plot combined 2DoF/3DoF NRMSE(range) from default full sweep (mean ± std)."""
import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
import mujoco
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
plots_root = project_root / 'plots'
metrics_root.mkdir(parents=True, exist_ok=True)
plots_root.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = metrics_root / 'nrmse_range_summary_Lle1000.csv'
MAX_LENGTH = 1000
MIN_SCALE_Q = 1e-3
MIN_SCALE_P = 1e-3

TORQUE_STYLES = {
    'sinusoidal': {'label': 'Sin.', 'guid_color': '#1f77b4', 'ung_color': '#6baed6'},
    'gp': {'label': 'GP', 'guid_color': '#d62728', 'ung_color': '#e6756b'},
    'zero': {'label': 'Zero', 'guid_color': '#2ca02c', 'ung_color': '#74c476'},
    'spline': {'label': 'Cubic Spline', 'guid_color': '#8b00ff', 'ung_color': '#c77dff'},
}
POLICIES = ['sinusoidal', 'gp', 'zero', 'spline']
FILL_ALPHA = 0.12


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


def _parse_combo_from_filename(path: Path):
    m = re.match(r'^(2dof|3dof)_(sinusoidal|gp|zero|spline)_L(\d+)_N(\d+)_seed(\d+)\.pt$', path.name)
    if not m:
        return None
    return {
        'system': m.group(1),
        'policy': m.group(2),
        'length': int(m.group(3)),
        'num_samples': int(m.group(4)),
        'seed': int(m.group(5)),
        'combo': path.stem,
    }


def _stats(arr):
    a = np.asarray(arr, dtype=float)
    return {
        'mean': float(np.mean(a)),
        'std': float(np.std(a)),
        'p25': float(np.percentile(a, 25)),
        'median': float(np.median(a)),
        'p75': float(np.percentile(a, 75)),
        'p99': float(np.percentile(a, 99)),
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


def build_summary():
    rows = []
    all_pt = sorted(traj_root.glob('*/*/*.pt'))
    all_pt = [p for p in all_pt if _parse_combo_from_filename(p) is not None]
    print(f'Found trajectory files: {len(all_pt)}')

    for p in tqdm(all_pt, desc='Combos'):
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
        ung_q, ung_p, gui_q, gui_p = [], [], [], []
        for i in range(n):
            nq, np_ = _compute_nrmse_range_for_sample(ung_states[i], ung_torques[i], mj_model, qpos_dim)
            gq, gp = _compute_nrmse_range_for_sample(gui_states[i], gui_torques[i], mj_model, qpos_dim)
            ung_q.append(nq)
            ung_p.append(np_)
            gui_q.append(gq)
            gui_p.append(gp)

        ung_q = np.asarray(ung_q, dtype=float)
        ung_p = np.asarray(ung_p, dtype=float)
        gui_q = np.asarray(gui_q, dtype=float)
        gui_p = np.asarray(gui_p, dtype=float)
        d_q = gui_q - ung_q
        d_p = gui_p - ung_p

        row = {
            'combo': meta['combo'],
            'system': meta['system'],
            'policy': meta['policy'],
            'length': meta['length'],
            'num_samples': int(n),
            'seed': meta['seed'],
        }
        for name, arr in [
            ('ung_nrmse_range_q', ung_q),
            ('gui_nrmse_range_q', gui_q),
            ('d_nrmse_range_q', d_q),
            ('ung_nrmse_range_p', ung_p),
            ('gui_nrmse_range_p', gui_p),
            ('d_nrmse_range_p', d_p),
        ]:
            st = _stats(arr)
            for k, v in st.items():
                row[f'{name}_{k}'] = v
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(['system', 'policy', 'length']).reset_index(drop=True)
    out.to_csv(SUMMARY_CSV, index=False)
    print(f'Saved summary: {SUMMARY_CSV} ({len(out)} rows)')
    return out


def _plot_subplot(ax, df_all, system, suffix, title):
    for pol in POLICIES:
        df = df_all[(df_all['system'] == system) & (df_all['policy'] == pol)].copy()
        if df.empty:
            continue
        df = df.sort_values('length')
        style = TORQUE_STYLES[pol]
        L = df['length'].to_numpy()

        mu = df[f'ung_nrmse_range_{suffix}_mean'].to_numpy()
        su = df[f'ung_nrmse_range_{suffix}_std'].to_numpy()
        ax.plot(L, mu, marker='o', linestyle='--', color=style['ung_color'],
                alpha=0.8, label=f'{style["label"]} (Unguided)', markersize=2, linewidth=1.5)
        fill_u = ax.fill_between(L, np.maximum(mu - su, 0), mu + su,
                                 color=style['ung_color'], alpha=FILL_ALPHA,
                                 edgecolor=style['ung_color'], linewidth=1.8)
        fill_u.set_linestyle('--')
        fill_u.set_edgecolor(style['guid_color'])

        mg = df[f'gui_nrmse_range_{suffix}_mean'].to_numpy()
        sg = df[f'gui_nrmse_range_{suffix}_std'].to_numpy()
        ax.plot(L, mg, marker='s', linestyle='-', color=style['guid_color'],
                label=f'{style["label"]} (Guided)', markersize=3, linewidth=2)
        ax.fill_between(L, np.maximum(mg - sg, 0), mg + sg,
                        color=style['guid_color'], alpha=FILL_ALPHA,
                        edgecolor=style['guid_color'], linewidth=1.5)

    ax.set_xlabel('Trajectory Length')
    ax.set_ylabel('NRMSE(range)')
    ax.set_title(title)


def make_plot(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _plot_subplot(axes[0, 0], df, '2dof', 'q', r'2DoF - $\mathrm{NRMSE}_{q,\mathrm{range}}$')
    _plot_subplot(axes[0, 1], df, '2dof', 'p', r'2DoF - $\mathrm{NRMSE}_{p,\mathrm{range}}$')
    _plot_subplot(axes[1, 0], df, '3dof', 'q', r'3DoF - $\mathrm{NRMSE}_{q,\mathrm{range}}$')
    _plot_subplot(axes[1, 1], df, '3dof', 'p', r'3DoF - $\mathrm{NRMSE}_{p,\mathrm{range}}$')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    n = len(POLICIES)
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
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out = plots_root / 'ablation_nrmse_range_combined.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'Saved plot: {out}')


def main():
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        print(f'Loaded cached summary: {SUMMARY_CSV} ({len(df)} rows)')
    else:
        df = build_summary()
    make_plot(df)


if __name__ == '__main__':
    main()
