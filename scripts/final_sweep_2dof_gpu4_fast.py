import os
import csv
import sys
from collections import defaultdict

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import scripts.quick_eval as qe


def eval_config(dpf, hnn, cfg_sys, mj_model, var_dq, var_dp, hamres_kwargs, qpos_dim, conf, settings, seeds, num_samples, device):
    rows = []
    for policy, L in settings:
        for seed in seeds:
            qe.set_seed(seed)
            torques = qe.load_torques(policy, num_samples, L, device, cfg_sys)
            noise = torch.randn(num_samples, L, dpf.state_dim, device=device)

            ung_states, ung_torques = qe.run_batch(
                dpf, torques, noise, L, qe.BATCH_SIZE_UNGUIDED,
                hnn=None, guidance_steps=0, smooth_sigma=0, num_diffusion_steps=20,
            )
            u_nq, u_np, u_hr, *_ = qe.compute_metrics_for_samples(
                ung_states, ung_torques, num_samples, mj_model, hnn, var_dq, var_dp, qpos_dim,
                desc=f"ung_{policy}_L{L}_s{seed}", hamres_kwargs=hamres_kwargs
            )
            u = (float(np.median(u_nq)), float(np.median(u_np)), float(np.median(u_hr)))

            gui_states, gui_torques = qe.run_batch(
                dpf, torques, noise, L, qe.BATCH_SIZE_GUIDED,
                hnn=hnn, guidance_method='normalized_sgd', guidance_steps=conf['steps'], guidance_lr=0.01,
                guidance_after_steps=conf['after'], guidance_before_steps=20, optimize_target='both',
                guidance_energy_mode=conf['mode'], guidance_hamres_smooth_sigma=conf.get('ham_sigma', 1.0),
                guidance_hamres_delta=conf.get('ham_delta', 1.0), guidance_hamres_min_scale_q=1e-3,
                guidance_hamres_min_scale_p=1e-3, guidance_trust_lambda=conf.get('trust', 0.0),
                smooth_sigma=0, smooth_guidance_only=False, smooth_last_step_only=False,
                alpha_q=conf['alpha'], alpha_p=conf['alpha'], num_diffusion_steps=20,
            )
            g_nq, g_np, g_hr, *_ = qe.compute_metrics_for_samples(
                gui_states, gui_torques, num_samples, mj_model, hnn, var_dq, var_dp, qpos_dim,
                desc=f"gui_{conf['name']}_{policy}_L{L}_s{seed}", hamres_kwargs=hamres_kwargs
            )
            g = (float(np.median(g_nq)), float(np.median(g_np)), float(np.median(g_hr)))
            d = (g[0] - u[0], g[1] - u[1], g[2] - u[2])
            rows.append(dict(
                config=conf['name'], policy=policy, length=L, seed=seed,
                ung_nq=u[0], ung_np=u[1], ung_hr=u[2], gui_nq=g[0], gui_np=g[1], gui_hr=g[2],
                d_nq=d[0], d_np=d[1], d_hr=d[2],
                win_nq=int(d[0] < 0), win_np=int(d[1] < 0), win_hr=int(d[2] < 0),
            ))
    return rows


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[r['config']].append(r)
    out = []
    for name, rs in by.items():
        dnq = np.array([r['d_nq'] for r in rs])
        dnp = np.array([r['d_np'] for r in rs])
        dhr = np.array([r['d_hr'] for r in rs])
        wnq = sum(r['win_nq'] for r in rs)
        wnp = sum(r['win_np'] for r in rs)
        whr = sum(r['win_hr'] for r in rs)
        wall = sum(int(r['win_nq'] and r['win_np'] and r['win_hr']) for r in rs)
        score = 0.4 * float(np.median(dnq)) + 0.4 * float(np.median(dnp)) + 0.2 * float(np.median(dhr))
        out.append(dict(
            config=name, n=len(rs),
            all_win=f"{wall}/{len(rs)}", win_nq=f"{wnq}/{len(rs)}", win_np=f"{wnp}/{len(rs)}", win_hr=f"{whr}/{len(rs)}",
            med_d_nq=float(np.median(dnq)), med_d_np=float(np.median(dnp)), med_d_hr=float(np.median(dhr)),
            score=score,
        ))
    out.sort(key=lambda x: (-int(x['all_win'].split('/')[0]), -int(x['win_nq'].split('/')[0]), -int(x['win_np'].split('/')[0]), -int(x['win_hr'].split('/')[0]), x['score']))
    return out


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    qe.SYSTEM = '2dof'
    cfg_sys = qe.SYSTEM_CONFIGS['2dof']
    dpf, hnn, var_dq, var_dp, mj_model = qe.load_models(cfg_sys, device)
    qpos_dim = cfg_sys['qpos_dim']
    hamres_kwargs = dict(
        smooth_sigma=qe.HAMRES_SMOOTH_SIGMA,
        delta=qe.HAMRES_PSEUDO_HUBER_DELTA,
        min_scale_q=qe.HAMRES_MIN_SCALE_Q,
        min_scale_p=qe.HAMRES_MIN_SCALE_P,
    )

    configs = [
        dict(name='one_step_best', mode='one_step', steps=50, after=15, alpha=1e-2, trust=0.0),
        dict(name='one_step_conservative', mode='one_step', steps=30, after=15, alpha=5e-3, trust=0.0),
        dict(name='one_step_aggressive', mode='one_step', steps=80, after=10, alpha=1e-2, trust=0.0),
        dict(name='robust_r3_sigma05', mode='robust_hamres', steps=50, after=15, alpha=1e-2, trust=0.0, ham_sigma=0.5, ham_delta=1.0),
        dict(name='robust_r9_comboA', mode='robust_hamres', steps=50, after=10, alpha=1e-2, trust=1e-3, ham_sigma=0.5, ham_delta=2.0),
        dict(name='robust_balanced', mode='robust_hamres', steps=50, after=10, alpha=5e-3, trust=1e-3, ham_sigma=0.5, ham_delta=1.0),
    ]

    # Stage A: smart shortlist
    stageA_settings = [('sinusoidal', 100), ('gp', 300), ('spline', 100), ('gp', 1000), ('zero', 300), ('spline', 1000)]
    stageA_seeds = [65, 66]
    num_samples = 6

    rowsA = []
    for c in configs:
        print('stageA', c['name'])
        rowsA.extend(eval_config(dpf, hnn, cfg_sys, mj_model, var_dq, var_dp, hamres_kwargs, qpos_dim, c, stageA_settings, stageA_seeds, num_samples, device))

    sumA = summarize(rowsA)
    top2_names = [sumA[0]['config'], sumA[1]['config']]
    top2 = [c for c in configs if c['name'] in top2_names]

    # Stage B: confirm top2 on full matrix
    stageB_settings = [(p, L) for p in ['sinusoidal', 'gp', 'zero', 'spline'] for L in [100, 300, 1000]]
    stageB_seeds = [65, 66]

    rowsB = []
    for c in top2:
        print('stageB', c['name'])
        rowsB.extend(eval_config(dpf, hnn, cfg_sys, mj_model, var_dq, var_dp, hamres_kwargs, qpos_dim, c, stageB_settings, stageB_seeds, num_samples, device))

    sumB = summarize(rowsB)

    out_dir = '/home/gsang/Projects/Perceiver_IO/output_ablation/sweep_outputs'
    os.makedirs(out_dir, exist_ok=True)
    pA = os.path.join(out_dir, 'final_fast_stageA_summary.csv')
    pB = os.path.join(out_dir, 'final_fast_stageB_summary.csv')
    pBR = os.path.join(out_dir, 'final_fast_stageB_rows.csv')

    for path, rows in [(pA, sumA), (pB, sumB), (pBR, rowsB)]:
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print('\n=== STAGE A SUMMARY ===')
    for s in sumA:
        print(s)
    print('\n=== STAGE B SUMMARY ===')
    for s in sumB:
        print(s)

    print('stageA_csv=', pA)
    print('stageB_csv=', pB)
    print('stageB_rows_csv=', pBR)


if __name__ == '__main__':
    main()
