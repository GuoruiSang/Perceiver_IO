import os
import csv
import math
import sys
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import scripts.quick_eval as qe


def cfg_name(c):
    keys = [
        'mode', 'steps', 'after', 'before', 'alpha_q', 'alpha_p',
        'trust', 'ham_sigma', 'ham_delta', 'num_diff_steps'
    ]
    parts = []
    for k in keys:
        if k in c:
            parts.append(f"{k}={c[k]}")
    return "|".join(parts)


def med3(a):
    return tuple(float(np.median(x)) for x in a)


def run_guided(dpf, hnn, torques, noise, L, c):
    kwargs = dict(
        hnn=hnn,
        guidance_method='normalized_sgd',
        guidance_steps=c['steps'],
        guidance_lr=0.01,
        guidance_after_steps=c['after'],
        guidance_before_steps=c['before'],
        optimize_target='both',
        guidance_energy_mode=c['mode'],
        guidance_hamres_smooth_sigma=c.get('ham_sigma', 1.0),
        guidance_hamres_delta=c.get('ham_delta', 1.0),
        guidance_hamres_min_scale_q=1e-3,
        guidance_hamres_min_scale_p=1e-3,
        guidance_trust_lambda=c.get('trust', 0.0),
        smooth_sigma=0,
        smooth_guidance_only=False,
        smooth_last_step_only=False,
        alpha_q=c['alpha_q'],
        alpha_p=c['alpha_p'],
        num_diffusion_steps=c['num_diff_steps'],
    )
    return qe.run_batch(dpf, torques, noise, L, qe.BATCH_SIZE_GUIDED, **kwargs)


def compute_metrics(states, torques, num_samples, mj_model, hnn, var_dq, var_dp, qpos_dim, hamres_kwargs, desc):
    nq, np_, hr, *_ = qe.compute_metrics_for_samples(
        states, torques, num_samples, mj_model, hnn, var_dq, var_dp, qpos_dim,
        desc=desc, hamres_kwargs=hamres_kwargs
    )
    return np.array(nq), np.array(np_), np.array(hr)


def rank_score(delta_nq, delta_np, delta_hr):
    # More negative is better. Put a bit more weight on NMSEs.
    return 0.4 * delta_nq + 0.4 * delta_np + 0.2 * delta_hr


def main():
    os.environ.setdefault('TQDM_DISABLE', '1')

    # Force GPU 4 by visible-device remap. If unavailable, fallback CPU.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    qe.SYSTEM = '2dof'
    cfg = qe.SYSTEM_CONFIGS['2dof']

    dpf, hnn, var_dq, var_dp, mj_model = qe.load_models(cfg, device)
    qpos_dim = cfg['qpos_dim']

    hamres_kwargs = dict(
        smooth_sigma=qe.HAMRES_SMOOTH_SIGMA if qe.HAMRES_SMOOTH_SIGMA_CFG is None else qe.HAMRES_SMOOTH_SIGMA_CFG,
        delta=qe.HAMRES_PSEUDO_HUBER_DELTA if qe.HAMRES_DELTA_CFG is None else qe.HAMRES_DELTA_CFG,
        min_scale_q=qe.HAMRES_MIN_SCALE_Q if qe.HAMRES_MIN_SCALE_Q_CFG is None else qe.HAMRES_MIN_SCALE_Q_CFG,
        min_scale_p=qe.HAMRES_MIN_SCALE_P if qe.HAMRES_MIN_SCALE_P_CFG is None else qe.HAMRES_MIN_SCALE_P_CFG,
    )

    # ---- Stage 1: coarse smart search on representative settings ----
    stage1_settings = [
        ('sinusoidal', 100),
        ('gp', 300),
    ]
    stage1_seeds = [65, 66]
    num_samples = 8

    coarse = []
    # one_step family (robust baseline)
    for steps in [30, 50, 80]:
        for after in [10, 15]:
            for aq in [5e-3, 1e-2]:
                coarse.append(dict(
                    mode='one_step', steps=steps, after=after, before=20,
                    alpha_q=aq, alpha_p=aq, trust=0.0, num_diff_steps=20,
                ))

    # robust_hamres family (focused around known good regions)
    for steps in [30, 50]:
        for after in [10, 15]:
            for aq in [5e-3, 1e-2]:
                for trust in [0.0, 1e-3]:
                    for ham_sigma in [0.5, 1.0]:
                        for ham_delta in [1.0, 2.0]:
                            coarse.append(dict(
                                mode='robust_hamres', steps=steps, after=after, before=20,
                                alpha_q=aq, alpha_p=aq, trust=trust,
                                ham_sigma=ham_sigma, ham_delta=ham_delta,
                                num_diff_steps=20,
                            ))

    print('stage1 configs:', len(coarse))

    ung_cache = {}
    stage1_rows = []

    for policy, L in stage1_settings:
        for seed in stage1_seeds:
            key = (policy, L, seed)
            qe.set_seed(seed)
            torques = qe.load_torques(policy, num_samples, L, device, cfg)
            noise = torch.randn(num_samples, L, dpf.state_dim, device=device)

            ung_kwargs = dict(hnn=None, guidance_steps=0, smooth_sigma=0, num_diffusion_steps=20)
            ung_states, ung_torques = qe.run_batch(dpf, torques, noise, L, qe.BATCH_SIZE_UNGUIDED, **ung_kwargs)
            ung_nq, ung_np, ung_hr = compute_metrics(
                ung_states, ung_torques, num_samples, mj_model, hnn, var_dq, var_dp, qpos_dim, hamres_kwargs,
                desc=f's1_ung_{policy}_L{L}_s{seed}'
            )
            ung_cache[key] = dict(torques=torques, noise=noise, ung=(ung_nq, ung_np, ung_hr))

            for c in coarse:
                gui_states, gui_torques = run_guided(dpf, hnn, torques, noise, L, c)
                g_nq, g_np, g_hr = compute_metrics(
                    gui_states, gui_torques, num_samples, mj_model, hnn, var_dq, var_dp, qpos_dim, hamres_kwargs,
                    desc=f's1_gui_{policy}_L{L}_s{seed}'
                )
                u_nq, u_np, u_hr = ung_cache[key]['ung']
                du = float(np.median(g_nq) - np.median(u_nq))
                dv = float(np.median(g_np) - np.median(u_np))
                dw = float(np.median(g_hr) - np.median(u_hr))
                wins = (int(du < 0), int(dv < 0), int(dw < 0))
                stage1_rows.append(dict(
                    config=cfg_name(c), mode=c['mode'], policy=policy, L=L, seed=seed,
                    d_nq=du, d_np=dv, d_hr=dw,
                    win_nq=wins[0], win_np=wins[1], win_hr=wins[2],
                ))

    # Aggregate stage1
    by_cfg = defaultdict(list)
    for r in stage1_rows:
        by_cfg[r['config']].append(r)

    stage1_rank = []
    for k, rs in by_cfg.items():
        dnq = np.array([x['d_nq'] for x in rs], dtype=float)
        dnp = np.array([x['d_np'] for x in rs], dtype=float)
        dhr = np.array([x['d_hr'] for x in rs], dtype=float)
        wnq = sum(int(x['win_nq']) for x in rs)
        wnp = sum(int(x['win_np']) for x in rs)
        whr = sum(int(x['win_hr']) for x in rs)
        wall = sum(int(x['win_nq'] and x['win_np'] and x['win_hr']) for x in rs)
        n = len(rs)
        score = rank_score(np.median(dnq), np.median(dnp), np.median(dhr))
        stage1_rank.append(dict(
            config=k, n=n, all_win=wall, win_nq=wnq, win_np=wnp, win_hr=whr,
            med_d_nq=float(np.median(dnq)), med_d_np=float(np.median(dnp)), med_d_hr=float(np.median(dhr)),
            score=score,
        ))

    stage1_rank.sort(key=lambda x: (-x['all_win'], -x['win_nq'], -x['win_np'], -x['win_hr'], x['score']))
    top = stage1_rank[:3]

    print('\nTop stage1 configs:')
    for t in top:
        print(t)

    # Parse top configs back to dicts
    def parse_config(s):
        d = {}
        for kv in s.split('|'):
            k, v = kv.split('=', 1)
            if k in {'steps', 'after', 'before', 'num_diff_steps'}:
                d[k] = int(float(v))
            elif k in {'alpha_q', 'alpha_p', 'trust', 'ham_sigma', 'ham_delta'}:
                d[k] = float(v)
            else:
                d[k] = v
        return d

    top_cfgs = [parse_config(t['config']) for t in top]

    # ---- Stage 2: full reliability on broad matrix ----
    stage2_settings = [(p, L) for p in ['sinusoidal', 'gp', 'zero', 'spline'] for L in [100, 300, 1000]]
    stage2_seeds = [65, 66, 67]
    stage2_rows = []

    for policy, L in stage2_settings:
        for seed in stage2_seeds:
            qe.set_seed(seed)
            torques = qe.load_torques(policy, num_samples, L, device, cfg)
            noise = torch.randn(num_samples, L, dpf.state_dim, device=device)

            ung_kwargs = dict(hnn=None, guidance_steps=0, smooth_sigma=0, num_diffusion_steps=20)
            ung_states, ung_torques = qe.run_batch(dpf, torques, noise, L, qe.BATCH_SIZE_UNGUIDED, **ung_kwargs)
            u_nq, u_np, u_hr = compute_metrics(
                ung_states, ung_torques, num_samples, mj_model, hnn, var_dq, var_dp, qpos_dim, hamres_kwargs,
                desc=f's2_ung_{policy}_L{L}_s{seed}'
            )
            u_med = (float(np.median(u_nq)), float(np.median(u_np)), float(np.median(u_hr)))

            for c in top_cfgs:
                gui_states, gui_torques = run_guided(dpf, hnn, torques, noise, L, c)
                g_nq, g_np, g_hr = compute_metrics(
                    gui_states, gui_torques, num_samples, mj_model, hnn, var_dq, var_dp, qpos_dim, hamres_kwargs,
                    desc=f's2_gui_{policy}_L{L}_s{seed}'
                )
                g_med = (float(np.median(g_nq)), float(np.median(g_np)), float(np.median(g_hr)))
                d = (g_med[0] - u_med[0], g_med[1] - u_med[1], g_med[2] - u_med[2])
                stage2_rows.append(dict(
                    config=cfg_name(c), policy=policy, L=L, seed=seed,
                    ung_nq=u_med[0], ung_np=u_med[1], ung_hr=u_med[2],
                    gui_nq=g_med[0], gui_np=g_med[1], gui_hr=g_med[2],
                    d_nq=d[0], d_np=d[1], d_hr=d[2],
                    win_nq=int(d[0] < 0), win_np=int(d[1] < 0), win_hr=int(d[2] < 0),
                ))

    # Aggregate stage2 globally and per setting
    by_cfg2 = defaultdict(list)
    for r in stage2_rows:
        by_cfg2[r['config']].append(r)

    global_rank = []
    for k, rs in by_cfg2.items():
        dnq = np.array([x['d_nq'] for x in rs], dtype=float)
        dnp = np.array([x['d_np'] for x in rs], dtype=float)
        dhr = np.array([x['d_hr'] for x in rs], dtype=float)
        n = len(rs)
        wnq = sum(x['win_nq'] for x in rs)
        wnp = sum(x['win_np'] for x in rs)
        whr = sum(x['win_hr'] for x in rs)
        wall = sum(int(x['win_nq'] and x['win_np'] and x['win_hr']) for x in rs)
        score = rank_score(np.median(dnq), np.median(dnp), np.median(dhr))
        global_rank.append(dict(
            config=k, n=n, all_win=wall, win_nq=wnq, win_np=wnp, win_hr=whr,
            med_d_nq=float(np.median(dnq)), med_d_np=float(np.median(dnp)), med_d_hr=float(np.median(dhr)),
            score=score,
        ))
    global_rank.sort(key=lambda x: (-x['all_win'], -x['win_nq'], -x['win_np'], -x['win_hr'], x['score']))

    out_dir = '/home/gsang/Projects/Perceiver_IO/output_ablation/sweep_outputs'
    os.makedirs(out_dir, exist_ok=True)

    stage1_csv = os.path.join(out_dir, 'smart_sweep_stage1_rank.csv')
    with open(stage1_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(stage1_rank[0].keys()))
        w.writeheader()
        w.writerows(stage1_rank)

    stage2_csv = os.path.join(out_dir, 'smart_sweep_stage2_rows.csv')
    with open(stage2_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(stage2_rows[0].keys()))
        w.writeheader()
        w.writerows(stage2_rows)

    global_csv = os.path.join(out_dir, 'smart_sweep_stage2_global_rank.csv')
    with open(global_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(global_rank[0].keys()))
        w.writeheader()
        w.writerows(global_rank)

    print('\n=== FINAL GLOBAL RANK ===')
    for r in global_rank:
        print(r)

    print('\nOutputs:')
    print(stage1_csv)
    print(stage2_csv)
    print(global_csv)


if __name__ == '__main__':
    main()
