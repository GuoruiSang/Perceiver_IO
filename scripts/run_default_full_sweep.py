#!/usr/bin/env python3
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import mujoco

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

import scripts.quick_eval as qe
from scripts.compute_ablation_2dof_with_smoothing import (
    SYSTEM_CONFIGS,
    BATCH_SIZE_UNGUIDED,
    BATCH_SIZE_GUIDED,
    HAMRES_SMOOTH_SIGMA,
    HAMRES_PSEUDO_HUBER_DELTA,
    HAMRES_MIN_SCALE_Q,
    HAMRES_MIN_SCALE_P,
    load_torques,
    compute_metrics_for_samples,
)


def resolve_default_guidance(system, cfg):
    preset_key = qe.GUIDANCE_PRESET.lower()
    if preset_key == 'auto':
        preset_key = 'robust_r9_comboa' if system == '2dof' else 'one_step_best'

    if preset_key == 'custom':
        return {
            'guidance_method': qe.GUIDANCE_METHOD,
            'optimize_target': qe.OPTIMIZE_TARGET,
            'guidance_energy_mode': qe.GUIDANCE_ENERGY_MODE,
            'guidance_steps': qe.GUIDANCE_STEPS if qe.GUIDANCE_STEPS is not None else cfg['guidance_steps'],
            'guidance_lr': qe.GUIDANCE_LR if qe.GUIDANCE_LR is not None else cfg['guidance_lr'],
            'guidance_after_steps': qe.GUIDANCE_AFTER if qe.GUIDANCE_AFTER is not None else cfg['guidance_after_steps'],
            'guidance_before_steps': qe.GUIDANCE_BEFORE,
            'guidance_trust_lambda': qe.GUIDANCE_TRUST_LAMBDA,
            'guidance_hamres_smooth_sigma': qe.GUIDANCE_HAMRES_SMOOTH_SIGMA,
            'guidance_hamres_delta': qe.GUIDANCE_HAMRES_DELTA,
            'guidance_hamres_min_scale_q': qe.GUIDANCE_HAMRES_MIN_SCALE_Q,
            'guidance_hamres_min_scale_p': qe.GUIDANCE_HAMRES_MIN_SCALE_P,
        }

    p = qe.GUIDANCE_PRESETS[preset_key]
    return {
        'guidance_method': p['guidance_method'],
        'optimize_target': p['optimize_target'],
        'guidance_energy_mode': p['guidance_energy_mode'],
        'guidance_steps': p['guidance_steps'],
        'guidance_lr': p['guidance_lr'],
        'guidance_after_steps': p['guidance_after'],
        'guidance_before_steps': p['guidance_before'],
        'guidance_trust_lambda': p['guidance_trust_lambda'],
        'guidance_hamres_smooth_sigma': p['guidance_hamres_smooth_sigma'],
        'guidance_hamres_delta': p['guidance_hamres_delta'],
        'guidance_hamres_min_scale_q': p['guidance_hamres_min_scale_q'],
        'guidance_hamres_min_scale_p': p['guidance_hamres_min_scale_p'],
    }


def metric_stats(arr):
    a = np.asarray(arr, dtype=float)
    return {
        'mean': float(np.mean(a)),
        'std': float(np.std(a)),
        'p25': float(np.percentile(a, 25)),
        'median': float(np.median(a)),
        'p75': float(np.percentile(a, 75)),
        'p99': float(np.percentile(a, 99)),
    }


def append_csv(path, fieldnames, row):
    new_file = not path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def combo_id(system, policy, length, seed, num_samples):
    return f"{system}_{policy}_L{length}_N{num_samples}_seed{seed}"


def _env_csv(name, default_values):
    raw = os.getenv(name, '').strip()
    if not raw:
        return list(default_values)
    vals = [v.strip() for v in raw.split(',') if v.strip()]
    return vals if vals else list(default_values)


def _resolve_path(raw_path):
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def main():
    systems = _env_csv('SWEEP_SYSTEMS', ['2dof', '3dof'])
    policies = ['sinusoidal', 'gp', 'zero', 'spline']
    lengths = list(range(50, 1201, 50))
    num_samples = 100
    seed = 10
    output_subdir = os.getenv('SWEEP_OUTPUT_SUBDIR', 'default_full_sweep')
    hnn_ckpt_3dof = os.getenv('SWEEP_HNN_CKPT_3DOF', '').strip()
    if hnn_ckpt_3dof:
        hnn_ckpt_3dof = _resolve_path(hnn_ckpt_3dof)

    # Respect CUDA_VISIBLE_DEVICES remapping (e.g., visible GPU 4 becomes cuda:0).
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    qe.DEVICE = str(device)
    print(f"[Sweep] device={device}")
    print(f"[Sweep] systems={systems}")
    print(f"[Sweep] output_subdir={output_subdir}")
    if hnn_ckpt_3dof:
        print(f"[Sweep] override 3dof hnn_ckpt={hnn_ckpt_3dof}")

    output_root = project_root / 'output_ablation' / output_subdir
    traj_root = output_root / 'trajectories'
    metrics_root = output_root / 'metrics'
    traj_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    summary_csv = metrics_root / 'metrics_summary.csv'
    per_sample_csv = metrics_root / 'metrics_per_sample.csv'
    done_json = output_root / 'completed_combos.json'
    manifest_json = output_root / 'run_manifest.json'

    completed = set()
    if done_json.exists():
        completed = set(json.loads(done_json.read_text()))

    manifest = {
        'systems': systems,
        'policies': policies,
        'lengths': lengths,
        'num_samples': num_samples,
        'seed': seed,
        'guidance_preset': qe.GUIDANCE_PRESET,
        'num_diffusion_steps': qe.NUM_DIFFUSION_STEPS,
        'smooth_sigma': qe.SMOOTH_SIGMA,
        'smooth_guidance_only': qe.SMOOTH_GUIDANCE_ONLY,
        'smooth_last_step_only': qe.SMOOTH_LAST_STEP_ONLY,
        'hamres_eval': {
            'smooth_sigma': HAMRES_SMOOTH_SIGMA if qe.HAMRES_SMOOTH_SIGMA_CFG is None else qe.HAMRES_SMOOTH_SIGMA_CFG,
            'delta': HAMRES_PSEUDO_HUBER_DELTA if qe.HAMRES_DELTA_CFG is None else qe.HAMRES_DELTA_CFG,
            'min_scale_q': HAMRES_MIN_SCALE_Q if qe.HAMRES_MIN_SCALE_Q_CFG is None else qe.HAMRES_MIN_SCALE_Q_CFG,
            'min_scale_p': HAMRES_MIN_SCALE_P if qe.HAMRES_MIN_SCALE_P_CFG is None else qe.HAMRES_MIN_SCALE_P_CFG,
        },
        'device': str(device),
        'output_subdir': output_subdir,
        'override_hnn_ckpt_3dof': str(hnn_ckpt_3dof) if hnn_ckpt_3dof else None,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2))

    summary_fields = [
        'combo', 'system', 'policy', 'length', 'num_samples', 'seed',
        'ung_nmse_q_mean', 'ung_nmse_q_std', 'ung_nmse_q_p25', 'ung_nmse_q_median', 'ung_nmse_q_p75', 'ung_nmse_q_p99',
        'gui_nmse_q_mean', 'gui_nmse_q_std', 'gui_nmse_q_p25', 'gui_nmse_q_median', 'gui_nmse_q_p75', 'gui_nmse_q_p99',
        'd_nmse_q_mean', 'd_nmse_q_std', 'd_nmse_q_p25', 'd_nmse_q_median', 'd_nmse_q_p75', 'd_nmse_q_p99',
        'ung_nmse_p_mean', 'ung_nmse_p_std', 'ung_nmse_p_p25', 'ung_nmse_p_median', 'ung_nmse_p_p75', 'ung_nmse_p_p99',
        'gui_nmse_p_mean', 'gui_nmse_p_std', 'gui_nmse_p_p25', 'gui_nmse_p_median', 'gui_nmse_p_p75', 'gui_nmse_p_p99',
        'd_nmse_p_mean', 'd_nmse_p_std', 'd_nmse_p_p25', 'd_nmse_p_median', 'd_nmse_p_p75', 'd_nmse_p_p99',
        'ung_hamres_mean', 'ung_hamres_std', 'ung_hamres_p25', 'ung_hamres_median', 'ung_hamres_p75', 'ung_hamres_p99',
        'gui_hamres_mean', 'gui_hamres_std', 'gui_hamres_p25', 'gui_hamres_median', 'gui_hamres_p75', 'gui_hamres_p99',
        'd_hamres_mean', 'd_hamres_std', 'd_hamres_p25', 'd_hamres_median', 'd_hamres_p75', 'd_hamres_p99',
        'all3_win_count', 'win_nmse_q_count', 'win_nmse_p_count', 'win_hamres_count',
    ]

    per_sample_fields = [
        'combo', 'system', 'policy', 'length', 'num_samples', 'seed', 'sample_idx',
        'ung_nmse_q', 'gui_nmse_q', 'd_nmse_q',
        'ung_nmse_p', 'gui_nmse_p', 'd_nmse_p',
        'ung_hamres', 'gui_hamres', 'd_hamres',
    ]

    for system in systems:
        cfg = dict(SYSTEM_CONFIGS[system])
        if system == '3dof' and hnn_ckpt_3dof:
            cfg['hnn_ckpt'] = hnn_ckpt_3dof
        print(f"\n[Sweep] loading models for {system} ...")
        dpf, hnn, var_dq, var_dp, mj_model = qe.load_models(cfg, device)
        qpos_dim = cfg['qpos_dim']

        guidance = resolve_default_guidance(system, cfg)

        shared_kwargs = {}
        if qe.NUM_DIFFUSION_STEPS is not None:
            shared_kwargs['num_diffusion_steps'] = qe.NUM_DIFFUSION_STEPS

        ung_kwargs = dict(hnn=None, guidance_steps=0, smooth_sigma=qe.SMOOTH_SIGMA, **shared_kwargs)
        gui_kwargs = dict(
            hnn=hnn,
            guidance_method=guidance['guidance_method'],
            guidance_steps=guidance['guidance_steps'],
            guidance_lr=guidance['guidance_lr'],
            guidance_after_steps=guidance['guidance_after_steps'],
            guidance_before_steps=guidance['guidance_before_steps'],
            optimize_target=guidance['optimize_target'],
            guidance_energy_mode=guidance['guidance_energy_mode'],
            guidance_hamres_smooth_sigma=guidance['guidance_hamres_smooth_sigma'],
            guidance_hamres_delta=guidance['guidance_hamres_delta'],
            guidance_hamres_min_scale_q=guidance['guidance_hamres_min_scale_q'],
            guidance_hamres_min_scale_p=guidance['guidance_hamres_min_scale_p'],
            guidance_trust_lambda=guidance['guidance_trust_lambda'],
            smooth_sigma=qe.SMOOTH_SIGMA,
            smooth_guidance_only=qe.SMOOTH_GUIDANCE_ONLY,
            smooth_last_step_only=qe.SMOOTH_LAST_STEP_ONLY,
            alpha_q=qe.ALPHA_Q,
            alpha_p=qe.ALPHA_P,
            langevin_step_size=qe.LANGEVIN_STEP_SIZE,
            langevin_noise_scale=qe.LANGEVIN_NOISE_SCALE,
            chunk_length=qe.CHUNK_LENGTH,
            **shared_kwargs,
        )

        hamres_kwargs = {
            'smooth_sigma': HAMRES_SMOOTH_SIGMA if qe.HAMRES_SMOOTH_SIGMA_CFG is None else qe.HAMRES_SMOOTH_SIGMA_CFG,
            'delta': HAMRES_PSEUDO_HUBER_DELTA if qe.HAMRES_DELTA_CFG is None else qe.HAMRES_DELTA_CFG,
            'min_scale_q': HAMRES_MIN_SCALE_Q if qe.HAMRES_MIN_SCALE_Q_CFG is None else qe.HAMRES_MIN_SCALE_Q_CFG,
            'min_scale_p': HAMRES_MIN_SCALE_P if qe.HAMRES_MIN_SCALE_P_CFG is None else qe.HAMRES_MIN_SCALE_P_CFG,
        }

        for policy in policies:
            for length in lengths:
                cid = combo_id(system, policy, length, seed, num_samples)
                if cid in completed:
                    print(f"[Skip] {cid}")
                    continue

                print(f"\n[Run] {cid}")
                torques = load_torques(policy, num_samples, length, device, cfg)
                qe.set_seed(seed)
                noise = torch.randn(num_samples, length, dpf.state_dim, device=device)

                ung_states, ung_torques = qe.run_batch(dpf, torques, noise, length, BATCH_SIZE_UNGUIDED, **ung_kwargs)
                gui_states, gui_torques = qe.run_batch(dpf, torques, noise, length, BATCH_SIZE_GUIDED, **gui_kwargs)

                # Save trajectories first (resumable artifact)
                traj_dir = traj_root / system / policy
                traj_dir.mkdir(parents=True, exist_ok=True)
                traj_path = traj_dir / f"{cid}.pt"
                torch.save({
                    'combo': cid,
                    'system': system,
                    'policy': policy,
                    'length': length,
                    'num_samples': num_samples,
                    'seed': seed,
                    'unguided_states': ung_states.detach().cpu(),
                    'unguided_torques': ung_torques.detach().cpu(),
                    'guided_states': gui_states.detach().cpu(),
                    'guided_torques': gui_torques.detach().cpu(),
                }, traj_path)

                ung_nq, ung_np, ung_hr, _, _ = compute_metrics_for_samples(
                    ung_states, ung_torques, num_samples, mj_model, hnn, var_dq, var_dp, qpos_dim,
                    desc=f"ung_{cid}", hamres_kwargs=hamres_kwargs,
                )
                gui_nq, gui_np, gui_hr, _, _ = compute_metrics_for_samples(
                    gui_states, gui_torques, num_samples, mj_model, hnn, var_dq, var_dp, qpos_dim,
                    desc=f"gui_{cid}", hamres_kwargs=hamres_kwargs,
                )

                ung_nq = np.array(ung_nq, dtype=float)
                gui_nq = np.array(gui_nq, dtype=float)
                ung_np = np.array(ung_np, dtype=float)
                gui_np = np.array(gui_np, dtype=float)
                ung_hr = np.array(ung_hr, dtype=float)
                gui_hr = np.array(gui_hr, dtype=float)

                d_nq = gui_nq - ung_nq
                d_np = gui_np - ung_np
                d_hr = gui_hr - ung_hr

                for i in range(num_samples):
                    append_csv(per_sample_csv, per_sample_fields, {
                        'combo': cid, 'system': system, 'policy': policy, 'length': length,
                        'num_samples': num_samples, 'seed': seed, 'sample_idx': i,
                        'ung_nmse_q': float(ung_nq[i]), 'gui_nmse_q': float(gui_nq[i]), 'd_nmse_q': float(d_nq[i]),
                        'ung_nmse_p': float(ung_np[i]), 'gui_nmse_p': float(gui_np[i]), 'd_nmse_p': float(d_np[i]),
                        'ung_hamres': float(ung_hr[i]), 'gui_hamres': float(gui_hr[i]), 'd_hamres': float(d_hr[i]),
                    })

                s_ung_nq = metric_stats(ung_nq)
                s_gui_nq = metric_stats(gui_nq)
                s_d_nq = metric_stats(d_nq)
                s_ung_np = metric_stats(ung_np)
                s_gui_np = metric_stats(gui_np)
                s_d_np = metric_stats(d_np)
                s_ung_hr = metric_stats(ung_hr)
                s_gui_hr = metric_stats(gui_hr)
                s_d_hr = metric_stats(d_hr)

                summary_row = {
                    'combo': cid, 'system': system, 'policy': policy, 'length': length,
                    'num_samples': num_samples, 'seed': seed,
                    **{f'ung_nmse_q_{k}': v for k, v in s_ung_nq.items()},
                    **{f'gui_nmse_q_{k}': v for k, v in s_gui_nq.items()},
                    **{f'd_nmse_q_{k}': v for k, v in s_d_nq.items()},
                    **{f'ung_nmse_p_{k}': v for k, v in s_ung_np.items()},
                    **{f'gui_nmse_p_{k}': v for k, v in s_gui_np.items()},
                    **{f'd_nmse_p_{k}': v for k, v in s_d_np.items()},
                    **{f'ung_hamres_{k}': v for k, v in s_ung_hr.items()},
                    **{f'gui_hamres_{k}': v for k, v in s_gui_hr.items()},
                    **{f'd_hamres_{k}': v for k, v in s_d_hr.items()},
                    'all3_win_count': int(np.sum((d_nq < 0) & (d_np < 0) & (d_hr < 0))),
                    'win_nmse_q_count': int(np.sum(d_nq < 0)),
                    'win_nmse_p_count': int(np.sum(d_np < 0)),
                    'win_hamres_count': int(np.sum(d_hr < 0)),
                }
                append_csv(summary_csv, summary_fields, summary_row)

                completed.add(cid)
                done_json.write_text(json.dumps(sorted(completed), indent=2))
                print(f"[Done] {cid}")

    print('\n[Sweep] Completed all combinations.')


if __name__ == '__main__':
    main()
