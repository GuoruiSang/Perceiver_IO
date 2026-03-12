#!/usr/bin/env python3
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import mujoco

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

import scripts.guidance_eval_config as gc
import scripts.guidance_sampling_utils as gs
from scripts.system_eval_utils import (
    SYSTEM_CONFIGS,
    BATCH_SIZE_UNGUIDED,
    BATCH_SIZE_GUIDED,
    HAMRES_SMOOTH_SIGMA,
    HAMRES_PSEUDO_HUBER_DELTA,
    HAMRES_MIN_SCALE_Q,
    HAMRES_MIN_SCALE_P,
    load_torques,
    compute_rmse_for_samples,
)
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


def _env_int(name, default):
    raw = os.getenv(name, '').strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _resolve_path(raw_path):
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def resolve_guided_batch_size(system, length, guidance):
    """Choose guided batch size with optional env override.

    Large-N strategy1 (resampling) on 2DoF can OOM at long lengths.
    """
    override = os.getenv('SWEEP_GUIDED_BATCH_SIZE', '').strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass

    base = BATCH_SIZE_GUIDED
    if system != '2dof' or guidance.get('guidance_method') != 'strategy1':
        return base

    n = int(guidance.get('guidance_num_candidates') or 0)
    if n >= 24:
        bs = 1
    elif n >= 20:
        bs = 2
    elif n >= 16:
        bs = 3
    elif n >= 12:
        bs = 4
    else:
        bs = base

    # Additional safety for long trajectories where autograd memory spikes.
    if n >= 20 and length >= 650:
        bs = 1
    elif n >= 12 and length >= 900:
        bs = min(bs, 2)

    return max(1, min(base, bs))


def main():
    systems = _env_csv('SWEEP_SYSTEMS', ['2dof', '3dof'])
    policies = _env_csv('SWEEP_POLICIES', ['sinusoidal', 'gp', 'zero', 'spline'])
    lengths_csv = os.getenv('SWEEP_LENGTHS_CSV', '').strip()
    if lengths_csv:
        vals = [v.strip() for v in lengths_csv.split(',') if v.strip()]
        lengths = sorted({int(v) for v in vals})
    else:
        length_min = _env_int('SWEEP_LENGTH_MIN', 50)
        length_max = _env_int('SWEEP_LENGTH_MAX', 1200)
        length_step = max(1, _env_int('SWEEP_LENGTH_STEP', 50))
        lengths = list(range(length_min, length_max + 1, length_step))
    num_samples = _env_int('SWEEP_NUM_SAMPLES', 100)
    seed = _env_int('SWEEP_SEED', 10)
    output_subdir = os.getenv('SWEEP_OUTPUT_SUBDIR', 'eval_runs/default_full_sweep')
    hnn_ckpt_3dof = os.getenv('SWEEP_HNN_CKPT_3DOF', '').strip()
    if hnn_ckpt_3dof:
        hnn_ckpt_3dof = _resolve_path(hnn_ckpt_3dof)

    alpha_q_env = os.getenv('SWEEP_ALPHA_Q', '').strip()
    alpha_p_env = os.getenv('SWEEP_ALPHA_P', '').strip()
    if alpha_q_env:
        gc.ALPHA_Q = float(alpha_q_env)
    if alpha_p_env:
        gc.ALPHA_P = float(alpha_p_env)

    # Respect CUDA_VISIBLE_DEVICES remapping (e.g., visible GPU 4 becomes cuda:0).
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Sweep] device={device}")
    print(f"[Sweep] systems={systems}")
    print(f"[Sweep] lengths={lengths}")
    print(f"[Sweep] alpha_q={gc.ALPHA_Q} alpha_p={gc.ALPHA_P}")
    print(f"[Sweep] output_subdir={output_subdir}")
    if hnn_ckpt_3dof:
        print(f"[Sweep] override 3dof hnn_ckpt={hnn_ckpt_3dof}")

    output_root = project_root / 'output' / output_subdir
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
        'guidance_preset': gc.GUIDANCE_PRESET,
        'alpha_q': float(gc.ALPHA_Q),
        'alpha_p': float(gc.ALPHA_P),
        'num_diffusion_steps': gc.NUM_DIFFUSION_STEPS,
        'smooth_sigma': gc.SMOOTH_SIGMA,
        'smooth_guidance_only': gc.SMOOTH_GUIDANCE_ONLY,
        'smooth_last_step_only': gc.SMOOTH_LAST_STEP_ONLY,
        'hamres_eval': {
            **gc.build_hamres_eval_kwargs(),
        },
        'device': str(device),
        'output_subdir': output_subdir,
        'override_hnn_ckpt_3dof': str(hnn_ckpt_3dof) if hnn_ckpt_3dof else None,
        'strategy1_label': 'resampling',
        'schema_note': 'CSV columns keep gui_* for backward compatibility; strategy1 should be interpreted as resampling.',
    }
    manifest_json.write_text(json.dumps(manifest, indent=2))

    summary_fields = [
        'combo', 'system', 'policy', 'length', 'num_samples', 'seed',
        'ung_rmse_q_mean', 'ung_rmse_q_std', 'ung_rmse_q_p25', 'ung_rmse_q_median', 'ung_rmse_q_p75', 'ung_rmse_q_p99',
        'gui_rmse_q_mean', 'gui_rmse_q_std', 'gui_rmse_q_p25', 'gui_rmse_q_median', 'gui_rmse_q_p75', 'gui_rmse_q_p99',
        'd_rmse_q_mean', 'd_rmse_q_std', 'd_rmse_q_p25', 'd_rmse_q_median', 'd_rmse_q_p75', 'd_rmse_q_p99',
        'ung_rmse_p_mean', 'ung_rmse_p_std', 'ung_rmse_p_p25', 'ung_rmse_p_median', 'ung_rmse_p_p75', 'ung_rmse_p_p99',
        'gui_rmse_p_mean', 'gui_rmse_p_std', 'gui_rmse_p_p25', 'gui_rmse_p_median', 'gui_rmse_p_p75', 'gui_rmse_p_p99',
        'd_rmse_p_mean', 'd_rmse_p_std', 'd_rmse_p_p25', 'd_rmse_p_median', 'd_rmse_p_p75', 'd_rmse_p_p99',
        'ung_hamres_mean', 'ung_hamres_std', 'ung_hamres_p25', 'ung_hamres_median', 'ung_hamres_p75', 'ung_hamres_p99',
        'gui_hamres_mean', 'gui_hamres_std', 'gui_hamres_p25', 'gui_hamres_median', 'gui_hamres_p75', 'gui_hamres_p99',
        'd_hamres_mean', 'd_hamres_std', 'd_hamres_p25', 'd_hamres_median', 'd_hamres_p75', 'd_hamres_p99',
        'all3_win_count', 'win_rmse_q_count', 'win_rmse_p_count', 'win_hamres_count',
    ]

    per_sample_fields = [
        'combo', 'system', 'policy', 'length', 'num_samples', 'seed', 'sample_idx',
        'ung_rmse_q', 'gui_rmse_q', 'd_rmse_q',
        'ung_rmse_p', 'gui_rmse_p', 'd_rmse_p',
        'ung_hamres', 'gui_hamres', 'd_hamres',
    ]

    for system in systems:
        cfg = dict(SYSTEM_CONFIGS[system])
        if system == '3dof' and hnn_ckpt_3dof:
            cfg['hnn_ckpt'] = hnn_ckpt_3dof
        print(f"\n[Sweep] loading models for {system} ...")
        dpf, hnn, var_dq, var_dp, mj_model = gs.load_models(cfg, device)
        qpos_dim = cfg['qpos_dim']

        guidance, _ = gc.resolve_guidance_config(
            system=system,
            auto_map={'2dof': 'robust_r9_comboa', '3dof': 'one_step_best'},
        )
        comparison_label = 'resampling' if guidance['guidance_method'] == 'strategy1' else 'guided'
        print(
            "[Sweep] guidance="
            f"{guidance['guidance_method']} "
            f"num_candidates={guidance['guidance_num_candidates']} "
            f"ham_sigma={guidance['guidance_hamres_smooth_sigma']} "
            f"ham_delta={guidance['guidance_hamres_delta']}"
        )
        print(f"[Sweep] comparison_label={comparison_label} (gui_* columns kept for compatibility)")

        ung_kwargs = gc.build_unguided_kwargs()
        gui_kwargs = gc.build_guided_kwargs(hnn, guidance)
        hamres_kwargs = gc.build_hamres_eval_kwargs()

        for policy in policies:
            for length in lengths:
                cid = combo_id(system, policy, length, seed, num_samples)
                if cid in completed:
                    print(f"[Skip] {cid}")
                    continue

                print(f"\n[Run] {cid}")
                torques = load_torques(policy, num_samples, length, device, cfg)
                gs.set_seed(seed)
                noise = torch.randn(num_samples, length, dpf.state_dim, device=device)

                guided_batch_size = resolve_guided_batch_size(system, length, guidance)
                if guided_batch_size != BATCH_SIZE_GUIDED:
                    print(
                        f"[Sweep] adjusted guided batch size: {guided_batch_size} "
                        f"(system={system}, N={guidance.get('guidance_num_candidates')}, L={length})"
                    )

                ung_states, ung_torques = gs.run_batch(dpf, torques, noise, length, BATCH_SIZE_UNGUIDED, **ung_kwargs)
                gui_states, gui_torques = gs.run_batch(dpf, torques, noise, length, guided_batch_size, **gui_kwargs)

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
                    'comparison_label': comparison_label,
                    'unguided_states': ung_states.detach().cpu(),
                    'unguided_torques': ung_torques.detach().cpu(),
                    'guided_states': gui_states.detach().cpu(),
                    'guided_torques': gui_torques.detach().cpu(),
                    'resampled_states': gui_states.detach().cpu(),
                    'resampled_torques': gui_torques.detach().cpu(),
                }, traj_path)

                ung_rq, ung_rp, ung_hr = compute_rmse_for_samples(
                    ung_states, ung_torques, num_samples, mj_model, hnn, var_dq, var_dp, qpos_dim,
                    desc=f"ung_{cid}", hamres_kwargs=hamres_kwargs,
                )
                gui_rq, gui_rp, gui_hr = compute_rmse_for_samples(
                    gui_states, gui_torques, num_samples, mj_model, hnn, var_dq, var_dp, qpos_dim,
                    desc=f"gui_{cid}", hamres_kwargs=hamres_kwargs,
                )

                ung_rq = np.array(ung_rq, dtype=float)
                gui_rq = np.array(gui_rq, dtype=float)
                ung_rp = np.array(ung_rp, dtype=float)
                gui_rp = np.array(gui_rp, dtype=float)
                ung_hr = np.array(ung_hr, dtype=float)
                gui_hr = np.array(gui_hr, dtype=float)

                d_rq = gui_rq - ung_rq
                d_rp = gui_rp - ung_rp
                d_hr = gui_hr - ung_hr

                for i in range(num_samples):
                    append_csv(per_sample_csv, per_sample_fields, {
                        'combo': cid, 'system': system, 'policy': policy, 'length': length,
                        'num_samples': num_samples, 'seed': seed, 'sample_idx': i,
                        'ung_rmse_q': float(ung_rq[i]), 'gui_rmse_q': float(gui_rq[i]), 'd_rmse_q': float(d_rq[i]),
                        'ung_rmse_p': float(ung_rp[i]), 'gui_rmse_p': float(gui_rp[i]), 'd_rmse_p': float(d_rp[i]),
                        'ung_hamres': float(ung_hr[i]), 'gui_hamres': float(gui_hr[i]), 'd_hamres': float(d_hr[i]),
                    })

                s_ung_rq = metric_stats(ung_rq)
                s_gui_rq = metric_stats(gui_rq)
                s_d_rq = metric_stats(d_rq)
                s_ung_rp = metric_stats(ung_rp)
                s_gui_rp = metric_stats(gui_rp)
                s_d_rp = metric_stats(d_rp)
                s_ung_hr = metric_stats(ung_hr)
                s_gui_hr = metric_stats(gui_hr)
                s_d_hr = metric_stats(d_hr)

                summary_row = {
                    'combo': cid, 'system': system, 'policy': policy, 'length': length,
                    'num_samples': num_samples, 'seed': seed,
                    **{f'ung_rmse_q_{k}': v for k, v in s_ung_rq.items()},
                    **{f'gui_rmse_q_{k}': v for k, v in s_gui_rq.items()},
                    **{f'd_rmse_q_{k}': v for k, v in s_d_rq.items()},
                    **{f'ung_rmse_p_{k}': v for k, v in s_ung_rp.items()},
                    **{f'gui_rmse_p_{k}': v for k, v in s_gui_rp.items()},
                    **{f'd_rmse_p_{k}': v for k, v in s_d_rp.items()},
                    **{f'ung_hamres_{k}': v for k, v in s_ung_hr.items()},
                    **{f'gui_hamres_{k}': v for k, v in s_gui_hr.items()},
                    **{f'd_hamres_{k}': v for k, v in s_d_hr.items()},
                    'all3_win_count': int(np.sum((d_rq < 0) & (d_rp < 0) & (d_hr < 0))),
                    'win_rmse_q_count': int(np.sum(d_rq < 0)),
                    'win_rmse_p_count': int(np.sum(d_rp < 0)),
                    'win_hamres_count': int(np.sum(d_hr < 0)),
                }
                append_csv(summary_csv, summary_fields, summary_row)

                completed.add(cid)
                done_json.write_text(json.dumps(sorted(completed), indent=2))
                print(f"[Done] {cid}")

    print('\n[Sweep] Completed all combinations.')


if __name__ == '__main__':
    main()
