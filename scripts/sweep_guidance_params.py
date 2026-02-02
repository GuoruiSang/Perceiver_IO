"""
Quick parameter sweep for HNN guidance settings.
Tests combinations of sampling + guidance parameters, reports final energy (e1+e2).

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/sweep_guidance_params.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import itertools
import numpy as np
import torch
import time
import h5py

from src.models.trajectory_dpf import TrajectoryDPF, EMA
from src.models.HNN import HNNWrapper
from src.models.utils import compute_hnn_physics_energy


def load_sinusoidal_torques(val_path, num_samples):
    torques = []
    with h5py.File(val_path, 'r') as f:
        for i in range(f.attrs['num_trajectories']):
            grp = f[f'traj_{i}']
            policy = grp.attrs.get('torque_policy', b'sinusoidal')
            if isinstance(policy, bytes):
                policy = policy.decode()
            if policy == 'sinusoidal':
                torques.append(grp['seq_torque'][:])
                if len(torques) >= num_samples:
                    break
    return np.stack(torques, axis=0)


def evaluate_config(model, hnn, torque_tensor, traj_len, device, config):
    """Generate trajectories with given config, return mean final energy (e1+e2)."""
    num_samples = torque_tensor.shape[0]
    trunc = torque_tensor[:, :traj_len, :]

    state, torque_out = model.sample_trajectories(
        num_samples=num_samples,
        trajectory_length=traj_len,
        num_diffusion_steps=config['diffusion_steps'],
        context_fraction=config['context_fraction'],
        use_ema=True,
        sampler='ddim',
        guidance_scale=1.0,
        hnn=hnn if config['guidance_steps'] > 0 else None,
        guidance_method='adam',
        guidance_after_steps=config['guidance_after_steps'],
        guidance_steps=config['guidance_steps'],
        guidance_lr=config['guidance_lr'],
        lambda_init=config['lambda_init'],
        torque=trunc,
    )

    # Compute final energy (needs grad for autograd inside compute_hnn_physics_energy)
    qpos = state[:, :, :model.qpos_dim].clone().requires_grad_(True)
    mom = state[:, :, model.qpos_dim:].clone().requires_grad_(True)

    energy, e1, e2, e3 = compute_hnn_physics_energy(
        qpos, mom, trunc, hnn, model.data_dt, config['lambda_init'],
        return_components=True,
    )

    return {
        'energy': energy.item(),
        'e1': e1.item(),
        'e2': e2.item(),
        'e3': e3.item(),
    }


def main():
    device = torch.device('cuda:0')

    # Load DPF
    ckpt_dir = project_root / 'checkpoints'
    dpf_ckpts = sorted(
        ckpt_dir.glob('trajectory_dpf_StateOnlyAdaLN*AbsoluteTimeEncoding&VariableTrajLength*.ckpt'),
        key=lambda p: p.stat().st_mtime
    )
    dpf_path = str(dpf_ckpts[-1])
    print(f"DPF: {dpf_path}")

    model = TrajectoryDPF.load_from_checkpoint(dpf_path, map_location=device).to(device)
    checkpoint = torch.load(dpf_path, map_location=device, weights_only=False)
    ema_shadow = checkpoint.get('ema_shadow', None)
    if ema_shadow:
        model.ema = EMA(model.model, decay=checkpoint.get('ema_decay', 0.9995))
        for name, tensor in ema_shadow.items():
            if name in model.ema.shadow:
                model.ema.shadow[name] = tensor.to(device=device, dtype=model.ema.shadow[name].dtype)
    del checkpoint

    # Load HNN
    hnn_ckpts = sorted(ckpt_dir.glob('SeperableHNN*.ckpt'), key=lambda p: p.stat().st_mtime)
    hnn_path = str(hnn_ckpts[-1])
    print(f"HNN: {hnn_path}")
    hnn = HNNWrapper.load_from_checkpoint(hnn_path, map_location=device).to(device)
    hnn.eval()

    # Load torques
    val_path = str(project_root / 'data' / 'traj_2000-steps_4000.h5')
    NUM_SAMPLES = 20
    TRAJ_LEN = 1000
    all_torques = load_sinusoidal_torques(val_path, NUM_SAMPLES)
    torque_tensor = torch.tensor(all_torques, dtype=torch.float32, device=device)
    print(f"Torques: {all_torques.shape}, using {NUM_SAMPLES} samples, length {TRAJ_LEN}")

    # Fix seed
    torch.manual_seed(42)
    np.random.seed(42)

    # ==========================================
    # Parameter grid
    # ==========================================
    sweep_configs = []

    # Baseline: no guidance
    sweep_configs.append({
        'name': 'no_guidance_50steps_ctx0.2',
        'diffusion_steps': 50, 'context_fraction': 0.2,
        'guidance_steps': 0, 'guidance_lr': 0.01,
        'guidance_after_steps': 0, 'lambda_init': 0.0,
    })
    sweep_configs.append({
        'name': 'no_guidance_100steps_ctx0.5',
        'diffusion_steps': 100, 'context_fraction': 0.5,
        'guidance_steps': 0, 'guidance_lr': 0.01,
        'guidance_after_steps': 0, 'lambda_init': 0.0,
    })

    # Sweep diffusion_steps x context_fraction (with fixed guidance)
    for diff_steps, ctx in [(50, 0.2), (100, 0.2), (100, 0.5), (200, 0.2)]:
        sweep_configs.append({
            'name': f'diff{diff_steps}_ctx{ctx}_g25_lr0.01_after{diff_steps-5}',
            'diffusion_steps': diff_steps, 'context_fraction': ctx,
            'guidance_steps': 25, 'guidance_lr': 0.01,
            'guidance_after_steps': diff_steps - 5, 'lambda_init': 0.0,
        })

    # Sweep guidance_steps
    for g_steps in [10, 25, 50, 100]:
        sweep_configs.append({
            'name': f'diff50_ctx0.2_g{g_steps}_lr0.01_after45',
            'diffusion_steps': 50, 'context_fraction': 0.2,
            'guidance_steps': g_steps, 'guidance_lr': 0.01,
            'guidance_after_steps': 45, 'lambda_init': 0.0,
        })

    # Sweep guidance_lr
    for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
        sweep_configs.append({
            'name': f'diff50_ctx0.2_g25_lr{lr}_after45',
            'diffusion_steps': 50, 'context_fraction': 0.2,
            'guidance_steps': 25, 'guidance_lr': lr,
            'guidance_after_steps': 45, 'lambda_init': 0.0,
        })

    # Sweep guidance_after_steps (when to start guidance)
    for after in [30, 35, 40, 45, 48]:
        sweep_configs.append({
            'name': f'diff50_ctx0.2_g25_lr0.01_after{after}',
            'diffusion_steps': 50, 'context_fraction': 0.2,
            'guidance_steps': 25, 'guidance_lr': 0.01,
            'guidance_after_steps': after, 'lambda_init': 0.0,
        })

    # Sweep lambda_init
    for lam in [0.0, 0.01, 0.1, 1.0]:
        sweep_configs.append({
            'name': f'diff50_ctx0.2_g25_lr0.01_after45_lam{lam}',
            'diffusion_steps': 50, 'context_fraction': 0.2,
            'guidance_steps': 25, 'guidance_lr': 0.01,
            'guidance_after_steps': 45, 'lambda_init': lam,
        })

    # Deduplicate by name
    seen = set()
    unique_configs = []
    for c in sweep_configs:
        if c['name'] not in seen:
            seen.add(c['name'])
            unique_configs.append(c)
    sweep_configs = unique_configs

    print(f"\n{'='*100}")
    print(f"Running {len(sweep_configs)} configurations")
    print(f"{'='*100}\n")

    results = []
    for i, config in enumerate(sweep_configs):
        torch.manual_seed(42)
        np.random.seed(42)

        t0 = time.time()
        print(f"[{i+1}/{len(sweep_configs)}] {config['name']}")

        metrics = evaluate_config(model, hnn, torque_tensor, TRAJ_LEN, device, config)
        elapsed = time.time() - t0

        row = {**config, **metrics, 'time': elapsed}
        results.append(row)
        print(f"  e1={metrics['e1']:.2f}  e2={metrics['e2']:.2f}  "
              f"e1+e2={metrics['e1']+metrics['e2']:.2f}  total={metrics['energy']:.2f}  "
              f"({elapsed:.1f}s)\n")

    # Sort by e1+e2
    results.sort(key=lambda r: r['e1'] + r['e2'])

    print("\n" + "=" * 120)
    print("RESULTS (sorted by e1+e2, ascending)")
    print("=" * 120)
    print(f"{'Rank':<5} {'Config':<55} {'e1':>10} {'e2':>10} {'e1+e2':>10} {'e3':>10} {'total':>10} {'time':>8}")
    print("-" * 120)
    for rank, r in enumerate(results, 1):
        print(f"{rank:<5} {r['name']:<55} {r['e1']:>10.2f} {r['e2']:>10.2f} "
              f"{r['e1']+r['e2']:>10.2f} {r['e3']:>10.4f} {r['energy']:>10.2f} {r['time']:>7.1f}s")

    print(f"\nBest: {results[0]['name']}")
    print(f"  e1={results[0]['e1']:.2f}, e2={results[0]['e2']:.2f}, e1+e2={results[0]['e1']+results[0]['e2']:.2f}")


if __name__ == "__main__":
    main()
