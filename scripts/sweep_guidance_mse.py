"""
Parameter sweep for HNN guidance — evaluated by MSE against MuJoCo reconstruction.
Reports MSE trajectory (qpos+mom) and MSE energy (H_gen vs H_recon).

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/sweep_guidance_mse.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import time
import h5py
import mujoco
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.models.trajectory_dpf import TrajectoryDPF, EMA
from src.models.HNN import HNNWrapper
from src.models.utils import reconstruct_traj_with_momentum

XML_PATH = str(project_root / 'configs' / 'rigid_arm_hinge.xml')


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


def compute_single_mse(args_tuple):
    """Compute MSE + full recon arrays for a single trajectory."""
    state_np, torque_np, qpos_dim, dt, data_dt = args_tuple

    qpos_gen = state_np[:, :qpos_dim]
    mom_gen = state_np[:, qpos_dim:]

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    data.qpos[:] = qpos_gen[0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    initial_qvel = np.linalg.solve(M, mom_gen[0])

    recon = reconstruct_traj_with_momentum(
        model, len(qpos_gen), dt,
        qpos_gen[0], initial_qvel, torque_np,
        data_dt=data_dt,
    )

    mse_qpos = np.mean((qpos_gen[1:] - recon['seq_qpos']) ** 2)
    mse_mom = np.mean((mom_gen[1:] - recon['seq_mom']) ** 2)

    qpos_recon = np.concatenate([qpos_gen[:1], recon['seq_qpos']], axis=0)
    mom_recon = np.concatenate([mom_gen[:1], recon['seq_mom']], axis=0)

    return mse_qpos, mse_mom, qpos_recon, mom_recon


def compute_hnn_energy_batch(hnn, qpos, mom, device, batch_size=10000):
    """Compute H(q,p) for trajectories. Returns [N, T] array."""
    N, T, _ = qpos.shape
    total = N * T
    q_flat = torch.tensor(qpos.reshape(total, -1), dtype=torch.float32, device=device)
    p_flat = torch.tensor(mom.reshape(total, -1), dtype=torch.float32, device=device)
    H_list = []
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            H_batch = hnn(p_flat[start:end], q_flat[start:end])
            H_list.append(H_batch.cpu().numpy().squeeze(-1))
    return np.concatenate(H_list).reshape(N, T)


def evaluate_config_mse(model, hnn, torque_tensor, traj_len, device, config, num_workers=4):
    """Generate trajectories, compute MSE against MuJoCo."""
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

    state_np = state.cpu().numpy()
    torque_np = torque_out.cpu().numpy()
    qpos_dim = model.qpos_dim
    dt = float(model.dt)
    data_dt = float(model.data_dt)

    # Parallel MuJoCo reconstruction
    task_args = [
        (state_np[i], torque_np[i], qpos_dim, dt, data_dt)
        for i in range(num_samples)
    ]

    mse_qpos_list, mse_mom_list = [], []
    recon_qpos_list, recon_mom_list = [], []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(compute_single_mse, a): i for i, a in enumerate(task_args)}
        results_by_idx = {}
        for future in as_completed(futures):
            idx = futures[future]
            results_by_idx[idx] = future.result()

        for i in range(num_samples):
            mse_q, mse_m, qr, mr = results_by_idx[i]
            mse_qpos_list.append(mse_q)
            mse_mom_list.append(mse_m)
            recon_qpos_list.append(qr)
            recon_mom_list.append(mr)

    mse_qpos_arr = np.array(mse_qpos_list)
    mse_mom_arr = np.array(mse_mom_list)

    gen_qpos = state_np[:, :, :qpos_dim]
    gen_mom = state_np[:, :, qpos_dim:]
    recon_qpos = np.stack(recon_qpos_list, axis=0)
    recon_mom = np.stack(recon_mom_list, axis=0)

    H_gen = compute_hnn_energy_batch(hnn, gen_qpos, gen_mom, device=device)
    H_recon = compute_hnn_energy_batch(hnn, recon_qpos, recon_mom, device=device)
    mse_energy_arr = np.mean((H_gen - H_recon) ** 2, axis=1)

    return {
        'mse_qpos_mean': np.mean(mse_qpos_arr),
        'mse_qpos_std': np.std(mse_qpos_arr),
        'mse_mom_mean': np.mean(mse_mom_arr),
        'mse_mom_std': np.std(mse_mom_arr),
        'mse_traj_mean': np.mean(mse_qpos_arr + mse_mom_arr),
        'mse_traj_std': np.std(mse_qpos_arr + mse_mom_arr),
        'mse_energy_mean': np.mean(mse_energy_arr),
        'mse_energy_std': np.std(mse_energy_arr),
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

    # ==========================================
    # Extensive parameter grid
    # ==========================================
    sweep_configs = []

    # --- (1) Baselines: no guidance, various diffusion_steps (ctx fixed at 0.2) ---
    for diff_steps in [25]:
        sweep_configs.append({
            'name': f'BASELINE_diff{diff_steps}_ctx0.2',
            'diffusion_steps': diff_steps, 'context_fraction': 0.2,
            'guidance_steps': 0, 'guidance_lr': 0.01,
            'guidance_after_steps': 0, 'lambda_init': 0.0,
        })

    # --- (2) Core sweep: guidance_steps x guidance_lr (fix diff50, ctx0.2, after=45) ---
    for g_steps in [1, 2, 5, 10, 25]:
        for lr in [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02]:
            sweep_configs.append({
                'name': f'diff50_ctx0.2_g{g_steps}_lr{lr}_after45',
                'diffusion_steps': 50, 'context_fraction': 0.2,
                'guidance_steps': g_steps, 'guidance_lr': lr,
                'guidance_after_steps': 45, 'lambda_init': 0.0,
            })

    # --- (3) guidance_after_steps sweep: when to start guidance ---
    # Use promising (g_steps, lr) combos from energy sweep
    for g_steps, lr in [(25, 0.01), (50, 0.01), (100, 0.01), (25, 0.005), (50, 0.005)]:
        for after in [10, 20, 30, 35, 40, 45, 50]:
            sweep_configs.append({
                'name': f'diff50_ctx0.2_g{g_steps}_lr{lr}_after{after}',
                'diffusion_steps': 50, 'context_fraction': 0.2,
                'guidance_steps': g_steps, 'guidance_lr': lr,
                'guidance_after_steps': after, 'lambda_init': 0.0,
            })

    # --- (4) Diffusion steps sweep (ctx fixed at 0.2, with best guidance combos) ---
    for diff_steps in [100, 200]:
        after = diff_steps - 5
        for g_steps, lr in [(25, 0.01), (50, 0.01), (100, 0.01), (50, 0.005), (100, 0.005)]:
            sweep_configs.append({
                'name': f'diff{diff_steps}_ctx0.2_g{g_steps}_lr{lr}_after{after}',
                'diffusion_steps': diff_steps, 'context_fraction': 0.2,
                'guidance_steps': g_steps, 'guidance_lr': lr,
                'guidance_after_steps': after, 'lambda_init': 0.0,
            })

    # Deduplicate by name
    seen = set()
    unique_configs = []
    for c in sweep_configs:
        if c['name'] not in seen:
            seen.add(c['name'])
            unique_configs.append(c)
    sweep_configs = unique_configs

    print(f"\n{'='*120}")
    print(f"Running {len(sweep_configs)} configurations (length={TRAJ_LEN}, samples={NUM_SAMPLES})")
    print(f"{'='*120}\n")

    results = []
    for i, config in enumerate(sweep_configs):
        torch.manual_seed(42)
        np.random.seed(42)

        t0 = time.time()
        print(f"[{i+1}/{len(sweep_configs)}] {config['name']}", flush=True)

        try:
            metrics = evaluate_config_mse(model, hnn, torque_tensor, TRAJ_LEN, device, config)
            elapsed = time.time() - t0

            row = {**config, **metrics, 'time': elapsed}
            results.append(row)
            print(f"  MSE_traj={metrics['mse_traj_mean']:.6f}±{metrics['mse_traj_std']:.6f}  "
                  f"MSE_energy={metrics['mse_energy_mean']:.4f}±{metrics['mse_energy_std']:.4f}  "
                  f"(qpos={metrics['mse_qpos_mean']:.6f}, mom={metrics['mse_mom_mean']:.6f})  "
                  f"({elapsed:.1f}s)\n", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED: {e} ({elapsed:.1f}s)\n", flush=True)

    # Print tables sorted by MSE trajectory
    print("\n" + "=" * 150)
    print("RESULTS SORTED BY MSE TRAJECTORY (qpos+mom)")
    print("=" * 150)
    results_traj = sorted(results, key=lambda r: r['mse_traj_mean'])
    print(f"{'Rank':<5} {'Config':<50} {'MSE_qpos':>14} {'MSE_mom':>14} {'MSE_traj':>14} {'MSE_energy':>14} {'time':>8}")
    print("-" * 150)
    for rank, r in enumerate(results_traj, 1):
        print(f"{rank:<5} {r['name']:<50} "
              f"{r['mse_qpos_mean']:>8.6f}±{r['mse_qpos_std']:<5.4f} "
              f"{r['mse_mom_mean']:>8.6f}±{r['mse_mom_std']:<5.4f} "
              f"{r['mse_traj_mean']:>8.6f}±{r['mse_traj_std']:<5.4f} "
              f"{r['mse_energy_mean']:>8.4f}±{r['mse_energy_std']:<5.4f} "
              f"{r['time']:>7.1f}s")

    # Print tables sorted by MSE energy
    print("\n" + "=" * 150)
    print("RESULTS SORTED BY MSE ENERGY")
    print("=" * 150)
    results_energy = sorted(results, key=lambda r: r['mse_energy_mean'])
    print(f"{'Rank':<5} {'Config':<50} {'MSE_qpos':>14} {'MSE_mom':>14} {'MSE_traj':>14} {'MSE_energy':>14} {'time':>8}")
    print("-" * 150)
    for rank, r in enumerate(results_energy, 1):
        print(f"{rank:<5} {r['name']:<50} "
              f"{r['mse_qpos_mean']:>8.6f}±{r['mse_qpos_std']:<5.4f} "
              f"{r['mse_mom_mean']:>8.6f}±{r['mse_mom_std']:<5.4f} "
              f"{r['mse_traj_mean']:>8.6f}±{r['mse_traj_std']:<5.4f} "
              f"{r['mse_energy_mean']:>8.4f}±{r['mse_energy_std']:<5.4f} "
              f"{r['time']:>7.1f}s")

    print(f"\nBest for MSE_traj: {results_traj[0]['name']} = {results_traj[0]['mse_traj_mean']:.6f}")
    print(f"Best for MSE_energy: {results_energy[0]['name']} = {results_energy[0]['mse_energy_mean']:.4f}")


if __name__ == "__main__":
    main()
