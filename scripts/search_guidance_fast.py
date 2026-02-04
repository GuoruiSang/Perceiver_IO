"""
Fast Guidance Hyperparameter Search using Successive Halving.

Uses multi-GPU parallel execution to quickly find optimal guidance parameters.

Usage:
    # Round 1: Test all 27 configs with 5 samples each
    python scripts/search_guidance_fast.py --round 1 --gpus 0,1,2,3

    # Round 2: Test top 9 configs with 15 samples each
    python scripts/search_guidance_fast.py --round 2 --gpus 0,1,2,3

    # Round 3: Test top 3 configs with 50 samples each
    python scripts/search_guidance_fast.py --round 3 --gpus 0,1,2
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import h5py
import mujoco
import torch
import torch.multiprocessing as mp
from itertools import product
import pandas as pd
import json
import time

from src.models.utils import reconstruct_traj_with_momentum, EMA
from src.models.HNN import HNNWrapper
from src.models.trajectory_dpf import TrajectoryDPF


# Configuration
DT = 0.0001
DATA_DT = 0.0002
TRAJECTORY_LENGTH = 500
NUM_DIFFUSION_STEPS = 50
CONTEXT_FRACTION = 0.2
QPOS_DIM = 3
MOM_DIM = 3

# Paths
MUJOCO_XML = project_root / 'configs' / 'rigid_arm_hinge.xml'
HNN_CHECKPOINT = project_root / 'checkpoints' / 'SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt'
DPF_CHECKPOINT = project_root / 'checkpoints' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt'
OUTPUT_DIR = project_root / 'output_ablation'

# Torque data paths
TORQUE_PATHS = {
    'sinusoidal': project_root / 'data' / 'sinusoidal_torques_1000_L1500.h5',
    'gp': project_root / 'data' / 'gp_torques_1000_L1500.h5',
    'spline': project_root / 'data' / 'training_torques_1000_L1500.h5',
}

# Search space (3x3x3 = 27 configs)
ALL_CONFIGS = list(product(
    [0.001, 0.005, 0.01],      # guidance_lr
    [10, 25, 50],              # guidance_steps
    [35, 40, 45],              # guidance_after_steps
))

# Round settings
ROUND_SETTINGS = {
    1: {'num_samples': 5, 'top_k': 9},
    2: {'num_samples': 15, 'top_k': 3},
    3: {'num_samples': 50, 'top_k': 1},
}


def load_mixed_torques(num_samples=50):
    """Load mixed torques from different policies (CPU tensors)."""
    samples_per_policy = num_samples // 4
    extra = num_samples - samples_per_policy * 4

    all_torques = []

    # Sinusoidal
    with h5py.File(TORQUE_PATHS['sinusoidal'], 'r') as f:
        n = samples_per_policy + extra
        torques = torch.tensor(f['torques'][:n], dtype=torch.float32)
        all_torques.append(torques)

    # GP
    with h5py.File(TORQUE_PATHS['gp'], 'r') as f:
        torques = torch.tensor(f['torques'][:samples_per_policy], dtype=torch.float32)
        all_torques.append(torques)

    # Zero
    zero_torques = torch.zeros(samples_per_policy, 1500, 3)
    all_torques.append(zero_torques)

    # Spline
    with h5py.File(TORQUE_PATHS['spline'], 'r') as f:
        torques = torch.tensor(f['torques'][:samples_per_policy], dtype=torch.float32)
        all_torques.append(torques)

    return torch.cat(all_torques, dim=0)


def compute_hamres(qpos, mom, torque, hnn, var_dq, var_dp, dt=DATA_DT):
    """Compute HamRes for a trajectory."""
    T = len(qpos)
    if T < 3:
        return float('nan')

    eps = 1e-8
    device = next(hnn.parameters()).device

    qdot = (qpos[2:] - qpos[:-2]) / (2 * dt)
    pdot = (mom[2:] - mom[:-2]) / (2 * dt)

    q_mid = qpos[1:-1]
    p_mid = mom[1:-1]
    tau_mid = torque[1:-1]

    q_t = torch.tensor(q_mid, dtype=torch.float32, device=device).requires_grad_(True)
    p_t = torch.tensor(p_mid, dtype=torch.float32, device=device).requires_grad_(True)

    with torch.enable_grad():
        H = hnn(p_t, q_t)
        dH_dp, dH_dq = torch.autograd.grad(H.sum(), (p_t, q_t))

    dH_dp = dH_dp.detach().cpu().numpy()
    dH_dq = dH_dq.detach().cpu().numpy()

    r_q = qdot - dH_dp
    r_p = pdot - (-dH_dq + tau_mid)

    mse_q = np.mean(r_q ** 2)
    mse_p = np.mean(r_p ** 2)

    return mse_q / (var_dq + eps) + mse_p / (var_dp + eps)


def compute_nmse(generated, reconstructed):
    """Compute NMSE (Normalized MSE) between generated and reconstructed trajectories."""
    min_len = min(len(generated) - 1, len(reconstructed))
    gen = generated[1:min_len+1]
    rec = reconstructed[:min_len]
    mse = np.mean((gen - rec) ** 2)
    var = np.var(gen) + 1e-8
    return mse / var


def evaluate_single_config(args):
    """Worker function to evaluate a single config on a specific GPU."""
    config, gpu_id, torques_cpu, num_samples, seed = args

    device = torch.device(f'cuda:{gpu_id}')
    lr, steps, after_steps = config

    try:
        # Load models on this GPU
        model = TrajectoryDPF.load_from_checkpoint(str(DPF_CHECKPOINT), map_location=device, strict=False)
        model = model.to(device)
        model.eval()

        checkpoint = torch.load(str(DPF_CHECKPOINT), map_location=device, weights_only=False)
        ema_shadow = checkpoint.get('ema_shadow', None)
        if ema_shadow is not None:
            model.ema = EMA(model.model, decay=0.9995)
            for name, tensor in ema_shadow.items():
                if name in model.ema.shadow:
                    model.ema.shadow[name] = tensor.to(device=device, dtype=model.ema.shadow[name].dtype)
            model._ema_loaded = True

        hnn = HNNWrapper.load_from_checkpoint(str(HNN_CHECKPOINT), map_location=device)
        hnn = hnn.to(device)
        hnn.eval()

        var_dq = hnn.qvel_var.mean().item()
        var_dp = hnn.mom_dot_var.mean().item()

        mj_model = mujoco.MjModel.from_xml_path(str(MUJOCO_XML))

        # Move torques to GPU
        torques = torques_cpu[:num_samples].to(device)
        traj_torques = torques[:, :TRAJECTORY_LENGTH, :]

        # Set seed for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Generate trajectories with guidance
        guidance_kwargs = dict(
            hnn=hnn,
            guidance_method='adam',
            guidance_steps=steps,
            guidance_lr=lr,
            guidance_after_steps=after_steps,
            lambda_init=0.0,
        )

        states, torques_out = model.sample_trajectories(
            num_samples=num_samples,
            trajectory_length=TRAJECTORY_LENGTH,
            num_diffusion_steps=NUM_DIFFUSION_STEPS,
            context_fraction=CONTEXT_FRACTION,
            use_ema=False,
            sampler='ddim',
            guidance_scale=1.0,
            torque=traj_torques,
            **guidance_kwargs,
        )

        states = states.cpu().numpy()
        torques_out = torques_out.cpu().numpy()

        # Compute metrics
        hamres_list = []
        nmse_list = []

        for i in range(num_samples):
            state = states[i]
            torque = torques_out[i]
            qpos_gen = state[:, :QPOS_DIM]
            mom_gen = state[:, QPOS_DIM:]

            # MuJoCo reconstruction
            M = np.zeros((mj_model.nv, mj_model.nv))
            data = mujoco.MjData(mj_model)
            data.qpos[:] = qpos_gen[0]
            data.qvel[:] = 0
            mujoco.mj_forward(mj_model, data)
            mujoco.mj_fullM(mj_model, M, data.qM)
            initial_qvel = np.linalg.solve(M, mom_gen[0])

            recon = reconstruct_traj_with_momentum(
                mj_model, TRAJECTORY_LENGTH, DT, qpos_gen[0], initial_qvel, torque, data_dt=DATA_DT
            )

            nmse_qpos = compute_nmse(qpos_gen, recon['seq_qpos'])
            nmse_mom = compute_nmse(mom_gen, recon['seq_mom'])
            nmse_list.append(nmse_qpos + nmse_mom)

            hamres = compute_hamres(qpos_gen, mom_gen, torque, hnn, var_dq, var_dp)
            hamres_list.append(hamres)

        result = {
            'guidance_lr': lr,
            'guidance_steps': steps,
            'guidance_after_steps': after_steps,
            'hamres_mean': np.nanmean(hamres_list),
            'hamres_median': np.nanmedian(hamres_list),
            'hamres_std': np.nanstd(hamres_list),
            'nmse_mean': np.mean(nmse_list),
            'nmse_std': np.std(nmse_list),
            'num_samples': num_samples,
        }

        # Clean up
        del model, hnn
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        return {
            'guidance_lr': lr,
            'guidance_steps': steps,
            'guidance_after_steps': after_steps,
            'error': str(e),
            'hamres_median': float('inf'),
        }


def evaluate_baseline(gpu_id, torques_cpu, num_samples, seed):
    """Evaluate baseline (no guidance)."""
    device = torch.device(f'cuda:{gpu_id}')

    model = TrajectoryDPF.load_from_checkpoint(str(DPF_CHECKPOINT), map_location=device, strict=False)
    model = model.to(device)
    model.eval()

    checkpoint = torch.load(str(DPF_CHECKPOINT), map_location=device, weights_only=False)
    ema_shadow = checkpoint.get('ema_shadow', None)
    if ema_shadow is not None:
        model.ema = EMA(model.model, decay=0.9995)
        for name, tensor in ema_shadow.items():
            if name in model.ema.shadow:
                model.ema.shadow[name] = tensor.to(device=device, dtype=model.ema.shadow[name].dtype)
        model._ema_loaded = True

    hnn = HNNWrapper.load_from_checkpoint(str(HNN_CHECKPOINT), map_location=device)
    hnn = hnn.to(device)
    hnn.eval()

    var_dq = hnn.qvel_var.mean().item()
    var_dp = hnn.mom_dot_var.mean().item()

    mj_model = mujoco.MjModel.from_xml_path(str(MUJOCO_XML))

    torques = torques_cpu[:num_samples].to(device)
    traj_torques = torques[:, :TRAJECTORY_LENGTH, :]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    states, torques_out = model.sample_trajectories(
        num_samples=num_samples,
        trajectory_length=TRAJECTORY_LENGTH,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        context_fraction=CONTEXT_FRACTION,
        use_ema=False,
        sampler='ddim',
        guidance_scale=1.0,
        torque=traj_torques,
        hnn=None,
        guidance_steps=0,
    )

    states = states.cpu().numpy()
    torques_out = torques_out.cpu().numpy()

    hamres_list = []
    nmse_list = []

    for i in range(num_samples):
        state = states[i]
        torque = torques_out[i]
        qpos_gen = state[:, :QPOS_DIM]
        mom_gen = state[:, QPOS_DIM:]

        M = np.zeros((mj_model.nv, mj_model.nv))
        data = mujoco.MjData(mj_model)
        data.qpos[:] = qpos_gen[0]
        data.qvel[:] = 0
        mujoco.mj_forward(mj_model, data)
        mujoco.mj_fullM(mj_model, M, data.qM)
        initial_qvel = np.linalg.solve(M, mom_gen[0])

        recon = reconstruct_traj_with_momentum(
            mj_model, TRAJECTORY_LENGTH, DT, qpos_gen[0], initial_qvel, torque, data_dt=DATA_DT
        )

        nmse_qpos = compute_nmse(qpos_gen, recon['seq_qpos'])
        nmse_mom = compute_nmse(mom_gen, recon['seq_mom'])
        nmse_list.append(nmse_qpos + nmse_mom)

        hamres = compute_hamres(qpos_gen, mom_gen, torque, hnn, var_dq, var_dp)
        hamres_list.append(hamres)

    del model, hnn
    torch.cuda.empty_cache()

    return {
        'hamres_mean': np.nanmean(hamres_list),
        'hamres_median': np.nanmedian(hamres_list),
        'hamres_std': np.nanstd(hamres_list),
        'nmse_mean': np.mean(nmse_list),
        'nmse_std': np.std(nmse_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='Comma-separated GPU IDs')
    parser.add_argument('--seed', type=int, default=228)
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    settings = ROUND_SETTINGS[args.round]
    num_samples = settings['num_samples']
    top_k = settings['top_k']

    print(f"{'='*60}")
    print(f"  ROUND {args.round}: {num_samples} samples per config")
    print(f"  GPUs: {gpu_ids}")
    print(f"{'='*60}")

    # Load torques (CPU)
    print("\nLoading mixed torques...")
    torques_cpu = load_mixed_torques(num_samples=50)
    print(f"  Loaded {torques_cpu.shape[0]} torques")

    # Get configs for this round
    if args.round == 1:
        configs = ALL_CONFIGS
    else:
        # Load previous round results
        prev_csv = OUTPUT_DIR / f'guidance_search_r{args.round - 1}.csv'
        print(f"\nLoading previous results from: {prev_csv}")
        prev_df = pd.read_csv(prev_csv)
        prev_df = prev_df.sort_values('hamres_median')
        prev_top_k = ROUND_SETTINGS[args.round - 1]['top_k']
        top_configs = prev_df.head(prev_top_k)
        configs = [
            (row['guidance_lr'], int(row['guidance_steps']), int(row['guidance_after_steps']))
            for _, row in top_configs.iterrows()
        ]
        print(f"  Top {prev_top_k} configs from Round {args.round - 1}:")
        for c in configs:
            print(f"    lr={c[0]}, steps={c[1]}, after={c[2]}")

    print(f"\nEvaluating {len(configs)} configurations...")

    # Evaluate baseline first (only in round 1)
    if args.round == 1:
        print("\nEvaluating baseline (no guidance)...")
        baseline = evaluate_baseline(gpu_ids[0], torques_cpu, num_samples, args.seed)
        print(f"  Baseline HamRes: mean={baseline['hamres_mean']:.4f}, median={baseline['hamres_median']:.4f}")
        print(f"  Baseline NNMSE: mean={baseline['nmse_mean']:.6f}")

    # Prepare work items
    work_items = []
    for i, config in enumerate(configs):
        gpu_id = gpu_ids[i % num_gpus]
        work_items.append((config, gpu_id, torques_cpu, num_samples, args.seed))

    # Sequential evaluation (simpler than multiprocessing for CUDA)
    results = []
    start_time = time.time()

    for i, item in enumerate(work_items):
        config = item[0]
        print(f"\n[{i+1}/{len(configs)}] lr={config[0]}, steps={config[1]}, after={config[2]}")

        result = evaluate_single_config(item)

        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  HamRes: mean={result['hamres_mean']:.4f}, median={result['hamres_median']:.4f}")
            print(f"  NMSE: mean={result['nmse_mean']:.6f}")

        results.append(result)

        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (len(configs) - i - 1)
        print(f"  Time: {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

    # Save results
    df = pd.DataFrame(results)
    df = df.sort_values('hamres_median')
    output_csv = OUTPUT_DIR / f'guidance_search_r{args.round}.csv'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  ROUND {args.round} SUMMARY")
    print(f"{'='*60}")

    if args.round == 1:
        print(f"\nBaseline (no guidance): HamRes median = {baseline['hamres_median']:.4f}")

    print(f"\nTop {top_k} configurations:")
    for i, row in df.head(top_k).iterrows():
        print(f"  lr={row['guidance_lr']}, steps={int(row['guidance_steps'])}, "
              f"after={int(row['guidance_after_steps'])} -> HamRes median={row['hamres_median']:.4f}")

    # Save best config if final round
    if args.round == 3:
        best = df.iloc[0]
        best_config = {
            'guidance_lr': float(best['guidance_lr']),
            'guidance_steps': int(best['guidance_steps']),
            'guidance_after_steps': int(best['guidance_after_steps']),
            'hamres_median': float(best['hamres_median']),
            'hamres_mean': float(best['hamres_mean']),
            'nmse_mean': float(best['nmse_mean']),
        }
        best_json = OUTPUT_DIR / 'guidance_best.json'
        with open(best_json, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"\nBest config saved to: {best_json}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
