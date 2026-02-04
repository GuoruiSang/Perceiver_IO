"""
Hyperparameter search for HNN guidance parameters.

Searches over:
- guidance_lr: learning rate for Adam optimizer
- guidance_steps: number of optimization steps per diffusion step
- guidance_after_steps: diffusion step to start applying guidance

Uses 50 mixed torque samples (from sinusoidal, gp, zero, spline).
Evaluates using HamRes metric.

Usage:
    python scripts/search_guidance_hyperparams.py --device cuda:3
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
from itertools import product
import pandas as pd

from src.models.utils import reconstruct_traj_with_momentum, EMA
from src.models.HNN import HNNWrapper
from src.models.trajectory_dpf import TrajectoryDPF


# Configuration
DT = 0.0001
DATA_DT = 0.0002
TRAJECTORY_LENGTH = 500  # Fixed length for evaluation
NUM_DIFFUSION_STEPS = 50
CONTEXT_FRACTION = 0.2
QPOS_DIM = 3
MOM_DIM = 3

# Paths
MUJOCO_XML = project_root / 'configs' / 'rigid_arm_hinge.xml'
HNN_CHECKPOINT = project_root / 'checkpoints' / 'SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt'
DPF_CHECKPOINT = project_root / 'checkpoints' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt'

# Torque data paths
TORQUE_PATHS = {
    'sinusoidal': project_root / 'data' / 'sinusoidal_torques_1000_L1500.h5',
    'gp': project_root / 'data' / 'gp_torques_1000_L1500.h5',
    'spline': project_root / 'data' / 'training_torques_1000_L1500.h5',
}

# Hyperparameter search space
GUIDANCE_LRS = [0.0005, 0.001, 0.005, 0.01, 0.02]
GUIDANCE_STEPS_LIST = [5, 10, 15, 25, 50]
GUIDANCE_AFTER_STEPS_LIST = [30, 35, 40, 45, 48]  # Out of 50 diffusion steps


def load_model(device):
    """Load DPF model with EMA + HNN."""
    print(f"Loading DPF model...")
    model = TrajectoryDPF.load_from_checkpoint(str(DPF_CHECKPOINT), map_location=device, strict=False)
    model = model.to(device)
    model.eval()

    checkpoint = torch.load(str(DPF_CHECKPOINT), map_location=device)
    ema_shadow = checkpoint.get('ema_shadow', None)
    ema_decay = checkpoint.get('ema_decay', 0.9995)
    if ema_shadow is not None:
        model.ema = EMA(model.model, decay=ema_decay)
        for name, tensor in ema_shadow.items():
            if name in model.ema.shadow:
                model.ema.shadow[name] = tensor.to(device=device, dtype=model.ema.shadow[name].dtype)
        model._ema_loaded = True
        print("  EMA shadow loaded")

    print(f"Loading HNN...")
    hnn = HNNWrapper.load_from_checkpoint(str(HNN_CHECKPOINT), map_location=device)
    hnn = hnn.to(device)
    hnn.eval()

    return model, hnn


def load_mixed_torques(device, num_samples=50):
    """Load mixed torques from different policies."""
    samples_per_policy = num_samples // 4  # 12 each, plus 2 extra for sinusoidal
    extra = num_samples - samples_per_policy * 4

    all_torques = []

    # Sinusoidal
    with h5py.File(TORQUE_PATHS['sinusoidal'], 'r') as f:
        n = samples_per_policy + extra
        torques = torch.tensor(f['torques'][:n], dtype=torch.float32, device=device)
        all_torques.append(torques)
        print(f"  Sinusoidal: {n} samples")

    # GP
    with h5py.File(TORQUE_PATHS['gp'], 'r') as f:
        torques = torch.tensor(f['torques'][:samples_per_policy], dtype=torch.float32, device=device)
        all_torques.append(torques)
        print(f"  GP: {samples_per_policy} samples")

    # Zero
    zero_torques = torch.zeros(samples_per_policy, 1500, 3, device=device)
    all_torques.append(zero_torques)
    print(f"  Zero: {samples_per_policy} samples")

    # Spline
    with h5py.File(TORQUE_PATHS['spline'], 'r') as f:
        torques = torch.tensor(f['torques'][:samples_per_policy], dtype=torch.float32, device=device)
        all_torques.append(torques)
        print(f"  Spline: {samples_per_policy} samples")

    mixed = torch.cat(all_torques, dim=0)
    print(f"Total mixed torques: {mixed.shape[0]}")
    return mixed


def compute_hamres(qpos: np.ndarray, mom: np.ndarray, torque: np.ndarray,
                   hnn: HNNWrapper, var_dq: float, var_dp: float,
                   dt: float = DATA_DT) -> float:
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

    hamres = mse_q / (var_dq + eps) + mse_p / (var_dp + eps)
    return hamres


def compute_mse(generated: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute MSE between generated and reconstructed trajectories."""
    min_len = min(len(generated) - 1, len(reconstructed))
    gen = generated[1:min_len+1]
    rec = reconstructed[:min_len]
    return np.mean((gen - rec) ** 2)


def evaluate_config(model, hnn, torques, mj_model, device,
                    guidance_lr, guidance_steps, guidance_after_steps,
                    var_dq, var_dp, seed=228):
    """Evaluate a single guidance configuration."""
    num_samples = torques.shape[0]
    traj_torques = torques[:, :TRAJECTORY_LENGTH, :]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    guidance_kwargs = dict(
        hnn=hnn,
        guidance_method='adam',
        guidance_steps=guidance_steps,
        guidance_lr=guidance_lr,
        guidance_after_steps=guidance_after_steps,
        lambda_init=0.0,
    )

    # Generate trajectories
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

    # Compute metrics for each sample
    hamres_list = []
    mse_list = []

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

        # Compute MSE
        mse_qpos = compute_mse(qpos_gen, recon['seq_qpos'])
        mse_mom = compute_mse(mom_gen, recon['seq_mom'])
        mse_list.append(mse_qpos + mse_mom)

        # Compute HamRes
        hamres = compute_hamres(qpos_gen, mom_gen, torque, hnn, var_dq, var_dp)
        hamres_list.append(hamres)

    return {
        'hamres_mean': np.nanmean(hamres_list),
        'hamres_median': np.nanmedian(hamres_list),
        'hamres_std': np.nanstd(hamres_list),
        'mse_mean': np.mean(mse_list),
        'mse_std': np.std(mse_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--output', type=str, default='guidance_hyperparam_search.csv')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load models
    model, hnn = load_model(device)
    var_dq = hnn.qvel_var.mean().item()
    var_dp = hnn.mom_dot_var.mean().item()
    print(f"HNN variance: var_dq={var_dq:.6f}, var_dp={var_dp:.6f}")

    # Load MuJoCo model
    print("Loading MuJoCo model...")
    mj_model = mujoco.MjModel.from_xml_path(str(MUJOCO_XML))

    # Load mixed torques
    print("Loading mixed torques...")
    torques = load_mixed_torques(device, args.num_samples)

    # First, evaluate baseline (no guidance)
    print("\n" + "="*60)
    print("Evaluating baseline (no guidance)...")
    print("="*60)

    torch.manual_seed(228)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(228)

    traj_torques = torques[:, :TRAJECTORY_LENGTH, :]
    states_baseline, torques_baseline = model.sample_trajectories(
        num_samples=args.num_samples,
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

    states_baseline = states_baseline.cpu().numpy()
    torques_baseline = torques_baseline.cpu().numpy()

    baseline_hamres = []
    baseline_mse = []
    for i in range(args.num_samples):
        state = states_baseline[i]
        torque = torques_baseline[i]
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

        mse_qpos = compute_mse(qpos_gen, recon['seq_qpos'])
        mse_mom = compute_mse(mom_gen, recon['seq_mom'])
        baseline_mse.append(mse_qpos + mse_mom)

        hamres = compute_hamres(qpos_gen, mom_gen, torque, hnn, var_dq, var_dp)
        baseline_hamres.append(hamres)

    print(f"Baseline - HamRes: mean={np.nanmean(baseline_hamres):.4f}, median={np.nanmedian(baseline_hamres):.4f}")
    print(f"Baseline - MSE: mean={np.mean(baseline_mse):.6f}, std={np.std(baseline_mse):.6f}")

    # Search over hyperparameters
    print("\n" + "="*60)
    print("Starting hyperparameter search...")
    print("="*60)

    configs = list(product(GUIDANCE_LRS, GUIDANCE_STEPS_LIST, GUIDANCE_AFTER_STEPS_LIST))
    total_configs = len(configs)
    print(f"Total configurations to evaluate: {total_configs}")

    results = []
    best_hamres = float('inf')
    best_config = None

    for idx, (lr, steps, after_steps) in enumerate(configs):
        print(f"\n[{idx+1}/{total_configs}] lr={lr}, steps={steps}, after_steps={after_steps}")

        try:
            metrics = evaluate_config(
                model, hnn, torques, mj_model, device,
                guidance_lr=lr,
                guidance_steps=steps,
                guidance_after_steps=after_steps,
                var_dq=var_dq,
                var_dp=var_dp,
            )

            result = {
                'guidance_lr': lr,
                'guidance_steps': steps,
                'guidance_after_steps': after_steps,
                **metrics,
            }
            results.append(result)

            print(f"  HamRes: mean={metrics['hamres_mean']:.4f}, median={metrics['hamres_median']:.4f}")
            print(f"  MSE: mean={metrics['mse_mean']:.6f}")

            if metrics['hamres_median'] < best_hamres:
                best_hamres = metrics['hamres_median']
                best_config = result
                print(f"  *** NEW BEST ***")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save results
    df = pd.DataFrame(results)
    output_path = project_root / 'output_ablation' / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nBaseline (no guidance):")
    print(f"  HamRes: mean={np.nanmean(baseline_hamres):.4f}, median={np.nanmedian(baseline_hamres):.4f}")
    print(f"  MSE: mean={np.mean(baseline_mse):.6f}")

    print(f"\nBest configuration:")
    print(f"  guidance_lr: {best_config['guidance_lr']}")
    print(f"  guidance_steps: {best_config['guidance_steps']}")
    print(f"  guidance_after_steps: {best_config['guidance_after_steps']}")
    print(f"  HamRes: mean={best_config['hamres_mean']:.4f}, median={best_config['hamres_median']:.4f}")
    print(f"  MSE: mean={best_config['mse_mean']:.6f}")

    # Print top 5 configs
    print("\nTop 5 configurations (by median HamRes):")
    df_sorted = df.sort_values('hamres_median')
    for i, row in df_sorted.head(5).iterrows():
        print(f"  lr={row['guidance_lr']}, steps={row['guidance_steps']}, "
              f"after={row['guidance_after_steps']} -> HamRes={row['hamres_median']:.4f}")


if __name__ == '__main__':
    main()
