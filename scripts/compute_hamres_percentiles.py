"""
Compute HamRes percentiles (P25, Median, P95, P99) for all torque policies.

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/compute_hamres_percentiles.py --policy sinusoidal
    CUDA_VISIBLE_DEVICES=4 python scripts/compute_hamres_percentiles.py --policy gp
    CUDA_VISIBLE_DEVICES=4 python scripts/compute_hamres_percentiles.py --policy zero
    CUDA_VISIBLE_DEVICES=4 python scripts/compute_hamres_percentiles.py --policy spline
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import pandas as pd
import torch
import h5py
from tqdm import tqdm

from src.models.trajectory_dpf import TrajectoryDPF
from src.models.HNN import HNNWrapper


# Configuration
DT = 0.0002  # Timestep matching training data
TRAJECTORY_LENGTHS = list(range(50, 1001, 50))  # 50, 100, ..., 1000
NUM_SAMPLES = 1000  # Samples per length

# Paths
CHECKPOINT_PATH = project_root / 'checkpoints' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt'
HNN_CHECKPOINT = project_root / 'checkpoints' / 'SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt'
OUTPUT_DIR = project_root / 'output_ablation' / 'results' / 'original'

# Torque data paths
TORQUE_PATHS = {
    'sinusoidal': project_root / 'data' / 'sinusoidal_torques_1000_L1500.h5',
    'gp': project_root / 'data' / 'gp_torques_1000_L1500.h5',
    'zero': None,  # Zero torque - no file needed
    'spline': project_root / 'data' / 'traj_80000-steps_4000.h5',  # Training data has spline torques
}

# Guidance parameters
GUIDANCE_STEPS = 25
GUIDANCE_AFTER_STEPS = 45
GUIDANCE_LR = 0.01


def load_torques(policy: str, num_samples: int, max_length: int, device: torch.device):
    """Load torque sequences for a given policy."""
    if policy == 'zero':
        # Zero torque
        return torch.zeros(num_samples, max_length, 3, device=device)

    path = TORQUE_PATHS[policy]
    with h5py.File(path, 'r') as f:
        if policy == 'spline':
            # Training data format - individual trajectory groups
            torques_list = []
            for i in range(num_samples):
                tau = f[f'traj_{i}']['seq_torque'][:max_length]
                torques_list.append(tau)
            torques = np.stack(torques_list, axis=0)
        else:
            # sinusoidal/gp format
            torques = f['torques'][:num_samples, :max_length]

    return torch.tensor(torques, dtype=torch.float32, device=device)


def compute_hamres_for_trajectory(
    qpos: torch.Tensor,  # [T, dim]
    mom: torch.Tensor,   # [T, dim]
    torque: torch.Tensor,  # [T, dim]
    hnn: HNNWrapper,
    var_dq: torch.Tensor,
    var_dp: torch.Tensor,
    dt: float = DT,
) -> float:
    """
    Compute HamRes for a single trajectory.

    HamRes = mean(|r_q|²)/Var_dq + mean(|r_p|²)/Var_dp
    where:
        r_q = qdot - dH/dp
        r_p = pdot - (-dH/dq + tau)
    """
    T = qpos.shape[0]
    if T < 3:
        return float('nan')

    device = qpos.device
    eps = 1e-8

    # Finite difference for velocities (central difference, skip boundaries)
    # qdot[t] ≈ (q[t+1] - q[t-1]) / (2*dt)
    qdot = (qpos[2:] - qpos[:-2]) / (2 * dt)  # [T-2, dim]
    pdot = (mom[2:] - mom[:-2]) / (2 * dt)    # [T-2, dim]

    # Corresponding positions/momenta/torques (skip boundaries)
    q_mid = qpos[1:-1]      # [T-2, dim]
    p_mid = mom[1:-1]       # [T-2, dim]
    tau_mid = torque[1:-1]  # [T-2, dim]

    # Compute HNN gradients
    with torch.inference_mode(False):
        with torch.enable_grad():
            p_grad = p_mid.detach().clone().requires_grad_(True)
            q_grad = q_mid.detach().clone().requires_grad_(True)

            H = hnn(p_grad, q_grad)

            dH_dp, dH_dq = torch.autograd.grad(
                H.sum(), (p_grad, q_grad), create_graph=False
            )

    # Compute residuals
    r_q = qdot - dH_dp  # [T-2, dim]
    r_p = pdot - (-dH_dq + tau_mid)  # [T-2, dim]

    # Compute normalized MSE
    mse_q = (r_q ** 2).mean()
    mse_p = (r_p ** 2).mean()

    hamres = (mse_q / (var_dq + eps) + mse_p / (var_dp + eps)).item()

    return hamres


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True,
                        choices=['sinusoidal', 'gp', 'zero', 'spline'])
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Computing HamRes for policy: {args.policy}")

    # Load models
    print("Loading DPF model...")
    dpf = TrajectoryDPF.load_from_checkpoint(
        CHECKPOINT_PATH, map_location=device, strict=False
    )
    dpf.eval()
    dpf.to(device)

    print("Loading HNN model...")
    hnn = HNNWrapper.load_from_checkpoint(HNN_CHECKPOINT, map_location=device)
    hnn.eval()
    hnn.to(device)

    # Get variance from HNN (used for normalization)
    var_dq = hnn.qvel_var.mean().to(device)
    var_dp = hnn.mom_dot_var.mean().to(device)
    print(f"Normalization variances: var_dq={var_dq.item():.4f}, var_dp={var_dp.item():.4f}")

    # Load torques
    max_length = max(TRAJECTORY_LENGTHS)
    print(f"Loading torques for policy '{args.policy}'...")
    torques = load_torques(args.policy, NUM_SAMPLES, max_length, device)

    # Results storage
    results = []

    for L in TRAJECTORY_LENGTHS:
        print(f"\n=== Length {L} ===")

        hamres_unguided = []
        hamres_guided = []

        # Process in batches
        batch_size = 50
        num_batches = (NUM_SAMPLES + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc=f"L={L}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, NUM_SAMPLES)
            batch_torques = torques[start_idx:end_idx, :L]
            actual_batch_size = end_idx - start_idx

            # Sample unguided trajectories
            with torch.no_grad():
                state_unguided, _ = dpf.sample_trajectories(
                    num_samples=actual_batch_size,
                    trajectory_length=L,
                    context_fraction=0.5,
                    use_ema=True,
                    hnn=None,  # No guidance
                    guidance_steps=0,
                    torque=batch_torques,  # Pass torques
                )

            # Sample guided trajectories (no torch.no_grad - HNN guidance needs gradients)
            state_guided, _ = dpf.sample_trajectories(
                num_samples=actual_batch_size,
                trajectory_length=L,
                context_fraction=0.5,
                use_ema=True,
                hnn=hnn,
                guidance_method="adam",
                guidance_steps=GUIDANCE_STEPS,
                guidance_after_steps=GUIDANCE_AFTER_STEPS,
                guidance_lr=GUIDANCE_LR,
                torque=batch_torques,  # Pass torques
            )

            # Compute HamRes for each trajectory
            for i in range(actual_batch_size):
                # Extract qpos and mom from trajectories
                # State format: [B, T, state_dim] where state_dim = 2*dim (qpos + mom)
                dim = state_unguided.shape[-1] // 2

                qpos_ung = state_unguided[i, :, :dim]
                mom_ung = state_unguided[i, :, dim:]
                tau = batch_torques[i]

                qpos_gui = state_guided[i, :, :dim]
                mom_gui = state_guided[i, :, dim:]

                hr_ung = compute_hamres_for_trajectory(qpos_ung, mom_ung, tau, hnn, var_dq, var_dp)
                hr_gui = compute_hamres_for_trajectory(qpos_gui, mom_gui, tau, hnn, var_dq, var_dp)

                if not np.isnan(hr_ung):
                    hamres_unguided.append(hr_ung)
                if not np.isnan(hr_gui):
                    hamres_guided.append(hr_gui)

        # Compute percentiles
        if hamres_unguided and hamres_guided:
            hr_ung = np.array(hamres_unguided)
            hr_gui = np.array(hamres_guided)

            results.append({
                'trajectory_length': L,
                'unguided_p25': np.percentile(hr_ung, 25),
                'unguided_median': np.percentile(hr_ung, 50),
                'unguided_p95': np.percentile(hr_ung, 95),
                'unguided_p99': np.percentile(hr_ung, 99),
                'unguided_mean': np.mean(hr_ung),
                'guided_p25': np.percentile(hr_gui, 25),
                'guided_median': np.percentile(hr_gui, 50),
                'guided_p95': np.percentile(hr_gui, 95),
                'guided_p99': np.percentile(hr_gui, 99),
                'guided_mean': np.mean(hr_gui),
            })

            print(f"  Unguided: P25={results[-1]['unguided_p25']:.4f}, "
                  f"Median={results[-1]['unguided_median']:.4f}, "
                  f"P95={results[-1]['unguided_p95']:.4f}")
            print(f"  Guided:   P25={results[-1]['guided_p25']:.4f}, "
                  f"Median={results[-1]['guided_median']:.4f}, "
                  f"P95={results[-1]['guided_p95']:.4f}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f'hamres_pct_{args.policy}.csv'
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
