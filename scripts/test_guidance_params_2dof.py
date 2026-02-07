"""
Quick test of guidance parameters for 2-DoF system.

Tests different combinations of guidance_lr, guidance_steps, guidance_after_steps
to find optimal settings.

Usage:
    python scripts/test_guidance_params_2dof.py
    python scripts/test_guidance_params_2dof.py --num_samples 10 --length 1000
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
import h5py
import mujoco
from itertools import product
from concurrent.futures import ThreadPoolExecutor

from src.models.trajectory_dpf import TrajectoryDPF
from src.models.HNN import HNNWrapper
from src.models.utils import EMA


# =============================================================================
# CONFIGURATION - Adjust these parameters as needed
# =============================================================================

# Test settings
NUM_SAMPLES = 100                    # Number of samples per test
TRAJECTORY_LENGTH = 1000           # Trajectory length
DEVICE = 'cuda:0'                  # GPU device
SEED = 42                          # Random seed for reproducibility

# Parameter grid to search
LR_VALUES = [1e-3, 1e-4, 1e-5]                 # guidance_lr options
STEPS_VALUES = [1, 2, 5, 10]                # guidance_steps options
AFTER_VALUES = [40, 45, 48]                # guidance_after_steps options

# Physics constants
DT = 0.0002                        # Data timestep (5000 Hz)
SIM_DT = 0.0001                    # MuJoCo simulation timestep
QPOS_DIM = 2                       # Position dimensions
MOM_DIM = 2                        # Momentum dimensions

# Model paths (2-DoF)
DPF_CHECKPOINT = project_root / 'checkpoints' / '2dof' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0008.ckpt'
HNN_CHECKPOINT = project_root / 'checkpoints' / '2dof' / 'SeperableHNN-2DOF-epoch-epoch=999.ckpt'
TORQUE_PATH = project_root / 'data' / '2dof' / 'sinusoidal_torques_2000_L1500.h5'
MUJOCO_XML = project_root / 'configs' / 'rigid_arm_hinge_2dof.xml'

# =============================================================================
# END CONFIGURATION
# =============================================================================


def reconstruct_single(args):
    """Reconstruct a single trajectory using MuJoCo (for parallel execution)."""
    qpos_init, mom_init, torque, mj_model, sim_dt, data_dt, qpos_dim = args

    T = len(torque)
    substeps = int(data_dt / sim_dt)

    # Set simulation timestep
    mj_model.opt.timestep = sim_dt

    # Compute initial velocity from momentum
    data = mujoco.MjData(mj_model)
    data.qpos[:qpos_dim] = qpos_init
    data.qvel[:] = 0
    mujoco.mj_forward(mj_model, data)

    M = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, M, data.qM)
    initial_qvel = np.linalg.solve(M[:qpos_dim, :qpos_dim], mom_init)

    # Forward simulation
    qpos_rec = [qpos_init.copy()]
    mom_rec = [mom_init.copy()]

    data.qpos[:qpos_dim] = qpos_init
    data.qvel[:qpos_dim] = initial_qvel
    mujoco.mj_forward(mj_model, data)

    for t in range(T - 1):
        data.ctrl[:qpos_dim] = torque[t]
        for _ in range(substeps):
            mujoco.mj_step(mj_model, data)

        qpos_rec.append(data.qpos[:qpos_dim].copy())
        mujoco.mj_fullM(mj_model, M, data.qM)
        mom = M[:qpos_dim, :qpos_dim] @ data.qvel[:qpos_dim]
        mom_rec.append(mom.copy())

    return np.array(qpos_rec), np.array(mom_rec)


def reconstruct_batch_parallel(states, torques, mj_model, sim_dt, data_dt, qpos_dim, max_workers=8):
    """Reconstruct trajectories in parallel using ThreadPoolExecutor."""
    B = states.shape[0]
    states_np = states.cpu().numpy()
    torques_np = torques.cpu().numpy()

    args_list = [
        (states_np[i, 0, :qpos_dim], states_np[i, 0, qpos_dim:], torques_np[i],
         mj_model, sim_dt, data_dt, qpos_dim)
        for i in range(B)
    ]

    qpos_rec_list = []
    mom_rec_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(reconstruct_single, args_list))

    for qpos_rec, mom_rec in results:
        qpos_rec_list.append(qpos_rec)
        mom_rec_list.append(mom_rec)

    return np.stack(qpos_rec_list), np.stack(mom_rec_list)


def compute_nmse_batch(states_gen, qpos_rec, mom_rec, qpos_dim):
    """Compute NMSE between generated and reconstructed trajectories.

    Uses per-trajectory variance (same as compute_ablation_mse*.py):
    - NMSE_q = MSE_q / Var(qpos_gen)
    - NMSE_p = MSE_p / Var(mom_gen)
    """
    states_np = states_gen.cpu().numpy()
    qpos_gen = states_np[:, :, :qpos_dim]
    mom_gen = states_np[:, :, qpos_dim:]

    # Skip first timestep and align lengths
    T = min(qpos_gen.shape[1], qpos_rec.shape[1])
    qpos_gen_skip = qpos_gen[:, 1:T]
    mom_gen_skip = mom_gen[:, 1:T]
    qpos_rec_skip = qpos_rec[:, 1:T]
    mom_rec_skip = mom_rec[:, 1:T]

    # Compute per-sample NMSE using per-trajectory variance
    nmse_q = []
    nmse_p = []

    for i in range(len(qpos_gen)):
        # MSE: generated[1:] vs reconstructed[1:]
        mse_q = np.mean((qpos_gen_skip[i] - qpos_rec_skip[i]) ** 2)
        mse_p = np.mean((mom_gen_skip[i] - mom_rec_skip[i]) ** 2)

        # Per-trajectory variance for normalization
        var_q = np.var(qpos_gen_skip[i])
        var_p = np.var(mom_gen_skip[i])

        nmse_q.append(mse_q / (var_q + 1e-12))
        nmse_p.append(mse_p / (var_p + 1e-12))

    return np.array(nmse_q), np.array(nmse_p)


def load_models(device):
    """Load DPF and HNN models."""
    print("Loading DPF model...")
    dpf = TrajectoryDPF.load_from_checkpoint(
        str(DPF_CHECKPOINT), map_location=device, strict=False
    )
    dpf.eval()
    dpf.to(device)

    checkpoint = torch.load(str(DPF_CHECKPOINT), map_location=device, weights_only=False)
    if 'ema_shadow' in checkpoint:
        dpf.ema = EMA(dpf.model, decay=0.9995)
        for name, tensor in checkpoint['ema_shadow'].items():
            if name in dpf.ema.shadow:
                dpf.ema.shadow[name] = tensor.to(device=device, dtype=dpf.ema.shadow[name].dtype)
        print("  EMA loaded")

    print("Loading HNN model...")
    hnn = HNNWrapper.load_from_checkpoint(str(HNN_CHECKPOINT), map_location=device)
    hnn.eval()
    hnn.to(device)

    return dpf, hnn


def load_torques(num_samples, length, device):
    """Load torque sequences."""
    with h5py.File(TORQUE_PATH, 'r') as f:
        torques = f['torques'][:num_samples, :length]
    return torch.tensor(torques, dtype=torch.float32, device=device)


def compute_hamres(qpos, mom, torque, hnn, var_dq, var_dp, dt=DT):
    """Compute HamRes for a batch of trajectories."""
    B, T, dim = qpos.shape
    if T < 3:
        return torch.tensor([float('nan')] * B)

    eps = 1e-8

    qdot = (qpos[:, 2:] - qpos[:, :-2]) / (2 * dt)
    pdot = (mom[:, 2:] - mom[:, :-2]) / (2 * dt)

    q_mid = qpos[:, 1:-1]
    p_mid = mom[:, 1:-1]
    tau_mid = torque[:, 1:-1]

    B_T = B * (T - 2)
    q_flat = q_mid.reshape(B_T, dim)
    p_flat = p_mid.reshape(B_T, dim)

    with torch.inference_mode(False):
        with torch.enable_grad():
            p_grad = p_flat.detach().clone().requires_grad_(True)
            q_grad = q_flat.detach().clone().requires_grad_(True)
            H = hnn(p_grad, q_grad)
            dH_dp, dH_dq = torch.autograd.grad(H.sum(), (p_grad, q_grad))

    dH_dp = dH_dp.reshape(B, T - 2, dim)
    dH_dq = dH_dq.reshape(B, T - 2, dim)

    r_q = qdot - dH_dp
    r_p = pdot - (-dH_dq + tau_mid)

    mse_q = (r_q ** 2).mean(dim=(1, 2))
    mse_p = (r_p ** 2).mean(dim=(1, 2))

    hamres = mse_q / (var_dq + eps) + mse_p / (var_dp + eps)
    return hamres


def compute_stats(arr):
    """Compute mean, std, median, and percentiles."""
    arr = np.array(arr)
    return {
        'mean': np.mean(arr),
        'std': np.std(arr),
        'median': np.median(arr),
        'p25': np.percentile(arr, 25),
        'p95': np.percentile(arr, 95),
        'p99': np.percentile(arr, 99),
    }


def test_params(dpf, hnn, torques, mj_model, device, num_samples, length, lr, steps, after, seed):
    """Test a specific parameter combination."""
    var_dq = hnn.qvel_var.mean().to(device)
    var_dp = hnn.mom_dot_var.mean().to(device)

    # Reset seed for unguided sampling (same initial noise for all combos)
    set_seed(seed)
    with torch.no_grad():
        state_ung, _ = dpf.sample_trajectories(
            num_samples=num_samples,
            trajectory_length=length,
            context_fraction=0.5,
            use_ema=True,
            hnn=None,
            torque=torques,
        )

    # Reset seed for guided sampling (same initial noise as unguided)
    set_seed(seed)
    state_gui, _ = dpf.sample_trajectories(
        num_samples=num_samples,
        trajectory_length=length,
        context_fraction=0.5,
        use_ema=True,
        hnn=hnn,
        guidance_method="adam",
        guidance_steps=steps,
        guidance_after_steps=after,
        guidance_lr=lr,
        torque=torques,
    )

    dim = QPOS_DIM
    qpos_ung, mom_ung = state_ung[:, :, :dim], state_ung[:, :, dim:]
    qpos_gui, mom_gui = state_gui[:, :, :dim], state_gui[:, :, dim:]

    # Compute HamRes
    hamres_ung = compute_hamres(qpos_ung, mom_ung, torques, hnn, var_dq, var_dp).cpu().numpy()
    hamres_gui = compute_hamres(qpos_gui, mom_gui, torques, hnn, var_dq, var_dp).cpu().numpy()

    # MuJoCo reconstruction for NMSE (parallel)
    print("    Reconstructing unguided...", end=" ", flush=True)
    qpos_rec_ung, mom_rec_ung = reconstruct_batch_parallel(
        state_ung, torques, mj_model, SIM_DT, DT, dim)
    print("guided...", end=" ", flush=True)
    qpos_rec_gui, mom_rec_gui = reconstruct_batch_parallel(
        state_gui, torques, mj_model, SIM_DT, DT, dim)
    print("done")

    # Compute NMSE
    nmse_q_ung, nmse_p_ung = compute_nmse_batch(state_ung, qpos_rec_ung, mom_rec_ung, dim)
    nmse_q_gui, nmse_p_gui = compute_nmse_batch(state_gui, qpos_rec_gui, mom_rec_gui, dim)

    return {
        'hamres_ung': compute_stats(hamres_ung),
        'hamres_gui': compute_stats(hamres_gui),
        'nmse_q_ung': compute_stats(nmse_q_ung),
        'nmse_q_gui': compute_stats(nmse_q_gui),
        'nmse_p_ung': compute_stats(nmse_p_ung),
        'nmse_p_gui': compute_stats(nmse_p_gui),
    }


def print_metric_table(name, ung_stats, gui_stats):
    """Print a comparison table for a single metric."""
    print(f"\n  {name}:")
    print(f"  {'Stat':<10} {'Unguided':<12} {'Guided':<12} {'Diff':<12}")
    print(f"  {'-'*46}")
    for stat in ['mean', 'std', 'median', 'p25', 'p95', 'p99']:
        u = ung_stats[stat]
        g = gui_stats[stat]
        print(f"  {stat:<10} {u:<12.6f} {g:<12.6f} {g-u:+.6f}")


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES)
    parser.add_argument('--length', type=int, default=TRAJECTORY_LENGTH)
    parser.add_argument('--device', type=str, default=DEVICE)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Samples: {args.num_samples}, Length: {args.length}")
    print(f"LR grid: {LR_VALUES}")
    print(f"Steps grid: {STEPS_VALUES}")
    print(f"After grid: {AFTER_VALUES}")

    # Load models
    dpf, hnn = load_models(device)
    torques = load_torques(args.num_samples, args.length, device)

    # Load MuJoCo model
    print(f"Loading MuJoCo model: {MUJOCO_XML}")
    mj_model = mujoco.MjModel.from_xml_path(str(MUJOCO_XML))

    results = []
    combos = list(product(LR_VALUES, STEPS_VALUES, AFTER_VALUES))

    print(f"\nTesting {len(combos)} combinations...\n")

    for lr, steps, after in combos:
        print(f">>> Testing lr={lr:.0e}, steps={steps}, after={after}")
        m = test_params(dpf, hnn, torques, mj_model, device, args.num_samples, args.length, lr, steps, after, args.seed)
        results.append({'lr': lr, 'steps': steps, 'after': after, **m})

    # Print detailed comparison table for each parameter combination
    print("\n" + "=" * 90)
    print("DETAILED RESULTS: Unguided vs Guided")
    print("=" * 90)

    for r in results:
        print(f"\n{'='*70}")
        print(f">>> lr={r['lr']:.0e}, steps={r['steps']}, after={r['after']}")
        print(f"{'='*70}")

        print_metric_table("HamRes", r['hamres_ung'], r['hamres_gui'])
        print_metric_table("NMSE_q (position)", r['nmse_q_ung'], r['nmse_q_gui'])
        print_metric_table("NMSE_p (momentum)", r['nmse_p_ung'], r['nmse_p_gui'])

    # Summary table
    if len(results) > 1:
        print("\n" + "=" * 120)
        print("SUMMARY: All Combinations (median values)")
        print("=" * 120)
        print(f"{'LR':<10} {'Steps':<8} {'After':<8} "
              f"{'HR_U':<10} {'HR_G':<10} "
              f"{'NQ_U':<10} {'NQ_G':<10} "
              f"{'NP_U':<10} {'NP_G':<10}")
        print("-" * 120)

        for r in results:
            print(f"{r['lr']:<10.0e} {r['steps']:<8} {r['after']:<8} "
                  f"{r['hamres_ung']['median']:<10.4f} {r['hamres_gui']['median']:<10.4f} "
                  f"{r['nmse_q_ung']['median']:<10.6f} {r['nmse_q_gui']['median']:<10.6f} "
                  f"{r['nmse_p_ung']['median']:<10.6f} {r['nmse_p_gui']['median']:<10.6f}")

        print("\n" + "=" * 90)
        print("TOP 5 by HamRes improvement (positive = guidance helps):")
        sorted_hr = sorted(results, key=lambda x: x['hamres_ung']['median'] - x['hamres_gui']['median'], reverse=True)
        for i, r in enumerate(sorted_hr[:5]):
            improv = r['hamres_ung']['median'] - r['hamres_gui']['median']
            print(f"{i+1}. lr={r['lr']:.0e}, steps={r['steps']}, after={r['after']} -> improvement: {improv:+.4f}")

        print("\nTOP 5 by lowest guided HamRes (median):")
        sorted_gui = sorted(results, key=lambda x: x['hamres_gui']['median'])
        for i, r in enumerate(sorted_gui[:5]):
            print(f"{i+1}. lr={r['lr']:.0e}, steps={r['steps']}, after={r['after']} -> HamRes: {r['hamres_gui']['median']:.4f}")


if __name__ == '__main__':
    main()
