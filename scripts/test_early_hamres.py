"""
Test: At which diffusion step does HamRes on predicted x0 become predictive of final NMSE?

For each trajectory:
1. Record HamRes at steps 10, 20, 30, 40, 50 (on predicted x0)
2. Compute final NMSE
3. Check correlation between early HamRes and final NMSE
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import h5py
import mujoco
import torch
from scipy import stats

from src.models.utils import reconstruct_traj_with_momentum, EMA
from src.models.HNN import HNNWrapper
from src.models.trajectory_dpf import TrajectoryDPF

# Config
DT = 0.0001
DATA_DT = 0.0002
TRAJECTORY_LENGTH = 500
NUM_DIFFUSION_STEPS = 50
NUM_SAMPLES = 30  # Need enough samples for correlation analysis
DEVICE = 'cuda:3'

# Paths
MUJOCO_XML = project_root / 'configs' / 'rigid_arm_hinge.xml'
HNN_CHECKPOINT = project_root / 'checkpoints' / 'SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt'
DPF_CHECKPOINT = project_root / 'checkpoints' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt'

CHECK_STEPS = [10, 20, 30, 40, 50]


def compute_hamres_batch(states, torques, hnn, var_dq, var_dp, dt=DATA_DT):
    """Compute HamRes for a batch of trajectories."""
    states_np = states.cpu().numpy()
    torques_np = torques.cpu().numpy()

    hamres_list = []
    for i in range(len(states_np)):
        qpos = states_np[i, :, :3]
        mom = states_np[i, :, 3:]
        torque = torques_np[i]

        T = len(qpos)
        if T < 3:
            hamres_list.append(float('nan'))
            continue

        eps = 1e-8
        device = next(hnn.parameters()).device

        qdot = (qpos[2:] - qpos[:-2]) / (2 * dt)
        pdot = (mom[2:] - mom[:-2]) / (2 * dt)
        q_mid, p_mid, tau_mid = qpos[1:-1], mom[1:-1], torque[1:-1]

        q_t = torch.tensor(q_mid, dtype=torch.float32, device=device).requires_grad_(True)
        p_t = torch.tensor(p_mid, dtype=torch.float32, device=device).requires_grad_(True)

        with torch.enable_grad():
            H = hnn(p_t, q_t)
            dH_dp, dH_dq = torch.autograd.grad(H.sum(), (p_t, q_t))

        dH_dp = dH_dp.detach().cpu().numpy()
        dH_dq = dH_dq.detach().cpu().numpy()

        r_q = qdot - dH_dp
        r_p = pdot - (-dH_dq + tau_mid)
        hamres = np.mean(r_q**2) / (var_dq + eps) + np.mean(r_p**2) / (var_dp + eps)
        hamres_list.append(hamres)

    return np.array(hamres_list)


def compute_nmse(gen, rec):
    min_len = min(len(gen) - 1, len(rec))
    g, r = gen[1:min_len+1], rec[:min_len]
    return np.mean((g - r)**2) / (np.var(g) + 1e-8)


def compute_nmse_batch(states, torques, mj_model, traj_len):
    """Compute NMSE for a batch of trajectories."""
    states_np = states.cpu().numpy()
    torques_np = torques.cpu().numpy()

    nmse_list = []
    for i in range(len(states_np)):
        qpos_gen = states_np[i, :, :3]
        mom_gen = states_np[i, :, 3:]
        torque = torques_np[i]

        M = np.zeros((mj_model.nv, mj_model.nv))
        data = mujoco.MjData(mj_model)
        data.qpos[:] = qpos_gen[0]
        mujoco.mj_forward(mj_model, data)
        mujoco.mj_fullM(mj_model, M, data.qM)
        initial_qvel = np.linalg.solve(M, mom_gen[0])

        recon = reconstruct_traj_with_momentum(
            mj_model, traj_len, DT, qpos_gen[0], initial_qvel, torque, data_dt=DATA_DT
        )

        nmse = compute_nmse(qpos_gen, recon['seq_qpos']) + compute_nmse(mom_gen, recon['seq_mom'])
        nmse_list.append(nmse)

    return np.array(nmse_list)


def main():
    device = torch.device(DEVICE)
    print(f"Device: {device}")
    print(f"Samples: {NUM_SAMPLES}, Length: {TRAJECTORY_LENGTH}")

    # Load models
    print("Loading models...")
    model = TrajectoryDPF.load_from_checkpoint(str(DPF_CHECKPOINT), map_location=device, strict=False)
    model = model.to(device).eval()

    checkpoint = torch.load(str(DPF_CHECKPOINT), map_location=device, weights_only=False)
    if 'ema_shadow' in checkpoint:
        model.ema = EMA(model.model, decay=0.9995)
        for name, tensor in checkpoint['ema_shadow'].items():
            if name in model.ema.shadow:
                model.ema.shadow[name] = tensor.to(device=device, dtype=model.ema.shadow[name].dtype)
        model._ema_loaded = True

    hnn = HNNWrapper.load_from_checkpoint(str(HNN_CHECKPOINT), map_location=device).to(device).eval()
    var_dq = hnn.qvel_var.mean().item()
    var_dp = hnn.mom_dot_var.mean().item()

    mj_model = mujoco.MjModel.from_xml_path(str(MUJOCO_XML))

    # Load torques - use different torques for each sample
    print("Loading torques...")
    torques = []
    with h5py.File(project_root / 'data' / 'sinusoidal_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:10], dtype=torch.float32, device=device))
    with h5py.File(project_root / 'data' / 'gp_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:10], dtype=torch.float32, device=device))
    torques.append(torch.zeros(5, 1500, 3, device=device))
    with h5py.File(project_root / 'data' / 'training_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:5], dtype=torch.float32, device=device))
    torques = torch.cat(torques, dim=0)[:NUM_SAMPLES, :TRAJECTORY_LENGTH, :]
    print(f"  Using {torques.shape[0]} different torques")

    # Generate trajectories and record HamRes at each checkpoint step
    print("\nGenerating trajectories with HamRes checkpoints...")

    # We need to modify the sampling to record intermediate x0 predictions
    # For now, let's generate multiple times with different numbers of steps
    # and check if early HamRes correlates with final NMSE

    # Storage for results
    hamres_at_step = {step: [] for step in CHECK_STEPS}
    final_nmse = []

    # Generate each sample individually with different seeds
    for sample_idx in range(NUM_SAMPLES):
        torch.manual_seed(sample_idx * 100)  # Different seed for each sample
        torch.cuda.manual_seed_all(sample_idx * 100)

        sample_torque = torques[sample_idx:sample_idx+1]

        # For each check step, run diffusion up to that step and get x0 prediction
        x0_at_steps = {}

        for check_step in CHECK_STEPS:
            torch.manual_seed(sample_idx * 100)  # Same seed to get same trajectory
            torch.cuda.manual_seed_all(sample_idx * 100)

            states, torques_out = model.sample_trajectories(
                num_samples=1,
                trajectory_length=TRAJECTORY_LENGTH,
                num_diffusion_steps=check_step,  # Run only up to this step
                context_fraction=0.2,
                use_ema=False,
                sampler='ddim',
                torque=sample_torque,
                hnn=None,
                guidance_steps=0,
            )

            # Compute HamRes on the x0 prediction at this step
            hamres = compute_hamres_batch(states, torques_out, hnn, var_dq, var_dp)[0]
            hamres_at_step[check_step].append(hamres)

            # Store final states for NMSE computation
            if check_step == 50:
                nmse = compute_nmse_batch(states, torques_out, mj_model, TRAJECTORY_LENGTH)[0]
                final_nmse.append(nmse)

        if (sample_idx + 1) % 10 == 0:
            print(f"  Processed {sample_idx + 1}/{NUM_SAMPLES} samples")

    # Convert to arrays
    final_nmse = np.array(final_nmse)
    for step in CHECK_STEPS:
        hamres_at_step[step] = np.array(hamres_at_step[step])

    # Compute correlations
    print("\n" + "="*70)
    print("CORRELATION: HamRes@Step vs Final NMSE")
    print("="*70)
    print(f"{'Step':<10} {'Pearson r':>12} {'p-value':>12} {'Spearman r':>12} {'p-value':>12}")
    print("-"*70)

    for step in CHECK_STEPS:
        # Remove NaN values
        mask = ~(np.isnan(hamres_at_step[step]) | np.isnan(final_nmse))
        h = hamres_at_step[step][mask]
        n = final_nmse[mask]

        if len(h) > 3:
            r_pearson, p_pearson = stats.pearsonr(h, n)
            r_spearman, p_spearman = stats.spearmanr(h, n)
            print(f"{step:<10} {r_pearson:>+12.3f} {p_pearson:>12.4f} {r_spearman:>+12.3f} {p_spearman:>12.4f}")
        else:
            print(f"{step:<10} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

    print("="*70)

    # Summary statistics
    print("\nSummary Statistics:")
    print(f"  Final NMSE: median={np.nanmedian(final_nmse):.4f}, mean={np.nanmean(final_nmse):.4f}")
    for step in CHECK_STEPS:
        print(f"  HamRes@{step}: median={np.nanmedian(hamres_at_step[step]):.4f}, mean={np.nanmean(hamres_at_step[step]):.4f}")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    for step in CHECK_STEPS:
        mask = ~(np.isnan(hamres_at_step[step]) | np.isnan(final_nmse))
        h = hamres_at_step[step][mask]
        n = final_nmse[mask]
        if len(h) > 3:
            r, p = stats.pearsonr(h, n)
            if r > 0.3 and p < 0.05:
                print(f"Step {step}: HamRes is PREDICTIVE (r={r:.3f}, p={p:.4f}) -> Can use for early rejection")
            else:
                print(f"Step {step}: HamRes is NOT predictive (r={r:.3f}, p={p:.4f})")


if __name__ == '__main__':
    main()
