"""
Correct test: Record x0 predictions during a SINGLE 50-step diffusion run.
Check if HamRes at intermediate steps correlates with final NMSE.
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
from tqdm import tqdm

from src.models.utils import reconstruct_traj_with_momentum, EMA
from src.models.HNN import HNNWrapper
from src.models.trajectory_dpf import TrajectoryDPF

# Config
DT = 0.0001
DATA_DT = 0.0002
TRAJECTORY_LENGTH = 500
NUM_DIFFUSION_STEPS = 50
NUM_SAMPLES = 30
DEVICE = 'cuda:3'

MUJOCO_XML = project_root / 'configs' / 'rigid_arm_hinge.xml'
HNN_CHECKPOINT = project_root / 'checkpoints' / 'SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt'
DPF_CHECKPOINT = project_root / 'checkpoints' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt'

CHECK_STEPS = [10, 20, 30, 40, 49]  # Steps at which to record x0


def compute_hamres_single(qpos, mom, torque, hnn, var_dq, var_dp, dt=DATA_DT):
    """Compute HamRes for a single trajectory."""
    T = len(qpos)
    if T < 3:
        return float('nan')
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
    return np.mean(r_q**2) / (var_dq + eps) + np.mean(r_p**2) / (var_dp + eps)


def compute_nmse(gen, rec):
    min_len = min(len(gen) - 1, len(rec))
    g, r = gen[1:min_len+1], rec[:min_len]
    return np.mean((g - r)**2) / (np.var(g) + 1e-8)


def sample_with_checkpoints(model, torque, hnn, var_dq, var_dp, check_steps):
    """
    Sample trajectories and record HamRes at checkpoint steps.
    Returns: dict {step: [hamres for each sample]}, final_states
    """
    device = model.device
    num_samples = torque.shape[0]
    trajectory_length = torque.shape[1]

    # Start with pure noise
    x = torch.randn(num_samples, trajectory_length, model.state_dim, device=device)

    # Normalized conditioning
    cond = model.normalize_cond(torque)

    # DDIM timesteps
    ts = torch.linspace(model.diffusion_steps - 1, 0, steps=NUM_DIFFUSION_STEPS, device=device, dtype=torch.long)

    # Context setup
    max_context_train = int(model.max_timesteps * 0.2)
    num_context = max(1, min(trajectory_length - 1, max_context_train))

    # Storage for HamRes at each checkpoint
    hamres_at_step = {step: [] for step in check_steps}

    for i, t in enumerate(tqdm(ts, desc="Sampling")):
        t_int = int(t.item())

        # Build tokens
        queries = model.build_tokens(x, t_int + 1, skip_normalize=True)
        contexts = queries[:, :num_context, :]

        # Predict noise
        with torch.no_grad():
            eps = model.model(contexts, queries, cond)

        # Get diffusion parameters
        a_bar_t = model.alpha_cumprod[t_int]
        if i < len(ts) - 1:
            t_prev_int = int(ts[i + 1].item())
            a_bar_prev = model.alpha_cumprod[t_prev_int]
        else:
            a_bar_prev = a_bar_t.new_tensor(1.0)

        a_bar_t = torch.clamp(a_bar_t, min=1e-6, max=1.0)
        a_bar_prev = torch.clamp(a_bar_prev, min=1e-6, max=1.0)

        # DDIM: predict x0
        x0 = model._predict_x0(x, eps, a_bar_t)

        # Record HamRes at checkpoint steps
        if i in check_steps:
            x0_phys = model.denormalize_state(x0)
            x0_np = x0_phys.cpu().numpy()
            torque_np = torque.cpu().numpy()

            for j in range(num_samples):
                qpos = x0_np[j, :, :3]
                mom = x0_np[j, :, 3:]
                tau = torque_np[j]
                hamres = compute_hamres_single(qpos, mom, tau, hnn, var_dq, var_dp)
                hamres_at_step[i].append(hamres)

        # DDIM update
        eps_coef = torch.sqrt(1.0 - a_bar_prev)
        x = torch.sqrt(a_bar_prev) * x0 + eps_coef * eps

    # Final x0
    final_states = model.denormalize_state(x0)
    return hamres_at_step, final_states, torque


def main():
    device = torch.device(DEVICE)
    print(f"Device: {device}")
    print(f"Samples: {NUM_SAMPLES}, Length: {TRAJECTORY_LENGTH}")
    print(f"Check steps: {CHECK_STEPS}")

    # Load models
    print("\nLoading models...")
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

    # Load torques
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
    print(f"  Using {torques.shape[0]} torques")

    # Sample with checkpoints
    print("\nSampling with HamRes checkpoints...")
    torch.manual_seed(228)
    torch.cuda.manual_seed_all(228)

    hamres_at_step, final_states, torque_out = sample_with_checkpoints(
        model, torques, hnn, var_dq, var_dp, CHECK_STEPS
    )

    # Compute final NMSE
    print("\nComputing final NMSE...")
    final_np = final_states.cpu().numpy()
    torque_np = torque_out.cpu().numpy()

    final_nmse = []
    for j in range(NUM_SAMPLES):
        qpos_gen = final_np[j, :, :3]
        mom_gen = final_np[j, :, 3:]
        tau = torque_np[j]

        M = np.zeros((mj_model.nv, mj_model.nv))
        data = mujoco.MjData(mj_model)
        data.qpos[:] = qpos_gen[0]
        mujoco.mj_forward(mj_model, data)
        mujoco.mj_fullM(mj_model, M, data.qM)
        initial_qvel = np.linalg.solve(M, mom_gen[0])

        recon = reconstruct_traj_with_momentum(
            mj_model, TRAJECTORY_LENGTH, DT, qpos_gen[0], initial_qvel, tau, data_dt=DATA_DT
        )

        nmse = compute_nmse(qpos_gen, recon['seq_qpos']) + compute_nmse(mom_gen, recon['seq_mom'])
        final_nmse.append(nmse)

    final_nmse = np.array(final_nmse)

    # Convert HamRes to arrays
    for step in CHECK_STEPS:
        hamres_at_step[step] = np.array(hamres_at_step[step])

    # Compute correlations
    print("\n" + "="*80)
    print("CORRELATION: HamRes@Step (same diffusion run) vs Final NMSE")
    print("="*80)
    print(f"{'Step':<10} {'Pearson r':>12} {'p-value':>12} {'Spearman r':>12} {'p-value':>12}")
    print("-"*80)

    for step in CHECK_STEPS:
        mask = ~(np.isnan(hamres_at_step[step]) | np.isnan(final_nmse))
        h = hamres_at_step[step][mask]
        n = final_nmse[mask]

        if len(h) > 3:
            r_pearson, p_pearson = stats.pearsonr(h, n)
            r_spearman, p_spearman = stats.spearmanr(h, n)
            sig_p = "*" if p_pearson < 0.05 else ""
            sig_s = "*" if p_spearman < 0.05 else ""
            print(f"{step:<10} {r_pearson:>+12.3f} {p_pearson:>11.4f}{sig_p} {r_spearman:>+12.3f} {p_spearman:>11.4f}{sig_s}")

    print("="*80)
    print("* = p < 0.05 (statistically significant)")

    # Summary
    print("\nSummary:")
    print(f"  Final NMSE: median={np.nanmedian(final_nmse):.4f}, mean={np.nanmean(final_nmse):.4f}")
    for step in CHECK_STEPS:
        print(f"  HamRes@step{step}: median={np.nanmedian(hamres_at_step[step]):.4f}")

    # Check if any step is predictive
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    best_step = None
    best_corr = 0
    for step in CHECK_STEPS:
        mask = ~(np.isnan(hamres_at_step[step]) | np.isnan(final_nmse))
        h = hamres_at_step[step][mask]
        n = final_nmse[mask]
        if len(h) > 3:
            r, p = stats.pearsonr(h, n)
            if r > best_corr and p < 0.1:
                best_corr = r
                best_step = step

    if best_step is not None and best_corr > 0.3:
        print(f"Step {best_step} has predictive power (r={best_corr:.3f})")
        print("-> Early pruning at this step may be effective")
    else:
        print("No step has strong predictive power for final NMSE")
        print("-> Early pruning based on HamRes may not be effective")


if __name__ == '__main__':
    main()
