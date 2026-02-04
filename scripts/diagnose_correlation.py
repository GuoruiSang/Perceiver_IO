"""
Diagnose correlation between HamRes, Energy Error, and NMSE.
This helps us understand if HNN-based metrics are useful for guidance/rejection.
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
NUM_SAMPLES = 30  # Generate 30 trajectories for correlation analysis
DEVICE = 'cuda:3'

# Paths
MUJOCO_XML = project_root / 'configs' / 'rigid_arm_hinge.xml'
HNN_CHECKPOINT = project_root / 'checkpoints' / 'SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt'
DPF_CHECKPOINT = project_root / 'checkpoints' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt'


def compute_hamres(qpos, mom, torque, hnn, var_dq, var_dp, dt=DATA_DT):
    """Compute Hamiltonian residual."""
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


def compute_energy_error(qpos, mom, torque, hnn, dt=DATA_DT):
    """
    Compute energy conservation error.
    E(T) - E(0) should equal Work done by torque.
    """
    device = next(hnn.parameters()).device

    # Energy at start and end
    q0 = torch.tensor(qpos[0:1], dtype=torch.float32, device=device)
    p0 = torch.tensor(mom[0:1], dtype=torch.float32, device=device)
    qT = torch.tensor(qpos[-1:], dtype=torch.float32, device=device)
    pT = torch.tensor(mom[-1:], dtype=torch.float32, device=device)

    with torch.no_grad():
        E0 = hnn(p0, q0).item()
        ET = hnn(pT, qT).item()

    # Work done by torque: W = integral of tau * qdot
    # qdot approximated by central difference
    qdot = (qpos[2:] - qpos[:-2]) / (2 * dt)  # shape: (T-2, 3)
    tau_mid = torque[1:-1]  # shape: (T-2, 3)

    # W = sum of tau * qdot * dt
    work = np.sum(tau_mid * qdot) * dt

    # Energy error
    delta_E = ET - E0
    energy_error = abs(delta_E - work)

    return energy_error, delta_E, work


def compute_nmse(gen, rec):
    min_len = min(len(gen) - 1, len(rec))
    g, r = gen[1:min_len+1], rec[:min_len]
    return np.mean((g - r)**2) / (np.var(g) + 1e-8)


def main():
    device = torch.device(DEVICE)
    print(f"Device: {device}")

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

    # Load mixed torques
    print("Loading torques...")
    torques = []
    with h5py.File(project_root / 'data' / 'sinusoidal_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:10], dtype=torch.float32, device=device))
    with h5py.File(project_root / 'data' / 'gp_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:10], dtype=torch.float32, device=device))
    torques.append(torch.zeros(5, 1500, 3, device=device))
    with h5py.File(project_root / 'data' / 'training_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:5], dtype=torch.float32, device=device))
    torques = torch.cat(torques, dim=0)
    print(f"  {torques.shape[0]} torques available")

    # Generate trajectories (baseline, no guidance)
    print(f"\nGenerating {NUM_SAMPLES} baseline trajectories (no guidance)...")
    traj_torques = torques[:NUM_SAMPLES, :TRAJECTORY_LENGTH, :]

    states, torques_out = model.sample_trajectories(
        num_samples=NUM_SAMPLES,
        trajectory_length=TRAJECTORY_LENGTH,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        context_fraction=0.2,
        use_ema=False,
        sampler='ddim',
        guidance_scale=1.0,
        torque=traj_torques,
        hnn=None,  # No guidance
        guidance_steps=0,
    )

    states = states.cpu().numpy()
    torques_out = torques_out.cpu().numpy()

    # Compute metrics for each trajectory
    print("\nComputing metrics...")
    hamres_list = []
    energy_error_list = []
    nmse_list = []

    for i in range(NUM_SAMPLES):
        qpos_gen = states[i, :, :3]
        mom_gen = states[i, :, 3:]
        torque = torques_out[i]

        # MuJoCo reconstruction
        M = np.zeros((mj_model.nv, mj_model.nv))
        data = mujoco.MjData(mj_model)
        data.qpos[:] = qpos_gen[0]
        mujoco.mj_forward(mj_model, data)
        mujoco.mj_fullM(mj_model, M, data.qM)
        initial_qvel = np.linalg.solve(M, mom_gen[0])

        recon = reconstruct_traj_with_momentum(
            mj_model, TRAJECTORY_LENGTH, DT, qpos_gen[0], initial_qvel, torque, data_dt=DATA_DT
        )

        # Compute metrics
        hamres = compute_hamres(qpos_gen, mom_gen, torque, hnn, var_dq, var_dp)
        energy_err, delta_E, work = compute_energy_error(qpos_gen, mom_gen, torque, hnn)
        nmse = compute_nmse(qpos_gen, recon['seq_qpos']) + compute_nmse(mom_gen, recon['seq_mom'])

        hamres_list.append(hamres)
        energy_error_list.append(energy_err)
        nmse_list.append(nmse)

        if i < 5:
            print(f"  Sample {i}: HamRes={hamres:.4f}, EnergyErr={energy_err:.4f}, NMSE={nmse:.4f}")

    hamres_arr = np.array(hamres_list)
    energy_arr = np.array(energy_error_list)
    nmse_arr = np.array(nmse_list)

    # Compute correlations
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)

    # Pearson correlation
    r_hamres_nmse, p_hamres = stats.pearsonr(hamres_arr, nmse_arr)
    r_energy_nmse, p_energy = stats.pearsonr(energy_arr, nmse_arr)
    r_hamres_energy, p_he = stats.pearsonr(hamres_arr, energy_arr)

    print(f"\nPearson Correlations:")
    print(f"  HamRes vs NMSE:       r = {r_hamres_nmse:+.3f}  (p = {p_hamres:.4f})")
    print(f"  EnergyErr vs NMSE:    r = {r_energy_nmse:+.3f}  (p = {p_energy:.4f})")
    print(f"  HamRes vs EnergyErr:  r = {r_hamres_energy:+.3f}  (p = {p_he:.4f})")

    # Spearman correlation (rank-based, more robust)
    rs_hamres_nmse, ps_hamres = stats.spearmanr(hamres_arr, nmse_arr)
    rs_energy_nmse, ps_energy = stats.spearmanr(energy_arr, nmse_arr)

    print(f"\nSpearman Correlations (rank-based):")
    print(f"  HamRes vs NMSE:       r = {rs_hamres_nmse:+.3f}  (p = {ps_hamres:.4f})")
    print(f"  EnergyErr vs NMSE:    r = {rs_energy_nmse:+.3f}  (p = {ps_energy:.4f})")

    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"  HamRes:    median={np.median(hamres_arr):.4f}, mean={np.mean(hamres_arr):.4f}, std={np.std(hamres_arr):.4f}")
    print(f"  EnergyErr: median={np.median(energy_arr):.4f}, mean={np.mean(energy_arr):.4f}, std={np.std(energy_arr):.4f}")
    print(f"  NMSE:      median={np.median(nmse_arr):.4f}, mean={np.mean(nmse_arr):.4f}, std={np.std(nmse_arr):.4f}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if r_hamres_nmse > 0.3:
        print("HamRes is POSITIVELY correlated with NMSE -> Can use for rejection sampling")
    elif r_hamres_nmse < -0.3:
        print("HamRes is NEGATIVELY correlated with NMSE -> HNN guidance hurts NMSE (as observed)")
    else:
        print("HamRes has WEAK correlation with NMSE -> Not a good proxy metric")

    if r_energy_nmse > 0.3:
        print("EnergyErr is POSITIVELY correlated with NMSE -> Can use for guidance/rejection!")
    elif r_energy_nmse < -0.3:
        print("EnergyErr is NEGATIVELY correlated with NMSE -> Energy check not useful")
    else:
        print("EnergyErr has WEAK correlation with NMSE -> Not a good proxy metric")


if __name__ == '__main__':
    main()
