"""
Quick test: Compare Energy guidance vs HamRes guidance vs Baseline.
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import h5py
import mujoco
import torch
import torch.nn as nn
from tqdm import trange

from src.models.utils import reconstruct_traj_with_momentum, EMA
from src.models.HNN import HNNWrapper
from src.models.trajectory_dpf import TrajectoryDPF

# Config
DT = 0.0001
DATA_DT = 0.0002
TRAJECTORY_LENGTH = 500
NUM_DIFFUSION_STEPS = 50
NUM_SAMPLES = 5
DEVICE = 'cuda:3'

# Paths
MUJOCO_XML = project_root / 'configs' / 'rigid_arm_hinge.xml'
HNN_CHECKPOINT = project_root / 'checkpoints' / 'SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt'
DPF_CHECKPOINT = project_root / 'checkpoints' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt'


def compute_energy_loss(seq_qpos, seq_mom, seq_torque, hnn, dt):
    """
    Compute energy conservation loss.
    E(T) - E(0) should equal Work done by torque.
    """
    B, T, _ = seq_qpos.shape

    # Energy at start and end
    E0 = hnn(seq_mom[:, 0, :], seq_qpos[:, 0, :])  # [B]
    ET = hnn(seq_mom[:, -1, :], seq_qpos[:, -1, :])  # [B]

    # Work done by torque: W = integral of tau * qdot
    # qdot approximated by central difference
    qdot = (seq_qpos[:, 2:, :] - seq_qpos[:, :-2, :]) / (2 * dt)  # [B, T-2, 3]
    tau_mid = seq_torque[:, 1:-1, :]  # [B, T-2, 3]

    # W = sum of tau * qdot * dt
    work = (tau_mid * qdot).sum(dim=(1, 2)) * dt  # [B]

    # Energy error
    delta_E = ET - E0
    energy_error = (delta_E - work).pow(2).mean()

    return energy_error


def run_energy_optimization(x, seq_torque, qpos_dim, mom_dim, dt, hnn, num_steps, lr=1e-3):
    """Optimize trajectory using energy conservation loss."""
    seq_qpos = nn.Parameter(x[:, :, :qpos_dim].clone())
    seq_mom = nn.Parameter(x[:, :, qpos_dim:qpos_dim + mom_dim].clone())

    optimizer = torch.optim.Adam([seq_qpos, seq_mom], lr=lr)

    for i in trange(num_steps, desc='Energy Optimization', leave=False):
        optimizer.zero_grad()
        energy_loss = compute_energy_loss(seq_qpos, seq_mom, seq_torque, hnn, dt)

        if i == 0 or i == num_steps - 1:
            print(f'  Energy Loss: {energy_loss.item():.6f}')

        energy_loss.backward()
        optimizer.step()

    return torch.cat([seq_qpos.data, seq_mom.data], dim=-1)


def compute_nmse(gen, rec):
    min_len = min(len(gen) - 1, len(rec))
    g, r = gen[1:min_len+1], rec[:min_len]
    return np.mean((g - r)**2) / (np.var(g) + 1e-8)


def compute_hamres(qpos, mom, torque, hnn, var_dq, var_dp, dt=DATA_DT):
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


def evaluate_trajectories(states, torques_out, hnn, mj_model, var_dq, var_dp):
    """Evaluate trajectories and return metrics."""
    states_np = states.cpu().numpy()
    torques_np = torques_out.cpu().numpy()

    hamres_list, nmse_list = [], []
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
            mj_model, len(qpos_gen), DT, qpos_gen[0], initial_qvel, torque, data_dt=DATA_DT
        )

        hamres_list.append(compute_hamres(qpos_gen, mom_gen, torque, hnn, var_dq, var_dp))
        nmse_list.append(compute_nmse(qpos_gen, recon['seq_qpos']) + compute_nmse(mom_gen, recon['seq_mom']))

    return np.median(hamres_list), np.mean(nmse_list), np.median(nmse_list)


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

    # Load torques
    print("Loading torques...")
    torques = []
    with h5py.File(project_root / 'data' / 'sinusoidal_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:2], dtype=torch.float32, device=device))
    with h5py.File(project_root / 'data' / 'gp_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:1], dtype=torch.float32, device=device))
    torques.append(torch.zeros(1, 1500, 3, device=device))
    with h5py.File(project_root / 'data' / 'training_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:1], dtype=torch.float32, device=device))
    torques = torch.cat(torques, dim=0)[:NUM_SAMPLES, :TRAJECTORY_LENGTH, :]

    print(f"\nTesting with {NUM_SAMPLES} trajectories, length {TRAJECTORY_LENGTH}")
    print("="*70)

    # Test 1: Baseline (no guidance)
    print("\n[1] Baseline (no guidance)")
    torch.manual_seed(42)
    states_base, torques_out = model.sample_trajectories(
        num_samples=NUM_SAMPLES,
        trajectory_length=TRAJECTORY_LENGTH,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        context_fraction=0.2,
        use_ema=False,
        sampler='ddim',
        torque=torques,
        hnn=None,
        guidance_steps=0,
    )
    hamres_base, nmse_mean_base, nmse_med_base = evaluate_trajectories(
        states_base, torques_out, hnn, mj_model, var_dq, var_dp
    )
    print(f"  HamRes median: {hamres_base:.4f}, NMSE mean: {nmse_mean_base:.4f}, NMSE median: {nmse_med_base:.4f}")

    # Test 2: HamRes guidance (current method)
    print("\n[2] HamRes guidance (50 steps, lr=0.001, last 10 diffusion steps)")
    torch.manual_seed(42)
    states_hamres, torques_out = model.sample_trajectories(
        num_samples=NUM_SAMPLES,
        trajectory_length=TRAJECTORY_LENGTH,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        context_fraction=0.2,
        use_ema=False,
        sampler='ddim',
        torque=torques,
        hnn=hnn,
        guidance_method='adam',
        guidance_steps=50,
        guidance_lr=0.001,
        guidance_after_steps=40,
    )
    hamres_hnn, nmse_mean_hnn, nmse_med_hnn = evaluate_trajectories(
        states_hamres, torques_out, hnn, mj_model, var_dq, var_dp
    )
    print(f"  HamRes median: {hamres_hnn:.4f}, NMSE mean: {nmse_mean_hnn:.4f}, NMSE median: {nmse_med_hnn:.4f}")

    # Test 3: Energy guidance (post-hoc optimization on baseline)
    print("\n[3] Energy guidance (50 steps, lr=0.001, post-hoc on baseline)")
    states_energy = run_energy_optimization(
        states_base.clone(), torques, qpos_dim=3, mom_dim=3, dt=DATA_DT,
        hnn=hnn, num_steps=50, lr=0.001
    )
    hamres_energy, nmse_mean_energy, nmse_med_energy = evaluate_trajectories(
        states_energy, torques_out, hnn, mj_model, var_dq, var_dp
    )
    print(f"  HamRes median: {hamres_energy:.4f}, NMSE mean: {nmse_mean_energy:.4f}, NMSE median: {nmse_med_energy:.4f}")

    # Test 4: Energy guidance with more steps
    print("\n[4] Energy guidance (200 steps, lr=0.01)")
    states_energy2 = run_energy_optimization(
        states_base.clone(), torques, qpos_dim=3, mom_dim=3, dt=DATA_DT,
        hnn=hnn, num_steps=200, lr=0.01
    )
    hamres_energy2, nmse_mean_energy2, nmse_med_energy2 = evaluate_trajectories(
        states_energy2, torques_out, hnn, mj_model, var_dq, var_dp
    )
    print(f"  HamRes median: {hamres_energy2:.4f}, NMSE mean: {nmse_mean_energy2:.4f}, NMSE median: {nmse_med_energy2:.4f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Method':<35} {'HamRes Med':>12} {'NMSE Med':>12} {'NMSE Mean':>12}")
    print("-"*70)
    print(f"{'Baseline':<35} {hamres_base:>12.4f} {nmse_med_base:>12.4f} {nmse_mean_base:>12.4f}")
    print(f"{'HamRes guidance':<35} {hamres_hnn:>12.4f} {nmse_med_hnn:>12.4f} {nmse_mean_hnn:>12.4f}")
    print(f"{'Energy guidance (50 steps)':<35} {hamres_energy:>12.4f} {nmse_med_energy:>12.4f} {nmse_mean_energy:>12.4f}")
    print(f"{'Energy guidance (200 steps)':<35} {hamres_energy2:>12.4f} {nmse_med_energy2:>12.4f} {nmse_mean_energy2:>12.4f}")
    print("="*70)


if __name__ == '__main__':
    main()
