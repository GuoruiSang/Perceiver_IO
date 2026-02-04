"""
Rigorous test: Compare HamRes/Energy guidance vs Baseline with controlled variables.
- Same seed (228)
- Multiple lengths (500, 1000)
- More samples (20)
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
NUM_DIFFUSION_STEPS = 50
NUM_SAMPLES = 20
SEED = 228
DEVICE = 'cuda:3'

# Paths
MUJOCO_XML = project_root / 'configs' / 'rigid_arm_hinge.xml'
HNN_CHECKPOINT = project_root / 'checkpoints' / 'SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt'
DPF_CHECKPOINT = project_root / 'checkpoints' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt'


def compute_nmse(gen, rec):
    min_len = min(len(gen) - 1, len(rec))
    g, r = gen[1:min_len+1], rec[:min_len]
    return np.mean((g - r)**2) / (np.var(g) + 1e-8)


def compute_energy_loss(seq_qpos, seq_mom, seq_torque, hnn, dt):
    """Energy conservation loss."""
    E0 = hnn(seq_mom[:, 0, :], seq_qpos[:, 0, :])
    ET = hnn(seq_mom[:, -1, :], seq_qpos[:, -1, :])
    qdot = (seq_qpos[:, 2:, :] - seq_qpos[:, :-2, :]) / (2 * dt)
    tau_mid = seq_torque[:, 1:-1, :]
    work = (tau_mid * qdot).sum(dim=(1, 2)) * dt
    delta_E = ET - E0
    return (delta_E - work).pow(2).mean()


def run_energy_optimization(x, seq_torque, qpos_dim, mom_dim, dt, hnn, num_steps, lr=0.001):
    """Post-hoc energy optimization."""
    seq_qpos = nn.Parameter(x[:, :, :qpos_dim].clone())
    seq_mom = nn.Parameter(x[:, :, qpos_dim:qpos_dim + mom_dim].clone())
    optimizer = torch.optim.Adam([seq_qpos, seq_mom], lr=lr)
    for _ in trange(num_steps, desc='Energy Opt', leave=False):
        optimizer.zero_grad()
        loss = compute_energy_loss(seq_qpos, seq_mom, seq_torque, hnn, dt)
        loss.backward()
        optimizer.step()
    return torch.cat([seq_qpos.data, seq_mom.data], dim=-1)


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


def evaluate(states, torques_out, hnn, mj_model, var_dq, var_dp, traj_len):
    """Evaluate trajectories."""
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
            mj_model, traj_len, DT, qpos_gen[0], initial_qvel, torque, data_dt=DATA_DT
        )

        hamres_list.append(compute_hamres(qpos_gen, mom_gen, torque, hnn, var_dq, var_dp))
        nmse_list.append(compute_nmse(qpos_gen, recon['seq_qpos']) + compute_nmse(mom_gen, recon['seq_mom']))

    return np.median(hamres_list), np.mean(nmse_list), np.median(nmse_list)


def test_config(model, hnn, mj_model, torques, traj_len, var_dq, var_dp, use_guidance):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    traj_torques = torques[:NUM_SAMPLES, :traj_len, :]

    if use_guidance:
        states, torques_out = model.sample_trajectories(
            num_samples=NUM_SAMPLES,
            trajectory_length=traj_len,
            num_diffusion_steps=NUM_DIFFUSION_STEPS,
            context_fraction=0.2,
            use_ema=False,
            sampler='ddim',
            torque=traj_torques,
            hnn=hnn,
            guidance_method='adam',
            guidance_steps=50,
            guidance_lr=0.001,
            guidance_after_steps=40,
        )
    else:
        states, torques_out = model.sample_trajectories(
            num_samples=NUM_SAMPLES,
            trajectory_length=traj_len,
            num_diffusion_steps=NUM_DIFFUSION_STEPS,
            context_fraction=0.2,
            use_ema=False,
            sampler='ddim',
            torque=traj_torques,
            hnn=None,
            guidance_steps=0,
        )

    states_np = states.cpu().numpy()
    torques_np = torques_out.cpu().numpy()

    hamres_list, nmse_list = [], []
    for i in range(NUM_SAMPLES):
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

        hamres_list.append(compute_hamres(qpos_gen, mom_gen, torque, hnn, var_dq, var_dp))
        nmse_list.append(compute_nmse(qpos_gen, recon['seq_qpos']) + compute_nmse(mom_gen, recon['seq_mom']))

    return np.median(hamres_list), np.mean(nmse_list), np.median(nmse_list)


def main():
    device = torch.device(DEVICE)
    print(f"Device: {device}, Seed: {SEED}, Samples: {NUM_SAMPLES}")

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
        torques.append(torch.tensor(f['torques'][:5], dtype=torch.float32, device=device))
    with h5py.File(project_root / 'data' / 'gp_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:5], dtype=torch.float32, device=device))
    torques.append(torch.zeros(5, 1500, 3, device=device))
    with h5py.File(project_root / 'data' / 'training_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:5], dtype=torch.float32, device=device))
    torques = torch.cat(torques, dim=0)
    print(f"  {torques.shape[0]} mixed torques")

    # Test
    lengths = [500, 1000]

    print("\n" + "="*90)
    print(f"{'Length':<10} {'Method':<20} {'HamRes Med':>12} {'NMSE Med':>12} {'NMSE Mean':>12}")
    print("="*90)

    for traj_len in lengths:
        # Baseline
        print(f"Testing L={traj_len} baseline...", end=" ", flush=True)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        traj_torques = torques[:NUM_SAMPLES, :traj_len, :]
        states_base, torques_out = model.sample_trajectories(
            num_samples=NUM_SAMPLES,
            trajectory_length=traj_len,
            num_diffusion_steps=NUM_DIFFUSION_STEPS,
            context_fraction=0.2,
            use_ema=False,
            sampler='ddim',
            torque=traj_torques,
            hnn=None,
            guidance_steps=0,
        )
        h_base, nmse_mean_base, nmse_med_base = evaluate(states_base, torques_out, hnn, mj_model, var_dq, var_dp, traj_len)
        print("done")

        # HamRes guidance
        print(f"Testing L={traj_len} HamRes...", end=" ", flush=True)
        h_hamres, nmse_mean_hamres, nmse_med_hamres = test_config(
            model, hnn, mj_model, torques, traj_len, var_dq, var_dp, use_guidance=True
        )
        print("done")

        # Energy guidance (post-hoc on baseline)
        print(f"Testing L={traj_len} Energy...", end=" ", flush=True)
        states_energy = run_energy_optimization(
            states_base.clone(), traj_torques, qpos_dim=3, mom_dim=3, dt=DATA_DT,
            hnn=hnn, num_steps=50, lr=0.001
        )
        h_energy, nmse_mean_energy, nmse_med_energy = evaluate(states_energy, torques_out, hnn, mj_model, var_dq, var_dp, traj_len)
        print("done")

        print(f"{traj_len:<10} {'Baseline':<20} {h_base:>12.4f} {nmse_med_base:>12.4f} {nmse_mean_base:>12.4f}")
        print(f"{'':<10} {'HamRes guidance':<20} {h_hamres:>12.4f} {nmse_med_hamres:>12.4f} {nmse_mean_hamres:>12.4f}")
        print(f"{'':<10} {'Energy guidance':<20} {h_energy:>12.4f} {nmse_med_energy:>12.4f} {nmse_mean_energy:>12.4f}")

        # Comparison
        hamres_change = (nmse_med_hamres - nmse_med_base) / nmse_med_base * 100
        energy_change = (nmse_med_energy - nmse_med_base) / nmse_med_base * 100
        print(f"{'':<10} {'HamRes vs Base':<20} {'':<12} {hamres_change:>+11.1f}%")
        print(f"{'':<10} {'Energy vs Base':<20} {'':<12} {energy_change:>+11.1f}%")
        print("-"*90)

    print("="*90)


if __name__ == '__main__':
    main()
