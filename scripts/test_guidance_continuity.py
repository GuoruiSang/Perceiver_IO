"""
Test guidance methods for continuity:
1. adam (per-step) - may cause discontinuity
2. adam_integration (chunked) - enforces continuity
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import h5py
import mujoco
import torch

from src.models.utils import reconstruct_traj_with_momentum, EMA
from src.models.HNN import HNNWrapper
from src.models.trajectory_dpf import TrajectoryDPF

# Config
DT = 0.0001
DATA_DT = 0.0002
TRAJECTORY_LENGTH = 500
NUM_DIFFUSION_STEPS = 50
NUM_SAMPLES = 20
SEED = 228
DEVICE = 'cuda:3'

# Paths
MUJOCO_XML = project_root / 'configs' / 'rigid_arm_hinge.xml'
HNN_CHECKPOINT = project_root / 'checkpoints' / 'SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt'
DPF_CHECKPOINT = project_root / 'checkpoints' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt'

# Test configs
TEST_CONFIGS = [
    ('adam', 50, 0.001, 40, 15, "adam (per-step)"),
    ('adam_integration', 50, 0.001, 40, 15, "adam_integration chunk=15"),
    ('adam_integration', 50, 0.001, 40, 30, "adam_integration chunk=30"),
    ('adam_integration', 50, 0.001, 40, 50, "adam_integration chunk=50"),
]


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


def compute_nmse(gen, rec):
    min_len = min(len(gen) - 1, len(rec))
    g, r = gen[1:min_len+1], rec[:min_len]
    return np.mean((g - r)**2) / (np.var(g) + 1e-8)


def compute_smoothness(traj):
    """Compute smoothness metric: mean squared second derivative (acceleration)"""
    # Second order finite difference: traj[t+1] - 2*traj[t] + traj[t-1]
    d2 = traj[2:] - 2*traj[1:-1] + traj[:-2]
    return np.mean(d2**2)


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
        torques.append(torch.tensor(f['torques'][:5], dtype=torch.float32, device=device))
    with h5py.File(project_root / 'data' / 'gp_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:5], dtype=torch.float32, device=device))
    torques.append(torch.zeros(5, 1500, 3, device=device))
    with h5py.File(project_root / 'data' / 'training_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:5], dtype=torch.float32, device=device))
    torques = torch.cat(torques, dim=0)
    print(f"  {torques.shape[0]} samples")

    # Test each config
    print("\n" + "="*80)
    print(f"{'Method':<30} {'HamRes':>10} {'NMSE':>10} {'Smoothness':>12}")
    print("="*80)

    for method, steps, lr, after, chunk, desc in TEST_CONFIGS:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        traj_torques = torques[:NUM_SAMPLES, :TRAJECTORY_LENGTH, :]

        kwargs = dict(
            num_samples=NUM_SAMPLES,
            trajectory_length=TRAJECTORY_LENGTH,
            num_diffusion_steps=NUM_DIFFUSION_STEPS,
            context_fraction=0.2,
            use_ema=False,
            sampler='ddim',
            guidance_scale=1.0,
            torque=traj_torques,
            hnn=hnn,
            guidance_method=method,
            guidance_steps=steps,
            guidance_lr=lr,
            guidance_after_steps=after,
            lambda_init=0.0,
        )
        if method == 'adam_integration':
            kwargs['chunk_length'] = chunk

        states, torques_out = model.sample_trajectories(**kwargs)

        states = states.cpu().numpy()
        torques_out = torques_out.cpu().numpy()

        hamres_list, nmse_list, smooth_list = [], [], []
        for i in range(NUM_SAMPLES):
            qpos_gen = states[i, :, :3]
            mom_gen = states[i, :, 3:]
            torque = torques_out[i]

            M = np.zeros((mj_model.nv, mj_model.nv))
            data = mujoco.MjData(mj_model)
            data.qpos[:] = qpos_gen[0]
            mujoco.mj_forward(mj_model, data)
            mujoco.mj_fullM(mj_model, M, data.qM)
            initial_qvel = np.linalg.solve(M, mom_gen[0])

            recon = reconstruct_traj_with_momentum(
                mj_model, TRAJECTORY_LENGTH, DT, qpos_gen[0], initial_qvel, torque, data_dt=DATA_DT
            )

            hamres_list.append(compute_hamres(qpos_gen, mom_gen, torque, hnn, var_dq, var_dp))
            nmse_list.append(compute_nmse(qpos_gen, recon['seq_qpos']) + compute_nmse(mom_gen, recon['seq_mom']))
            smooth_list.append(compute_smoothness(qpos_gen) + compute_smoothness(mom_gen))

        print(f"{desc:<30} {np.nanmedian(hamres_list):>10.4f} {np.mean(nmse_list):>10.4f} {np.mean(smooth_list):>12.6f}")

    print("="*80)


if __name__ == '__main__':
    main()
