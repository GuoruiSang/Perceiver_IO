"""
Search for guidance configs that achieve NMSE median < 0.001.

Current best NMSE median: ~0.037, need ~37x improvement.
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
NUM_DIFFUSION_STEPS = 50
NUM_SAMPLES = 5  # 5 mixed torques
SEED = 228
DEVICE = 'cuda:3'

# Paths
MUJOCO_XML = project_root / 'configs' / 'rigid_arm_hinge.xml'
HNN_CHECKPOINT = project_root / 'checkpoints' / 'SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt'
DPF_CHECKPOINT = project_root / 'checkpoints' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt'

# Search space - more aggressive guidance
CONFIGS = [
    # (guidance_steps, guidance_lr, after, before, lambda_init, desc)
    # Baseline
    (0, 0.001, 0, 0, 0.0, "Baseline"),
    # Current best
    (50, 0.001, 40, 9999, 0.0, "50 steps, lr=0.001, 40-50"),
    # More steps
    (100, 0.001, 40, 9999, 0.0, "100 steps, lr=0.001, 40-50"),
    (200, 0.001, 40, 9999, 0.0, "200 steps, lr=0.001, 40-50"),
    # Full range guidance (all diffusion steps)
    (50, 0.001, 0, 9999, 0.0, "50 steps, lr=0.001, full"),
    (100, 0.001, 0, 9999, 0.0, "100 steps, lr=0.001, full"),
    (200, 0.001, 0, 9999, 0.0, "200 steps, lr=0.001, full"),
    # Higher learning rate
    (50, 0.01, 40, 9999, 0.0, "50 steps, lr=0.01, 40-50"),
    (100, 0.01, 40, 9999, 0.0, "100 steps, lr=0.01, 40-50"),
    (50, 0.01, 0, 9999, 0.0, "50 steps, lr=0.01, full"),
    (100, 0.01, 0, 9999, 0.0, "100 steps, lr=0.01, full"),
    # With lambda_init (enforce initial conditions)
    (100, 0.001, 0, 9999, 1.0, "100 steps, full, lambda_init=1"),
    (100, 0.01, 0, 9999, 1.0, "100 steps, full, lr=0.01, lambda=1"),
    (200, 0.01, 0, 9999, 1.0, "200 steps, full, lr=0.01, lambda=1"),
]


def compute_nmse(gen, rec):
    min_len = min(len(gen) - 1, len(rec))
    g, r = gen[1:min_len+1], rec[:min_len]
    return np.mean((g - r)**2) / (np.var(g) + 1e-8)


def test_config(model, hnn, mj_model, torques, traj_len, gsteps, glr, after, before, lambda_init):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    traj_torques = torques[:NUM_SAMPLES, :traj_len, :]

    states, torques_out = model.sample_trajectories(
        num_samples=NUM_SAMPLES,
        trajectory_length=traj_len,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        context_fraction=0.2,
        use_ema=False,
        sampler='ddim',
        guidance_scale=1.0,
        torque=traj_torques,
        hnn=hnn,
        guidance_method='adam',
        guidance_steps=gsteps,
        guidance_lr=glr,
        guidance_after_steps=after,
        guidance_before_steps=before,
        lambda_init=lambda_init,
    )

    states = states.cpu().numpy()
    torques_out = torques_out.cpu().numpy()

    nmse_list = []
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
            mj_model, traj_len, DT, qpos_gen[0], initial_qvel, torque, data_dt=DATA_DT
        )

        nmse_list.append(compute_nmse(qpos_gen, recon['seq_qpos']) + compute_nmse(mom_gen, recon['seq_mom']))

    return np.median(nmse_list), np.mean(nmse_list)


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
    mj_model = mujoco.MjModel.from_xml_path(str(MUJOCO_XML))

    # Load 5 mixed torques
    print("Loading torques...")
    torques = []
    with h5py.File(project_root / 'data' / 'sinusoidal_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:2], dtype=torch.float32, device=device))
    with h5py.File(project_root / 'data' / 'gp_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:1], dtype=torch.float32, device=device))
    torques.append(torch.zeros(1, 1500, 3, device=device))
    with h5py.File(project_root / 'data' / 'training_torques_1000_L1500.h5', 'r') as f:
        torques.append(torch.tensor(f['torques'][:1], dtype=torch.float32, device=device))
    torques = torch.cat(torques, dim=0)
    print(f"  {torques.shape[0]} mixed torques (2 sin, 1 gp, 1 zero, 1 spline)")

    # Test lengths
    lengths = [250, 500, 1000]

    print("\n" + "="*100)
    print(f"{'Config':<40} {'L=250 Med':>12} {'L=500 Med':>12} {'L=1000 Med':>12}")
    print("="*100)

    for gsteps, glr, after, before, lambda_init, desc in CONFIGS:
        results = []
        for traj_len in lengths:
            med, mean = test_config(model, hnn, mj_model, torques, traj_len, gsteps, glr, after, before, lambda_init)
            results.append(med)

        # Highlight if any config achieves < 0.01 (closer to target)
        marker = " *" if all(r < 0.01 for r in results) else ""
        print(f"{desc:<40} {results[0]:>12.4f} {results[1]:>12.4f} {results[2]:>12.4f}{marker}")

    print("="*100)
    print("* = NMSE median < 0.01 across all lengths")


if __name__ == '__main__':
    main()
