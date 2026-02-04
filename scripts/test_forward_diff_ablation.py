"""
Ablation: Forward Difference + Proximal Term for HNN Guidance.

Based on GPT's suggestions:
1. Replace central difference with forward difference in HamRes guidance
2. Sweep lambda_prox to control trajectory drift

Test groups:
- Baseline: No guidance
- G0: Original (central diff, lambda=1.0)
- G1: Forward diff only (lambda=0)
- G2a-d: Forward diff + varying lambda_prox (1e-4, 1e-3, 1e-2, 1e-1)

Config: T=1000, 30 samples, DDIM 50 steps, CFG=1.0, guidance in last 5 steps
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
TRAJECTORY_LENGTH = 1000
NUM_DIFFUSION_STEPS = 50
NUM_SAMPLES = 30
SEED = 228
DEVICE = 'cuda:3'

# Paths
MUJOCO_XML = project_root / 'configs' / 'rigid_arm_hinge.xml'
HNN_CHECKPOINT = project_root / 'checkpoints' / 'SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt'
DPF_CHECKPOINT = project_root / 'checkpoints' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt'

# Guidance config: last 5 steps (45-50 out of 50)
GUIDANCE_AFTER_STEPS = 45
GUIDANCE_STEPS = 50  # Adam steps per diffusion step
GUIDANCE_LR = 0.001


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


def evaluate(states, torques_out, hnn, mj_model, var_dq, var_dp, traj_len):
    """Evaluate trajectories and return NMSE/HamRes arrays."""
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

    return np.array(hamres_list), np.array(nmse_list)


def run_config(model, hnn, torques, traj_len, use_guidance=True, use_forward_diff=False, lambda_init=1.0, normalize_energy=False):
    """Run a single configuration."""
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
            guidance_scale=1.0,  # No CFG
            torque=traj_torques,
            hnn=hnn,
            guidance_method='adam',
            guidance_after_steps=GUIDANCE_AFTER_STEPS,
            guidance_steps=GUIDANCE_STEPS,
            guidance_lr=GUIDANCE_LR,
            lambda_init=lambda_init,
            use_forward_diff=use_forward_diff,
            normalize_energy=normalize_energy,
        )
    else:
        states, torques_out = model.sample_trajectories(
            num_samples=NUM_SAMPLES,
            trajectory_length=traj_len,
            num_diffusion_steps=NUM_DIFFUSION_STEPS,
            context_fraction=0.2,
            use_ema=False,
            sampler='ddim',
            guidance_scale=1.0,
            torque=traj_torques,
            hnn=None,
            guidance_steps=0,
        )

    return states, torques_out


def main():
    device = torch.device(DEVICE)
    print(f"Device: {device}, Seed: {SEED}, Samples: {NUM_SAMPLES}, Length: {TRAJECTORY_LENGTH}")

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
    print(f"  {torques.shape[0]} mixed torques available")

    # Define test configurations
    # normalize=False: raw MSE, normalize=True: NMSE (divided by variance)
    configs = [
        {"name": "Baseline", "use_guidance": False, "use_forward_diff": False, "lambda_init": 0, "normalize": False},
        # Without normalization (original behavior)
        {"name": "G0 (central, λ=1, raw)", "use_guidance": True, "use_forward_diff": False, "lambda_init": 1.0, "normalize": False},
        {"name": "G1 (forward, λ=0, raw)", "use_guidance": True, "use_forward_diff": True, "lambda_init": 0, "normalize": False},
        # With normalization (balanced e1/e2/e3)
        {"name": "N0 (central, λ=1, norm)", "use_guidance": True, "use_forward_diff": False, "lambda_init": 1.0, "normalize": True},
        {"name": "N1 (forward, λ=0, norm)", "use_guidance": True, "use_forward_diff": True, "lambda_init": 0, "normalize": True},
        {"name": "N2a (forward, λ=0.1, norm)", "use_guidance": True, "use_forward_diff": True, "lambda_init": 0.1, "normalize": True},
        {"name": "N2b (forward, λ=1.0, norm)", "use_guidance": True, "use_forward_diff": True, "lambda_init": 1.0, "normalize": True},
        {"name": "N2c (forward, λ=10, norm)", "use_guidance": True, "use_forward_diff": True, "lambda_init": 10.0, "normalize": True},
    ]

    results = []

    for cfg in configs:
        print(f"\nTesting: {cfg['name']}...")
        states, torques_out = run_config(
            model, hnn, torques, TRAJECTORY_LENGTH,
            use_guidance=cfg["use_guidance"],
            use_forward_diff=cfg["use_forward_diff"],
            lambda_init=cfg["lambda_init"],
            normalize_energy=cfg["normalize"]
        )
        hamres_arr, nmse_arr = evaluate(states, torques_out, hnn, mj_model, var_dq, var_dp, TRAJECTORY_LENGTH)

        nmse_med = np.nanmedian(nmse_arr)
        nmse_p95 = np.nanpercentile(nmse_arr, 95)
        hamres_p99 = np.nanpercentile(hamres_arr, 99)

        results.append({
            "name": cfg["name"],
            "nmse_med": nmse_med,
            "nmse_p95": nmse_p95,
            "hamres_p99": hamres_p99,
        })
        print(f"  NMSE med={nmse_med:.4f}, P95={nmse_p95:.4f}, HamRes P99={hamres_p99:.4f}")

    # Print summary table
    print("\n" + "="*110)
    print(f"ABLATION: Forward Difference + Normalization @ T={TRAJECTORY_LENGTH}")
    print("="*110)
    print(f"{'Group':<28} {'FwdDiff':<8} {'Norm':<6} {'λ':<8} {'NMSE Med':>10} {'NMSE P95':>10} {'HamRes P99':>12} {'Change':>12}")
    print("-"*110)

    baseline_nmse = results[0]["nmse_med"]
    for i, cfg in enumerate(configs):
        r = results[i]
        fd_str = "-" if not cfg["use_guidance"] else ("Yes" if cfg["use_forward_diff"] else "No")
        norm_str = "-" if not cfg["use_guidance"] else ("Yes" if cfg["normalize"] else "No")
        lam_str = "-" if not cfg["use_guidance"] else f"{cfg['lambda_init']}"
        change = ((r["nmse_med"] - baseline_nmse) / baseline_nmse * 100) if i > 0 else 0
        change_str = f"{change:+.1f}%" if i > 0 else ""
        print(f"{r['name']:<28} {fd_str:<8} {norm_str:<6} {lam_str:<8} {r['nmse_med']:>10.4f} {r['nmse_p95']:>10.4f} {r['hamres_p99']:>12.4f} {change_str:>12}")

    print("="*110)

    # Highlight best config
    guided_results = results[1:]  # Exclude baseline
    best_idx = np.argmin([r["nmse_med"] for r in guided_results])
    best = guided_results[best_idx]
    print(f"\nBest guided config: {best['name']}")
    print(f"  NMSE median: {best['nmse_med']:.4f} (vs baseline {baseline_nmse:.4f})")
    improvement = (baseline_nmse - best["nmse_med"]) / baseline_nmse * 100
    print(f"  Improvement: {improvement:+.1f}%")


if __name__ == '__main__':
    main()
