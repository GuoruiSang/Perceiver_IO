"""
Quick single-config evaluation: generate unguided vs guided trajectories,
compare NMSE and HamRes.

改参数直接改下面 CONFIG 区域，然后点运行即可。
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

import os
import time
import numpy as np
import torch
import mujoco
import matplotlib.pyplot as plt

from src.models.trajectory_dpf import TrajectoryDPF
from src.models.HNN import HNNWrapper
from src.models.utils import EMA, reconstruct_traj_with_momentum

from compute_ablation_2dof_with_smoothing import (
    SYSTEM_CONFIGS, DT, SIM_DT,
    BATCH_SIZE_UNGUIDED, BATCH_SIZE_GUIDED,
    HAMRES_SMOOTH_SIGMA, HAMRES_PSEUDO_HUBER_DELTA,
    HAMRES_MIN_SCALE_Q, HAMRES_MIN_SCALE_P,
    load_torques, compute_metrics_for_samples,
)

# ============================================================
# CONFIG — 改这里，然后直接运行脚本
# ============================================================
SYSTEM          = '3dof'        # '2dof' or '3dof'
POLICY          = 'sinusoidal'  # 'sinusoidal', 'gp', 'zero', 'spline'
LENGTH          = 1000           # 轨迹长度
NUM_SAMPLES     = 4            # 样本数 (小值快速迭代)
SEED            = 10           # 随机种子
DEVICE          = 'cuda:4'      # GPU 设备

# 平滑
SMOOTH_SIGMA          = 0           # 高斯平滑 sigma (0=关闭)
SMOOTH_GUIDANCE_ONLY  = False       # True=只在guidance能量计算时平滑，输出不平滑
SMOOTH_LAST_STEP_ONLY = False       # True=只在最后一步扩散时平滑

# Guidance (None = 用 SYSTEM_CONFIGS 里的系统默认值)
GUIDANCE_PRESET = 'one_step_best'
# 预设:
#   one_step_best     -> one-step guidance 最佳配置（same-budget search, 3DoF）
#   robust_r3_sigma05 -> robust_hamres 推荐配置
#   robust_r9_comboA  -> robust_hamres 更激进配置
#   auto              -> 按系统自动选择（2dof: robust_r9_comboA, 3dof: one_step_best）
#   custom            -> 使用下方自定义参数
GUIDANCE_METHOD    = 'normalized_sgd'  # 仅 custom 使用
OPTIMIZE_TARGET    = 'both'            # 仅 custom 使用
GUIDANCE_ENERGY_MODE = 'one_step'      # 仅 custom 使用
GUIDANCE_STEPS  = 100                  # 仅 custom 使用
GUIDANCE_LR     = 0.01                 # 仅 custom 使用
GUIDANCE_AFTER  = 15                   # 仅 custom 使用
GUIDANCE_BEFORE = 20                   # 仅 custom 使用
# Trust region (防止 guidance 偏离 DPF 先验过远)
GUIDANCE_TRUST_LAMBDA = 0.0            # 仅 custom 使用
# robust_hamres guidance 能量参数
GUIDANCE_HAMRES_SMOOTH_SIGMA = 1.0     # 仅 custom 使用
GUIDANCE_HAMRES_DELTA = 1.0            # 仅 custom 使用
GUIDANCE_HAMRES_MIN_SCALE_Q = 1e-3     # 仅 custom 使用
GUIDANCE_HAMRES_MIN_SCALE_P = 1e-3     # 仅 custom 使用
# Langevin 专用
LANGEVIN_STEP_SIZE   = 1e-6    # Langevin 步长
LANGEVIN_NOISE_SCALE = 1e-5    # Langevin 噪声尺度
# normalized_sgd 专用
ALPHA_Q         = 1e-2          # q 步长 (normalized SGD)
ALPHA_P         = 1e-2          # p 步长 (normalized SGD)
# adam_integration 专用
CHUNK_LENGTH    = 15            # 积分 chunk 长度

# 扩散
NUM_DIFFUSION_STEPS = 20      # DDIM 总步数 (None=模型默认 ~50)

# HamRes（评估）稳定性参数：None=使用 compute_ablation_2dof_with_smoothing.py 默认值
HAMRES_SMOOTH_SIGMA_CFG = None
HAMRES_DELTA_CFG        = None
HAMRES_MIN_SCALE_Q_CFG  = None
HAMRES_MIN_SCALE_P_CFG  = None
# ============================================================

GUIDANCE_PRESETS = {
    'one_step_best': dict(
        guidance_method='normalized_sgd',
        optimize_target='both',
        guidance_energy_mode='one_step',
        guidance_steps=50,
        guidance_lr=0.01,
        guidance_after=10,
        guidance_before=20,
        guidance_trust_lambda=1e-3,
        guidance_hamres_smooth_sigma=1.0,
        guidance_hamres_delta=1.0,
        guidance_hamres_min_scale_q=1e-3,
        guidance_hamres_min_scale_p=1e-3,
    ),
    'robust_r3_sigma05': dict(
        guidance_method='normalized_sgd',
        optimize_target='both',
        guidance_energy_mode='robust_hamres',
        guidance_steps=50,
        guidance_lr=0.01,
        guidance_after=15,
        guidance_before=20,
        guidance_trust_lambda=0.0,
        guidance_hamres_smooth_sigma=0.5,
        guidance_hamres_delta=1.0,
        guidance_hamres_min_scale_q=1e-3,
        guidance_hamres_min_scale_p=1e-3,
    ),
    'robust_r9_comboa': dict(
        guidance_method='normalized_sgd',
        optimize_target='both',
        guidance_energy_mode='robust_hamres',
        guidance_steps=50,
        guidance_lr=0.01,
        guidance_after=10,
        guidance_before=20,
        guidance_trust_lambda=1e-3,
        guidance_hamres_smooth_sigma=0.5,
        guidance_hamres_delta=2.0,
        guidance_hamres_min_scale_q=1e-3,
        guidance_hamres_min_scale_p=1e-3,
    ),
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_models(cfg, device):
    """Load DPF (with EMA), HNN (with variance buffers), and MuJoCo model."""
    dpf = TrajectoryDPF.load_from_checkpoint(
        str(cfg['dpf_ckpt']), map_location=device, strict=False)
    dpf.eval().to(device)

    ckpt = torch.load(str(cfg['dpf_ckpt']), map_location=device, weights_only=False)
    if 'ema_shadow' in ckpt:
        dpf.ema = EMA(dpf.model, decay=0.9995)
        for name, tensor in ckpt['ema_shadow'].items():
            if name in dpf.ema.shadow:
                dpf.ema.shadow[name] = tensor.to(
                    device=device, dtype=dpf.ema.shadow[name].dtype)
    del ckpt

    # Old checkpoints lack model_type in hparams; detect and pass explicitly
    hnn_ckpt = torch.load(str(cfg['hnn_ckpt']), map_location=device, weights_only=False)
    hnn_hparams = hnn_ckpt.get('hyper_parameters', {})
    hnn_kwargs = {}
    if 'model_type' not in hnn_hparams:
        hnn_kwargs['model_type'] = 'separable'
    del hnn_ckpt
    hnn = HNNWrapper.load_from_checkpoint(
        str(cfg['hnn_ckpt']), map_location=device, **hnn_kwargs)
    hnn.eval().to(device)
    var_dq = hnn.qvel_var.mean().to(device)
    var_dp = hnn.mom_dot_var.mean().to(device)

    mj_model = mujoco.MjModel.from_xml_path(cfg['xml_path'])
    mj_model.opt.timestep = SIM_DT

    return dpf, hnn, var_dq, var_dp, mj_model


def _reconstruct_gt(qpos, mom, torque, xml_path):
    """Run MuJoCo reconstruction for a single sample."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    # Initial velocity from momentum: v = M^{-1} @ p
    data.qpos[:] = qpos[0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    initial_qvel = np.linalg.solve(M, mom[0])
    recon = reconstruct_traj_with_momentum(
        model, len(qpos), SIM_DT,
        qpos[0], initial_qvel, torque, data_dt=DT)
    return recon  # keys: seq_qpos, seq_mom, seq_torque (length T-1)


def plot_trajectories(ung_states, ung_torques, gui_states, gui_torques,
                      ung_nq, gui_nq, qpos_dim, xml_path):
    """Plot same sample: row1=torque, row2=unguided vs GT, row3=guided vs GT."""
    # Pick sample closest to guided median NMSE_q (same index for both)
    idx = int(np.argsort(gui_nq)[len(gui_nq) // 2])

    save_dir = str(project_root / 'plots')
    os.makedirs(save_dir, exist_ok=True)

    mom_dim = ung_states.shape[-1] - qpos_dim
    torque_dim = ung_torques.shape[-1]
    ncols = qpos_dim + mom_dim  # e.g. 6 for 3DoF

    # Extract sample data (numpy)
    ung_s = ung_states[idx].detach().cpu().numpy() if torch.is_tensor(ung_states) else ung_states[idx]
    gui_s = gui_states[idx].detach().cpu().numpy() if torch.is_tensor(gui_states) else gui_states[idx]
    tau = ung_torques[idx].detach().cpu().numpy() if torch.is_tensor(ung_torques) else ung_torques[idx]

    # MuJoCo GT reconstruction (same initial state & torques → same GT for both)
    ung_recon = _reconstruct_gt(ung_s[:, :qpos_dim], ung_s[:, qpos_dim:], tau, xml_path)
    gui_recon = _reconstruct_gt(gui_s[:, :qpos_dim], gui_s[:, qpos_dim:], tau, xml_path)

    fig, axes = plt.subplots(3, ncols, figsize=(5 * ncols, 10))

    # Row 0: Torque
    for d in range(torque_dim):
        ax = axes[0, d]
        ax.plot(tau[:, d], linewidth=0.8)
        ax.set_title(f'torque[{d}]')
    for d in range(torque_dim, ncols):
        axes[0, d].set_visible(False)

    # Prepend initial state to MuJoCo reconstruction so both start from t=0
    ung_gt_qpos = np.concatenate([[ung_s[0, :qpos_dim]], ung_recon['seq_qpos']], axis=0)
    ung_gt_mom = np.concatenate([[ung_s[0, qpos_dim:]], ung_recon['seq_mom']], axis=0)
    gui_gt_qpos = np.concatenate([[gui_s[0, :qpos_dim]], gui_recon['seq_qpos']], axis=0)
    gui_gt_mom = np.concatenate([[gui_s[0, qpos_dim:]], gui_recon['seq_mom']], axis=0)

    # Row 1: Unguided vs GT (both from t=0)
    for d in range(qpos_dim):
        ax = axes[1, d]
        ax.scatter(range(len(ung_s)), ung_s[:, d], s=1, c='blue', alpha=0.7, label='Generated')
        ax.scatter(range(len(ung_gt_qpos)), ung_gt_qpos[:, d], s=1, c='red', alpha=0.7, label='MuJoCo GT')
        ax.set_title(f'unguided qpos[{d}]')
        ax.legend(markerscale=5, fontsize=7)
    for d in range(mom_dim):
        ax = axes[1, qpos_dim + d]
        ax.scatter(range(len(ung_s)), ung_s[:, qpos_dim + d], s=1, c='blue', alpha=0.7, label='Generated')
        ax.scatter(range(len(ung_gt_mom)), ung_gt_mom[:, d], s=1, c='red', alpha=0.7, label='MuJoCo GT')
        ax.set_title(f'unguided mom[{d}]')
        ax.legend(markerscale=5, fontsize=7)

    # Row 2: Guided vs GT (both from t=0)
    for d in range(qpos_dim):
        ax = axes[2, d]
        ax.scatter(range(len(gui_s)), gui_s[:, d], s=1, c='blue', alpha=0.7, label='Generated')
        ax.scatter(range(len(gui_gt_qpos)), gui_gt_qpos[:, d], s=1, c='red', alpha=0.7, label='MuJoCo GT')
        ax.set_title(f'guided qpos[{d}]')
        ax.legend(markerscale=5, fontsize=7)
    for d in range(mom_dim):
        ax = axes[2, qpos_dim + d]
        ax.scatter(range(len(gui_s)), gui_s[:, qpos_dim + d], s=1, c='blue', alpha=0.7, label='Generated')
        ax.scatter(range(len(gui_gt_mom)), gui_gt_mom[:, d], s=1, c='red', alpha=0.7, label='MuJoCo GT')
        ax.set_title(f'guided mom[{d}]')
        ax.legend(markerscale=5, fontsize=7)

    # Sync y-axis between unguided (row1) and guided (row2) for each column
    for col in range(ncols):
        ymin = min(axes[1, col].get_ylim()[0], axes[2, col].get_ylim()[0])
        ymax = max(axes[1, col].get_ylim()[1], axes[2, col].get_ylim()[1])
        axes[1, col].set_ylim(ymin, ymax)
        axes[2, col].set_ylim(ymin, ymax)

    fig.suptitle(
        f'Sample {idx}  |  NMSE_q: ung={ung_nq[idx]:.6f}, gui={gui_nq[idx]:.6f}',
        fontsize=14, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(save_dir, f'quick_eval_{SYSTEM}.jpg')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Plot (sample {idx}): NMSE_q ung={ung_nq[idx]:.6f}, gui={gui_nq[idx]:.6f}")
    print(f"  -> {out_path}")


def run_batch(dpf, torques, noise, L, batch_size, **kwargs):
    """Generate trajectories in batches with full kwarg passthrough."""
    num_samples = torques.shape[0]
    all_states, all_torques = [], []
    for i in range(0, num_samples, batch_size):
        j = min(i + batch_size, num_samples)
        bt = torques[i:j]
        bn = noise[i:j] if noise is not None else None
        state, tau_out = dpf.sample_trajectories(
            num_samples=bt.shape[0],
            trajectory_length=L,
            context_fraction=0.5,
            use_ema=True,
            torque=bt,
            initial_noise=bn,
            **kwargs,
        )
        all_states.append(state)
        all_torques.append(tau_out)
    return torch.cat(all_states, dim=0), torch.cat(all_torques, dim=0)


def main():
    # Resolve defaults from system config
    cfg = SYSTEM_CONFIGS[SYSTEM]
    preset_key = GUIDANCE_PRESET.lower()
    if preset_key == 'auto':
        preset_key = 'robust_r9_comboa' if SYSTEM == '2dof' else 'one_step_best'
    if preset_key == 'custom':
        guidance_method = GUIDANCE_METHOD
        optimize_target = OPTIMIZE_TARGET
        guidance_energy_mode = GUIDANCE_ENERGY_MODE
        guidance_steps = GUIDANCE_STEPS if GUIDANCE_STEPS is not None else cfg['guidance_steps']
        guidance_lr = GUIDANCE_LR if GUIDANCE_LR is not None else cfg['guidance_lr']
        guidance_after = GUIDANCE_AFTER if GUIDANCE_AFTER is not None else cfg['guidance_after_steps']
        guidance_before = GUIDANCE_BEFORE
        guidance_trust_lambda = GUIDANCE_TRUST_LAMBDA
        guidance_hamres_smooth_sigma = GUIDANCE_HAMRES_SMOOTH_SIGMA
        guidance_hamres_delta = GUIDANCE_HAMRES_DELTA
        guidance_hamres_min_scale_q = GUIDANCE_HAMRES_MIN_SCALE_Q
        guidance_hamres_min_scale_p = GUIDANCE_HAMRES_MIN_SCALE_P
    else:
        if preset_key not in GUIDANCE_PRESETS:
            raise ValueError(
                f"Unknown GUIDANCE_PRESET={GUIDANCE_PRESET!r}. "
                f"Choose from {list(GUIDANCE_PRESETS.keys()) + ['custom']}"
            )
        p = GUIDANCE_PRESETS[preset_key]
        guidance_method = p['guidance_method']
        optimize_target = p['optimize_target']
        guidance_energy_mode = p['guidance_energy_mode']
        guidance_steps = p['guidance_steps']
        guidance_lr = p['guidance_lr']
        guidance_after = p['guidance_after']
        guidance_before = p['guidance_before']
        guidance_trust_lambda = p['guidance_trust_lambda']
        guidance_hamres_smooth_sigma = p['guidance_hamres_smooth_sigma']
        guidance_hamres_delta = p['guidance_hamres_delta']
        guidance_hamres_min_scale_q = p['guidance_hamres_min_scale_q']
        guidance_hamres_min_scale_p = p['guidance_hamres_min_scale_p']
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    hamres_kwargs = dict(
        smooth_sigma=HAMRES_SMOOTH_SIGMA if HAMRES_SMOOTH_SIGMA_CFG is None else HAMRES_SMOOTH_SIGMA_CFG,
        delta=HAMRES_PSEUDO_HUBER_DELTA if HAMRES_DELTA_CFG is None else HAMRES_DELTA_CFG,
        min_scale_q=HAMRES_MIN_SCALE_Q if HAMRES_MIN_SCALE_Q_CFG is None else HAMRES_MIN_SCALE_Q_CFG,
        min_scale_p=HAMRES_MIN_SCALE_P if HAMRES_MIN_SCALE_P_CFG is None else HAMRES_MIN_SCALE_P_CFG,
    )

    # Print config
    print(f"{'='*64}")
    print(f"Quick Eval: {SYSTEM} | {POLICY} | L={LENGTH} | N={NUM_SAMPLES} | seed={SEED}")
    print(f"{'-'*64}")
    if GUIDANCE_PRESET.lower() == 'auto':
        print(f"Guidance preset: auto -> {preset_key}")
    else:
        print(f"Guidance preset: {GUIDANCE_PRESET}")
    print(f"Guidance: method={guidance_method}  energy={guidance_energy_mode}  target={optimize_target}  steps={guidance_steps}  lr={guidance_lr}  after={guidance_after}  before={guidance_before}  trust_lambda={guidance_trust_lambda}")
    if guidance_method == 'langevin':
        print(f"  Langevin: step_size={LANGEVIN_STEP_SIZE}  noise_scale={LANGEVIN_NOISE_SCALE}")
    elif guidance_method == 'normalized_sgd':
        print(f"  Normalized SGD: alpha_q={ALPHA_Q}  alpha_p={ALPHA_P}")
    elif guidance_method == 'adam_integration':
        print(f"  Integration: chunk_length={CHUNK_LENGTH}")
    if guidance_energy_mode == 'robust_hamres':
        print(f"  robust_hamres: smooth_sigma={guidance_hamres_smooth_sigma}  delta={guidance_hamres_delta}  min_scale_q={guidance_hamres_min_scale_q}  min_scale_p={guidance_hamres_min_scale_p}")
    print(f"Smoothing: sigma={SMOOTH_SIGMA}  guidance_only={SMOOTH_GUIDANCE_ONLY}  last_step_only={SMOOTH_LAST_STEP_ONLY}")
    print(f"HamRes eval: smooth_sigma={hamres_kwargs['smooth_sigma']}  delta={hamres_kwargs['delta']}  min_scale_q={hamres_kwargs['min_scale_q']}  min_scale_p={hamres_kwargs['min_scale_p']}")
    if NUM_DIFFUSION_STEPS is not None:
        print(f"Diffusion steps: {NUM_DIFFUSION_STEPS}")
    print(f"{'='*64}")

    # Setup
    print("Loading models...")
    dpf, hnn, var_dq, var_dp, mj_model = load_models(cfg, device)
    torques = load_torques(POLICY, NUM_SAMPLES, LENGTH, device, cfg)
    # Set seed AFTER model loading to avoid random state being consumed by load
    set_seed(SEED)
    noise = torch.randn(NUM_SAMPLES, LENGTH, dpf.state_dim, device=device)

    qpos_dim = cfg['qpos_dim']

    # Build kwargs
    shared_kwargs = {}
    if NUM_DIFFUSION_STEPS is not None:
        shared_kwargs['num_diffusion_steps'] = NUM_DIFFUSION_STEPS

    ung_kwargs = dict(hnn=None, guidance_steps=0, smooth_sigma=SMOOTH_SIGMA,
                      **shared_kwargs)
    gui_kwargs = dict(
        hnn=hnn, guidance_method=guidance_method,
        guidance_steps=guidance_steps, guidance_lr=guidance_lr,
        guidance_after_steps=guidance_after, guidance_before_steps=guidance_before,
        optimize_target=optimize_target,
        guidance_energy_mode=guidance_energy_mode,
        guidance_hamres_smooth_sigma=guidance_hamres_smooth_sigma,
        guidance_hamres_delta=guidance_hamres_delta,
        guidance_hamres_min_scale_q=guidance_hamres_min_scale_q,
        guidance_hamres_min_scale_p=guidance_hamres_min_scale_p,
        guidance_trust_lambda=guidance_trust_lambda,
        smooth_sigma=SMOOTH_SIGMA, smooth_guidance_only=SMOOTH_GUIDANCE_ONLY,
        smooth_last_step_only=SMOOTH_LAST_STEP_ONLY,
        langevin_step_size=LANGEVIN_STEP_SIZE,
        langevin_noise_scale=LANGEVIN_NOISE_SCALE,
        alpha_q=ALPHA_Q,
        alpha_p=ALPHA_P,
        chunk_length=CHUNK_LENGTH,
        **shared_kwargs,
    )

    # Generate unguided
    print("Generating unguided...")
    t0 = time.time()
    ung_states, ung_torques = run_batch(
        dpf, torques, noise, LENGTH, BATCH_SIZE_UNGUIDED, **ung_kwargs)
    t_ung = time.time() - t0

    # Generate guided
    print("Generating guided...")
    t0 = time.time()
    gui_states, gui_torques = run_batch(
        dpf, torques, noise, LENGTH, BATCH_SIZE_GUIDED, **gui_kwargs)
    t_gui = time.time() - t0

    # Compute metrics
    print("Computing metrics...")
    t0 = time.time()
    ung_nq, ung_np, ung_hr, ung_nq_pd, ung_np_pd = compute_metrics_for_samples(
        ung_states, ung_torques, NUM_SAMPLES, mj_model, hnn,
        var_dq, var_dp, qpos_dim, desc="unguided", hamres_kwargs=hamres_kwargs)
    gui_nq, gui_np, gui_hr, gui_nq_pd, gui_np_pd = compute_metrics_for_samples(
        gui_states, gui_torques, NUM_SAMPLES, mj_model, hnn,
        var_dq, var_dp, qpos_dim, desc="guided", hamres_kwargs=hamres_kwargs)
    t_met = time.time() - t0

    # Per-dim arrays: shape (num_samples, num_dims)
    ung_nq_pd = np.array(ung_nq_pd)
    ung_np_pd = np.array(ung_np_pd)
    gui_nq_pd = np.array(gui_nq_pd)
    gui_np_pd = np.array(gui_np_pd)

    # Print results
    print(f"\n{'='*80}")
    print(f"{'':>14} {'mean':>10} {'std':>10} {'p25':>10} {'median':>10} {'p75':>10} {'p99':>10}")
    print(f"{'-'*80}")
    for label, ung_arr, gui_arr in [
        ('NMSE_q', ung_nq, gui_nq),
        ('NMSE_p', ung_np, gui_np),
        ('HamRes', ung_hr, gui_hr),
    ]:
        u_mean = np.mean(ung_arr)
        u_std = np.std(ung_arr)
        u_p25 = np.percentile(ung_arr, 25)
        u_med = np.median(ung_arr)
        u_p75 = np.percentile(ung_arr, 75)
        u_p99 = np.percentile(ung_arr, 99)
        g_mean = np.mean(gui_arr)
        g_std = np.std(gui_arr)
        g_p25 = np.percentile(gui_arr, 25)
        g_med = np.median(gui_arr)
        g_p75 = np.percentile(gui_arr, 75)
        g_p99 = np.percentile(gui_arr, 99)
        print(f"{label:>8} ung {u_mean:>10.6f} {u_std:>10.6f} {u_p25:>10.6f} {u_med:>10.6f} {u_p75:>10.6f} {u_p99:>10.6f}")
        print(f"{'':>8} gui {g_mean:>10.6f} {g_std:>10.6f} {g_p25:>10.6f} {g_med:>10.6f} {g_p75:>10.6f} {g_p99:>10.6f}")
        d_mean = g_mean - u_mean
        d_std = g_std - u_std
        d_p25 = g_p25 - u_p25
        d_med = g_med - u_med
        d_p75 = g_p75 - u_p75
        d_p99 = g_p99 - u_p99
        print(f"{'':>8}   Δ {d_mean:>+10.6f} {d_std:>+10.6f} {d_p25:>+10.6f} {d_med:>+10.6f} {d_p75:>+10.6f} {d_p99:>+10.6f}")
    print(f"{'='*80}")
    print(f"Timing: unguided={t_ung:.1f}s  guided={t_gui:.1f}s  metrics={t_met:.1f}s")

    # Per-dimension NMSE breakdown
    num_q = ung_nq_pd.shape[1]
    num_p = ung_np_pd.shape[1]
    print(f"\n{'='*80}")
    print(f"Per-dimension NMSE (median over samples)")
    print(f"{'-'*80}")
    print(f"{'':>12} ", end="")
    for d in range(num_q):
        print(f"{'dim'+str(d):>10}", end="")
    print(f" {'mean':>10}")
    print(f"{'-'*80}")
    for label, pd_arr in [
        ('NMSE_q ung', ung_nq_pd),
        ('NMSE_q gui', gui_nq_pd),
    ]:
        print(f"{label:>12} ", end="")
        dim_medians = [np.median(pd_arr[:, d]) for d in range(num_q)]
        for v in dim_medians:
            print(f"{v:>10.6f}", end="")
        print(f" {np.mean(dim_medians):>10.6f}")
    # Delta row
    print(f"{'NMSE_q Δ':>12} ", end="")
    dim_deltas = [np.median(gui_nq_pd[:, d]) - np.median(ung_nq_pd[:, d]) for d in range(num_q)]
    for v in dim_deltas:
        print(f"{v:>+10.6f}", end="")
    print(f" {np.mean(dim_deltas):>+10.6f}")

    print(f"{'-'*80}")
    print(f"{'':>12} ", end="")
    for d in range(num_p):
        print(f"{'dim'+str(d):>10}", end="")
    print(f" {'mean':>10}")
    print(f"{'-'*80}")
    for label, pd_arr in [
        ('NMSE_p ung', ung_np_pd),
        ('NMSE_p gui', gui_np_pd),
    ]:
        print(f"{label:>12} ", end="")
        dim_medians = [np.median(pd_arr[:, d]) for d in range(num_p)]
        for v in dim_medians:
            print(f"{v:>10.6f}", end="")
        print(f" {np.mean(dim_medians):>10.6f}")
    # Delta row
    print(f"{'NMSE_p Δ':>12} ", end="")
    dim_deltas = [np.median(gui_np_pd[:, d]) - np.median(ung_np_pd[:, d]) for d in range(num_p)]
    for v in dim_deltas:
        print(f"{v:>+10.6f}", end="")
    print(f" {np.mean(dim_deltas):>+10.6f}")
    print(f"{'='*80}")

    # Plot trajectory comparison
    plot_trajectories(ung_states, ung_torques, gui_states, gui_torques,
                      ung_nq, gui_nq, qpos_dim, cfg['xml_path'])


if __name__ == '__main__':
    main()
