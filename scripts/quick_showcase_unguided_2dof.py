"""
2DoF unguided showcase: DPF sampling (no guidance) vs MuJoCo reconstruction.
Uses wandb-consistent settings: 100 DDIM steps, context_fraction=0.5, no fixed seed.
Generates many samples and picks the best ones to display.
"""
import sys, os, warnings
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import torch
import matplotlib.pyplot as plt
import mujoco

# ── Config ──
DPF_CKPT = str(project_root / 'checkpoints' / '2dof' /
    'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding'
    '&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions'
    ':epoch=2999_val_loss:val_loss=0.0008.ckpt')
MUJOCO_XML = str(project_root / 'configs' / 'rigid_arm_hinge_2dof.xml')
DT = 0.0001
DATA_DT = 0.0002
QPOS_DIM = 2
LENGTH = 1000
NUM_DISPLAY = 2        # How many samples to show
NUM_GENERATE = 20      # Generate more, pick best

# Sampling params (match wandb callback)
NUM_DIFFUSION_STEPS = 100
CONTEXT_FRACTION = 0.5

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'axes.grid': False,
})


def reconstruct_trajectory(qpos_init, mom_init, torque, mj_model):
    T = len(torque)
    mj_model.opt.timestep = DT  # Must set explicitly (XML default is 0.002)
    M = np.zeros((mj_model.nv, mj_model.nv))
    data = mujoco.MjData(mj_model)
    data.qpos[:QPOS_DIM] = qpos_init
    data.qvel[:] = 0
    mujoco.mj_forward(mj_model, data)
    mujoco.mj_fullM(mj_model, M, data.qM)
    initial_qvel = np.linalg.solve(M[:QPOS_DIM, :QPOS_DIM], mom_init)

    substeps = int(DATA_DT / DT)
    qpos_rec = [qpos_init.copy()]
    mom_rec = [mom_init.copy()]

    data.qpos[:QPOS_DIM] = qpos_init
    data.qvel[:QPOS_DIM] = initial_qvel
    mujoco.mj_forward(mj_model, data)

    for t in range(T - 1):
        data.ctrl[:QPOS_DIM] = torque[t]
        for _ in range(substeps):
            mujoco.mj_step(mj_model, data)
        qpos_rec.append(data.qpos[:QPOS_DIM].copy())
        mujoco.mj_fullM(mj_model, M, data.qM)
        mom = M[:QPOS_DIM, :QPOS_DIM] @ data.qvel[:QPOS_DIM]
        mom_rec.append(mom.copy())

    return np.array(qpos_rec), np.array(mom_rec)


if __name__ == '__main__':
    device = torch.device('cuda:0')

    from src.models.trajectory_dpf import TrajectoryDPF
    print("Loading model...")
    dpf = TrajectoryDPF.load_from_checkpoint(DPF_CKPT, map_location=device, strict=False)
    ckpt = torch.load(DPF_CKPT, map_location=device, weights_only=False)
    if 'ema_state_dict' in ckpt:
        dpf.ema.load_state_dict(ckpt['ema_state_dict'])
    dpf.eval()
    del ckpt
    torch.cuda.empty_cache()

    # Sample (unguided, wandb-like settings, NO fixed seed)
    print(f"Sampling {NUM_GENERATE} trajectories (unguided, {NUM_DIFFUSION_STEPS} DDIM steps)...")
    state, torque = dpf.sample_trajectories(
        num_samples=NUM_GENERATE,
        trajectory_length=LENGTH,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        context_fraction=CONTEXT_FRACTION,
        use_ema=True,
        sampler='ddim',
    )
    state = state.cpu().numpy()
    torque = torque.cpu().numpy()

    # MuJoCo reconstruction & MSE
    mj_model = mujoco.MjModel.from_xml_path(MUJOCO_XML)
    print("Reconstructing with MuJoCo...")

    mse_list = []
    recon_list = []
    for s in range(NUM_GENERATE):
        qpos = state[s, :, :QPOS_DIM]
        mom = state[s, :, QPOS_DIM:]
        qpos_rec, mom_rec = reconstruct_trajectory(qpos[0], mom[0], torque[s], mj_model)
        # Align: [1:] vs reconstructed[1:] (skip initial which matches by construction)
        mse_q = np.mean((qpos[1:] - qpos_rec[1:LENGTH]) ** 2)
        mse_p = np.mean((mom[1:] - mom_rec[1:LENGTH]) ** 2)
        mse_list.append(mse_q + mse_p)
        recon_list.append((qpos_rec, mom_rec))
        print(f"  Sample {s}: MSE={mse_q+mse_p:.6f} (qpos={mse_q:.6f}, mom={mse_p:.6f})")

    # Sort by MSE, pick median samples (not best/worst, representative)
    sorted_idx = np.argsort(mse_list)
    median_start = len(sorted_idx) // 2 - NUM_DISPLAY // 2
    selected = sorted_idx[median_start:median_start + NUM_DISPLAY]
    print(f"\nAll MSEs: mean={np.mean(mse_list):.6f}, median={np.median(mse_list):.6f}")
    print(f"Selected median samples: {selected.tolist()} with MSE={[mse_list[i] for i in selected]}")

    # Plot showcase
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, 3 * (1 + NUM_DISPLAY)))
    gs = GridSpec(1 + NUM_DISPLAY, 4, figure=fig, hspace=0.35, wspace=0.3)
    time = np.arange(LENGTH)
    colors = {'dpf': '#1f77b4', 'mujoco': '#d62728'}

    # Row 0: Torque from first selected sample
    first_s = selected[0]
    for dim in range(QPOS_DIM):
        ax = fig.add_subplot(gs[0, dim * 2:(dim + 1) * 2])
        ax.plot(time, torque[first_s, :, dim], color='black', linewidth=1.2)
        ax.set_title(rf'$\tau_{dim}$', fontsize=14)
        if dim == 0:
            ax.set_ylabel('Torque', fontsize=12)
        ax.set_xlim(0, LENGTH)

    # Sample rows
    labels = [r'$q_0$', r'$q_1$', r'$p_0$', r'$p_1$']
    for row, s in enumerate(selected):
        qpos = state[s, :, :QPOS_DIM]
        mom = state[s, :, QPOS_DIM:]
        qpos_rec, mom_rec = recon_list[s]

        data_pairs = [
            (qpos[:, 0], qpos_rec[:, 0]),
            (qpos[:, 1], qpos_rec[:, 1]),
            (mom[:, 0], mom_rec[:, 0]),
            (mom[:, 1], mom_rec[:, 1]),
        ]
        for col, (gen, rec) in enumerate(data_pairs):
            ax = fig.add_subplot(gs[row + 1, col])
            ax.plot(time[:len(gen)], gen, color=colors['dpf'], linewidth=1,
                    label='DPF' if col == 0 else None)
            ax.plot(np.arange(len(rec)), rec, color=colors['mujoco'], linewidth=1,
                    label='MuJoCo' if col == 0 else None)
            if row == 0:
                ax.set_title(labels[col], fontsize=12)
            if col == 0:
                ax.set_ylabel(f'Sample {row+1}\nMSE={mse_list[s]:.6f}', fontsize=11)
                ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
            if row == NUM_DISPLAY - 1:
                ax.set_xlabel('Timestep', fontsize=10)
            ax.set_xlim(0, LENGTH)

    fig.suptitle(f'2DoF Unguided DPF (DDIM {NUM_DIFFUSION_STEPS} steps, context={CONTEXT_FRACTION})\n'
                 f'Median MSE={np.median(mse_list):.6f} (n={NUM_GENERATE})', fontsize=14)

    out_path = project_root / 'plots' / 'traj_showcase_2dof_unguided_only.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"\nSaved: {out_path}")
