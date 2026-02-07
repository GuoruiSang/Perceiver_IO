"""
2DoF showcase: Unguided vs Guided DPF, both compared with MuJoCo reconstruction.
Format: Torque row + alternating Unguided/Guided rows with gray/white backgrounds.
DPF shown as dots, MuJoCo as lines. Same seed for fair comparison.

Usage:
    python scripts/quick_showcase_2dof.py --policy sinusoidal
    python scripts/quick_showcase_2dof.py --all
"""
import sys, os, warnings, argparse
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import torch
import matplotlib.pyplot as plt
import mujoco
import h5py

# ── Config ──
DPF_CKPT = str(project_root / 'checkpoints' / '2dof' /
    'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding'
    '&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions'
    ':epoch=2999_val_loss:val_loss=0.0008.ckpt')
HNN_CKPT = str(project_root / 'checkpoints' / '2dof' /
    'SeperableHNN-2DOF-epoch-epoch=999.ckpt')
MUJOCO_XML = str(project_root / 'configs' / 'rigid_arm_hinge_2dof.xml')
DT = 0.0001
DATA_DT = 0.0002
QPOS_DIM = 2
TORQUE_DIM = 2
LENGTH = 1000
NUM_DISPLAY = 2
NUM_GENERATE = 20
SEED = 228

# Sampling params (match ablation pipeline)
NUM_DIFFUSION_STEPS = 100
CONTEXT_FRACTION = 0.5

# Guidance params (2DoF optimal from ablation)
GUIDANCE_STEPS = 10
GUIDANCE_LR = 0.0001
GUIDANCE_AFTER_STEPS = 45
SMOOTH_SIGMA = 3.0

# Torque data paths
TORQUE_PATHS = {
    'sinusoidal': project_root / 'data' / '2dof' / 'sinusoidal_torques_2000_L1500.h5',
    'gp': project_root / 'data' / '2dof' / 'gp_torques_2000_L1500.h5',
    'spline': project_root / 'data' / '2dof' / 'spline_torques_1000_L1500.h5',
    'zero': None,
}

POLICY_LABELS = {
    'sinusoidal': 'Sinusoidal',
    'gp': 'GP',
    'zero': 'Zero',
    'spline': 'Cubic Spline',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 12,
    'font.weight': 'bold',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'figure.titleweight': 'bold',
})


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_torques(policy, num_samples, device):
    """Load torque sequences for a given policy."""
    if policy == 'zero':
        return torch.zeros(num_samples, LENGTH, TORQUE_DIM, device=device)
    path = TORQUE_PATHS[policy]
    with h5py.File(path, 'r') as f:
        torques = f['torques'][:num_samples, :LENGTH]
    return torch.tensor(torques, dtype=torch.float32, device=device)


def reconstruct_trajectory(qpos_init, mom_init, torque, mj_model):
    T = len(torque)
    mj_model.opt.timestep = DT
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


def compute_mse(state, torque_np, mj_model):
    N = len(state)
    mse_list, recon_list = [], []
    for s in range(N):
        qpos = state[s, :, :QPOS_DIM]
        mom = state[s, :, QPOS_DIM:]
        qpos_rec, mom_rec = reconstruct_trajectory(qpos[0], mom[0], torque_np[s], mj_model)
        mse_q = np.mean((qpos[1:] - qpos_rec[1:LENGTH]) ** 2)
        mse_p = np.mean((mom[1:] - mom_rec[1:LENGTH]) ** 2)
        mse_list.append(mse_q + mse_p)
        recon_list.append((qpos_rec, mom_rec))
    return mse_list, recon_list


def generate_and_plot(dpf, hnn, device, policy):
    """Generate unguided+guided for one torque policy and plot."""
    print(f"\n{'='*60}")
    print(f"Policy: {policy}")
    print(f"{'='*60}")

    # Load torque
    torque_gpu = load_torques(policy, NUM_GENERATE, device)
    print(f"Torque shape: {torque_gpu.shape}")

    # ── Unguided (same seed) ──
    print(f"\nSampling {NUM_GENERATE} UNGUIDED (seed={SEED})...")
    set_seed(SEED)
    state_ung, _ = dpf.sample_trajectories(
        num_samples=NUM_GENERATE,
        trajectory_length=LENGTH,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        context_fraction=CONTEXT_FRACTION,
        use_ema=True,
        sampler='ddim',
        smooth_sigma=0.0,
        torque=torque_gpu,
        hnn=None,
        guidance_steps=0,
    )
    state_ung = state_ung.cpu().numpy()
    torque_np = torque_gpu.cpu().numpy()

    # ── Guided (SAME seed, same torque) ──
    print(f"\nSampling {NUM_GENERATE} GUIDED (seed={SEED}, smooth_sigma={SMOOTH_SIGMA})...")
    set_seed(SEED)
    state_gui, _ = dpf.sample_trajectories(
        num_samples=NUM_GENERATE,
        trajectory_length=LENGTH,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        context_fraction=CONTEXT_FRACTION,
        use_ema=True,
        sampler='ddim',
        torque=torque_gpu,
        hnn=hnn,
        guidance_method='adam',
        guidance_steps=GUIDANCE_STEPS,
        guidance_lr=GUIDANCE_LR,
        guidance_after_steps=GUIDANCE_AFTER_STEPS,
        smooth_sigma=SMOOTH_SIGMA,
    )
    state_gui = state_gui.cpu().numpy()

    # ── MuJoCo reconstruction ──
    mj_model = mujoco.MjModel.from_xml_path(MUJOCO_XML)

    print("\nReconstructing UNGUIDED...")
    mse_ung, recon_ung = compute_mse(state_ung, torque_np, mj_model)
    for s in range(NUM_GENERATE):
        print(f"  Ung {s}: MSE={mse_ung[s]:.6f}")

    print("\nReconstructing GUIDED...")
    mse_gui, recon_gui = compute_mse(state_gui, torque_np, mj_model)
    for s in range(NUM_GENERATE):
        print(f"  Gui {s}: MSE={mse_gui[s]:.6f}")

    # ── Pick best samples: unguided decent, guided very good, big ratio ──
    candidates = []
    for i in range(NUM_GENERATE):
        ratio = mse_ung[i] / max(mse_gui[i], 1e-10)
        candidates.append((i, mse_ung[i], mse_gui[i], ratio))

    # Filter: unguided not in worst 25%, guided in best 50%
    ung_threshold = np.percentile(mse_ung, 75)
    gui_threshold = np.percentile(mse_gui, 50)
    good = [c for c in candidates if c[1] <= ung_threshold and c[2] <= gui_threshold]
    if len(good) < NUM_DISPLAY:
        good = candidates
    good.sort(key=lambda x: x[3], reverse=True)
    selected = [g[0] for g in good[:NUM_DISPLAY]]

    print(f"\nUnguided MSE: mean={np.mean(mse_ung):.6f}, median={np.median(mse_ung):.6f}")
    print(f"Guided MSE:   mean={np.mean(mse_gui):.6f}, median={np.median(mse_gui):.6f}")
    print(f"Selected samples: {selected}")
    for s in selected:
        print(f"  Sample {s}: ung={mse_ung[s]:.6f}, gui={mse_gui[s]:.6f}, "
              f"ratio={mse_ung[s]/max(mse_gui[s],1e-10):.1f}x")

    # ── Plot ──
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, 12))
    nrows = 1 + 2 * NUM_DISPLAY
    gs = GridSpec(nrows, 4, figure=fig, hspace=0.35, wspace=0.3)
    time = np.arange(LENGTH)
    colors = {'dpf': '#0066FF', 'mujoco': '#FF0000'}

    # Row 0: Torque from first selected sample
    first_s = selected[0]
    for dim in range(QPOS_DIM):
        ax = fig.add_subplot(gs[0, dim * 2:(dim + 1) * 2])
        ax.plot(time, torque_np[first_s, :, dim], color='black', linewidth=1.2)
        ax.set_title(rf'$\tau_{dim}$', fontsize=14)
        if dim == 0:
            ax.set_ylabel('Torque', fontsize=12)
        ax.set_xlim(0, LENGTH)

    # Row configs
    row_configs = []
    for i in range(NUM_DISPLAY):
        row_configs.append((1 + 2*i,     i, 'ung', f'Sample {i+1}\nUnguided', True))
        row_configs.append((1 + 2*i + 1, i, 'gui', f'Sample {i+1}\nGuided',   False))

    labels = [r'$q_0$', r'$q_1$', r'$p_0$', r'$p_1$']

    axes_map = {}
    for row_idx, sample_list_idx, mode, ylabel, gray_bg in row_configs:
        s = selected[sample_list_idx]
        if mode == 'ung':
            st = state_ung[s]
            qpos_rec, mom_rec = recon_ung[s]
            label_dpf = 'Unguided'
        else:
            st = state_gui[s]
            qpos_rec, mom_rec = recon_gui[s]
            label_dpf = 'Guided'

        qpos = st[:, :QPOS_DIM]
        mom = st[:, QPOS_DIM:]
        time_rec = np.arange(len(qpos_rec))

        data_pairs = [
            (qpos[:, 0], qpos_rec[:, 0]),
            (qpos[:, 1], qpos_rec[:, 1]),
            (mom[:, 0], mom_rec[:, 0]),
            (mom[:, 1], mom_rec[:, 1]),
        ]

        for col, (gen, rec) in enumerate(data_pairs):
            ax = fig.add_subplot(gs[row_idx, col])
            if gray_bg:
                ax.set_facecolor('#E8E8E8')
            ax.plot(time[:len(gen)], gen, color=colors['dpf'],
                    marker='.', markersize=1, linewidth=0,
                    label=label_dpf if col == 0 else None)
            ax.plot(time_rec, rec, color=colors['mujoco'], linewidth=1,
                    label='MuJoCo' if col == 0 else None)
            if row_idx == 1:
                ax.set_title(labels[col], fontsize=12)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=11)
                ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
            if row_idx == nrows - 1:
                ax.set_xlabel('Trajectory Length', fontsize=10)
            ax.set_xlim(0, LENGTH)

            key = (sample_list_idx, col)
            if key not in axes_map:
                axes_map[key] = []
            axes_map[key].append(ax)

    # Align y-axis for each (sample, col) pair
    for key, ax_pair in axes_map.items():
        ymin = min(a.get_ylim()[0] for a in ax_pair)
        ymax = max(a.get_ylim()[1] for a in ax_pair)
        margin = (ymax - ymin) * 0.05
        for a in ax_pair:
            a.set_ylim(ymin - margin, ymax + margin)

    policy_label = POLICY_LABELS[policy]
    fig.suptitle(f'2DoF-{policy_label} Torque Policy (Length={LENGTH})',
                 fontsize=16, fontweight='bold', y=0.98)

    out_name = f'2DoF-{policy_label.replace(" ", "_")}_Torque_Policy_Length_{LENGTH}.png'
    out_path = project_root / 'plots' / out_name
    (project_root / 'plots').mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='sinusoidal',
                        choices=['sinusoidal', 'gp', 'zero', 'spline'])
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0')

    from src.models.trajectory_dpf import TrajectoryDPF
    from src.models.HNN import HNNWrapper

    # Load models once
    print("Loading DPF...")
    dpf = TrajectoryDPF.load_from_checkpoint(DPF_CKPT, map_location=device, strict=False)
    ckpt = torch.load(DPF_CKPT, map_location=device, weights_only=False)
    if 'ema_state_dict' in ckpt:
        dpf.ema.load_state_dict(ckpt['ema_state_dict'])
    dpf.eval()
    del ckpt
    torch.cuda.empty_cache()

    print("Loading HNN...")
    hnn = HNNWrapper.load_from_checkpoint(HNN_CKPT, map_location=device)
    hnn.eval()

    if args.all:
        for policy in ['sinusoidal', 'gp', 'zero', 'spline']:
            generate_and_plot(dpf, hnn, device, policy)
    else:
        generate_and_plot(dpf, hnn, device, args.policy)
