"""
Preview all candidate torque policies and their resulting MuJoCo trajectories.

Generates one sample trajectory per policy, plots torque + qpos + momentum
side by side for visual comparison.

Usage:
    python scripts/preview_torque_policies.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import math
import mujoco
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline


# ─── Configuration ───────────────────────────────────────────────────────────
XML_PATH = str(project_root / 'configs' / 'rigid_arm_hinge.xml')
NUM_STEPS = 1000        # data collection points
DT = 0.0001             # fine simulation timestep
DATA_DT = 0.0002        # data collection timestep (skip_steps * DT = 2 * 0.0001)
SKIP_STEPS = int(round(DATA_DT / DT))
NUM_SIM_STEPS = NUM_STEPS * SKIP_STEPS
SEED = 42

# Sinusoidal defaults (same as training)
NUM_SIN = 5
LIM_AMPLITUDE = 0.5
LIM_FREQUENCY = 6 * math.pi
LIM_PHASE = 2 * math.pi


# ─── Amplitude matching ─────────────────────────────────────────────────────
def _amplitude_match(torque_raw, ref_std_per_dim):
    """Rescale raw torque per-dimension to match sinusoidal reference std."""
    torque = torque_raw.copy()
    for d in range(torque.shape[1]):
        current_std = torque[:, d].std()
        if current_std > 0:
            torque[:, d] *= ref_std_per_dim[d] / current_std
    return torque


def compute_reference_std(model, num_ref=1000):
    """Compute per-dim std from sinusoidal reference torques."""
    rng = np.random.RandomState(0)
    torque_dim = model.nu
    all_torques = []
    for _ in range(num_ref):
        amplitudes = rng.uniform(0, LIM_AMPLITUDE, (torque_dim, NUM_SIN, 1))
        frequencies = rng.uniform(0, LIM_FREQUENCY, (torque_dim, NUM_SIN, 1))
        phases = rng.uniform(0, LIM_PHASE, (torque_dim, NUM_SIN, 1))
        t = np.arange(NUM_SIM_STEPS) * model.opt.timestep
        t = t[None, None, :]
        t = np.tile(t, (torque_dim, NUM_SIN, 1))
        torque = np.sum(amplitudes * np.sin(frequencies * t + phases), axis=1).T
        all_torques.append(torque)
    ref = np.array(all_torques)
    return ref.std(axis=(0, 1))


# ─── Torque generators ──────────────────────────────────────────────────────
# Each returns (NUM_SIM_STEPS, torque_dim) at fine simulation resolution.

def generate_sinusoidal(model, rng, ref_std_per_dim):
    torque_dim = model.nu
    amplitudes = rng.uniform(0, LIM_AMPLITUDE, (torque_dim, NUM_SIN, 1))
    frequencies = rng.uniform(0, LIM_FREQUENCY, (torque_dim, NUM_SIN, 1))
    phases = rng.uniform(0, LIM_PHASE, (torque_dim, NUM_SIN, 1))
    t = np.arange(NUM_SIM_STEPS) * model.opt.timestep
    t = t[None, None, :]
    t = np.tile(t, (torque_dim, NUM_SIN, 1))
    return np.sum(amplitudes * np.sin(frequencies * t + phases), axis=1).T


def generate_gp(model, rng, ref_std_per_dim, length_scale=100):
    torque_dim = model.nu
    noise = rng.randn(NUM_SIM_STEPS, torque_dim)
    smoothed = gaussian_filter1d(noise, sigma=length_scale, axis=0)
    return _amplitude_match(smoothed, ref_std_per_dim)


def generate_chirp(model, rng, ref_std_per_dim, num_chirps=3):
    torque_dim = model.nu
    t = np.arange(NUM_SIM_STEPS) * model.opt.timestep
    T = t[-1]
    torque = np.zeros((NUM_SIM_STEPS, torque_dim))
    for d in range(torque_dim):
        for _ in range(num_chirps):
            f0 = rng.uniform(0, LIM_FREQUENCY / 3)
            f1 = rng.uniform(LIM_FREQUENCY / 3, LIM_FREQUENCY)
            A = rng.uniform(0, 1)
            phase = rng.uniform(0, 2 * np.pi)
            phi = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * T))
            torque[:, d] += A * np.sin(phi + phase)
    return _amplitude_match(torque, ref_std_per_dim)


def generate_step(model, rng, ref_std_per_dim, mean_hold=100, smooth_sigma=5):
    torque_dim = model.nu
    torque = np.zeros((NUM_SIM_STEPS, torque_dim))
    for d in range(torque_dim):
        idx = 0
        while idx < NUM_SIM_STEPS:
            hold = max(1, int(rng.exponential(mean_hold)))
            level = rng.uniform(-LIM_AMPLITUDE, LIM_AMPLITUDE)
            torque[idx:idx + hold, d] = level
            idx += hold
        if smooth_sigma > 0:
            torque[:, d] = gaussian_filter1d(torque[:, d], sigma=smooth_sigma)
    return _amplitude_match(torque, ref_std_per_dim)


def generate_ou(model, rng, ref_std_per_dim, theta=5.0, sigma=1.0):
    torque_dim = model.nu
    dt_sim = model.opt.timestep
    torque = np.zeros((NUM_SIM_STEPS, torque_dim))
    for d in range(torque_dim):
        th = rng.uniform(1.0, 10.0)
        sig = rng.uniform(0.5, 2.0)
        x = 0.0
        for i in range(NUM_SIM_STEPS):
            x += th * (0.0 - x) * dt_sim + sig * np.sqrt(dt_sim) * rng.randn()
            torque[i, d] = x
    return _amplitude_match(torque, ref_std_per_dim)


def generate_spline(model, rng, ref_std_per_dim, n_points_range=(8, 15)):
    torque_dim = model.nu
    torque = np.zeros((NUM_SIM_STEPS, torque_dim))
    t_fine = np.arange(NUM_SIM_STEPS)
    for d in range(torque_dim):
        n_pts = rng.randint(n_points_range[0], n_points_range[1] + 1)
        t_ctrl = np.sort(rng.choice(NUM_SIM_STEPS, n_pts, replace=False))
        t_ctrl[0] = 0
        t_ctrl[-1] = NUM_SIM_STEPS - 1
        y_ctrl = rng.uniform(-LIM_AMPLITUDE, LIM_AMPLITUDE, n_pts)
        cs = CubicSpline(t_ctrl, y_ctrl, bc_type='natural')
        torque[:, d] = cs(t_fine)
    return _amplitude_match(torque, ref_std_per_dim)


# ─── MuJoCo forward simulation ──────────────────────────────────────────────

def simulate(model, seq_torque_fine, initial_qpos, initial_qvel):
    """Run forward dynamics and collect qpos + momentum at data collection rate.

    Args:
        model: MuJoCo model (timestep already set)
        seq_torque_fine: (NUM_SIM_STEPS, torque_dim) torques at fine resolution
        initial_qpos, initial_qvel: initial conditions

    Returns:
        seq_qpos: (NUM_STEPS, nq)
        seq_mom: (NUM_STEPS, nv)
        seq_torque: (NUM_STEPS, nu) subsampled torque
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    mujoco.mj_forward(model, data)

    nq, nv = model.nq, model.nv
    seq_qpos = np.empty((NUM_STEPS, nq))
    seq_mom = np.empty((NUM_STEPS, nv))
    M = np.zeros((nv, nv))

    for data_idx in range(NUM_STEPS):
        for sub_step in range(SKIP_STEPS):
            sim_idx = data_idx * SKIP_STEPS + sub_step
            data.ctrl[:] = seq_torque_fine[sim_idx]
            mujoco.mj_step(model, data)

        seq_qpos[data_idx] = data.qpos
        mujoco.mj_fullM(model, M, data.qM)
        seq_mom[data_idx] = M @ data.qvel

    seq_torque = seq_torque_fine[SKIP_STEPS - 1::SKIP_STEPS]
    return seq_qpos, seq_mom, seq_torque


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)

    # Load model
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    model.opt.timestep = DT
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY

    # Compute amplitude reference
    print("Computing sinusoidal reference statistics...")
    ref_std = compute_reference_std(model, num_ref=1000)
    print(f"  ref_std_per_dim = {ref_std}")

    # Fixed initial condition for fair comparison
    rng_init = np.random.RandomState(SEED)
    initial_qpos = rng_init.uniform(-math.pi / 3, math.pi / 3, model.nq)
    initial_qvel = rng_init.uniform(-math.pi / 3, math.pi / 3, model.nv)

    policies = [
        ('sinusoidal', generate_sinusoidal),
        ('gp', generate_gp),
        ('chirp', generate_chirp),
        ('step', generate_step),
        ('ou', generate_ou),
        ('spline', generate_spline),
    ]

    fig, axes = plt.subplots(len(policies), 3, figsize=(18, 3 * len(policies)),
                             dpi=120, squeeze=False)
    fig.suptitle('Torque Policy Preview (same initial conditions, amplitude-matched)',
                 fontsize=14, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # one per joint dimension
    time_data = np.arange(NUM_STEPS) * DATA_DT * 1000  # ms

    for row, (name, gen_fn) in enumerate(policies):
        rng = np.random.RandomState(SEED + row)
        torque_fine = gen_fn(model, rng, ref_std)

        qpos, mom, torque_sub = simulate(model, torque_fine, initial_qpos, initial_qvel)

        # Column 0: Torque
        ax = axes[row, 0]
        for d in range(model.nu):
            ax.plot(time_data, torque_sub[:, d], color=colors[d], alpha=0.8,
                    linewidth=0.5, label=f'dim {d}')
        ax.set_ylabel(name, fontsize=11, fontweight='bold', rotation=0, labelpad=60, va='center')
        if row == 0:
            ax.set_title('Torque', fontsize=12)
        if row == len(policies) - 1:
            ax.set_xlabel('Time (ms)')

        # Column 1: qpos
        ax = axes[row, 1]
        for d in range(model.nq):
            ax.plot(time_data, qpos[:, d], color=colors[d], alpha=0.8, linewidth=0.5)
        if row == 0:
            ax.set_title('Joint Position (qpos)', fontsize=12)
        if row == len(policies) - 1:
            ax.set_xlabel('Time (ms)')

        # Column 2: momentum
        ax = axes[row, 2]
        for d in range(model.nv):
            ax.plot(time_data, mom[:, d], color=colors[d], alpha=0.8, linewidth=0.5)
        if row == 0:
            ax.set_title('Momentum', fontsize=12)
        if row == len(policies) - 1:
            ax.set_xlabel('Time (ms)')

        # Print statistics
        print(f"  {name:12s}  torque std={torque_sub.std(axis=0)}  "
              f"qpos range=[{qpos.min():.2f}, {qpos.max():.2f}]  "
              f"mom range=[{mom.min():.2f}, {mom.max():.2f}]")

    # Add legend to first row
    axes[0, 0].legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    out_path = project_root / 'plots' / 'torque_policy_preview.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
