import mujoco
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import h5py
import os
import tempfile
from tqdm import tqdm
import math
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline

# Worker-global MuJoCo model cache.
# Reusing the parsed model in each process avoids expensive XML reload per trajectory.
_WORKER_MODEL = None
_WORKER_XML_PATH = None
_WORKER_DT = None
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = PROJECT_ROOT / "plots"
DATA_DIR = PROJECT_ROOT / "data"

TRAJ_KEYS = ['seq_qpos', 'seq_qvel', 'seq_qacc', 'seq_mom', 'seq_mom_dot', 'seq_torque', 'seq_energy']

def _sample_reacher_goal_in_disk(max_radius: float = 0.2):
    """Sample Gym-style Reacher target position uniformly in a disk."""
    while True:
        goal = np.random.uniform(low=-max_radius, high=max_radius, size=(2,))
        if np.linalg.norm(goal) < max_radius:
            return goal


def _get_reacher_target_indices(model):
    """Return target qpos/dof indices if the model has Reacher target joints."""
    jtx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_x")
    jty = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_y")
    if jtx < 0 or jty < 0:
        return None
    qx = int(model.jnt_qposadr[jtx])
    qy = int(model.jnt_qposadr[jty])
    dx = int(model.jnt_dofadr[jtx])
    dy = int(model.jnt_dofadr[jty])
    return qx, qy, dx, dy


def randomly_initialize_qpos_qvel_qacc(
    model,
    lim_qpos=1,
    lim_qvel=1,
    gym_reacher_target_sampling: bool = True,
    gym_reacher_state_sampling: bool = True,
):
    """
        Args: 
            model: The mujoco model.
        Returns:
            initial_qpos: (model.nq, ). A randomly sampled position.
            initial_qvel: (model.nv, ). A randomly sampled velocity
    """

    # Get the dimensions of qpos, qvel, and qcc
    qpos_dim = model.nq
    qvel_dim = model.nv

    target_indices = _get_reacher_target_indices(model)
    use_gym_reacher_state = bool(gym_reacher_state_sampling and target_indices is not None)

    if use_gym_reacher_state:
        # Match Gymnasium Reacher-v5 reset_model():
        # qpos = init_qpos + U(-0.1, 0.1), qvel = init_qvel + U(-0.005, 0.005)
        if hasattr(model, "qpos0"):
            initial_qpos = np.array(model.qpos0, dtype=np.float64).copy()
        else:
            initial_qpos = np.zeros((qpos_dim,), dtype=np.float64)
        # MuJoCo MjModel does not expose qvel0; Reacher-v5 init_qvel is zero anyway.
        initial_qvel = np.zeros((qvel_dim,), dtype=np.float64)
        initial_qpos += np.random.uniform(low=-0.1, high=0.1, size=(qpos_dim,))
        initial_qvel += np.random.uniform(low=-0.005, high=0.005, size=(qvel_dim,))
    else:
        # Legacy broad random initialization.
        initial_qpos = np.random.uniform(low=-lim_qpos, high=lim_qpos, size=(qpos_dim,))
        initial_qvel = np.random.uniform(low=-lim_qvel, high=lim_qvel, size=(qvel_dim,))

    # For Gym/Gymnasium Reacher comparability:
    # place the target uniformly in a radius-0.2 disk.
    if gym_reacher_target_sampling:
        if target_indices is not None:
            qx, qy, dx, dy = target_indices
            goal_xy = _sample_reacher_goal_in_disk(max_radius=0.2)
            initial_qpos[qx] = goal_xy[0]
            initial_qpos[qy] = goal_xy[1]
            # Reacher-v5 sets target velocity components to zero at reset.
            initial_qvel[dx] = 0.0
            initial_qvel[dy] = 0.0

    # In forward dynamics, qacc is an output, not an input state.
    # initial_qacc = np.zeros((qvel_dim,))

    return initial_qpos, initial_qvel

def generate_random_seq_torque(model, num_steps: int = 2000, skip_steps: int = 1, num_sin: int = 20, lim_amplitude: int = 10, lim_frequency: int = 1, lim_phase: float = 3.14):
    """
    Generate a smooth random torque sequence using sum of sinusoids.
    
    Args: 
        model: MuJoCo model
        num_steps: Number of data collection points (output length after subsampling)
        skip_steps: Number of simulation steps between data collection points.
                   Total simulation steps = num_steps * skip_steps
        num_sin: Number of sinusoids to sum for each torque dimension
        lim_amplitude: Maximum amplitude for sinusoids
        lim_frequency: Maximum frequency for sinusoids
        lim_phase: Maximum phase shift for sinusoids

    Returns:
        seq_torque: (num_sim_steps, torque_dim) torque at fine simulation resolution
                   where num_sim_steps = num_steps * skip_steps
    """
    torque_dim = model.nu
    num_sim_steps = num_steps * skip_steps

    amplitudes = np.random.uniform(low=0, high=lim_amplitude, size=(torque_dim, num_sin, 1))
    frequencies = np.random.uniform(low=0, high=lim_frequency, size=(torque_dim, num_sin, 1))
    phases = np.random.uniform(low=0, high=lim_phase, size=(torque_dim, num_sin, 1))

    # Create a steps array with shape (1, 1, num_sim_steps) and broadcast to (torque_dim, num_sin, num_sim_steps)
    # Time array at fine simulation resolution
    steps = np.arange(num_sim_steps) * model.opt.timestep  # shape (num_sim_steps,)
    steps = steps[None, None, :]  # shape (1, 1, num_sim_steps)
    steps = np.tile(steps, (torque_dim, num_sin, 1))  # shape (torque_dim, num_sin, num_sim_steps)
    
    # Use sin functions to generate seq_torque
    seq_torque = np.sum(amplitudes * np.sin(frequencies * steps + phases), axis=1).T
    return seq_torque


# ─── Diverse torque policies ────────────────────────────────────────────────

def _amplitude_match(torque_raw, ref_std_per_dim):
    """Rescale raw torque per-dimension to match sinusoidal reference std."""
    torque = torque_raw.copy()
    for d in range(torque.shape[1]):
        current_std = torque[:, d].std()
        if current_std > 0:
            torque[:, d] *= ref_std_per_dim[d] / current_std
    return torque


def compute_reference_std(xml_path, num_steps, dt, data_dt, num_ref=1000):
    """Compute per-dim std from sinusoidal reference torques.

    Loads a fresh model internally so this can be called before the main loop.
    Returns ref_std_per_dim as a numpy array of shape (torque_dim,).
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = dt
    skip_steps = max(1, int(round(data_dt / dt)))
    rng = np.random.RandomState(0)
    all_torques = []
    for _ in range(num_ref):
        amplitudes = rng.uniform(0, 0.5, (model.nu, 5, 1))
        frequencies = rng.uniform(0, 6 * math.pi, (model.nu, 5, 1))
        phases = rng.uniform(0, 2 * math.pi, (model.nu, 5, 1))
        t = np.arange(num_steps * skip_steps) * dt
        t = t[None, None, :]
        t = np.tile(t, (model.nu, 5, 1))
        torque = np.sum(amplitudes * np.sin(frequencies * t + phases), axis=1).T
        all_torques.append(torque)
    ref = np.array(all_torques)
    return ref.std(axis=(0, 1))


def generate_gp_torque(model, num_steps, skip_steps, ref_std_per_dim, length_scale=100):
    """Gaussian-filtered white noise torque (smooth, non-periodic, broadband)."""
    num_sim_steps = num_steps * skip_steps
    noise = np.random.randn(num_sim_steps, model.nu)
    smoothed = gaussian_filter1d(noise, sigma=length_scale, axis=0)
    return _amplitude_match(smoothed, ref_std_per_dim)


def generate_chirp_torque(model, num_steps, skip_steps, ref_std_per_dim,
                          lim_frequency=6*math.pi, num_chirps=3):
    """Swept-frequency sinusoid torque (smooth, time-varying frequency)."""
    num_sim_steps = num_steps * skip_steps
    t = np.arange(num_sim_steps) * model.opt.timestep
    T = t[-1]
    torque = np.zeros((num_sim_steps, model.nu))
    for d in range(model.nu):
        for _ in range(num_chirps):
            f0 = np.random.uniform(0, lim_frequency / 3)
            f1 = np.random.uniform(lim_frequency / 3, lim_frequency)
            A = np.random.uniform(0, 1)
            phase = np.random.uniform(0, 2 * np.pi)
            phi = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * T))
            torque[:, d] += A * np.sin(phi + phase)
    return _amplitude_match(torque, ref_std_per_dim)


def generate_step_torque(model, num_steps, skip_steps, ref_std_per_dim,
                         lim_amplitude=0.5, mean_hold=100, smooth_sigma=5):
    """Piecewise constant torque with light smoothing (near-discontinuous)."""
    num_sim_steps = num_steps * skip_steps
    torque = np.zeros((num_sim_steps, model.nu))
    for d in range(model.nu):
        idx = 0
        while idx < num_sim_steps:
            hold = max(1, int(np.random.exponential(mean_hold)))
            level = np.random.uniform(-lim_amplitude, lim_amplitude)
            torque[idx:idx + hold, d] = level
            idx += hold
        if smooth_sigma > 0:
            torque[:, d] = gaussian_filter1d(torque[:, d], sigma=smooth_sigma)
    return _amplitude_match(torque, ref_std_per_dim)


def generate_ou_torque(model, num_steps, skip_steps, ref_std_per_dim):
    """Ornstein-Uhlenbeck process torque (stochastic, mean-reverting)."""
    num_sim_steps = num_steps * skip_steps
    dt_sim = model.opt.timestep
    torque = np.zeros((num_sim_steps, model.nu))
    for d in range(model.nu):
        theta = np.random.uniform(1.0, 10.0)
        sigma = np.random.uniform(0.5, 2.0)
        x = 0.0
        for i in range(num_sim_steps):
            x += theta * (0.0 - x) * dt_sim + sigma * np.sqrt(dt_sim) * np.random.randn()
            torque[i, d] = x
    return _amplitude_match(torque, ref_std_per_dim)


def generate_drift_ou_torque(
    model,
    num_steps,
    skip_steps,
    ref_std_per_dim=None,
    rho=0.97,
    mean_rho=0.998,
    eps_scale=0.35,
    ctrl_margin=0.05,
):
    """
    OU process around a slowly drifting latent mean.

    This produces smooth controls without piecewise jumps, while preserving
    broad exploration and target reachability better than plain high-rho OU.
    """
    num_sim_steps = num_steps * skip_steps
    low, high = _get_control_ranges(model)
    low_safe = low + ctrl_margin
    high_safe = high - ctrl_margin

    finite = np.isfinite(low_safe) & np.isfinite(high_safe) & (high_safe > low_safe)
    low_safe = np.where(finite, low_safe, -1.0)
    high_safe = np.where(finite, high_safe, 1.0)
    target_std = (high_safe - low_safe) / np.sqrt(12.0)

    torque = np.zeros((num_sim_steps, model.nu), dtype=np.float64)
    mean_state = np.zeros((model.nu,), dtype=np.float64)

    sigma_mean = np.sqrt(max(1e-12, 1.0 - mean_rho * mean_rho))
    sigma_ou = np.sqrt(max(1e-12, 1.0 - rho * rho))

    for i in range(1, num_sim_steps):
        mean_state = mean_rho * mean_state + sigma_mean * target_std * np.random.randn(model.nu)
        mean_state = np.clip(mean_state, low_safe, high_safe)
        torque[i] = (
            rho * torque[i - 1]
            + (1.0 - rho) * mean_state
            + eps_scale * sigma_ou * target_std * np.random.randn(model.nu)
        )
        torque[i] = np.clip(torque[i], low_safe, high_safe)

    return torque


def generate_spline_torque(model, num_steps, skip_steps, ref_std_per_dim,
                           lim_amplitude=0.5, n_points_range=(8, 15)):
    """Cubic spline through random waypoints (smooth, aperiodic)."""
    num_sim_steps = num_steps * skip_steps
    torque = np.zeros((num_sim_steps, model.nu))
    t_fine = np.arange(num_sim_steps)
    for d in range(model.nu):
        n_pts = np.random.randint(n_points_range[0], n_points_range[1] + 1)
        t_ctrl = np.sort(np.random.choice(num_sim_steps, n_pts, replace=False))
        t_ctrl[0] = 0
        t_ctrl[-1] = num_sim_steps - 1
        y_ctrl = np.random.uniform(-lim_amplitude, lim_amplitude, n_pts)
        cs = CubicSpline(t_ctrl, y_ctrl, bc_type='natural')
        torque[:, d] = cs(t_fine)
    return _amplitude_match(torque, ref_std_per_dim)


def generate_uniform_iid_torque(model, num_steps, skip_steps, ctrl_margin=0.05):
    """IID uniform torque at each simulation step, inside actuator bounds minus margin."""
    num_sim_steps = num_steps * skip_steps
    low, high = _get_control_ranges(model)
    low_safe = low + ctrl_margin
    high_safe = high - ctrl_margin

    # Fallback for any actuator without finite limits.
    finite = np.isfinite(low_safe) & np.isfinite(high_safe) & (high_safe > low_safe)
    low_safe = np.where(finite, low_safe, -1.0)
    high_safe = np.where(finite, high_safe, 1.0)
    return np.random.uniform(low=low_safe[None, :], high=high_safe[None, :], size=(num_sim_steps, model.nu))


def generate_lpf_uniform_torque(model, num_steps, skip_steps, beta=0.992, ctrl_margin=0.05):
    """Per-step IID uniform torque filtered by a first-order low-pass recursion."""
    num_sim_steps = num_steps * skip_steps
    low, high = _get_control_ranges(model)
    low_safe = low + ctrl_margin
    high_safe = high - ctrl_margin

    finite = np.isfinite(low_safe) & np.isfinite(high_safe) & (high_safe > low_safe)
    low_safe = np.where(finite, low_safe, -1.0)
    high_safe = np.where(finite, high_safe, 1.0)

    beta = float(np.clip(beta, 0.0, 0.99999))
    torque = np.zeros((num_sim_steps, model.nu), dtype=np.float64)
    for t in range(1, num_sim_steps):
        u = np.random.uniform(low=low_safe, high=high_safe, size=(model.nu,))
        torque[t] = beta * torque[t - 1] + (1.0 - beta) * u

    return np.clip(torque, low_safe, high_safe)


def generate_torque_sequence(model, num_steps, skip_steps, policy='sinusoidal',
                             ref_std_per_dim=None, **kwargs):
    """Dispatch to the appropriate torque generator based on policy name."""
    if policy == 'sinusoidal':
        return generate_random_seq_torque(model, num_steps=num_steps,
                                          skip_steps=skip_steps, **kwargs)
    elif policy == 'gp':
        return generate_gp_torque(model, num_steps, skip_steps, ref_std_per_dim)
    elif policy == 'chirp':
        return generate_chirp_torque(model, num_steps, skip_steps, ref_std_per_dim)
    elif policy == 'step':
        return generate_step_torque(model, num_steps, skip_steps, ref_std_per_dim)
    elif policy == 'ou':
        return generate_ou_torque(model, num_steps, skip_steps, ref_std_per_dim)
    elif policy == 'drift_ou':
        return generate_drift_ou_torque(
            model,
            num_steps,
            skip_steps,
            ref_std_per_dim,
            rho=float(kwargs.get("drift_ou_rho", 0.97)),
            mean_rho=float(kwargs.get("drift_ou_mean_rho", 0.998)),
            eps_scale=float(kwargs.get("drift_ou_eps_scale", 0.35)),
            ctrl_margin=float(kwargs.get("ctrl_margin", 0.05)),
        )
    elif policy == 'spline':
        return generate_spline_torque(model, num_steps, skip_steps, ref_std_per_dim)
    elif policy == 'uniform_iid':
        ctrl_margin = float(kwargs.get("ctrl_margin", 0.05))
        return generate_uniform_iid_torque(model, num_steps, skip_steps, ctrl_margin=ctrl_margin)
    elif policy == 'lpf_uniform':
        ctrl_margin = float(kwargs.get("ctrl_margin", 0.05))
        beta = float(kwargs.get("lpf_uniform_beta", 0.992))
        return generate_lpf_uniform_torque(
            model,
            num_steps,
            skip_steps,
            beta=beta,
            ctrl_margin=ctrl_margin,
        )
    elif policy == 'zero':
        return np.zeros((num_steps * skip_steps, model.nu))
    else:
        raise ValueError(f"Unknown torque policy: {policy}")


def _build_non_dissipative_xml(xml_path: str) -> str:
    """Create a temporary XML with dissipative terms removed.

    This keeps the original task geometry/actuation intact while removing
    damping/friction losses so the data is closer to conservative dynamics.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Remove default joint damping/friction losses.
    for default_joint in root.findall(".//default/joint"):
        default_joint.set("damping", "0")
        default_joint.set("frictionloss", "0")

    # Explicitly zero damping/frictionloss for Reacher arm joints.
    for joint in root.findall(".//joint"):
        name = joint.get("name", "")
        if name in {"joint0", "joint1"}:
            joint.set("damping", "0")
            joint.set("frictionloss", "0")

    # Zero geom friction for completeness (contacts are usually disabled anyway).
    for geom in root.findall(".//geom"):
        geom.set("friction", "0 0 0")

    fd, tmp_path = tempfile.mkstemp(prefix="reacher_non_diss_", suffix=".xml")
    os.close(fd)
    tree.write(tmp_path, encoding="utf-8", xml_declaration=False)
    return tmp_path


def _get_joint1_qpos_index(model):
    """Return qpos index for joint1, or None if not present."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    if joint_id < 0:
        return None
    return int(model.jnt_qposadr[joint_id])


def _get_control_ranges(model):
    """Return actuator control bounds at simulation resolution."""
    low = np.full(model.nu, -np.inf, dtype=np.float64)
    high = np.full(model.nu, np.inf, dtype=np.float64)
    if model.nu > 0:
        limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
        if np.any(limited):
            low[limited] = model.actuator_ctrlrange[limited, 0]
            high[limited] = model.actuator_ctrlrange[limited, 1]
    return low, high


def _saturation_fraction(seq_torque_fine, ctrl_low, ctrl_high, eps=1e-10):
    """Fraction of control entries that exceed actuator ranges."""
    below = seq_torque_fine < (ctrl_low[None, :] - eps)
    above = seq_torque_fine > (ctrl_high[None, :] + eps)
    violated = np.logical_or(below, above)
    return float(np.mean(violated))


def _init_worker_model(xml_path: str, dt: float) -> None:
    """Process initializer: load MuJoCo model once per worker process."""
    global _WORKER_MODEL, _WORKER_XML_PATH, _WORKER_DT
    _WORKER_MODEL = mujoco.MjModel.from_xml_path(xml_path)
    _WORKER_MODEL.opt.timestep = dt
    _WORKER_MODEL.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    _WORKER_XML_PATH = xml_path
    _WORKER_DT = dt


def _get_or_create_worker_model(xml_path: str, dt: float):
    """Get worker-local cached model or lazily create it (single-process fallback)."""
    global _WORKER_MODEL, _WORKER_XML_PATH, _WORKER_DT
    if _WORKER_MODEL is None or _WORKER_XML_PATH != xml_path or _WORKER_DT != dt:
        _init_worker_model(xml_path, dt)
    return _WORKER_MODEL


def generate_one_trajectory(
    model,
    initial_qpos,
    initial_qvel,
    seq_torque,
    num_steps: int = 2000,
    skip_steps: int = 1,
    data_dt: float = None,
    early_max_velocity: float | None = None,
    early_max_acceleration: float | None = None,
    early_joint1_abs_limit: float | None = None,
):
    """
    Run forward dynamics simulation and collect data at specified intervals.
    
    Args:
        model: MuJoCo model (with timestep already set to dt)
        initial_qpos: Initial joint positions
        initial_qvel: Initial joint velocities
        seq_torque: Torque sequence at fine simulation resolution (num_sim_steps, torque_dim)
                   where num_sim_steps = num_steps * skip_steps
        num_steps: Number of data points to collect
        skip_steps: Number of simulation steps between data collection points
        data_dt: Data collection timestep (for computing derivatives). If None, uses model.opt.timestep * skip_steps
        
    Returns:
        seq_qpos: (num_steps, nq) pre-step positions
        seq_qvel: (num_steps, nv) pre-step velocities
        seq_qacc: (num_steps, nv) forward-difference acceleration over [t, t + data_dt)
        seq_mom: (num_steps, nv) pre-step momenta
        seq_mom_dot: (num_steps, nv) forward-difference momentum derivative over [t, t + data_dt)
        seq_energy: (num_steps,) total energy (kinetic + potential) at each pre-step state
    """
    data = mujoco.MjData(model)
    
    # Compute data_dt if not provided
    if data_dt is None:
        data_dt = model.opt.timestep * skip_steps

    # Reset data
    mujoco.mj_resetData(model, data)

    # Pass initial qpos and qvel to data
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel

    mujoco.mj_forward(model, data)

    # Pre-allocate arrays for speed (avoid list appends and conversion)
    nq, nv = model.nq, model.nv
    seq_qpos = np.empty((num_steps, nq), dtype=np.float64)
    seq_qvel = np.empty((num_steps, nv), dtype=np.float64)
    seq_qacc = np.empty((num_steps, nv), dtype=np.float64)
    seq_mom = np.empty((num_steps, nv), dtype=np.float64)
    seq_energy = np.empty((num_steps,), dtype=np.float64)
    seq_qacc = np.empty((num_steps, nv), dtype=np.float64)
    seq_mom_dot = np.empty((num_steps, nv), dtype=np.float64)
    
    # Pre-allocate mass matrix
    M = np.zeros((nv, nv), dtype=np.float64)
    
    unstable = False
    
    q1_idx = _get_joint1_qpos_index(model)

    for data_idx in range(num_steps):
        # Check stability/safety at the current pre-step state.
        if np.any(np.isnan(data.qpos)) or np.any(np.abs(data.qvel) > 1e3):
            seq_qpos[data_idx:] = np.nan
            seq_qvel[data_idx:] = np.nan
            seq_qacc[data_idx:] = np.nan
            seq_mom[data_idx:] = np.nan
            seq_mom_dot[data_idx:] = np.nan
            seq_energy[data_idx:] = np.nan
            unstable = True
            break

        if early_max_velocity is not None and np.any(np.abs(data.qvel) > early_max_velocity):
            seq_qpos[data_idx:] = np.nan
            seq_qvel[data_idx:] = np.nan
            seq_qacc[data_idx:] = np.nan
            seq_mom[data_idx:] = np.nan
            seq_mom_dot[data_idx:] = np.nan
            seq_energy[data_idx:] = np.nan
            unstable = True
            break

        if early_max_acceleration is not None and np.any(np.abs(data.qacc) > early_max_acceleration):
            seq_qpos[data_idx:] = np.nan
            seq_qvel[data_idx:] = np.nan
            seq_qacc[data_idx:] = np.nan
            seq_mom[data_idx:] = np.nan
            seq_mom_dot[data_idx:] = np.nan
            seq_energy[data_idx:] = np.nan
            unstable = True
            break

        if (
            early_joint1_abs_limit is not None
            and q1_idx is not None
            and abs(float(data.qpos[q1_idx])) > early_joint1_abs_limit
        ):
            seq_qpos[data_idx:] = np.nan
            seq_qvel[data_idx:] = np.nan
            seq_qacc[data_idx:] = np.nan
            seq_mom[data_idx:] = np.nan
            seq_mom_dot[data_idx:] = np.nan
            seq_energy[data_idx:] = np.nan
            unstable = True
            break

        # Save the synchronized pre-step state x_t.
        seq_qpos[data_idx] = data.qpos
        qvel_curr = data.qvel.copy()
        seq_qvel[data_idx] = qvel_curr

        mujoco.mj_fullM(model, M, data.qM)
        mom_curr = M @ qvel_curr
        seq_mom[data_idx] = mom_curr
        seq_energy[data_idx] = data.energy[0] + data.energy[1]

        # Advance dynamics over [t, t + data_dt).
        for sub_step in range(skip_steps):
            sim_idx = data_idx * skip_steps + sub_step
            data.ctrl[:] = seq_torque[sim_idx]
            mujoco.mj_step(model, data)

        # If the rollout becomes unstable during the interval, invalidate the current
        # sample as well because its forward-difference targets depend on x_{t+1}.
        if np.any(np.isnan(data.qpos)) or np.any(np.abs(data.qvel) > 1e3):
            seq_qpos[data_idx:] = np.nan
            seq_qvel[data_idx:] = np.nan
            seq_qacc[data_idx:] = np.nan
            seq_mom[data_idx:] = np.nan
            seq_mom_dot[data_idx:] = np.nan
            seq_energy[data_idx:] = np.nan
            unstable = True
            break

        if early_max_velocity is not None and np.any(np.abs(data.qvel) > early_max_velocity):
            seq_qpos[data_idx:] = np.nan
            seq_qvel[data_idx:] = np.nan
            seq_qacc[data_idx:] = np.nan
            seq_mom[data_idx:] = np.nan
            seq_mom_dot[data_idx:] = np.nan
            seq_energy[data_idx:] = np.nan
            unstable = True
            break

        if early_max_acceleration is not None and np.any(np.abs(data.qacc) > early_max_acceleration):
            seq_qpos[data_idx:] = np.nan
            seq_qvel[data_idx:] = np.nan
            seq_qacc[data_idx:] = np.nan
            seq_mom[data_idx:] = np.nan
            seq_mom_dot[data_idx:] = np.nan
            seq_energy[data_idx:] = np.nan
            unstable = True
            break

        if (
            early_joint1_abs_limit is not None
            and q1_idx is not None
            and abs(float(data.qpos[q1_idx])) > early_joint1_abs_limit
        ):
            seq_qpos[data_idx:] = np.nan
            seq_qvel[data_idx:] = np.nan
            seq_qacc[data_idx:] = np.nan
            seq_mom[data_idx:] = np.nan
            seq_mom_dot[data_idx:] = np.nan
            seq_energy[data_idx:] = np.nan
            unstable = True
            break

        mujoco.mj_fullM(model, M, data.qM)
        mom_next = M @ data.qvel
        seq_qacc[data_idx] = (data.qvel - qvel_curr) / data_dt
        seq_mom_dot[data_idx] = (mom_next - mom_curr) / data_dt
        
    return seq_qpos, seq_qvel, seq_qacc, seq_mom, seq_mom_dot, seq_energy

def generate(
    model,
    lim_qpos=math.pi / 3,
    lim_qvel=math.pi / 3,
    num_sin=5,
    lim_amplitude=0.5,
    lim_frequency=6 * math.pi,
    lim_phase=2 * math.pi,
    num_steps: int = 1000,
    skip_steps: int = 1,
    data_dt: float = None,
    torque_policy='sinusoidal',
    ref_std_per_dim=None,
    torque_scale: float = 1.0,
    ctrl_margin: float = 0.05,
    lpf_uniform_beta: float = 0.992,
    drop_target_states: bool = True,
    gym_reacher_target_sampling: bool = True,
    gym_reacher_state_sampling: bool = True,
    early_max_velocity: float | None = None,
    early_max_acceleration: float | None = None,
    early_joint1_abs_limit: float | None = None,
):
    """
    Generate a single trajectory with random initial conditions and torque.

    Args:
        model: MuJoCo model (with timestep already set to dt)
        lim_qpos: Limit for random initial position sampling
        lim_qvel: Limit for random initial velocity sampling
        num_sin: Number of sinusoids for torque generation (default: 5 for smoother torques)
        lim_amplitude: Maximum torque amplitude (default: 0.5 for bounded energy injection)
        lim_frequency: Maximum torque frequency (default: 25 for more work cancellation)
        lim_phase: Maximum torque phase
        num_steps: Number of data points to collect
        skip_steps: Number of simulation steps between data collection points
        data_dt: Data collection timestep (for metadata). If None, uses model.opt.timestep * skip_steps
        torque_policy: Torque generation policy name
            (sinusoidal, gp, chirp, step, ou, drift_ou, spline, uniform_iid, lpf_uniform, zero)
        ref_std_per_dim: Reference per-dim std for amplitude matching (required for non-sinusoidal policies)
        drop_target_states: Compatibility behavior:
            - True: save only actuated arm dimensions in canonical keys.
            - False: still keep canonical keys arm-only for downstream compatibility,
              and additionally store full-state arrays in optional *_full keys.

    Returns:
        result: dict with seq_qpos, seq_qvel, seq_qacc, seq_mom, seq_mom_dot, seq_torque, seq_energy, torque_policy
               All sequences have length num_steps (data collection points)
               seq_torque is subsampled from fine resolution to match data collection
    """
    # Compute data_dt if not provided
    if data_dt is None:
        data_dt = model.opt.timestep * skip_steps

    # Randomly sample initial position and velocity
    initial_qpos, initial_qvel = randomly_initialize_qpos_qvel_qacc(
        model,
        lim_qpos=lim_qpos,
        lim_qvel=lim_qvel,
        gym_reacher_target_sampling=gym_reacher_target_sampling,
        gym_reacher_state_sampling=gym_reacher_state_sampling,
    )

    # Generate torque at fine simulation resolution using selected policy
    if torque_policy == 'sinusoidal':
        seq_torque_fine = generate_random_seq_torque(
            model, num_steps=num_steps, skip_steps=skip_steps,
            num_sin=num_sin, lim_amplitude=lim_amplitude,
            lim_frequency=lim_frequency, lim_phase=lim_phase
        )
    elif torque_policy == 'zero':
        num_sim_steps = num_steps * skip_steps
        seq_torque_fine = np.zeros((num_sim_steps, model.nu))
    else:
        seq_torque_fine = generate_torque_sequence(
            model, num_steps, skip_steps,
            policy=torque_policy, ref_std_per_dim=ref_std_per_dim,
            ctrl_margin=ctrl_margin,
            lpf_uniform_beta=lpf_uniform_beta,
        )

    # Lowering torque scale increases acceptance under strict safety filters.
    seq_torque_fine = float(torque_scale) * seq_torque_fine

    # Generate trajectories
    seq_qpos, seq_qvel, seq_qacc, seq_mom, seq_mom_dot, seq_energy = generate_one_trajectory(
        model, initial_qpos, initial_qvel, seq_torque_fine,
        num_steps=num_steps, skip_steps=skip_steps, data_dt=data_dt,
        early_max_velocity=early_max_velocity,
        early_max_acceleration=early_max_acceleration,
        early_joint1_abs_limit=early_joint1_abs_limit,
    )

    # Store the interval-mean control so (q_t, p_t, tau_t, dpdt_t) share the same interval.
    seq_torque = seq_torque_fine.reshape(num_steps, skip_steps, model.nu).mean(axis=1)
    ctrl_low, ctrl_high = _get_control_ranges(model)
    q1_idx = _get_joint1_qpos_index(model)
    if q1_idx is not None:
        seq_joint1 = seq_qpos[:, q1_idx].copy()
    else:
        seq_joint1 = np.full((num_steps,), np.nan, dtype=np.float64)

    # Preserve full states for optional diagnostics/reachability export.
    # Canonical keys remain arm-only so existing DPF/HNN pipelines are unchanged.
    seq_qpos_full = seq_qpos.copy()
    seq_qvel_full = seq_qvel.copy()
    seq_qacc_full = seq_qacc.copy()
    seq_mom_full = seq_mom.copy()
    seq_mom_dot_full = seq_mom_dot.copy()

    arm_dim = int(model.nu)
    seq_qpos_arm = seq_qpos_full[:, :arm_dim]
    seq_qvel_arm = seq_qvel_full[:, :arm_dim]
    seq_qacc_arm = seq_qacc_full[:, :arm_dim]
    seq_mom_arm = seq_mom_full[:, :arm_dim]
    seq_mom_dot_arm = seq_mom_dot_full[:, :arm_dim]

    result = {
        # Canonical training keys (arm-only, stable schema for existing scripts).
        'seq_qpos': seq_qpos_arm[:],
        'seq_qvel': seq_qvel_arm[:],
        'seq_qacc': seq_qacc_arm[:],
        'seq_mom': seq_mom_arm[:],
        'seq_mom_dot': seq_mom_dot_arm[:],
        'seq_torque': seq_torque[:],
        'seq_energy': seq_energy[:],
        'torque_policy': torque_policy,
        # Extra metadata used only for filtering/diagnostics (not saved to HDF5).
        'ctrl_low': ctrl_low,
        'ctrl_high': ctrl_high,
        'saturation_frac': _saturation_fraction(seq_torque_fine, ctrl_low, ctrl_high),
        'seq_joint1': seq_joint1,
    }

    # Optional full-state export for target-aware analysis/reachability.
    # These extra keys are ignored by existing training/eval loaders.
    if not drop_target_states:
        result['seq_qpos_full'] = seq_qpos_full[:]
        result['seq_qvel_full'] = seq_qvel_full[:]
        result['seq_qacc_full'] = seq_qacc_full[:]
        result['seq_mom_full'] = seq_mom_full[:]
        result['seq_mom_dot_full'] = seq_mom_dot_full[:]
        if seq_qpos_full.shape[1] > arm_dim:
            result['seq_target'] = seq_qpos_full[:, arm_dim:]

    return result

def init_plot(dim: int, title: str):
    """
    Initializes a figure with subplots for each dimension.
    Args:
        dim: Number of dimensions (rows).
        title: Title of the figure.
    Returns:
        fig: The matplotlib figure object.
        axes: Array of axes objects.
    """
    fig, axes = plt.subplots(dim, 1, figsize=(10, 10), dpi=150, sharex=True)
    if dim == 1:
        axes = np.array([axes])
    fig.suptitle(title)
    return fig, axes

def plot_on_axes(axes, seq, label: str = None):
    """
    Plots a sequence onto existing axes.
    Args:
        axes: Array of axes objects.
        seq: Sequence data of shape (num_steps, dim).
        label: Label for the plot legend.
    """
    steps = np.arange(seq.shape[0])
    dim = seq.shape[1]

    for i in range(dim):
        axes[i].scatter(steps, seq[:,i], s=1, alpha=0.6)

def generate_single_trajectory_task(args):
    """
    Worker function for multiprocessing.
    MuJoCo models cannot be pickled, so we load the model in each worker process.

    Args:
        args: Tuple of (xml_path, num_steps, dt, data_dt, seed, torque_policy, ref_std_per_dim)
              - xml_path: Path to MuJoCo XML model
              - num_steps: Number of data points to collect
              - dt: Fine simulation timestep
              - data_dt: Data collection timestep (coarser)
              - seed: Random seed for this trajectory
              - torque_policy: Name of torque generation policy
              - ref_std_per_dim: Reference per-dim std for amplitude matching

    Returns:
        Result from generate() function
    """
    (
        xml_path,
        num_steps,
        dt,
        data_dt,
        seed,
        torque_policy,
        ref_std_per_dim,
        torque_scale,
        early_max_velocity,
        early_max_acceleration,
        early_joint1_abs_limit,
        ctrl_margin,
        lpf_uniform_beta,
        drop_target_states,
        gym_reacher_target_sampling,
        gym_reacher_state_sampling,
    ) = args
    # Set the seed for this process to ensure different trajectories
    np.random.seed(seed)

    # Compute skip_steps from timesteps and enforce consistency
    skip_steps = max(1, int(round(data_dt / dt)))
    data_dt = skip_steps * dt  # ensure no mismatch with actual collection interval

    # Reuse a cached model in each process to avoid XML parsing overhead per task.
    model = _get_or_create_worker_model(xml_path, dt)

    return generate(model, num_steps=num_steps, skip_steps=skip_steps, data_dt=data_dt,
                    torque_policy=torque_policy, ref_std_per_dim=ref_std_per_dim,
                    torque_scale=torque_scale,
                    ctrl_margin=ctrl_margin,
                    lpf_uniform_beta=lpf_uniform_beta,
                    drop_target_states=drop_target_states,
                    gym_reacher_target_sampling=gym_reacher_target_sampling,
                    gym_reacher_state_sampling=gym_reacher_state_sampling,
                    early_max_velocity=early_max_velocity,
                    early_max_acceleration=early_max_acceleration,
                    early_joint1_abs_limit=early_joint1_abs_limit)

def plot_coverage(results):
    nrows = results[0]['seq_qpos'].shape[1]
    ncols = results[0]['seq_mom'].shape[1]
    fig, axes = plt.subplots(nrows, ncols, sharey=True, figsize=(40, 20), dpi=150)

    for result in results:
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j].scatter(result['seq_qpos'][:, i], result['seq_mom'][:,j], s=0.1, alpha=0.5)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / 'coverage_forward.jpg')
    plt.close()

def plot_data(results):
    """
    Plot each seq_xxx (pos, vel, acc, momentum, momentum_dot, torque).
    Each plot shows all dimensions for all trajectories over time.
    Uses vectorized plotting for speed.
    """
    if not results:
        print("No results to plot.")
        return
    
    keys = ['seq_qpos', 'seq_qvel', 'seq_qacc', 'seq_mom', 'seq_mom_dot', 'seq_torque']
    save_dir = PLOTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    colors = plt.cm.tab10.colors  # Use a colormap for dimensions
    
    for key in keys:
        # Stack as (num_trajectories, num_steps, dim)
        arr = np.stack([result[key] for result in results], axis=0)
        num_traj, num_steps, dim = arr.shape

        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Create x-coordinates: tile timesteps for all trajectories
        x = np.tile(np.arange(num_steps), num_traj)  # shape: (num_traj * num_steps,)
        
        for d in range(dim):
            # Flatten all trajectories for this dimension
            y = arr[:, :, d].ravel()  # shape: (num_traj * num_steps,)
            ax.scatter(x, y, s=1, alpha=0.3, c=[colors[d % len(colors)]], label=f'dim {d}')

        ax.set_xlabel('Timestep')
        ax.set_ylabel(key)
        ax.set_title(f"{key}: all trajectories & all dims")
        ax.legend(loc='upper right', markerscale=5)
        fig.tight_layout()
        fig.savefig(save_dir / f"{key}_trajectories.png")
        plt.close(fig)

def show_statistics(results):
    """
    Concisely print statistics for the list of result dicts.
    Each dict contains arrays for seq_qpos, seq_qvel, seq_qacc, seq_mom, seq_mom_dot, seq_torque, seq_energy.
    """
    if not results:
        print("No results to compute statistics.")
        return
    
    keys = TRAJ_KEYS
    stats = {}
    for key in keys:
        data = np.concatenate([r[key] for r in results], axis=0)
        stats[key] = {
            "shape": data.shape,
            "min": np.min(data),
            "max": np.max(data),
            "mean": np.mean(data),
            "std": np.std(data),
        }
    print("---- Statistics ----")
    for key in keys:
        s = stats[key]
        print(f"{key}: shape={s['shape']}, min={s['min']:.4f}, max={s['max']:.4f}, mean={s['mean']:.4f}, std={s['std']:.4f}")
    print("--------------------")


def init_running_stats():
    """Initialize streaming statistics accumulators.

    We keep running min/max/mean/std without storing all trajectories in memory.
    """
    stats = {}
    for key in TRAJ_KEYS:
        stats[key] = {
            "rows": 0,
            "tail_shape": None,
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
        }
    return stats


def update_running_stats(stats, result):
    """Update running statistics from one accepted trajectory."""
    for key in TRAJ_KEYS:
        arr = np.asarray(result[key])
        entry = stats[key]
        if entry["tail_shape"] is None:
            entry["tail_shape"] = arr.shape[1:]
        entry["rows"] += int(arr.shape[0]) if arr.ndim > 0 else 1
        entry["count"] += int(arr.size)
        entry["sum"] += float(arr.sum())
        entry["sum_sq"] += float(np.square(arr).sum())
        entry["min"] = min(entry["min"], float(arr.min()))
        entry["max"] = max(entry["max"], float(arr.max()))


def print_running_stats(stats):
    """Print stats computed from running accumulators."""
    print("---- Statistics ----")
    for key in TRAJ_KEYS:
        entry = stats[key]
        if entry["count"] <= 0:
            print(f"{key}: empty")
            continue
        mean = entry["sum"] / entry["count"]
        var = max(0.0, entry["sum_sq"] / entry["count"] - mean * mean)
        std = float(np.sqrt(var))
        shape = (entry["rows"],) + tuple(entry["tail_shape"] or ())
        print(
            f"{key}: shape={shape}, min={entry['min']:.4f}, max={entry['max']:.4f}, "
            f"mean={mean:.4f}, std={std:.4f}"
        )
    print("--------------------")


def _write_h5_header(file, xml_path, num_steps, num_trajectories, dt, data_dt, policy_weights=None):
    """Write HDF5 metadata header in the existing training-data format."""
    with open(xml_path, 'r') as xml_file:
        file.attrs['xml'] = xml_file.read()
    file.attrs['num_steps'] = num_steps
    file.attrs['num_trajectories'] = num_trajectories
    file.attrs['dt'] = dt
    file.attrs['data_dt'] = data_dt
    file.attrs['skip_steps'] = int(round(data_dt / dt))
    file.attrs['state_alignment'] = 'pre_step'
    file.attrs['torque_alignment'] = 'interval_mean'
    file.attrs['derivative_alignment'] = 'forward_difference'
    if policy_weights is not None:
        file.attrs['torque_policies'] = json.dumps(policy_weights)


def _write_trajectory_group(file, traj_index, result):
    """Write one accepted trajectory group to HDF5."""
    group = file.create_group(f'traj_{traj_index}')
    group.create_dataset('seq_qpos', data=result['seq_qpos'], dtype='f4')
    group.create_dataset('seq_qvel', data=result['seq_qvel'], dtype='f4')
    group.create_dataset('seq_qacc', data=result['seq_qacc'], dtype='f4')
    group.create_dataset('seq_mom', data=result['seq_mom'], dtype='f4')
    group.create_dataset('seq_mom_dot', data=result['seq_mom_dot'], dtype='f4')
    group.create_dataset('seq_torque', data=result['seq_torque'], dtype='f4')
    group.create_dataset('seq_energy', data=result['seq_energy'], dtype='f4')
    # Optional extra channels for target-aware analysis while keeping canonical keys unchanged.
    if 'seq_qpos_full' in result:
        group.create_dataset('seq_qpos_full', data=result['seq_qpos_full'], dtype='f4')
    if 'seq_qvel_full' in result:
        group.create_dataset('seq_qvel_full', data=result['seq_qvel_full'], dtype='f4')
    if 'seq_qacc_full' in result:
        group.create_dataset('seq_qacc_full', data=result['seq_qacc_full'], dtype='f4')
    if 'seq_mom_full' in result:
        group.create_dataset('seq_mom_full', data=result['seq_mom_full'], dtype='f4')
    if 'seq_mom_dot_full' in result:
        group.create_dataset('seq_mom_dot_full', data=result['seq_mom_dot_full'], dtype='f4')
    if 'seq_target' in result:
        group.create_dataset('seq_target', data=result['seq_target'], dtype='f4')
    group.attrs['torque_policy'] = np.bytes_(result.get('torque_policy', 'sinusoidal'))

def is_available(
    result,
    lim_energy_ratio=10.0,
    max_energy_limit=1000.0,
    max_velocity=50.0,
    max_acceleration=500.0,
    max_abs_mom_dot=None,
    safe_mode=True,
    joint_margin=0.2,
    ctrl_margin=0.2,
    max_saturation_frac=0.0,
):
    """
    Check if trajectory is stable using energy and kinematic filtering.
    
    Args:
        result: dict containing arrays for 'seq_qpos', 'seq_qvel', 'seq_qacc', 'seq_energy'
        lim_energy_ratio: Maximum allowed ratio of max/initial energy
        max_energy_limit: Hard limit on maximum energy value
        max_velocity: Maximum allowed angular velocity (rad/s)
        max_acceleration: Maximum allowed angular acceleration (rad/s²)
        
    Returns:
        tuple[bool, str]: (accepted, reason)
        
    Filtering criteria:
        - NaN/Inf values (numerical instability)
        - Gimbal lock avoidance for 3-hinge model
        - Energy bounds (ratio and absolute)
        - Kinematic bounds (velocity and acceleration)
    """
    seq_qpos = result['seq_qpos']    # shape: (num_steps, nq)
    seq_qvel = result['seq_qvel']    # shape: (num_steps, nv)
    seq_qacc = result['seq_qacc']    # shape: (num_steps, na)
    seq_energy = result['seq_energy']  # shape: (num_steps,)

    # Check for NaNs or Infs (numerical instability)
    if np.any(np.isnan(seq_qpos)) or np.any(np.isnan(seq_qvel)) or np.any(np.isnan(seq_qacc)):
        return False, "nan_or_inf"
    if np.any(np.isinf(seq_qpos)) or np.any(np.isinf(seq_qvel)) or np.any(np.isinf(seq_qacc)):
        return False, "nan_or_inf"
    if np.any(np.isnan(seq_energy)) or np.any(np.isinf(seq_energy)):
        return False, "nan_or_inf"
    
    # Kinematic limits (physically plausible dynamics)
    if np.any(np.abs(seq_qvel) > max_velocity):
        return False, "velocity"
    if np.any(np.abs(seq_qacc) > max_acceleration):
        return False, "acceleration"
    if max_abs_mom_dot is not None:
        seq_mom_dot = result.get("seq_mom_dot", None)
        if seq_mom_dot is None:
            return False, "mom_dot_missing"
        if np.any(np.abs(seq_mom_dot) > float(max_abs_mom_dot)):
            return False, "mom_dot"

    # Energy-based filtering: check that energy doesn't grow unboundedly
    initial_energy = np.abs(seq_energy[0]) + 1.0  # Add 1.0 to avoid division by zero
    max_energy = np.max(np.abs(seq_energy))
    energy_ratio = max_energy / initial_energy
    
    if energy_ratio > lim_energy_ratio:
        return False, "energy_ratio"
    
    # Hard limit on maximum energy
    if max_energy > max_energy_limit:
        return False, "energy_max"

    if safe_mode:
        # Safety margin avoids non-smooth dynamics near hard joint limits.
        q1 = result.get("seq_joint1", None)
        if q1 is not None and np.any(np.isfinite(q1)):
            if np.any(np.abs(q1) > (3.0 - joint_margin)):
                return False, "joint_limit_margin"

        # Keep controls away from actuator bounds to avoid clipping artifacts.
        torque = result.get("seq_torque", None)
        ctrl_low = result.get("ctrl_low", None)
        ctrl_high = result.get("ctrl_high", None)
        if torque is not None and ctrl_low is not None and ctrl_high is not None:
            low_safe = ctrl_low + ctrl_margin
            high_safe = ctrl_high - ctrl_margin
            if np.any(torque < low_safe[None, :]) or np.any(torque > high_safe[None, :]):
                return False, "ctrl_margin"

        if result.get("saturation_frac", 0.0) > max_saturation_frac:
            return False, "saturation"

    return True, "accepted"

def save_as_h5py(results, save_path, xml_path, num_steps, num_trajectories, dt, data_dt,
                 policy_weights=None):
    """
    Save trajectory results to HDF5 file.

    Args:
        results: List of trajectory result dicts
        save_path: Directory to save the file
        xml_path: Path to MuJoCo XML model
        num_steps: Number of data points per trajectory
        num_trajectories: Number of trajectories
        dt: Fine simulation timestep
        data_dt: Data collection timestep
        policy_weights: Dict of {policy_name: weight} used for generation
    """
    # Create directory if it doesn't exist (including parent directories)
    # Don't use os.mkdir(), cause it only creates one level
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f'traj_{num_trajectories}-steps_{num_steps}.h5')

    # plot_coverage(results)
    # plot_data(results)
    show_statistics(results)

    # Don't forget to set the write mode
    with h5py.File(file_path, 'w') as file:
        _write_h5_header(file, xml_path, num_steps, num_trajectories, dt, data_dt, policy_weights)

        for i, result in tqdm(enumerate(results), total=len(results), desc='Saving'):
            _write_trajectory_group(file, i, result)


def parse_torque_policies(policy_str):
    """Parse torque policy specification string into (names, weights).

    Format: "sinusoidal:0.25,gp:0.25,chirp:0.25,spline:0.25"
    Returns:
        policy_names: list of str
        policy_weights: list of float (normalized to sum to 1)
    """
    names, weights = [], []
    for item in policy_str.split(','):
        item = item.strip()
        if ':' in item:
            name, weight = item.split(':')
            names.append(name.strip())
            weights.append(float(weight.strip()))
        else:
            names.append(item)
            weights.append(1.0)
    total = sum(weights)
    weights = [w / total for w in weights]
    return names, weights


def compute_policy_targets(policy_names, policy_weights, num_trajectories):
    """Compute exact integer accepted counts per policy using largest remainder.

    This enforces strict final policy ratios in the saved dataset while keeping
    totals exact and deterministic for a given (weights, num_trajectories).
    """
    expected = np.array(policy_weights, dtype=np.float64) * float(num_trajectories)
    base = np.floor(expected).astype(int)
    remainder = expected - base
    shortfall = int(num_trajectories - int(base.sum()))
    if shortfall > 0:
        # Give leftover samples to the largest fractional remainders.
        order = np.argsort(-remainder)
        for idx in order[:shortfall]:
            base[idx] += 1
    return {name: int(cnt) for name, cnt in zip(policy_names, base)}


def generate_split(xml_path, num_steps, dt, data_dt, num_trajectories, batch_size,
                   policy_names, policy_weights, ref_std_per_dim, num_workers,
                   lim_energy_ratio, max_energy_limit, max_velocity, max_acceleration,
                   max_abs_mom_dot,
                   safe_mode, joint_margin, ctrl_margin, max_saturation_frac,
                   torque_scale, lpf_uniform_beta, drop_target_states, gym_reacher_target_sampling,
                   gym_reacher_state_sampling, strict_policy_ratios, stream_file=None,
                   seed_offset=0, desc="Collecting trajectories"):
    """Generate a dataset split (train or val) with diverse torque policies.

    Returns list of result dicts.
    """
    available_results = [] if stream_file is None else None
    total_generated = 0
    seed_counter = seed_offset
    accepted_count = 0
    running_stats = init_running_stats()
    reject_counts = {
        "nan_or_inf": 0,
        "velocity": 0,
        "acceleration": 0,
        "mom_dot": 0,
        "mom_dot_missing": 0,
        "energy_ratio": 0,
        "energy_max": 0,
        "joint_limit_margin": 0,
        "ctrl_margin": 0,
        "saturation": 0,
        "other": 0,
    }
    accepted_policy_counts = {name: 0 for name in policy_names}
    policy_targets = compute_policy_targets(policy_names, policy_weights, num_trajectories)

    pbar = tqdm(total=num_trajectories, desc=desc)

    # Use single-process fallback for restricted environments where multiprocessing
    # semaphores are unavailable; this also helps small pilot/debug runs.
    use_parallel = num_workers is not None and num_workers > 1
    executor = None
    if use_parallel:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker_model,
            initargs=(xml_path, dt),
        )
    else:
        # Also cache model in single-process mode for faster pilot/debug runs.
        _init_worker_model(xml_path, dt)

    try:
        while accepted_count < num_trajectories:
            seeds = np.random.randint(0, 999999, (batch_size,)) + seed_counter
            seed_counter += batch_size

            # Strict-ratio mode samples only from policies that still need accepted
            # trajectories. This makes final saved counts match target ratios.
            if strict_policy_ratios:
                active_names = [p for p in policy_names if accepted_policy_counts[p] < policy_targets[p]]
                if not active_names:
                    break
                remaining = np.array(
                    [policy_targets[p] - accepted_policy_counts[p] for p in active_names],
                    dtype=np.float64,
                )
                remaining = remaining / remaining.sum()
                policies = np.random.choice(active_names, size=batch_size, p=remaining)
            else:
                # Non-strict mode keeps historical behavior: weighted proposal mix.
                policies = np.random.choice(policy_names, size=batch_size, p=policy_weights)

            tasks = [
                (
                    xml_path,
                    num_steps,
                    dt,
                    data_dt,
                    int(seed),
                    policy,
                    ref_std_per_dim,
                    torque_scale,
                    max_velocity if safe_mode else None,
                    max_acceleration if safe_mode else None,
                    (3.0 - joint_margin) if safe_mode else None,
                    ctrl_margin,
                    lpf_uniform_beta,
                    drop_target_states,
                    gym_reacher_target_sampling,
                    gym_reacher_state_sampling,
                )
                for seed, policy in zip(seeds, policies)
            ]
            if use_parallel:
                # Chunksize>1 reduces multiprocessing scheduling overhead.
                chunksize = max(1, len(tasks) // max(1, num_workers * 8))
                results = list(executor.map(generate_single_trajectory_task, tasks, chunksize=chunksize))
            else:
                results = [generate_single_trajectory_task(task) for task in tasks]
            total_generated += len(results)

            for r in results:
                ok, reason = is_available(
                    r,
                    lim_energy_ratio=lim_energy_ratio,
                    max_energy_limit=max_energy_limit,
                    max_velocity=max_velocity,
                    max_acceleration=max_acceleration,
                    max_abs_mom_dot=max_abs_mom_dot,
                    safe_mode=safe_mode,
                    joint_margin=joint_margin,
                    ctrl_margin=ctrl_margin,
                    max_saturation_frac=max_saturation_frac,
                )
                if ok:
                    p = r.get("torque_policy", "sinusoidal")
                    if strict_policy_ratios and accepted_policy_counts.get(p, 0) >= policy_targets.get(p, 0):
                        # In strict mode, never exceed a policy quota.
                        continue
                    if stream_file is None:
                        available_results.append(r)
                    else:
                        _write_trajectory_group(stream_file, accepted_count, r)
                    update_running_stats(running_stats, r)
                    accepted_count += 1
                    accepted_policy_counts[p] = accepted_policy_counts.get(p, 0) + 1
                    if accepted_count >= num_trajectories:
                        break
                else:
                    reject_counts[reason] = reject_counts.get(reason, 0) + 1

            pbar.n = min(accepted_count, num_trajectories)
            pbar.set_postfix(generated=total_generated,
                             accept_rate=f"{accepted_count/total_generated:.1%}")
            pbar.refresh()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    pbar.close()

    if available_results is not None:
        available_results = available_results[:num_trajectories]

    print(f"  Collected {accepted_count} trajectories "
          f"(generated {total_generated}, accept rate: {accepted_count/total_generated:.1%})")
    print("  Rejections by reason:")
    for key in [
        "nan_or_inf",
        "velocity",
        "acceleration",
        "mom_dot",
        "mom_dot_missing",
        "energy_ratio",
        "energy_max",
        "joint_limit_margin",
        "ctrl_margin",
        "saturation",
        "other",
    ]:
        print(f"    - {key}: {reject_counts.get(key, 0)}")
    print(f"  Policy targets: {policy_targets}")
    print(f"  Accepted per-policy counts: {accepted_policy_counts}")

    return {
        "results": available_results,
        "running_stats": running_stats,
        "accepted_count": accepted_count,
        "generated_count": total_generated,
        "policy_targets": policy_targets,
        "accepted_policy_counts": accepted_policy_counts,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate forward-dynamics trajectory dataset")
    parser.add_argument("--save_path", type=str,
                        default=str(DATA_DIR))
    parser.add_argument("--xml_path", type=str,
                        default='/home/gsang/miniconda3/envs/perceiver/lib/python3.10/site-packages/gymnasium/envs/mujoco/assets/reacher.xml')
    parser.add_argument("--num_steps", type=int, default=2000,
                        help="Data collection points per trajectory")
    parser.add_argument("--num_trajectories", type=int, default=80000,
                        help="Number of training trajectories")
    parser.add_argument("--num_val", type=int, default=4000,
                        help="Number of validation trajectories")
    # dt < data_dt reduces integration error while keeping the same saved sample rate.
    parser.add_argument("--dt", type=float, default=0.00005,
                        help="Fine simulation timestep")
    parser.add_argument("--data_dt", type=float, default=0.0002,
                        help="Data collection timestep (enforced to skip_steps * dt)")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Batch size for parallel generation")
    parser.add_argument("--num_workers", type=int, default=24,
                        help="Number of parallel workers")
    parser.add_argument("--torque_policies", type=str,
                        # Restrict to smoother policies for Reacher to improve safe acceptance.
                        default="sinusoidal:0.45,gp:0.25,spline:0.20,zero:0.10",
                        help="Comma-separated policy:weight pairs")
    parser.add_argument("--safe_mode", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable safety-margin filtering for joint/control limits")
    parser.add_argument("--joint_margin", type=float, default=0.05,
                        help="Safety margin from joint1 hard limit ±3.0 rad")
    parser.add_argument("--ctrl_margin", type=float, default=0.05,
                        help="Safety margin from actuator ctrlrange bounds")
    parser.add_argument("--max_saturation_frac", type=float, default=0.0,
                        help="Max allowed fraction of control proposals outside actuator range")
    parser.add_argument("--non_dissipative", action=argparse.BooleanOptionalAction, default=True,
                        help="Patch XML to zero damping/friction losses before generation")
    parser.add_argument("--torque_scale", type=float, default=0.35,
                        help="Global multiplier on generated torques to improve safe acceptance")
    parser.add_argument("--lpf_uniform_beta", type=float, default=0.992,
                        help="Smoothing factor for lpf_uniform torque policy (higher = smoother)")
    parser.add_argument("--stream_write", action=argparse.BooleanOptionalAction, default=True,
                        help="Write accepted trajectories incrementally to HDF5 (lower RAM, slightly slower)")
    parser.add_argument("--drop_target_states", action=argparse.BooleanOptionalAction, default=True,
                        help="Save only arm states (joint0/joint1) instead of including target_x/target_y")
    parser.add_argument(
        "--gym_reacher_target_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If target_x/target_y joints exist, sample target uniformly in disk (||goal||<0.2) like Gym Reacher",
    )
    parser.add_argument(
        "--gym_reacher_state_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If Reacher target joints exist, sample qpos/qvel with Reacher-v5 reset style "
             "(qpos noise in [-0.1,0.1], qvel noise in [-0.005,0.005], target qvel set to 0).",
    )
    parser.add_argument("--strict_policy_ratios", action=argparse.BooleanOptionalAction, default=True,
                        help="Enforce exact accepted trajectory count per policy from --torque_policies")
    # Filter limits
    parser.add_argument("--lim_energy_ratio", type=float, default=5.0)
    parser.add_argument("--max_energy_limit", type=float, default=500.0)
    parser.add_argument("--max_velocity", type=float, default=10.0)
    parser.add_argument("--max_acceleration", type=float, default=100.0)
    parser.add_argument("--max_abs_mom_dot", type=float, default=None,
                        help="Reject a trajectory if any |mom_dot| exceeds this threshold")
    args = parser.parse_args()

    # Parse policies
    policy_names, policy_weights = parse_torque_policies(args.torque_policies)
    policy_dict = dict(zip(policy_names, policy_weights))

    skip_steps = max(1, int(round(args.data_dt / args.dt)))
    # Enforce data_dt = skip_steps * dt so there is no mismatch between
    # the actual collection interval and the value used for derivatives.
    args.data_dt = skip_steps * args.dt
    xml_path_for_run = args.xml_path
    tmp_xml_to_cleanup = None
    if args.non_dissipative:
        # Non-dissipative patch makes collected data closer to controlled Hamiltonian dynamics.
        tmp_xml_to_cleanup = _build_non_dissipative_xml(args.xml_path)
        xml_path_for_run = tmp_xml_to_cleanup

    print(f"Configuration:")
    print(f"  XML path: {xml_path_for_run}")
    print(f"  Simulation timestep (dt): {args.dt}")
    print(f"  Data collection timestep (data_dt): {args.data_dt}")
    print(f"  Skip steps: {skip_steps} (collect every {skip_steps} simulation steps)")
    print(f"  Total simulation steps per trajectory: {args.num_steps * skip_steps}")
    print(f"  Data points per trajectory: {args.num_steps}")
    print(f"  Training trajectories: {args.num_trajectories}")
    print(f"  Validation trajectories: {args.num_val}")
    print(f"  Torque policies: {policy_dict}")
    print(f"  Safe mode: {args.safe_mode}")
    print(f"  Joint margin: {args.joint_margin}")
    print(f"  Control margin: {args.ctrl_margin}")
    print(f"  Max saturation fraction: {args.max_saturation_frac}")
    print(f"  Non-dissipative: {args.non_dissipative}")
    print(f"  Torque scale: {args.torque_scale}")
    print(f"  LPF-uniform beta: {args.lpf_uniform_beta}")
    print(f"  Stream write: {args.stream_write}")
    print(f"  Drop target states: {args.drop_target_states}")
    print(f"  Gym Reacher target sampling: {args.gym_reacher_target_sampling}")
    print(f"  Gym Reacher state sampling: {args.gym_reacher_state_sampling}")
    print(f"  Strict policy ratios: {args.strict_policy_ratios}")
    print(f"  Max |mom_dot|: {args.max_abs_mom_dot}")
    print(f"  Workers: {args.num_workers}")

    # Compute reference std for amplitude matching
    print("\nComputing sinusoidal reference statistics for amplitude matching...")
    ref_std_per_dim = compute_reference_std(xml_path_for_run, args.num_steps,
                                            args.dt, args.data_dt)
    print(f"  ref_std_per_dim = {ref_std_per_dim}")

    try:
        # ── Generate training set ─────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  TRAINING SET ({args.num_trajectories} trajectories)")
        print(f"{'='*60}")
        os.makedirs(args.save_path, exist_ok=True)
        if args.stream_write:
            train_file_path = os.path.join(
                args.save_path, f'traj_{args.num_trajectories}-steps_{args.num_steps}.h5'
            )
            with h5py.File(train_file_path, 'w') as train_file:
                _write_h5_header(
                    train_file,
                    xml_path_for_run,
                    args.num_steps,
                    args.num_trajectories,
                    args.dt,
                    args.data_dt,
                    policy_weights=policy_dict,
                )
                train_summary = generate_split(
                    xml_path_for_run, args.num_steps, args.dt, args.data_dt,
                    args.num_trajectories, args.batch_size,
                    policy_names, policy_weights, ref_std_per_dim, args.num_workers,
                    args.lim_energy_ratio, args.max_energy_limit,
                    args.max_velocity, args.max_acceleration, args.max_abs_mom_dot,
                    args.safe_mode, args.joint_margin, args.ctrl_margin, args.max_saturation_frac,
                    args.torque_scale, args.lpf_uniform_beta, args.drop_target_states,
                    args.gym_reacher_target_sampling, args.gym_reacher_state_sampling,
                    args.strict_policy_ratios, stream_file=train_file,
                    seed_offset=0, desc="Collecting training trajectories",
                )
            print_running_stats(train_summary["running_stats"])
        else:
            # Peak-speed mode: keep accepted trajectories in memory, then save in one shot.
            # This is faster but can use very large RAM for big datasets.
            train_summary = generate_split(
                xml_path_for_run, args.num_steps, args.dt, args.data_dt,
                args.num_trajectories, args.batch_size,
                policy_names, policy_weights, ref_std_per_dim, args.num_workers,
                args.lim_energy_ratio, args.max_energy_limit,
                args.max_velocity, args.max_acceleration, args.max_abs_mom_dot,
                args.safe_mode, args.joint_margin, args.ctrl_margin, args.max_saturation_frac,
                args.torque_scale, args.lpf_uniform_beta, args.drop_target_states,
                args.gym_reacher_target_sampling, args.gym_reacher_state_sampling,
                args.strict_policy_ratios, stream_file=None,
                seed_offset=0, desc="Collecting training trajectories",
            )
            print_running_stats(train_summary["running_stats"])
            save_as_h5py(
                train_summary["results"],
                args.save_path,
                xml_path_for_run,
                args.num_steps,
                args.num_trajectories,
                args.dt,
                args.data_dt,
                policy_weights=policy_dict,
            )
        print(f"Training set saved to {args.save_path}")

        # ── Generate validation set ───────────────────────────────────────────
        if args.num_val > 0:
            print(f"\n{'='*60}")
            print(f"  VALIDATION SET ({args.num_val} trajectories)")
            print(f"{'='*60}")
            if args.stream_write:
                val_file_path = os.path.join(
                    args.save_path, f'traj_{args.num_val}-steps_{args.num_steps}.h5'
                )
                with h5py.File(val_file_path, 'w') as val_file:
                    _write_h5_header(
                        val_file,
                        xml_path_for_run,
                        args.num_steps,
                        args.num_val,
                        args.dt,
                        args.data_dt,
                        policy_weights=policy_dict,
                    )
                    val_summary = generate_split(
                        xml_path_for_run, args.num_steps, args.dt, args.data_dt,
                        args.num_val, args.batch_size,
                        policy_names, policy_weights, ref_std_per_dim, args.num_workers,
                        args.lim_energy_ratio, args.max_energy_limit,
                        args.max_velocity, args.max_acceleration, args.max_abs_mom_dot,
                        args.safe_mode, args.joint_margin, args.ctrl_margin, args.max_saturation_frac,
                        args.torque_scale, args.lpf_uniform_beta, args.drop_target_states,
                        args.gym_reacher_target_sampling, args.gym_reacher_state_sampling,
                        args.strict_policy_ratios, stream_file=val_file,
                        seed_offset=10000000, desc="Collecting validation trajectories",
                    )
                print_running_stats(val_summary["running_stats"])
            else:
                val_summary = generate_split(
                    xml_path_for_run, args.num_steps, args.dt, args.data_dt,
                    args.num_val, args.batch_size,
                    policy_names, policy_weights, ref_std_per_dim, args.num_workers,
                    args.lim_energy_ratio, args.max_energy_limit,
                    args.max_velocity, args.max_acceleration, args.max_abs_mom_dot,
                    args.safe_mode, args.joint_margin, args.ctrl_margin, args.max_saturation_frac,
                    args.torque_scale, args.lpf_uniform_beta, args.drop_target_states,
                    args.gym_reacher_target_sampling, args.gym_reacher_state_sampling,
                    args.strict_policy_ratios, stream_file=None,
                    seed_offset=10000000, desc="Collecting validation trajectories",
                )
                print_running_stats(val_summary["running_stats"])
                save_as_h5py(
                    val_summary["results"],
                    args.save_path,
                    xml_path_for_run,
                    args.num_steps,
                    args.num_val,
                    args.dt,
                    args.data_dt,
                    policy_weights=policy_dict,
                )
            print(f"Validation set saved to {args.save_path}")

        print(f"\nDone!")
    finally:
        if tmp_xml_to_cleanup is not None and os.path.exists(tmp_xml_to_cleanup):
            os.remove(tmp_xml_to_cleanup)


if __name__ == '__main__':
    main()
