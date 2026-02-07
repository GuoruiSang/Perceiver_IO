"""
Benchmark speed comparison: MuJoCo vs Unguided DPF vs Guided DPF (3DoF).

Finds the maximum batch size that fits GPU memory for each DPF mode,
then measures wall-clock time for generating trajectories of length 1000.
"""
import sys, os, io, time, math, warnings
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TQDM_DISABLE'] = '1'  # Disable tqdm globally
warnings.filterwarnings('ignore')

import numpy as np
import torch
import mujoco
from concurrent.futures import ProcessPoolExecutor

# ── Config ──
TRAJ_LENGTH = 1000
DT = 0.0001
DATA_DT = 0.0002
QPOS_DIM = 3

DPF_CKPT = str(project_root / 'checkpoints' /
    'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding'
    '&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions'
    ':epoch=2999_val_loss:val_loss=0.0010.ckpt')
HNN_CKPT = str(project_root / 'checkpoints' /
    'SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt')
XML_PATH = str(project_root / 'configs' / 'rigid_arm_hinge.xml')

# Guidance params (3DoF)
GUIDANCE_STEPS = 25
GUIDANCE_LR = 0.01
GUIDANCE_AFTER_STEPS = 45
CONTEXT_FRACTION = 0.2
NUM_DIFFUSION_STEPS = 50

# Number of warm-up and timing runs
WARMUP_RUNS = 2
TIMING_RUNS = 5

# Batch sizes to probe (for fast OOM search)
PROBE_DIFFUSION_STEPS = 3  # Minimal steps for memory probing


# ── Suppress ALL output (stdout + stderr) ──
class SuppressAllOutput:
    """Suppress both stdout and stderr to eliminate ALL print/tqdm output."""
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self
    def __exit__(self, *a):
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def load_models(device):
    """Load DPF + HNN, apply EMA permanently, return ready-to-use models."""
    from src.models.trajectory_dpf import TrajectoryDPF
    from src.models.HNN import HNNWrapper

    with SuppressAllOutput():
        dpf = TrajectoryDPF.load_from_checkpoint(DPF_CKPT, map_location=device, strict=False)
        ckpt = torch.load(DPF_CKPT, map_location=device, weights_only=False)
        if 'ema_state_dict' in ckpt:
            dpf.ema.load_state_dict(ckpt['ema_state_dict'])
        dpf.eval()

        # Apply EMA permanently
        dpf_device = next(dpf.model.parameters()).device
        for name, param in dpf.model.named_parameters():
            if param.requires_grad and name in dpf.ema.shadow:
                dpf.ema.shadow[name] = dpf.ema.shadow[name].to(device=dpf_device, dtype=param.dtype)
        dpf.ema.store(dpf.model)
        dpf.ema.copy_to(dpf.model)
        dpf.model.eval()

        hnn = HNNWrapper.load_from_checkpoint(HNN_CKPT, map_location=device)
        hnn.eval()

    del ckpt
    torch.cuda.empty_cache()
    return dpf, hnn


def generate_torque_batch(n, length, device):
    """Generate random sinusoidal torque matching training distribution."""
    lim_amp, lim_freq, lim_phase = 0.5, 6 * math.pi, 2 * math.pi
    num_sin, torque_dim = 5, QPOS_DIM
    all_t = []
    for _ in range(n):
        amp = np.random.uniform(0, lim_amp, (torque_dim, num_sin, 1))
        freq = np.random.uniform(0, lim_freq, (torque_dim, num_sin, 1))
        phase = np.random.uniform(0, lim_phase, (torque_dim, num_sin, 1))
        steps = np.arange(length)[None, None, :] * DATA_DT
        t = (amp * np.sin(freq * steps + phase)).sum(axis=1).T  # [L, dim]
        all_t.append(t)
    return torch.tensor(np.stack(all_t), dtype=torch.float32, device=device)


# ── MuJoCo benchmark (CPU, multiprocessing) ──
def _mujoco_single(args):
    """Simulate one trajectory in a worker process."""
    torque_np, xml_path, traj_length = args
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    qpos_dim = 3
    substeps = int(DATA_DT / DT)

    data.qpos[:qpos_dim] = np.random.uniform(-0.5, 0.5, qpos_dim)
    data.qvel[:qpos_dim] = np.random.uniform(-1, 1, qpos_dim)
    mujoco.mj_forward(model, data)

    for t in range(traj_length - 1):
        data.ctrl[:qpos_dim] = torque_np[t]
        for _ in range(substeps):
            mujoco.mj_step(model, data)
    return 0


def benchmark_mujoco(num_traj, num_workers=None):
    """Benchmark MuJoCo simulation with multiprocessing."""
    if num_workers is None:
        num_workers = min(os.cpu_count(), num_traj)
    np.random.seed(0)
    torques = [np.random.randn(TRAJ_LENGTH, QPOS_DIM).astype(np.float32) * 0.5
               for _ in range(num_traj)]
    args = [(t, XML_PATH, TRAJ_LENGTH) for t in torques]

    # Warm up (spawn workers)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        list(pool.map(_mujoco_single, args[:min(num_workers, len(args))]))

    # Timing
    times = []
    for _ in range(TIMING_RUNS):
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            t0 = time.perf_counter()
            list(pool.map(_mujoco_single, args))
            times.append(time.perf_counter() - t0)
    return times


# ── DPF helpers ──
def _sample_dpf(dpf, hnn, device, bs, torque, guided):
    """Run one DPF sampling pass. use_ema=False since EMA applied permanently."""
    with SuppressAllOutput():
        if guided:
            dpf.sample_trajectories(
                num_samples=bs, trajectory_length=TRAJ_LENGTH,
                num_diffusion_steps=NUM_DIFFUSION_STEPS,
                context_fraction=CONTEXT_FRACTION, use_ema=False, sampler='ddim',
                hnn=hnn, guidance_method='adam',
                guidance_steps=GUIDANCE_STEPS, guidance_lr=GUIDANCE_LR,
                guidance_after_steps=GUIDANCE_AFTER_STEPS,
                torque=torque)
        else:
            dpf.sample_trajectories(
                num_samples=bs, trajectory_length=TRAJ_LENGTH,
                num_diffusion_steps=NUM_DIFFUSION_STEPS,
                context_fraction=CONTEXT_FRACTION, use_ema=False, sampler='ddim',
                hnn=None, guidance_steps=0,
                torque=torque)


def _probe_batch(dpf, hnn, device, bs, guided):
    """Fast OOM probe: run with minimal diffusion steps to test memory.

    For guided mode, set guidance_after_steps=0 so guidance runs immediately,
    testing peak memory from Adam optimizer states.
    """
    torch.cuda.empty_cache()
    try:
        torque = generate_torque_batch(bs, TRAJ_LENGTH, device)
        with SuppressAllOutput():
            if guided:
                dpf.sample_trajectories(
                    num_samples=bs, trajectory_length=TRAJ_LENGTH,
                    num_diffusion_steps=PROBE_DIFFUSION_STEPS,
                    context_fraction=CONTEXT_FRACTION, use_ema=False, sampler='ddim',
                    hnn=hnn, guidance_method='adam',
                    guidance_steps=GUIDANCE_STEPS, guidance_lr=GUIDANCE_LR,
                    guidance_after_steps=0,  # Run guidance immediately for memory test
                    torque=torque)
            else:
                dpf.sample_trajectories(
                    num_samples=bs, trajectory_length=TRAJ_LENGTH,
                    num_diffusion_steps=PROBE_DIFFUSION_STEPS,
                    context_fraction=CONTEXT_FRACTION, use_ema=False, sampler='ddim',
                    hnn=None, guidance_steps=0,
                    torque=torque)
        del torque
        torch.cuda.empty_cache()
        return True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False


def find_max_batch(dpf, hnn, device, guided):
    """Binary search for max batch size that fits in GPU memory."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    lo, hi, best = 1, 2048, 1

    # Exponential probe (fast: only PROBE_DIFFUSION_STEPS per test)
    bs = 1
    while bs <= hi:
        if _probe_batch(dpf, hnn, device, bs, guided):
            best = bs
            bs *= 2
        else:
            break
    lo, hi = best, bs

    # Binary search
    while lo <= hi:
        mid = (lo + hi) // 2
        if _probe_batch(dpf, hnn, device, mid, guided):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def benchmark_dpf(dpf, hnn, device, bs, guided):
    """Benchmark DPF sampling at given batch size."""
    torch.cuda.empty_cache()
    torque = generate_torque_batch(bs, TRAJ_LENGTH, device)

    # Warm up
    for _ in range(WARMUP_RUNS):
        _sample_dpf(dpf, hnn, device, bs, torque, guided)
    torch.cuda.synchronize()

    # Timing
    times = []
    for _ in range(TIMING_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _sample_dpf(dpf, hnn, device, bs, torque, guided)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    del torque
    torch.cuda.empty_cache()
    return times


# ── Main ──
if __name__ == '__main__':
    device = torch.device('cuda:0')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Trajectory Length: {TRAJ_LENGTH}")
    print(f"Diffusion Steps: {NUM_DIFFUSION_STEPS}")
    print(f"Timing runs: {TIMING_RUNS}\n")

    # Load models (EMA applied permanently)
    print("Loading models...")
    dpf, hnn = load_models(device)
    print("Models loaded.\n")

    # ── 1. Find max batch sizes (fast probing) ──
    print("=" * 60)
    print("Finding max batch sizes (GPU memory limit)...")
    print("=" * 60)

    max_bs_ung = find_max_batch(dpf, hnn, device, guided=False)
    print(f"  Unguided max batch: {max_bs_ung}")

    max_bs_gui = find_max_batch(dpf, hnn, device, guided=True)
    print(f"  Guided   max batch: {max_bs_gui}")
    print()

    common_bs = min(max_bs_ung, max_bs_gui)

    # ── 2. Benchmark MuJoCo ──
    print("=" * 60)
    print(f"Benchmarking MuJoCo ({common_bs} trajectories, {os.cpu_count()} workers)...")
    print("=" * 60)
    mj_times = benchmark_mujoco(common_bs)
    mj_mean = np.mean(mj_times)
    mj_std = np.std(mj_times)
    print(f"  Time: {mj_mean:.3f} +/- {mj_std:.3f} s")
    print(f"  Throughput: {common_bs / mj_mean:.1f} traj/s\n")

    # ── 3. Benchmark Unguided ──
    print("=" * 60)
    print(f"Benchmarking Unguided DPF (batch={max_bs_ung})...")
    print("=" * 60)
    ung_times = benchmark_dpf(dpf, hnn, device, max_bs_ung, guided=False)
    ung_mean = np.mean(ung_times)
    ung_std = np.std(ung_times)
    print(f"  Time: {ung_mean:.3f} +/- {ung_std:.3f} s")
    print(f"  Throughput: {max_bs_ung / ung_mean:.1f} traj/s\n")

    # ── 4. Benchmark Guided ──
    print("=" * 60)
    print(f"Benchmarking Guided DPF (batch={max_bs_gui})...")
    print("=" * 60)
    gui_times = benchmark_dpf(dpf, hnn, device, max_bs_gui, guided=True)
    gui_mean = np.mean(gui_times)
    gui_std = np.std(gui_times)
    print(f"  Time: {gui_mean:.3f} +/- {gui_std:.3f} s")
    print(f"  Throughput: {max_bs_gui / gui_mean:.1f} traj/s\n")

    # ── 5. Fair comparison (same batch size) ──
    if common_bs != max_bs_ung or common_bs != max_bs_gui:
        print("=" * 60)
        print(f"Fair comparison (same batch={common_bs}):")
        print("=" * 60)
        ung_fair = benchmark_dpf(dpf, hnn, device, common_bs, guided=False)
        gui_fair = benchmark_dpf(dpf, hnn, device, common_bs, guided=True)
        mj_fair_mean = mj_mean
        print(f"  MuJoCo:   {mj_fair_mean:.3f} s  ({common_bs / mj_fair_mean:.1f} traj/s)")
        print(f"  Unguided: {np.mean(ung_fair):.3f} +/- {np.std(ung_fair):.3f} s  ({common_bs / np.mean(ung_fair):.1f} traj/s)")
        print(f"  Guided:   {np.mean(gui_fair):.3f} +/- {np.std(gui_fair):.3f} s  ({common_bs / np.mean(gui_fair):.1f} traj/s)")
        print()

    # ── Summary ──
    print("=" * 60)
    print("SUMMARY (max throughput per method)")
    print("=" * 60)
    print(f"{'Method':<20} {'Batch':>6} {'Time (s)':>12} {'Traj/s':>10}")
    print("-" * 60)
    print(f"{'MuJoCo':<20} {common_bs:>6} {mj_mean:>9.3f}+-{mj_std:.3f} {common_bs/mj_mean:>10.1f}")
    print(f"{'Unguided DPF':<20} {max_bs_ung:>6} {ung_mean:>9.3f}+-{ung_std:.3f} {max_bs_ung/ung_mean:>10.1f}")
    print(f"{'Guided DPF':<20} {max_bs_gui:>6} {gui_mean:>9.3f}+-{gui_std:.3f} {max_bs_gui/gui_mean:>10.1f}")
    print("=" * 60)
