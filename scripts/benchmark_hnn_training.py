"""
Benchmark StructuredHNN training throughput on a single A100 80GB GPU.

Finds the optimal batch_size, num_workers, and torch.compile setting
by measuring actual training step time (forward + autograd.grad + backward + optimizer).

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/benchmark_hnn_training.py
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '4')

import sys
import time
import gc
import json
from pathlib import Path
from statistics import median

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from scripts.dataset import TrajectoryHNNCached
from src.models.HNN import StructuredHNN

# ── Config ──────────────────────────────────────────────────────────────────
TRAIN_FILE = str(project_root / 'data' / 'traj_80000-steps_4000.h5')

BATCH_SIZES = [8192, 16384, 32768, 65536, 131072, 262144, 524288]
NUM_WORKERS_LIST = [4, 8, 16]
NUM_WARMUP = 5
NUM_MEASURE = 50

DEVICE = torch.device('cuda:0')
COORDINATE_DIM = 3


# ── Core training step (replicates HNNWrapper.calculate_loss exactly) ───────
def train_step(model, batch, eps, p_std, q_std, qvel_var, mom_dot_var, optimizer):
    """Single training step matching HNNWrapper.calculate_loss (HNN.py L396-442)."""
    p = batch['mom'].to(DEVICE, non_blocking=True)
    q = batch['qpos'].to(DEVICE, non_blocking=True)
    dqdt_target = batch['qvel'].to(DEVICE, non_blocking=True)
    dpdt_target = batch['mom_dot'].to(DEVICE, non_blocking=True)
    torque = batch['torque'].to(DEVICE, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    p_raw = p.detach().requires_grad_(True)
    q_raw = q.detach().requires_grad_(True)

    p_scaled = p_raw / (p_std + eps)
    q_scaled = q_raw / (q_std + eps)

    H = model(p_scaled, q_scaled)

    grads = torch.autograd.grad(H.sum(), (p_raw, q_raw), create_graph=True)
    dqdt_pred = grads[0]
    dpdt_pred = -grads[1] + torque  # ground-truth torque (use_torque=True, predict_torque=False)

    loss_dqdt = torch.mean((dqdt_target - dqdt_pred) ** 2 / (qvel_var + eps))
    loss_dpdt = torch.mean((dpdt_target - dpdt_pred) ** 2 / (mom_dot_var + eps))
    loss = loss_dqdt + loss_dpdt

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


# ── Benchmark runner ────────────────────────────────────────────────────────
def benchmark(model, dataloader, eps, p_std, q_std, qvel_var, mom_dot_var,
              optimizer, num_warmup=NUM_WARMUP, num_measure=NUM_MEASURE):
    """Run warmup + timed iterations. Returns dict of stats."""
    model.train()
    data_iter = iter(dataloader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            return next(data_iter)

    # Warmup
    for _ in range(num_warmup):
        train_step(model, next_batch(), eps, p_std, q_std, qvel_var, mom_dot_var, optimizer)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Measure
    times = []
    for _ in range(num_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_step(model, next_batch(), eps, p_std, q_std, qvel_var, mom_dot_var, optimizer)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    bs = dataloader.batch_size
    mean_t = sum(times) / len(times)

    return {
        'batch_size': bs,
        'mean_time_ms': mean_t * 1000,
        'median_time_ms': median(times) * 1000,
        'std_time_ms': (sum((t - mean_t) ** 2 for t in times) / len(times)) ** 0.5 * 1000,
        'samples_per_sec': bs / mean_t,
        'peak_memory_gb': peak_mem,
    }


# ── Helpers ─────────────────────────────────────────────────────────────────
def make_model(compiled=False):
    model = StructuredHNN(COORDINATE_DIM, COORDINATE_DIM,
                          hidden_dim=256, num_layers=4).to(DEVICE)
    if compiled:
        model = torch.compile(model, mode='reduce-overhead')
    return model


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0, fused=True)


def make_dataloader(dataset, batch_size, num_workers=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=True, drop_last=True)


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def try_batch_size(dataset, batch_size, eps, p_std, q_std, qvel_var, mom_dot_var,
                   compiled=False, num_workers=8):
    """Try one training step at given batch_size. Returns True if no OOM."""
    cleanup()
    try:
        model = make_model(compiled)
        optimizer = make_optimizer(model)
        dl = make_dataloader(dataset, batch_size, num_workers)
        batch = next(iter(dl))
        train_step(model, batch, eps, p_std, q_std, qvel_var, mom_dot_var, optimizer)
        torch.cuda.synchronize()
        del model, optimizer, dl, batch
        cleanup()
        return True
    except torch.cuda.OutOfMemoryError:
        cleanup()
        return False


def find_max_batch_size(dataset, eps, p_std, q_std, qvel_var, mom_dot_var,
                        compiled=False, num_workers=8):
    """Exponential probe + binary search for max batch size."""
    # Exponential probe
    best = 8192
    bs = 8192
    while bs <= 8 * 1024 * 1024:
        print(f'  Probing bs={bs:>10,} ...', end=' ', flush=True)
        if try_batch_size(dataset, bs, eps, p_std, q_std, qvel_var, mom_dot_var,
                          compiled, num_workers):
            print('OK')
            best = bs
            bs *= 2
        else:
            print('OOM')
            break

    # Binary search refinement
    lo, hi = best, bs
    while hi - lo > 4096:
        mid = ((lo + hi) // 2 // 1024) * 1024  # align to 1024
        if mid == lo:
            break
        print(f'  Refining bs={mid:>10,} ...', end=' ', flush=True)
        if try_batch_size(dataset, mid, eps, p_std, q_std, qvel_var, mom_dot_var,
                          compiled, num_workers):
            print('OK')
            lo = mid
        else:
            print('OOM')
            hi = mid

    return lo


def run_config(dataset, batch_size, eps, p_std, q_std, qvel_var, mom_dot_var,
               compiled=False, num_workers=8):
    """Benchmark a single config. Returns result dict or None on OOM."""
    cleanup()
    try:
        model = make_model(compiled)
        optimizer = make_optimizer(model)
        dl = make_dataloader(dataset, batch_size, num_workers)
        result = benchmark(model, dl, eps, p_std, q_std, qvel_var, mom_dot_var, optimizer)
        result['compiled'] = compiled
        result['num_workers'] = num_workers
        del model, optimizer, dl
        cleanup()
        return result
    except torch.cuda.OutOfMemoryError:
        cleanup()
        return None


def print_header():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f'\n{"=" * 80}')
    print(f'StructuredHNN Training Benchmark')
    print(f'{"=" * 80}')
    print(f'GPU:       {gpu_name} ({gpu_mem:.0f} GB)')
    print(f'PyTorch:   {torch.__version__}  |  CUDA: {torch.version.cuda}')
    print(f'Model:     StructuredHNN (hidden=256, layers=4)')
    print(f'Precision: float32 (TF32 matmul enabled)')
    print(f'Measure:   {NUM_WARMUP} warmup + {NUM_MEASURE} timed iterations per config')


def print_table(results, title):
    print(f'\n{"=" * 80}')
    print(title)
    print(f'{"=" * 80}')
    print(f'  {"Batch Size":>12} | {"Throughput":>14} | {"Time/Batch":>12} | '
          f'{"Peak Mem":>10} | {"Samples/Epoch":>15}')
    print(f'  {"-" * 12}-+-{"-" * 14}-+-{"-" * 12}-+-{"-" * 10}-+-{"-" * 15}')
    for r in results:
        samples_per_epoch = r['batch_size'] * 2000
        print(f'  {r["batch_size"]:>12,} | {r["samples_per_sec"]:>11,.0f} s/s | '
              f'{r["mean_time_ms"]:>9.1f} ms | '
              f'{r["peak_memory_gb"]:>7.1f} GB | {samples_per_epoch:>13,}')


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print_header()

    # Load dataset
    print(f'\nLoading dataset from {TRAIN_FILE} ...')
    dataset = TrajectoryHNNCached(TRAIN_FILE, trajectory_length=1000)
    print(f'Loaded {len(dataset):,} samples')

    # Compute statistics (same as training)
    eps = torch.tensor(1e-8, device=DEVICE)
    qvel_var = dataset.qvel.var(dim=0, unbiased=False).to(DEVICE)
    mom_dot_var = dataset.mom_dot.var(dim=0, unbiased=False).to(DEVICE)
    q_std = dataset.qpos.std(dim=0, unbiased=False).to(DEVICE)
    p_std = dataset.mom.std(dim=0, unbiased=False).to(DEVICE)

    all_results = []

    # ── Phase 1: Find max batch size ────────────────────────────────────────
    print(f'\n{"=" * 80}')
    print('Phase 1: Max Batch Size Search')
    print(f'{"=" * 80}')

    print('\nEager mode:')
    max_bs_eager = find_max_batch_size(dataset, eps, p_std, q_std, qvel_var, mom_dot_var,
                                       compiled=False)
    print(f'  → Max batch size (eager): {max_bs_eager:,}')

    print('\nCompiled mode:')
    max_bs_compiled = find_max_batch_size(dataset, eps, p_std, q_std, qvel_var, mom_dot_var,
                                          compiled=True)
    print(f'  → Max batch size (compiled): {max_bs_compiled:,}')

    # ── Phase 2: Batch size sweep (eager, workers=8) ────────────────────────
    print(f'\n{"=" * 80}')
    print('Phase 2: Batch Size Sweep (eager, workers=8)')
    print(f'{"=" * 80}')

    eager_results = []
    for bs in BATCH_SIZES:
        if bs > max_bs_eager:
            print(f'  bs={bs:>10,}: skip (> max {max_bs_eager:,})')
            break
        print(f'  bs={bs:>10,}: benchmarking ...', end=' ', flush=True)
        r = run_config(dataset, bs, eps, p_std, q_std, qvel_var, mom_dot_var,
                       compiled=False, num_workers=8)
        if r is None:
            print('OOM')
            break
        print(f'{r["samples_per_sec"]:,.0f} s/s | {r["mean_time_ms"]:.1f} ms | {r["peak_memory_gb"]:.1f} GB')
        eager_results.append(r)
        all_results.append(r)

    print_table(eager_results, 'Batch Size Sweep — Eager (workers=8)')

    # ── Phase 3: torch.compile comparison ───────────────────────────────────
    print(f'\n{"=" * 80}')
    print('Phase 3: torch.compile Effect')
    print(f'{"=" * 80}')

    # Test compile at a few key batch sizes
    compile_test_sizes = [bs for bs in [8192, 65536, 262144] if bs <= max_bs_compiled]
    compile_results = []
    for bs in compile_test_sizes:
        print(f'  bs={bs:>10,}: benchmarking compiled ...', end=' ', flush=True)
        r = run_config(dataset, bs, eps, p_std, q_std, qvel_var, mom_dot_var,
                       compiled=True, num_workers=8)
        if r is None:
            print('OOM')
            continue
        # Find corresponding eager result
        eager_r = next((er for er in eager_results if er['batch_size'] == bs), None)
        speedup = eager_r['samples_per_sec'] / r['samples_per_sec'] if eager_r else float('nan')
        speedup = r['samples_per_sec'] / eager_r['samples_per_sec'] if eager_r else float('nan')
        print(f'{r["samples_per_sec"]:,.0f} s/s | speedup={speedup:.2f}x')
        compile_results.append(r)
        all_results.append(r)

    if compile_results and eager_results:
        print(f'\n  {"Batch Size":>12} | {"Eager":>14} | {"Compiled":>14} | {"Speedup":>8}')
        print(f'  {"-" * 12}-+-{"-" * 14}-+-{"-" * 14}-+-{"-" * 8}')
        for cr in compile_results:
            er = next((e for e in eager_results if e['batch_size'] == cr['batch_size']), None)
            if er:
                sp = cr['samples_per_sec'] / er['samples_per_sec']
                print(f'  {cr["batch_size"]:>12,} | {er["samples_per_sec"]:>11,.0f} s/s | '
                      f'{cr["samples_per_sec"]:>11,.0f} s/s | {sp:>6.2f}x')

    # ── Phase 4: num_workers sweep ──────────────────────────────────────────
    # Use the best batch size from eager results
    if eager_results:
        best_eager = max(eager_results, key=lambda r: r['samples_per_sec'])
        best_bs = best_eager['batch_size']

        print(f'\n{"=" * 80}')
        print(f'Phase 4: num_workers Sweep (bs={best_bs:,}, eager)')
        print(f'{"=" * 80}')

        worker_results = []
        for nw in NUM_WORKERS_LIST:
            print(f'  workers={nw:>2}: benchmarking ...', end=' ', flush=True)
            r = run_config(dataset, best_bs, eps, p_std, q_std, qvel_var, mom_dot_var,
                           compiled=False, num_workers=nw)
            if r is None:
                print('OOM')
                continue
            print(f'{r["samples_per_sec"]:,.0f} s/s | {r["mean_time_ms"]:.1f} ms')
            worker_results.append(r)
            all_results.append(r)

        if worker_results:
            print(f'\n  {"Workers":>8} | {"Throughput":>14} | {"Time/Batch":>12}')
            print(f'  {"-" * 8}-+-{"-" * 14}-+-{"-" * 12}')
            for wr in worker_results:
                print(f'  {wr["num_workers"]:>8} | {wr["samples_per_sec"]:>11,.0f} s/s | '
                      f'{wr["mean_time_ms"]:>9.1f} ms')

    # ── Summary ─────────────────────────────────────────────────────────────
    if all_results:
        best = max(all_results, key=lambda r: r['samples_per_sec'])
        baseline = next((r for r in all_results
                         if r['batch_size'] == 8192 and not r.get('compiled', False)
                         and r.get('num_workers', 8) == 8), None)

        print(f'\n{"=" * 80}')
        print('OPTIMAL CONFIGURATION')
        print(f'{"=" * 80}')
        print(f'  batch_size:   {best["batch_size"]:,}')
        print(f'  num_workers:  {best.get("num_workers", 8)}')
        print(f'  compile:      {best.get("compiled", False)}')
        print(f'  throughput:   {best["samples_per_sec"]:,.0f} samples/sec')
        print(f'  time/batch:   {best["mean_time_ms"]:.1f} ms')
        print(f'  peak memory:  {best["peak_memory_gb"]:.1f} / 80.0 GB')

        if baseline:
            # Training time estimate: 1000 epochs × 2000 batches
            t_baseline = baseline['mean_time_ms'] * 2000 * 1000 / 1000 / 3600
            t_optimal = best['mean_time_ms'] * 2000 * 1000 / 1000 / 3600
            speedup = t_baseline / t_optimal
            print(f'\n  Training time (1000 epochs × 2000 batches):')
            print(f'    Current (bs={baseline["batch_size"]:,}): {t_baseline:.1f} hours')
            print(f'    Optimal (bs={best["batch_size"]:,}): {t_optimal:.1f} hours')
            print(f'    Speedup: {speedup:.2f}x')

        # LR scaling recommendations
        base_lr = 3e-4
        base_bs = 8192
        print(f'\n{"=" * 80}')
        print('Learning Rate Scaling (sqrt rule for AdamW)')
        print(f'{"=" * 80}')
        print(f'  {"Batch Size":>12} | {"LR (sqrt)":>12} | {"LR (linear)":>12}')
        print(f'  {"-" * 12}-+-{"-" * 12}-+-{"-" * 12}')
        for r in eager_results:
            bs = r['batch_size']
            lr_sqrt = base_lr * (bs / base_bs) ** 0.5
            lr_linear = base_lr * (bs / base_bs)
            print(f'  {bs:>12,} | {lr_sqrt:>12.2e} | {lr_linear:>12.2e}')

    # Save JSON
    out_path = project_root / 'logs' / 'hnn_benchmark_results.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {out_path}')
    print(f'{"=" * 80}\n')


if __name__ == '__main__':
    main()
