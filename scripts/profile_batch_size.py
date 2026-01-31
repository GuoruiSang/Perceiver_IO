"""
Profile batch sizes for ablation trajectory generation.

Benchmarks different batch sizes for both unguided and guided generation
to find optimal throughput on the current GPU.

Usage:
    python scripts/profile_batch_size.py --device cuda:0
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
import torch
import numpy as np

from src.models.trajectory_dpf import TrajectoryDPF
from src.models.utils import EMA


def load_model(checkpoint_path, device):
    print(f"Loading model from: {checkpoint_path}")
    model = TrajectoryDPF.load_from_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    ema_shadow = checkpoint.get('ema_shadow', None)
    ema_decay = checkpoint.get('ema_decay', 0.9995)
    if ema_shadow is not None:
        model.ema = EMA(model.model, decay=ema_decay)
        for name, tensor in ema_shadow.items():
            if name in model.ema.shadow:
                model.ema.shadow[name] = tensor.to(device=device, dtype=model.ema.shadow[name].dtype)
        model._ema_loaded = True
        print("  EMA shadow loaded")
    return model


def load_hnn(hnn_checkpoint_path, device):
    from src.models.HNN import HNNWrapper
    hnn = HNNWrapper.load_from_checkpoint(hnn_checkpoint_path, map_location=device)
    hnn = hnn.to(device)
    hnn.eval()
    return hnn


def profile_batch_size(model, hnn, device, batch_size, trajectory_length, num_samples,
                       num_diffusion_steps, context_fraction):
    """Profile a single batch_size for both unguided and guided."""
    torque = torch.randn(num_samples, trajectory_length, model.torque_dim, device=device) * 0.5

    results = {}

    for mode_name, guidance_kwargs in [
        ('unguided', dict(hnn=None, guidance_steps=0)),
        ('guided', dict(hnn=hnn, guidance_method='adam', guidance_steps=25,
                        guidance_lr=0.01, guidance_after_steps=45, lambda_init=0.0)),
    ]:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        try:
            start = time.time()
            all_states = []

            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch_torques = torque[batch_start:batch_end]

                state, _ = model.sample_trajectories(
                    num_samples=batch_torques.shape[0],
                    trajectory_length=trajectory_length,
                    num_diffusion_steps=num_diffusion_steps,
                    context_fraction=context_fraction,
                    use_ema=False,
                    sampler='ddim',
                    guidance_scale=1.0,
                    torque=batch_torques,
                    **guidance_kwargs,
                )
                all_states.append(state)

            elapsed = time.time() - start
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3

            results[mode_name] = {
                'throughput': num_samples / elapsed,
                'time': elapsed,
                'peak_mem_gb': peak_mem,
                'status': 'OK',
            }

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                results[mode_name] = {
                    'throughput': 0,
                    'time': float('inf'),
                    'peak_mem_gb': float('inf'),
                    'status': 'OOM',
                }
            else:
                raise

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile batch sizes")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/gsang/Projects/Perceiver_IO/checkpoints/"
                        "trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding"
                        "&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions"
                        ":epoch=2999_val_loss:val_loss=0.0010.ckpt")
    parser.add_argument("--hnn_checkpoint", type=str,
                        default="/home/gsang/Projects/Perceiver_IO/checkpoints/"
                        "SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt")
    parser.add_argument("--trajectory_length", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Total samples to generate per test (keep small for speed)")
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--context_fraction", type=float, default=0.2)
    parser.add_argument("--batch_sizes", type=int, nargs="+",
                        default=[25, 50, 100, 200, 400, 500])
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    model = load_model(args.checkpoint, device)
    hnn = load_hnn(args.hnn_checkpoint, device)

    print(f"\n{'='*90}")
    print(f"  BATCH SIZE PROFILING")
    print(f"  trajectory_length={args.trajectory_length}, num_samples={args.num_samples}, "
          f"diffusion_steps={args.num_diffusion_steps}")
    print(f"{'='*90}")

    # Warmup
    print("\nWarmup run...")
    _ = profile_batch_size(model, hnn, device, 25, args.trajectory_length,
                           50, args.num_diffusion_steps, args.context_fraction)

    all_results = {}
    for bs in args.batch_sizes:
        print(f"\nProfiling batch_size={bs}...")
        results = profile_batch_size(
            model, hnn, device, bs, args.trajectory_length,
            args.num_samples, args.num_diffusion_steps, args.context_fraction,
        )
        all_results[bs] = results

        for mode, r in results.items():
            if r['status'] == 'OK':
                print(f"  {mode:>10}: {r['throughput']:.1f} samples/s, "
                      f"{r['time']:.1f}s total, {r['peak_mem_gb']:.1f} GB VRAM")
            else:
                print(f"  {mode:>10}: OOM")

    # Summary table
    print(f"\n{'='*90}")
    print(f"  SUMMARY")
    print(f"{'='*90}")
    print(f"{'Batch Size':>12} | {'Unguided samp/s':>16} | {'Unguided VRAM':>14} | "
          f"{'Guided samp/s':>14} | {'Guided VRAM':>12} |")
    print("-" * 90)

    best_unguided_bs, best_unguided_tp = 0, 0
    best_guided_bs, best_guided_tp = 0, 0

    for bs in args.batch_sizes:
        r = all_results[bs]
        ug = r['unguided']
        gd = r['guided']

        ug_str = f"{ug['throughput']:.1f}" if ug['status'] == 'OK' else 'OOM'
        ug_mem = f"{ug['peak_mem_gb']:.1f} GB" if ug['status'] == 'OK' else 'OOM'
        gd_str = f"{gd['throughput']:.1f}" if gd['status'] == 'OK' else 'OOM'
        gd_mem = f"{gd['peak_mem_gb']:.1f} GB" if gd['status'] == 'OK' else 'OOM'

        print(f"{bs:>12} | {ug_str:>16} | {ug_mem:>14} | {gd_str:>14} | {gd_mem:>12} |")

        if ug['status'] == 'OK' and ug['throughput'] > best_unguided_tp:
            best_unguided_tp = ug['throughput']
            best_unguided_bs = bs
        if gd['status'] == 'OK' and gd['throughput'] > best_guided_tp:
            best_guided_tp = gd['throughput']
            best_guided_bs = bs

    print(f"\n  Best unguided batch_size: {best_unguided_bs} ({best_unguided_tp:.1f} samples/s)")
    print(f"  Best guided batch_size:   {best_guided_bs} ({best_guided_tp:.1f} samples/s)")
    print(f"\nUse these values in generate_ablation_trajectories.py:")
    print(f"  --batch_size_unguided {best_unguided_bs} --batch_size_guided {best_guided_bs}")


if __name__ == "__main__":
    main()
