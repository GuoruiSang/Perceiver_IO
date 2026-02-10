"""
Ablation Trajectory Generation Pipeline - Single Model Runner

Generates trajectories for all experimental conditions and saves them to H5.
Metric computation is handled by downstream evaluation/report scripts.

Experiments:
  A) Variable trajectory lengths: 4 torque conditions x 30 lengths x 2 modes (unguided/guided)
  B) Variable context fractions: 1 torque condition x 20 fractions x 2 modes

Usage:
    python scripts/generate_ablation_trajectories.py \
        --model_name original \
        --checkpoint checkpoints/model.ckpt \
        --device cuda:0 \
        --output_dir output_ablation

    # Split torque_concat across 2 GPUs:
    python scripts/generate_ablation_trajectories.py \
        --model_name torque_concat --device cuda:2 \
        --torque_conditions training,sinusoidal --run_exp_b \
        ...
    python scripts/generate_ablation_trajectories.py \
        --model_name torque_concat --device cuda:3 \
        --torque_conditions gp,zero --no_exp_b \
        ...
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import h5py
import torch
import numpy as np
import time
import os

from src.models.trajectory_dpf import TrajectoryDPF
from src.models.utils import EMA


def load_model_and_hnn(checkpoint_path, hnn_checkpoint_path, device):
    """Load model with EMA + HNN."""
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

    hnn = None
    if hnn_checkpoint_path:
        from src.models.HNN import HNNWrapper
        print(f"Loading HNN from: {hnn_checkpoint_path}")
        hnn = HNNWrapper.load_from_checkpoint(hnn_checkpoint_path, map_location=device)
        hnn = hnn.to(device)
        hnn.eval()

    return model, hnn


def load_torques(torque_file, device):
    """Load torque sequences from H5 file."""
    print(f"Loading torques from: {torque_file}")
    with h5py.File(torque_file, 'r') as f:
        torques = torch.tensor(f['torques'][:], dtype=torch.float32, device=device)
        print(f"  Shape: {torques.shape}, range: [{torques.min():.3f}, {torques.max():.3f}]")
    return torques


def generate_trajectories(
    model, hnn, torques, device,
    trajectory_length, num_samples, batch_size,
    num_diffusion_steps, context_fraction,
    guided, seed=228,
):
    """
    Generate trajectories and return state + torque tensors.

    Returns:
        state: [num_samples, trajectory_length, state_dim] on CPU
        torque_out: [num_samples, trajectory_length, torque_dim] on CPU
    """
    traj_torques = torques[:num_samples, :trajectory_length, :]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    all_states = []
    all_torques = []

    guidance_kwargs = {}
    if guided and hnn is not None:
        guidance_kwargs = dict(
            hnn=hnn,
            guidance_method='adam',
            guidance_steps=25,
            guidance_lr=0.01,
            guidance_after_steps=45,
            lambda_init=0.0,
        )
    else:
        guidance_kwargs = dict(hnn=None, guidance_steps=0)

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_torques = traj_torques[batch_start:batch_end]

        try:
            state, torque_out = model.sample_trajectories(
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
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                # Retry with half batch
                half = batch_torques.shape[0] // 2
                if half == 0:
                    raise
                for sub_start in range(0, batch_torques.shape[0], half):
                    sub_end = min(sub_start + half, batch_torques.shape[0])
                    sub_torques = batch_torques[sub_start:sub_end]
                    state, torque_out = model.sample_trajectories(
                        num_samples=sub_torques.shape[0],
                        trajectory_length=trajectory_length,
                        num_diffusion_steps=num_diffusion_steps,
                        context_fraction=context_fraction,
                        use_ema=False,
                        sampler='ddim',
                        guidance_scale=1.0,
                        torque=sub_torques,
                        **guidance_kwargs,
                    )
                    all_states.append(state.cpu())
                    all_torques.append(torque_out.cpu())
                continue
            else:
                raise

        all_states.append(state.cpu())
        all_torques.append(torque_out.cpu())

    return torch.cat(all_states, dim=0), torch.cat(all_torques, dim=0)


def save_to_h5(h5_file, group_name, state, torque):
    """Save state and torque tensors to an H5 group."""
    g = h5_file.create_group(group_name)
    g.create_dataset('state', data=state.numpy(), dtype='f4')
    g.create_dataset('torque', data=torque.numpy(), dtype='f4')


def run_experiment_a(
    model, hnn, device, output_dir, model_name,
    torque_dict, trajectory_lengths, num_samples,
    batch_size_unguided, batch_size_guided,
    num_diffusion_steps, context_fraction, seed,
):
    """Run Experiment A: variable trajectory lengths for each torque condition."""
    total_evals = len(torque_dict) * len(trajectory_lengths) * 2
    eval_count = 0
    total_start = time.time()

    for torque_label, torques in torque_dict.items():
        h5_path = os.path.join(output_dir, model_name, f'exp_a_{torque_label}.h5')
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)

        # Check for existing file to enable resuming
        existing_groups = set()
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                existing_groups = set(f.keys())

        with h5py.File(h5_path, 'a') as h5f:
            for traj_length in trajectory_lengths:
                for mode_name, guided in [('unguided', False), ('guided', True)]:
                    group_name = f'{mode_name}/L{traj_length}'
                    eval_count += 1

                    if group_name in existing_groups:
                        elapsed_total = time.time() - total_start
                        print(f"  [{eval_count}/{total_evals}] SKIP {torque_label}/{group_name} (exists) "
                              f"[{elapsed_total:.0f}s elapsed]")
                        continue

                    batch_size = batch_size_guided if guided else batch_size_unguided
                    start = time.time()

                    state, torque_out = generate_trajectories(
                        model, hnn, torques, device,
                        traj_length, num_samples, batch_size,
                        num_diffusion_steps, context_fraction,
                        guided=guided, seed=seed,
                    )

                    save_to_h5(h5f, group_name, state, torque_out)
                    h5f.flush()

                    elapsed = time.time() - start
                    elapsed_total = time.time() - total_start
                    throughput = num_samples / elapsed
                    print(f"  [{eval_count}/{total_evals}] {torque_label}/{group_name}: "
                          f"{elapsed:.1f}s ({throughput:.0f} samp/s) "
                          f"[{elapsed_total:.0f}s elapsed]")

        print(f"  Saved: {h5_path}")


def run_experiment_b(
    model, hnn, device, output_dir, model_name,
    torques, trajectory_length, context_fractions, num_samples,
    batch_size_unguided, batch_size_guided,
    num_diffusion_steps, seed,
):
    """Run Experiment B: variable context fractions."""
    total_evals = len(context_fractions) * 2
    eval_count = 0
    total_start = time.time()

    h5_path = os.path.join(output_dir, model_name, 'exp_b_context_fractions.h5')
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    existing_groups = set()
    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            existing_groups = set(f.keys())

    with h5py.File(h5_path, 'a') as h5f:
        for ctx_frac in context_fractions:
            for mode_name, guided in [('unguided', False), ('guided', True)]:
                group_name = f'{mode_name}/cf_{ctx_frac:.2f}'
                eval_count += 1

                if group_name in existing_groups:
                    elapsed_total = time.time() - total_start
                    print(f"  [{eval_count}/{total_evals}] SKIP {group_name} (exists) "
                          f"[{elapsed_total:.0f}s elapsed]")
                    continue

                batch_size = batch_size_guided if guided else batch_size_unguided
                start = time.time()

                state, torque_out = generate_trajectories(
                    model, hnn, torques, device,
                    trajectory_length, num_samples, batch_size,
                    num_diffusion_steps, ctx_frac,
                    guided=guided, seed=seed,
                )

                save_to_h5(h5f, group_name, state, torque_out)
                h5f.flush()

                elapsed = time.time() - start
                elapsed_total = time.time() - total_start
                throughput = num_samples / elapsed
                print(f"  [{eval_count}/{total_evals}] {group_name}: "
                      f"{elapsed:.1f}s ({throughput:.0f} samp/s) "
                      f"[{elapsed_total:.0f}s elapsed]")

    print(f"  Saved: {h5_path}")


def main():
    parser = argparse.ArgumentParser(description="Ablation Trajectory Generation")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["original", "global_cond", "torque_concat"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hnn_checkpoint", type=str,
                        default="/home/gsang/Projects/Perceiver_IO/checkpoints/"
                        "SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt")

    # Torque data paths
    parser.add_argument("--training_torques", type=str,
                        default="data/training_torques_1000_L1500.h5")
    parser.add_argument("--sinusoidal_torques", type=str,
                        default="data/sinusoidal_torques_1000_L1500.h5")
    parser.add_argument("--gp_torques", type=str,
                        default="data/gp_torques_1000_L1500.h5")

    # Experiment parameters
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size_unguided", type=int, default=200)
    parser.add_argument("--batch_size_guided", type=int, default=50)
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--context_fraction", type=float, default=0.2,
                        help="Context fraction for Experiment A")
    parser.add_argument("--seed", type=int, default=228)

    # Output
    parser.add_argument("--output_dir", type=str, default="output_ablation/trajectories")

    # Experiment selection
    parser.add_argument("--torque_conditions", type=str, default="training,sinusoidal,gp,zero",
                        help="Comma-separated torque conditions for Exp A")
    parser.add_argument("--no_exp_a", action="store_true", help="Skip Experiment A")
    parser.add_argument("--no_exp_b", action="store_true", help="Skip Experiment B")

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # Load model
    model, hnn = load_model_and_hnn(
        str(project_root / args.checkpoint),
        str(project_root / args.hnn_checkpoint),
        device,
    )

    # Load torques
    torque_conditions = [t.strip() for t in args.torque_conditions.split(',')]
    torque_dict = {}
    for cond in torque_conditions:
        if cond == 'training':
            torque_dict['training'] = load_torques(
                str(project_root / args.training_torques), device)
        elif cond == 'sinusoidal':
            torque_dict['sinusoidal'] = load_torques(
                str(project_root / args.sinusoidal_torques), device)
        elif cond == 'gp':
            torque_dict['gp'] = load_torques(
                str(project_root / args.gp_torques), device)
        elif cond == 'zero':
            torque_dict['zero'] = torch.zeros(
                args.num_samples, 1500, model.torque_dim, device=device)
            print(f"Created zero torques: [{args.num_samples}, 1500, {model.torque_dim}]")

    # Trajectory lengths for Experiment A
    trajectory_lengths = list(range(50, 1501, 50))  # [50, 100, ..., 1500]

    # Context fractions for Experiment B
    context_fractions = [round(x * 0.05, 2) for x in range(20)]  # [0.0, 0.05, ..., 0.95]

    print(f"\n{'='*80}")
    print(f"  ABLATION TRAJECTORY GENERATION: {args.model_name}")
    print(f"{'='*80}")
    print(f"  Torque conditions: {torque_conditions}")
    print(f"  Trajectory lengths: {trajectory_lengths[0]}-{trajectory_lengths[-1]} ({len(trajectory_lengths)} values)")
    print(f"  Context fractions: {context_fractions[0]}-{context_fractions[-1]} ({len(context_fractions)} values)")
    print(f"  Samples: {args.num_samples}, diffusion steps: {args.num_diffusion_steps}")
    print(f"  Batch sizes: unguided={args.batch_size_unguided}, guided={args.batch_size_guided}")
    print(f"  Run Exp A: {not args.no_exp_a}, Run Exp B: {not args.no_exp_b}")
    print(f"{'='*80}")

    overall_start = time.time()

    # Experiment A
    if not args.no_exp_a:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT A: Variable Trajectory Lengths")
        print(f"{'='*60}")
        run_experiment_a(
            model, hnn, device, args.output_dir, args.model_name,
            torque_dict, trajectory_lengths, args.num_samples,
            args.batch_size_unguided, args.batch_size_guided,
            args.num_diffusion_steps, args.context_fraction, args.seed,
        )

    # Experiment B
    if not args.no_exp_b:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT B: Variable Context Fractions")
        print(f"{'='*60}")
        # Use sinusoidal torques for Exp B
        if 'sinusoidal' in torque_dict:
            exp_b_torques = torque_dict['sinusoidal']
        else:
            # Load sinusoidal torques even if not in torque_conditions
            exp_b_torques = load_torques(
                str(project_root / args.sinusoidal_torques), device)

        run_experiment_b(
            model, hnn, device, args.output_dir, args.model_name,
            exp_b_torques, 1000, context_fractions, args.num_samples,
            args.batch_size_unguided, args.batch_size_guided,
            args.num_diffusion_steps, args.seed,
        )

    total_elapsed = time.time() - overall_start
    hours = total_elapsed / 3600
    print(f"\n{'='*80}")
    print(f"  DONE: {args.model_name} completed in {hours:.1f} hours ({total_elapsed:.0f}s)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
