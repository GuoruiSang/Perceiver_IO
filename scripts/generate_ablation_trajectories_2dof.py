"""
Ablation Trajectory Generation for 2-DoF System.

Generates trajectories for all experimental conditions and saves them to H5.

Experiments:
  A) Variable trajectory lengths: 4 torque conditions x 20 lengths x 2 modes (unguided/guided)

Usage:
    python scripts/generate_ablation_trajectories_2dof.py --device cuda:0
    python scripts/generate_ablation_trajectories_2dof.py --device cuda:0 --torque_conditions sinusoidal,gp
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


def set_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 2-DoF specific paths
DPF_CHECKPOINT = project_root / 'checkpoints' / '2dof' / 'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0008.ckpt'
HNN_CHECKPOINT = project_root / 'checkpoints' / '2dof' / 'SeperableHNN-2DOF-epoch-epoch=999.ckpt'
OUTPUT_DIR = project_root / 'output_ablation' / 'trajectories' / '2dof'

# Torque data paths (2-DoF)
TORQUE_PATHS = {
    'sinusoidal': project_root / 'data' / '2dof' / 'sinusoidal_torques_2000_L1500.h5',
    'gp': project_root / 'data' / '2dof' / 'gp_torques_2000_L1500.h5',
    'spline': project_root / 'data' / '2dof' / 'spline_torques_1000_L1500.h5',
    'zero': None,  # Generated on-the-fly
}

# Dimensions for 2-DoF system
QPOS_DIM = 2
MOM_DIM = 2
TORQUE_DIM = 2


def load_model_and_hnn(checkpoint_path, hnn_checkpoint_path, device):
    """Load model with EMA + HNN."""
    print(f"Loading DPF from: {checkpoint_path}")
    model = TrajectoryDPF.load_from_checkpoint(str(checkpoint_path), map_location=device, strict=False)
    model = model.to(device)
    model.eval()

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
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
        hnn = HNNWrapper.load_from_checkpoint(str(hnn_checkpoint_path), map_location=device)
        hnn = hnn.to(device)
        hnn.eval()

    return model, hnn


def load_torques(torque_label, num_samples, max_length, device):
    """Load torque sequences for a given policy."""
    if torque_label == 'zero':
        print(f"Creating zero torques: [{num_samples}, {max_length}, {TORQUE_DIM}]")
        return torch.zeros(num_samples, max_length, TORQUE_DIM, device=device)

    path = TORQUE_PATHS[torque_label]
    if not path.exists():
        raise FileNotFoundError(f"Torque file not found: {path}")

    print(f"Loading torques from: {path}")
    with h5py.File(path, 'r') as f:
        torques = f['torques'][:num_samples, :max_length]

    torques = torch.tensor(torques, dtype=torch.float32, device=device)
    print(f"  Shape: {torques.shape}, range: [{torques.min():.3f}, {torques.max():.3f}]")
    return torques


def generate_trajectories(
    model, hnn, torques, device,
    trajectory_length, num_samples, batch_size,
    num_diffusion_steps, context_fraction,
    guided, seed=228,
):
    """Generate trajectories and return state + torque tensors.

    IMPORTANT: For fair comparison between unguided and guided:
    - Both calls must use the SAME seed
    - This ensures sample i gets the same initial noise in both modes
    """
    traj_torques = torques[:num_samples, :trajectory_length, :]

    # Reset all random states for reproducibility
    set_seed(seed)

    all_states = []
    all_torques = []

    guidance_kwargs = {}
    if guided and hnn is not None:
        # Optimal parameters from guidance parameter search
        guidance_kwargs = dict(
            hnn=hnn,
            guidance_method='adam',
            guidance_steps=10,
            guidance_lr=0.0001,       # lr=1e-4
            guidance_after_steps=45,
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
                use_ema=True,
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
                        use_ema=True,
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
    model, hnn, device,
    torque_dict, trajectory_lengths, num_samples,
    batch_size_unguided, batch_size_guided,
    num_diffusion_steps, context_fraction, seed,
):
    """Run Experiment A: variable trajectory lengths for each torque condition."""
    total_evals = len(torque_dict) * len(trajectory_lengths) * 2
    eval_count = 0
    total_start = time.time()

    for torque_label, torques in torque_dict.items():
        h5_path = OUTPUT_DIR / f'exp_a_{torque_label}.h5'
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Check for existing file to enable resuming
        existing_groups = set()
        if h5_path.exists():
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


def main():
    parser = argparse.ArgumentParser(description="2-DoF Ablation Trajectory Generation")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Experiment parameters
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size_unguided", type=int, default=200)
    parser.add_argument("--batch_size_guided", type=int, default=50)
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--context_fraction", type=float, default=0.2,
                        help="Context fraction for Experiment A")
    parser.add_argument("--seed", type=int, default=228)

    # Experiment selection
    parser.add_argument("--torque_conditions", type=str, default="sinusoidal,gp,zero,spline",
                        help="Comma-separated torque conditions for Exp A")

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # Load model
    model, hnn = load_model_and_hnn(DPF_CHECKPOINT, HNN_CHECKPOINT, device)

    # Load torques
    torque_conditions = [t.strip() for t in args.torque_conditions.split(',')]
    torque_dict = {}
    max_length = 1500

    for cond in torque_conditions:
        torque_dict[cond] = load_torques(cond, args.num_samples, max_length, device)

    # Trajectory lengths for Experiment A (match 3-DoF: 50 to 1000 in steps of 50)
    trajectory_lengths = list(range(50, 1001, 50))  # [50, 100, ..., 1000]

    print(f"\n{'='*80}")
    print(f"  2-DoF ABLATION TRAJECTORY GENERATION")
    print(f"{'='*80}")
    print(f"  DPF checkpoint: {DPF_CHECKPOINT.name}")
    print(f"  HNN checkpoint: {HNN_CHECKPOINT.name}")
    print(f"  Torque conditions: {torque_conditions}")
    print(f"  Trajectory lengths: {trajectory_lengths[0]}-{trajectory_lengths[-1]} ({len(trajectory_lengths)} values)")
    print(f"  Samples: {args.num_samples}, diffusion steps: {args.num_diffusion_steps}")
    print(f"  Batch sizes: unguided={args.batch_size_unguided}, guided={args.batch_size_guided}")
    print(f"{'='*80}")

    overall_start = time.time()

    # Experiment A
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT A: Variable Trajectory Lengths")
    print(f"{'='*60}")
    run_experiment_a(
        model, hnn, device,
        torque_dict, trajectory_lengths, args.num_samples,
        args.batch_size_unguided, args.batch_size_guided,
        args.num_diffusion_steps, args.context_fraction, args.seed,
    )

    total_elapsed = time.time() - overall_start
    hours = total_elapsed / 3600
    print(f"\n{'='*80}")
    print(f"  DONE: 2-DoF trajectories completed in {hours:.1f} hours ({total_elapsed:.0f}s)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
