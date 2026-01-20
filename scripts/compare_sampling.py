"""
Compare energy-guided sampling vs standard sampling.

Evaluates physical consistency by computing MSE between:
- Generated trajectory (from diffusion model)
- Reconstructed trajectory (from MuJoCo physics simulation)

If the diffusion model generates physically plausible trajectories,
the MSE to reconstruction should be low.

Tests:
1. Validation torque: Uses torques from validation data
2. Random torque: Uses random torques (same number as validation)

Both tests use the same seed for fair comparison between guided and unguided sampling.
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import mujoco

from scripts.dataset import TrajectoryDPFCached
from src.models.HNN import HNNWrapper
from src.models.utils import compute_hnn_physics_energy, EMA, reconstruct_traj_with_momentum
from src.models.trajectory_dpf import TrajectoryDPF


def load_model_and_hnn(checkpoint_path, hnn_checkpoint_path, device):
    """Load diffusion model and HNN."""
    # Load diffusion model
    model = TrajectoryDPF.load_from_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    
    # Load EMA weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ema_shadow = checkpoint.get('ema_shadow', None)
    ema_decay = checkpoint.get('ema_decay', 0.9995)
    if ema_shadow is not None:
        model.ema = EMA(model.model, decay=ema_decay)
        for name, tensor in ema_shadow.items():
            if name in model.ema.shadow:
                model.ema.shadow[name] = tensor.to(device=device, dtype=model.ema.shadow[name].dtype)
        model._ema_loaded = True
        print("✓ EMA weights loaded")
    
    # Load HNN
    hnn = HNNWrapper.load_from_checkpoint(hnn_checkpoint_path, map_location=device)
    hnn = hnn.to(device)
    hnn.eval()
    print("✓ HNN loaded")
    
    return model, hnn


def reconstruct_batch(seq_qpos, seq_mom, seq_torque, mj_model, dt, data_dt):
    """
    Reconstruct trajectories using MuJoCo physics.
    
    Args:
        seq_qpos: [B, T, qpos_dim] generated positions
        seq_mom: [B, T, mom_dim] generated momenta
        seq_torque: [B, T, torque_dim] torque conditioning
        mj_model: MuJoCo model
        dt: Fine simulation timestep
        data_dt: Data collection timestep
    
    Returns:
        recon_qpos: [B, T-1, qpos_dim] reconstructed positions
        recon_mom: [B, T-1, mom_dim] reconstructed momenta
    """
    batch_size = seq_qpos.shape[0]
    num_steps = seq_qpos.shape[1]
    
    recon_qpos_list = []
    recon_mom_list = []
    
    for b in range(batch_size):
        qpos_b = seq_qpos[b].cpu().numpy()
        mom_b = seq_mom[b].cpu().numpy()
        torque_b = seq_torque[b].cpu().numpy()
        
        # Compute initial velocity from momentum: v = M^{-1} @ p
        data = mujoco.MjData(mj_model)
        data.qpos[:] = qpos_b[0]
        data.qvel[:] = 0
        mujoco.mj_forward(mj_model, data)
        
        M = np.zeros((mj_model.nv, mj_model.nv))
        mujoco.mj_fullM(mj_model, M, data.qM)
        initial_qvel = np.linalg.solve(M, mom_b[0])
        
        # Reconstruct
        recon = reconstruct_traj_with_momentum(
            mj_model, num_steps, dt,
            qpos_b[0], initial_qvel, torque_b,
            data_dt=data_dt
        )
        
        recon_qpos_list.append(recon['seq_qpos'])
        recon_mom_list.append(recon['seq_mom'])
    
    recon_qpos = torch.tensor(np.stack(recon_qpos_list), dtype=seq_qpos.dtype, device=seq_qpos.device)
    recon_mom = torch.tensor(np.stack(recon_mom_list), dtype=seq_mom.dtype, device=seq_mom.device)
    
    return recon_qpos, recon_mom


def compute_reconstruction_mse(seq_qpos, seq_mom, recon_qpos, recon_mom):
    """
    Compute MSE between generated and reconstructed trajectories.
    
    Note: generated[1:] is compared with reconstructed (both have length T-1)
    """
    # Align: generated[1:] vs reconstructed
    mse_qpos = nn.functional.mse_loss(seq_qpos[:, 1:, :], recon_qpos).item()
    mse_mom = nn.functional.mse_loss(seq_mom[:, 1:, :], recon_mom).item()
    mse_total = mse_qpos + mse_mom
    
    return mse_qpos, mse_mom, mse_total


def sample_batch(model, torque, seed, use_hnn=False, hnn=None, 
                 num_diffusion_steps=200, context_fraction=0.5,
                 guidance_scale=1.5, guidance_steps=200, guidance_lr=0.002,
                 guidance_after_steps=199, lambda_init=1.0):
    """Sample a batch of trajectories with given torque."""
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    trajectory_length = torque.shape[1]
    num_samples = torque.shape[0]
    
    state, _ = model.sample_trajectories(
        num_samples=num_samples,
        trajectory_length=trajectory_length,
        num_diffusion_steps=num_diffusion_steps,
        context_fraction=context_fraction,
        use_ema=True,
        sampler="ddim",
        guidance_scale=guidance_scale,
        hnn=hnn if use_hnn else None,
        guidance_method="adam",
        guidance_after_steps=guidance_after_steps,
        guidance_steps=guidance_steps if use_hnn else 0,
        guidance_lr=guidance_lr,
        lambda_init=lambda_init,
        torque=torque,
    )
    
    return state


def main():
    parser = argparse.ArgumentParser(description="Compare energy-guided vs standard sampling")
    parser.add_argument("--checkpoint", type=str, 
                        default="/home/gsang/Projects/Perceiver_IO/checkpoints/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&ContextLengthCap&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0008.ckpt")
    parser.add_argument("--hnn_checkpoint", type=str,
                        default="/home/gsang/Projects/Perceiver_IO/checkpoints/SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt")
    parser.add_argument("--val_data", type=str,
                        default="/home/gsang/Projects/Perceiver_IO/data/traj_4000-steps_4000.h5")
    parser.add_argument("--mujoco_model", type=str,
                        default="/home/gsang/Projects/Perceiver_IO/configs/rigid_arm_hinge.xml")
    parser.add_argument("--trajectory_length", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_batches", type=int, default=1, 
                        help="Number of batches to test. If None, use all validation data.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_steps", type=int, default=5)
    parser.add_argument("--guidance_lr", type=float, default=0.002)
    parser.add_argument("--guidance_after", type=int, default=25)
    parser.add_argument("--lambda_init", type=float, default=1.0)
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--context_fraction", type=float, default=0.5)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--skip_random", action="store_true", help="Skip random torque test")
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and HNN
    print("\n" + "="*70)
    print("Loading models...")
    print("="*70)
    model, hnn = load_model_and_hnn(args.checkpoint, args.hnn_checkpoint, device)
    
    dt = float(model.dt)
    data_dt = float(model.data_dt)
    qpos_dim = model.qpos_dim
    mom_dim = model.mom_dim
    
    print(f"Model dt: {dt}, data_dt: {data_dt}")
    print(f"Dimensions: qpos={qpos_dim}, mom={mom_dim}")
    
    # Load MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path(args.mujoco_model)
    print(f"✓ MuJoCo model loaded: {args.mujoco_model}")
    
    # Load validation dataset
    print(f"\nLoading validation data: {args.val_data}")
    val_dataset = TrajectoryDPFCached(args.val_data, trajectory_length=args.trajectory_length)
    num_val_samples = len(val_dataset)
    print(f"Validation dataset size: {num_val_samples}")
    
    # Determine number of batches
    if args.num_batches is None:
        num_batches = num_val_samples // args.batch_size
    else:
        num_batches = min(args.num_batches, num_val_samples // args.batch_size)
    
    total_samples = num_batches * args.batch_size
    print(f"Testing on {total_samples} samples ({num_batches} batches of {args.batch_size})")
    
    # =========================================================================
    # TEST 1: Validation torque
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: Validation Torque - MSE(Generated, Reconstructed)")
    print("="*70)
    
    results_val = {
        'standard': {'mse_qpos': [], 'mse_mom': [], 'mse_total': [], 'hnn_energy': []},
        'guided': {'mse_qpos': [], 'mse_mom': [], 'mse_total': [], 'hnn_energy': []}
    }
    
    for batch_idx in tqdm(range(num_batches), desc="Validation batches"):
        # Get batch of validation data
        start_idx = batch_idx * args.batch_size
        end_idx = start_idx + args.batch_size
        
        batch_torque = []
        for i in range(start_idx, end_idx):
            sample = val_dataset[i]
            batch_torque.append(sample['seq_torque'])
        
        batch_torque = torch.stack(batch_torque).to(device)  # [B, T, torque_dim]
        
        # Use same seed for both runs
        batch_seed = args.seed + batch_idx
        
        # === Standard sampling (no HNN guidance) ===
        state_std = sample_batch(
            model, batch_torque, batch_seed, use_hnn=False, hnn=hnn,
            num_diffusion_steps=args.num_diffusion_steps,
            context_fraction=args.context_fraction,
            guidance_scale=args.guidance_scale,
        )
        
        # Denormalize
        state_std_denorm = model.denormalize_state(state_std)
        qpos_std = state_std_denorm[:, :, :qpos_dim]
        mom_std = state_std_denorm[:, :, qpos_dim:]
        
        # Reconstruct using MuJoCo
        recon_qpos_std, recon_mom_std = reconstruct_batch(
            qpos_std, mom_std, batch_torque, mj_model, dt, data_dt
        )
        
        # Compute MSE(generated, reconstructed)
        mse_qpos_std, mse_mom_std, mse_total_std = compute_reconstruction_mse(
            qpos_std, mom_std, recon_qpos_std, recon_mom_std
        )
        
        # Compute HNN energy
        with torch.enable_grad():
            qpos_grad = qpos_std.clone().requires_grad_(True)
            mom_grad = mom_std.clone().requires_grad_(True)
            energy_std, _, _, _ = compute_hnn_physics_energy(
                qpos_grad, mom_grad, batch_torque, hnn, data_dt, 
                lambda_init=0.0, return_components=True
            )
        
        results_val['standard']['mse_qpos'].append(mse_qpos_std)
        results_val['standard']['mse_mom'].append(mse_mom_std)
        results_val['standard']['mse_total'].append(mse_total_std)
        results_val['standard']['hnn_energy'].append(energy_std.item())
        
        # === Energy-guided sampling ===
        state_guided = sample_batch(
            model, batch_torque, batch_seed, use_hnn=True, hnn=hnn,
            num_diffusion_steps=args.num_diffusion_steps,
            context_fraction=args.context_fraction,
            guidance_scale=args.guidance_scale,
            guidance_steps=args.guidance_steps,
            guidance_lr=args.guidance_lr,
            guidance_after_steps=args.guidance_after,
            lambda_init=args.lambda_init,
        )
        
        # Denormalize
        state_guided_denorm = model.denormalize_state(state_guided)
        qpos_guided = state_guided_denorm[:, :, :qpos_dim]
        mom_guided = state_guided_denorm[:, :, qpos_dim:]
        
        # Reconstruct using MuJoCo
        recon_qpos_guided, recon_mom_guided = reconstruct_batch(
            qpos_guided, mom_guided, batch_torque, mj_model, dt, data_dt
        )
        
        # Compute MSE(generated, reconstructed)
        mse_qpos_guided, mse_mom_guided, mse_total_guided = compute_reconstruction_mse(
            qpos_guided, mom_guided, recon_qpos_guided, recon_mom_guided
        )
        
        # Compute HNN energy
        with torch.enable_grad():
            qpos_grad = qpos_guided.clone().requires_grad_(True)
            mom_grad = mom_guided.clone().requires_grad_(True)
            energy_guided, _, _, _ = compute_hnn_physics_energy(
                qpos_grad, mom_grad, batch_torque, hnn, data_dt,
                lambda_init=0.0, return_components=True
            )
        
        results_val['guided']['mse_qpos'].append(mse_qpos_guided)
        results_val['guided']['mse_mom'].append(mse_mom_guided)
        results_val['guided']['mse_total'].append(mse_total_guided)
        results_val['guided']['hnn_energy'].append(energy_guided.item())
    
    # Print results for validation torque
    print("\n" + "-"*70)
    print("VALIDATION TORQUE RESULTS: MSE(Generated, Reconstructed)")
    print("-"*70)
    print(f"{'Metric':<20} {'Standard':<25} {'Energy-Guided':<25} {'Improvement':<15}")
    print("-"*70)
    
    for metric in ['mse_qpos', 'mse_mom', 'mse_total', 'hnn_energy']:
        std_mean = np.mean(results_val['standard'][metric])
        std_std = np.std(results_val['standard'][metric])
        guided_mean = np.mean(results_val['guided'][metric])
        guided_std = np.std(results_val['guided'][metric])
        
        if std_mean > 0:
            improvement = (std_mean - guided_mean) / std_mean * 100
            imp_str = f"{improvement:+.1f}%"
        else:
            imp_str = "N/A"
        
        print(f"{metric:<20} {std_mean:.6f}±{std_std:.6f}  {guided_mean:.6f}±{guided_std:.6f}  {imp_str}")
    
    # =========================================================================
    # TEST 2: Random torque (same number as validation)
    # =========================================================================
    if not args.skip_random:
        print("\n" + "="*70)
        print("TEST 2: Random Torque - MSE(Generated, Reconstructed)")
        print("="*70)
        
        results_random = {
            'standard': {'mse_qpos': [], 'mse_mom': [], 'mse_total': [], 'hnn_energy': []},
            'guided': {'mse_qpos': [], 'mse_mom': [], 'mse_total': [], 'hnn_energy': []}
        }
        
        for batch_idx in tqdm(range(num_batches), desc="Random torque batches"):
            batch_seed = args.seed + 10000 + batch_idx  # Different seed range
            
            # Generate random torque
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)
            
            random_torque = model._generate_random_torque(
                args.batch_size, args.trajectory_length, data_dt
            )
            
            # === Standard sampling ===
            state_std = sample_batch(
                model, random_torque, batch_seed, use_hnn=False, hnn=hnn,
                num_diffusion_steps=args.num_diffusion_steps,
                context_fraction=args.context_fraction,
                guidance_scale=args.guidance_scale,
            )
            
            state_std_denorm = model.denormalize_state(state_std)
            qpos_std = state_std_denorm[:, :, :qpos_dim]
            mom_std = state_std_denorm[:, :, qpos_dim:]
            
            recon_qpos_std, recon_mom_std = reconstruct_batch(
                qpos_std, mom_std, random_torque, mj_model, dt, data_dt
            )
            
            mse_qpos_std, mse_mom_std, mse_total_std = compute_reconstruction_mse(
                qpos_std, mom_std, recon_qpos_std, recon_mom_std
            )
            
            with torch.enable_grad():
                qpos_grad = qpos_std.clone().requires_grad_(True)
                mom_grad = mom_std.clone().requires_grad_(True)
                energy_std, _, _, _ = compute_hnn_physics_energy(
                    qpos_grad, mom_grad, random_torque, hnn, data_dt,
                    lambda_init=0.0, return_components=True
                )
            
            results_random['standard']['mse_qpos'].append(mse_qpos_std)
            results_random['standard']['mse_mom'].append(mse_mom_std)
            results_random['standard']['mse_total'].append(mse_total_std)
            results_random['standard']['hnn_energy'].append(energy_std.item())
            
            # === Energy-guided sampling ===
            state_guided = sample_batch(
                model, random_torque, batch_seed, use_hnn=True, hnn=hnn,
                num_diffusion_steps=args.num_diffusion_steps,
                context_fraction=args.context_fraction,
                guidance_scale=args.guidance_scale,
                guidance_steps=args.guidance_steps,
                guidance_lr=args.guidance_lr,
                guidance_after_steps=args.guidance_after,
                lambda_init=args.lambda_init,
            )
            
            state_guided_denorm = model.denormalize_state(state_guided)
            qpos_guided = state_guided_denorm[:, :, :qpos_dim]
            mom_guided = state_guided_denorm[:, :, qpos_dim:]
            
            recon_qpos_guided, recon_mom_guided = reconstruct_batch(
                qpos_guided, mom_guided, random_torque, mj_model, dt, data_dt
            )
            
            mse_qpos_guided, mse_mom_guided, mse_total_guided = compute_reconstruction_mse(
                qpos_guided, mom_guided, recon_qpos_guided, recon_mom_guided
            )
            
            with torch.enable_grad():
                qpos_grad = qpos_guided.clone().requires_grad_(True)
                mom_grad = mom_guided.clone().requires_grad_(True)
                energy_guided, _, _, _ = compute_hnn_physics_energy(
                    qpos_grad, mom_grad, random_torque, hnn, data_dt,
                    lambda_init=0.0, return_components=True
                )
            
            results_random['guided']['mse_qpos'].append(mse_qpos_guided)
            results_random['guided']['mse_mom'].append(mse_mom_guided)
            results_random['guided']['mse_total'].append(mse_total_guided)
            results_random['guided']['hnn_energy'].append(energy_guided.item())
        
        # Print results for random torque
        print("\n" + "-"*70)
        print("RANDOM TORQUE RESULTS: MSE(Generated, Reconstructed)")
        print("-"*70)
        print(f"{'Metric':<20} {'Standard':<25} {'Energy-Guided':<25} {'Improvement':<15}")
        print("-"*70)
        
        for metric in ['mse_qpos', 'mse_mom', 'mse_total', 'hnn_energy']:
            std_mean = np.mean(results_random['standard'][metric])
            std_std = np.std(results_random['standard'][metric])
            guided_mean = np.mean(results_random['guided'][metric])
            guided_std = np.std(results_random['guided'][metric])
            
            if std_mean > 0:
                improvement = (std_mean - guided_mean) / std_mean * 100
                imp_str = f"{improvement:+.1f}%"
            else:
                imp_str = "N/A"
            
            print(f"{metric:<20} {std_mean:.6f}±{std_std:.6f}  {guided_mean:.6f}±{guided_std:.6f}  {imp_str}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Lower MSE = Better physical consistency with MuJoCo simulation")
    print("Lower HNN Energy = Better consistency with learned Hamiltonian")
    print("\nReference: Ground truth training data has HNN Energy ~0.72")


if __name__ == '__main__':
    main()
