"""
Three-stage hyperparameter sweep for guidance tuning.

Stage 1: Find best CFG guidance_scale (no energy guidance)
Stage 2: Fix best guidance_scale, sweep energy-based guidance configs
Stage 3: Validation with visualization using the best config

Usage:
    # Run all stages automatically
    python scripts/evaluate_guidance_sweep.py --stage all --config configs/guidance_sweep.yaml
    
    # Or run individual stages
    python scripts/evaluate_guidance_sweep.py --stage 1 --config configs/guidance_sweep.yaml
    python scripts/evaluate_guidance_sweep.py --stage 2 --config configs/guidance_sweep.yaml --guidance_scale 1.2
    python scripts/evaluate_guidance_sweep.py --stage 3 --config configs/guidance_sweep.yaml --guidance_scale 1.2 --best_config "adam,20,0.002,150,1.0,None"
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
import h5py
import torch
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import os

from src.models.trajectory_dpf import TrajectoryDPF
from src.models.utils import EMA, compare_generated_with_reconstructed


def load_model_and_hnn(checkpoint_path, hnn_checkpoint_path, device):
    """Load the trajectory model and HNN from checkpoints."""
    print(f"Loading model from: {checkpoint_path}")
    model = TrajectoryDPF.load_from_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    
    # Load EMA shadow if present
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
    
    # Load HNN
    hnn = None
    if hnn_checkpoint_path:
        from src.models.HNN import HNNWrapper
        print(f"Loading HNN from: {hnn_checkpoint_path}")
        hnn = HNNWrapper.load_from_checkpoint(hnn_checkpoint_path, map_location=device)
        hnn = hnn.to(device)
        hnn.eval()
    
    return model, hnn


def load_test_torques(torque_file, device):
    """Load pre-generated test torques from HDF5."""
    print(f"Loading test torques from: {torque_file}")
    with h5py.File(torque_file, 'r') as f:
        torques = torch.tensor(f['torques'][:], dtype=torch.float32, device=device)
        print(f"  Loaded {torques.shape[0]} torque sequences, shape: {torques.shape}")
        print(f"  Metadata: seed={f.attrs['seed']}, dt={f.attrs['dt']}")
    return torques


def compute_mse_batch(model, state_tensor, torque_tensor):
    """Compute MSE for a batch of trajectories (no plotting)."""
    mse_qpos_list = []
    mse_mom_list = []
    
    for i in range(state_tensor.shape[0]):
        state_np = state_tensor[i].cpu().numpy()
        torque_np = torque_tensor[i].cpu().numpy()
        generated = {
            'seq_qpos': state_np[:, :model.qpos_dim],
            'seq_mom': state_np[:, model.qpos_dim:],
            'seq_torque': torque_np,
        }
        mse_dict = compare_generated_with_reconstructed(
            generated, 
            str(project_root / 'configs/rigid_arm_hinge.xml'),
            str(project_root / 'plots'),
            dt=float(model.dt),
            data_dt=float(model.data_dt),
            name=None  # No plotting
        )
        mse_qpos_list.append(mse_dict['mse_qpos'])
        mse_mom_list.append(mse_dict['mse_mom'])
    
    return {
        'qpos': np.array(mse_qpos_list),
        'mom': np.array(mse_mom_list),
        'total': np.array(mse_qpos_list) + np.array(mse_mom_list)
    }


def run_stage1(config, model, torques, device, output_dir, batch_size=50):
    """Stage 1: Sweep guidance_scale values (no energy guidance)."""
    print("\n" + "="*70)
    print("  STAGE 1: Finding best CFG guidance_scale")
    print("="*70)
    
    guidance_scales = config['stage1']['guidance_scale']
    seed = config['seed']
    results = []
    
    num_samples = torques.shape[0]
    trajectory_length = config['trajectory_length']
    
    for guidance_scale in guidance_scales:
        print(f"\nTesting guidance_scale={guidance_scale}...")
        
        # Reset seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        all_mse_qpos = []
        all_mse_mom = []
        
        # Process in batches to manage memory
        for batch_start in tqdm(range(0, num_samples, batch_size), desc=f"  scale={guidance_scale}"):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_torques = torques[batch_start:batch_end]
            
            # Sample without energy guidance
            state, torque_out = model.sample_trajectories(
                num_samples=batch_torques.shape[0],
                trajectory_length=trajectory_length,
                num_diffusion_steps=config['num_diffusion_steps'],
                context_fraction=config['context_fraction'],
                use_ema=config['use_ema'],
                sampler=config['sampler'],
                guidance_scale=guidance_scale,
                hnn=None,  # No energy guidance
                guidance_steps=0,
                torque=batch_torques,
            )
            
            # Compute MSE
            mse = compute_mse_batch(model, state, torque_out)
            all_mse_qpos.extend(mse['qpos'].tolist())
            all_mse_mom.extend(mse['mom'].tolist())
        
        mse_qpos = np.mean(all_mse_qpos)
        mse_mom = np.mean(all_mse_mom)
        mse_total = mse_qpos + mse_mom
        
        results.append({
            'guidance_scale': guidance_scale,
            'mse_qpos': mse_qpos,
            'mse_mom': mse_mom,
            'mse_total': mse_total,
            'mse_qpos_std': np.std(all_mse_qpos),
            'mse_mom_std': np.std(all_mse_mom),
        })
        
        print(f"  MSE: qpos={mse_qpos:.6f}, mom={mse_mom:.6f}, total={mse_total:.6f}")
    
    # Save results
    df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, 'stage1_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\nStage 1 results saved to: {output_path}")
    
    # Find best
    best_idx = df['mse_total'].idxmin()
    best_scale = df.loc[best_idx, 'guidance_scale']
    best_mse = df.loc[best_idx, 'mse_total']
    
    print(f"\n" + "="*70)
    print(f"  STAGE 1 RESULTS")
    print(f"="*70)
    print(df.to_string(index=False))
    print(f"\n  Best guidance_scale: {best_scale} (MSE total: {best_mse:.6f})")
    print(f"="*70)
    
    return best_scale


def run_stage2(config, model, hnn, torques, guidance_scale, device, output_dir, batch_size=50):
    """Stage 2: Sweep energy guidance configs with fixed guidance_scale."""
    print("\n" + "="*70)
    print(f"  STAGE 2: Finding best energy guidance config")
    print(f"  Fixed guidance_scale: {guidance_scale}")
    print("="*70)
    
    seed = config['seed']
    num_samples = torques.shape[0]
    trajectory_length = config['trajectory_length']
    
    # First, compute baseline (no energy guidance)
    print("\nComputing baseline (no energy guidance)...")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    baseline_mse_qpos = []
    baseline_mse_mom = []
    
    for batch_start in tqdm(range(0, num_samples, batch_size), desc="  Baseline"):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_torques = torques[batch_start:batch_end]
        
        state, torque_out = model.sample_trajectories(
            num_samples=batch_torques.shape[0],
            trajectory_length=trajectory_length,
            num_diffusion_steps=config['num_diffusion_steps'],
            context_fraction=config['context_fraction'],
            use_ema=config['use_ema'],
            sampler=config['sampler'],
            guidance_scale=guidance_scale,
            hnn=None,
            guidance_steps=0,
            torque=batch_torques,
        )
        
        mse = compute_mse_batch(model, state, torque_out)
        baseline_mse_qpos.extend(mse['qpos'].tolist())
        baseline_mse_mom.extend(mse['mom'].tolist())
    
    baseline_qpos = np.mean(baseline_mse_qpos)
    baseline_mom = np.mean(baseline_mse_mom)
    baseline_total = baseline_qpos + baseline_mom
    print(f"  Baseline MSE: qpos={baseline_qpos:.6f}, mom={baseline_mom:.6f}, total={baseline_total:.6f}")
    
    # Generate all config combinations
    stage2_config = config['stage2']
    
    # For adam_integration, include chunk_length; for adam, skip it
    configs_to_test = []
    
    for method in stage2_config['guidance_method']:
        for steps in stage2_config['guidance_steps']:
            for lr in stage2_config['guidance_lr']:
                for after_steps in stage2_config['guidance_after_steps']:
                    for lambda_init in stage2_config['lambda_init']:
                        if method == 'adam_integration':
                            for chunk_length in stage2_config['chunk_length']:
                                configs_to_test.append({
                                    'guidance_method': method,
                                    'guidance_steps': steps,
                                    'guidance_lr': lr,
                                    'guidance_after_steps': after_steps,
                                    'lambda_init': lambda_init,
                                    'chunk_length': chunk_length,
                                })
                        else:
                            configs_to_test.append({
                                'guidance_method': method,
                                'guidance_steps': steps,
                                'guidance_lr': lr,
                                'guidance_after_steps': after_steps,
                                'lambda_init': lambda_init,
                                'chunk_length': None,
                            })
    
    print(f"\nTesting {len(configs_to_test)} configurations...")
    
    results = []
    
    for cfg_idx, cfg in enumerate(configs_to_test):
        cfg_str = f"{cfg['guidance_method']}, steps={cfg['guidance_steps']}, lr={cfg['guidance_lr']}, after={cfg['guidance_after_steps']}, lambda={cfg['lambda_init']}"
        if cfg['chunk_length']:
            cfg_str += f", chunk={cfg['chunk_length']}"
        
        print(f"\n[{cfg_idx+1}/{len(configs_to_test)}] {cfg_str}")
        
        # Reset seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        guided_mse_qpos = []
        guided_mse_mom = []
        
        for batch_start in tqdm(range(0, num_samples, batch_size), desc="  Progress", leave=False):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_torques = torques[batch_start:batch_end]
            
            state, torque_out = model.sample_trajectories(
                num_samples=batch_torques.shape[0],
                trajectory_length=trajectory_length,
                num_diffusion_steps=config['num_diffusion_steps'],
                context_fraction=config['context_fraction'],
                use_ema=config['use_ema'],
                sampler=config['sampler'],
                guidance_scale=guidance_scale,
                hnn=hnn,
                guidance_method=cfg['guidance_method'],
                guidance_after_steps=cfg['guidance_after_steps'],
                guidance_steps=cfg['guidance_steps'],
                guidance_lr=cfg['guidance_lr'],
                lambda_init=cfg['lambda_init'],
                chunk_length=cfg['chunk_length'] if cfg['chunk_length'] else 15,
                torque=batch_torques,
            )
            
            mse = compute_mse_batch(model, state, torque_out)
            guided_mse_qpos.extend(mse['qpos'].tolist())
            guided_mse_mom.extend(mse['mom'].tolist())
        
        guided_qpos = np.mean(guided_mse_qpos)
        guided_mom = np.mean(guided_mse_mom)
        guided_total = guided_qpos + guided_mom
        
        improvement_abs = baseline_total - guided_total
        improvement_pct = (improvement_abs / baseline_total) * 100 if baseline_total > 0 else 0
        
        results.append({
            'guidance_method': cfg['guidance_method'],
            'guidance_steps': cfg['guidance_steps'],
            'guidance_lr': cfg['guidance_lr'],
            'guidance_after_steps': cfg['guidance_after_steps'],
            'chunk_length': cfg['chunk_length'],
            'lambda_init': cfg['lambda_init'],
            'mse_qpos_baseline': baseline_qpos,
            'mse_qpos_guided': guided_qpos,
            'mse_mom_baseline': baseline_mom,
            'mse_mom_guided': guided_mom,
            'mse_total_baseline': baseline_total,
            'mse_total_guided': guided_total,
            'improvement_abs': improvement_abs,
            'improvement_pct': improvement_pct,
        })
        
        print(f"  MSE: {guided_total:.6f} (baseline: {baseline_total:.6f}), Improvement: {improvement_pct:+.1f}%")
    
    # Save results
    df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, 'stage2_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\nStage 2 results saved to: {output_path}")
    
    # Show top 10 by improvement
    df_sorted = df.sort_values('improvement_pct', ascending=False)
    
    print(f"\n" + "="*70)
    print(f"  STAGE 2 RESULTS - TOP 10 CONFIGURATIONS")
    print(f"="*70)
    print(f"  Baseline MSE total: {baseline_total:.6f}")
    print(f"  Fixed guidance_scale: {guidance_scale}")
    print("-"*70)
    
    top10 = df_sorted.head(10)
    for idx, row in top10.iterrows():
        cfg_str = f"{row['guidance_method']}, steps={int(row['guidance_steps'])}, lr={row['guidance_lr']}, after={int(row['guidance_after_steps'])}, lambda={row['lambda_init']}"
        if pd.notna(row['chunk_length']):
            cfg_str += f", chunk={int(row['chunk_length'])}"
        print(f"  {row['improvement_pct']:+6.1f}%  MSE={row['mse_total_guided']:.6f}  | {cfg_str}")
    
    print("="*70)
    
    # Return best config as dict
    best_row = df_sorted.iloc[0]
    best_config = {
        'guidance_method': best_row['guidance_method'],
        'guidance_steps': int(best_row['guidance_steps']),
        'guidance_lr': best_row['guidance_lr'],
        'guidance_after_steps': int(best_row['guidance_after_steps']),
        'lambda_init': best_row['lambda_init'],
        'chunk_length': int(best_row['chunk_length']) if pd.notna(best_row['chunk_length']) else None,
        'improvement_pct': best_row['improvement_pct'],
        'mse_total_guided': best_row['mse_total_guided'],
        'baseline_total': baseline_total,
    }
    
    return df_sorted, best_config


def run_stage3(config, model, hnn, torques, guidance_scale, best_config, device, output_dir, num_visualize=8):
    """Stage 3: Validation with visualization using best config."""
    print("\n" + "="*70)
    print(f"  STAGE 3: Validation with Visualization")
    print(f"  Best config: {best_config['guidance_method']}, steps={best_config['guidance_steps']}, "
          f"lr={best_config['guidance_lr']}, after={best_config['guidance_after_steps']}, "
          f"lambda={best_config['lambda_init']}")
    if best_config['chunk_length']:
        print(f"             chunk_length={best_config['chunk_length']}")
    print("="*70)
    
    seed = config['seed']
    trajectory_length = config['trajectory_length']
    
    # Use first num_visualize torques for visualization
    vis_torques = torques[:num_visualize]
    
    # Create visualization output directory
    vis_dir = os.path.join(output_dir, 'stage3_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate baseline (no energy guidance) for comparison
    print(f"\nGenerating {num_visualize} baseline trajectories...")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    baseline_state, baseline_torque = model.sample_trajectories(
        num_samples=num_visualize,
        trajectory_length=trajectory_length,
        num_diffusion_steps=config['num_diffusion_steps'],
        context_fraction=config['context_fraction'],
        use_ema=config['use_ema'],
        sampler=config['sampler'],
        guidance_scale=guidance_scale,
        hnn=None,
        guidance_steps=0,
        torque=vis_torques,
    )
    
    # Generate guided trajectories with best config
    print(f"Generating {num_visualize} guided trajectories with best config...")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    guided_state, guided_torque = model.sample_trajectories(
        num_samples=num_visualize,
        trajectory_length=trajectory_length,
        num_diffusion_steps=config['num_diffusion_steps'],
        context_fraction=config['context_fraction'],
        use_ema=config['use_ema'],
        sampler=config['sampler'],
        guidance_scale=guidance_scale,
        hnn=hnn,
        guidance_method=best_config['guidance_method'],
        guidance_after_steps=best_config['guidance_after_steps'],
        guidance_steps=best_config['guidance_steps'],
        guidance_lr=best_config['guidance_lr'],
        lambda_init=best_config['lambda_init'],
        chunk_length=best_config['chunk_length'] if best_config['chunk_length'] else 15,
        torque=vis_torques,
    )
    
    # Compute MSE and create visualizations
    print("\nGenerating comparison plots...")
    baseline_mse_list = []
    guided_mse_list = []
    
    for i in range(num_visualize):
        # Baseline
        baseline_np = baseline_state[i].cpu().numpy()
        torque_np = baseline_torque[i].cpu().numpy()
        baseline_gen = {
            'seq_qpos': baseline_np[:, :model.qpos_dim],
            'seq_mom': baseline_np[:, model.qpos_dim:],
            'seq_torque': torque_np,
        }
        baseline_mse = compare_generated_with_reconstructed(
            baseline_gen,
            str(project_root / 'configs/rigid_arm_hinge.xml'),
            vis_dir,
            dt=float(model.dt),
            data_dt=float(model.data_dt),
            name=f'sample{i}_baseline'
        )
        baseline_mse_list.append(baseline_mse['mse_total'])
        
        # Guided
        guided_np = guided_state[i].cpu().numpy()
        guided_gen = {
            'seq_qpos': guided_np[:, :model.qpos_dim],
            'seq_mom': guided_np[:, model.qpos_dim:],
            'seq_torque': torque_np,
        }
        guided_mse = compare_generated_with_reconstructed(
            guided_gen,
            str(project_root / 'configs/rigid_arm_hinge.xml'),
            vis_dir,
            dt=float(model.dt),
            data_dt=float(model.data_dt),
            name=f'sample{i}_guided'
        )
        guided_mse_list.append(guided_mse['mse_total'])
        
        print(f"  Sample {i}: Baseline MSE={baseline_mse['mse_total']:.6f}, "
              f"Guided MSE={guided_mse['mse_total']:.6f}, "
              f"Improvement={((baseline_mse['mse_total'] - guided_mse['mse_total']) / baseline_mse['mse_total'] * 100):+.1f}%")
    
    # Summary
    avg_baseline = np.mean(baseline_mse_list)
    avg_guided = np.mean(guided_mse_list)
    avg_improvement = (avg_baseline - avg_guided) / avg_baseline * 100
    
    print(f"\n" + "="*70)
    print(f"  STAGE 3 SUMMARY")
    print(f"="*70)
    print(f"  Samples visualized: {num_visualize}")
    print(f"  Average Baseline MSE: {avg_baseline:.6f}")
    print(f"  Average Guided MSE:   {avg_guided:.6f}")
    print(f"  Average Improvement:  {avg_improvement:+.1f}%")
    print(f"\n  Visualization plots saved to: {vis_dir}")
    print("="*70)
    
    # Save final summary
    summary = {
        'guidance_scale': guidance_scale,
        'best_config': best_config,
        'validation_results': {
            'num_samples': num_visualize,
            'avg_baseline_mse': avg_baseline,
            'avg_guided_mse': avg_guided,
            'avg_improvement_pct': avg_improvement,
            'per_sample_baseline_mse': baseline_mse_list,
            'per_sample_guided_mse': guided_mse_list,
        }
    }
    
    import json
    summary_path = os.path.join(output_dir, 'final_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Final summary saved to: {summary_path}")
    
    return summary


def run_all_stages(config, model, hnn, torques, device, output_dir, batch_size=50, num_visualize=8):
    """Run all three stages automatically."""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  AUTOMATED HYPERPARAMETER SWEEP - ALL STAGES".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # Stage 1: Find best guidance_scale
    best_scale = run_stage1(config, model, torques, device, output_dir, batch_size)
    
    # Stage 2: Find best energy guidance config
    df_sorted, best_config = run_stage2(config, model, hnn, torques, best_scale, device, output_dir, batch_size)
    
    # Stage 3: Validation with visualization
    summary = run_stage3(config, model, hnn, torques, best_scale, best_config, device, output_dir, num_visualize)
    
    # Final report
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  HYPERPARAMETER SWEEP COMPLETE".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    print(f"\n  Best guidance_scale: {best_scale}")
    print(f"  Best energy guidance config:")
    print(f"    - method: {best_config['guidance_method']}")
    print(f"    - steps: {best_config['guidance_steps']}")
    print(f"    - lr: {best_config['guidance_lr']}")
    print(f"    - after_steps: {best_config['guidance_after_steps']}")
    print(f"    - lambda_init: {best_config['lambda_init']}")
    if best_config['chunk_length']:
        print(f"    - chunk_length: {best_config['chunk_length']}")
    print(f"\n  Improvement over baseline: {best_config['improvement_pct']:+.1f}%")
    print(f"\n  Results saved to: {output_dir}/")
    print("#"*70)
    
    return best_scale, best_config, summary


def parse_best_config(config_str):
    """Parse best config string: 'method,steps,lr,after,lambda,chunk'"""
    parts = config_str.split(',')
    return {
        'guidance_method': parts[0],
        'guidance_steps': int(parts[1]),
        'guidance_lr': float(parts[2]),
        'guidance_after_steps': int(parts[3]),
        'lambda_init': float(parts[4]),
        'chunk_length': int(parts[5]) if parts[5] != 'None' else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Three-stage guidance hyperparameter sweep")
    parser.add_argument("--stage", type=str, default="3", choices=['1', '2', '3', 'all'],
                        help="Stage 1: sweep guidance_scale, Stage 2: sweep energy guidance, Stage 3: validation, all: run everything")
    parser.add_argument("--config", type=str, default="configs/guidance_sweep.yaml",
                        help="Path to sweep configuration YAML")
    parser.add_argument("--guidance_scale", type=float, default=1,
                        help="Fixed guidance_scale for Stage 2/3 (required for stage 2, 3)")
    parser.add_argument("--best_config", type=str, default="adam,20,0.005,40,1.0,None",
                        help="Best config string for Stage 3: 'method,steps,lr,after,lambda,chunk' (required for stage 3)")
    parser.add_argument("--checkpoint", type=str, 
                        default="/home/gsang/Projects/Perceiver_IO/checkpoints/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt",
                        help="Path to model checkpoint")
    parser.add_argument("--hnn_checkpoint", type=str,
                        default="/home/gsang/Projects/Perceiver_IO/checkpoints/SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt",
                        help="Path to HNN checkpoint")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for processing (to manage GPU memory)")
    parser.add_argument("--num_visualize", type=int, default=64,
                        help="Number of samples to visualize in Stage 3")
    parser.add_argument("--num_diffusion_steps", type=int, default=50,
                        help="Number of diffusion steps (default: 50)")
    
    args = parser.parse_args()
    
    # Validate args
    if args.stage == '2' and args.guidance_scale is None:
        parser.error("--guidance_scale is required for stage 2")
    if args.stage == '3' and (args.guidance_scale is None or args.best_config is None):
        parser.error("--guidance_scale and --best_config are required for stage 3")
    
    # Load config
    config_path = project_root / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with CLI arguments
    config['num_diffusion_steps'] = args.num_diffusion_steps
    
    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and HNN
    model, hnn = load_model_and_hnn(args.checkpoint, args.hnn_checkpoint, device)
    
    # Load test torques
    torque_path = project_root / config['test_set_path']
    torques = load_test_torques(str(torque_path), device)
    
    # Run appropriate stage
    if args.stage == 'all':
        run_all_stages(config, model, hnn, torques, device, args.output_dir, args.batch_size, args.num_visualize)
    elif args.stage == '1':
        best_scale = run_stage1(config, model, torques, device, args.output_dir, args.batch_size)
        print(f"\nRecommended command for Stage 2:")
        print(f"  python scripts/evaluate_guidance_sweep.py --stage 2 --config {args.config} --guidance_scale {best_scale}")
    elif args.stage == '2':
        df_sorted, best_config = run_stage2(config, model, hnn, torques, args.guidance_scale, device, args.output_dir, args.batch_size)
        best = best_config
        config_str = f"{best['guidance_method']},{best['guidance_steps']},{best['guidance_lr']},{best['guidance_after_steps']},{best['lambda_init']},{best['chunk_length']}"
        print(f"\nRecommended command for Stage 3:")
        print(f"  python scripts/evaluate_guidance_sweep.py --stage 3 --config {args.config} --guidance_scale {args.guidance_scale} --best_config \"{config_str}\"")
    elif args.stage == '3':
        best_config = parse_best_config(args.best_config)
        run_stage3(config, model, hnn, torques, args.guidance_scale, best_config, device, args.output_dir, args.num_visualize)


if __name__ == "__main__":
    main()
