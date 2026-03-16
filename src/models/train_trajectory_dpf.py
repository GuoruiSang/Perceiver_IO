"""Train CLI entrypoint for Trajectory DPF."""

import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from scripts.data.dataset import TrajectoryDPFCached
from src import config
from src.models.trajectory_dpf_model import TrajectoryDPF
from src.training.callbacks import WandBTrajectoryCallback
from src.training.utils import compute_normalization_stats
from src.models.utils import visualize_trajectory


def main():
    parser = argparse.ArgumentParser(description="Train Trajectory DPF")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["train"], default=config.DEFAULT_MODE,
                        help="Mode: 'train' to train the model")
    
    # Data and model paths
    parser.add_argument("--h5_path", type=str, default=config.DEFAULT_H5_PATH,
                        help="Path to training data h5 file")
    parser.add_argument("--val_h5_path", type=str, default=None,
                        help="Optional path to a separate validation h5 file. "
                             "If set, uses strict train/val split across files.")
    parser.add_argument("--checkpoint_dir", type=str, default=config.DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume_from_checkpoint", type=str, default=config.DEFAULT_RESUME_CHECKPOINT,
                        help="Path to checkpoint to resume training from")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 2],
                        help="GPU device IDs to use for training (e.g., --devices 0 1)")
    parser.add_argument("--num_workers", type=int, default=config.DEFAULT_NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="AdamW beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.99,
                        help="AdamW beta2")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="LR warmup steps (<=0 keeps legacy schedule)")
    parser.add_argument("--grad_clip_val", type=float, default=1.0,
                        help="Global gradient clipping value")
    parser.add_argument("--grad_warn_threshold", type=float, default=10.0,
                        help="Warn when pre-clip grad norm exceeds this value (<=0 disables warning)")
    parser.add_argument("--use_fused_adamw", action="store_true",
                        help="Use fused AdamW on CUDA when available (speed optimization, same math target).")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1,
                        help="Run validation every N epochs (higher is faster, lower monitoring frequency).")
    parser.add_argument("--num_sanity_val_steps", type=int, default=2,
                        help="Sanity validation batches before training (0 skips for faster startup).")
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=10,
                        help="Save model checkpoint every N epochs.")
    parser.add_argument("--disable_wandb_traj_callback", action="store_true",
                        help="Disable expensive WandB trajectory image callback during training.")
    parser.add_argument("--wandb_traj_log_every_n_epochs", type=int, default=10,
                        help="Log the W&B trajectory image callback every N validation epochs.")
    parser.add_argument("--disable_startup_visualization", action="store_true",
                        help="Skip the pre-training debug trajectory plots saved under plots/training_debug.")
    parser.add_argument("--compile_model", action="store_true",
                        help="Enable torch.compile for model forward/backward speedup.")
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (if --compile_model is set).")
    parser.add_argument("--num_latents", type=int, default=config.DEFAULT_NUM_LATENTS)
    parser.add_argument("--num_latent_channels", type=int, default=config.DEFAULT_NUM_LATENT_CHANNELS)
    parser.add_argument("--diffusion_steps", type=int, default=config.DEFAULT_DIFFUSION_STEPS)
    parser.add_argument("--backbone", type=str, default="perceiverio",
                        choices=["perceiverio", "transformer"],
                        help="Backbone type: perceiverio (default) or transformer baseline")
    parser.add_argument("--num_decoder_blocks", type=int, default=4,
                        help="Number of self-attention blocks in the decoder for trajectory refinement")
    parser.add_argument("--max_trajectories", type=int, default=0,
                        help="Max trajectories to use from dataset (0 = all)")
    parser.add_argument("--fixed_trajectory_length", type=int, default=None,
                        help="Fixed trajectory length for non-DPF training (disables variable-length). "
                             "When set, all training samples use this exact length instead of random lengths from 100-1000.")
    parser.add_argument(
        "--conditioning_mode",
        type=str,
        default="adaln_torque",
        choices=["adaln_torque", "concat_torque_in_state"],
        help="How torque enters the model: separate AdaLN conditioning or concatenated into the denoised state.",
    )
    parser.add_argument(
        "--query_context_mode",
        type=str,
        default="random_subset",
        choices=["random_subset", "future_context", "clean_prefix_noisy_suffix"],
        help="Perceiver training/validation query-context construction.",
    )
    parser.add_argument(
        "--token_layout",
        type=str,
        default="aligned_tau",
        choices=["aligned_tau", "shifted_tau"],
        help="Token layout for concatenated-state torque: (state_t, tau_t) or (state_t, tau_{t-1}).",
    )
    parser.add_argument("--unconditional_tau_in_state", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--training_context_mode",
        type=str,
        default=None,
        choices=[
            "random_subset",
            "future_context",
            "clean_prefix_noisy_suffix",
            "shifted_tau_tokens",
            "shifted_future_context",
            "shifted_future_context_cleanprefix",
        ],
        help=argparse.SUPPRESS,
    )
    
    # Training callback sampling parameters
    parser.add_argument(
        "--sampling_torque_policy",
        type=str,
        default="mixed_reacher",
        choices=["sinusoidal", "lpf_uniform", "drift_ou", "mixed_reacher"],
        help="Torque policy used by training-time trajectory callbacks.",
    )
    parser.add_argument(
        "--sampling_torque_mix",
        type=str,
        default="sinusoidal:0.32,lpf_uniform:0.25,drift_ou:0.43",
        help="Policy mixture used when --sampling_torque_policy=mixed_reacher.",
    )
    parser.add_argument(
        "--sampling_lpf_uniform_beta",
        type=float,
        default=0.9992,
        help="LPF beta for lpf_uniform torque sampling policy.",
    )
    parser.add_argument(
        "--sampling_torque_scale",
        type=float,
        default=0.35,
        help="Global scale applied to callback-generated torque.",
    )
    
    # W&B arguments
    parser.add_argument("--wandb", type=bool, default=config.DEFAULT_WANDB_ENABLED, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=config.DEFAULT_WANDB_PROJECT, help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (username or team)")
    parser.add_argument("--wandb_run_name", type=str, default=config.DEFAULT_WANDB_RUN_NAME, help="W&B run name")
    
    args = parser.parse_args()

    # Backward-compatible mapping from the old coupled flags onto the new orthogonal axes.
    if args.unconditional_tau_in_state:
        args.conditioning_mode = "concat_torque_in_state"

    if args.training_context_mode is not None:
        legacy_context_mode_map = {
            "random_subset": ("random_subset", "aligned_tau", None),
            "future_context": ("future_context", "aligned_tau", None),
            "clean_prefix_noisy_suffix": ("clean_prefix_noisy_suffix", "aligned_tau", None),
            "shifted_tau_tokens": ("random_subset", "shifted_tau", "concat_torque_in_state"),
            "shifted_future_context": ("future_context", "shifted_tau", "concat_torque_in_state"),
            "shifted_future_context_cleanprefix": (
                "clean_prefix_noisy_suffix",
                "shifted_tau",
                "concat_torque_in_state",
            ),
        }
        query_context_mode, token_layout, conditioning_mode = legacy_context_mode_map[args.training_context_mode]
        args.query_context_mode = query_context_mode
        args.token_layout = token_layout
        if conditioning_mode is not None:
            args.conditioning_mode = conditioning_mode

    if args.token_layout == "shifted_tau" and args.conditioning_mode != "concat_torque_in_state":
        raise ValueError(
            "--token_layout shifted_tau is only valid when --conditioning_mode=concat_torque_in_state"
        )

    # Speed knobs for modern NVIDIA GPUs (A100 etc.)
    # - TF32 accelerates float32 matmuls on Tensor Cores with negligible impact for most training.
    # - bf16 mixed precision enables Flash SDP kernels for attention in torch 2.0 (big speedup).
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception as e:
            print(f"[Perf] Warning: failed to set float32 matmul precision: {e}")
    
    # Training mode - load dataset and setup training
    dataset_traj_length = args.fixed_trajectory_length if args.fixed_trajectory_length else 1000
    print(f"Loading dataset from {args.h5_path}...")
    full_dataset = TrajectoryDPFCached(args.h5_path, trajectory_length=dataset_traj_length)

    if args.max_trajectories > 0 and args.max_trajectories < len(full_dataset):
        dataset = torch.utils.data.Subset(full_dataset, range(args.max_trajectories))
        print(f"  Subset to {args.max_trajectories} trajectories")
    else:
        dataset = full_dataset

    # Get dimensions from first sample
    sample = dataset[0]
    qpos_dim = sample['seq_qpos'].shape[-1]
    mom_dim = sample['seq_mom'].shape[-1]
    torque_dim = sample['seq_torque'].shape[-1]
    max_timesteps = full_dataset.num_steps
    qpos_representation = getattr(full_dataset, "qpos_representation", "raw")

    # Get simulation metadata from dataset
    dt = full_dataset.dt
    data_dt = full_dataset.data_dt
    xml_content = full_dataset.xml
    
    print(f"Dataset info:")
    print(f"  Trajectories: {len(dataset)}")
    print(f"  Timesteps: {max_timesteps}")
    print(f"  qpos_dim: {qpos_dim}, mom_dim: {mom_dim}, torque_dim: {torque_dim}")
    print(f"  qpos_representation: {qpos_representation}")
    print(f"  dt: {dt}, data_dt: {data_dt}")
    print(f"  XML content: {'loaded' if xml_content else 'not available'}")
    print(f"  conditioning_mode: {args.conditioning_mode}")
    print(f"  query_context_mode: {args.query_context_mode}")
    print(f"  token_layout: {args.token_layout}")
    
    # Create dataloaders:
    # - If val_h5_path is provided, keep strict split across files.
    # - Otherwise preserve existing behavior (random split from one file).
    if args.val_h5_path:
        print(f"Loading validation dataset from {args.val_h5_path}...")
        val_dataset_full = TrajectoryDPFCached(args.val_h5_path, trajectory_length=dataset_traj_length)

        # Guardrail: strict split only makes sense if dimensions align.
        val_sample = val_dataset_full[0]
        if val_sample['seq_qpos'].shape[-1] != qpos_dim or \
           val_sample['seq_mom'].shape[-1] != mom_dim or \
           val_sample['seq_torque'].shape[-1] != torque_dim:
            raise ValueError(
                "Validation dataset dimensions do not match training dataset: "
                f"train(q,p,u)=({qpos_dim},{mom_dim},{torque_dim}), "
                f"val=({val_sample['seq_qpos'].shape[-1]},{val_sample['seq_mom'].shape[-1]},{val_sample['seq_torque'].shape[-1]})"
            )

        train_dataset = dataset
        val_dataset = val_dataset_full
        print(f"Dataset info (strict split):")
        print(f"  Train trajectories: {len(train_dataset)}")
        print(f"  Val trajectories: {len(val_dataset)}")
    else:
        train_size = int(config.DEFAULT_TRAIN_VAL_SPLIT * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        print(f"Dataset info (random split):")
        print(f"  Train trajectories: {len(train_dataset)}")
        print(f"  Val trajectories: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True,
                           persistent_workers=(args.num_workers > 0))
    
    # Compute normalization stats (min-max for scaling to [-1, 1])
    stats_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    qpos_min, qpos_max, mom_min, mom_max, torque_min, torque_max = compute_normalization_stats(
        stats_loader, qpos_dim, mom_dim, torque_dim, max_timesteps,
        qpos_representation=qpos_representation,
    )
    
    print("[Startup] Normalization stats computed.", flush=True)
    # Visualize one trajectory before and after normalization
    if args.disable_startup_visualization:
        print("[Startup] Skipping startup visualization.", flush=True)
    else:
        print("[Startup] Saving trajectory before/after normalization...", flush=True)
        sample = dataset[0]
        sample_qpos = sample['seq_qpos']  # [T, qpos_dim]
        sample_mom = sample['seq_mom']    # [T, mom_dim]
        sample_torque = sample['seq_torque']  # [T, torque_dim]
        
        # Original trajectory dict
        original_traj_dict = {
            'seq_qpos': sample_qpos,
            'seq_mom': sample_mom,
            'seq_torque': sample_torque,
        }
        
        # Save original trajectory
        debug_plot_dir = str(project_root / 'plots' / 'training_debug')
        os.makedirs(debug_plot_dir, exist_ok=True)
        visualize_trajectory(original_traj_dict, debug_plot_dir)
        src_path = f'{debug_plot_dir}/trajectory.jpg'
        dst_path = f'{debug_plot_dir}/trajectory_original.jpg'
        if os.path.exists(src_path):
            os.rename(src_path, dst_path)
            print(f"[Visualization] Saved {dst_path}", flush=True)
        else:
            print(f"[Visualization] Warning: {src_path} not found, skipping rename", flush=True)
        
        # Normalize the trajectory
        if args.conditioning_mode == "concat_torque_in_state":
            state_min = torch.cat([qpos_min, mom_min, torque_min], dim=-1)
            state_max = torch.cat([qpos_max, mom_max, torque_max], dim=-1)
        else:
            state_min = torch.cat([qpos_min, mom_min], dim=-1)
            state_max = torch.cat([qpos_max, mom_max], dim=-1)
        state_range = state_max - state_min
        
        if args.conditioning_mode == "concat_torque_in_state":
            sample_torque_state = sample_torque
            if args.token_layout == "shifted_tau":
                zero_torque = torch.zeros_like(sample_torque[:1, :])
                sample_torque_state = torch.cat([zero_torque, sample_torque[:-1, :]], dim=0)
            full_state = torch.cat([sample_qpos, sample_mom, sample_torque_state], dim=-1)  # [T, state_dim]
        else:
            full_state = torch.cat([sample_qpos, sample_mom], dim=-1)  # [T, state_dim]
        normalized_state = (full_state - state_min) / state_range * 2.0 - 1.0
        
        # Normalize torque separately
        cond_range = torque_max - torque_min
        normalized_torque = (sample_torque - torque_min) / cond_range * 2.0 - 1.0
        
        normalized_traj_dict = {
            'seq_qpos': normalized_state[:, :qpos_dim],
            'seq_mom': normalized_state[:, qpos_dim:qpos_dim + mom_dim],
            'seq_torque': normalized_torque,
        }
        
        # Save normalized trajectory
        visualize_trajectory(normalized_traj_dict, debug_plot_dir)
        src_path_norm = f'{debug_plot_dir}/trajectory.jpg'
        dst_path_norm = f'{debug_plot_dir}/trajectory_normalized.jpg'
        if os.path.exists(src_path_norm):
            os.rename(src_path_norm, dst_path_norm)
            print(f"[Visualization] Saved {dst_path_norm}", flush=True)
        else:
            print(f"[Visualization] Warning: {src_path_norm} not found, skipping rename", flush=True)
        print(f"[Visualization] Original state range: [{full_state.min():.4f}, {full_state.max():.4f}]", flush=True)
        print(f"[Visualization] Normalized state range: [{normalized_state.min():.4f}, {normalized_state.max():.4f}]", flush=True)

    # Determine trajectory length training options
    if args.fixed_trajectory_length:
        traj_length_options = (args.fixed_trajectory_length,)
        print(f"[Fixed-Length] Training with fixed trajectory length: {args.fixed_trajectory_length}")
    else:
        traj_length_options = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)

    print("[Startup] Building model...", flush=True)
    # Create model (per-step state-torque interaction conditioning, prefix context)
    model = TrajectoryDPF(
        qpos_dim=qpos_dim,
        mom_dim=mom_dim,
        torque_dim=torque_dim,
        max_timesteps=max_timesteps,
        diffusion_steps=args.diffusion_steps,
        num_latents=args.num_latents,
        num_latent_channels=args.num_latent_channels,
        cond_dim=256,  # AdaLN conditioning embedding dimension
        num_decoder_blocks=args.num_decoder_blocks,
        trajectory_length_training_options=traj_length_options,
        lr=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        warmup_steps=args.warmup_steps,
        grad_warn_threshold=args.grad_warn_threshold,
        use_fused_adamw=args.use_fused_adamw,
        encoder_cond_mode="none",  # Encoder conditioning: "per_step", "mean", "rnn" or "none"
        backbone=args.backbone,
        conditioning_mode=args.conditioning_mode,
        query_context_mode=args.query_context_mode,
        token_layout=args.token_layout,
        dt=dt,
        data_dt=data_dt,
        xml_content=xml_content,
        qpos_representation=qpos_representation,
        qpos_min=qpos_min,
        qpos_max=qpos_max,
        mom_min=mom_min,
        mom_max=mom_max,
        torque_min=torque_min,
        torque_max=torque_max,
    )

    if args.compile_model:
        if hasattr(torch, "compile"):
            try:
                model.model = torch.compile(model.model, mode=args.compile_mode)
                print(f"[Perf] Enabled torch.compile with mode='{args.compile_mode}'.")
            except Exception as e:
                print(f"[Perf] torch.compile failed ({e}); continuing without compile.")
        else:
            print("[Perf] torch.compile not available in this PyTorch version; continuing without compile.")
    
    print("[Startup] Model constructed.", flush=True)
    # Training mode - setup W&B logger
    logger = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb is not installed. Run 'pip install wandb' to enable W&B logging.")
            print("Continuing without W&B logging.")
        else:
            from pytorch_lightning.loggers import WandbLogger
            wandb_run_name = args.wandb_run_name
            if args.fixed_trajectory_length:
                # Replace VariableTrajLength with FixedTrajLength in the run name
                if wandb_run_name and "VariableTrajLength" in wandb_run_name:
                    wandb_run_name = wandb_run_name.replace("VariableTrajLength", f"FixedTrajLength{args.fixed_trajectory_length}")
                else:
                    wandb_run_name = f"{wandb_run_name}_FixedTrajLength{args.fixed_trajectory_length}" if wandb_run_name else f"FixedTrajLength{args.fixed_trajectory_length}"
            if args.backbone == "transformer":
                wandb_run_name = f"{wandb_run_name}_backbone-transformer" if wandb_run_name else "backbone-transformer"
            if args.conditioning_mode != "adaln_torque":
                cond_tag = f"cond-{args.conditioning_mode}"
                wandb_run_name = f"{wandb_run_name}_{cond_tag}" if wandb_run_name else cond_tag
            if args.query_context_mode != "random_subset":
                ctx_tag = f"ctxMode-{args.query_context_mode}"
                wandb_run_name = f"{wandb_run_name}_{ctx_tag}" if wandb_run_name else ctx_tag
            if args.token_layout != "aligned_tau":
                layout_tag = f"layout-{args.token_layout}"
                wandb_run_name = f"{wandb_run_name}_{layout_tag}" if wandb_run_name else layout_tag
            logger = WandbLogger(
                project=args.wandb_project,
                name=wandb_run_name,
                entity=args.wandb_entity,
                save_dir=args.checkpoint_dir,
                log_model=True,
            )
            
            # Log dataset and training info as hyperparameters
            logger.log_hyperparams({
                'dataset_path': args.h5_path,
                'num_trajectories': len(dataset),
                'trajectory_length': max_timesteps,
                'qpos_dim': qpos_dim,
                'mom_dim': mom_dim,
                'torque_dim': torque_dim,
                'batch_size': args.batch_size,
                'num_workers': args.num_workers,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'adam_beta1': args.adam_beta1,
                'adam_beta2': args.adam_beta2,
                'warmup_steps': args.warmup_steps,
                'grad_clip_val': args.grad_clip_val,
                'grad_warn_threshold': args.grad_warn_threshold,
                'use_fused_adamw': args.use_fused_adamw,
                'accumulate_grad_batches': args.accumulate_grad_batches,
                'check_val_every_n_epoch': args.check_val_every_n_epoch,
                'num_sanity_val_steps': args.num_sanity_val_steps,
                'checkpoint_every_n_epochs': args.checkpoint_every_n_epochs,
                'disable_wandb_traj_callback': args.disable_wandb_traj_callback,
                'wandb_traj_log_every_n_epochs': args.wandb_traj_log_every_n_epochs,
                'compile_model': args.compile_model,
                'compile_mode': args.compile_mode,
                'num_latents': args.num_latents,
                'num_latent_channels': args.num_latent_channels,
                'num_decoder_blocks': args.num_decoder_blocks,
                'diffusion_steps': args.diffusion_steps,
                'epochs': args.epochs,
                'backbone': args.backbone,
                'conditioning_mode': args.conditioning_mode,
                'query_context_mode': args.query_context_mode,
                'token_layout': args.token_layout,
                'fixed_trajectory_length': args.fixed_trajectory_length,
                'trajectory_length_training_options': list(traj_length_options),
            })
            print(f"Initialized W&B logging: project={args.wandb_project}", flush=True)
    
    print("[Startup] Setting up callbacks...", flush=True)
    # Setup callbacks
    callbacks = []
    
    backbone_tag = "_backbone-transformer" if args.backbone == "transformer" else ""
    conditioning_tag = (
        f"_cond-{args.conditioning_mode}"
        if args.conditioning_mode != "adaln_torque"
        else ""
    )
    layout_tag = f"_layout-{args.token_layout}" if args.token_layout != "aligned_tau" else ""
    length_tag = f"FixedTrajLength{args.fixed_trajectory_length}" if args.fixed_trajectory_length else "VariableTrajLength"
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f'trajectory_dpf_x0Stabilized&AbsoluteTimeEncoding&{length_tag}&UniformContext&EncoderNone&DecoderAttentions{backbone_tag}{conditioning_tag}{layout_tag}:{{epoch:03d}}_val_loss:{{val_loss:.4f}}',
        every_n_epochs=args.checkpoint_every_n_epochs,
    )
    callbacks.append(checkpoint_callback)
    
    # Add W&B trajectory logging callback if W&B is enabled
    if args.wandb and WANDB_AVAILABLE and not args.disable_wandb_traj_callback:
        wandb_traj_callback = WandBTrajectoryCallback(
            log_every_n_epochs=max(1, args.wandb_traj_log_every_n_epochs),
            num_samples=1,
            sampling_torque_policy=args.sampling_torque_policy,
            sampling_torque_mix=args.sampling_torque_mix,
            sampling_lpf_uniform_beta=args.sampling_lpf_uniform_beta,
            sampling_torque_scale=args.sampling_torque_scale,
        )
        callbacks.append(wandb_traj_callback)
    
    print("[Startup] Building trainer...", flush=True)
    # Setup trainer
    effective_grad_clip_val = args.grad_clip_val if args.grad_clip_val > 0 else None
    if effective_grad_clip_val is None:
        print("Gradient clipping disabled (grad_clip_val <= 0).")
    trainer_strategy = "auto"
    if (
        torch.cuda.is_available()
        and len(args.devices) > 1
        and args.backbone == "perceiverio"
        and args.conditioning_mode == "concat_torque_in_state"
    ):
        # In concat-state Perceiver mode, torque-conditioning submodules are intentionally unused.
        # DDP needs find_unused_parameters=True to avoid bucket rebuild failures.
        trainer_strategy = "ddp_find_unused_parameters_true"
        print("[Trainer] Using strategy=ddp_find_unused_parameters_true for concat-state PerceiverIO multi-GPU.")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.devices,
        strategy=trainer_strategy,
        callbacks=callbacks,
        logger=logger,
        # precision="bf16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=effective_grad_clip_val,  # Prevents gradient explosion when enabled
        gradient_clip_algorithm="norm",  # Clip by global norm (more stable than value)
        accumulate_grad_batches=max(1, args.accumulate_grad_batches),
        check_val_every_n_epoch=max(1, args.check_val_every_n_epoch),
        num_sanity_val_steps=max(0, args.num_sanity_val_steps),
        log_every_n_steps=10,
    )
    
    print("[Startup] Trainer constructed.", flush=True)
    # Train
    print("Starting training...", flush=True)
    ckpt_path = None
    if args.resume_from_checkpoint:
        # Only resume when the checkpoint path exists; otherwise start from scratch.
        if os.path.exists(args.resume_from_checkpoint):
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            # Let Lightning restore full training state (epoch, optimizer, schedulers).
            ckpt_path = args.resume_from_checkpoint

            # Additionally ensure model weights can load even if there are benign mismatches.
            try:
                checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
                model.load_state_dict(checkpoint.get('state_dict', {}), strict=False)
            except Exception as e:
                print(f"[Training] Non-strict model weight preload skipped due to: {e}")
        else:
            print(f"[Training] Resume checkpoint not found, starting fresh: {args.resume_from_checkpoint}")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    print("Training complete!")


if __name__ == "__main__":
    main()
