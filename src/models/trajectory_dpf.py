
import sys
from pathlib import Path

# Add project root to Python path (for running as script)
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import h5py
import math
import os
import json
import mujoco
from typing import Optional, Tuple
import argparse
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from perceiver.model.core import (
    FourierPositionEncoding,
    OutputAdapter,
    PerceiverDecoder,
    PerceiverEncoder,
)
from src.models.utils import (
    EMA,
    visualize_trajectory,
    compare_generated_with_reconstructed,
    compare_multiple_generated_with_reconstructed,
)
from scripts.data.dataset import TrajectoryDPFCached
from scripts.data.generate_dataset_forward import (
    generate_torque_sequence,
    parse_torque_policies,
)
from src import config
from src.training.utils import compute_normalization_stats

from src.models.architectures import (
    TrajectoryOutputAdapter,
    TrajectoryPerceiverIO,
    TrajectoryTransformerDiffusion,
    ConditionedTrajectoryPerceiverIO,
    AblationConfig,
)
import tempfile
import numpy as np
import shutil
from typing import List
import faulthandler


# -------------------------
# W&B Trajectory Logger Callback
# -------------------------

class WandBTrajectoryCallback(pl.Callback):
    """Callback to log sampled trajectory visualizations to W&B during training."""
    
    def __init__(
        self,
        log_every_n_epochs: int = 10,
        num_samples: int = 1,
        sampling_torque_policy: str = "mixed_reacher",
        sampling_torque_mix: str = "sinusoidal:0.32,lpf_uniform:0.25,drift_ou:0.43",
        sampling_lpf_uniform_beta: float = 0.9992,
        sampling_torque_scale: float = 0.35,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.sampling_torque_policy = sampling_torque_policy
        self.sampling_torque_mix = sampling_torque_mix
        self.sampling_lpf_uniform_beta = sampling_lpf_uniform_beta
        self.sampling_torque_scale = sampling_torque_scale

    def _sample_training_pattern_trajectory(self, trainer, pl_module):
        val_loaders = trainer.val_dataloaders
        if isinstance(val_loaders, (list, tuple)):
            if len(val_loaders) == 0:
                raise RuntimeError("No validation dataloader available for W&B trajectory callback.")
            val_loader = val_loaders[0]
        else:
            val_loader = val_loaders

        batch = next(iter(val_loader))
        device = pl_module.device
        # Sample multiple branches from the same validation prefix rather than
        # different trajectories, so the callback image shows a true branch fan-out.
        qpos = batch["seq_qpos"][:1].to(device=device, dtype=torch.float32).repeat(self.num_samples, 1, 1)
        mom = batch["seq_mom"][:1].to(device=device, dtype=torch.float32).repeat(self.num_samples, 1, 1)
        torque = batch["seq_torque"][:1].to(device=device, dtype=torch.float32).repeat(self.num_samples, 1, 1)
        return pl_module.sample_training_pattern_trajectory_from_batch(
            qpos=qpos,
            mom=mom,
            torque=torque,
            num_diffusion_steps=100,
            sampler="ddim",
            use_ema=True,
            resample_context_every_step=True,
            sampling_torque_policy=self.sampling_torque_policy,
            sampling_torque_mix=self.sampling_torque_mix,
            sampling_lpf_uniform_beta=self.sampling_lpf_uniform_beta,
            sampling_torque_scale=self.sampling_torque_scale,
        )
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log sample trajectory after validation epoch."""
        current_epoch = trainer.current_epoch + 1
        
        # Only log from the main process (rank 0) in DDP
        if not trainer.is_global_zero:
            return
        
        print(f"\n[W&B Callback] on_validation_epoch_end called (epoch {current_epoch})")
        
        if not WANDB_AVAILABLE:
            print("[W&B Callback] wandb not available, skipping")
            return
        
        if trainer.logger is None:
            print("[W&B Callback] trainer.logger is None, skipping")
            return
        
        if current_epoch % self.log_every_n_epochs != 0:
            print(f"[W&B Callback] Skipping (epoch {current_epoch} % {self.log_every_n_epochs} != 0)")
            return
        
        try:
            print(f"[W&B] Generating sample trajectory for logging (epoch {current_epoch})...")
            
            # Match the training context/query pattern when the checkpoint uses
            # future-context training; otherwise fall back to generic full sampling.
            state, torque, sample_meta = self._sample_training_pattern_trajectory(trainer, pl_module)
            print(f"[W&B] Sample metadata: {sample_meta}", flush=True)
            
            # Split state into components: [qpos | mom]
            qpos_dim = pl_module.qpos_dim
            mom_dim = pl_module.mom_dim
            
            state_np = state.detach().cpu()
            torque_np = torque.detach().cpu()
            generated_list = []
            for sample_idx in range(state_np.shape[0]):
                state_traj = state_np[sample_idx]
                torque_traj = torque_np[sample_idx]
                generated_list.append(
                    {
                        'seq_qpos': state_traj[:, :qpos_dim],
                        'seq_mom': state_traj[:, qpos_dim:qpos_dim + mom_dim],
                        'seq_torque': torque_traj,
                    }
                )
            
            # Create temporary directory for the plot
            with tempfile.TemporaryDirectory() as tmp_dir:
                print(f"[W&B] Temporary directory: {tmp_dir}")
                # Check if XML content is available for comparison plot
                if pl_module.xml_content is not None:
                    # Write XML to temp file for MuJoCo model loading
                    xml_path = os.path.join(tmp_dir, 'model.xml')
                    with open(xml_path, 'w') as f:
                        f.write(pl_module.xml_content)
                    print(f"[W&B] XML written to: {xml_path}")
                    
                    # Use comparison plot (generated vs physics-reconstructed)
                    print("[W&B] Starting compare_generated_with_reconstructed(...)")
                    faulthandler.dump_traceback_later(60, repeat=False)
                    prefix_len = int(sample_meta.get("prefix_len") or 0)
                    if len(generated_list) > 1:
                        compare_multiple_generated_with_reconstructed(
                            generated_list=generated_list,
                            mujoco_model_path=xml_path,
                            save_path=tmp_dir,
                            dt=pl_module.dt,
                            data_dt=pl_module.data_dt,
                            name='comparison',
                            prefix_len=prefix_len,
                        )
                    else:
                        compare_generated_with_reconstructed(
                            generated_list[0], xml_path, tmp_dir,
                            dt=pl_module.dt, data_dt=pl_module.data_dt,
                            name='comparison'
                        )
                    faulthandler.cancel_dump_traceback_later()
                    print("[W&B] compare_generated_with_reconstructed(...) finished")
                    plot_path = os.path.join(tmp_dir, 'comparison.jpg')
                else:
                    # Fallback to simple visualization if XML not available
                    print("[W&B] XML content not available, using simple visualization")
                    print("[W&B] Starting visualize_trajectory(...)")
                    faulthandler.dump_traceback_later(60, repeat=False)
                    visualize_trajectory(trajectory_dict, tmp_dir)
                    faulthandler.cancel_dump_traceback_later()
                    print("[W&B] visualize_trajectory(...) finished")
                    plot_path = os.path.join(tmp_dir, 'trajectory.jpg')
                
                print(f"[W&B] Plot exists={os.path.exists(plot_path)} path={plot_path}")
                print(f"[W&B] Logging image from: {plot_path}")
                print("[W&B] Creating wandb.Image(...)")
                wandb_image = wandb.Image(plot_path)
                print("[W&B] wandb.Image(...) created")
                print("[W&B] Calling wandb.log(...)")
                wandb.log({
                    'sampled_trajectory': wandb_image,
                })
                print("[W&B] wandb.log(...) finished")
            
            print(f"[W&B] Sample trajectory logged successfully!")
            
        except Exception as e:
            import traceback
            print(f"[W&B] Failed to log sample trajectory: {e}")
            traceback.print_exc()


# -------------------------
# Trajectory DPF Module
# -------------------------

class TrajectoryDPF(pl.LightningModule):
    """
    Diffusion Probabilistic Fields for Trajectory Generation.
    
    Generates (qpos, mom) trajectories conditioned on torque using classifier-free guidance.
    
    Token structure: [qpos | mom | diffusion_enc | temporal_enc]
    - State (qpos, mom): noised and denoised
    - Torque: passed separately for per-step AdaLN conditioning (NOT in tokens)
    
    Per-step control mechanism (implements torque_t ⊗ state_t → state_{t+1}):
    - Compute interaction embedding from (state_t, torque_t)
    - Shift-right so it modulates state_{t+1} prediction
    
    Temporal information is encoded explicitly via Fourier position encodings:
    - Diffusion timestep: Which denoising step (1→diffusion_steps)
    - Temporal position: Position in trajectory sequence (0→T)
    """
    
    def __init__(
        self,
        qpos_dim: int,
        mom_dim: int,
        torque_dim: int,
        max_timesteps: int = 1000,
        diffusion_steps: int = 1000,
        num_frequency_bands_for_diffusion: int = 64,
        num_latents: int = 256,
        num_latent_channels: int = 256,
        cond_dim: int = 256,  # Dimension of conditioning embeddings
        num_decoder_blocks: int = 0,  # NEW: Number of decoder self-attention blocks
        trajectory_length_training_options: Tuple[int, ...] = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000),
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.99,
        warmup_steps: int = 1000,
        grad_warn_threshold: float = 10.0,
        use_fused_adamw: bool = False,
        use_ema: bool = True,
        p_uncond: float = 0.1,  # Probability of dropping conditioning for CFG
        lambda_cond: float = 0.1,  # Weight for conditioning regularization loss
        encoder_cond_mode: str = "none",  # "per_step", "mean", "rnn" or "none" for encoder conditioning
        backbone: str = "perceiverio",  # "perceiverio" or "transformer"
        unconditional_tau_in_state: bool = False,  # If True, model denoises [qpos|mom|torque] unconditionally
        tau_loss_mask_prob: float = 0.0,  # In unconditional mode, probability to drop tau channels from denoise loss
        training_context_mode: str = "random_subset",  # "random_subset", "future_context", "shifted_future_context", or "shifted_future_context_cleanprefix"
        query_loss_decay: str = "none",  # Optional front-loaded weighting over query tokens
        query_loss_decay_strength: float = 1.0,  # Strength parameter for the decay profile
        # Ablation study configuration
        ablation_config: Optional[AblationConfig] = None,
        # Simulation metadata (loaded from dataset)
        dt: float = 0.0001,  # Fine simulation timestep
        data_dt: float = 0.0002,  # Data collection timestep (skip_steps * dt = 2 * 0.0001)
        xml_content: Optional[str] = None,  # MuJoCo model XML content
        # Min-max normalization stats (optional, will be computed if not provided)
        qpos_min: Optional[torch.Tensor] = None,
        qpos_max: Optional[torch.Tensor] = None,
        mom_min: Optional[torch.Tensor] = None,
        mom_max: Optional[torch.Tensor] = None,
        torque_min: Optional[torch.Tensor] = None,
        torque_max: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.qpos_dim = qpos_dim
        self.mom_dim = mom_dim
        self.torque_dim = torque_dim
        self.unconditional_tau_in_state = bool(unconditional_tau_in_state)
        self.state_dim = qpos_dim + mom_dim + (torque_dim if self.unconditional_tau_in_state else 0)
        self.adaln_cond_dim = cond_dim  # Conditioning embedding dimension for AdaLN
        self.num_decoder_blocks = num_decoder_blocks
        self.max_timesteps = max_timesteps
        self.diffusion_steps = diffusion_steps
        self.trajectory_length_training_options = trajectory_length_training_options
        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.warmup_steps = warmup_steps
        self.grad_warn_threshold = grad_warn_threshold
        self.use_fused_adamw = use_fused_adamw
        self.p_uncond = p_uncond  # CFG dropout probability
        self.lambda_cond = lambda_cond  # Conditioning regularization weight
        self.encoder_cond_mode = encoder_cond_mode
        self.backbone = backbone
        self.ablation_config = ablation_config if ablation_config is not None else AblationConfig()
        self.tau_loss_mask_prob = float(max(0.0, min(1.0, tau_loss_mask_prob)))
        if training_context_mode not in {
            "random_subset",
            "future_context",
            "shifted_future_context",
            "shifted_future_context_cleanprefix",
        }:
            raise ValueError(f"Unsupported training_context_mode: {training_context_mode}")
        if training_context_mode in {"shifted_future_context", "shifted_future_context_cleanprefix"} and not self.unconditional_tau_in_state:
            raise ValueError(
                f"training_context_mode={training_context_mode!r} requires unconditional_tau_in_state=True."
            )
        self.training_context_mode = training_context_mode
        if query_loss_decay not in {"none", "linear", "exp", "power"}:
            raise ValueError(f"Unsupported query_loss_decay: {query_loss_decay}")
        self.query_loss_decay = query_loss_decay
        self.query_loss_decay_strength = float(max(0.0, query_loss_decay_strength))

        # Simulation metadata for physics-consistent sampling
        self.dt = dt
        self.data_dt = data_dt
        self.xml_content = xml_content
        
        # Minimum allowed scale for any normalized dimension to avoid division blow-ups
        self.range_epsilon = config.DEFAULT_NORMALIZATION_RANGE_EPSILON
        
        # Fourier position encoding for diffusion timestep
        self.fpe_diffusion = FourierPositionEncoding(
            input_shape=(diffusion_steps,),
            num_frequency_bands=num_frequency_bands_for_diffusion
        )
        self.diffusion_encoding_channels = self.fpe_diffusion.num_position_encoding_channels()
        
        # Temporal position encoding: use absolute sinusoidal (length-independent)
        # This ensures timestep t gets the same encoding regardless of total trajectory length T
        self.num_temporal_frequency_bands = num_frequency_bands_for_diffusion // 2  # Use fewer bands for temporal
        # Output channels = num_bands * 2 (sin + cos for each frequency band)
        self.temporal_encoding_channels = self.num_temporal_frequency_bands * 2
        
        # Total input channels per token:
        # - conditional mode: [qpos | mom | diffusion_enc | temporal_enc]
        # - unconditional mode: [qpos | mom | torque | diffusion_enc | temporal_enc]
        # NOTE: In conditional mode, torque is NOT in tokens unless torque_in_tokens ablation is active.
        num_input_channels_raw = self.state_dim + self.diffusion_encoding_channels + self.temporal_encoding_channels
        if self.ablation_config.torque_in_tokens and not self.unconditional_tau_in_state:
            num_input_channels_raw += self.torque_dim
        
        # Ensure num_input_channels is divisible by num_heads (8) for attention
        num_heads = 8
        if num_input_channels_raw % num_heads != 0:
            padding = num_heads - (num_input_channels_raw % num_heads)
            self.temporal_encoding_channels += padding
            print(f"[Init] Padded temporal_encoding_channels by {padding} to make num_input_channels divisible by {num_heads}")
        num_input_channels = self.state_dim + self.diffusion_encoding_channels + self.temporal_encoding_channels
        if self.ablation_config.torque_in_tokens and not self.unconditional_tau_in_state:
            num_input_channels += self.torque_dim

        # ------------------------------------------------------------------
        # Token encoding caches (major speed win)
        # ------------------------------------------------------------------
        # `build_tokens()` is called every step. Previously it regenerated:
        # - the full diffusion Fourier table [B, diffusion_steps, C] every step
        #   just to slice one timestep, and then copied it to GPU.
        # - the temporal sin/cos table [T, C] every step.
        #
        # Cache both as buffers so they live on the right device and are reused.
        with torch.no_grad():
            # [diffusion_steps, diff_enc_dim]
            diffusion_table = self.fpe_diffusion(1)[0].contiguous()
            # [max_timesteps, temp_enc_dim]
            temporal_table = self._get_temporal_encoding(max_timesteps, device=torch.device("cpu")).contiguous()

        self.register_buffer("diffusion_encoding_table", diffusion_table, persistent=True)
        self.register_buffer("temporal_encoding_table", temporal_table, persistent=True)
        
        if self.backbone == "transformer":
            # Transformer baseline:
            # - conditional mode: [state|torque|diff|temp]
            # - unconditional mode: [state|diff|temp] where state already includes torque
            transformer_input_channels = num_input_channels + (0 if self.unconditional_tau_in_state else self.torque_dim)
            self.model = TrajectoryTransformerDiffusion(
                num_input_channels=transformer_input_channels,
                num_output_channels=self.state_dim,
                d_model=num_latent_channels,
                num_layers=max(1, num_decoder_blocks),
                nhead=8,
            )
        else:
            # PerceiverIO backbone with per-step state-torque interaction conditioning
            self.model = ConditionedTrajectoryPerceiverIO(
                num_input_channels=num_input_channels,
                num_output_channels=self.state_dim,  # Predict noise for (qpos, mom) only
                state_dim=self.state_dim,
                torque_dim=torque_dim,
                num_latents=num_latents,
                num_latent_channels=num_latent_channels,
                cond_dim=cond_dim,
                num_decoder_blocks=num_decoder_blocks,  # Pass it down
                encoder_cond_mode=encoder_cond_mode,
                ablation_config=self.ablation_config,
            )
        
        # Diffusion schedule (cosine)
        s = 0.008
        t_vals = torch.linspace(0, diffusion_steps, diffusion_steps + 1, dtype=torch.float32)
        f = torch.cos(((t_vals / diffusion_steps + s) / (1.0 + s)) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = torch.clamp(1.0 - (alpha_bar[1:] / alpha_bar[:-1]), min=1e-8, max=0.999)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1.0 - alpha_cumprod))
        
        # Normalization parameters (min-max scaling)
        self._setup_normalization(qpos_min, qpos_max, mom_min, mom_max, torque_min, torque_max)
        
        # EMA for better sampling
        if use_ema:
            self.ema = EMA(self.model, decay=0.9995)
        else:
            self.ema = None
        # Track whether EMA shadow was restored from checkpoint
        self._ema_loaded = False
    
    def _get_temporal_encoding(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Generate absolute sinusoidal temporal position encoding (length-independent).
        
        Timestep t always gets the same encoding regardless of total sequence length T.
        This is crucial for trajectory extension: timestep 500 has identical encoding
        whether T=1000 or T=1500.
        
        Uses standard Transformer-style sinusoidal encoding:
            PE(t, 2i)   = sin(t / 10000^(2i/d))
            PE(t, 2i+1) = cos(t / 10000^(2i/d))
        
        Args:
            T: sequence length (number of timesteps)
            device: torch device
        
        Returns:
            temporal_enc: [T, temporal_encoding_channels] - position encodings
        """
        d = self.temporal_encoding_channels
        positions = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)  # [T, 1]
        
        # Number of sin/cos pairs (use floor division for odd d)
        num_pairs = d // 2
        dim_indices = torch.arange(0, num_pairs, device=device, dtype=torch.float32)  # [num_pairs]
        freqs = 1.0 / (10000.0 ** (2.0 * dim_indices / d))  # [num_pairs]
        
        # Compute sin/cos encodings
        angles = positions * freqs  # [T, num_pairs]
        temporal_enc = torch.zeros(T, d, device=device)
        temporal_enc[:, 0:num_pairs] = torch.sin(angles)
        temporal_enc[:, num_pairs:2*num_pairs] = torch.cos(angles)
        # Any remaining channels (if d is odd) stay as zeros (padding)
        
        return temporal_enc
    
    def _setup_normalization(self, qpos_min, qpos_max, mom_min, mom_max, torque_min, torque_max):
        """
        Setup min-max normalization parameters to scale all components to [-1, 1].
        
        Normalization stats are per-dimension (global across time) with shape [dim].
        Separate normalization for state (qpos, mom) and conditioning (torque).
        """
        # Default to no normalization if not provided
        if qpos_min is None:
            qpos_min = torch.zeros(self.qpos_dim) - 1
            qpos_max = torch.ones(self.qpos_dim)
        if mom_min is None:
            mom_min = torch.zeros(self.mom_dim) - 1
            mom_max = torch.ones(self.mom_dim)
        if torque_min is None:
            torque_min = torch.zeros(self.torque_dim) - 1
            torque_max = torch.ones(self.torque_dim)
        
        # State normalization:
        # - conditional mode: [qpos | mom]
        # - unconditional mode: [qpos | mom | torque]
        if self.unconditional_tau_in_state:
            state_min = torch.cat([qpos_min, mom_min, torque_min], dim=-1)
            state_max = torch.cat([qpos_max, mom_max, torque_max], dim=-1)
        else:
            state_min = torch.cat([qpos_min, mom_min], dim=-1)
            state_max = torch.cat([qpos_max, mom_max], dim=-1)
        
        # Ensure minimum range for stability
        small_range_mask = (state_max - state_min) < self.range_epsilon
        state_max = torch.where(small_range_mask, state_min + self.range_epsilon, state_max)
        state_range = state_max - state_min
        
        self.register_buffer('state_min', state_min.float())
        self.register_buffer('state_max', state_max.float())
        self.register_buffer('state_range', state_range.float())
        
        # Conditioning normalization: [torque]
        cond_min = torque_min
        cond_max = torque_max
        small_range_mask_cond = (cond_max - cond_min) < self.range_epsilon
        cond_max = torch.where(small_range_mask_cond, cond_min + self.range_epsilon, cond_max)
        cond_range = cond_max - cond_min
        
        self.register_buffer('cond_min', cond_min.float())
        self.register_buffer('cond_max', cond_max.float())
        self.register_buffer('cond_range', cond_range.float())
    
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize state to [-1, 1] range using min-max scaling.
        
        Args:
            state: [B, T, state_dim] trajectories where state_dim = qpos_dim + mom_dim
        
        Returns:
            normalized_state: [B, T, state_dim] with all components in [-1, 1]
        """
        return (state - self.state_min) / self.state_range * 2.0 - 1.0
    
    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state from [-1, 1] back to original scale.
        
        Args:
            state: [B, T, state_dim] normalized trajectories with all components in [-1, 1]
        
        Returns:
            denormalized_state: [B, T, state_dim] in original scale
        """
        return (state + 1.0) / 2.0 * self.state_range + self.state_min
    
    def normalize_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Normalize conditioning (torque) to [-1, 1]."""
        return (cond - self.cond_min) / self.cond_range * 2.0 - 1.0
    
    def denormalize_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Denormalize conditioning (torque) from [-1, 1] to original scale."""
        return (cond + 1.0) / 2.0 * self.cond_range + self.cond_min

    def _shift_torque_sequence(self, torque: torch.Tensor) -> torch.Tensor:
        """
        Build shifted torque tokens for x_1=(s_1,0), x_t=(s_t,tau_{t-1}) for t>=2.
        """
        if torque.ndim != 3:
            raise ValueError(f"torque must have shape [B,T,D], got {tuple(torque.shape)}")
        zero = torch.zeros_like(torque[:, :1, :])
        return torch.cat([zero, torque[:, :-1, :]], dim=1)
    
    def configure_optimizers(self):
        adamw_kwargs = dict(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
        )
        optimizer = None
        if self.use_fused_adamw and torch.cuda.is_available():
            try:
                optimizer = torch.optim.AdamW(**adamw_kwargs, fused=True)
                print("[Perf] Using fused AdamW optimizer.")
            except TypeError:
                print("[Perf] fused AdamW unsupported in this build; using standard AdamW.")
        if optimizer is None:
            optimizer = torch.optim.AdamW(**adamw_kwargs)
        
        # Total steps for scheduling
        total_steps = self.trainer.estimated_stepping_batches
        
        # Warmup + Cosine Annealing for stability.
        # Allow explicit warmup override from CLI for quick tuning.
        if self.warmup_steps > 0:
            warmup_steps = min(int(self.warmup_steps), int(total_steps))
        else:
            warmup_steps = min(1000, total_steps // 10)  # 10% warmup, max 1000 steps
        
        min_lr_ratio = 0.1  # Never go below 10% of initial LR

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, warmup_steps))
            else:
                # Cosine decay after warmup (with floor)
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return max(min_lr_ratio, cosine)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
    
    def build_tokens(
        self, 
        state: torch.Tensor,
        diffusion_t: int, 
        skip_normalize: bool = False,
        torque: Optional[torch.Tensor] = None,
        include_torque: bool = False,
        time_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build tokens from state with explicit temporal position encoding.
        
        PerceiverIO mode:
            [state | diffusion_enc | temporal_enc]
        Transformer baseline mode (include_torque=True):
            [state | torque | diffusion_enc | temporal_enc]
        
        Args:
            state: [B, T, state_dim] - batch of state trajectories (qpos, mom)
            diffusion_t: diffusion timestep (1 to diffusion_steps)
            skip_normalize: if True, assumes inputs are already in normalized space
            torque: [B, T, torque_dim] normalized torque
            include_torque: include torque after state slice
            time_indices: optional absolute timestep indices [T] used for temporal encoding

        Returns:
            tokens: [B, T, C_in]
        """
        B, T, _ = state.shape
        device = state.device
        
        # Normalize unless already in normalized space
        normalized_state = state if skip_normalize else self.normalize_state(state)
        
        # Diffusion timestep encoding (same for all timesteps)
        # Cached table: [diffusion_steps, diff_enc_dim] -> take one row and expand.
        diffusion_vec = self.diffusion_encoding_table[diffusion_t - 1].to(device=device, dtype=normalized_state.dtype)
        diffusion_enc = diffusion_vec.view(1, 1, -1).expand(B, T, -1)  # [B, T, diff_enc_dim]
        
        # Temporal position encoding: absolute sinusoidal (LENGTH-INDEPENDENT)
        # Timestep t always gets the same encoding regardless of total T
        # This is crucial for trajectory extension without performance degradation
        if time_indices is None:
            time_indices = torch.arange(T, device=device, dtype=torch.long)
        else:
            time_indices = time_indices.to(device=device, dtype=torch.long)
            if time_indices.ndim != 1 or time_indices.shape[0] != T:
                raise ValueError(
                    f"time_indices must have shape ({T},), got {tuple(time_indices.shape)}"
                )

        max_time_index = int(time_indices.max().item()) if time_indices.numel() > 0 else -1
        if max_time_index < self.temporal_encoding_table.shape[0]:
            temporal_enc = self.temporal_encoding_table.index_select(0, time_indices).to(
                device=device,
                dtype=normalized_state.dtype,
            )  # [T, temp_enc_dim]
        else:
            # Fallback: build on the fly for longer absolute horizons.
            temporal_full = self._get_temporal_encoding(max_time_index + 1, device).to(dtype=normalized_state.dtype)
            temporal_enc = temporal_full.index_select(0, time_indices)
        temporal_enc = temporal_enc.unsqueeze(0).expand(B, -1, -1)  # [B, T, temp_enc_dim]

        if include_torque:
            if torque is None:
                raise ValueError("torque must be provided when include_torque=True")
            tokens = torch.cat([normalized_state, torque, diffusion_enc, temporal_enc], dim=-1)
        else:
            tokens = torch.cat([normalized_state, diffusion_enc, temporal_enc], dim=-1)
        
        return tokens
    
    def apply_noise(self, tokens: torch.Tensor, diffusion_t: int, return_noise: bool = False):
        """
        Apply noise to the state part of tokens only.
        
        Token structure: [state | diffusion_enc | temporal_enc]
        Only state (positions 0 to state_dim) gets noised.
        
        Args:
            tokens: [B, N, C_in] - tokens
            diffusion_t: diffusion timestep
            return_noise: whether to return the noise
        """
        # Extract state slice (at the beginning of token structure)
        state_start = 0
        state_end = self.state_dim
        
        state_slice = tokens[:, :, state_start:state_end]
        noise = torch.randn_like(state_slice)
        
        # Apply noise: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        noisy_state = (
            self.sqrt_alpha_cumprod[diffusion_t - 1] * state_slice +
            self.sqrt_one_minus_alpha_cumprod[diffusion_t - 1] * noise
        )
        
        # Replace state slice with noisy version (in-place: tokens are freshly built per step)
        tokens[:, :, state_start:state_end] = noisy_state
        
        if return_noise:
            return tokens, noise
        return tokens

    def _sample_subset_indices(self, seq_len: int, subset_len: int, device: torch.device) -> torch.Tensor:
        """Uniform random subset of timesteps, returned sorted for stable batching."""
        subset_len = max(1, min(int(subset_len), int(seq_len)))
        idx = torch.randperm(seq_len, device=device)[:subset_len]
        return torch.sort(idx).values

    def _build_future_context_views(
        self,
        state: torch.Tensor,
        torque: Optional[torch.Tensor],
        diffusion_t: int,
        time_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, int, int]:
        B, T, _ = state.shape
        clean_tokens = self.build_tokens(state, diffusion_t, time_indices=time_indices)
        noisy_tokens, noise = self.apply_noise(clean_tokens.clone(), diffusion_t, return_noise=True)

        cut_t = torch.randint(1, T, (1,), device=state.device).item()
        query_idx = torch.arange(cut_t, T, device=state.device, dtype=torch.long)
        history_idx = torch.arange(0, cut_t, device=state.device, dtype=torch.long)

        num_future_context = torch.randint(1, len(query_idx) + 1, (1,), device=state.device).item()
        future_subset_local = self._sample_subset_indices(len(query_idx), num_future_context, state.device)
        future_context_idx = query_idx.index_select(0, future_subset_local)
        context_idx = torch.cat([history_idx, future_context_idx], dim=0)

        if self.training_context_mode == "shifted_future_context_cleanprefix":
            history_context = clean_tokens.index_select(dim=1, index=history_idx)
            future_context = noisy_tokens.index_select(dim=1, index=future_context_idx)
            noisy_contexts = torch.cat([history_context, future_context], dim=1)
        else:
            noisy_contexts = noisy_tokens.index_select(dim=1, index=context_idx)
        noisy_queries = noisy_tokens.index_select(dim=1, index=query_idx)
        noise_target = noise.index_select(dim=1, index=query_idx)
        cond = torque.index_select(dim=1, index=query_idx) if torque is not None else None
        return noisy_contexts, noisy_queries, cond, noise_target, int(context_idx.numel()), int(query_idx.numel())

    def _build_shifted_future_context_views(
        self,
        state: torch.Tensor,
        torque: Optional[torch.Tensor],
        diffusion_t: int,
        time_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, int, int]:
        """
        Shifted continuation mode:
        - x_1 = (s_1, 0)
        - x_t = (s_t, tau_{t-1}) for t >= 2

        Sample a current-state index c in [0, T-2].
        Context covers x_1..x_{c+1} (history through the current observed state),
        query covers x_{c+2}..x_T so the first query token is (s_{t+1}, tau_t).
        """
        B, T, _ = state.shape
        if T < 2:
            raise ValueError("shifted_future_context requires sequence length at least 2")

        tokens = self.build_tokens(state, diffusion_t, time_indices=time_indices)
        noisy_tokens, noise = self.apply_noise(tokens, diffusion_t, return_noise=True)

        current_idx = torch.randint(0, T - 1, (1,), device=state.device).item()
        history_idx = torch.arange(0, current_idx + 1, device=state.device, dtype=torch.long)
        query_idx = torch.arange(current_idx + 1, T, device=state.device, dtype=torch.long)

        num_future_context = torch.randint(1, len(query_idx) + 1, (1,), device=state.device).item()
        future_subset_local = self._sample_subset_indices(len(query_idx), num_future_context, state.device)
        future_context_idx = query_idx.index_select(0, future_subset_local)
        context_idx = torch.cat([history_idx, future_context_idx], dim=0)

        noisy_contexts = noisy_tokens.index_select(dim=1, index=context_idx)
        noisy_queries = noisy_tokens.index_select(dim=1, index=query_idx)
        noise_target = noise.index_select(dim=1, index=query_idx)
        cond = torque.index_select(dim=1, index=query_idx) if torque is not None else None
        return noisy_contexts, noisy_queries, cond, noise_target, int(context_idx.numel()), int(query_idx.numel())

    def _build_perceiver_train_views(
        self,
        state: torch.Tensor,
        torque: Optional[torch.Tensor],
        diffusion_t: int,
        time_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, int, int]:
        """
        Build Perceiver context/query tensors for training or validation.

        Returns:
            contexts, queries, cond, noise_target, num_context, num_query
        """
        B, T, _ = state.shape
        if time_indices is None:
            time_indices = torch.arange(T, device=state.device, dtype=torch.long)

        tokens = self.build_tokens(state, diffusion_t, time_indices=time_indices)
        noisy_tokens, noise = self.apply_noise(tokens, diffusion_t, return_noise=True)
        if self.training_context_mode == "future_context":
            return self._build_future_context_views(state, torque, diffusion_t, time_indices)
        if self.training_context_mode in {"shifted_future_context", "shifted_future_context_cleanprefix"}:
            return self._build_shifted_future_context_views(state, torque, diffusion_t, time_indices)

        num_context = torch.randint(1, T + 1, (1,), device=state.device).item()
        num_query = torch.randint(1, T + 1, (1,), device=state.device).item()
        context_idx = self._sample_subset_indices(T, num_context, state.device)
        query_idx = self._sample_subset_indices(T, num_query, state.device)

        noisy_contexts = noisy_tokens.index_select(dim=1, index=context_idx)
        noisy_queries = noisy_tokens.index_select(dim=1, index=query_idx)
        noise_target = noise.index_select(dim=1, index=query_idx)
        cond = torque.index_select(dim=1, index=query_idx) if torque is not None else None
        return noisy_contexts, noisy_queries, cond, noise_target, num_context, num_query

    def _get_query_loss_weights(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if self.query_loss_decay == "none" or num_tokens == 1:
            return torch.ones((num_tokens,), device=device, dtype=torch.float32)

        strength = self.query_loss_decay_strength
        steps = torch.arange(num_tokens, device=device, dtype=torch.float32)
        denom = max(float(num_tokens - 1), 1.0)

        if self.query_loss_decay == "linear":
            end_weight = 1.0 / (1.0 + strength)
            weights = torch.linspace(1.0, end_weight, num_tokens, device=device, dtype=torch.float32)
        elif self.query_loss_decay == "exp":
            weights = torch.exp(-strength * (steps / denom))
        elif self.query_loss_decay == "power":
            weights = torch.pow(steps + 1.0, -strength)
        else:
            raise RuntimeError(f"Unhandled query_loss_decay: {self.query_loss_decay}")

        return weights / weights.mean().clamp_min(1e-8)

    def _compute_denoise_loss(
        self,
        predictions: torch.Tensor,
        noise_target: torch.Tensor,
        tau_loss_masked: bool,
    ) -> torch.Tensor:
        if tau_loss_masked:
            pq_dim = self.qpos_dim + self.mom_dim
            predictions = predictions[:, :, :pq_dim]
            noise_target = noise_target[:, :, :pq_dim]

        if self.query_loss_decay == "none":
            return F.mse_loss(predictions, noise_target)

        sq_error = (predictions - noise_target) ** 2
        token_mse = sq_error.mean(dim=-1)
        weights = self._get_query_loss_weights(token_mse.shape[1], token_mse.device)
        return (token_mse * weights.unsqueeze(0)).mean()

    def _shifted_token_torque_to_rollout_torque(self, shifted_token_torque: torch.Tensor) -> torch.Tensor:
        """
        Convert shifted torque tokens x_t=(s_t, tau_{t-1}) into rollout-aligned torque.

        For transitions state[:, i] -> state[:, i+1], the correct control lives in the
        next token, so rollout_torque[:, i] = shifted_token_torque[:, i+1].
        """
        if shifted_token_torque.ndim != 3:
            raise ValueError(
                f"shifted_token_torque must have shape [B,T,{self.torque_dim}], got {tuple(shifted_token_torque.shape)}"
            )
        if shifted_token_torque.shape[-1] != self.torque_dim:
            raise ValueError(
                f"shifted_token_torque last dim must equal torque_dim={self.torque_dim}, got {shifted_token_torque.shape[-1]}"
            )
        if shifted_token_torque.shape[1] == 0:
            return shifted_token_torque
        return torch.cat([shifted_token_torque[:, 1:, :], shifted_token_torque[:, -1:, :]], dim=1)

    def sample_training_pattern_trajectory_from_batch(
        self,
        qpos: torch.Tensor,
        mom: torch.Tensor,
        torque: torch.Tensor,
        num_diffusion_steps: int = 100,
        sampler: str = "ddim",
        use_ema: bool = True,
        resample_context_every_step: bool = True,
        sampling_torque_policy: str = "mixed_reacher",
        sampling_torque_mix: str = "sinusoidal:0.32,lpf_uniform:0.25,drift_ou:0.43",
        sampling_lpf_uniform_beta: float = 0.9992,
        sampling_torque_scale: float = 0.35,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Sample a trajectory using the same context/query pattern as training when possible.

        Returns rollout-aligned torque for visualization/reconstruction plus metadata
        describing the sampled prefix/query split.
        """
        device = self.device
        qpos = qpos.to(device=device, dtype=torch.float32)
        mom = mom.to(device=device, dtype=torch.float32)
        torque = torque.to(device=device, dtype=torch.float32)
        if qpos.ndim != 3 or mom.ndim != 3 or torque.ndim != 3:
            raise ValueError("qpos, mom, and torque must have shape [B,T,D].")
        if qpos.shape[:2] != mom.shape[:2] or qpos.shape[:2] != torque.shape[:2]:
            raise ValueError(
                f"Mismatched batch/time dims: qpos={tuple(qpos.shape)}, mom={tuple(mom.shape)}, torque={tuple(torque.shape)}"
            )

        horizon = min(int(qpos.shape[1]), int(self.max_timesteps))
        qpos = qpos[:, :horizon, :]
        mom = mom[:, :horizon, :]
        torque = torque[:, :horizon, :]
        batch_size = int(qpos.shape[0])

        if horizon < 2:
            raise RuntimeError(f"Need at least 2 timesteps for training-pattern sampling, got horizon={horizon}.")

        metadata = {
            "training_context_mode": self.training_context_mode,
            "horizon": int(horizon),
        }

        if self.training_context_mode in {"shifted_future_context", "shifted_future_context_cleanprefix"} and self.unconditional_tau_in_state:
            shifted_tau = self._shift_torque_sequence(torque)
            full_state = torch.cat([qpos, mom, shifted_tau], dim=-1)

            prefix_len = int(np.random.randint(1, horizon))
            query_length = int(horizon - prefix_len)
            num_future_context = int(np.random.randint(1, query_length + 1))
            context_fraction = float(num_future_context) / float(query_length)
            prefix_state = full_state[:, :prefix_len, :]

            print(
                f"[Sample] Using shifted training-pattern sampler with prefix_len={prefix_len}, "
                f"query_length={query_length}, future_context={num_future_context}/{query_length}",
                flush=True,
            )
            state, torque_tokens = self.sample_shifted_suffix_trajectory(
                prefix_state=prefix_state,
                query_length=query_length,
                num_diffusion_steps=num_diffusion_steps,
                context_fraction=context_fraction,
                use_ema=use_ema,
                sampler=sampler,
                resample_context_every_step=resample_context_every_step,
            )
            rollout_torque = self._shifted_token_torque_to_rollout_torque(torque_tokens)
            metadata.update(
                {
                    "mode": self.training_context_mode,
                    "prefix_len": prefix_len,
                    "query_length": query_length,
                    "num_future_context": num_future_context,
                    "context_fraction": context_fraction,
                    "batch_size": batch_size,
                }
            )
            return state, rollout_torque, metadata

        if self.training_context_mode == "future_context" and self.unconditional_tau_in_state:
            full_state = torch.cat([qpos, mom, torque], dim=-1)
            cut_t = int(np.random.randint(1, horizon))
            query_length = int(horizon - cut_t)
            num_future_context = int(np.random.randint(1, query_length + 1))
            context_fraction = float(num_future_context) / float(query_length)
            known_mask = torch.zeros_like(full_state, dtype=torch.bool)
            known_mask[:, :cut_t, :] = True

            print(
                f"[Sample] Using future_context training-pattern sampler with prefix_len={cut_t}, "
                f"query_length={query_length}, future_context={num_future_context}/{query_length}",
                flush=True,
            )
            state, rollout_torque = self.sample_inpainted_trajectory(
                known_state=full_state,
                known_mask=known_mask,
                current_index=cut_t,
                num_diffusion_steps=num_diffusion_steps,
                context_fraction=context_fraction,
                use_ema=use_ema,
                sampler=sampler,
                target_guidance_w=0.0,
                query_mode="suffix",
                resample_context_every_step=resample_context_every_step,
            )
            metadata.update(
                {
                    "mode": "future_context",
                    "prefix_len": cut_t,
                    "query_length": query_length,
                    "num_future_context": num_future_context,
                    "context_fraction": context_fraction,
                    "batch_size": batch_size,
                }
            )
            return state, rollout_torque, metadata

        print(
            "[Sample] Falling back to generic sample_trajectories() path "
            f"(training_context_mode={self.training_context_mode}, "
            f"unconditional_tau_in_state={self.unconditional_tau_in_state})",
            flush=True,
        )
        state, rollout_torque = self.sample_trajectories(
            num_samples=batch_size,
            trajectory_length=horizon,
            num_diffusion_steps=num_diffusion_steps,
            context_fraction=0.5,
            use_ema=use_ema,
            sampler=sampler,
            sampling_torque_policy=sampling_torque_policy,
            sampling_torque_mix=sampling_torque_mix,
            sampling_lpf_uniform_beta=sampling_lpf_uniform_beta,
            sampling_torque_scale=sampling_torque_scale,
        )
        metadata.update(
            {
                "mode": "generic_full_trajectory",
                "prefix_len": None,
                "query_length": int(horizon),
                "num_future_context": None,
                "context_fraction": 0.5,
                "batch_size": batch_size,
            }
        )
        return state, rollout_torque, metadata

    def _apply_shifted_hnn_guidance(
        self,
        prefix_phys: torch.Tensor,
        query_phys: torch.Tensor,
        hnn: nn.Module,
        guidance_method: str,
        guidance_energy_mode: str,
        alpha_q: float,
        alpha_p: float,
        alpha_tau: float,
        guidance_trust_lambda: float,
        guidance_normalize_grad: bool,
        guidance_joint_update: bool,
        guidance_hamres_smooth_sigma: float,
        guidance_hamres_delta: float,
        guidance_hamres_min_scale_q: float,
        guidance_hamres_min_scale_p: float,
        freeze_first_query_token: bool,
        dt: Optional[float],
    ) -> torch.Tensor:
        from src.models.utils import compute_hnn_guidance_energy

        if guidance_method != "strategy2":
            raise ValueError(
                "sample_shifted_suffix_trajectory currently supports HNN guidance_method='strategy2' only."
            )
        if query_phys.shape[1] == 0:
            return query_phys

        rollout_dt = float(self.data_dt) if dt is None else float(dt)
        q_ref = query_phys[:, :, :self.qpos_dim].detach()
        p_ref = query_phys[:, :, self.qpos_dim:self.qpos_dim + self.mom_dim].detach()
        tau_ref = query_phys[:, :, self.qpos_dim + self.mom_dim:self.qpos_dim + self.mom_dim + self.torque_dim].detach()

        query_q = q_ref.clone().requires_grad_(True)
        query_p = p_ref.clone().requires_grad_(True)
        query_tau = tau_ref.clone().requires_grad_(True)
        full_q = torch.cat([prefix_phys[:, :, :self.qpos_dim].detach(), query_q], dim=1)
        full_p = torch.cat(
            [prefix_phys[:, :, self.qpos_dim:self.qpos_dim + self.mom_dim].detach(), query_p],
            dim=1,
        )
        full_shifted_tau = torch.cat(
            [
                prefix_phys[:, :, self.qpos_dim + self.mom_dim:self.qpos_dim + self.mom_dim + self.torque_dim].detach(),
                query_tau,
            ],
            dim=1,
        )
        rollout_tau = self._shifted_token_torque_to_rollout_torque(full_shifted_tau)
        torque_gain = getattr(hnn, "torque_gain", None)
        if torque_gain is not None:
            gain = torch.as_tensor(torque_gain, device=rollout_tau.device, dtype=rollout_tau.dtype).flatten()
            if gain.numel() == 1:
                gain = gain.repeat(self.torque_dim)
            gain = gain[: self.torque_dim].view(1, 1, self.torque_dim)
            rollout_tau = rollout_tau * gain

        energy = compute_hnn_guidance_energy(
            full_q,
            full_p,
            rollout_tau,
            hnn,
            rollout_dt,
            mode=guidance_energy_mode,
            use_forward_diff=False,
            hamres_smooth_sigma=guidance_hamres_smooth_sigma,
            hamres_delta=guidance_hamres_delta,
            hamres_min_scale_q=guidance_hamres_min_scale_q,
            hamres_min_scale_p=guidance_hamres_min_scale_p,
        )
        if guidance_trust_lambda > 0.0:
            energy = energy + guidance_trust_lambda * (
                ((query_q - q_ref) ** 2).mean()
                + ((query_p - p_ref) ** 2).mean()
                + ((query_tau - tau_ref) ** 2).mean()
            )
        grad_q, grad_p, grad_tau = torch.autograd.grad(energy, [query_q, query_p, query_tau])

        eps = 1e-12
        if guidance_joint_update:
            if guidance_normalize_grad:
                grad_joint = torch.cat([grad_q, grad_p, grad_tau], dim=-1)
                norm_joint = grad_joint.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1).clamp_min(eps)
                grad_q_use = grad_q / norm_joint
                grad_p_use = grad_p / norm_joint
                grad_tau_use = grad_tau / norm_joint
            else:
                grad_p_use = grad_p
                grad_q_use = grad_q
                grad_tau_use = grad_tau
        else:
            if guidance_normalize_grad:
                norm_p = grad_p.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1).clamp_min(eps)
                norm_q = grad_q.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1).clamp_min(eps)
                norm_tau = grad_tau.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1).clamp_min(eps)
                grad_q_use = grad_q / norm_q
                grad_p_use = grad_p / norm_p
                grad_tau_use = grad_tau / norm_tau
            else:
                grad_p_use = grad_p
                grad_q_use = grad_q
                grad_tau_use = grad_tau

        guided_query = query_phys.detach().clone()
        guided_query[:, :, :self.qpos_dim] = (query_q - float(alpha_q) * grad_q_use).detach()
        guided_query[:, :, self.qpos_dim:self.qpos_dim + self.mom_dim] = (
            query_p - float(alpha_p) * grad_p_use
        ).detach()
        guided_query[:, :, self.qpos_dim + self.mom_dim:self.qpos_dim + self.mom_dim + self.torque_dim] = (
            query_tau - float(alpha_tau) * grad_tau_use
        ).detach()
        if freeze_first_query_token and guided_query.shape[1] > 0:
            # Keep overlap token fixed (after boundary guidance) while still letting
            # it influence token-1 updates through coupled HNN energy terms.
            guided_query[:, 0, : self.qpos_dim] = q_ref[:, 0, :]
            guided_query[:, 0, self.qpos_dim : self.qpos_dim + self.mom_dim] = p_ref[:, 0, :]
            guided_query[:, 0, self.qpos_dim + self.mom_dim : self.qpos_dim + self.mom_dim + self.torque_dim] = (
                tau_ref[:, 0, :]
            )
        return guided_query

    def training_step(self, batch, batch_idx):
        """Training step with prefix context and CFG dropout."""
        # batch is a dict with keys: 'seq_qpos', 'seq_mom', 'seq_torque'
        qpos = batch['seq_qpos']  # [B, T, qpos_dim]
        mom = batch['seq_mom']    # [B, T, mom_dim]
        torque = batch['seq_torque']  # [B, T, torque_dim]
        
        # State:
        # - conditional mode: [qpos | mom]
        # - unconditional mode: [qpos | mom | torque]
        if self.unconditional_tau_in_state:
            torque_tokens = (
                self._shift_torque_sequence(torque)
                if self.training_context_mode in {"shifted_future_context", "shifted_future_context_cleanprefix"}
                else torque
            )
            state = torch.cat([qpos, mom, torque_tokens], dim=-1)
        else:
            state = torch.cat([qpos, mom], dim=-1)  # [B, T, state_dim]
        
        B, T, _ = state.shape
        
        # Sample trajectory length uniformly from training options
        T_train = self.trajectory_length_training_options[
            torch.randint(0, len(self.trajectory_length_training_options), (1,)).item()
        ]
        T_train = min(T_train, T)  # Clamp to actual batch length
        
        # Random slice of T_train timesteps (data augmentation)
        max_start = T - T_train
        start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
        state = state[:, start:start + T_train, :]
        torque = torque[:, start:start + T_train, :]
        time_indices = torch.arange(start, start + T_train, device=state.device, dtype=torch.long)

        # Random diffusion timestep
        diffusion_t = torch.randint(1, self.diffusion_steps + 1, (1,)).item()
        
        # Sample context length uniformly from [1, T_train-1]
        num_context = torch.randint(1, T_train, (1,)).item()
        
        if self.unconditional_tau_in_state:
            if self.backbone == "transformer":
                tokens = self.build_tokens(state, diffusion_t)
                noisy_tokens, noise = self.apply_noise(tokens, diffusion_t, return_noise=True)
                predictions = self.model(noisy_tokens)
                noise_target = noise
                num_query = T_train
            else:
                noisy_contexts, noisy_queries, _, noise_target, num_context, num_query = self._build_perceiver_train_views(
                    state, torque=None, diffusion_t=diffusion_t, time_indices=time_indices
                )
                predictions = self.model(noisy_contexts, noisy_queries, torque=None)
        else:
            # Normalize torque for conditioning
            torque_norm = self.normalize_cond(torque)
            
            # ========== CFG DROPOUT (classifier-free guidance training) ==========
            # With probability p_uncond, train unconditionally by zeroing torque
            if torch.rand(1).item() < self.p_uncond:
                torque_norm = torch.zeros_like(torque_norm)

            if self.backbone == "transformer":
                tokens = self.build_tokens(
                    state, diffusion_t, torque=torque_norm, include_torque=True
                )
                noisy_tokens, noise = self.apply_noise(tokens, diffusion_t, return_noise=True)
                predictions = self.model(noisy_tokens)
                noise_target = noise
                num_query = T_train
            else:
                noisy_contexts, noisy_queries, torque_for_queries, noise_target, num_context, num_query = (
                    self._build_perceiver_train_views(
                        state,
                        torque=torque_norm,
                        diffusion_t=diffusion_t,
                        time_indices=time_indices,
                    )
                )
                predictions = self.model(noisy_contexts, noisy_queries, torque_for_queries)
            
        # Main loss: predict noise with optional stochastic tau masking in unconditional mode.
        tau_loss_masked = False
        if (
            self.unconditional_tau_in_state
            and self.tau_loss_mask_prob > 0.0
            and self.torque_dim > 0
        ):
            tau_loss_masked = bool(torch.rand(1, device=predictions.device).item() < self.tau_loss_mask_prob)

        loss_denoise = self._compute_denoise_loss(predictions, noise_target, tau_loss_masked)
        
        # ========== STABILITY: Skip NaN/Inf losses ==========
        if not torch.isfinite(loss_denoise):
            print(f"[WARNING] NaN/Inf loss detected at step {batch_idx}, skipping batch")
            return None  # PyTorch Lightning will skip this batch
        
        loss = loss_denoise
        
        # Log training loss (step-level only)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log('denoise_loss', loss_denoise, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        if self.unconditional_tau_in_state:
            self.log(
                'tau_loss_masked',
                float(tau_loss_masked),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        
        # Log trajectory length and context length for debugging
        self.log('T_train', float(T_train), prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log('num_context', float(num_context), prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log('num_query', float(num_query), prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with prefix context (always conditional, no CFG dropout)."""
        qpos = batch['seq_qpos']
        mom = batch['seq_mom']
        torque = batch['seq_torque']
        
        if self.unconditional_tau_in_state:
            torque_tokens = (
                self._shift_torque_sequence(torque)
                if self.training_context_mode in {"shifted_future_context", "shifted_future_context_cleanprefix"}
                else torque
            )
            state = torch.cat([qpos, mom, torque_tokens], dim=-1)
        else:
            # State: [qpos | mom], Conditioning: torque (passed separately for AdaLN)
            state = torch.cat([qpos, mom], dim=-1)
        
        B, T, _ = state.shape
        time_indices = torch.arange(T, device=state.device, dtype=torch.long)
        
        diffusion_t = torch.randint(1, self.diffusion_steps + 1, (1,)).item()
        
        if self.unconditional_tau_in_state:
            if self.backbone == "transformer":
                tokens = self.build_tokens(state, diffusion_t)
                noisy_tokens, noise = self.apply_noise(tokens, diffusion_t, return_noise=True)
                predictions = self.model(noisy_tokens)
                noise_target = noise
            else:
                noisy_contexts, noisy_queries, _, noise_target, _, _ = self._build_perceiver_train_views(
                    state, torque=None, diffusion_t=diffusion_t, time_indices=time_indices
                )
                predictions = self.model(noisy_contexts, noisy_queries, torque=None)
        else:
            # Normalize torque for conditioning
            torque_norm = self.normalize_cond(torque)

            if self.backbone == "transformer":
                tokens = self.build_tokens(
                    state, diffusion_t, torque=torque_norm, include_torque=True
                )
                noisy_tokens, noise = self.apply_noise(tokens, diffusion_t, return_noise=True)
                predictions = self.model(noisy_tokens)
                noise_target = noise
            else:
                noisy_contexts, noisy_queries, torque_for_queries, noise_target, _, _ = (
                    self._build_perceiver_train_views(
                        state,
                        torque=torque_norm,
                        diffusion_t=diffusion_t,
                        time_indices=time_indices,
                    )
                )
                predictions = self.model(noisy_contexts, noisy_queries, torque_for_queries)
        loss = self._compute_denoise_loss(predictions, noise_target, tau_loss_masked=False)
        
        # Log validation loss (epoch-level only)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    
    def on_save_checkpoint(self, checkpoint):
        """Persist EMA shadow so it survives resume and can be used for sampling."""
        if self.ema is not None and len(self.ema.shadow) > 0:
            checkpoint['ema_decay'] = self.ema.decay
            checkpoint['ema_shadow'] = {name: tensor.detach().cpu() for name, tensor in self.ema.shadow.items()}
    
    def on_load_checkpoint(self, checkpoint):
        """Restore EMA shadow from checkpoint if present."""
        ema_shadow = checkpoint.get('ema_shadow', None)
        ema_decay = checkpoint.get('ema_decay', 0.9995)
        if ema_shadow is not None:
            # Recreate EMA and load shadow to correct device/dtype later
            self.ema = EMA(self.model, decay=ema_decay)
            device = next(self.model.parameters()).device
            for name, tensor in ema_shadow.items():
                if name in self.ema.shadow:
                    self.ema.shadow[name] = tensor.to(device=device, dtype=self.ema.shadow[name].dtype)
            self._ema_loaded = True
        else:
            self._ema_loaded = False

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict that handles EMA model state gracefully.
        
        During training resume, PyTorch Lightning calls load_state_dict with strict=True,
        which fails if EMA state is missing or has changed structure.
        
        Solution: Use strict=False to match sampling behavior, which allows missing keys.
        This lets us load the model checkpoint even if EMA state doesn't match perfectly.
        """
        # For checkpoint loading during training (when EMA may not exist or has changed),
        # use strict=False to handle missing/mismatched keys gracefully
        if strict and any(key.startswith('ema.') for key in state_dict.keys()):
            # If checkpoint contains EMA state but we're loading with strict=True,
            # fall back to strict=False to avoid errors
            print("[TrajectoryDPF] Checkpoint contains EMA state. Loading with strict=False to handle mismatches.")
            strict = False
        
        return super().load_state_dict(state_dict, strict=strict)
    
    def on_fit_start(self):
        """Ensure EMA tensors live on same device/dtype as model before training/sampling."""
        if self.ema is not None:
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.ema.shadow:
                    self.ema.shadow[name] = self.ema.shadow[name].to(device=param.device, dtype=param.dtype)
    
    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms for monitoring training stability."""
        # Compute gradient norm across all parameters
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Log gradient norm (helps detect exploding gradients early)
        self.log('grad_norm', total_norm, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        
        # Warn if gradient norm is suspiciously high (pre-clipping value).
        # Threshold is configurable so we can reduce warning noise during stable clipping.
        if self.grad_warn_threshold > 0 and total_norm > self.grad_warn_threshold:
            print(f"[WARNING] High gradient norm: {total_norm:.2f} (pre-clip)")
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA after each training batch."""
        if self.ema is not None:
            self.ema.update(self.model)
    
    def _predict_x0(
        self, 
        x_t: torch.Tensor, 
        eps: torch.Tensor, 
        a_bar_t: torch.Tensor,
        dynamic_threshold: bool = True,
        percentile: float = 0.995,
        clamp_range: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict x0 from noisy state x_t and predicted noise eps, with stabilization.
        
        Uses dynamic thresholding (Imagen-style) to prevent CFG blow-up:
        1. Compute raw x0 prediction
        2. Find the percentile threshold of |x0| across all dimensions
        3. If threshold > clamp_range, scale x0 so the percentile lands at clamp_range
        4. Finally clamp to [-clamp_range, clamp_range]
        
        This keeps x0 in a reasonable range without hard clipping artifacts.
        
        Args:
            x_t: [B, T, state_dim] - noisy state
            eps: [B, T, state_dim] - predicted noise
            a_bar_t: scalar - cumulative alpha at timestep t
            dynamic_threshold: whether to use dynamic thresholding (vs hard clamp)
            percentile: percentile for dynamic threshold (0.995 = 99.5th percentile)
            clamp_range: target range for normalized data (typically 1.0 for [-1,1])
        
        Returns:
            x0: [B, T, state_dim] - stabilized x0 prediction
        """
        # Raw x0 prediction
        x0 = (x_t - torch.sqrt(1.0 - a_bar_t) * eps) / torch.sqrt(a_bar_t)
        
        if dynamic_threshold:
            # Compute percentile threshold per sample (flatten T and state_dim)
            B = x0.shape[0]
            x0_flat = x0.reshape(B, -1).abs()  # [B, T*state_dim]
            
            # Get the percentile value for each sample
            k = int(percentile * x0_flat.shape[1])
            k = max(1, min(k, x0_flat.shape[1] - 1))
            threshold = x0_flat.kthvalue(k, dim=1, keepdim=True).values  # [B, 1]
            
            # Scale down if threshold exceeds clamp_range
            threshold = torch.clamp(threshold, min=clamp_range)  # At least clamp_range
            scale = clamp_range / threshold  # [B, 1]
            scale = scale.unsqueeze(-1)  # [B, 1, 1] for broadcasting
            
            x0 = x0 * scale
        
        # Final hard clamp as safety net
        x0 = torch.clamp(x0, -clamp_range, clamp_range)
        
        return x0
    
    def _compute_legacy_sigma_t(
        self, a_bar_t: torch.Tensor, a_bar_prev: torch.Tensor, is_final_step: bool
    ) -> torch.Tensor:
        """Compute sigma_t for legacy DDPM sampler."""
        if is_final_step:
            return a_bar_t.new_tensor(0.0)
        eta = 1.0
        return eta * torch.sqrt(torch.clamp((1.0 - a_bar_prev) / (1.0 - a_bar_t), min=0.0, max=1.0)) * \
               torch.sqrt(torch.clamp(1.0 - (a_bar_t / a_bar_prev), min=0.0, max=1.0))
    
    @staticmethod
    def _legacy_generate_sinusoidal_torque(
        model,
        trajectory_length: int,
        dt: float,
        num_sin: int = 5,
        lim_amplitude: float = 0.5,
        lim_frequency: float = 6 * math.pi,
        lim_phase: float = 2 * math.pi,
    ) -> np.ndarray:
        """Legacy sinusoidal torque used by older sampling runs."""
        torque_dim = model.nu
        amplitudes = np.random.uniform(0, lim_amplitude, (torque_dim, num_sin, 1))
        frequencies = np.random.uniform(0, lim_frequency, (torque_dim, num_sin, 1))
        phases = np.random.uniform(0, lim_phase, (torque_dim, num_sin, 1))
        steps = np.arange(trajectory_length) * dt
        steps = steps[None, None, :]
        steps = np.tile(steps, (torque_dim, num_sin, 1))
        return np.sum(amplitudes * np.sin(frequencies * steps + phases), axis=1).T

    def _generate_random_torque(
        self,
        num_samples: int,
        trajectory_length: int,
        dt: float,
        sampling_torque_policy: str = "legacy_sinusoidal",
        sampling_torque_mix: str = "sinusoidal:0.32,lpf_uniform:0.25,drift_ou:0.43",
        sampling_lpf_uniform_beta: float = 0.9992,
        sampling_torque_scale: float = 0.35,
    ) -> torch.Tensor:
        """
        Generate torque for sampling.

        Policies:
        - legacy_sinusoidal: old in-file sinusoid generator (backward compatible)
        - sinusoidal/lpf_uniform/drift_ou: reuse dataset generators
        - mixed_reacher: per-trajectory mixture over sinusoidal/lpf_uniform/drift_ou
        """
        device = self.device
        model = mujoco.MjModel.from_xml_string(self.xml_content) if self.xml_content is not None else None
        if model is None:
            raise RuntimeError("XML content is required to generate torque from policy.")
        model.opt.timestep = float(self.dt)
        skip_steps = max(1, int(round(float(dt) / float(self.dt))))

        policy = sampling_torque_policy.strip()
        if policy == "mixed_reacher":
            mix_names, mix_weights = parse_torque_policies(sampling_torque_mix)
            valid = {"sinusoidal", "lpf_uniform", "drift_ou"}
            if any(name not in valid for name in mix_names):
                raise ValueError(
                    f"sampling_torque_mix contains unsupported policies {mix_names}. "
                    f"Only {sorted(valid)} are supported for mixed_reacher."
                )
            sampled = np.random.choice(mix_names, size=num_samples, p=mix_weights)
        else:
            sampled = [policy] * num_samples

        all_torques = []
        for p in sampled:
            if p == "legacy_sinusoidal":
                torque = self._legacy_generate_sinusoidal_torque(
                    model=model,
                    trajectory_length=trajectory_length,
                    dt=dt,
                )
            else:
                torque = generate_torque_sequence(
                    model=model,
                    num_steps=trajectory_length,
                    skip_steps=skip_steps,
                    policy=p,
                    ref_std_per_dim=None,
                    ctrl_margin=0.05,
                    lpf_uniform_beta=float(sampling_lpf_uniform_beta),
                )
                # Dataset generation scales controls globally after policy sampling.
                torque = float(sampling_torque_scale) * torque
            # Convert from simulation resolution to collected resolution.
            if torque.shape[0] == trajectory_length * skip_steps and skip_steps > 1:
                torque = torque[skip_steps - 1 :: skip_steps]
            all_torques.append(torque)

        torque_np = np.asarray(all_torques, dtype=np.float32)
        if torque_np.shape != (num_samples, trajectory_length, self.torque_dim):
            raise RuntimeError(
                f"Generated torque shape mismatch: got {torque_np.shape}, "
                f"expected {(num_samples, trajectory_length, self.torque_dim)}"
            )
        return torch.tensor(torque_np, dtype=torch.float32, device=device)

    def sample_inpainted_trajectory(
        self,
        known_state: torch.Tensor,
        known_mask: torch.Tensor,
        current_index: int,
        target_positions: Optional[torch.Tensor] = None,
        num_diffusion_steps: int = 100,
        context_fraction: float = 0.2,
        use_ema: bool = True,
        sampler: str = "ddim",
        target_guidance_w: float = 0.01,
        target_lambda_power: float = 4.0,
        reacher_link1: float = 0.1,
        reacher_link2: float = 0.11,
        initial_noise: Optional[torch.Tensor] = None,
        anchor_noise: Optional[torch.Tensor] = None,
        resample_context_every_step: bool = True,
        query_mode: str = "suffix",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a full trajectory while inpainting known state/control prefixes.

        This path is intended for receding-horizon control on the unconditional
        Perceiver DPF where tau is part of the denoised state. The caller
        provides:
        - known_state: clean physical state/control tuples [B, T, state_dim]
        - known_mask:  boolean mask over the same tensor indicating which
          channels are observed and should be overwritten during denoising
        - current_index: zero-based timestep whose state is observed but whose
          control may remain unknown (mask can encode that partial observation)

        ``query_mode='suffix'``:
        - query tokens are suffix-only and begin at the current control interval
          ``current_index`` so the first query token can generate ``tau_t``
          while its state channels remain anchored to the observed ``s_t``.
        - context tokens are the fully observed history prefix plus a subset of
          the suffix query tokens.

        ``query_mode='full'``:
        - query tokens span the full trajectory [0, T).
        - the executed history and current observed state are inpainted inside
          the query.
        - context contains the fully observed history slice from the query plus
          an optional subset of future query tokens; the partial current token
          is excluded from context.
        """
        if not self.unconditional_tau_in_state:
            raise NotImplementedError("sample_inpainted_trajectory currently requires unconditional_tau_in_state=True.")
        if self.backbone == "transformer":
            raise NotImplementedError("sample_inpainted_trajectory is implemented for the Perceiver backbone only.")
        if sampler not in {"ddim", "ddpm"}:
            raise ValueError("sample_inpainted_trajectory supports sampler in {'ddim', 'ddpm'}.")
        if query_mode not in {"suffix", "full"}:
            raise ValueError(f"query_mode must be one of {{'suffix', 'full'}}; got {query_mode!r}")

        self.model.eval()
        device = self.device

        if use_ema and self.ema is not None:
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.ema.shadow:
                    self.ema.shadow[name] = self.ema.shadow[name].to(device=param.device, dtype=param.dtype)
            self.ema.store(self.model)
            self.ema.copy_to(self.model)

        known_state = known_state.to(device=device, dtype=torch.float32)
        known_mask = known_mask.to(device=device, dtype=torch.bool)
        if known_state.ndim != 3 or known_state.shape != known_mask.shape:
            raise ValueError(
                f"known_state and known_mask must have the same [B,T,D] shape; "
                f"got {tuple(known_state.shape)} vs {tuple(known_mask.shape)}"
            )
        if known_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"known_state last dim must match state_dim={self.state_dim}; got {known_state.shape[-1]}"
            )

        batch_size, plan_horizon, _ = known_state.shape
        if not (0 <= int(current_index) < int(plan_horizon)):
            raise ValueError(f"current_index={current_index} must be in [0, {plan_horizon - 1}]")

        if target_positions is not None:
            target_positions = target_positions.to(device=device, dtype=known_state.dtype)
            if target_positions.shape != (batch_size, 2):
                raise ValueError(
                    f"target_positions must have shape {(batch_size, 2)}; got {tuple(target_positions.shape)}"
                )

        clean_norm_full = self.normalize_state(known_state)
        if query_mode == "suffix":
            query_clean = clean_norm_full[:, current_index:, :]
            query_known_mask = known_mask[:, current_index:, :]
            query_time_indices = torch.arange(current_index, plan_horizon, device=device, dtype=torch.long)
            context_history_len = current_index
            future_context_pool = torch.arange(query_clean.shape[1], device=device, dtype=torch.long)
        else:
            query_clean = clean_norm_full
            query_known_mask = known_mask
            query_time_indices = torch.arange(plan_horizon, device=device, dtype=torch.long)
            context_history_len = current_index
            if current_index + 1 < plan_horizon:
                future_context_pool = torch.arange(current_index + 1, plan_horizon, device=device, dtype=torch.long)
            else:
                future_context_pool = torch.empty(0, device=device, dtype=torch.long)
        query_len = query_clean.shape[1]

        if initial_noise is not None:
            x = initial_noise.to(device=device, dtype=query_clean.dtype)
            if x.shape != query_clean.shape:
                raise ValueError(
                    f"initial_noise must have shape {tuple(query_clean.shape)}; got {tuple(x.shape)}"
                )
        else:
            x = torch.randn_like(query_clean)

        if anchor_noise is None:
            anchor_noise = torch.randn_like(query_clean)
        else:
            anchor_noise = anchor_noise.to(device=device, dtype=query_clean.dtype)
            if anchor_noise.shape != query_clean.shape:
                raise ValueError(
                    f"anchor_noise must have shape {tuple(query_clean.shape)}; got {tuple(anchor_noise.shape)}"
                )

        ts = torch.linspace(
            self.diffusion_steps - 1,
            0,
            steps=int(num_diffusion_steps),
            device=device,
            dtype=torch.long,
        )

        cached_query_subset: Optional[torch.Tensor] = None

        def _overwrite_known(x_in: torch.Tensor, a_bar: torch.Tensor, stochastic: bool) -> torch.Tensor:
            noise = torch.randn_like(query_clean) if stochastic else anchor_noise
            noisy_known = (
                torch.sqrt(a_bar) * query_clean
                + torch.sqrt(torch.clamp(1.0 - a_bar, min=0.0)) * noise
            )
            return torch.where(query_known_mask, noisy_known, x_in)

        def _build_context_and_query(
            x_in: torch.Tensor,
            diffusion_t: int,
            _a_bar: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            nonlocal cached_query_subset
            queries = self.build_tokens(
                x_in,
                diffusion_t,
                skip_normalize=True,
                time_indices=query_time_indices,
            )
            context_parts = []
            if context_history_len > 0:
                context_parts.append(queries[:, :context_history_len, :])

            if future_context_pool.numel() > 0 and context_fraction > 0.0:
                if cached_query_subset is None or resample_context_every_step:
                    num_query_context = max(1, int(round(float(context_fraction) * float(future_context_pool.numel()))))
                    num_query_context = min(num_query_context, int(future_context_pool.numel()))
                    subset_local = self._sample_subset_indices(int(future_context_pool.numel()), num_query_context, device)
                    cached_query_subset = future_context_pool.index_select(dim=0, index=subset_local)
                query_subset = queries.index_select(dim=1, index=cached_query_subset)
                context_parts.append(query_subset)

            if context_parts:
                contexts = torch.cat(context_parts, dim=1)
            else:
                contexts = queries.new_zeros((queries.shape[0], 0, queries.shape[-1]))
            return contexts, queries

        def _apply_target_guidance(x0_norm: torch.Tensor, a_bar: torch.Tensor) -> torch.Tensor:
            future_start = 1 if query_mode == "suffix" else current_index + 1
            if target_positions is None or target_guidance_w <= 0.0 or future_start >= query_len:
                return x0_norm

            x0_norm = torch.where(query_known_mask, query_clean, x0_norm)
            x0_phys = self.denormalize_state(x0_norm)
            q = x0_phys[:, :, :self.qpos_dim].clone().requires_grad_(True)
            future_q = q[:, future_start:, :]
            if future_q.shape[1] == 0:
                return x0_norm

            q1 = future_q[:, :, 0]
            q2 = future_q[:, :, 1]
            ee_x = float(reacher_link1) * torch.cos(q1) + float(reacher_link2) * torch.cos(q1 + q2)
            ee_y = float(reacher_link1) * torch.sin(q1) + float(reacher_link2) * torch.sin(q1 + q2)
            ee = torch.stack([ee_x, ee_y], dim=-1)
            delta = ee - target_positions.unsqueeze(1)

            # Weight all future tokens equally for target guidance.
            gamma = torch.full(
                (1, future_q.shape[1], 1),
                1.0 / float(future_q.shape[1]),
                device=device,
                dtype=q.dtype,
            )

            loss = (gamma * torch.sum(delta * delta, dim=-1, keepdim=True)).sum(dim=1).mean()
            grad_q = torch.autograd.grad(loss, q)[0]
            # Use smaller guidance steps near the end of denoising for finer adjustment.
            step_size = float(target_guidance_w) * torch.sqrt(torch.clamp(1.0 - a_bar, min=1e-6)).item()
            q_new = q - step_size * grad_q

            x0_phys_new = x0_phys.detach().clone()
            x0_phys_new[:, :, :self.qpos_dim] = q_new.detach()
            x0_norm_new = self.normalize_state(x0_phys_new).detach()
            return torch.where(query_known_mask, query_clean, x0_norm_new)

        for i, t in enumerate(tqdm(ts, total=len(ts), desc="Inpaint Sampling")):
            t_int = int(t.item())
            a_bar_t = torch.clamp(self.alpha_cumprod[t_int], min=1e-6, max=1.0)

            # Re-anchor known channels before every denoising prediction.
            x = _overwrite_known(x, a_bar_t, stochastic=(sampler == "ddpm"))
            contexts, queries = _build_context_and_query(x, t_int + 1, a_bar_t)

            with torch.no_grad():
                eps = self.model(contexts, queries, torque=None)

            x0 = self._predict_x0(x, eps, a_bar_t)
            x0 = _apply_target_guidance(x0, a_bar_t)
            eps_guided = (x - torch.sqrt(a_bar_t) * x0) / torch.sqrt(torch.clamp(1.0 - a_bar_t, min=1e-6))

            if sampler == "ddim":
                if i < len(ts) - 1:
                    a_bar_prev = torch.clamp(self.alpha_cumprod[int(ts[i + 1].item())], min=1e-6, max=1.0)
                else:
                    a_bar_prev = a_bar_t.new_tensor(1.0)
                x = torch.sqrt(a_bar_prev) * x0 + torch.sqrt(torch.clamp(1.0 - a_bar_prev, min=0.0)) * eps_guided
            else:
                alpha_t = self.alphas[t_int]
                beta_t = self.betas[t_int]
                coef1 = 1.0 / torch.sqrt(alpha_t)
                coef2 = beta_t / torch.sqrt(torch.clamp(1.0 - a_bar_t, min=1e-6))
                mean = coef1 * (x - coef2 * eps_guided)
                if t_int > 0:
                    sigma_t = torch.sqrt(beta_t)
                    x = mean + sigma_t * torch.randn_like(x)
                else:
                    x = mean

        x = torch.where(query_known_mask, query_clean, x)
        query_state = self.denormalize_state(x)
        if query_mode == "suffix":
            state = known_state.detach().clone()
            state[:, current_index:, :] = query_state
        else:
            state = query_state
        torque = state[:, :, self.qpos_dim + self.mom_dim : self.qpos_dim + self.mom_dim + self.torque_dim]

        if use_ema and self.ema is not None:
            self.ema.restore(self.model)

        self.model.train()
        return state, torque

    def sample_shifted_suffix_trajectory(
        self,
        prefix_state: torch.Tensor,
        query_length: int,
        num_diffusion_steps: int = 100,
        context_fraction: float = 0.2,
        use_ema: bool = True,
        sampler: str = "ddim",
        initial_noise: Optional[torch.Tensor] = None,
        prefix_start_index: int = 0,
        query_time_index_offset: int = 0,
        resample_context_every_step: bool = True,
        hnn: Optional[nn.Module] = None,
        guidance_method: str = "strategy2",
        guidance_energy_mode: str = "one_step",
        alpha_q: float = 1e-4,
        alpha_p: float = 1e-4,
        alpha_tau: Optional[float] = None,
        guidance_normalize_grad: bool = True,
        guidance_joint_update: bool = False,
        guidance_hamres_smooth_sigma: float = 1.0,
        guidance_hamres_delta: float = 1.0,
        guidance_hamres_min_scale_q: float = 1e-3,
        guidance_hamres_min_scale_p: float = 1e-3,
        freeze_first_query_token_in_hnn_guidance: bool = False,
        guidance_trust_lambda: float = 0.0,
        target_positions: Optional[torch.Tensor] = None,
        target_guidance_w: float = 0.0,
        boundary_guidance_alpha: float = 0.0,
        target_lambda_power: float = 4.0,
        reacher_link1: float = 0.1,
        reacher_link2: float = 0.11,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a future suffix under the shifted tokenization:
        - x_1 = (s_1, 0)
        - x_t = (s_t, tau_{t-1}) for t >= 2

        The prefix contains observed history through the current state, so the first
        query token is (s_{t+1}, tau_t). This avoids partial-token inpainting.
        """
        if self.training_context_mode not in {"shifted_future_context", "shifted_future_context_cleanprefix"}:
            raise RuntimeError(
                "sample_shifted_suffix_trajectory requires a checkpoint trained with "
                "training_context_mode in {'shifted_future_context', 'shifted_future_context_cleanprefix'}."
            )
        if not self.unconditional_tau_in_state:
            raise NotImplementedError("sample_shifted_suffix_trajectory requires unconditional_tau_in_state=True.")
        if self.backbone == "transformer":
            raise NotImplementedError("sample_shifted_suffix_trajectory is implemented for the Perceiver backbone only.")
        if sampler not in {"ddim", "ddpm"}:
            raise ValueError("sample_shifted_suffix_trajectory supports sampler in {'ddim', 'ddpm'}.")
        if int(query_length) < 0:
            raise ValueError(f"query_length must be non-negative, got {query_length}")

        self.model.eval()
        device = self.device

        if use_ema and self.ema is not None:
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.ema.shadow:
                    self.ema.shadow[name] = self.ema.shadow[name].to(device=param.device, dtype=param.dtype)
            self.ema.store(self.model)
            self.ema.copy_to(self.model)

        prefix_state = prefix_state.to(device=device, dtype=torch.float32)
        if prefix_state.ndim != 3 or prefix_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"prefix_state must have shape [B,T,{self.state_dim}], got {tuple(prefix_state.shape)}"
            )

        batch_size, prefix_len, _ = prefix_state.shape
        if target_positions is not None:
            target_positions = target_positions.to(device=device, dtype=prefix_state.dtype)
            if target_positions.shape != (batch_size, 2):
                raise ValueError(
                    f"target_positions must have shape {(batch_size, 2)}; got {tuple(target_positions.shape)}"
                )
        if alpha_tau is None:
            alpha_tau = float(alpha_p)

        if query_length == 0:
            state = prefix_state.detach().clone()
            torque = state[:, :, self.qpos_dim + self.mom_dim : self.qpos_dim + self.mom_dim + self.torque_dim]
            if use_ema and self.ema is not None:
                self.ema.restore(self.model)
            self.model.train()
            return state, torque

        prefix_clean = self.normalize_state(prefix_state)
        prefix_anchor_noise = torch.randn_like(prefix_clean)
        query_time_indices = torch.arange(
            int(prefix_start_index) + prefix_len + int(query_time_index_offset),
            int(prefix_start_index) + prefix_len + int(query_time_index_offset) + int(query_length),
            device=device,
            dtype=torch.long,
        )
        prefix_time_indices = torch.arange(
            int(prefix_start_index),
            int(prefix_start_index) + prefix_len,
            device=device,
            dtype=torch.long,
        )

        if initial_noise is not None:
            x = initial_noise.to(device=device, dtype=prefix_clean.dtype)
            expected_shape = (batch_size, int(query_length), self.state_dim)
            if x.shape != expected_shape:
                raise ValueError(
                    f"initial_noise must have shape {expected_shape}, got {tuple(x.shape)}"
                )
        else:
            x = torch.randn(batch_size, int(query_length), self.state_dim, device=device, dtype=prefix_clean.dtype)

        ts = torch.linspace(
            self.diffusion_steps - 1,
            0,
            steps=int(num_diffusion_steps),
            device=device,
            dtype=torch.long,
        )
        future_context_pool = torch.arange(int(query_length), device=device, dtype=torch.long)
        cached_query_subset: Optional[torch.Tensor] = None

        def _build_context_and_query(
            x_in: torch.Tensor,
            diffusion_t: int,
            a_bar: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            nonlocal cached_query_subset

            if self.training_context_mode == "shifted_future_context_cleanprefix":
                prefix_tokens = self.build_tokens(
                    prefix_clean,
                    diffusion_t,
                    skip_normalize=True,
                    time_indices=prefix_time_indices,
                )
            else:
                prefix_noise = torch.randn_like(prefix_clean) if sampler == "ddpm" else prefix_anchor_noise
                noisy_prefix = (
                    torch.sqrt(a_bar) * prefix_clean
                    + torch.sqrt(torch.clamp(1.0 - a_bar, min=0.0)) * prefix_noise
                )
                prefix_tokens = self.build_tokens(
                    noisy_prefix,
                    diffusion_t,
                    skip_normalize=True,
                    time_indices=prefix_time_indices,
                )
            query_tokens = self.build_tokens(
                x_in,
                diffusion_t,
                skip_normalize=True,
                time_indices=query_time_indices,
            )

            context_parts = [prefix_tokens]
            if future_context_pool.numel() > 0 and context_fraction > 0.0:
                if cached_query_subset is None or resample_context_every_step:
                    num_query_context = max(1, int(round(float(context_fraction) * float(future_context_pool.numel()))))
                    num_query_context = min(num_query_context, int(future_context_pool.numel()))
                    subset_local = self._sample_subset_indices(int(future_context_pool.numel()), num_query_context, device)
                    cached_query_subset = future_context_pool.index_select(dim=0, index=subset_local)
                query_subset = query_tokens.index_select(dim=1, index=cached_query_subset)
                context_parts.append(query_subset)

            contexts = torch.cat(context_parts, dim=1)
            return contexts, query_tokens

        def _apply_target_guidance(x0_norm: torch.Tensor, a_bar: torch.Tensor) -> torch.Tensor:
            if target_positions is None or target_guidance_w <= 0.0 or int(query_length) <= 0:
                return x0_norm

            x0_phys = self.denormalize_state(x0_norm)
            q = x0_phys[:, :, :self.qpos_dim].clone().requires_grad_(True)
            if q.shape[1] == 0:
                return x0_norm

            q1 = q[:, :, 0]
            q2 = q[:, :, 1]
            ee_x = float(reacher_link1) * torch.cos(q1) + float(reacher_link2) * torch.cos(q1 + q2)
            ee_y = float(reacher_link1) * torch.sin(q1) + float(reacher_link2) * torch.sin(q1 + q2)
            ee = torch.stack([ee_x, ee_y], dim=-1)
            delta = ee - target_positions.unsqueeze(1)

            # Weight all future tokens equally for target guidance.
            gamma = torch.full(
                (1, q.shape[1], 1),
                1.0 / float(q.shape[1]),
                device=device,
                dtype=q.dtype,
            )

            loss = (gamma * torch.sum(delta * delta, dim=-1, keepdim=True)).sum(dim=1).mean()
            grad_q = torch.autograd.grad(loss, q)[0]
            # Use smaller guidance steps near the end of denoising for finer adjustment.
            step_size = float(target_guidance_w) * torch.sqrt(torch.clamp(1.0 - a_bar, min=1e-6)).item()
            q_new = q - step_size * grad_q

            x0_phys_new = x0_phys.detach().clone()
            x0_phys_new[:, :, :self.qpos_dim] = q_new.detach()
            return self.normalize_state(x0_phys_new).detach()

        def _apply_boundary_alignment_guidance(x0_norm: torch.Tensor, a_bar: torch.Tensor) -> torch.Tensor:
            if boundary_guidance_alpha <= 0.0 or int(query_length) <= 0:
                return x0_norm

            token_dim = min(
                int(self.qpos_dim + self.mom_dim + self.torque_dim),
                int(self.state_dim),
            )
            if token_dim <= 0:
                return x0_norm

            x0_phys = self.denormalize_state(x0_norm)
            if x0_phys.shape[1] == 0 or prefix_state.shape[1] == 0:
                return x0_norm

            x0_var = x0_phys.detach().clone().requires_grad_(True)
            boundary_target = prefix_state[:, -1, :token_dim].detach()
            boundary_pred = x0_var[:, 0, :token_dim]
            loss = torch.mean((boundary_pred - boundary_target) ** 2)
            grad = torch.autograd.grad(loss, x0_var)[0]
            step_size = float(boundary_guidance_alpha) * torch.sqrt(torch.clamp(1.0 - a_bar, min=1e-6)).item()
            x0_phys_new = x0_var - step_size * grad
            return self.normalize_state(x0_phys_new.detach()).detach()

        for i, t in enumerate(tqdm(ts, total=len(ts), desc="Shifted Suffix Sampling")):
            t_int = int(t.item())
            a_bar_t = torch.clamp(self.alpha_cumprod[t_int], min=1e-6, max=1.0)
            contexts, queries = _build_context_and_query(x, t_int + 1, a_bar_t)

            with torch.no_grad():
                eps = self.model(contexts, queries, torque=None)

            x0 = self._predict_x0(x, eps, a_bar_t)
            x0 = _apply_target_guidance(x0, a_bar_t)
            x0 = _apply_boundary_alignment_guidance(x0, a_bar_t)
            if hnn is not None:
                x0_phys = self.denormalize_state(x0)
                noise_step_scale = torch.sqrt(torch.clamp(1.0 - a_bar_t, min=1e-6)).item()
                x0_phys = self._apply_shifted_hnn_guidance(
                    prefix_phys=prefix_state,
                    query_phys=x0_phys,
                    hnn=hnn,
                    guidance_method=guidance_method,
                    guidance_energy_mode=guidance_energy_mode,
                    alpha_q=float(alpha_q) * noise_step_scale,
                    alpha_p=float(alpha_p) * noise_step_scale,
                    alpha_tau=float(alpha_tau) * noise_step_scale,
                    guidance_trust_lambda=guidance_trust_lambda,
                    guidance_normalize_grad=guidance_normalize_grad,
                    guidance_joint_update=guidance_joint_update,
                    guidance_hamres_smooth_sigma=guidance_hamres_smooth_sigma,
                    guidance_hamres_delta=guidance_hamres_delta,
                    guidance_hamres_min_scale_q=guidance_hamres_min_scale_q,
                    guidance_hamres_min_scale_p=guidance_hamres_min_scale_p,
                    freeze_first_query_token=freeze_first_query_token_in_hnn_guidance,
                    dt=dt,
                )
                x0 = self.normalize_state(x0_phys).detach()
            eps_guided = (x - torch.sqrt(a_bar_t) * x0) / torch.sqrt(torch.clamp(1.0 - a_bar_t, min=1e-6))

            if sampler == "ddim":
                if i < len(ts) - 1:
                    a_bar_prev = torch.clamp(self.alpha_cumprod[int(ts[i + 1].item())], min=1e-6, max=1.0)
                else:
                    a_bar_prev = a_bar_t.new_tensor(1.0)
                x = torch.sqrt(a_bar_prev) * x0 + torch.sqrt(torch.clamp(1.0 - a_bar_prev, min=0.0)) * eps_guided
            else:
                alpha_t = self.alphas[t_int]
                beta_t = self.betas[t_int]
                coef1 = 1.0 / torch.sqrt(alpha_t)
                coef2 = beta_t / torch.sqrt(torch.clamp(1.0 - a_bar_t, min=1e-6))
                mean = coef1 * (x - coef2 * eps_guided)
                if t_int > 0:
                    sigma_t = torch.sqrt(beta_t)
                    x = mean + sigma_t * torch.randn_like(x)
                else:
                    x = mean

        query_state = self.denormalize_state(x)
        state = torch.cat([prefix_state.detach().clone(), query_state], dim=1)
        torque = state[:, :, self.qpos_dim + self.mom_dim : self.qpos_dim + self.mom_dim + self.torque_dim]

        if use_ema and self.ema is not None:
            self.ema.restore(self.model)

        self.model.train()
        return state, torque

    def sample_trajectories(
        self,
        num_samples: int,
        trajectory_length: int,
        num_diffusion_steps: int = None,
        context_fraction: float = 0.7,
        use_ema: bool = True,
        resample_context_every_step: bool = True,
        sampler: str = "ddim",
        # CFG parameters
        guidance_scale: float = 1.0,  # CFG scale (1.0 = no CFG, >1.0 = stronger conditioning)
        # HNN guidance parameters
        hnn: nn.Module = None,
        guidance_method: str = "strategy2",
        langevin_step_size: float = 1e-5,
        langevin_noise_scale: float = 1e-6,
        chunk_length: int = 15,  # Chunk length for integration-based guidance
        use_forward_diff: bool = False,  # Use forward difference instead of central difference for HamRes
        guidance_energy_mode: str = "one_step",  # 'one_step' or 'robust_hamres'
        guidance_hamres_smooth_sigma: float = 1.0,
        guidance_hamres_delta: float = 1.0,
        guidance_hamres_min_scale_q: float = 1e-3,
        guidance_hamres_min_scale_p: float = 1e-3,
        guidance_trust_lambda: float = 0.0,
        dt: Optional[float] = None,
        # Torque generation parameters
        torque: torch.Tensor = None,  # Optional: provide torque directly
        initial_noise: torch.Tensor = None,  # Optional: fixed initial noise for reproducible sampling
        sampling_torque_policy: str = "legacy_sinusoidal",
        sampling_torque_mix: str = "sinusoidal:0.32,lpf_uniform:0.25,drift_ou:0.43",
        sampling_lpf_uniform_beta: float = 0.9992,
        sampling_torque_scale: float = 0.35,
        # Temporal smoothing
        smooth_sigma: float = 0.0,  # Gaussian smoothing sigma (0 = disabled, 1-3 recommended)
        smooth_guidance_only: bool = False,  # If True, smooth only for guidance input; output stays unsmoothed
        smooth_last_step_only: bool = False,  # If True, only smooth at the final diffusion step
        optimize_target: str = "both",  # 'both', 'q' (position only), or 'p' (momentum only)
        alpha_q: float = 1e-4,  # Normalized SGD step size for q
        alpha_p: float = 1e-4,  # Normalized SGD step size for p
        guidance_normalize_grad: bool = True,  # Strategy 2: normalize guidance gradients
        guidance_joint_update: bool = False,  # Strategy 2: share one norm across q/p
        guidance_num_candidates: int = 16,  # Strategy 1 particle count
        guidance_dynamic_steps_inv_sqrt: bool = False,  # steps_t ~ 1/sqrt(1-a_bar_t)
        guidance_dynamic_steps_max: int = 5,  # clamp upper bound for dynamic steps
        guidance_dynamic_steps_multiplier: int = 1,  # multiply per-step guidance steps
        target_guidance_k: float = 0.01,
        target_lambda_power: float = 4.0,
        target_x: Optional[float] = None,
        target_y: Optional[float] = None,
        reacher_link1: float = 0.1,
        reacher_link2: float = 0.11,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample trajectories using diffusion with classifier-free guidance.
        
        Generates (qpos, mom) trajectories conditioned on torque.
        Uses CFG: eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        
        Args:
            num_samples: number of trajectories to generate
            trajectory_length: length of each trajectory
            num_diffusion_steps: number of denoising steps
            context_fraction: fraction of timesteps to use as context
            use_ema: whether to use EMA weights
            resample_context_every_step: whether to resample context at each step
            sampler: sampling method ('ddpm', 'ddim', or 'ddpm_legacy')
            guidance_scale: CFG scale (1.0 = no CFG, >1.0 = stronger conditioning)
            hnn: HNN for physics-based guidance
            guidance_method: strategy selector
                - 'strategy1': sample 16 stochastic x_{t-1} candidates from x_t (eta=1),
                  score corresponding x0 candidates with robust HamRes, and sigmoid-sample one
                - 'strategy2': one-step normalized guidance at each sampling step
                - 'strategy12_hybrid': strategy1 candidate generation + one-step guidance on each
                  candidate x0 before robust-HamRes scoring and resampling
            guidance_normalize_grad: strategy2 only; normalize per-sample guidance gradients before step
            guidance_joint_update: strategy2 only; use one shared norm across q/p instead of separate norms
            guidance_num_candidates: particle count for strategy1
            dt: timestep used to parameterize random torque generation (seconds between torque samples).
                If None, defaults to self.data_dt (dataset control timestep).
            torque: optional pre-generated torque [num_samples, trajectory_length, torque_dim]
            initial_noise: optional fixed initial noise [num_samples, trajectory_length, state_dim]
                for reproducible sampling with different parameters (e.g., context_fraction)

        Returns:
            Tuple of:
                - state: [num_samples, trajectory_length, state_dim] (qpos, mom)
                - torque: [num_samples, trajectory_length, torque_dim]
        """
        from src.models.utils import (
            run_one_step_guidance_hnn,
            compute_hnn_robust_hamres_energy,
        )
        
        self.model.eval()
        if guidance_method == "strategy1_target" and hnn is None:
            raise ValueError("guidance_method='strategy1_target' requires an HNN for resampling.")
        
        # Use EMA weights if available
        if use_ema and self.ema is not None:
            print(f"[Sampling] Applying EMA weights for inference...")
            
            # CRITICAL: Ensure EMA shadow tensors are on the same device as the model
            # This is necessary because EMA is not an nn.Module and doesn't follow model.to(device)
            device = next(self.model.parameters()).device
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.ema.shadow:
                    self.ema.shadow[name] = self.ema.shadow[name].to(device=device, dtype=param.dtype)
            
            self.ema.store(self.model)
            self.ema.copy_to(self.model)
            print(f"[Sampling] ✓ EMA weights applied to model (device: {device})")
        elif use_ema and self.ema is None:
            print(f"[Sampling] WARNING: use_ema=True but no EMA available!")
        else:
            print(f"[Sampling] Using regular model weights (use_ema=False)")
        
        device = self.device
        
        # Generate or use provided torque conditioning unless tau is part of state.
        if self.unconditional_tau_in_state:
            torque = None
            cond = None
            cond_uncond = None
        else:
            if torque is None:
                # NOTE: use dataset control timestep by default so sampling torque matches training distribution
                torque_dt = float(self.data_dt) if dt is None else float(dt)
                print(
                    f"[Sampling] Generating torque sequences with policy={sampling_torque_policy}, "
                    f"mix={sampling_torque_mix}"
                )
                torque = self._generate_random_torque(
                    num_samples,
                    trajectory_length,
                    torque_dt,
                    sampling_torque_policy=sampling_torque_policy,
                    sampling_torque_mix=sampling_torque_mix,
                    sampling_lpf_uniform_beta=sampling_lpf_uniform_beta,
                    sampling_torque_scale=sampling_torque_scale,
                )
            else:
                torque = torque.to(device)
        
        # Start with pure noise for state (qpos, mom)
        if initial_noise is not None:
            x = initial_noise.to(device)
        else:
            x = torch.randn(num_samples, trajectory_length, self.state_dim, device=device)
        
        if not self.unconditional_tau_in_state:
            # Normalized conditioning (torque passed separately for AdaLN)
            cond = self.normalize_cond(torque)
            cond_uncond = torch.zeros_like(cond)  # Unconditional: zeros (for CFG)
        
        # Setup timesteps based on sampler
        if sampler == "ddpm":
            if num_diffusion_steps is None:
                num_diffusion_steps = self.diffusion_steps
            ts = torch.arange(self.diffusion_steps - 1, -1, -1, device=device, dtype=torch.long)
            print(f"[Sampling] Using DDPM sampler with {len(ts)} steps")
        elif sampler == "ddim":
            if num_diffusion_steps is None:
                num_diffusion_steps = 50
            ts = torch.linspace(self.diffusion_steps - 1, 0, steps=num_diffusion_steps, device=device, dtype=torch.long)
            print(f"[Sampling] Using DDIM sampler with {len(ts)} steps")
        elif sampler == "ddpm_legacy":
            if num_diffusion_steps is None:
                num_diffusion_steps = 50
            ts = torch.linspace(self.diffusion_steps - 1, 0, steps=num_diffusion_steps, device=device, dtype=torch.long)
            print(f"[Sampling] Using Legacy sampler with {len(ts)} steps")
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
        
        if self.unconditional_tau_in_state:
            print("[Sampling] Unconditional mode: tau is part of state; CFG/conditioning disabled.")
        else:
            print(f"[Sampling] CFG guidance_scale={guidance_scale}")
        
        if self.backbone != "transformer":
            # PREFIX context: use first num_context timesteps (not random)
            # IMPORTANT: Cap context length to training max to avoid OOD encoder behavior when extending
            max_context_train = int(self.max_timesteps * context_fraction)
            num_context = max(1, min(trajectory_length - 1, max_context_train))
            print(f"[Sampling] Context length: {num_context} (capped at {max_context_train} from training length {self.max_timesteps})")

        def _sample_reacher_targets(batch_size: int, dev: torch.device, dtype: torch.dtype) -> torch.Tensor:
            if target_x is not None and target_y is not None:
                tgt = torch.tensor([float(target_x), float(target_y)], device=dev, dtype=dtype)
                return tgt.unsqueeze(0).repeat(batch_size, 1)
            out = []
            while len(out) < batch_size:
                cand = np.random.uniform(low=-0.2, high=0.2, size=(batch_size * 2, 2))
                mask = np.linalg.norm(cand, axis=1) < 0.2
                for xy in cand[mask]:
                    out.append(xy)
                    if len(out) >= batch_size:
                        break
            return torch.tensor(np.asarray(out[:batch_size]), device=dev, dtype=dtype)

        target_positions = _sample_reacher_targets(num_samples, device, x.dtype)
        self._last_sampled_targets = target_positions.detach().cpu().numpy()
        t_idx = torch.linspace(0.0, 1.0, steps=trajectory_length, device=device, dtype=x.dtype)
        lambda_w = (t_idx + 1e-6) ** float(target_lambda_power)
        lambda_w = (lambda_w / lambda_w.sum()).view(1, trajectory_length, 1)

        def _apply_reacher_terminal_guidance(x_phys: torch.Tensor, step_size: float) -> torch.Tensor:
            if self.qpos_dim < 2:
                return x_phys
            q = x_phys[:, :, :self.qpos_dim].clone().requires_grad_(True)
            q1 = q[:, :, 0]
            q2 = q[:, :, 1]
            ee_x = float(reacher_link1) * torch.cos(q1) + float(reacher_link2) * torch.cos(q1 + q2)
            ee_y = float(reacher_link1) * torch.sin(q1) + float(reacher_link2) * torch.sin(q1 + q2)
            ee = torch.stack([ee_x, ee_y], dim=-1)
            delta = ee - target_positions.unsqueeze(1)
            sq_l2 = torch.sum(delta * delta, dim=-1, keepdim=True)
            loss = (lambda_w * sq_l2).sum(dim=1).mean()
            grad_q = torch.autograd.grad(loss, q)[0]
            q_new = q - float(step_size) * grad_q
            x_new = x_phys.detach().clone()
            x_new[:, :, :self.qpos_dim] = q_new.detach()
            return x_new

        # Strategy1-only cached views reused across diffusion steps.
        strategy1_num_cands = max(2, int(guidance_num_candidates))
        strategy1_cond_flat = None
        strategy1_cond_uncond_flat = None
        strategy1_tau_flat = None
        strategy1_uniform = None
        if hnn is not None and guidance_method in {"strategy1", "strategy1_target"} and (not self.unconditional_tau_in_state):
            strategy1_cond_flat = (
                cond.unsqueeze(1)
                .expand(-1, strategy1_num_cands, -1, -1)
                .reshape(num_samples * strategy1_num_cands, trajectory_length, -1)
            )
            strategy1_cond_uncond_flat = (
                cond_uncond.unsqueeze(1)
                .expand(-1, strategy1_num_cands, -1, -1)
                .reshape(num_samples * strategy1_num_cands, trajectory_length, -1)
            )
            strategy1_tau_flat = (
                torque.unsqueeze(1)
                .expand(-1, strategy1_num_cands, -1, -1)
                .reshape(num_samples * strategy1_num_cands, trajectory_length, -1)
            )
            strategy1_uniform = torch.full(
                (num_samples, strategy1_num_cands),
                1.0 / float(strategy1_num_cands),
                device=device,
                dtype=cond.dtype,
            )

        def _predict_eps_cfg(x_in: torch.Tensor, timestep_int: int, cond_in: torch.Tensor, cond_uncond_in: torch.Tensor) -> torch.Tensor:
            """Predict epsilon with optional CFG for arbitrary batch size."""
            if self.unconditional_tau_in_state:
                tokens = self.build_tokens(x_in, timestep_int + 1, skip_normalize=True)
                with torch.no_grad():
                    if self.backbone == "transformer":
                        return self.model(tokens)
                    contexts_local = tokens[:, :num_context, :]
                    return self.model(contexts_local, tokens, torque=None)
            if self.backbone == "transformer":
                cond_tokens = self.build_tokens(
                    x_in, timestep_int + 1, skip_normalize=True, torque=cond_in, include_torque=True
                )
                with torch.no_grad():
                    eps_cond_local = self.model(cond_tokens)
                if guidance_scale != 1.0:
                    uncond_tokens = self.build_tokens(
                        x_in, timestep_int + 1, skip_normalize=True, torque=cond_uncond_in, include_torque=True
                    )
                    with torch.no_grad():
                        eps_uncond_local = self.model(uncond_tokens)
                    return eps_uncond_local + guidance_scale * (eps_cond_local - eps_uncond_local)
                return eps_cond_local

            queries_local = self.build_tokens(x_in, timestep_int + 1, skip_normalize=True)
            contexts_local = queries_local[:, :num_context, :]
            with torch.no_grad():
                eps_cond_local = self.model(contexts_local, queries_local, cond_in)
            if guidance_scale != 1.0:
                with torch.no_grad():
                    eps_uncond_local = self.model(contexts_local, queries_local, cond_uncond_in)
                return eps_uncond_local + guidance_scale * (eps_cond_local - eps_uncond_local)
            return eps_cond_local

        def _strategy1_pick_xt_prev(
            x_t_local: torch.Tensor,
            eps_t_local: torch.Tensor,
            a_bar_t_local: torch.Tensor,
            a_bar_prev_local: torch.Tensor,
            is_last_step: bool,
            t_prev_local_int: int,
        ) -> torch.Tensor:
            """
            Strategy 1:
            Sample N stochastic x_{t-1} candidates from x_t (eta=1), compute each candidate's
            x0 via an extra model call at t-1, score with robust HamRes, and sample by sigmoid weights.
            """
            x0_t_local = self._predict_x0(x_t_local, eps_t_local, a_bar_t_local)
            if is_last_step:
                return x0_t_local

            bsz_local, tlen_local, state_dim_local = x_t_local.shape
            num_cands = strategy1_num_cands
            sigma_t_local = self._compute_legacy_sigma_t(a_bar_t_local, a_bar_prev_local, is_last_step)
            c_local = torch.sqrt(torch.clamp(1.0 - a_bar_prev_local - sigma_t_local * sigma_t_local, min=0.0))

            if sigma_t_local.item() > 0.0:
                z_local = torch.randn(
                    bsz_local, num_cands, tlen_local, state_dim_local,
                    device=x_t_local.device, dtype=x_t_local.dtype
                )
            else:
                z_local = torch.zeros(
                    bsz_local, num_cands, tlen_local, state_dim_local,
                    device=x_t_local.device, dtype=x_t_local.dtype
                )

            x_prev_cands = (
                torch.sqrt(a_bar_prev_local) * x0_t_local.unsqueeze(1)
                + c_local * eps_t_local.unsqueeze(1)
                + sigma_t_local * z_local
            )

            x_prev_flat = x_prev_cands.reshape(bsz_local * num_cands, tlen_local, state_dim_local)
            if self.unconditional_tau_in_state:
                cond_flat = None
                cond_uncond_flat = None
                x_prev_phys_for_tau = self.denormalize_state(x_prev_flat)
                tau_prev = x_prev_phys_for_tau[:, :, self.qpos_dim + self.mom_dim:self.qpos_dim + self.mom_dim + self.torque_dim]
            elif strategy1_cond_flat is not None and bsz_local == num_samples:
                cond_flat = strategy1_cond_flat
                cond_uncond_flat = strategy1_cond_uncond_flat
                tau_prev = strategy1_tau_flat
            else:
                cond_flat = cond.unsqueeze(1).expand(-1, num_cands, -1, -1).reshape(bsz_local * num_cands, tlen_local, -1)
                cond_uncond_flat = cond_uncond.unsqueeze(1).expand(-1, num_cands, -1, -1).reshape(bsz_local * num_cands, tlen_local, -1)
                tau_prev = torque.unsqueeze(1).expand(-1, num_cands, -1, -1).reshape(bsz_local * num_cands, tlen_local, -1)
            eps_prev_flat = _predict_eps_cfg(x_prev_flat, t_prev_local_int, cond_flat, cond_uncond_flat)

            x0_prev_flat = self._predict_x0(x_prev_flat, eps_prev_flat, a_bar_prev_local)
            x0_prev_phys = self.denormalize_state(x0_prev_flat)

            # Hybrid (strategy1 + strategy2):
            # apply one-step guidance to each candidate x0 before scoring.
            if guidance_method == "strategy12_hybrid":
                noise_step_scale_local = torch.sqrt(torch.clamp(1.0 - a_bar_prev_local, min=1e-6)).item()
                x0_prev_phys = run_one_step_guidance_hnn(
                    x0_prev_phys,
                    tau_prev,
                    self.qpos_dim,
                    self.mom_dim,
                    self.data_dt,
                    hnn,
                    alpha_q=alpha_q * noise_step_scale_local,
                    alpha_p=alpha_p * noise_step_scale_local,
                    guidance_trust_lambda=guidance_trust_lambda,
                    guidance_normalize_grad=guidance_normalize_grad,
                    guidance_joint_update=guidance_joint_update,
                )

            q_prev = x0_prev_phys[:, :, :self.qpos_dim]
            p_prev = x0_prev_phys[:, :, self.qpos_dim:self.qpos_dim + self.mom_dim]

            residual = compute_hnn_robust_hamres_energy(
                q_prev,
                p_prev,
                tau_prev,
                hnn,
                self.data_dt,
                smooth_sigma=guidance_hamres_smooth_sigma,
                delta=guidance_hamres_delta,
                min_scale_q=guidance_hamres_min_scale_q,
                min_scale_p=guidance_hamres_min_scale_p,
                reduction="none_batch",
                create_graph=False,
            ).reshape(bsz_local, num_cands)

            eps_score = 1e-12
            residual = torch.nan_to_num(residual, nan=0.0, posinf=1e6, neginf=-1e6)
            std_r = residual.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps_score)
            scaled_r = torch.nan_to_num(residual / std_r, nan=0.0, posinf=1e6, neginf=-1e6)
            weights = torch.sigmoid(-scaled_r).clamp_min(eps_score)
            weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
            sum_w = weights.sum(dim=1, keepdim=True)
            if strategy1_uniform is not None and strategy1_uniform.shape[0] == bsz_local:
                uniform = strategy1_uniform
            else:
                uniform = torch.full_like(weights, 1.0 / float(num_cands))
            probs = torch.where(sum_w > eps_score, weights / sum_w.clamp_min(eps_score), uniform)
            chosen = torch.multinomial(probs, num_samples=1).squeeze(1)
            bidx = torch.arange(bsz_local, device=x_t_local.device)

            # For hybrid mode, selected x_{t-1} is reconstructed from guided x0 candidate
            # and the same per-candidate diffusion noise.
            if guidance_method == "strategy12_hybrid":
                x0_prev_norm = self.normalize_state(x0_prev_phys).reshape(bsz_local, num_cands, tlen_local, state_dim_local)
                chosen_x0 = x0_prev_norm[bidx, chosen]
                chosen_z = z_local[bidx, chosen]
                return (
                    torch.sqrt(a_bar_prev_local) * chosen_x0
                    + c_local * eps_t_local
                    + sigma_t_local * chosen_z
                )

            return x_prev_cands[bidx, chosen]

        for i, t in enumerate(tqdm(ts, total=len(ts), desc="Sampling")):
            t_int = int(t.item())
            eps = _predict_eps_cfg(x, t_int, cond, cond_uncond)
            
            # Extract current state
            x_t = x  # Already just the state part (not in tokens)
            
            # Get diffusion parameters
            a_bar_t = self.alpha_cumprod[t_int]
            if i < len(ts) - 1:
                t_prev_int = int(ts[i + 1].item())
                a_bar_prev = self.alpha_cumprod[t_prev_int]
            else:
                a_bar_prev = a_bar_t.new_tensor(1.0)

            a_bar_t = torch.clamp(a_bar_t, min=1e-6, max=1.0)
            a_bar_prev = torch.clamp(a_bar_prev, min=1e-6, max=1.0)

            # Sampler update
            if sampler == "ddpm":
                alpha_t = self.alphas[t_int]
                beta_t = self.betas[t_int]
                coef1 = 1.0 / torch.sqrt(alpha_t)
                coef2 = beta_t / torch.sqrt(1.0 - a_bar_t)
                mean = coef1 * (x_t - coef2 * eps)
                
                if t_int > 0:
                    sigma_t = torch.sqrt(beta_t)
                    z = torch.randn_like(x_t)
                    x = mean + sigma_t * z
                else:
                    x = mean
                
            elif sampler == "ddim":
                x0 = self._predict_x0(x_t, eps, a_bar_t)
                is_last = (i == len(ts) - 1)
                # Per-step output smoothing (disabled if guidance_only or last_step_only)
                do_output_smooth = smooth_sigma > 0 and not smooth_guidance_only and not smooth_last_step_only

                # Smooth x0 at every step
                if do_output_smooth:
                    from scipy.ndimage import gaussian_filter1d
                    x0_phys = self.denormalize_state(x0)
                    x0_np = x0_phys.cpu().numpy()
                    x0_smoothed = gaussian_filter1d(x0_np, sigma=smooth_sigma, axis=1)
                    x0_phys = torch.tensor(x0_smoothed, dtype=x0_phys.dtype, device=x0_phys.device)
                    x0 = self.normalize_state(x0_phys)

                # Apply HNN-based guidance
                if hnn is not None:
                    # Strategy 1: stochastic candidate selection over x_{t-1} (eta=1), no gradient descent.
                    if guidance_method in {"strategy1", "strategy12_hybrid", "strategy1_target"}:
                        if is_last:
                            if guidance_method == "strategy12_hybrid":
                                x0_phys_last = self.denormalize_state(x0)
                                torque_live = torque
                                if self.unconditional_tau_in_state:
                                    torque_live = x0_phys_last[:, :, self.qpos_dim + self.mom_dim:self.qpos_dim + self.mom_dim + self.torque_dim]
                                x0_phys_last = run_one_step_guidance_hnn(
                                    x0_phys_last,
                                    torque_live,
                                    self.qpos_dim,
                                    self.mom_dim,
                                    self.data_dt,
                                    hnn,
                                    alpha_q=alpha_q,
                                    alpha_p=alpha_p,
                                    guidance_trust_lambda=guidance_trust_lambda,
                                    guidance_normalize_grad=guidance_normalize_grad,
                                    guidance_joint_update=guidance_joint_update,
                                )
                                x = self.normalize_state(x0_phys_last).detach()
                            elif guidance_method == "strategy1_target":
                                x0_phys_last = self.denormalize_state(x0)
                                target_step = float(target_guidance_k) / torch.sqrt(torch.clamp(1.0 - a_bar_t, min=1e-6)).item()
                                x0_phys_last = _apply_reacher_terminal_guidance(x0_phys_last, target_step)
                                x = self.normalize_state(x0_phys_last).detach()
                            else:
                                x = x0
                        else:
                            t_prev_local = int(ts[i + 1].item())
                            x = _strategy1_pick_xt_prev(
                                x_t_local=x_t,
                                eps_t_local=eps,
                                a_bar_t_local=a_bar_t,
                                a_bar_prev_local=a_bar_prev,
                                is_last_step=is_last,
                                t_prev_local_int=t_prev_local,
                            ).detach()
                            if guidance_method == "strategy1_target":
                                eps_prev = _predict_eps_cfg(x, t_prev_local, cond, cond_uncond)
                                x0_prev = self._predict_x0(x, eps_prev, a_bar_prev)
                                x0_prev_phys = self.denormalize_state(x0_prev)
                                target_step = float(target_guidance_k) / torch.sqrt(torch.clamp(1.0 - a_bar_prev, min=1e-6)).item()
                                x0_prev_phys = _apply_reacher_terminal_guidance(x0_prev_phys, target_step)
                                x0_prev = self.normalize_state(x0_prev_phys).detach()
                                eps_coef_prev = torch.sqrt(torch.clamp(1.0 - a_bar_prev, min=1e-6))
                                x = (torch.sqrt(a_bar_prev) * x0_prev + eps_coef_prev * eps_prev).detach()
                        continue

                    x0_phys = self.denormalize_state(x0)

                    # smooth_guidance_only: smooth a copy for guidance, keep original for output
                    if smooth_guidance_only and smooth_sigma > 0:
                        from scipy.ndimage import gaussian_filter1d
                        x0_gui_np = x0_phys.detach().cpu().numpy()
                        x0_gui_smooth = gaussian_filter1d(x0_gui_np, sigma=smooth_sigma, axis=1)
                        x0_gui = torch.tensor(x0_gui_smooth, dtype=x0_phys.dtype, device=x0_phys.device)
                    else:
                        x0_gui = x0_phys

                    # Scale step size by current diffusion noise level.
                    noise_step_scale = torch.sqrt(torch.clamp(1.0 - a_bar_t, min=1e-6)).item()
                    if guidance_method == "strategy2":
                        x0_phys = run_one_step_guidance_hnn(
                            x0_gui,
                            torque if (not self.unconditional_tau_in_state) else x0_gui[:, :, self.qpos_dim + self.mom_dim:self.qpos_dim + self.mom_dim + self.torque_dim],
                            self.qpos_dim,
                            self.mom_dim,
                            self.data_dt,
                            hnn,
                            alpha_q=alpha_q * noise_step_scale,
                            alpha_p=alpha_p * noise_step_scale,
                            guidance_trust_lambda=guidance_trust_lambda,
                            guidance_normalize_grad=guidance_normalize_grad,
                            guidance_joint_update=guidance_joint_update,
                        )
                    else:
                        raise ValueError(
                            f"Unknown guidance_method={guidance_method}. "
                            "Valid: {'strategy1', 'strategy2', 'strategy12_hybrid', 'strategy1_target'}"
                        )
                    # Smooth after guidance (per-step mode only)
                    if do_output_smooth:
                        from scipy.ndimage import gaussian_filter1d
                        x0_np = x0_phys.detach().cpu().numpy()
                        x0_smoothed = gaussian_filter1d(x0_np, sigma=smooth_sigma, axis=1)
                        x0_phys = torch.tensor(x0_smoothed, dtype=x0_phys.dtype, device=x0_phys.device)
                    x0 = self.normalize_state(x0_phys).detach()

                eps_coef = torch.sqrt(1.0 - a_bar_prev)
                x = torch.sqrt(a_bar_prev) * x0 + eps_coef * eps

            elif sampler == "ddpm_legacy":
                x0 = self._predict_x0(x_t, eps, a_bar_t)
                is_last = (i == len(ts) - 1)
                do_output_smooth = smooth_sigma > 0 and not smooth_guidance_only and not smooth_last_step_only

                # Apply HNN-based guidance
                if hnn is not None:
                    if guidance_method in {"strategy1", "strategy12_hybrid", "strategy1_target"}:
                        if is_last:
                            if guidance_method == "strategy12_hybrid":
                                x0_phys_last = self.denormalize_state(x0)
                                torque_live = torque
                                if self.unconditional_tau_in_state:
                                    torque_live = x0_phys_last[:, :, self.qpos_dim + self.mom_dim:self.qpos_dim + self.mom_dim + self.torque_dim]
                                x0_phys_last = run_one_step_guidance_hnn(
                                    x0_phys_last,
                                    torque_live,
                                    self.qpos_dim,
                                    self.mom_dim,
                                    self.data_dt,
                                    hnn,
                                    alpha_q=alpha_q,
                                    alpha_p=alpha_p,
                                    guidance_trust_lambda=guidance_trust_lambda,
                                    guidance_normalize_grad=guidance_normalize_grad,
                                    guidance_joint_update=guidance_joint_update,
                                )
                                x = self.normalize_state(x0_phys_last).detach()
                            elif guidance_method == "strategy1_target":
                                x0_phys_last = self.denormalize_state(x0)
                                target_step = float(target_guidance_k) / torch.sqrt(torch.clamp(1.0 - a_bar_t, min=1e-6)).item()
                                x0_phys_last = _apply_reacher_terminal_guidance(x0_phys_last, target_step)
                                x = self.normalize_state(x0_phys_last).detach()
                            else:
                                x = x0
                        else:
                            t_prev_local = int(ts[i + 1].item())
                            x = _strategy1_pick_xt_prev(
                                x_t_local=x_t,
                                eps_t_local=eps,
                                a_bar_t_local=a_bar_t,
                                a_bar_prev_local=a_bar_prev,
                                is_last_step=is_last,
                                t_prev_local_int=t_prev_local,
                            ).detach()
                            if guidance_method == "strategy1_target":
                                eps_prev = _predict_eps_cfg(x, t_prev_local, cond, cond_uncond)
                                x0_prev = self._predict_x0(x, eps_prev, a_bar_prev)
                                x0_prev_phys = self.denormalize_state(x0_prev)
                                target_step = float(target_guidance_k) / torch.sqrt(torch.clamp(1.0 - a_bar_prev, min=1e-6)).item()
                                x0_prev_phys = _apply_reacher_terminal_guidance(x0_prev_phys, target_step)
                                x0_prev = self.normalize_state(x0_prev_phys).detach()
                                c_prev = torch.sqrt(torch.clamp(1.0 - a_bar_prev, min=1e-6))
                                x = (torch.sqrt(a_bar_prev) * x0_prev + c_prev * eps_prev).detach()
                        continue

                    x0 = torch.clamp(x0, -1.0, 1.0)
                    x0_phys = self.denormalize_state(x0)

                    # smooth_guidance_only: smooth a copy for guidance, keep original for output
                    if smooth_guidance_only and smooth_sigma > 0:
                        from scipy.ndimage import gaussian_filter1d
                        x0_gui_np = x0_phys.detach().cpu().numpy()
                        x0_gui_smooth = gaussian_filter1d(x0_gui_np, sigma=smooth_sigma, axis=1)
                        x0_gui = torch.tensor(x0_gui_smooth, dtype=x0_phys.dtype, device=x0_phys.device)
                    else:
                        x0_gui = x0_phys

                    noise_step_scale = torch.sqrt(torch.clamp(1.0 - a_bar_t, min=1e-6)).item()
                    if guidance_method == "strategy2":
                        x0_phys = run_one_step_guidance_hnn(
                            x0_gui,
                            torque if (not self.unconditional_tau_in_state) else x0_gui[:, :, self.qpos_dim + self.mom_dim:self.qpos_dim + self.mom_dim + self.torque_dim],
                            self.qpos_dim,
                            self.mom_dim,
                            self.data_dt,
                            hnn,
                            alpha_q=alpha_q * noise_step_scale,
                            alpha_p=alpha_p * noise_step_scale,
                            guidance_trust_lambda=guidance_trust_lambda,
                            guidance_normalize_grad=guidance_normalize_grad,
                            guidance_joint_update=guidance_joint_update,
                        )
                    else:
                        raise ValueError(
                            f"Unknown guidance_method={guidance_method}. "
                            "Valid: {'strategy1', 'strategy2', 'strategy12_hybrid', 'strategy1_target'}"
                        )
                    # Smooth after guidance (per-step mode only)
                    if do_output_smooth:
                        from scipy.ndimage import gaussian_filter1d
                        x0_np = x0_phys.detach().cpu().numpy()
                        x0_smoothed = gaussian_filter1d(x0_np, sigma=smooth_sigma, axis=1)
                        x0_phys = torch.tensor(x0_smoothed, dtype=x0_phys.dtype, device=x0_phys.device)
                    x0 = self.normalize_state(x0_phys).detach()

                sigma_t = self._compute_legacy_sigma_t(a_bar_t, a_bar_prev, i == len(ts) - 1)
                c = torch.sqrt(torch.clamp(1.0 - a_bar_prev - sigma_t * sigma_t, min=0.0))
                z = torch.randn_like(x_t) if (sigma_t.item() > 0.0) else torch.zeros_like(x_t)
                x = torch.sqrt(a_bar_prev) * x0 + c * eps + sigma_t * z
        
        # Denormalize state
        state = self.denormalize_state(x)
        if self.unconditional_tau_in_state:
            torque = state[:, :, self.qpos_dim + self.mom_dim:self.qpos_dim + self.mom_dim + self.torque_dim]

        # Post-loop smoothing: smooth final output after all sampling is done
        if smooth_last_step_only and smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            state_np = state.detach().cpu().numpy()
            state_smoothed = gaussian_filter1d(state_np, sigma=smooth_sigma, axis=1)
            state = torch.tensor(state_smoothed, dtype=state.dtype, device=state.device)

        # Restore original weights if EMA was applied
        if use_ema and self.ema is not None:
            self.ema.restore(self.model)

        self.model.train()
        return state, torque
    
    def save_trajectories_to_h5(
        self,
        output_path: str,
        num_samples: int = 100,
        trajectory_length: int = None,
        **sample_kwargs
    ):
        """
        Generate and save trajectories to h5 file (consistent with training format).
        
        Args:
            output_path: path to output h5 file
            num_samples: number of trajectories to generate
            trajectory_length: length of each trajectory (default: same as training trajectory length)
            **sample_kwargs: additional kwargs for sample_trajectories
        """
        if trajectory_length is None:
            trajectory_length = self.max_timesteps
        
        print(f"Generating {num_samples} trajectories of length {trajectory_length}...")
        state, torque = self.sample_trajectories(
            num_samples, trajectory_length,
            **sample_kwargs
        )
        
        # Move to CPU and convert to numpy
        state_np = state.cpu().numpy()
        torque_np = torque.cpu().numpy()
        
        # Split state into components: [qpos | mom]
        qpos = state_np[:, :, :self.qpos_dim]
        mom = state_np[:, :, self.qpos_dim:self.qpos_dim + self.mom_dim]
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        
        print(f"Saving to {output_path}...")
        with h5py.File(output_path, 'w') as f:
            f.attrs['num_trajectories'] = num_samples
            f.attrs['num_steps'] = trajectory_length
            
            for i in range(num_samples):
                traj_group = f.create_group(f'traj_{i}')
                traj_group.create_dataset('seq_qpos', data=qpos[i], dtype='f8')
                traj_group.create_dataset('seq_mom', data=mom[i], dtype='f8')
                traj_group.create_dataset('seq_torque', data=torque_np[i], dtype='f8')
        
        print(f"Saved {num_samples} trajectories to {output_path}")



# -------------------------
# Main training script
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Train or Generate Samples with Trajectory DPF")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["train", "generate_samples"], default=config.DEFAULT_MODE,
                        help="Mode: 'train' to train the model, 'generate_samples' to generate samples from a checkpoint")
    
    # Data and model paths
    parser.add_argument("--h5_path", type=str, default=config.DEFAULT_H5_PATH,
                        help="Path to training data h5 file")
    parser.add_argument("--val_h5_path", type=str, default=None,
                        help="Optional path to a separate validation h5 file. "
                             "If set, uses strict train/val split across files.")
    parser.add_argument("--checkpoint_dir", type=str, default=config.DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume_from_checkpoint", type=str, default=config.DEFAULT_RESUME_CHECKPOINT, 
                        help="Path to checkpoint to resume training from (for train mode) or to load for generation (for generate_samples mode)")
    
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
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["global_cond", "no_shift_right", "no_state_interaction", "torque_concat"],
                        help="Ablation study mode (default: None = full model)")
    parser.add_argument("--fixed_trajectory_length", type=int, default=None,
                        help="Fixed trajectory length for non-DPF training (disables variable-length). "
                             "When set, all training samples use this exact length instead of random lengths from 100-1000.")
    parser.add_argument(
        "--unconditional_tau_in_state",
        action="store_true",
        help="Train unconditional diffusion over full state [qpos|mom|torque] (no torque conditioning input).",
    )
    parser.add_argument(
        "--tau_loss_mask_prob",
        type=float,
        default=0.0,
        help="In unconditional mode, probability to exclude tau channels from denoising loss for a batch.",
    )
    parser.add_argument(
        "--training_context_mode",
        type=str,
        default="random_subset",
        choices=["random_subset", "future_context", "shifted_future_context", "shifted_future_context_cleanprefix"],
        help="Perceiver training/validation context-query construction.",
    )
    parser.add_argument(
        "--query_loss_decay",
        type=str,
        default="none",
        choices=["none", "linear", "exp", "power"],
        help="Optional front-loaded weighting over query-token denoising loss.",
    )
    parser.add_argument(
        "--query_loss_decay_strength",
        type=float,
        default=1.0,
        help="Strength parameter for the query loss decay profile.",
    )
    
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=config.DEFAULT_NUM_SAMPLES, help="Number of trajectories to generate")
    parser.add_argument("--trajectory_length", type=int, default=1000,
                        help="Length of generated trajectories (default: same as training trajectory length)")
    parser.add_argument("--output_path", type=str, default=config.DEFAULT_OUTPUT_PATH,
                        help="Output path for generated samples")
    parser.add_argument("--sampler", type=str, choices=["ddpm", "ddim", "ddpm_legacy"], default=config.DEFAULT_SAMPLER,
                        help="Sampling method: 'ddpm' (full schedule, stochastic), 'ddim' (fast, deterministic), or 'ddpm_legacy' (subsampled stochastic)")
    parser.add_argument("--num_diffusion_steps", type=int, default=config.DEFAULT_NUM_DIFFUSION_STEPS,
                        help="Number of diffusion steps for sampling (can be less than training steps for DDIM)")
    parser.add_argument("--context_fraction", type=float, default=config.DEFAULT_CONTEXT_FRACTION,
                        help="Fraction of timesteps to use as context during sampling")
    parser.add_argument("--use_ema", type=bool, default=config.DEFAULT_USE_EMA,
                        help="Whether to use EMA weights for sampling")
    
    # CFG and guidance parameters
    parser.add_argument("--guidance_scale", type=float, default=1.2,
                        help="Classifier-free guidance scale (1.0 = no CFG, >1.0 = stronger conditioning)")
    parser.add_argument("--hnn_checkpoint", type=str, default='/home/gsang/Projects/Perceiver_IO/checkpoints/SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt',
                        help="Path to HNN checkpoint for physics-based guidance during sampling")
    parser.add_argument(
        "--guidance_method",
        type=str,
        choices=["strategy1", "strategy2", "strategy12_hybrid", "strategy1_target"],
        default="strategy2",
        help="Guidance strategy: strategy1 (candidate resampling), strategy2 (one-step guidance), strategy12_hybrid (resampling + one-step guidance per candidate), or strategy1_target (resampling + terminal target guidance).",
    )
    parser.add_argument(
        "--guidance_no_normalize_grad",
        action="store_true",
        help="Strategy2 only: disable guidance gradient normalization.",
    )
    parser.add_argument(
        "--guidance_joint_update",
        action="store_true",
        help="Strategy2 only: use one shared gradient norm across q/p.",
    )
    parser.add_argument(
        "--guidance_num_candidates",
        type=int,
        default=16,
        help="Number of candidates per sampling step for strategy1.",
    )
    parser.add_argument("--target_guidance_k", type=float, default=0.01,
                        help="Step-size coefficient k for strategy1_target guidance.")
    parser.add_argument("--target_lambda_power", type=float, default=4.0,
                        help="Terminal-heavy lambda schedule power for strategy1_target.")
    parser.add_argument("--target_x", type=float, default=None,
                        help="Optional fixed target x for strategy1_target. If omitted, sample random Reacher targets.")
    parser.add_argument("--target_y", type=float, default=None,
                        help="Optional fixed target y for strategy1_target. If omitted, sample random Reacher targets.")
    parser.add_argument(
        "--guidance_dynamic_steps_inv_sqrt",
        action="store_true",
        help="Deprecated (ignored): legacy dynamic-step option from removed guidance strategies.",
    )
    parser.add_argument(
        "--guidance_dynamic_steps_max",
        type=int,
        default=5,
        help="Deprecated (ignored): legacy dynamic-step cap from removed guidance strategies.",
    )
    parser.add_argument("--langevin_step_size", type=float, default=2e-4,
                        help="Deprecated (ignored): legacy Langevin option.")
    parser.add_argument("--langevin_noise_scale", type=float, default=0,
                        help="Deprecated (ignored): legacy Langevin option.")
    parser.add_argument("--lambda_init", type=float, default=1,
                        help="Deprecated (ignored): legacy trust regularization option.")
    parser.add_argument("--chunk_length", type=int, default=50,
                        help="Deprecated (ignored): legacy integration-guidance option.")
    parser.add_argument("--seed", type=int, default=228,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--use_trained_torque", action="store_true", default=False,
                        help="Use torque sequences from training data instead of generating new random ones")
    parser.add_argument("--torque_file", type=str, default=None,
                        help="Path to HDF5 file with pre-generated torque sequences (overrides --use_trained_torque)")
    parser.add_argument(
        "--sampling_torque_policy",
        type=str,
        default="mixed_reacher",
        choices=["legacy_sinusoidal", "sinusoidal", "lpf_uniform", "drift_ou", "mixed_reacher"],
        help="Torque policy used when sampling must generate torque from scratch.",
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
        help="Global scale applied to policy-generated torque when sampling from scratch.",
    )
    parser.add_argument(
        "--comparison_output_dir",
        type=str,
        default="/home/gsang/Projects/Perceiver_IO/plots/sampling_comparisons",
        help="Directory for generated-vs-reconstructed comparison images and summary files.",
    )
    
    # W&B arguments
    parser.add_argument("--wandb", type=bool, default=config.DEFAULT_WANDB_ENABLED, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=config.DEFAULT_WANDB_PROJECT, help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (username or team)")
    parser.add_argument("--wandb_run_name", type=str, default=config.DEFAULT_WANDB_RUN_NAME, help="W&B run name")
    
    args = parser.parse_args()

    # Speed knobs for modern NVIDIA GPUs (A100 etc.)
    # - TF32 accelerates float32 matmuls on Tensor Cores with negligible impact for most training.
    # - bf16 mixed precision enables Flash SDP kernels for attention in torch 2.0 (big speedup).
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception as e:
            print(f"[Perf] Warning: failed to set float32 matmul precision: {e}")
    
    # Execute based on mode
    if args.mode == "generate_samples":
        # For generation mode, load everything from checkpoint - no dataset needed
        if args.resume_from_checkpoint is None:
            print("Error: --resume_from_checkpoint is required for generate_samples mode")
            return
        
        print(f"Loading model from checkpoint: {args.resume_from_checkpoint}")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load model directly from checkpoint (includes hyperparameters and buffers)
        model = TrajectoryDPF.load_from_checkpoint(
            args.resume_from_checkpoint,
            map_location=device,
        )
        model = model.to(device)
        print(f"[Sampling] Model loaded and moved to device: {device}")
        print(f"[Sampling] Model dimensions: qpos={model.qpos_dim}, mom={model.mom_dim}, torque={model.torque_dim}")
        print(f"[Sampling] Trajectory length: {model.max_timesteps}")
        
        # Load EMA shadow if present in checkpoint
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        ema_shadow = checkpoint.get('ema_shadow', None)
        ema_decay = checkpoint.get('ema_decay', 0.9995)
        if ema_shadow is not None:
            model.ema = EMA(model.model, decay=ema_decay)
            print(f"[Sampling] Loading EMA shadow with {len(ema_shadow)} parameters...")
            
            loaded_count = 0
            for name, tensor in ema_shadow.items():
                if name in model.ema.shadow:
                    model.ema.shadow[name] = tensor.to(device=device, dtype=model.ema.shadow[name].dtype)
                    loaded_count += 1
            
            print(f"[Sampling] Loaded {loaded_count}/{len(ema_shadow)} EMA parameters")
            model._ema_loaded = True
            print("[Sampling] ✓ EMA shadow restored from checkpoint.")
        else:
            print("[Sampling] WARNING: No EMA shadow found in checkpoint!")
            print("[Sampling] Proceeding with regular model weights.")
        
        # Load HNN for guidance
        hnn = None
        if args.hnn_checkpoint:
            from src.models.HNN import HNNWrapper
            print(f"[Sampling] Loading HNN from: {args.hnn_checkpoint}")
            
            # Load HNNWrapper (includes input scaling q_std, p_std)
            hnn = HNNWrapper.load_from_checkpoint(args.hnn_checkpoint, map_location=device)
            hnn = hnn.to(device)
            hnn.eval()
            print(f"[Sampling] ✓ HNNWrapper loaded for energy consistency")
            print(f"[Sampling]   Scaling: q_std={hnn.q_std.mean().item():.4f}, p_std={hnn.p_std.mean().item():.4f}")
        
        # Determine trajectory length (default to training length if not specified)
        trajectory_length = args.trajectory_length if args.trajectory_length is not None else model.max_timesteps
        
        # Generate samples
        print(f"\nGenerating {args.num_samples} sample trajectories of length {trajectory_length}...")
        print(f"[Sampling] Training trajectory length: {model.max_timesteps}")
        if trajectory_length != model.max_timesteps:
            print(f"[Sampling] Extending trajectory length to: {trajectory_length}")
        print(f"[Sampling] Using sampler={args.sampler}, num_diffusion_steps={args.num_diffusion_steps}, "
              f"context_fraction={args.context_fraction}, use_ema={args.use_ema}, seed={args.seed}")
        print(
            "[Sampling] Torque policy config: "
            f"policy={args.sampling_torque_policy}, mix={args.sampling_torque_mix}, "
            f"lpf_beta={args.sampling_lpf_uniform_beta}, scale={args.sampling_torque_scale}"
        )
        os.makedirs(args.comparison_output_dir, exist_ok=True)
        print(f"[Sampling] Comparison outputs will be saved under: {args.comparison_output_dir}")
        if hnn is not None:
            print(f"[Sampling] Guidance enabled: method={args.guidance_method}")
            if args.guidance_method == "strategy2":
                print(f"[Sampling] Strategy2 options: normalize_grad={not args.guidance_no_normalize_grad}, "
                      f"joint_update={args.guidance_joint_update}")
            if args.guidance_method == "strategy12_hybrid":
                print(f"[Sampling] Strategy12 options: num_candidates={args.guidance_num_candidates}, "
                      f"normalize_grad={not args.guidance_no_normalize_grad}, joint_update={args.guidance_joint_update}")
            if args.guidance_method == "strategy1_target":
                print(
                    f"[Sampling] Strategy1-target options: num_candidates={args.guidance_num_candidates}, "
                    f"k={args.target_guidance_k}, lambda_power={args.target_lambda_power}, "
                    f"target=({args.target_x},{args.target_y})"
                )
        
        # Resolve XML for reconstruction/comparison using checkpoint metadata first.
        xml_cleanup_path = None
        if model.xml_content is not None:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as tf:
                tf.write(model.xml_content)
                comparison_xml_path = tf.name
            xml_cleanup_path = comparison_xml_path
            print(f"[Sampling] Using XML from checkpoint for comparison: {comparison_xml_path}")
        else:
            comparison_xml_path = '/home/gsang/Projects/Perceiver_IO/configs/rigid_arm_hinge.xml'
            print(f"[Sampling] WARNING: checkpoint has no XML content; fallback XML: {comparison_xml_path}")
        shutil.copy2(comparison_xml_path, os.path.join(args.comparison_output_dir, "model.xml"))

        # Helper function to compute MSE statistics for a set of trajectories
        def compute_mse_stats(state_tensor, torque_tensor, save_plots=False, params_str=""):
            mse_qpos_list = []
            mse_mom_list = []
            mse_total_list = []
            
            for i in range(state_tensor.shape[0]):
                state_np = state_tensor[i].cpu().numpy()
                torque_np = torque_tensor[i].cpu().numpy()
                generated = {
                    'seq_qpos': state_np[:, :model.qpos_dim],
                    'seq_mom': state_np[:, model.qpos_dim:model.qpos_dim + model.mom_dim],
                    'seq_torque': torque_np,
                }
                mse_dict = compare_generated_with_reconstructed(
                    generated, comparison_xml_path,
                    args.comparison_output_dir,
                    dt=float(model.dt),
                    data_dt=float(model.data_dt),
                    name=f'{params_str}_{i}' if save_plots else None
                )
                mse_qpos_list.append(mse_dict['mse_qpos'])
                mse_mom_list.append(mse_dict['mse_mom'])
                mse_total_list.append(mse_dict['mse_total'])
            
            return {
                'qpos': np.array(mse_qpos_list),
                'mom': np.array(mse_mom_list),
                'total': np.array(mse_total_list)
            }
        
        # Load torque from file or training data if requested
        trained_torque = None
        torque_source = "random"
        
        if args.torque_file:
            # Load pre-generated torques from HDF5 file
            print(f"[Sampling] Loading torque sequences from file: {args.torque_file}")
            with h5py.File(args.torque_file, 'r') as f:
                torques_np = f['torques'][:]
                print(f"[Sampling] File contains {torques_np.shape[0]} torque sequences")
                print(f"[Sampling] File metadata: seed={f.attrs.get('seed', 'N/A')}, dt={f.attrs.get('dt', 'N/A')}")
                
                # Select the requested number of samples
                if args.num_samples > torques_np.shape[0]:
                    print(f"[Warning] Requested {args.num_samples} samples but file only has {torques_np.shape[0]}. Using all available.")
                    args.num_samples = torques_np.shape[0]
                
                trained_torque = torch.tensor(torques_np[:args.num_samples], dtype=torch.float32, device=device)
                print(f"[Sampling] Loaded {trained_torque.shape[0]} torque sequences from file")
            torque_source = "file"
        elif args.use_trained_torque:
            print(f"[Sampling] Loading torque sequences from training data: {args.h5_path}")
            dataset = TrajectoryDPFCached(args.h5_path, trajectory_length=trajectory_length)
            
            # Sample random trajectories from dataset and extract their torques
            import random
            random.seed(args.seed)
            indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
            torque_list = []
            for idx in indices:
                sample = dataset[idx]
                torque_list.append(sample['seq_torque'])
            
            trained_torque = torch.stack(torque_list).to(device)
            print(f"[Sampling] Loaded {len(torque_list)} torque sequences from training data")
            torque_source = "trained"
        
        # Build naming strings
        ema_str = "ema" if args.use_ema else "noema"
        base_params = f"seed{args.seed}_{args.sampler}_diff{args.num_diffusion_steps}_ctx{args.context_fraction}_{ema_str}_cfg{args.guidance_scale}"
        
        # ============================================================
        # STEP 1: Run baseline (no guidance) if guidance is enabled
        # ============================================================
        baseline_mse = None
        shared_torque = trained_torque  # Will store torque from baseline run to reuse
        
        if hnn is not None:
            print(f"\n[Step 1/2] Generating BASELINE trajectories (no guidance)...")
            
            # Set seed for reproducibility
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            
            state_baseline, torque_baseline = model.sample_trajectories(
                num_samples=args.num_samples,
                trajectory_length=trajectory_length,
                num_diffusion_steps=args.num_diffusion_steps,
                context_fraction=args.context_fraction,
                use_ema=args.use_ema,
                sampler=args.sampler,
                guidance_scale=args.guidance_scale,
                hnn=None,  # No HNN guidance
                torque=trained_torque,
                sampling_torque_policy=args.sampling_torque_policy,
                sampling_torque_mix=args.sampling_torque_mix,
                sampling_lpf_uniform_beta=args.sampling_lpf_uniform_beta,
                sampling_torque_scale=args.sampling_torque_scale,
                target_guidance_k=args.target_guidance_k,
                target_lambda_power=args.target_lambda_power,
                target_x=args.target_x,
                target_y=args.target_y,
            )
            
            # Store torque from baseline to reuse in guided run (ensures same torque)
            shared_torque = torque_baseline
            
            # Compute baseline MSE (no plots)
            print("[Step 1/2] Computing baseline MSE...")
            baseline_mse = compute_mse_stats(state_baseline, torque_baseline, save_plots=False)
        
        # ============================================================
        # STEP 2: Run with guidance (or just regular sampling if no guidance)
        # ============================================================
        if hnn is not None:
            print(f"\n[Step 2/2] Generating GUIDED trajectories ({args.guidance_method})...")
        
        # Set seed again for reproducibility (same diffusion noise)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        state, torque = model.sample_trajectories(
            num_samples=args.num_samples,
            trajectory_length=trajectory_length,
            num_diffusion_steps=args.num_diffusion_steps,
            context_fraction=args.context_fraction,
            use_ema=args.use_ema,
            sampler=args.sampler,
            guidance_scale=args.guidance_scale,
            hnn=hnn,
            guidance_method=args.guidance_method,
            guidance_normalize_grad=(not args.guidance_no_normalize_grad),
            guidance_joint_update=args.guidance_joint_update,
            guidance_num_candidates=args.guidance_num_candidates,
            torque=shared_torque,  # Use same torque as baseline
            sampling_torque_policy=args.sampling_torque_policy,
            sampling_torque_mix=args.sampling_torque_mix,
            sampling_lpf_uniform_beta=args.sampling_lpf_uniform_beta,
            sampling_torque_scale=args.sampling_torque_scale,
            target_guidance_k=args.target_guidance_k,
            target_lambda_power=args.target_lambda_power,
            target_x=args.target_x,
            target_y=args.target_y,
        )
        sampled_targets = getattr(model, "_last_sampled_targets", None)
        
        # Build guidance string for naming
        if hnn is not None:
            if args.guidance_method == "strategy1":
                guidance_str = f"hnn-{args.guidance_method}_cands{args.guidance_num_candidates}"
            else:
                guidance_str = f"hnn-{args.guidance_method}"
        else:
            guidance_str = "hnn-off"
        
        params_str = f"{base_params}_{guidance_str}_len{trajectory_length}_torque-{torque_source}"
        
        # Compute guided MSE and save plots
        print(f"\nSaving comparison plots...")
        guided_mse = compute_mse_stats(state, torque, save_plots=True, params_str=params_str)
        
        # ============================================================
        # Print MSE Results
        # ============================================================
        print(f"\n{'='*70}")
        print(f"  MSE RESULTS: Physics Reconstruction Error (Generated vs MuJoCo)")
        print(f"{'='*70}")
        print(f"  Seed: {args.seed}  |  Samples: {args.num_samples}  |  Length: {trajectory_length}")
        print(f"{'='*70}")
        
        if baseline_mse is not None:
            # Show comparison
            print(f"\n  {'Metric':<12} {'Baseline':>14} {'Guided':>14} {'Improvement':>14} {'% Improv':>10}")
            print(f"  {'-'*64}")
            
            for metric in ['qpos', 'mom', 'total']:
                base_mean = baseline_mse[metric].mean()
                guid_mean = guided_mse[metric].mean()
                improvement = base_mean - guid_mean
                pct_improv = (improvement / base_mean) * 100 if base_mean > 0 else 0
                
                print(f"  MSE {metric:<7} {base_mean:>14.6f} {guid_mean:>14.6f} {improvement:>+14.6f} {pct_improv:>+9.1f}%")
            
            print(f"\n  Guidance: {guidance_str}")
        else:
            # No guidance, just show results
            print(f"\n  {'Metric':<12} {'Mean':>14} {'Std':>14}")
            print(f"  {'-'*42}")
            for metric in ['qpos', 'mom', 'total']:
                print(f"  MSE {metric:<7} {guided_mse[metric].mean():>14.6f} {guided_mse[metric].std():>14.6f}")
        
        print(f"{'='*70}\n")

        # Save run metadata and summary metrics for reproducibility/inspection.
        torque_np_for_stats = torque.detach().cpu().numpy()
        tau_diff = np.diff(torque_np_for_stats, axis=1)
        summary_path = os.path.join(args.comparison_output_dir, "summary_metrics.txt")
        with open(summary_path, "w") as f:
            f.write("Physics reconstruction metrics\n")
            f.write(f"seed: {args.seed}\n")
            f.write(f"num_samples: {args.num_samples}\n")
            f.write(f"trajectory_length: {trajectory_length}\n")
            f.write(f"sampler: {args.sampler}\n")
            f.write(f"num_diffusion_steps: {args.num_diffusion_steps}\n")
            f.write(f"context_fraction: {args.context_fraction}\n")
            f.write(f"use_ema: {args.use_ema}\n")
            f.write(f"torque_source: {torque_source}\n")
            f.write(f"sampling_torque_policy: {args.sampling_torque_policy}\n")
            f.write(f"sampling_torque_mix: {args.sampling_torque_mix}\n")
            f.write(f"sampling_lpf_uniform_beta: {args.sampling_lpf_uniform_beta}\n")
            f.write(f"sampling_torque_scale: {args.sampling_torque_scale}\n")
            f.write(f"target_guidance_k: {args.target_guidance_k}\n")
            f.write(f"target_lambda_power: {args.target_lambda_power}\n")
            f.write(f"target_xy: ({args.target_x}, {args.target_y})\n")
            f.write(f"torque_abs_mean: {np.abs(torque_np_for_stats).mean():.8f}\n")
            f.write(f"torque_abs_max: {np.abs(torque_np_for_stats).max():.8f}\n")
            f.write(f"tau_diff_std: {tau_diff.std() if tau_diff.size > 0 else 0.0:.8f}\n")
            f.write("\nPer-sample guided reconstruction MSE\n")
            for i in range(args.num_samples):
                f.write(
                    f"sample_{i}: qpos={guided_mse['qpos'][i]:.8f}, "
                    f"mom={guided_mse['mom'][i]:.8f}, total={guided_mse['total'][i]:.8f}\n"
                )
            if baseline_mse is not None:
                f.write("\nAggregate means (baseline vs guided)\n")
                for metric in ["qpos", "mom", "total"]:
                    f.write(
                        f"{metric}: baseline_mean={baseline_mse[metric].mean():.8f}, "
                        f"guided_mean={guided_mse[metric].mean():.8f}\n"
                    )
            else:
                f.write("\nAggregate means (guided only)\n")
                for metric in ["qpos", "mom", "total"]:
                    f.write(f"{metric}: guided_mean={guided_mse[metric].mean():.8f}\n")

        run_config_path = os.path.join(args.comparison_output_dir, "run_config.json")
        run_config = {
            "checkpoint": args.resume_from_checkpoint,
            "output_path": args.output_path,
            "comparison_output_dir": args.comparison_output_dir,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "trajectory_length": trajectory_length,
            "sampler": args.sampler,
            "num_diffusion_steps": args.num_diffusion_steps,
            "context_fraction": args.context_fraction,
            "use_ema": bool(args.use_ema),
            "guidance_scale": float(args.guidance_scale),
            "guidance_method": args.guidance_method if hnn is not None else "none",
            "hnn_checkpoint": args.hnn_checkpoint if hnn is not None else None,
            "torque_source": torque_source,
            "sampling_torque_policy": args.sampling_torque_policy,
            "sampling_torque_mix": args.sampling_torque_mix,
            "sampling_lpf_uniform_beta": float(args.sampling_lpf_uniform_beta),
            "sampling_torque_scale": float(args.sampling_torque_scale),
            "target_guidance_k": float(args.target_guidance_k),
            "target_lambda_power": float(args.target_lambda_power),
            "target_x": None if args.target_x is None else float(args.target_x),
            "target_y": None if args.target_y is None else float(args.target_y),
            "targets_saved_in_h5": sampled_targets is not None,
            "guided_mse_mean": {
                "qpos": float(guided_mse["qpos"].mean()),
                "mom": float(guided_mse["mom"].mean()),
                "total": float(guided_mse["total"].mean()),
            },
            "baseline_mse_mean": None if baseline_mse is None else {
                "qpos": float(baseline_mse["qpos"].mean()),
                "mom": float(baseline_mse["mom"].mean()),
                "total": float(baseline_mse["total"].mean()),
            },
        }
        with open(run_config_path, "w") as f:
            json.dump(run_config, f, indent=2, sort_keys=True)
        print(f"[Sampling] Wrote summary metrics: {summary_path}")
        print(f"[Sampling] Wrote run config: {run_config_path}")
        if xml_cleanup_path is not None and os.path.exists(xml_cleanup_path):
            os.remove(xml_cleanup_path)
        
        # Save to h5
        state_all = state.cpu().numpy()
        torque_all = torque.cpu().numpy()
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        with h5py.File(args.output_path, 'w') as f:
            f.attrs['num_trajectories'] = args.num_samples
            f.attrs['num_steps'] = trajectory_length
            if sampled_targets is not None:
                f.create_dataset('targets_xy', data=sampled_targets, dtype='f8')
            for i in range(args.num_samples):
                g = f.create_group(f'traj_{i}')
                g.create_dataset('seq_qpos', data=state_all[i, :, :model.qpos_dim], dtype='f8')
                g.create_dataset('seq_mom', data=state_all[i, :, model.qpos_dim:model.qpos_dim + model.mom_dim], dtype='f8')
                g.create_dataset('seq_torque', data=torque_all[i], dtype='f8')
                if sampled_targets is not None:
                    g.create_dataset('target_xy', data=sampled_targets[i], dtype='f8')
        
        print(f"Sample generation complete! Saved to {args.output_path}")
        return
    
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

    # Get simulation metadata from dataset
    dt = full_dataset.dt
    data_dt = full_dataset.data_dt
    xml_content = full_dataset.xml
    
    print(f"Dataset info:")
    print(f"  Trajectories: {len(dataset)}")
    print(f"  Timesteps: {max_timesteps}")
    print(f"  qpos_dim: {qpos_dim}, mom_dim: {mom_dim}, torque_dim: {torque_dim}")
    print(f"  dt: {dt}, data_dt: {data_dt}")
    print(f"  XML content: {'loaded' if xml_content else 'not available'}")
    print(f"  unconditional_tau_in_state: {args.unconditional_tau_in_state}")
    print(f"  tau_loss_mask_prob: {args.tau_loss_mask_prob}")
    print(f"  training_context_mode: {args.training_context_mode}")
    print(f"  query_loss_decay: {args.query_loss_decay}")
    print(f"  query_loss_decay_strength: {args.query_loss_decay_strength}")
    
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
        stats_loader, qpos_dim, mom_dim, torque_dim, max_timesteps
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
        debug_plot_dir = '/home/gsang/Projects/Perceiver_IO/plots/training_debug'
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
        # State:
        # - conditional mode: [qpos | mom]
        # - unconditional mode: [qpos | mom | torque]
        if args.unconditional_tau_in_state:
            state_min = torch.cat([qpos_min, mom_min, torque_min], dim=-1)
            state_max = torch.cat([qpos_max, mom_max, torque_max], dim=-1)
        else:
            state_min = torch.cat([qpos_min, mom_min], dim=-1)
            state_max = torch.cat([qpos_max, mom_max], dim=-1)
        state_range = state_max - state_min
        
        if args.unconditional_tau_in_state:
            full_state = torch.cat([sample_qpos, sample_mom, sample_torque], dim=-1)  # [T, state_dim]
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

    # Build ablation config from CLI argument (None = full model)
    ablation_config = None
    if args.ablation is not None:
        ablation_map = {
            "global_cond": AblationConfig.global_cond,
            "no_shift_right": AblationConfig.no_shift_right,
            "no_state_interaction": AblationConfig.no_state_interaction,
            "torque_concat": AblationConfig.torque_concat,
        }
        ablation_config = ablation_map[args.ablation]()
        print(f"[Ablation] Running ablation study: {args.ablation}")
        print(f"[Ablation] Config: {ablation_config}")

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
        unconditional_tau_in_state=args.unconditional_tau_in_state,
        tau_loss_mask_prob=args.tau_loss_mask_prob,
        training_context_mode=args.training_context_mode,
        query_loss_decay=args.query_loss_decay,
        query_loss_decay_strength=args.query_loss_decay_strength,
        ablation_config=ablation_config,
        dt=dt,
        data_dt=data_dt,
        xml_content=xml_content,
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
            if args.ablation:
                wandb_run_name = f"{wandb_run_name}_ablation-{args.ablation}" if wandb_run_name else f"ablation-{args.ablation}"
            if args.backbone == "transformer":
                wandb_run_name = f"{wandb_run_name}_backbone-transformer" if wandb_run_name else "backbone-transformer"
            if args.unconditional_tau_in_state:
                wandb_run_name = f"{wandb_run_name}_uncond-qpt" if wandb_run_name else "uncond-qpt"
            if args.tau_loss_mask_prob > 0:
                mask_tag = f"tauMask{args.tau_loss_mask_prob:g}"
                wandb_run_name = f"{wandb_run_name}_{mask_tag}" if wandb_run_name else mask_tag
            if args.training_context_mode != "random_subset":
                ctx_tag = f"ctxMode-{args.training_context_mode}"
                wandb_run_name = f"{wandb_run_name}_{ctx_tag}" if wandb_run_name else ctx_tag
            if args.query_loss_decay != "none":
                decay_tag = f"qLoss-{args.query_loss_decay}{args.query_loss_decay_strength:g}"
                wandb_run_name = f"{wandb_run_name}_{decay_tag}" if wandb_run_name else decay_tag
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
                'compile_model': args.compile_model,
                'compile_mode': args.compile_mode,
                'num_latents': args.num_latents,
                'num_latent_channels': args.num_latent_channels,
                'num_decoder_blocks': args.num_decoder_blocks,
                'diffusion_steps': args.diffusion_steps,
                'epochs': args.epochs,
                'backbone': args.backbone,
                'unconditional_tau_in_state': args.unconditional_tau_in_state,
                'tau_loss_mask_prob': args.tau_loss_mask_prob,
                'training_context_mode': args.training_context_mode,
                'query_loss_decay': args.query_loss_decay,
                'query_loss_decay_strength': args.query_loss_decay_strength,
                'fixed_trajectory_length': args.fixed_trajectory_length,
                'trajectory_length_training_options': list(traj_length_options),
            })
            print(f"Initialized W&B logging: project={args.wandb_project}", flush=True)
    
    print("[Startup] Setting up callbacks...", flush=True)
    # Setup callbacks
    callbacks = []
    
    ablation_tag = f"_ablation-{args.ablation}" if args.ablation else ""
    backbone_tag = "_backbone-transformer" if args.backbone == "transformer" else ""
    unconditional_tag = "_uncond-qpt" if args.unconditional_tau_in_state else ""
    query_loss_tag = ""
    if args.query_loss_decay != "none":
        query_loss_tag = f"_qLoss-{args.query_loss_decay}{args.query_loss_decay_strength:g}"
    length_tag = f"FixedTrajLength{args.fixed_trajectory_length}" if args.fixed_trajectory_length else "VariableTrajLength"
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f'trajectory_dpf_x0Stabilized&AbsoluteTimeEncoding&{length_tag}&UniformContext&EncoderNone&DecoderAttentions{ablation_tag}{backbone_tag}{unconditional_tag}{query_loss_tag}:{{epoch:03d}}_val_loss:{{val_loss:.4f}}',
        every_n_epochs=args.checkpoint_every_n_epochs,
    )
    callbacks.append(checkpoint_callback)
    
    # Add W&B trajectory logging callback if W&B is enabled
    if args.wandb and WANDB_AVAILABLE and not args.disable_wandb_traj_callback:
        wandb_traj_callback = WandBTrajectoryCallback(
            log_every_n_epochs=1,  # TEMP: immediate callback verification after matplotlib backend fix
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
        and args.unconditional_tau_in_state
    ):
        # In unconditional Perceiver mode, torque-conditioning submodules are intentionally unused.
        # DDP needs find_unused_parameters=True to avoid bucket rebuild failures.
        trainer_strategy = "ddp_find_unused_parameters_true"
        print("[Trainer] Using strategy=ddp_find_unused_parameters_true for unconditional PerceiverIO multi-GPU.")

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
