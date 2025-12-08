"""
Trajectory DPF: Diffusion Probabilistic Fields for Trajectory Generation

Each timestep is a token with:
- Value: (qpos, qvel, torque)
- Diffusion timestep encoding (which denoising step: 1→1000)
- Temporal position encoding (position in trajectory sequence: 0→T)

Training & Sampling (consistent approach):
- Query tokens: ALL timesteps with noised values
- Context tokens: Random subset of queries (can overlap)
- Predict noise for all queries using context subset
- Loss computed on all query predictions

Key insights:
- Context provides conditioning, queries are where we predict
- They don't need to be disjoint - overlap allows the model to learn
  self-consistency and matches the sampling procedure
- Temporal position encoding: Explicit Fourier encoding for each timestep's position
  in the trajectory sequence, providing the model with temporal ordering information

Sampling:
- Start with all timesteps as pure noise
- Iteratively denoise with random context/query splits using DDIM or DDPM Legacy
- DDIM: Deterministic denoising for faster sampling
- Save generated trajectories to h5 file
"""

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
from typing import Optional, Tuple
import argparse
from tqdm import tqdm
from einops import rearrange

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
from src.models.utils import EMA, visualize_trajectory, compare_generated_with_reconstructed, run_adam_optimization, run_langevin_dynamics
from scripts.dataset import TrajectoryDPFCached
from src import config
from src.training.utils import compute_normalization_stats

from src.models.architectures import TrajectoryOutputAdapter, TrajectoryPerceiverIO
import tempfile


# -------------------------
# W&B Trajectory Logger Callback
# -------------------------

class WandBTrajectoryCallback(pl.Callback):
    """Callback to log sampled trajectory visualizations to W&B during training."""
    
    def __init__(self, log_every_n_epochs: int = 10, num_samples: int = 1):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log sample trajectory after validation epoch."""
        current_epoch = trainer.current_epoch + 1
        
        # Debug: always print to confirm callback is running
        print(f"\n[W&B Callback] on_validation_epoch_end called (epoch {current_epoch})")
        
        if not WANDB_AVAILABLE:
            print("[W&B Callback] wandb not available, skipping")
            return
        
        if trainer.logger is None:
            print("[W&B Callback] trainer.logger is None, skipping")
            return
        
        # Only log every N epochs
        if current_epoch % self.log_every_n_epochs != 0:
            print(f"[W&B Callback] Skipping (epoch {current_epoch} % {self.log_every_n_epochs} != 0)")
            return
        
        try:
            print(f"[W&B] Generating sample trajectory for logging (epoch {current_epoch})...")
            
            # Generate sample trajectories
            trajectories = pl_module.sample_trajectories(
                num_samples=self.num_samples,
                trajectory_length=min(500, pl_module.max_timesteps),
                num_diffusion_steps=100,
                context_fraction=0.5,
                use_ema=True,
                sampler='ddim'
            )
            
            # Split into components (order: [torque | qacc | qvel | qpos])
            qpos_dim = pl_module.qpos_dim
            qvel_dim = pl_module.qvel_dim
            qacc_dim = pl_module.qacc_dim
            torque_dim = pl_module.torque_dim
            
            traj = trajectories[0]  # Take first sample [T, state_dim]
            trajectory_dict = {
                'seq_torque': traj[:, :torque_dim],
                'seq_qacc': traj[:, torque_dim:torque_dim + qacc_dim],
                'seq_qvel': traj[:, torque_dim + qacc_dim:torque_dim + qacc_dim + qvel_dim],
                'seq_qpos': traj[:, torque_dim + qacc_dim + qvel_dim:],
            }
            
            # Create temporary directory for the plot
            with tempfile.TemporaryDirectory() as tmp_dir:
                visualize_trajectory(trajectory_dict, tmp_dir)
                plot_path = os.path.join(tmp_dir, 'trajectory.jpg')
                
                # Log to wandb using the experiment directly
                print(f"[W&B] Logging image from: {plot_path}")
                wandb.log({
                    'sampled_trajectory': wandb.Image(plot_path),
                })
            
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
    
    Each timestep is a token: (qpos, qvel, torque, diffusion_t_enc, temporal_pos_enc)
    
    Temporal information is encoded explicitly via Fourier position encodings:
    - Diffusion timestep: Which denoising step (1→diffusion_steps)
    - Temporal position: Position in trajectory sequence (0→T)
    """
    
    def __init__(
        self,
        qpos_dim: int,
        qvel_dim: int,
        qacc_dim: int,
        torque_dim: int,
        max_timesteps: int = 1000,
        diffusion_steps: int = 1000,
        num_frequency_bands_for_diffusion: int = 64,
        num_latents: int = 256,
        num_latent_channels: int = 256,
        context_fraction_range: Tuple[float, float] = (0.3, 0.7),
        lr: float = 1e-4,
        use_ema: bool = True,
        # Min-max normalization stats (optional, will be computed if not provided)
        qpos_min: Optional[torch.Tensor] = None,
        qpos_max: Optional[torch.Tensor] = None,
        qvel_min: Optional[torch.Tensor] = None,
        qvel_max: Optional[torch.Tensor] = None,
        qacc_min: Optional[torch.Tensor] = None,
        qacc_max: Optional[torch.Tensor] = None,
        torque_min: Optional[torch.Tensor] = None,
        torque_max: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.qpos_dim = qpos_dim
        self.qvel_dim = qvel_dim
        self.qacc_dim = qacc_dim
        self.torque_dim = torque_dim
        self.state_dim = qpos_dim + qvel_dim + qacc_dim + torque_dim
        self.max_timesteps = max_timesteps
        self.diffusion_steps = diffusion_steps
        self.context_fraction_range = context_fraction_range
        self.lr = lr
        # Minimum allowed scale for any normalized dimension to avoid division blow-ups
        self.range_epsilon = config.DEFAULT_NORMALIZATION_RANGE_EPSILON
        
        # Fourier position encoding for diffusion timestep
        self.fpe_diffusion = FourierPositionEncoding(
            input_shape=(diffusion_steps,),
            num_frequency_bands=num_frequency_bands_for_diffusion
        )
        self.diffusion_encoding_channels = self.fpe_diffusion.num_position_encoding_channels()
        
        # Fourier position encoding for temporal position in trajectory
        self.num_temporal_frequency_bands = num_frequency_bands_for_diffusion // 2  # Use fewer bands for temporal
        self.fpe_temporal = FourierPositionEncoding(
            input_shape=(max_timesteps,),
            num_frequency_bands=self.num_temporal_frequency_bands
        )
        self.temporal_encoding_channels = self.fpe_temporal.num_position_encoding_channels()
        
        # Total input channels per token: state + diffusion_enc + temporal_enc
        # Token structure: [state (26) | diffusion_enc (129) | temporal_enc (65)] = 220 channels
        # - state: torque (6) + qacc (6) + qvel (6) + qpos (8) = 26  [order: torque | qacc | qvel | qpos]
        # - diffusion_enc: 2 * num_frequency_bands_for_diffusion + 1 = 2*64+1 = 129
        # - temporal_enc: 2 * (num_frequency_bands_for_diffusion // 2) + 1 = 2*32+1 = 65
        num_input_channels = self.state_dim + self.diffusion_encoding_channels + self.temporal_encoding_channels
        
        # PerceiverIO backbone
        self.model = TrajectoryPerceiverIO(
            num_input_channels=num_input_channels,
            num_output_channels=self.state_dim,  # Predict noise for (torque, qacc, qvel, qpos)
            num_latents=num_latents,
            num_latent_channels=num_latent_channels,
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
        self._setup_normalization(qpos_min, qpos_max, qvel_min, qvel_max, qacc_min, qacc_max, torque_min, torque_max)
        
        # EMA for better sampling
        if use_ema:
            self.ema = EMA(self.model, decay=0.9995)
        else:
            self.ema = None
        # Track whether EMA shadow was restored from checkpoint
        self._ema_loaded = False
    
    def _setup_normalization(self, qpos_min, qpos_max, qvel_min, qvel_max, qacc_min, qacc_max, torque_min, torque_max):
        """
        Setup min-max normalization parameters to scale all state components to [-1, 1].
        
        Normalization stats are per-dimension (global across time) with shape [dim].
        """
        # Default to no normalization if not provided
        if qpos_min is None:
            qpos_min = torch.zeros(self.qpos_dim) - 1
            qpos_max = torch.ones(self.qpos_dim)
        if qvel_min is None:
            qvel_min = torch.zeros(self.qvel_dim) - 1
            qvel_max = torch.ones(self.qvel_dim)
        if qacc_min is None:
            qacc_min = torch.zeros(self.qacc_dim) - 1
            qacc_max = torch.ones(self.qacc_dim)
        if torque_min is None:
            torque_min = torch.zeros(self.torque_dim) - 1
            torque_max = torch.ones(self.torque_dim)
        
        # Concatenate all normalization stats: [torque | qacc | qvel | qpos]
        # Shape: [state_dim]
        state_min = torch.cat([torque_min, qacc_min, qvel_min, qpos_min], dim=-1)
        state_max = torch.cat([torque_max, qacc_max, qvel_max, qpos_max], dim=-1)
        
        # Ensure minimum range for stability
        small_range_mask = (state_max - state_min) < self.range_epsilon
        state_max = torch.where(small_range_mask, state_min + self.range_epsilon, state_max)
        
        # Compute range for: x_norm = (x - min) / (max - min) * 2 - 1
        # Which maps [min, max] -> [-1, 1]
        state_range = state_max - state_min
        
        self.register_buffer('state_min', state_min.float())
        self.register_buffer('state_max', state_max.float())
        self.register_buffer('state_range', state_range.float())
    
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize state to [-1, 1] range using min-max scaling.
        
        Args:
            state: [B, T, state_dim] trajectories where state_dim = qpos_dim + qvel_dim + torque_dim
        
        Returns:
            normalized_state: [B, T, state_dim] with all components in [-1, 1]
        """
        # Normalize all state components: (x - min) / (max - min) * 2 - 1
        # state_min, state_max, state_range have shape [state_dim]
        # Broadcasting handles [B, T, dim] - [dim] correctly
        return (state - self.state_min) / self.state_range * 2.0 - 1.0
    
    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state from [-1, 1] back to original scale.
        
        Args:
            state: [B, T, state_dim] normalized trajectories with all components in [-1, 1]
        
        Returns:
            denormalized_state: [B, T, state_dim] in original scale
        """
        # Denormalize all state components: (x_norm + 1) / 2 * (max - min) + min
        # Maps [-1, 1] -> [min, max]
        return (state + 1.0) / 2.0 * self.state_range + self.state_min
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
    
    def build_tokens(
        self, 
        trajectories: torch.Tensor, 
        diffusion_t: int, 
        skip_normalize: bool = False,
    ) -> torch.Tensor:
        """
        Build tokens from trajectories with explicit temporal position encoding.
        
        Args:
            trajectories: [B, T, state_dim] - batch of trajectories
            diffusion_t: diffusion timestep (1 to diffusion_steps)
            skip_normalize: if True, assumes trajectories are already in normalized space
        
        Returns:
            tokens: [B, T, C_in] where C_in = state + diffusion_enc + temporal_enc
        """
        B, T, _ = trajectories.shape
        device = trajectories.device
        
        # Normalize trajectories unless already in normalized space
        normalized_traj = trajectories if skip_normalize else self.normalize_state(trajectories)
        
        # Diffusion timestep encoding (same for all timesteps)
        diffusion_enc = self.fpe_diffusion(B).to(device)[:, diffusion_t-1:diffusion_t, :]  # [B, 1, diff_enc_dim]
        diffusion_enc = diffusion_enc.expand(-1, T, -1)  # [B, T, diff_enc_dim]
        
        # Temporal position encoding (different for each timestep in trajectory)
        # Handle variable lengths: if T > max_timesteps, generate extended encoding
        if T > self.max_timesteps:
            # Create a temporary FourierPositionEncoding with the required length
            from perceiver.model.core import FourierPositionEncoding
            fpe_temporal_extended = FourierPositionEncoding(
                input_shape=(T,),
                num_frequency_bands=self.num_temporal_frequency_bands
            )
            temporal_enc = fpe_temporal_extended(B).to(device)[:, :T, :]  # [B, T, temp_enc_dim]
        else:
            temporal_enc = self.fpe_temporal(B).to(device)[:, :T, :]  # [B, T, temp_enc_dim]

        # Concatenate: [state | diffusion_enc | temporal_enc]
        tokens = torch.cat([normalized_traj, diffusion_enc, temporal_enc], dim=-1)
        
        return tokens
    
    def apply_noise(self, tokens: torch.Tensor, diffusion_t: int, return_noise: bool = False):
        """
        Apply noise to the state part of tokens.
        
        Args:
            tokens: [B, N, C_in] - tokens with structure [state | diffusion_enc | temporal_enc]
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
        
        # Replace state slice with noisy version
        tokens = tokens.clone()
        tokens[:, :, state_start:state_end] = noisy_state
        
        if return_noise:
            return tokens, noise
        return tokens
    
    def training_step(self, batch, batch_idx):
        """Training step with random context/query split."""
        # batch is a dict with keys: 'seq_qpos', 'seq_qvel', 'seq_qacc', 'seq_torque'
        qpos = batch['seq_qpos']  # [B, T, qpos_dim]
        qvel = batch['seq_qvel']  # [B, T, qvel_dim]
        qacc = batch['seq_qacc']  # [B, T, qacc_dim]
        torque = batch['seq_torque']  # [B, T, torque_dim]
        
        # Concatenate into full state: [B, T, 26] - order: [torque | qacc | qvel | qpos]
        trajectories = torch.cat([torque, qacc, qvel, qpos], dim=-1)
        
        B, T, _ = trajectories.shape
        
        # Random diffusion timestep
        diffusion_t = torch.randint(1, self.diffusion_steps + 1, (1,)).item()
        
        # Build tokens
        tokens = self.build_tokens(trajectories, diffusion_t)  # [B, T, C_in]
        
        # Random context/query split (following the sampling approach)
        # Queries = ALL tokens, Contexts = random subset of queries (can overlap)
        context_fraction = torch.empty(1).uniform_(*self.context_fraction_range).item()
        num_context = max(1, min(T, int(T * context_fraction)))
        
        # All tokens are queries (same as during sampling)
        queries = tokens  # [B, T, C_in]
        
        # Context is a random subset of queries (can overlap with loss computation)
        ctx_idx = torch.stack([torch.randperm(T, device=tokens.device)[:num_context] 
                               for _ in range(B)], dim=0)
        contexts = torch.gather(tokens, 1, ctx_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        
        # Apply noise to both context and query
        noisy_contexts = self.apply_noise(contexts, diffusion_t)
        noisy_queries, noise = self.apply_noise(queries, diffusion_t, return_noise=True)
        
        # Predict noise for queries
        predictions = self.model(noisy_contexts, noisy_queries)
        
        # Loss: predict noise only for queries
        loss = F.mse_loss(predictions, noise)
        
        # Log training loss (step-level only)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        qpos = batch['seq_qpos']
        qvel = batch['seq_qvel']
        qacc = batch['seq_qacc']
        torque = batch['seq_torque']
        trajectories = torch.cat([torque, qacc, qvel, qpos], dim=-1)  # [B, T, 26] - order: [torque | qacc | qvel | qpos]
        
        B, T, _ = trajectories.shape
        
        diffusion_t = torch.randint(1, self.diffusion_steps + 1, (1,)).item()
        tokens = self.build_tokens(trajectories, diffusion_t)
        
        # Use fixed context fraction for validation (matching sampling approach)
        context_fraction = 0.5
        num_context = max(1, min(T, int(T * context_fraction)))
        
        # All tokens are queries (same as training and sampling)
        queries = tokens  # [B, T, C_in]
        
        # Context is a random subset of queries
        ctx_idx = torch.stack([torch.randperm(T, device=tokens.device)[:num_context] 
                               for _ in range(B)], dim=0)
        contexts = torch.gather(tokens, 1, ctx_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        
        noisy_contexts = self.apply_noise(contexts, diffusion_t)
        noisy_queries, noise = self.apply_noise(queries, diffusion_t, return_noise=True)
        
        predictions = self.model(noisy_contexts, noisy_queries)
        loss = F.mse_loss(predictions, noise)
        
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
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA after each training batch."""
        if self.ema is not None:
            self.ema.update(self.model)
    
    def _predict_x0(self, x_t: torch.Tensor, eps: torch.Tensor, a_bar_t: torch.Tensor) -> torch.Tensor:
        """Predict x0 from noisy state x_t and predicted noise eps."""
        x0 = (x_t - torch.sqrt(1.0 - a_bar_t) * eps) / torch.sqrt(a_bar_t)
        # return torch.clamp(x0, -10, 10)
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
    
    def sample_trajectories(
        self,
        num_samples: int,
        trajectory_length: int,
        num_diffusion_steps: int = None,  # None means use full schedule for DDPM
        context_fraction: float = 0.7,
        use_ema: bool = True,
        resample_context_every_step: bool = True,
        sampler: str = "ddim",
        # Guidance parameters (adam or langevin, mutually exclusive)
        torque_predictor: nn.Module = None,
        guidance_method: str = "adam",  # "adam" or "langevin"
        guidance_after_steps: int = 0,  # Start guidance after this many diffusion steps (0 = always)
        guidance_steps: int = 0,  # Number of optimization steps per diffusion step
        guidance_lr: float = 1e-3,  # Learning rate for adam
        langevin_step_size: float = 1e-5,  # Step size for langevin
        langevin_noise_scale: float = 1e-6,  # Noise scale for langevin
        dt: float = 0.0005,  # Timestep for physics consistency
    ) -> torch.Tensor:
        """
        Sample trajectories using DDPM, DDIM, or Legacy DDPM.
        
        DDPM (default): Standard DDPM sampling with full schedule
            - Uses ALL diffusion steps (e.g., 1000 steps)
            - Update rule: x_{t-1} = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) * ε) + σ_t * z
            - Stochastic: adds noise at each step
            - Slower but more stable
        
        DDIM: Deterministic denoising with subsampling (Song et al. 2021)
            - CAN skip steps (T → T-k → T-2k → ... → 0)
            - Update rule: x_{t-1} = sqrt(ᾱ_{t-1}) * x_0 + sqrt(1-ᾱ_{t-1}) * ε
            - Deterministic: no noise injection
            - Much faster (e.g., only 50 steps instead of 1000)
        
        DDPM Legacy: Generalized DDIM with eta=1 (can subsample)
            - Based on DDIM formulation, so CAN subsample
            - Uses eta=1 for stochastic sampling
        
        Key insight: Query = ALL timesteps, Context = random subset of queries
        At each denoising step:
        - Queries: complete trajectory (all timesteps)
        - Contexts: random subset of those same timesteps
        - Predict noise for all queries using the context subset
        
        Args:
            num_samples: number of trajectories to generate
            trajectory_length: length of each trajectory
            num_diffusion_steps: number of denoising steps
                - For DDPM: defaults to full schedule (diffusion_steps)
                - For DDIM/legacy: can use fewer (e.g., 50 for speed)
            context_fraction: fraction of timesteps to use as context
            use_ema: whether to use EMA weights
            resample_context_every_step: whether to resample context at each step
            sampler: sampling method ('ddpm', 'ddim', or 'ddpm_legacy')
        
        Returns:
            trajectories: [num_samples, trajectory_length, state_dim]
        """
        self.model.eval()
        
        # Use EMA weights if available (EMA is updated after each training batch)
        if use_ema and self.ema is not None:
            print(f"[Sampling] Applying EMA weights for inference...")
            print(f"[Sampling] EMA has {len(self.ema.shadow)} shadow parameters")
            self.ema.store(self.model)
            self.ema.copy_to(self.model)
            print(f"[Sampling] ✓ EMA weights applied to model")
        elif use_ema and self.ema is None:
            print(f"[Sampling] WARNING: use_ema=True but no EMA available!")
            print(f"[Sampling] Using regular model weights instead")
        else:
            print(f"[Sampling] Using regular model weights (use_ema=False)")
        
        device = self.device
        
        # Start with pure noise
        x = torch.randn(num_samples, trajectory_length, self.state_dim, device=device)
        
        # Setup timesteps based on sampler
        if sampler == "ddpm":
            # DDPM uses full schedule (all steps from T-1 to 0)
            if num_diffusion_steps is None:
                num_diffusion_steps = self.diffusion_steps
            ts = torch.arange(self.diffusion_steps - 1, -1, -1, device=device, dtype=torch.long)
            print(f"[Sampling] Using DDPM sampler with {len(ts)} steps (full schedule)")
        elif sampler == "ddim":
            # DDIM: CAN use subsampled schedule (DDIM was designed for this!)
            if num_diffusion_steps is None:
                num_diffusion_steps = 50  # Default for DDIM
            ts = torch.linspace(self.diffusion_steps - 1, 0, steps=num_diffusion_steps, device=device, dtype=torch.long)
            print(f"[Sampling] Using DDIM sampler with {len(ts)} subsampled steps")
        elif sampler == "ddpm_legacy":
            # Legacy: generalized DDIM with eta=1 (can subsample like DDIM)
            if num_diffusion_steps is None:
                num_diffusion_steps = 50  # Default for legacy
            ts = torch.linspace(self.diffusion_steps - 1, 0, steps=num_diffusion_steps, device=device, dtype=torch.long)
            print(f"[Sampling] Using Legacy sampler (generalized DDIM, eta=1) with {len(ts)} subsampled steps")
        else:
            raise ValueError(f"Unknown sampler: {sampler}. Choose 'ddpm', 'ddim', or 'ddpm_legacy'.")
        
        num_context = max(1, min(trajectory_length, int(trajectory_length * context_fraction)))
        
        # Generate context indices once if not resampling
        if not resample_context_every_step:
            ctx_idx = torch.stack([torch.randperm(trajectory_length, device=device)[:num_context] 
                                   for _ in range(num_samples)], dim=0)

        for i, t in enumerate(tqdm(ts, total=len(ts), desc="Sampling")):
            # Build tokens from current noisy state - ALL timesteps are queries
            t_int = int(t.item())
            queries = self.build_tokens(x, t_int + 1, skip_normalize=True)
            # queries: [B, T, C] - complete trajectory
            
            # Resample context at each step if requested
            if resample_context_every_step:
                ctx_idx = torch.stack([torch.randperm(trajectory_length, device=device)[:num_context]
                                       for _ in range(num_samples)], dim=0)
            
            # Context is a subset of queries
            contexts = torch.gather(queries, 1, ctx_idx.unsqueeze(-1).expand(-1, -1, queries.shape[-1]))
            # contexts: [B, num_context, C] - subset of trajectory
            
            # Predict noise for ALL queries using context subset (no grad to save memory)
            with torch.no_grad():
                eps = self.model(contexts, queries)  # eps: [B, T, state_dim]
            
            # Extract state from queries (at the beginning of token structure)
            state_start = 0
            state_end = self.state_dim
            x_t = queries[:, :, state_start:state_end]  # [B, T, state_dim]
            
            # Get diffusion parameters (cumulative alphas)
            a_bar_t = self.alpha_cumprod[t_int]
            if i < len(ts) - 1:
                t_prev_int = int(ts[i + 1].item())
                a_bar_prev = self.alpha_cumprod[t_prev_int]
            else:
                # Final step brings us to t=0 -> set a_bar_prev = 1 for x_0
                a_bar_prev = a_bar_t.new_tensor(1.0)

            # Clamp alpha values to prevent numerical issues
            a_bar_t = torch.clamp(a_bar_t, min=1e-6, max=1.0)
            a_bar_prev = torch.clamp(a_bar_prev, min=1e-6, max=1.0)

            # Branch based on sampler type
            if sampler == "ddpm":
                # Standard DDPM update rule
                # x_{t-1} = (1/√α_t) * (x_t - β_t/√(1-ᾱ_t) * ε) + σ_t * z
                alpha_t = self.alphas[t_int]
                beta_t = self.betas[t_int]
                
                # Compute mean
                coef1 = 1.0 / torch.sqrt(alpha_t)
                coef2 = beta_t / torch.sqrt(1.0 - a_bar_t)
                mean = coef1 * (x_t - coef2 * eps)
                
                # Add noise (except at final step)
                if t_int > 0:
                    # sigma_t = sqrt(beta_t) or sqrt(beta_t * (1-alpha_bar_{t-1})/(1-alpha_bar_t))
                    sigma_t = torch.sqrt(beta_t)
                    z = torch.randn_like(x_t)
                    x = mean + sigma_t * z
                else:
                    x = mean
                
                if i == 0:
                    print(f"[Sampling Debug] Step {i}: mean range: [{mean.min():.4f}, {mean.max():.4f}]")
                
            elif sampler == "ddim":
                # DDIM update rule (deterministic, faster)
                x0 = self._predict_x0(x_t, eps, a_bar_t)
                
                # Apply guidance (adam or langevin) after guidance_after_steps
                if torque_predictor is not None and guidance_steps > 0 and i >= guidance_after_steps:
                    x0_phys = self.denormalize_state(x0)
                    if guidance_method == "adam":
                        x0_phys = run_adam_optimization(
                            x0_phys, self.torque_dim, self.qacc_dim, self.qvel_dim,
                            dt, torque_predictor, guidance_steps, guidance_lr
                        )
                    elif guidance_method == "langevin":
                        x0_phys = run_langevin_dynamics(
                            x0_phys, self.torque_dim, self.qacc_dim, self.qvel_dim,
                            dt, torque_predictor, guidance_steps, langevin_step_size, langevin_noise_scale
                        )
                    x0 = self.normalize_state(x0_phys).detach()
                
                if i == 0:
                    print(f"[Sampling Debug] Step {i}: x0 after prediction - range: [{x0.min():.4f}, {x0.max():.4f}], has NaN: {torch.isnan(x0).any()}")
                
                # DDIM is deterministic (eta=0)
                eps_coef = torch.sqrt(1.0 - a_bar_prev)
                x = torch.sqrt(a_bar_prev) * x0 + eps_coef * eps
                
            elif sampler == "ddpm_legacy":
                # Legacy DDPM: generalized DDIM with eta=1 (matches training code)
                x0 = self._predict_x0(x_t, eps, a_bar_t)
                
                # Apply guidance (adam or langevin) after guidance_after_steps
                if torque_predictor is not None and guidance_steps > 0 and i >= guidance_after_steps:
                    x0_phys = self.denormalize_state(x0)
                    if guidance_method == "adam":
                        x0_phys = run_adam_optimization(
                            x0_phys, self.torque_dim, self.qacc_dim, self.qvel_dim,
                            dt, torque_predictor, guidance_steps, guidance_lr
                        )
                    elif guidance_method == "langevin":
                        x0_phys = run_langevin_dynamics(
                            x0_phys, self.torque_dim, self.qacc_dim, self.qvel_dim,
                            dt, torque_predictor, guidance_steps, langevin_step_size, langevin_noise_scale
                        )
                    x0 = self.normalize_state(x0_phys).detach()
                
                if i == 0:
                    print(f"[Sampling Debug] Step {i}: x0 after prediction - range: [{x0.min():.4f}, {x0.max():.4f}], has NaN: {torch.isnan(x0).any()}")
                
                sigma_t = self._compute_legacy_sigma_t(a_bar_t, a_bar_prev, i == len(ts) - 1)
                c = torch.sqrt(torch.clamp(1.0 - a_bar_prev - sigma_t * sigma_t, min=0.0))
                
                z = torch.randn_like(x_t) if (sigma_t.item() > 0.0) else torch.zeros_like(x_t)
                x = torch.sqrt(a_bar_prev) * x0 + c * eps + sigma_t * z
            
            if i == 0:
                print(f"[Sampling Debug] Step {i}: x after update - range: [{x.min():.4f}, {x.max():.4f}], has NaN: {torch.isnan(x).any()}")
        
        # Denormalize
        print(f"[Sampling Debug] x before denormalize - range: [{x.min():.4f}, {x.max():.4f}], has NaN: {torch.isnan(x).any()}")
        trajectories = self.denormalize_state(x)
        print(f"[Sampling Debug] trajectories after denormalize - range: [{trajectories.min():.4f}, {trajectories.max():.4f}], has NaN: {torch.isnan(trajectories).any()}")
        
        # No post-processing needed - all dimensions treated equally
        
        # Restore original weights if EMA was applied
        if use_ema and self.ema is not None:
            self.ema.restore(self.model)
        
        self.model.train()
        return trajectories
    
    def save_trajectories_to_h5(
        self,
        output_path: str,
        num_samples: int = 100,
        trajectory_length: int = None,
        guidance_after_steps: int = 0,
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
        # Default to training trajectory length if not specified
        if trajectory_length is None:
            trajectory_length = self.max_timesteps
        
        print(f"Generating {num_samples} trajectories of length {trajectory_length}...")
        trajectories = self.sample_trajectories(
            num_samples, trajectory_length, 
            guidance_after_steps=guidance_after_steps, 
            **sample_kwargs
        )
        
        # Move to CPU and convert to numpy
        trajectories = trajectories.cpu().numpy()
        
        # Split into components (order: [torque | qacc | qvel | qpos])
        torque = trajectories[:, :, :self.torque_dim]
        qacc = trajectories[:, :, self.torque_dim:self.torque_dim + self.qacc_dim]
        qvel = trajectories[:, :, self.torque_dim + self.qacc_dim:self.torque_dim + self.qacc_dim + self.qvel_dim]
        qpos = trajectories[:, :, self.torque_dim + self.qacc_dim + self.qvel_dim:]
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        
        print(f"Saving to {output_path}...")
        with h5py.File(output_path, 'w') as f:
            # Match training format: root attrs
            f.attrs['num_trajectories'] = num_samples
            f.attrs['num_steps'] = trajectory_length
            
            # Match training format: traj_{i}/seq_* structure
            for i in range(num_samples):
                traj_group = f.create_group(f'traj_{i}')
                traj_group.create_dataset('seq_qpos', data=qpos[i], dtype='f8')
                traj_group.create_dataset('seq_qvel', data=qvel[i], dtype='f8')
                traj_group.create_dataset('seq_qacc', data=qacc[i], dtype='f8')
                traj_group.create_dataset('seq_torque', data=torque[i], dtype='f8')
        
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
    parser.add_argument("--checkpoint_dir", type=str, default=config.DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume_from_checkpoint", type=str, default=config.DEFAULT_RESUME_CHECKPOINT, 
                        help="Path to checkpoint to resume training from (for train mode) or to load for generation (for generate_samples mode)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=config.DEFAULT_NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LEARNING_RATE)
    parser.add_argument("--num_latents", type=int, default=config.DEFAULT_NUM_LATENTS)
    parser.add_argument("--num_latent_channels", type=int, default=config.DEFAULT_NUM_LATENT_CHANNELS)
    parser.add_argument("--diffusion_steps", type=int, default=config.DEFAULT_DIFFUSION_STEPS)
    
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=config.DEFAULT_NUM_SAMPLES, help="Number of trajectories to generate")
    parser.add_argument("--trajectory_length", type=int, default=500,
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
    
    # Guidance and post-processing
    parser.add_argument("--hnn_checkpoint", type=str, default='/home/gsang/Projects/Perceiver_IO/checkpoints/SeperableHNN-Tanh-epoch-epoch=459.ckpt',
                        help="Path to HNN checkpoint for guidance during sampling and/or torque correction after sampling")
    parser.add_argument("--guidance_method", type=str, choices=["adam", "langevin"], default="adam",
                        help="Guidance method: 'adam' or 'langevin' (mutually exclusive)")
    parser.add_argument("--guidance_after_steps", type=int, default=0,
                        help="Start guidance after this many diffusion steps (0 = from the beginning)")
    parser.add_argument("--guidance_steps", type=int, default=10,
                        help="Number of optimization steps per diffusion step (0 = disabled)")
    parser.add_argument("--guidance_lr", type=float, default=1e-2,
                        help="Learning rate for adam guidance")
    parser.add_argument("--langevin_step_size", type=float, default=1e-5,
                        help="Step size for langevin guidance")
    parser.add_argument("--langevin_noise_scale", type=float, default=1e-6,
                        help="Noise scale for langevin guidance")
    parser.add_argument("--correct_torque", action="store_true",
                        help="Replace generated torque with physics-consistent torque after sampling")
    parser.add_argument("--seed", type=int, default=11,
                        help="Random seed for reproducible sampling")
    
    # W&B arguments
    parser.add_argument("--wandb", type=bool, default=config.DEFAULT_WANDB_ENABLED, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=config.DEFAULT_WANDB_PROJECT, help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (username or team)")
    parser.add_argument("--wandb_run_name", type=str, default=config.DEFAULT_WANDB_RUN_NAME, help="W&B run name")
    
    args = parser.parse_args()
    
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
        print(f"[Sampling] Model dimensions: qpos={model.qpos_dim}, qvel={model.qvel_dim}, "
              f"qacc={model.qacc_dim}, torque={model.torque_dim}")
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
        
        # Load torque predictor if needed for guidance or post-processing
        torque_predictor = None
        if args.hnn_checkpoint and (args.guidance_steps > 0 or args.correct_torque):
            from src.models.HNN import TorquePredictor
            print(f"[Sampling] Loading torque predictor from: {args.hnn_checkpoint}")
            hnn_ckpt = torch.load(args.hnn_checkpoint, map_location=device)
            hnn_state = {k.replace('torque_predictor.', ''): v 
                         for k, v in hnn_ckpt['state_dict'].items() if k.startswith('torque_predictor.')}
            torque_predictor = TorquePredictor(coordinate_dim=model.qpos_dim).to(device)
            torque_predictor.load_state_dict(hnn_state)
            torque_predictor.eval()
            print(f"[Sampling] ✓ Torque predictor loaded")
        
        # Determine trajectory length (default to training length if not specified)
        trajectory_length = args.trajectory_length if args.trajectory_length is not None else model.max_timesteps
        
        # Generate samples
        print(f"\nGenerating {args.num_samples} sample trajectories of length {trajectory_length}...")
        print(f"[Sampling] Training trajectory length: {model.max_timesteps}")
        if trajectory_length != model.max_timesteps:
            print(f"[Sampling] Extending trajectory length to: {trajectory_length}")
        print(f"[Sampling] Using sampler={args.sampler}, num_diffusion_steps={args.num_diffusion_steps}, "
              f"context_fraction={args.context_fraction}, use_ema={args.use_ema}, seed={args.seed}")
        if args.guidance_steps > 0:
            print(f"[Sampling] Guidance enabled: method={args.guidance_method}, after_steps={args.guidance_after_steps}, "
                  f"steps={args.guidance_steps}, lr={args.guidance_lr}")
        
        # Set seed for reproducibility
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        trajectories = model.sample_trajectories(
            num_samples=args.num_samples,
            trajectory_length=trajectory_length,
            num_diffusion_steps=args.num_diffusion_steps,
            context_fraction=args.context_fraction,
            use_ema=args.use_ema,
            sampler=args.sampler,
            torque_predictor=torque_predictor if args.guidance_steps > 0 else None,
            guidance_method=args.guidance_method,
            guidance_after_steps=args.guidance_after_steps,
            guidance_steps=args.guidance_steps,
            guidance_lr=args.guidance_lr,
            langevin_step_size=args.langevin_step_size,
            langevin_noise_scale=args.langevin_noise_scale,
        )
        
        # Post-process: Replace torque with physics-consistent torque
        if args.correct_torque and torque_predictor is not None:
            with torch.no_grad():
                qpos = trajectories[:, :, model.torque_dim + model.qacc_dim + model.qvel_dim:]
                qvel = trajectories[:, :, model.torque_dim + model.qacc_dim:model.torque_dim + model.qacc_dim + model.qvel_dim]
                qacc = trajectories[:, :, model.torque_dim:model.torque_dim + model.qacc_dim]
                corrected_torque = torque_predictor(qpos, qvel, qacc)
                trajectories[:, :, :model.torque_dim] = corrected_torque
            print(f"[Post-process] ✓ Torque corrected using HNN")
        
        # Compare first trajectory with physics reconstruction
        traj_np = trajectories[0].cpu().numpy()
        generated = {
            'seq_torque': traj_np[:, :model.torque_dim],
            'seq_qacc': traj_np[:, model.torque_dim:model.torque_dim + model.qacc_dim],
            'seq_qvel': traj_np[:, model.torque_dim + model.qacc_dim:model.torque_dim + model.qacc_dim + model.qvel_dim],
            'seq_qpos': traj_np[:, model.torque_dim + model.qacc_dim + model.qvel_dim:],
        }
        compare_generated_with_reconstructed(
            generated, '/home/gsang/Projects/Perceiver_IO/configs/rigid_arm_hinge.xml',
            '/home/gsang/Projects/Perceiver_IO/plots'
        )
        
        # Save to h5
        trajectories_np = trajectories.cpu().numpy()
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        import h5py
        with h5py.File(args.output_path, 'w') as f:
            f.attrs['num_trajectories'] = args.num_samples
            f.attrs['num_steps'] = trajectory_length
            for i in range(args.num_samples):
                g = f.create_group(f'traj_{i}')
                g.create_dataset('seq_torque', data=trajectories_np[i, :, :model.torque_dim], dtype='f8')
                g.create_dataset('seq_qacc', data=trajectories_np[i, :, model.torque_dim:model.torque_dim + model.qacc_dim], dtype='f8')
                g.create_dataset('seq_qvel', data=trajectories_np[i, :, model.torque_dim + model.qacc_dim:model.torque_dim + model.qacc_dim + model.qvel_dim], dtype='f8')
                g.create_dataset('seq_qpos', data=trajectories_np[i, :, model.torque_dim + model.qacc_dim + model.qvel_dim:], dtype='f8')
        
        print(f"Sample generation complete! Saved to {args.output_path}")
        return
    
    # Training mode - load dataset and setup training
    print(f"Loading dataset from {args.h5_path}...")
    dataset = TrajectoryDPFCached(args.h5_path)
    
    # Get dimensions from first sample
    sample = dataset[0]
    qpos_dim = sample['seq_qpos'].shape[-1]
    qvel_dim = sample['seq_qvel'].shape[-1]
    qacc_dim = sample['seq_qacc'].shape[-1]
    torque_dim = sample['seq_torque'].shape[-1]
    max_timesteps = dataset.num_steps
    
    print(f"Dataset info:")
    print(f"  Trajectories: {len(dataset)}")
    print(f"  Timesteps: {max_timesteps}")
    print(f"  qpos_dim: {qpos_dim}, qvel_dim: {qvel_dim}, qacc_dim: {qacc_dim}, torque_dim: {torque_dim}")
    
    # Create dataloaders
    train_size = int(config.DEFAULT_TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    
    # Compute normalization stats (min-max for scaling to [-1, 1])
    stats_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    qpos_min, qpos_max, qvel_min, qvel_max, qacc_min, qacc_max, torque_min, torque_max = compute_normalization_stats(
        stats_loader, qpos_dim, qvel_dim, qacc_dim, torque_dim, max_timesteps
    )
    
    # Visualize one trajectory before and after normalization
    print("\n[Visualization] Saving trajectory before/after normalization...")
    sample = dataset[0]
    sample_qpos = sample['seq_qpos']  # [T, qpos_dim]
    sample_qvel = sample['seq_qvel']  # [T, qvel_dim]
    sample_qacc = sample['seq_qacc']  # [T, qacc_dim]
    sample_torque = sample['seq_torque']  # [T, torque_dim]
    
    # Original trajectory dict
    original_traj_dict = {
        'seq_qpos': sample_qpos,
        'seq_qvel': sample_qvel,
        'seq_qacc': sample_qacc,
        'seq_torque': sample_torque,
    }
    
    # Save original trajectory
    os.makedirs('plots', exist_ok=True)
    visualize_trajectory(original_traj_dict, '/home/gsang/Projects/Perceiver_IO/plots')
    os.rename('/home/gsang/Projects/Perceiver_IO/plots/trajectory.jpg', '/home/gsang/Projects/Perceiver_IO/plots/trajectory_original.jpg')
    print("[Visualization] Saved /home/gsang/Projects/Perceiver_IO/plots/trajectory_original.jpg")
    
    # Normalize the trajectory
    # Concatenate normalization stats (order: [torque | qacc | qvel | qpos])
    state_min = torch.cat([torque_min, qacc_min, qvel_min, qpos_min], dim=-1)
    state_max = torch.cat([torque_max, qacc_max, qvel_max, qpos_max], dim=-1)
    state_range = state_max - state_min
    
    # Concatenate trajectory (order: [torque | qacc | qvel | qpos])
    full_state = torch.cat([sample_torque, sample_qacc, sample_qvel, sample_qpos], dim=-1)  # [T, state_dim]
    
    # Normalize: (x - min) / range * 2 - 1
    normalized_state = (full_state - state_min) / state_range * 2.0 - 1.0
    
    # Split back into components for visualization (order: [torque | qacc | qvel | qpos])
    normalized_traj_dict = {
        'seq_torque': normalized_state[:, :torque_dim],
        'seq_qacc': normalized_state[:, torque_dim:torque_dim + qacc_dim],
        'seq_qvel': normalized_state[:, torque_dim + qacc_dim:torque_dim + qacc_dim + qvel_dim],
        'seq_qpos': normalized_state[:, torque_dim + qacc_dim + qvel_dim:],
    }
    
    # Save normalized trajectory
    visualize_trajectory(normalized_traj_dict, '/home/gsang/Projects/Perceiver_IO/plots')
    os.rename('/home/gsang/Projects/Perceiver_IO/plots/trajectory.jpg', '/home/gsang/Projects/Perceiver_IO/plots/trajectory_normalized.jpg')
    print("[Visualization] Saved /home/gsang/Projects/Perceiver_IO/plots/trajectory_normalized.jpg")
    print(f"[Visualization] Original range: [{full_state.min():.4f}, {full_state.max():.4f}]")
    print(f"[Visualization] Normalized range: [{normalized_state.min():.4f}, {normalized_state.max():.4f}]")

    # Create model
    model = TrajectoryDPF(
        qpos_dim=qpos_dim,
        qvel_dim=qvel_dim,
        qacc_dim=qacc_dim,
        torque_dim=torque_dim,
        max_timesteps=max_timesteps,
        diffusion_steps=args.diffusion_steps,
        num_latents=args.num_latents,
        num_latent_channels=args.num_latent_channels,
        lr=args.lr,
        qpos_min=qpos_min,
        qpos_max=qpos_max,
        qvel_min=qvel_min,
        qvel_max=qvel_max,
        qacc_min=qacc_min,
        qacc_max=qacc_max,
        torque_min=torque_min,
        torque_max=torque_max,
    )
    
    # Training mode - setup W&B logger
    logger = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb is not installed. Run 'pip install wandb' to enable W&B logging.")
            print("Continuing without W&B logging.")
        else:
            from pytorch_lightning.loggers import WandbLogger
            logger = WandbLogger(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                save_dir=args.checkpoint_dir,
                log_model=True,  # Log model checkpoints to W&B
            )
            
            # Log dataset and training info as hyperparameters
            logger.log_hyperparams({
                'dataset_path': args.h5_path,
                'num_trajectories': len(dataset),
                'trajectory_length': max_timesteps,
                'qpos_dim': qpos_dim,
                'qvel_dim': qvel_dim,
                'qacc_dim': qacc_dim,
                'torque_dim': torque_dim,
                'batch_size': args.batch_size,
                'num_workers': args.num_workers,
                'lr': args.lr,
                'num_latents': args.num_latents,
                'num_latent_channels': args.num_latent_channels,
                'diffusion_steps': args.diffusion_steps,
                'epochs': args.epochs,
            })
            print(f"Initialized W&B logging: project={args.wandb_project}")
    
    # Setup callbacks
    callbacks = []
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='trajectory_dpf_foward:{epoch:03d}_val_loss:{val_loss:.4f}',
        every_n_epochs=10,  # Save checkpoint every 10 epochs
    )
    callbacks.append(checkpoint_callback)
    
    # Add W&B trajectory logging callback if W&B is enabled
    if args.wandb and WANDB_AVAILABLE:
        wandb_traj_callback = WandBTrajectoryCallback(
            log_every_n_epochs=10,  # Log trajectory every 10 epochs
            num_samples=1,
        )
        callbacks.append(wandb_traj_callback)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        # gradient_clip_val=1.0,
        log_every_n_steps=10,
    )
    
    # Train
    print("Starting training...")
    ckpt_path = None
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        # Option 1: Let Lightning restore full training state (epoch, optimizer, schedulers)
        ckpt_path = args.resume_from_checkpoint
        
        # Additionally ensure model weights can load even if there are benign mismatches
        try:
            checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
            model.load_state_dict(checkpoint.get('state_dict', {}), strict=False)
        except Exception as e:
            print(f"[Training] Non-strict model weight preload skipped due to: {e}")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    print("Training complete!")


if __name__ == "__main__":
    main()

