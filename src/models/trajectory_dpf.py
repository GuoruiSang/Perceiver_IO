
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
from src.models.utils import EMA, visualize_trajectory, compare_generated_with_reconstructed
from scripts.dataset import TrajectoryDPFCached
from src import config
from src.training.utils import compute_normalization_stats

from src.models.architectures import TrajectoryOutputAdapter, TrajectoryPerceiverIO, ConditionedTrajectoryPerceiverIO
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
            
            # Generate sample trajectories (returns state, torque tuple)
            state, torque = pl_module.sample_trajectories(
                num_samples=self.num_samples,
                trajectory_length=min(1000, pl_module.max_timesteps),
                num_diffusion_steps=100,
                context_fraction=0.5,
                use_ema=True,
                sampler='ddim'
            )
            
            # Split state into components: [qpos | mom]
            qpos_dim = pl_module.qpos_dim
            mom_dim = pl_module.mom_dim
            
            state_traj = state[0]  # Take first sample [T, state_dim]
            torque_traj = torque[0]  # [T, torque_dim]
            
            trajectory_dict = {
                'seq_qpos': state_traj[:, :qpos_dim],
                'seq_mom': state_traj[:, qpos_dim:],
                'seq_torque': torque_traj,
            }
            
            # Create temporary directory for the plot
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Check if XML content is available for comparison plot
                if pl_module.xml_content is not None:
                    # Write XML to temp file for MuJoCo model loading
                    xml_path = os.path.join(tmp_dir, 'model.xml')
                    with open(xml_path, 'w') as f:
                        f.write(pl_module.xml_content)
                    
                    # Use comparison plot (generated vs physics-reconstructed)
                    compare_generated_with_reconstructed(
                        trajectory_dict, xml_path, tmp_dir,
                        dt=pl_module.dt, data_dt=pl_module.data_dt,
                        name='comparison'
                    )
                    plot_path = os.path.join(tmp_dir, 'comparison.jpg')
                else:
                    # Fallback to simple visualization if XML not available
                    print("[W&B] XML content not available, using simple visualization")
                    visualize_trajectory(trajectory_dict, tmp_dir)
                    plot_path = os.path.join(tmp_dir, 'trajectory.jpg')
                
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
        context_fraction_range: Tuple[float, float] = (0.3, 0.7),
        lr: float = 1e-4,
        use_ema: bool = True,
        p_uncond: float = 0.1,  # Probability of dropping conditioning for CFG
        lambda_cond: float = 0.1,  # Weight for conditioning regularization loss
        encoder_cond_mode: str = "mean",  # "mean" or "none" for encoder global conditioning
        # Simulation metadata (loaded from dataset)
        dt: float = 0.0001,  # Fine simulation timestep
        data_dt: float = 0.00025,  # Data collection timestep
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
        self.state_dim = qpos_dim + mom_dim  # State to denoise (qpos + mom)
        self.adaln_cond_dim = cond_dim  # Conditioning embedding dimension for AdaLN
        self.max_timesteps = max_timesteps
        self.diffusion_steps = diffusion_steps
        self.context_fraction_range = context_fraction_range
        self.lr = lr
        self.p_uncond = p_uncond  # CFG dropout probability
        self.lambda_cond = lambda_cond  # Conditioning regularization weight
        self.encoder_cond_mode = encoder_cond_mode
        
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
        
        # Total input channels per token: state + diffusion_enc + temporal_enc
        # Token structure: [qpos | mom | diffusion_enc | temporal_enc]
        # NOTE: Torque is NOT in tokens - it's passed separately for AdaLN conditioning
        num_input_channels_raw = self.state_dim + self.diffusion_encoding_channels + self.temporal_encoding_channels
        
        # Ensure num_input_channels is divisible by num_heads (8) for attention
        num_heads = 8
        if num_input_channels_raw % num_heads != 0:
            padding = num_heads - (num_input_channels_raw % num_heads)
            self.temporal_encoding_channels += padding
            print(f"[Init] Padded temporal_encoding_channels by {padding} to make num_input_channels divisible by {num_heads}")
        num_input_channels = self.state_dim + self.diffusion_encoding_channels + self.temporal_encoding_channels
        
        # PerceiverIO backbone with per-step state-torque interaction conditioning
        self.model = ConditionedTrajectoryPerceiverIO(
            num_input_channels=num_input_channels,
            num_output_channels=self.state_dim,  # Predict noise for (qpos, mom) only
            state_dim=self.state_dim,
            torque_dim=torque_dim,
            num_latents=num_latents,
            num_latent_channels=num_latent_channels,
            cond_dim=cond_dim,
            encoder_cond_mode=encoder_cond_mode,
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
        
        # State normalization: [qpos | mom]
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
        state: torch.Tensor,
        diffusion_t: int, 
        skip_normalize: bool = False,
    ) -> torch.Tensor:
        """
        Build tokens from state with explicit temporal position encoding.
        
        NOTE: Torque is NOT included in tokens - it's passed separately to the model
        for per-step AdaLN conditioning with state-torque interaction.
        
        Args:
            state: [B, T, state_dim] - batch of state trajectories (qpos, mom)
            diffusion_t: diffusion timestep (1 to diffusion_steps)
            skip_normalize: if True, assumes inputs are already in normalized space
        
        Returns:
            tokens: [B, T, C_in] where C_in = state + diffusion_enc + temporal_enc
        """
        B, T, _ = state.shape
        device = state.device
        
        # Normalize unless already in normalized space
        normalized_state = state if skip_normalize else self.normalize_state(state)
        
        # Diffusion timestep encoding (same for all timesteps)
        diffusion_enc = self.fpe_diffusion(B).to(device)[:, diffusion_t-1:diffusion_t, :]  # [B, 1, diff_enc_dim]
        diffusion_enc = diffusion_enc.expand(-1, T, -1)  # [B, T, diff_enc_dim]
        
        # Temporal position encoding: absolute sinusoidal (LENGTH-INDEPENDENT)
        # Timestep t always gets the same encoding regardless of total T
        # This is crucial for trajectory extension without performance degradation
        temporal_enc = self._get_temporal_encoding(T, device)  # [T, temp_enc_dim]
        temporal_enc = temporal_enc.unsqueeze(0).expand(B, -1, -1)  # [B, T, temp_enc_dim]

        # Concatenate: [state | diffusion_enc | temporal_enc]
        # NOTE: Torque is NOT included - passed separately for AdaLN conditioning
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
        
        # Replace state slice with noisy version
        tokens = tokens.clone()
        tokens[:, :, state_start:state_end] = noisy_state
        
        if return_noise:
            return tokens, noise
        return tokens
    
    def training_step(self, batch, batch_idx):
        """Training step with prefix context and CFG dropout."""
        # batch is a dict with keys: 'seq_qpos', 'seq_mom', 'seq_torque'
        qpos = batch['seq_qpos']  # [B, T, qpos_dim]
        mom = batch['seq_mom']    # [B, T, mom_dim]
        torque = batch['seq_torque']  # [B, T, torque_dim]
        
        # State: [qpos | mom], Conditioning: torque (passed separately for AdaLN)
        state = torch.cat([qpos, mom], dim=-1)  # [B, T, state_dim]
        
        B, T, _ = state.shape
        
        # Random diffusion timestep
        diffusion_t = torch.randint(1, self.diffusion_steps + 1, (1,)).item()
        
        # PREFIX context selection (not random) for extension capability
        context_fraction = torch.empty(1).uniform_(*self.context_fraction_range).item()
        num_context = max(1, min(T - 1, int(T * context_fraction)))
        
        # Build full token array once (NO torque in tokens), then apply noise once.
        # IMPORTANT: keep contexts as a slice of the (noisy) queries so training matches sampling
        tokens = self.build_tokens(state, diffusion_t)  # [B, T, C_in]
        noisy_tokens, noise = self.apply_noise(tokens, diffusion_t, return_noise=True)  # noise on state slice only

        noisy_queries = noisy_tokens
        noisy_contexts = noisy_tokens[:, :num_context, :]
        
        # Normalize torque for conditioning (passed separately to model)
        torque_norm = self.normalize_cond(torque)
        
        # ========== CFG DROPOUT (classifier-free guidance training) ==========
        # With probability p_uncond, train unconditionally by zeroing torque
        if torch.rand(1).item() < self.p_uncond:
            torque_norm = torch.zeros_like(torque_norm)

        # Predict noise (torque passed separately for per-step AdaLN conditioning)
        predictions = self.model(noisy_contexts, noisy_queries, torque_norm)
            
        # Main loss: predict noise
        loss_denoise = F.mse_loss(predictions, noise)
        
        loss = loss_denoise
        
        # Log training loss (step-level only)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log('denoise_loss', loss_denoise, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with prefix context (always conditional, no CFG dropout)."""
        qpos = batch['seq_qpos']
        mom = batch['seq_mom']
        torque = batch['seq_torque']
        
        # State: [qpos | mom], Conditioning: torque (passed separately for AdaLN)
        state = torch.cat([qpos, mom], dim=-1)
        
        B, T, _ = state.shape
        
        diffusion_t = torch.randint(1, self.diffusion_steps + 1, (1,)).item()
        
        # Build tokens (NO torque in tokens)
        tokens = self.build_tokens(state, diffusion_t)
        
        # Use fixed context fraction for validation - PREFIX context
        context_fraction = 0.5
        num_context = max(1, min(T - 1, int(T * context_fraction)))
        
        # Apply noise once, then slice PREFIX context from the same noisy token array.
        noisy_tokens, noise = self.apply_noise(tokens, diffusion_t, return_noise=True)
        noisy_queries = noisy_tokens  # [B, T, C_in]
        noisy_contexts = noisy_tokens[:, :num_context, :]  # [B, num_context, C_in]
        
        # Normalize torque for conditioning (passed separately)
        torque_norm = self.normalize_cond(torque)
        
        # Predict noise (torque passed separately for per-step AdaLN conditioning)
        predictions = self.model(noisy_contexts, noisy_queries, torque_norm)
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
    
    def _generate_random_torque(
        self, 
        num_samples: int, 
        trajectory_length: int, 
        dt: float,
        num_sin: int = 5,
        lim_amplitude: float = 0.5,
        lim_frequency: float = 6 * math.pi,
        lim_phase: float = 2 * math.pi
    ) -> torch.Tensor:
        """
        Generate random smooth torque sequences using sum of sinusoids.
        
        Parameters match generate_dataset_forward.py for consistency.
        
        Returns:
            torque: [num_samples, trajectory_length, torque_dim]
        """
        import numpy as np
        
        torque_dim = self.torque_dim
        device = self.device
        
        all_torques = []
        for _ in range(num_samples):
            amplitudes = np.random.uniform(0, lim_amplitude, (torque_dim, num_sin, 1))
            frequencies = np.random.uniform(0, lim_frequency, (torque_dim, num_sin, 1))
            phases = np.random.uniform(0, lim_phase, (torque_dim, num_sin, 1))
            
            steps = np.arange(trajectory_length) * dt
            steps = steps[None, None, :]
            steps = np.tile(steps, (torque_dim, num_sin, 1))
            
            torque = np.sum(amplitudes * np.sin(frequencies * steps + phases), axis=1).T
            all_torques.append(torque)
        
        return torch.tensor(np.array(all_torques), dtype=torch.float32, device=device)
    
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
        guidance_method: str = "adam",
        guidance_after_steps: int = 0,
        guidance_steps: int = 0,
        guidance_lr: float = 1e-3,
        langevin_step_size: float = 1e-5,
        langevin_noise_scale: float = 1e-6,
        lambda_init: float = 1.0,  # Weight for initial consistency term
        dt: Optional[float] = None,
        # Torque generation parameters
        torque: torch.Tensor = None,  # Optional: provide torque directly
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
            guidance_method: 'adam' or 'langevin'
            guidance_after_steps: start HNN guidance after this many steps
            guidance_steps: number of HNN optimization steps
            guidance_lr: learning rate for adam guidance
            langevin_step_size: step size for langevin guidance
            langevin_noise_scale: noise scale for langevin guidance
            lambda_init: weight for initial consistency term in HNN energy
            dt: timestep used to parameterize random torque generation (seconds between torque samples).
                If None, defaults to self.data_dt (dataset control timestep).
            torque: optional pre-generated torque [num_samples, trajectory_length, torque_dim]
        
        Returns:
            Tuple of:
                - state: [num_samples, trajectory_length, state_dim] (qpos, mom)
                - torque: [num_samples, trajectory_length, torque_dim]
        """
        from src.models.utils import run_adam_optimization_hnn, run_langevin_dynamics_hnn
        
        self.model.eval()
        
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
        
        # Generate or use provided torque conditioning
        if torque is None:
            # NOTE: use dataset control timestep by default so sampling torque matches training distribution
            torque_dt = float(self.data_dt) if dt is None else float(dt)
            print(f"[Sampling] Generating random torque sequences...")
            torque = self._generate_random_torque(num_samples, trajectory_length, torque_dt)
        else:
            torque = torque.to(device)
        
        # Start with pure noise for state (qpos, mom)
        x = torch.randn(num_samples, trajectory_length, self.state_dim, device=device)
        
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
        
        print(f"[Sampling] CFG guidance_scale={guidance_scale}")
        
        # PREFIX context: use first num_context timesteps (not random)
        # IMPORTANT: Cap context length to training max to avoid OOD encoder behavior when extending
        max_context_train = int(self.max_timesteps * context_fraction)
        num_context = max(1, min(trajectory_length - 1, max_context_train))
        print(f"[Sampling] Context length: {num_context} (capped at {max_context_train} from training length {self.max_timesteps})")

        for i, t in enumerate(tqdm(ts, total=len(ts), desc="Sampling")):
            t_int = int(t.item())
            
            # Build tokens (NO torque in tokens): [state | diffusion_enc | temporal_enc]
            queries = self.build_tokens(x, t_int + 1, skip_normalize=True)
            
            # PREFIX context: first num_context timesteps
            contexts = queries[:, :num_context, :]
            
            # Predict noise conditionally (torque passed separately for per-step AdaLN)
            with torch.no_grad():
                eps_cond = self.model(contexts, queries, cond)
            
            # CFG: if guidance_scale != 1.0, also predict unconditionally
            if guidance_scale != 1.0:
                # Predict with zeroed torque for unconditional
                with torch.no_grad():
                    eps_uncond = self.model(contexts, queries, cond_uncond)
                
                # CFG combination
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                
                # DEBUG: Print diagnostic info at key steps
                if i in [0, len(ts)//2, len(ts)-1]:
                    eps_diff = (eps_cond - eps_uncond).abs().mean().item()
                    print(f"[DEBUG] Step {i}, t={t_int}: eps_cond range=[{eps_cond.min():.4f}, {eps_cond.max():.4f}], "
                          f"eps_uncond range=[{eps_uncond.min():.4f}, {eps_uncond.max():.4f}], "
                          f"|eps_cond - eps_uncond| mean={eps_diff:.6f}")
            else:
                eps = eps_cond
                
                # DEBUG: Print diagnostic info at key steps  
                if i in [0, len(ts)//2, len(ts)-1]:
                    print(f"[DEBUG] Step {i}, t={t_int}: eps range=[{eps.min():.4f}, {eps.max():.4f}], "
                          f"eps mean={eps.mean():.4f}, eps std={eps.std():.4f}")
            
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
                
                # DEBUG: Check x0 temporal variation at key steps
                if i in [0, len(ts)//2, len(ts)-1]:
                    # Check if x0 varies across time (dim 1) - should NOT be constant
                    x0_time_std = x0.std(dim=1).mean().item()  # Std across timesteps, averaged over batch and dims
                    x0_dim_std = x0.std(dim=2).mean().item()   # Std across state dims
                    print(f"[DEBUG] Step {i}: x0 temporal_std={x0_time_std:.6f}, dim_std={x0_dim_std:.6f}, "
                          f"x0 range=[{x0.min():.4f}, {x0.max():.4f}]")
                
                # Apply HNN-based guidance
                if hnn is not None and guidance_steps > 0 and i >= guidance_after_steps:
                    # IMPORTANT:
                    # x0 lives in "normalized state space" (intended ~[-1, 1]) but can exceed that range.
                    # If we denormalize an unclamped x0, we can push qpos/mom far outside the training
                    # distribution, which explodes finite differences and HNN physics energy.
                    x0_phys = self.denormalize_state(x0)
                    
                    if guidance_method == "adam":
                        x0_phys = run_adam_optimization_hnn(
                            x0_phys, torque, self.qpos_dim, self.mom_dim,
                            self.data_dt, hnn, guidance_steps, guidance_lr, lambda_init=lambda_init
                        )
                    elif guidance_method == "langevin":
                        x0_phys = run_langevin_dynamics_hnn(
                            x0_phys, torque, self.qpos_dim, self.mom_dim,
                            self.data_dt, hnn, guidance_steps, langevin_step_size, langevin_noise_scale, lambda_init=lambda_init
                        )
                    x0 = self.normalize_state(x0_phys).detach()
                
                eps_coef = torch.sqrt(1.0 - a_bar_prev)
                x = torch.sqrt(a_bar_prev) * x0 + eps_coef * eps
                
            elif sampler == "ddpm_legacy":
                x0 = self._predict_x0(x_t, eps, a_bar_t)
                
                # Apply HNN-based guidance
                if hnn is not None and guidance_steps > 0 and i >= guidance_after_steps:
                    x0 = torch.clamp(x0, -1.0, 1.0)
                    x0_phys = self.denormalize_state(x0)
                    
                    if guidance_method == "adam":
                        x0_phys = run_adam_optimization_hnn(
                            x0_phys, torque, self.qpos_dim, self.mom_dim,
                            self.data_dt, hnn, guidance_steps, guidance_lr, lambda_init=lambda_init
                        )
                    elif guidance_method == "langevin":
                        x0_phys = run_langevin_dynamics_hnn(
                            x0_phys, torque, self.qpos_dim, self.mom_dim,
                            self.data_dt, hnn, guidance_steps, langevin_step_size, langevin_noise_scale, lambda_init=lambda_init
                        )
                    x0 = self.normalize_state(x0_phys).detach()
                
                sigma_t = self._compute_legacy_sigma_t(a_bar_t, a_bar_prev, i == len(ts) - 1)
                c = torch.sqrt(torch.clamp(1.0 - a_bar_prev - sigma_t * sigma_t, min=0.0))
                z = torch.randn_like(x_t) if (sigma_t.item() > 0.0) else torch.zeros_like(x_t)
                x = torch.sqrt(a_bar_prev) * x0 + c * eps + sigma_t * z
            
            if i == 0:
                print(f"[Sampling Debug] Step {i}: x range: [{x.min():.4f}, {x.max():.4f}]")
        
        # Denormalize state
        state = self.denormalize_state(x)
        print(f"[Sampling Debug] Final state range: [{state.min():.4f}, {state.max():.4f}]")
        
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
        if trajectory_length is None:
            trajectory_length = self.max_timesteps
        
        print(f"Generating {num_samples} trajectories of length {trajectory_length}...")
        state, torque = self.sample_trajectories(
            num_samples, trajectory_length, 
            guidance_after_steps=guidance_after_steps, 
            **sample_kwargs
        )
        
        # Move to CPU and convert to numpy
        state_np = state.cpu().numpy()
        torque_np = torque.cpu().numpy()
        
        # Split state into components: [qpos | mom]
        qpos = state_np[:, :, :self.qpos_dim]
        mom = state_np[:, :, self.qpos_dim:]
        
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
    parser.add_argument("--guidance_scale", type=float, default=4.0,
                        help="Classifier-free guidance scale (1.0 = no CFG, >1.0 = stronger conditioning)")
    parser.add_argument("--hnn_checkpoint", type=str, default='/home/gsang/Projects/Perceiver_IO/Projects/Perceiver_IO/checkpoints/SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt',
                        help="Path to HNN checkpoint for physics-based guidance during sampling")
    parser.add_argument("--guidance_method", type=str, choices=["adam", "langevin"], default="adam",
                        help="HNN guidance method: 'adam' or 'langevin'")
    parser.add_argument("--guidance_after_steps", type=int, default=100,
                        help="Start HNN guidance after this many diffusion steps (0 = from beginning)")
    parser.add_argument("--guidance_steps", type=int, default=10,
                        help="Number of HNN optimization steps per diffusion step (0 = disabled)")
    parser.add_argument("--guidance_lr", type=float, default=2e-2,
                        help="Learning rate for adam HNN guidance")
    parser.add_argument("--langevin_step_size", type=float, default=1e-5,
                        help="Step size for langevin HNN guidance")
    parser.add_argument("--langevin_noise_scale", type=float, default=1e-6,
                        help="Noise scale for langevin HNN guidance")
    parser.add_argument("--lambda_init", type=float, default=1.0,
                        help="Weight for initial consistency term in HNN energy")
    parser.add_argument("--seed", type=int, default=228,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--use_trained_torque", action="store_true", default=False,
                        help="Use torque sequences from training data instead of generating new random ones")
    
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
        if args.hnn_checkpoint and args.guidance_steps > 0:
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
        if args.guidance_steps > 0:
            print(f"[Sampling] Guidance enabled: method={args.guidance_method}, after_steps={args.guidance_after_steps}, "
                  f"steps={args.guidance_steps}, lr={args.guidance_lr}")
        
        # Set seed for reproducibility
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        # Load torque from training data if requested
        trained_torque = None
        if args.use_trained_torque:
            print(f"[Sampling] Loading torque sequences from training data: {args.h5_path}")
            dataset = TrajectoryDPFCached(args.h5_path, trajectory_length=trajectory_length)
            
            # Sample random trajectories from dataset and extract their torques
            import random
            indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
            torque_list = []
            for idx in indices:
                sample = dataset[idx]
                torque_list.append(sample['seq_torque'])
            
            trained_torque = torch.stack(torque_list).to(device)
            print(f"[Sampling] Loaded {len(torque_list)} torque sequences from training data")
            print(f"[Sampling] Torque shape: {trained_torque.shape}")
        
        state, torque = model.sample_trajectories(
            num_samples=args.num_samples,
            trajectory_length=trajectory_length,
            num_diffusion_steps=args.num_diffusion_steps,
            context_fraction=args.context_fraction,
            use_ema=args.use_ema,
            sampler=args.sampler,
            guidance_scale=args.guidance_scale,
            hnn=hnn if args.guidance_steps > 0 else None,
            guidance_method=args.guidance_method,
            guidance_after_steps=args.guidance_after_steps,
            guidance_steps=args.guidance_steps,
            guidance_lr=args.guidance_lr,
            langevin_step_size=args.langevin_step_size,
            langevin_noise_scale=args.langevin_noise_scale,
            lambda_init=args.lambda_init,
            torque=trained_torque,  # Use training data torque if --use_trained_torque is set
        )
        
        # Compare all trajectories with physics reconstruction
        # Build descriptive name with inference parameters for reproducibility
        torque_source = "trained" if args.use_trained_torque else "random"
        ema_str = "ema" if args.use_ema else "noema"
        
        # Build guidance string based on method
        if args.guidance_steps > 0:
            if args.guidance_method == "adam":
                guidance_str = f"hnn-{args.guidance_method}_after{args.guidance_after_steps}_steps{args.guidance_steps}_lr{args.guidance_lr}_lambda{args.lambda_init}"
            else:  # langevin
                guidance_str = f"hnn-{args.guidance_method}_after{args.guidance_after_steps}_steps{args.guidance_steps}_ss{args.langevin_step_size}_ns{args.langevin_noise_scale}_lambda{args.lambda_init}"
        else:
            guidance_str = "hnn-off"
        
        params_str = (
            f"seed{args.seed}_"
            f"{args.sampler}_"
            f"diff{args.num_diffusion_steps}_"
            f"ctx{args.context_fraction}_"
            f"{ema_str}_"
            f"cfg{args.guidance_scale}_"
            f"{guidance_str}_"
            f"len{trajectory_length}_"
            f"torque-{torque_source}"
        )
        
        print(f"[Sampling] Saving comparison plots for {args.num_samples} samples...")
        print(f"[Sampling] Plot name format: {params_str}_<idx>.jpg")
        for i in range(args.num_samples):
            state_np = state[i].cpu().numpy()
            torque_np = torque[i].cpu().numpy()
            generated = {
                'seq_qpos': state_np[:, :model.qpos_dim],
                'seq_mom': state_np[:, model.qpos_dim:],
                'seq_torque': torque_np,
            }
            compare_generated_with_reconstructed(
                generated, '/home/gsang/Projects/Perceiver_IO/configs/rigid_arm_hinge.xml',
                '/home/gsang/Projects/Perceiver_IO/plots',
                dt=float(model.dt),
                data_dt=float(model.data_dt),
                name=f'{params_str}_{i}'
            )
        print(f"[Sampling] Saved {args.num_samples} comparison plots to /home/gsang/Projects/Perceiver_IO/plots/")
        
        # Save to h5
        state_all = state.cpu().numpy()
        torque_all = torque.cpu().numpy()
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        import h5py
        with h5py.File(args.output_path, 'w') as f:
            f.attrs['num_trajectories'] = args.num_samples
            f.attrs['num_steps'] = trajectory_length
            for i in range(args.num_samples):
                g = f.create_group(f'traj_{i}')
                g.create_dataset('seq_qpos', data=state_all[i, :, :model.qpos_dim], dtype='f8')
                g.create_dataset('seq_mom', data=state_all[i, :, model.qpos_dim:], dtype='f8')
                g.create_dataset('seq_torque', data=torque_all[i], dtype='f8')
        
        print(f"Sample generation complete! Saved to {args.output_path}")
        return
    
    # Training mode - load dataset and setup training
    print(f"Loading dataset from {args.h5_path}...")
    dataset = TrajectoryDPFCached(args.h5_path, trajectory_length=1000)
    
    # Get dimensions from first sample
    sample = dataset[0]
    qpos_dim = sample['seq_qpos'].shape[-1]
    mom_dim = sample['seq_mom'].shape[-1]
    torque_dim = sample['seq_torque'].shape[-1]
    max_timesteps = dataset.num_steps
    
    # Get simulation metadata from dataset
    dt = dataset.dt
    data_dt = dataset.data_dt
    xml_content = dataset.xml
    
    print(f"Dataset info:")
    print(f"  Trajectories: {len(dataset)}")
    print(f"  Timesteps: {max_timesteps}")
    print(f"  qpos_dim: {qpos_dim}, mom_dim: {mom_dim}, torque_dim: {torque_dim}")
    print(f"  dt: {dt}, data_dt: {data_dt}")
    print(f"  XML content: {'loaded' if xml_content else 'not available'}")
    
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
    qpos_min, qpos_max, mom_min, mom_max, torque_min, torque_max = compute_normalization_stats(
        stats_loader, qpos_dim, mom_dim, torque_dim, max_timesteps
    )
    
    # Visualize one trajectory before and after normalization
    print("\n[Visualization] Saving trajectory before/after normalization...")
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
    os.makedirs('plots', exist_ok=True)
    visualize_trajectory(original_traj_dict, '/home/gsang/Projects/Perceiver_IO/plots')
    os.rename('/home/gsang/Projects/Perceiver_IO/plots/trajectory.jpg', '/home/gsang/Projects/Perceiver_IO/plots/trajectory_original.jpg')
    print("[Visualization] Saved /home/gsang/Projects/Perceiver_IO/plots/trajectory_original.jpg")
    
    # Normalize the trajectory
    # State: [qpos | mom]
    state_min = torch.cat([qpos_min, mom_min], dim=-1)
    state_max = torch.cat([qpos_max, mom_max], dim=-1)
    state_range = state_max - state_min
    
    full_state = torch.cat([sample_qpos, sample_mom], dim=-1)  # [T, state_dim]
    normalized_state = (full_state - state_min) / state_range * 2.0 - 1.0
    
    # Normalize torque separately
    cond_range = torque_max - torque_min
    normalized_torque = (sample_torque - torque_min) / cond_range * 2.0 - 1.0
    
    normalized_traj_dict = {
        'seq_qpos': normalized_state[:, :qpos_dim],
        'seq_mom': normalized_state[:, qpos_dim:],
        'seq_torque': normalized_torque,
    }
    
    # Save normalized trajectory
    visualize_trajectory(normalized_traj_dict, '/home/gsang/Projects/Perceiver_IO/plots')
    os.rename('/home/gsang/Projects/Perceiver_IO/plots/trajectory.jpg', '/home/gsang/Projects/Perceiver_IO/plots/trajectory_normalized.jpg')
    print("[Visualization] Saved /home/gsang/Projects/Perceiver_IO/plots/trajectory_normalized.jpg")
    print(f"[Visualization] Original state range: [{full_state.min():.4f}, {full_state.max():.4f}]")
    print(f"[Visualization] Normalized state range: [{normalized_state.min():.4f}, {normalized_state.max():.4f}]")

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
        lr=args.lr,
        encoder_cond_mode="mean",  # Global encoder conditioning: "mean" or "none"
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
        filename='trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&ContextLengthCap:{epoch:03d}_val_loss:{val_loss:.4f}',
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
        devices=[0,1],
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

