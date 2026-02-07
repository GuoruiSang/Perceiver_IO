
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

from src.models.architectures import TrajectoryOutputAdapter, TrajectoryPerceiverIO, ConditionedTrajectoryPerceiverIO, AblationConfig
import tempfile
import numpy as np


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
        num_decoder_blocks: int = 0,  # NEW: Number of decoder self-attention blocks
        trajectory_length_training_options: Tuple[int, ...] = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000),
        lr: float = 1e-4,
        use_ema: bool = True,
        p_uncond: float = 0.1,  # Probability of dropping conditioning for CFG
        lambda_cond: float = 0.1,  # Weight for conditioning regularization loss
        encoder_cond_mode: str = "none",  # "per_step", "mean", "rnn" or "none" for encoder conditioning
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
        self.state_dim = qpos_dim + mom_dim  # State to denoise (qpos + mom)
        self.adaln_cond_dim = cond_dim  # Conditioning embedding dimension for AdaLN
        self.num_decoder_blocks = num_decoder_blocks
        self.max_timesteps = max_timesteps
        self.diffusion_steps = diffusion_steps
        self.trajectory_length_training_options = trajectory_length_training_options
        self.lr = lr
        self.p_uncond = p_uncond  # CFG dropout probability
        self.lambda_cond = lambda_cond  # Conditioning regularization weight
        self.encoder_cond_mode = encoder_cond_mode
        self.ablation_config = ablation_config if ablation_config is not None else AblationConfig()

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
        # NOTE: Torque is NOT in tokens unless torque_in_tokens ablation is active
        num_input_channels_raw = self.state_dim + self.diffusion_encoding_channels + self.temporal_encoding_channels
        if self.ablation_config.torque_in_tokens:
            num_input_channels_raw += self.torque_dim
        
        # Ensure num_input_channels is divisible by num_heads (8) for attention
        num_heads = 8
        if num_input_channels_raw % num_heads != 0:
            padding = num_heads - (num_input_channels_raw % num_heads)
            self.temporal_encoding_channels += padding
            print(f"[Init] Padded temporal_encoding_channels by {padding} to make num_input_channels divisible by {num_heads}")
        num_input_channels = self.state_dim + self.diffusion_encoding_channels + self.temporal_encoding_channels
        if self.ablation_config.torque_in_tokens:
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
        
        # Total steps for scheduling
        total_steps = self.trainer.estimated_stepping_batches
        
        # Warmup + Cosine Annealing for stability
        # Warmup prevents early gradient explosions
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
        # Cached table: [diffusion_steps, diff_enc_dim] -> take one row and expand.
        diffusion_vec = self.diffusion_encoding_table[diffusion_t - 1].to(device=device, dtype=normalized_state.dtype)
        diffusion_enc = diffusion_vec.view(1, 1, -1).expand(B, T, -1)  # [B, T, diff_enc_dim]
        
        # Temporal position encoding: absolute sinusoidal (LENGTH-INDEPENDENT)
        # Timestep t always gets the same encoding regardless of total T
        # This is crucial for trajectory extension without performance degradation
        if T <= self.temporal_encoding_table.shape[0]:
            temporal_enc = self.temporal_encoding_table[:T].to(device=device, dtype=normalized_state.dtype)  # [T, temp_enc_dim]
        else:
            # Fallback (rare in training): build on the fly for longer sequences.
            temporal_enc = self._get_temporal_encoding(T, device).to(dtype=normalized_state.dtype)
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
        
        # Replace state slice with noisy version (in-place: tokens are freshly built per step)
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

        # Random diffusion timestep
        diffusion_t = torch.randint(1, self.diffusion_steps + 1, (1,)).item()
        
        # Sample context length uniformly from [1, T_train-1]
        num_context = torch.randint(1, T_train, (1,)).item()
        
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
            
        # Main loss: predict noise with stability safeguards
        loss_denoise = F.mse_loss(predictions, noise)
        
        # ========== STABILITY: Skip NaN/Inf losses ==========
        if not torch.isfinite(loss_denoise):
            print(f"[WARNING] NaN/Inf loss detected at step {batch_idx}, skipping batch")
            return None  # PyTorch Lightning will skip this batch
        
        loss = loss_denoise
        
        # Log training loss (step-level only)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log('denoise_loss', loss_denoise, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        
        # Log trajectory length and context length for debugging
        self.log('T_train', float(T_train), prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log('num_context', float(num_context), prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        
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
        
        # Warn if gradient norm is suspiciously high (pre-clipping value)
        if total_norm > 10.0:
            print(f"[WARNING] High gradient norm: {total_norm:.2f} (will be clipped to 1.0)")
    
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
        guidance_before_steps: int = 9999,  # Stop guidance before this step (default: never stop)
        guidance_steps: int = 0,
        guidance_lr: float = 1e-3,
        langevin_step_size: float = 1e-5,
        langevin_noise_scale: float = 1e-6,
        chunk_length: int = 15,  # Chunk length for integration-based guidance
        use_forward_diff: bool = False,  # Use forward difference instead of central difference for HamRes
        dt: Optional[float] = None,
        # Torque generation parameters
        torque: torch.Tensor = None,  # Optional: provide torque directly
        initial_noise: torch.Tensor = None,  # Optional: fixed initial noise for reproducible sampling
        # Temporal smoothing
        smooth_sigma: float = 0.0,  # Gaussian smoothing sigma (0 = disabled, 1-3 recommended)
        smooth_guidance_only: bool = False,  # If True, smooth only for guidance input; output stays unsmoothed
        smooth_last_step_only: bool = False,  # If True, only smooth at the final diffusion step
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
            guidance_method: 'adam', 'langevin', or 'adam_integration'
            guidance_after_steps: start HNN guidance after this many steps
            guidance_steps: number of HNN optimization steps
            guidance_lr: learning rate for adam guidance
            langevin_step_size: step size for langevin guidance
            langevin_noise_scale: noise scale for langevin guidance
            lambda_init: weight for initial consistency term in HNN energy
            chunk_length: length of integration chunks for 'adam_integration' method
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
        from src.models.utils import run_adam_optimization_hnn, run_langevin_dynamics_hnn, run_adam_optimization_hnn_integration
        
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
        if initial_noise is not None:
            x = initial_noise.to(device)
        else:
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
            else:
                eps = eps_cond
            
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
                if hnn is not None and guidance_steps > 0 and guidance_after_steps <= i < guidance_before_steps:
                    x0_phys = self.denormalize_state(x0)

                    # smooth_guidance_only: smooth a copy for guidance, keep original for output
                    if smooth_guidance_only and smooth_sigma > 0:
                        from scipy.ndimage import gaussian_filter1d
                        x0_gui_np = x0_phys.detach().cpu().numpy()
                        x0_gui_smooth = gaussian_filter1d(x0_gui_np, sigma=smooth_sigma, axis=1)
                        x0_gui = torch.tensor(x0_gui_smooth, dtype=x0_phys.dtype, device=x0_phys.device)
                    else:
                        x0_gui = x0_phys

                    if guidance_method == "adam":
                        x0_phys = run_adam_optimization_hnn(
                            x0_gui, torque, self.qpos_dim, self.mom_dim,
                            self.data_dt, hnn, guidance_steps, guidance_lr,
                            use_forward_diff=use_forward_diff
                        )
                    elif guidance_method == "langevin":
                        x0_phys = run_langevin_dynamics_hnn(
                            x0_gui, torque, self.qpos_dim, self.mom_dim,
                            self.data_dt, hnn, guidance_steps, langevin_step_size, langevin_noise_scale
                        )
                    elif guidance_method == "adam_integration":
                        x0_phys = run_adam_optimization_hnn_integration(
                            x0_gui, torque, self.qpos_dim, self.mom_dim,
                            self.data_dt, hnn, guidance_steps, guidance_lr,
                            chunk_length=chunk_length
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
                if hnn is not None and guidance_steps > 0 and guidance_after_steps <= i < guidance_before_steps:
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

                    if guidance_method == "adam":
                        x0_phys = run_adam_optimization_hnn(
                            x0_gui, torque, self.qpos_dim, self.mom_dim,
                            self.data_dt, hnn, guidance_steps, guidance_lr,
                            use_forward_diff=use_forward_diff
                        )
                    elif guidance_method == "langevin":
                        x0_phys = run_langevin_dynamics_hnn(
                            x0_gui, torque, self.qpos_dim, self.mom_dim,
                            self.data_dt, hnn, guidance_steps, langevin_step_size, langevin_noise_scale
                        )
                    elif guidance_method == "adam_integration":
                        x0_phys = run_adam_optimization_hnn_integration(
                            x0_gui, torque, self.qpos_dim, self.mom_dim,
                            self.data_dt, hnn, guidance_steps, guidance_lr,
                            chunk_length=chunk_length
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
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 2],
                        help="GPU device IDs to use for training (e.g., --devices 0 1)")
    parser.add_argument("--num_workers", type=int, default=config.DEFAULT_NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LEARNING_RATE)
    parser.add_argument("--num_latents", type=int, default=config.DEFAULT_NUM_LATENTS)
    parser.add_argument("--num_latent_channels", type=int, default=config.DEFAULT_NUM_LATENT_CHANNELS)
    parser.add_argument("--diffusion_steps", type=int, default=config.DEFAULT_DIFFUSION_STEPS)
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
    parser.add_argument("--guidance_method", type=str, choices=["adam", "langevin", "adam_integration"], default="adam",
                        help="HNN guidance method: 'adam', 'langevin', or 'adam_integration' (integration-based)")
    parser.add_argument("--guidance_after_steps", type=int, default=150,
                        help="Start HNN guidance after this many diffusion steps (0 = from beginning)")
    parser.add_argument("--guidance_steps", type=int, default=20,
                        help="Number of HNN optimization steps per diffusion step (0 = disabled)")
    parser.add_argument("--guidance_lr", type=float, default=2e-3,
                        help="Learning rate for adam HNN guidance")
    parser.add_argument("--langevin_step_size", type=float, default=2e-4,
                        help="Step size for langevin HNN guidance")
    parser.add_argument("--langevin_noise_scale", type=float, default=0,
                        help="Noise scale for langevin HNN guidance")
    parser.add_argument("--lambda_init", type=float, default=1,
                        help="Weight for initial consistency term in HNN energy")
    parser.add_argument("--chunk_length", type=int, default=50,
                        help="Chunk length for integration-based guidance (adam_integration method)")
    parser.add_argument("--seed", type=int, default=228,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--use_trained_torque", action="store_true", default=False,
                        help="Use torque sequences from training data instead of generating new random ones")
    parser.add_argument("--torque_file", type=str, default=None,
                        help="Path to HDF5 file with pre-generated torque sequences (overrides --use_trained_torque)")
    
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
                    'seq_mom': state_np[:, model.qpos_dim:],
                    'seq_torque': torque_np,
                }
                mse_dict = compare_generated_with_reconstructed(
                    generated, '/home/gsang/Projects/Perceiver_IO/configs/rigid_arm_hinge.xml',
                    '/home/gsang/Projects/Perceiver_IO/plots',
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
        
        if args.guidance_steps > 0:
            print(f"\n[Step 1/2] Generating BASELINE trajectories (no guidance)...")
            
            # Set seed for reproducibility
            torch.manual_seed(args.seed)
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
                guidance_steps=0,
                torque=trained_torque,
            )
            
            # Store torque from baseline to reuse in guided run (ensures same torque)
            shared_torque = torque_baseline
            
            # Compute baseline MSE (no plots)
            print("[Step 1/2] Computing baseline MSE...")
            baseline_mse = compute_mse_stats(state_baseline, torque_baseline, save_plots=False)
        
        # ============================================================
        # STEP 2: Run with guidance (or just regular sampling if no guidance)
        # ============================================================
        if args.guidance_steps > 0:
            print(f"\n[Step 2/2] Generating GUIDED trajectories ({args.guidance_method})...")
        
        # Set seed again for reproducibility (same diffusion noise)
        torch.manual_seed(args.seed)
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
            hnn=hnn if args.guidance_steps > 0 else None,
            guidance_method=args.guidance_method,
            guidance_after_steps=args.guidance_after_steps,
            guidance_steps=args.guidance_steps,
            guidance_lr=args.guidance_lr,
            langevin_step_size=args.langevin_step_size,
            langevin_noise_scale=args.langevin_noise_scale,
            lambda_init=args.lambda_init,
            chunk_length=args.chunk_length,
            torque=shared_torque,  # Use same torque as baseline
        )
        
        # Build guidance string for naming
        if args.guidance_steps > 0:
            if args.guidance_method == "adam":
                guidance_str = f"hnn-{args.guidance_method}_after{args.guidance_after_steps}_steps{args.guidance_steps}_lr{args.guidance_lr}_lambda{args.lambda_init}"
            elif args.guidance_method == "adam_integration":
                guidance_str = f"hnn-{args.guidance_method}_after{args.guidance_after_steps}_steps{args.guidance_steps}_lr{args.guidance_lr}_chunk{args.chunk_length}_lambda{args.lambda_init}"
            else:  # langevin
                guidance_str = f"hnn-{args.guidance_method}_after{args.guidance_after_steps}_steps{args.guidance_steps}_ss{args.langevin_step_size}_ns{args.langevin_noise_scale}_lambda{args.lambda_init}"
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
        
        # Save to h5
        state_all = state.cpu().numpy()
        torque_all = torque.cpu().numpy()
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
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
    
    # Create dataloaders
    train_size = int(config.DEFAULT_TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
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
    src_path = '/home/gsang/Projects/Perceiver_IO/plots/trajectory.jpg'
    dst_path = '/home/gsang/Projects/Perceiver_IO/plots/trajectory_original.jpg'
    if os.path.exists(src_path):
        os.rename(src_path, dst_path)
        print(f"[Visualization] Saved {dst_path}")
    else:
        print(f"[Visualization] Warning: {src_path} not found, skipping rename")
    
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
    src_path_norm = '/home/gsang/Projects/Perceiver_IO/plots/trajectory.jpg'
    dst_path_norm = '/home/gsang/Projects/Perceiver_IO/plots/trajectory_normalized.jpg'
    if os.path.exists(src_path_norm):
        os.rename(src_path_norm, dst_path_norm)
        print(f"[Visualization] Saved {dst_path_norm}")
    else:
        print(f"[Visualization] Warning: {src_path_norm} not found, skipping rename")
    print(f"[Visualization] Original state range: [{full_state.min():.4f}, {full_state.max():.4f}]")
    print(f"[Visualization] Normalized state range: [{normalized_state.min():.4f}, {normalized_state.max():.4f}]")

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
        encoder_cond_mode="none",  # Encoder conditioning: "per_step", "mean", "rnn" or "none"
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
                'num_latents': args.num_latents,
                'num_latent_channels': args.num_latent_channels,
                'num_decoder_blocks': args.num_decoder_blocks,
                'diffusion_steps': args.diffusion_steps,
                'epochs': args.epochs,
                'fixed_trajectory_length': args.fixed_trajectory_length,
                'trajectory_length_training_options': list(traj_length_options),
            })
            print(f"Initialized W&B logging: project={args.wandb_project}")
    
    # Setup callbacks
    callbacks = []
    
    ablation_tag = f"_ablation-{args.ablation}" if args.ablation else ""
    length_tag = f"FixedTrajLength{args.fixed_trajectory_length}" if args.fixed_trajectory_length else "VariableTrajLength"
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&{length_tag}&UniformContext&EncoderNone&DecoderAttentions{ablation_tag}:{{epoch:03d}}_val_loss:{{val_loss:.4f}}',
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
        devices=args.devices,
        callbacks=callbacks,
        logger=logger,
        # precision="bf16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,  # ENABLED: Prevents gradient explosion
        gradient_clip_algorithm="norm",  # Clip by global norm (more stable than value)
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

