
"""Main Trajectory DPF model definition."""

import torch
import pytorch_lightning as pl
import math
from typing import Optional, Tuple


from perceiver.model.core import (
    FourierPositionEncoding,
)
from src.models.utils import (
    EMA,
)
from src import config

from src.models.trajectory_dpf_sampling import TrajectoryDPFSampling
from src.models.trajectory_dpf_training import TrajectoryDPFTraining
from src.models.architectures import (
    TrajectoryTransformerDiffusion,
    ConditionedTrajectoryPerceiverIO,
)
# -------------------------
# Trajectory DPF Module
# -------------------------

class TrajectoryDPF(TrajectoryDPFSampling, TrajectoryDPFTraining, pl.LightningModule):
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
        encoder_cond_mode: str = "none",  # "per_step", "mean", "rnn" or "none" for encoder conditioning
        backbone: str = "perceiverio",  # "perceiverio" or "transformer"
        unconditional_tau_in_state: bool = False,  # If True, model denoises [qpos|mom|torque] unconditionally
        training_context_mode: str = "random_subset",  # "random_subset", "future_context", or shifted-token mode
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
        self.encoder_cond_mode = encoder_cond_mode
        self.backbone = backbone
        legacy_context_mode_map = {
            "shifted_future_context": "shifted_tau_tokens",
            "shifted_future_context_cleanprefix": "shifted_tau_tokens",
        }
        training_context_mode = legacy_context_mode_map.get(training_context_mode, training_context_mode)
        if training_context_mode not in {
            "random_subset",
            "future_context",
            "shifted_tau_tokens",
        }:
            raise ValueError(f"Unsupported training_context_mode: {training_context_mode}")
        if training_context_mode == "shifted_tau_tokens" and not self.unconditional_tau_in_state:
            raise ValueError(
                f"training_context_mode={training_context_mode!r} requires unconditional_tau_in_state=True."
            )
        self.training_context_mode = training_context_mode
        self.use_shifted_tau_tokens = training_context_mode == "shifted_tau_tokens"
        if self.use_shifted_tau_tokens:
            print(
                "[Init] Using shifted tau tokens only: state_i pairs with tau_{i-1}; "
                "context/query construction stays on the generic path."
            )

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
        num_input_channels_raw = self.state_dim + self.diffusion_encoding_channels + self.temporal_encoding_channels
        
        # Ensure num_input_channels is divisible by num_heads (8) for attention
        num_heads = 8
        if num_input_channels_raw % num_heads != 0:
            padding = num_heads - (num_input_channels_raw % num_heads)
            self.temporal_encoding_channels += padding
            print(f"[Init] Padded temporal_encoding_channels by {padding} to make num_input_channels divisible by {num_heads}")
        num_input_channels = self.state_dim + self.diffusion_encoding_channels + self.temporal_encoding_channels

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

# -------------------------
# Main training script
# -------------------------
