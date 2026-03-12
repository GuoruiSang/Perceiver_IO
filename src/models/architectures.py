"""
Model architectures for Trajectory DPF.

This module contains the core model architecture classes used by the Trajectory DPF subsystem:
- TrajectoryOutputAdapter: Output adapter for projecting decoder outputs to state dimensions
- TrajectoryPerceiverIO: PerceiverIO backbone for trajectory generation (original, uses pip package)
- ConditionedTrajectoryPerceiverIO: PerceiverIO with DiT-style AdaLN conditioning on torque
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from perceiver.model.core import (
    InputAdapter as PerceiverInputAdapter,
    OutputAdapter,
    PerceiverDecoder,
    PerceiverEncoder,
)

class InputAdapter(PerceiverInputAdapter):
    """Pass-through input adapter that returns input unchanged."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return input as-is (assumes input is already properly formatted tokens)."""
        return x


class OutputQueryProvider(nn.Module):
    """Simple output query provider that returns input queries unchanged.
    
    Stateless provider exposing num_query_channels for decoder construction.
    """
    
    def __init__(self, num_query_channels: int):
        super().__init__()
        self._num_query_channels = num_query_channels
    
    @property
    def num_query_channels(self) -> int:
        return self._num_query_channels
    
    def forward(self, x=None) -> torch.Tensor:
        """Identity: pass provided queries through unmodified."""
        return x


class TrajectoryOutputAdapter(OutputAdapter):
    """Output adapter for trajectory generation that projects decoder output to state dimensions."""
    
    def __init__(self, num_output_channels: int, num_decoder_channels: int):
        super().__init__()
        self.linear = nn.Linear(num_decoder_channels, num_output_channels)
    
    def forward(self, x):
        return self.linear(x)


class TrajectoryPerceiverIO(nn.Module):
    """PerceiverIO for trajectory token arrays [B, T, C_in]."""
    
    def __init__(
        self,
        num_input_channels: int,
        num_output_channels: int,
        num_latents: int = 256,
        num_latent_channels: int = 256,
        num_self_attention_layers_per_block: int = 2,
        num_self_attention_blocks: int = 8,
        num_self_attention_heads: int = 4,
        num_cross_attention_heads: int = 4,
    ):
        super().__init__()
        self.input_adapter = InputAdapter(num_input_channels=num_input_channels)
        self.encoder = PerceiverEncoder(
            input_adapter=self.input_adapter,
            num_latents=num_latents,
            num_latent_channels=num_latent_channels,
            num_self_attention_layers_per_block=num_self_attention_layers_per_block,
            num_self_attention_blocks=num_self_attention_blocks,
            num_self_attention_heads=num_self_attention_heads,
            num_cross_attention_heads=num_cross_attention_heads,
        )
        self.output_query_provider = OutputQueryProvider(
            num_query_channels=num_input_channels
        )
        self.output_adapter = TrajectoryOutputAdapter(
            num_output_channels=num_output_channels,
            num_decoder_channels=num_input_channels,
        )
        self.decoder = PerceiverDecoder(
            output_adapter=self.output_adapter,
            output_query_provider=self.output_query_provider,
            num_latent_channels=num_latent_channels,
            num_cross_attention_qk_channels=num_latent_channels // 2,
        )

    def forward(
        self, 
        contexts: torch.Tensor, 
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            contexts: [B, N_ctx, C_in] - context tokens
            queries: [B, N_qry, C_in] - query tokens
        Returns:
            predictions: [B, N_qry, C_out] - predicted noise
        """
        latents = self.encoder(contexts)
        predictions = self.decoder(latents, queries)
        return predictions


class TrajectoryTransformerDiffusion(nn.Module):
    """Transformer diffusion baseline without latent bottleneck.

    Expects full-sequence tokens and predicts full-sequence noise in one pass.
    """

    def __init__(
        self,
        num_input_channels: int,
        num_output_channels: int,
        d_model: int = 256,
        num_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_input_channels, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_output_channels)

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_tokens: [B, T, C_in]
        Returns:
            predictions: [B, T, C_out]
        """
        h = self.input_norm(self.input_proj(x_tokens))
        h = self.encoder(h)
        return self.output_proj(self.output_norm(h))


# =============================================================================
# DiT-style AdaLN Conditioned PerceiverIO
# =============================================================================

class TorqueConditioner(nn.Module):
    """Encodes torque sequence to conditioning embeddings for AdaLN modulation."""
    
    def __init__(self, torque_dim: int, cond_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(torque_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
    
    def forward(self, torque: torch.Tensor) -> torch.Tensor:
        """
        Args:
            torque: [B, T, torque_dim] - raw torque sequence
        Returns:
            cond: [B, T, cond_dim] - per-timestep conditioning embeddings
        """
        return self.mlp(torque)


class StateConditioner(nn.Module):
    """Encodes state sequence to conditioning embeddings for interaction with torque."""
    
    def __init__(self, state_dim: int, cond_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, T, state_dim] - state sequence (e.g., noisy estimate)
        Returns:
            emb: [B, T, cond_dim] - per-timestep state embeddings
        """
        return self.mlp(state)


class InteractionMLP(nn.Module):
    """Computes interaction embedding from concatenated state and torque embeddings."""
    
    def __init__(self, cond_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
    
    def forward(self, state_emb: torch.Tensor, torque_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_emb: [B, T, cond_dim] - state embeddings
            torque_emb: [B, T, cond_dim] - torque embeddings
        Returns:
            interaction: [B, T, cond_dim] - interaction embeddings encoding (state_t, torque_t)
        """
        combined = torch.cat([state_emb, torque_emb], dim=-1)
        return self.mlp(combined)


class AdaLNSelfAttentionBlock(nn.Module):
    """
    Self-attention block with AdaLN-Zero conditioning (DiT-style).
    
    Modulation parameters (scale, shift, gate) are initialized to zero for stable training.
    """
    
    def __init__(self, dim: int, cond_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        
        # AdaLN-Zero: 6 modulation vectors (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 6),
        )
        # Initialize to zero for stable training
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim] - input features (latents)
            cond: [B, 1, cond_dim] - global conditioning (broadcasted to all positions)
        Returns:
            output: [B, N, dim] - modulated features
        """
        B, N, C = x.shape
        
        # Get modulation parameters from conditioning
        mod = self.adaLN_modulation(cond)  # [B, 1, dim*6]
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)
        
        # Self-attention with AdaLN
        h = self.norm1(x) * (1 + scale1) + shift1
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, num_heads, N, head_dim]
        
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        attn = self.attn_dropout(attn)
        x = x + gate1 * self.proj(attn)
        
        # MLP with AdaLN
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        
        return x


class AdaLNCrossAttentionBlock(nn.Module):
    """
    Cross-attention block with STATE-ONLY AdaLN-Zero conditioning.
    
    Queries attend to latents (key-value), with per-timestep torque modulation 
    applied ONLY to the state portion of queries. Diffusion and temporal encodings
    are left untouched to preserve temporal coherence.
    
    Token structure: [state | diffusion_enc | temporal_enc]
    AdaLN modulates: [state] only
    """
    
    def __init__(
        self, 
        query_dim: int, 
        latent_dim: int, 
        cond_dim: int,
        state_dim: int,  # NEW: dimension of state slice to modulate
        num_heads: int = 8, 
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.state_dim = state_dim  # Only modulate this portion
        assert query_dim % num_heads == 0, f"query_dim {query_dim} must be divisible by num_heads {num_heads}"
        
        self.norm_q = nn.LayerNorm(query_dim, elementwise_affine=False)
        self.norm_kv = nn.LayerNorm(latent_dim)  # KV from latents, no conditioning needed
        
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.kv_proj = nn.Linear(latent_dim, query_dim * 2)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.attn_dropout = nn.Dropout(dropout)
        
        # AdaLN-Zero for STATE-ONLY modulation: only modulate state_dim channels
        # Output scale/shift for state portion only
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, state_dim * 2),  # Only state_dim, not query_dim
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(
        self, 
        queries: torch.Tensor, 
        latents: torch.Tensor, 
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, T, query_dim] - query tokens [state | diffusion_enc | temporal_enc]
            latents: [B, N, latent_dim] - latent array from encoder
            cond: [B, T, cond_dim] - per-timestep conditioning
        Returns:
            output: [B, T, query_dim] - cross-attended features
        """
        B, T, C = queries.shape
        N = latents.shape[1]
        
        # Get per-timestep modulation parameters (for state slice only)
        mod = self.adaLN_modulation(cond)  # [B, T, state_dim*2]
        shift, scale = mod.chunk(2, dim=-1)  # Each: [B, T, state_dim]
        
        # STATE-ONLY AdaLN: modulate only the state portion, leave encodings untouched
        queries_normed = self.norm_q(queries)  # [B, T, query_dim]
        
        # Split into state and encoding portions
        state_normed = queries_normed[:, :, :self.state_dim]  # [B, T, state_dim]
        enc_normed = queries_normed[:, :, self.state_dim:]    # [B, T, query_dim - state_dim]
        
        # Apply AdaLN only to state portion
        state_modulated = state_normed * (1 + scale) + shift  # [B, T, state_dim]
        
        # Recombine: modulated state + untouched encodings
        queries_modulated = torch.cat([state_modulated, enc_normed], dim=-1)  # [B, T, query_dim]
        
        # Cross-attention with partially modulated queries
        q = self.q_proj(queries_modulated)
        kv = self.kv_proj(self.norm_kv(latents))
        k, v = kv.chunk(2, dim=-1)
        
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, T, C)
        attn = self.attn_dropout(attn)
        
        return queries + self.out_proj(attn)


class AdaLNContextModulation(nn.Module):
    """
    STATE-ONLY AdaLN modulation for context tokens in the encoder cross-attention.

    Modulates only the state portion of context tokens (not diffusion/temporal encodings)
    using per-timestep conditioning. This changes how context presents itself as K/V
    in cross-attention with learnable latent queries.

    Token structure: [state | diffusion_enc | temporal_enc]
    AdaLN modulates: [state] only
    """

    def __init__(self, context_dim: int, cond_dim: int, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.norm = nn.LayerNorm(context_dim, elementwise_affine=False)

        # AdaLN-Zero: scale and shift for state portion only
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, state_dim * 2),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, contexts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            contexts: [B, N_ctx, context_dim] - context tokens
            cond: [B, N_ctx, cond_dim] - per-timestep conditioning (shifted)
        Returns:
            modulated: [B, N_ctx, context_dim] - modulated context tokens
        """
        mod = self.adaLN_modulation(cond)  # [B, N_ctx, state_dim*2]
        shift, scale = mod.chunk(2, dim=-1)  # Each: [B, N_ctx, state_dim]

        normed = self.norm(contexts)

        # Split state and encoding portions
        state_normed = normed[:, :, :self.state_dim]
        enc_normed = normed[:, :, self.state_dim:]

        # Modulate only state portion
        state_modulated = state_normed * (1 + scale) + shift

        # Recombine
        return torch.cat([state_modulated, enc_normed], dim=-1)


class StandardCrossAttentionBlock(nn.Module):
    """
    Standard cross-attention block WITHOUT AdaLN conditioning.

    Same interface as AdaLNCrossAttentionBlock (accepts ``cond``) but ignores it.
    Used for the torque-concat ablation where conditioning is in the tokens.
    """

    def __init__(
        self,
        query_dim: int,
        latent_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert query_dim % num_heads == 0

        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.kv_proj = nn.Linear(latent_dim, query_dim * 2)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        latents: torch.Tensor,
        cond: torch.Tensor = None,
    ) -> torch.Tensor:
        """cond is accepted for interface compatibility but ignored."""
        B, T, C = queries.shape
        N = latents.shape[1]

        q = self.q_proj(self.norm_q(queries))
        kv = self.kv_proj(self.norm_kv(latents))
        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, T, C)
        attn = self.attn_dropout(attn)

        return queries + self.out_proj(attn)


class ConditionedPerceiverEncoder(nn.Module):
    """
    PerceiverIO encoder with AdaLN-conditioned self-attention blocks and optional
    per-step AdaLN modulation of context tokens before cross-attention.

    Architecture:
    1. (Optional) Per-step STATE-ONLY AdaLN on context tokens (K/V side)
    2. Initial cross-attention: context tokens → learnable latents
    3. N self-attention blocks on latents with global torque conditioning (AdaLN)
    """

    def __init__(
        self,
        num_input_channels: int,
        num_latents: int = 256,
        num_latent_channels: int = 256,
        cond_dim: int = 256,
        state_dim: int = None,
        num_self_attention_blocks: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_cross_attn_adaln: bool = False,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.num_latent_channels = num_latent_channels

        # Optional per-step AdaLN context modulation for cross-attention K/V
        self.use_cross_attn_adaln = use_cross_attn_adaln
        if use_cross_attn_adaln and state_dim is not None:
            self.context_modulation = AdaLNContextModulation(
                context_dim=num_input_channels,
                cond_dim=cond_dim,
                state_dim=state_dim,
            )
        else:
            self.context_modulation = None

        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(1, num_latents, num_latent_channels) * 0.02)

        # Initial cross-attention: context → latents
        self.cross_attn_norm_latent = nn.LayerNorm(num_latent_channels)
        self.cross_attn_norm_context = nn.LayerNorm(num_input_channels)
        self.cross_attn_q = nn.Linear(num_latent_channels, num_latent_channels)
        self.cross_attn_kv = nn.Linear(num_input_channels, num_latent_channels * 2)
        self.cross_attn_proj = nn.Linear(num_latent_channels, num_latent_channels)
        self.cross_attn_num_heads = num_heads
        self.cross_attn_head_dim = num_latent_channels // num_heads

        # Self-attention blocks with AdaLN conditioning
        self.self_attn_blocks = nn.ModuleList([
            AdaLNSelfAttentionBlock(num_latent_channels, cond_dim, num_heads, dropout)
            for _ in range(num_self_attention_blocks)
        ])

    def forward(
        self,
        contexts: torch.Tensor,
        cond: torch.Tensor,
        per_step_cond: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            contexts: [B, N_ctx, num_input_channels] - context tokens (state only, no torque)
            cond: [B, 1, cond_dim] - global conditioning for self-attention blocks
            per_step_cond: [B, N_ctx, cond_dim] - per-timestep conditioning for cross-attention
                           context modulation (only used when use_cross_attn_adaln=True)
        Returns:
            latents: [B, num_latents, num_latent_channels] - encoded latent array
        """
        B = contexts.shape[0]
        N_ctx = contexts.shape[1]

        # Apply per-step AdaLN to context tokens before cross-attention (if enabled)
        if self.context_modulation is not None and per_step_cond is not None:
            contexts = self.context_modulation(contexts, per_step_cond)

        # Initialize latents
        latents = self.latents.expand(B, -1, -1)  # [B, num_latents, num_latent_channels]

        # Initial cross-attention: latents attend to (optionally modulated) context
        q = self.cross_attn_q(self.cross_attn_norm_latent(latents))
        kv = self.cross_attn_kv(self.cross_attn_norm_context(contexts))
        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(B, self.num_latents, self.cross_attn_num_heads, self.cross_attn_head_dim).transpose(1, 2)
        k = k.reshape(B, N_ctx, self.cross_attn_num_heads, self.cross_attn_head_dim).transpose(1, 2)
        v = v.reshape(B, N_ctx, self.cross_attn_num_heads, self.cross_attn_head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, self.num_latents, self.num_latent_channels)
        latents = latents + self.cross_attn_proj(attn)

        # Self-attention blocks with global AdaLN conditioning
        for block in self.self_attn_blocks:
            latents = block(latents, cond)

        return latents


class ConditionedPerceiverDecoder(nn.Module):
    """
    PerceiverIO decoder with STATE-ONLY AdaLN-conditioned cross-attention
    AND optional self-attention blocks for refinement.
    
    Architecture:
    1. Cross-attention: queries attend to latents with per-timestep torque modulation (AdaLN)
       - AdaLN only modulates state portion, leaving diffusion/temporal encodings untouched
    2. Optional Self-Attention Blocks: refinement with per-timestep torque modulation (AdaLN)
    3. Output projection to state dimensions
    """
    
    def __init__(
        self,
        num_query_channels: int,
        num_latent_channels: int,
        num_output_channels: int,
        state_dim: int,  # NEW: dimension of state slice for state-only AdaLN
        cond_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        num_decoder_blocks: int = 0,  # NEW: Number of self-attention blocks
    ):
        super().__init__()
        
        # Cross-attention with STATE-ONLY AdaLN conditioning
        self.cross_attn = AdaLNCrossAttentionBlock(
            query_dim=num_query_channels,
            latent_dim=num_latent_channels,
            cond_dim=cond_dim,
            state_dim=state_dim,  # Pass state_dim for state-only modulation
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Self-attention blocks with AdaLN conditioning (refinement)
        self.self_attn_blocks = nn.ModuleList([
            AdaLNSelfAttentionBlock(
                dim=num_query_channels,
                cond_dim=cond_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_decoder_blocks)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(num_query_channels)
        self.output_proj = nn.Linear(num_query_channels, num_output_channels)
    
    def forward(
        self, 
        latents: torch.Tensor, 
        queries: torch.Tensor, 
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            latents: [B, N, num_latent_channels] - latent array from encoder
            queries: [B, T, num_query_channels] - query tokens (state + positional encodings)
            cond: [B, T, cond_dim] - per-timestep conditioning
        Returns:
            output: [B, T, num_output_channels] - predicted noise
        """
        # Cross-attention with per-timestep AdaLN modulation
        output = self.cross_attn(queries, latents, cond)
        
        # Self-attention blocks for refinement (with per-timestep AdaLN)
        for block in self.self_attn_blocks:
            output = block(output, cond)
        
        # Output projection
        output = self.output_proj(self.output_norm(output))
        
        return output


class ConditionedTrajectoryPerceiverIO(nn.Module):
    """
    PerceiverIO with per-step state-torque interaction conditioning.

    Token structure (context and query):
        [state | diffusion_enc | temporal_enc]  (NO torque - it's used for AdaLN modulation)

    Per-step control mechanism (implements torque_t ⊗ state_t → state_{t+1}):
        1. Extract current state estimate x_t from tokens[:, :, :state_dim]
        2. Compute embeddings: x_emb = StateConditioner(x_t), u_emb = TorqueConditioner(u_t)
        3. Compute interaction: c_t = InteractionMLP(x_emb, u_emb)
        4. Shift-right: c_next[t+1] = c[t], c_next[0] = learnable c0 parameter
        5. Use c_next as per-timestep conditioning

    Dual conditioning streams:
        - Decoder: conditioning from full query sequence [B, T, cond_dim]
        - Encoder (per_step mode): conditioning from first N_ctx timesteps only [B, N_ctx, cond_dim]
          Applied as STATE-ONLY AdaLN on context tokens before encoder cross-attention.

    Encoder conditioning modes:
        - "per_step": per-step causal AdaLN on context tokens + global cond for self-attention
        - "mean": global conditioning (mean of full query conditioning) for self-attention only
        - "rnn": GRU-based global conditioning for self-attention only
        - "none": no encoder conditioning
    """
    
    def __init__(
        self,
        num_input_channels: int,
        num_output_channels: int,
        state_dim: int,
        torque_dim: int,
        num_latents: int = 256,
        num_latent_channels: int = 256,
        cond_dim: int = 256,
        num_self_attention_blocks: int = 8,
        num_decoder_blocks: int = 0,  # NEW: Number of decoder self-attention blocks
        num_heads: int = 8,
        dropout: float = 0.0,
        encoder_cond_mode: str = "mean",  # "per_step", "mean", "rnn", or "none"
    ):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.state_dim = state_dim
        self.torque_dim = torque_dim
        self.cond_dim = cond_dim
        self.encoder_cond_mode = encoder_cond_mode
        self.torque_conditioner = TorqueConditioner(torque_dim, cond_dim)
        # Learnable initial conditioning for c[0] (replaces zero-fill at t=0)
        self.c0 = nn.Parameter(torch.zeros(1, 1, cond_dim))
        self.state_conditioner = StateConditioner(state_dim, cond_dim)
        self.interaction_mlp = InteractionMLP(cond_dim)

        # Global conditioning projection (for encoder, if enabled)
        if encoder_cond_mode in ("mean", "per_step"):
            self.global_cond_proj = nn.Linear(cond_dim, cond_dim)
            self.cond_rnn = None
        elif encoder_cond_mode == "rnn":
            self.global_cond_proj = nn.Linear(cond_dim, cond_dim)
            self.cond_rnn = nn.GRU(cond_dim, cond_dim, batch_first=True)
        else:
            self.global_cond_proj = None
            self.cond_rnn = None

        # Conditioned encoder
        self.encoder = ConditionedPerceiverEncoder(
            num_input_channels=num_input_channels,
            num_latents=num_latents,
            num_latent_channels=num_latent_channels,
            cond_dim=cond_dim,
            state_dim=state_dim,
            num_self_attention_blocks=num_self_attention_blocks,
            num_heads=num_heads,
            dropout=dropout,
            use_cross_attn_adaln=(encoder_cond_mode == "per_step"),
        )

        self.decoder = ConditionedPerceiverDecoder(
            num_query_channels=num_input_channels,
            num_latent_channels=num_latent_channels,
            num_output_channels=num_output_channels,
            state_dim=state_dim,
            cond_dim=cond_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_decoder_blocks=num_decoder_blocks,
        )
    
    def _compute_conditioning(
        self,
        tokens: torch.Tensor,
        torque_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute interaction conditioning from tokens and torque embeddings.

        Args:
            tokens: [B, L, num_input_channels] - tokens (queries or contexts)
            torque_emb: [B, L, cond_dim] - torque embeddings for the same L positions
        Returns:
            c: [B, L, cond_dim] - conditioning embeddings (before shift-right)
        """
        x_est = tokens[:, :, :self.state_dim]
        x_emb = self.state_conditioner(x_est)
        return self.interaction_mlp(x_emb, torque_emb)

    def _shift_right(self, c: torch.Tensor) -> torch.Tensor:
        """Apply causal shift-right with learnable c0.

        Args:
            c: [B, L, cond_dim] - conditioning embeddings
        Returns:
            c_shifted: [B, L, cond_dim] - shifted conditioning (c_shifted[0] = c0, c_shifted[t] = c[t-1])
        """
        B = c.shape[0]
        c_shifted = torch.zeros_like(c)
        c_shifted[:, 0:1, :] = self.c0.expand(B, -1, -1)
        c_shifted[:, 1:, :] = c[:, :-1, :]
        return c_shifted

    def forward(
        self,
        contexts: torch.Tensor,
        queries: torch.Tensor,
        torque: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            contexts: [B, N_ctx, num_input_channels] - context tokens (state + pos encodings, no torque)
            queries: [B, N_qry, num_input_channels] - query tokens (state + pos encodings, no torque)
            torque: [B, N_qry, torque_dim] - optional torque conditioning (normalized to [-1, 1])
        Returns:
            predictions: [B, N_qry, num_output_channels] - predicted noise

        Key mechanism (full model):
            - Query at index t gets modulated by c_next[t] = interaction(x_{t-1}, u_{t-1})
            - This implements: torque_{t-1} applied to state_{t-1} affects state_t
            - Equivalently: torque_t applied to state_t affects state_{t+1}
        """
        B, T, _ = queries.shape
        N_ctx = contexts.shape[1]
        # Compute conditioning for the full model path.
        if torque is not None:
            # Torque embedding for full query sequence
            u_emb = self.torque_conditioner(torque)  # [B, T, cond_dim]

            # ========== DECODER CONDITIONING (full query sequence) ==========
            c = self._compute_conditioning(queries, u_emb)  # [B, T, cond_dim]
            c_shifted = self._shift_right(c)

            # Per-step vs global conditioning for decoder
            if self.encoder_cond_mode == "per_step":
                c_next = c_shifted
            else:
                c_next = c_shifted.mean(dim=1, keepdim=True).expand(-1, T, -1)

            # ========== ENCODER CONDITIONING ==========
            per_step_enc_cond = None

            if self.encoder_cond_mode == "per_step":
                # Compute encoder-specific conditioning from first N_ctx timesteps only
                u_emb_enc = u_emb[:, :N_ctx, :]  # Reuse torque embeddings, sliced
                c_enc = self._compute_conditioning(contexts, u_emb_enc)  # [B, N_ctx, cond_dim]
                per_step_enc_cond = self._shift_right(c_enc)  # [B, N_ctx, cond_dim]

                # Global conditioning for encoder self-attention from context-only conditioning
                global_cond = self.global_cond_proj(c_enc.mean(dim=1, keepdim=True))

            elif self.encoder_cond_mode == "mean" and self.global_cond_proj is not None:
                global_cond = self.global_cond_proj(c.mean(dim=1, keepdim=True))

            elif self.encoder_cond_mode == "rnn" and self.global_cond_proj is not None:
                _, h_n = self.cond_rnn(c)
                global_cond = self.global_cond_proj(h_n.permute(1, 0, 2))

            else:  # "none"
                global_cond = torch.zeros(B, 1, self.cond_dim, device=queries.device, dtype=queries.dtype)

        else:
            # No torque conditioning provided: fall back to zero conditioning.
            c_next = torch.zeros(B, T, self.cond_dim, device=queries.device, dtype=queries.dtype)
            global_cond = torch.zeros(B, 1, self.cond_dim, device=queries.device, dtype=queries.dtype)
            per_step_enc_cond = None

        # Encode contexts with global + per-step conditioning
        latents = self.encoder(contexts, global_cond, per_step_enc_cond)

        # Decode queries with per-timestep conditioning
        predictions = self.decoder(latents, queries, c_next)

        return predictions
