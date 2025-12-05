"""
Model architectures for Trajectory DPF.

This module contains the core model architecture classes extracted from trajectory_dpf.py:
- TrajectoryOutputAdapter: Output adapter for projecting decoder outputs to state dimensions
- TrajectoryPerceiverIO: PerceiverIO backbone for trajectory generation
"""

import torch
import torch.nn as nn
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
