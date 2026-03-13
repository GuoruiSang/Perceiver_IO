"""Training/validation helpers for Trajectory DPF."""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from src.models.utils import EMA


class TrajectoryDPFTraining:
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
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], int, int]:
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

        noisy_contexts = noisy_tokens.index_select(dim=1, index=context_idx)
        noisy_queries = noisy_tokens.index_select(dim=1, index=query_idx)
        noise_target = noise.index_select(dim=1, index=query_idx)
        cond = torque.index_select(dim=1, index=query_idx) if torque is not None else None
        return noisy_contexts, noisy_queries, cond, noise_target, None, int(context_idx.numel()), int(query_idx.numel())

    def _apply_state_noise_with_mask(
        self,
        clean_tokens: torch.Tensor,
        diffusion_t: int,
        state_noise_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply diffusion noise only where `state_noise_mask` is True.

        Args:
            clean_tokens: [B, T, C]
            diffusion_t: diffusion timestep
            state_noise_mask: [T] or [B, T] boolean mask over token positions
        """
        if state_noise_mask.ndim == 1:
            state_noise_mask = state_noise_mask.unsqueeze(0).expand(clean_tokens.shape[0], -1)
        if state_noise_mask.shape != clean_tokens.shape[:2]:
            raise ValueError(
                "state_noise_mask must match the [B,T] token layout: "
                f"got {tuple(state_noise_mask.shape)} for tokens {tuple(clean_tokens.shape[:2])}"
            )

        mask = state_noise_mask.to(device=clean_tokens.device, dtype=torch.bool).unsqueeze(-1)
        noisy_tokens = clean_tokens.clone()
        clean_state = noisy_tokens[:, :, :self.state_dim]
        noise = torch.randn_like(clean_state)

        noisy_state = torch.where(
            mask,
            self.sqrt_alpha_cumprod[diffusion_t - 1] * clean_state +
            self.sqrt_one_minus_alpha_cumprod[diffusion_t - 1] * noise,
            clean_state,
        )
        noisy_tokens[:, :, :self.state_dim] = noisy_state
        noise = noise * mask.to(dtype=noise.dtype)
        return noisy_tokens, noise

    def _build_clean_prefix_noisy_suffix_views(
        self,
        state: torch.Tensor,
        torque: Optional[torch.Tensor],
        diffusion_t: int,
        time_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, int, int]:
        """
        Build query=[clean prefix, noisy suffix] and train only on the suffix.

        Contexts are sampled as a subset of the query tokens.
        """
        _, T, _ = state.shape
        clean_tokens = self.build_tokens(state, diffusion_t, time_indices=time_indices)

        suffix_start = torch.randint(1, T, (1,), device=state.device).item() if T > 1 else 0
        loss_mask = torch.zeros(T, device=state.device, dtype=torch.bool)
        loss_mask[suffix_start:] = True
        noisy_queries, noise = self._apply_state_noise_with_mask(
            clean_tokens,
            diffusion_t,
            state_noise_mask=loss_mask,
        )

        num_context = torch.randint(1, T + 1, (1,), device=state.device).item()
        context_idx = self._sample_subset_indices(T, num_context, state.device)
        noisy_contexts = noisy_queries.index_select(dim=1, index=context_idx)
        return noisy_contexts, noisy_queries, torque, noise, loss_mask, int(context_idx.numel()), T

    def _build_perceiver_train_views(
        self,
        state: torch.Tensor,
        torque: Optional[torch.Tensor],
        diffusion_t: int,
        time_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], int, int]:
        """
        Build Perceiver context/query tensors for training or validation.

        Returns:
            contexts, queries, cond, noise_target, loss_mask, num_context, num_query
        """
        B, T, _ = state.shape
        if time_indices is None:
            time_indices = torch.arange(T, device=state.device, dtype=torch.long)

        if self.query_context_mode == "future_context":
            return self._build_future_context_views(state, torque, diffusion_t, time_indices)
        if self.query_context_mode == "clean_prefix_noisy_suffix":
            return self._build_clean_prefix_noisy_suffix_views(state, torque, diffusion_t, time_indices)

        tokens = self.build_tokens(state, diffusion_t, time_indices=time_indices)
        noisy_tokens, noise = self.apply_noise(tokens, diffusion_t, return_noise=True)

        num_context = torch.randint(1, T + 1, (1,), device=state.device).item()
        num_query = torch.randint(1, T + 1, (1,), device=state.device).item()
        context_idx = self._sample_subset_indices(T, num_context, state.device)
        query_idx = self._sample_subset_indices(T, num_query, state.device)

        noisy_contexts = noisy_tokens.index_select(dim=1, index=context_idx)
        noisy_queries = noisy_tokens.index_select(dim=1, index=query_idx)
        noise_target = noise.index_select(dim=1, index=query_idx)
        cond = torque.index_select(dim=1, index=query_idx) if torque is not None else None
        return noisy_contexts, noisy_queries, cond, noise_target, None, num_context, num_query

    def _compute_denoise_loss(
        self,
        predictions: torch.Tensor,
        noise_target: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if loss_mask is not None:
            if loss_mask.ndim == 1:
                loss_mask = loss_mask.unsqueeze(0).expand(predictions.shape[0], -1)
            if loss_mask.shape != predictions.shape[:2]:
                raise ValueError(
                    "loss_mask must match the [B,T] prediction layout: "
                    f"got {tuple(loss_mask.shape)} for predictions {tuple(predictions.shape[:2])}"
                )
            weight = loss_mask.to(device=predictions.device, dtype=predictions.dtype).unsqueeze(-1)
            sq_error = (predictions - noise_target) ** 2
            denom = weight.sum() * predictions.shape[-1]
            if denom <= 0:
                raise ValueError("loss_mask selected zero tokens for denoising loss")
            return (sq_error * weight).sum() / denom
        return F.mse_loss(predictions, noise_target)

    def training_step(self, batch, batch_idx):
        """Training step with optional concat-state torque or AdaLN torque conditioning."""
        # batch is a dict with keys: 'seq_qpos', 'seq_mom', 'seq_torque'
        qpos = batch['seq_qpos']  # [B, T, qpos_dim]
        mom = batch['seq_mom']    # [B, T, mom_dim]
        torque = batch['seq_torque']  # [B, T, torque_dim]
        
        if self.unconditional_tau_in_state:
            torque_tokens = self._shift_torque_sequence(torque) if self.use_shifted_tau_tokens else torque
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
                tokens = self.build_tokens(state, diffusion_t, time_indices=time_indices)
                if self.query_context_mode == "clean_prefix_noisy_suffix":
                    loss_mask = torch.zeros(T_train, device=state.device, dtype=torch.bool)
                    suffix_start = torch.randint(1, T_train, (1,), device=state.device).item() if T_train > 1 else 0
                    loss_mask[suffix_start:] = True
                    noisy_tokens, noise = self._apply_state_noise_with_mask(tokens, diffusion_t, loss_mask)
                else:
                    noisy_tokens, noise = self.apply_noise(tokens, diffusion_t, return_noise=True)
                    loss_mask = None
                predictions = self.model(noisy_tokens)
                noise_target = noise
                num_query = T_train
            else:
                noisy_contexts, noisy_queries, _, noise_target, loss_mask, num_context, num_query = self._build_perceiver_train_views(
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
                    state, diffusion_t, torque=torque_norm, include_torque=True, time_indices=time_indices
                )
                if self.query_context_mode == "clean_prefix_noisy_suffix":
                    loss_mask = torch.zeros(T_train, device=state.device, dtype=torch.bool)
                    suffix_start = torch.randint(1, T_train, (1,), device=state.device).item() if T_train > 1 else 0
                    loss_mask[suffix_start:] = True
                    noisy_tokens, noise = self._apply_state_noise_with_mask(tokens, diffusion_t, loss_mask)
                else:
                    noisy_tokens, noise = self.apply_noise(tokens, diffusion_t, return_noise=True)
                    loss_mask = None
                predictions = self.model(noisy_tokens)
                noise_target = noise
                num_query = T_train
            else:
                noisy_contexts, noisy_queries, torque_for_queries, noise_target, loss_mask, num_context, num_query = (
                    self._build_perceiver_train_views(
                        state,
                        torque=torque_norm,
                        diffusion_t=diffusion_t,
                        time_indices=time_indices,
                    )
                )
                predictions = self.model(noisy_contexts, noisy_queries, torque_for_queries)
            
        loss_denoise = self._compute_denoise_loss(predictions, noise_target, loss_mask=loss_mask)
        
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
        self.log('num_query', float(num_query), prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with prefix context (always conditional, no CFG dropout)."""
        qpos = batch['seq_qpos']
        mom = batch['seq_mom']
        torque = batch['seq_torque']
        
        if self.unconditional_tau_in_state:
            torque_tokens = self._shift_torque_sequence(torque) if self.use_shifted_tau_tokens else torque
            state = torch.cat([qpos, mom, torque_tokens], dim=-1)
        else:
            state = torch.cat([qpos, mom], dim=-1)
        
        B, T, _ = state.shape
        time_indices = torch.arange(T, device=state.device, dtype=torch.long)
        
        diffusion_t = torch.randint(1, self.diffusion_steps + 1, (1,)).item()
        
        if self.unconditional_tau_in_state:
            if self.backbone == "transformer":
                tokens = self.build_tokens(state, diffusion_t, time_indices=time_indices)
                if self.query_context_mode == "clean_prefix_noisy_suffix":
                    loss_mask = torch.zeros(T, device=state.device, dtype=torch.bool)
                    suffix_start = torch.randint(1, T, (1,), device=state.device).item() if T > 1 else 0
                    loss_mask[suffix_start:] = True
                    noisy_tokens, noise = self._apply_state_noise_with_mask(tokens, diffusion_t, loss_mask)
                else:
                    noisy_tokens, noise = self.apply_noise(tokens, diffusion_t, return_noise=True)
                    loss_mask = None
                predictions = self.model(noisy_tokens)
                noise_target = noise
            else:
                noisy_contexts, noisy_queries, _, noise_target, loss_mask, _, _ = self._build_perceiver_train_views(
                    state, torque=None, diffusion_t=diffusion_t, time_indices=time_indices
                )
                predictions = self.model(noisy_contexts, noisy_queries, torque=None)
        else:
            # Normalize torque for conditioning
            torque_norm = self.normalize_cond(torque)

            if self.backbone == "transformer":
                tokens = self.build_tokens(
                    state, diffusion_t, torque=torque_norm, include_torque=True, time_indices=time_indices
                )
                if self.query_context_mode == "clean_prefix_noisy_suffix":
                    loss_mask = torch.zeros(T, device=state.device, dtype=torch.bool)
                    suffix_start = torch.randint(1, T, (1,), device=state.device).item() if T > 1 else 0
                    loss_mask[suffix_start:] = True
                    noisy_tokens, noise = self._apply_state_noise_with_mask(tokens, diffusion_t, loss_mask)
                else:
                    noisy_tokens, noise = self.apply_noise(tokens, diffusion_t, return_noise=True)
                    loss_mask = None
                predictions = self.model(noisy_tokens)
                noise_target = noise
            else:
                noisy_contexts, noisy_queries, torque_for_queries, noise_target, loss_mask, _, _ = (
                    self._build_perceiver_train_views(
                        state,
                        torque=torque_norm,
                        diffusion_t=diffusion_t,
                        time_indices=time_indices,
                    )
                )
                predictions = self.model(noisy_contexts, noisy_queries, torque_for_queries)
        loss = self._compute_denoise_loss(predictions, noise_target, loss_mask=loss_mask)
        
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
        if strict and hasattr(self.model, "c0") and "model.c0" not in state_dict:
            # Older full-model checkpoints predate the learnable c0 token. The current
            # default initialization is zero, so loading without that key is safe.
            print("[TrajectoryDPF] Checkpoint is missing model.c0. Loading with strict=False and using default zero init.")
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
    
