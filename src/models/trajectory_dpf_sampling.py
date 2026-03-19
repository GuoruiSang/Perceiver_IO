"""Generic diffusion sampling helpers for Trajectory DPF."""

import numpy as np
import mujoco
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Tuple

from scripts.data.generate_dataset_forward import (
    generate_torque_sequence,
    parse_torque_policies,
)


class TrajectoryDPFSampling:
    def _expand_sampling_tensor(
        self,
        tensor: Optional[torch.Tensor],
        num_samples: int,
        trajectory_length: int,
        expected_last_dim: int,
        name: str,
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.ndim != 3:
            raise ValueError(f"{name} must have shape [B,T,{expected_last_dim}], got {tuple(tensor.shape)}")
        if tensor.shape[-1] != expected_last_dim:
            raise ValueError(
                f"{name} last dim must equal {expected_last_dim}, got {tensor.shape[-1]}"
            )
        if tensor.shape[1] < trajectory_length:
            raise ValueError(
                f"{name} must have at least trajectory_length={trajectory_length} timesteps, got {tensor.shape[1]}"
            )
        if tensor.shape[0] == 1 and num_samples > 1:
            tensor = tensor.expand(num_samples, -1, -1)
        elif tensor.shape[0] != num_samples:
            raise ValueError(
                f"{name} batch dim must be 1 or num_samples={num_samples}, got {tensor.shape[0]}"
            )
        return tensor[:, :trajectory_length, :].to(device=self.device, dtype=torch.float32)

    def _build_sampling_state(
        self,
        qpos: torch.Tensor,
        mom: torch.Tensor,
        torque: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.unconditional_tau_in_state:
            if torque is None:
                raise ValueError("torque is required when conditioning_mode concatenates torque into state")
            torque_tokens = self._shift_torque_sequence(torque) if self.use_shifted_tau_tokens else torque
            return torch.cat([qpos, mom, torque_tokens], dim=-1)
        return torch.cat([qpos, mom], dim=-1)

    def _apply_observed_prefix_constraint(
        self,
        x: torch.Tensor,
        observed_prefix_state_norm: Optional[torch.Tensor],
        prefix_len: int,
    ) -> torch.Tensor:
        if observed_prefix_state_norm is None or prefix_len <= 0:
            return x
        x[:, :prefix_len, :] = observed_prefix_state_norm[:, :prefix_len, :]
        return x

    def _build_sampling_context_indices(
        self,
        seq_len: int,
        subset_len: int,
        device: torch.device,
        strategy: str = "prefix",
    ) -> torch.Tensor:
        subset_len = max(1, min(int(subset_len), int(seq_len)))
        if strategy == "prefix":
            return torch.arange(subset_len, device=device, dtype=torch.long)
        if strategy == "query_subset":
            idx = torch.randperm(seq_len, device=device)[:subset_len]
            return torch.sort(idx).values
        raise ValueError(f"Unsupported context selection strategy: {strategy}")

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
        sampling_torque_policy: str = "sinusoidal",
        sampling_torque_mix: str = "sinusoidal:0.32,lpf_uniform:0.25,drift_ou:0.43",
        sampling_lpf_uniform_beta: float = 0.9992,
        sampling_torque_scale: float = 0.35,
    ) -> torch.Tensor:
        """
        Generate torque for sampling.

        Policies:
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
            torque_kwargs = {}
            if p in {"drift_ou", "lpf_uniform"}:
                torque_kwargs["ctrl_margin"] = 0.05
            if p == "lpf_uniform":
                torque_kwargs["lpf_uniform_beta"] = float(sampling_lpf_uniform_beta)
            torque = generate_torque_sequence(
                model=model,
                num_steps=trajectory_length,
                skip_steps=skip_steps,
                policy=p,
                ref_std_per_dim=None,
                **torque_kwargs,
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


    def sample_trajectories(
        self,
        num_samples: int,
        trajectory_length: int,
        num_diffusion_steps: int = None,
        context_fraction: float = 0.7,
        sample_mode: Optional[str] = None,
        prefix_len: Optional[int] = None,
        observed_qpos: torch.Tensor = None,
        observed_mom: torch.Tensor = None,
        observed_torque: torch.Tensor = None,
        time_indices: Optional[torch.Tensor] = None,
        use_ema: bool = True,
        sampler: str = "ddim",
        # CFG parameters
        guidance_scale: float = 1.0,  # CFG scale (1.0 = no CFG, >1.0 = stronger conditioning)
        # HNN guidance parameters
        hnn: nn.Module = None,
        guidance_method: str = "strategy2",
        guidance_hamres_smooth_sigma: float = 1.0,
        guidance_hamres_delta: float = 1.0,
        guidance_hamres_min_scale_q: float = 1e-3,
        guidance_hamres_min_scale_p: float = 1e-3,
        guidance_trust_lambda: float = 0.0,
        dt: Optional[float] = None,
        # Torque generation parameters
        torque: torch.Tensor = None,  # Optional: provide torque directly
        initial_noise: torch.Tensor = None,  # Optional: fixed initial noise for reproducible sampling
        sampling_torque_policy: str = "sinusoidal",
        sampling_torque_mix: str = "sinusoidal:0.32,lpf_uniform:0.25,drift_ou:0.43",
        sampling_lpf_uniform_beta: float = 0.9992,
        sampling_torque_scale: float = 0.35,
        # Temporal smoothing
        smooth_sigma: float = 0.0,  # Gaussian smoothing sigma (0 = disabled, 1-3 recommended)
        smooth_guidance_only: bool = False,  # If True, smooth only for guidance input; output stays unsmoothed
        smooth_last_step_only: bool = False,  # If True, only smooth at the final diffusion step
        alpha_q: float = 1e-4,  # Normalized SGD step size for q
        alpha_p: float = 1e-4,  # Normalized SGD step size for p
        guidance_normalize_grad: bool = True,  # Strategy 2: normalize guidance gradients
        guidance_joint_update: bool = False,  # Strategy 2: share one norm across q/p
        guidance_num_candidates: int = 16,  # Strategy 1 particle count
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
            sampler: sampling method ('ddpm' or 'ddim')
            guidance_scale: CFG scale (1.0 = no CFG, >1.0 = stronger conditioning)
            hnn: HNN for physics-based guidance
            guidance_method: strategy selector
                - 'strategy1': sample 16 stochastic x_{t-1} candidates from x_t (eta=1),
                  score corresponding x0 candidates with robust HamRes, and sigmoid-sample one
                - 'strategy2': one-step normalized guidance at each sampling step
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

        resolved_sample_mode = (
            sample_mode
            if sample_mode is not None
            else (
                "observed_prefix_completion"
                if getattr(self, "query_context_mode", "random_subset") == "clean_prefix_noisy_suffix"
                else "generic_full_trajectory"
            )
        )
        if resolved_sample_mode not in {"generic_full_trajectory", "observed_prefix_completion"}:
            raise ValueError(f"Unsupported sample_mode: {resolved_sample_mode}")

        observed_qpos = self._expand_sampling_tensor(
            observed_qpos, num_samples, trajectory_length, self.qpos_dim, "observed_qpos"
        )
        observed_mom = self._expand_sampling_tensor(
            observed_mom, num_samples, trajectory_length, self.mom_dim, "observed_mom"
        )
        observed_torque = self._expand_sampling_tensor(
            observed_torque, num_samples, trajectory_length, self.torque_dim, "observed_torque"
        )
        if time_indices is not None:
            time_indices = time_indices.to(device=device, dtype=torch.long)
            if time_indices.ndim != 1 or time_indices.shape[0] != trajectory_length:
                raise ValueError(
                    f"time_indices must have shape ({trajectory_length},), got {tuple(time_indices.shape)}"
                )

        if resolved_sample_mode == "observed_prefix_completion":
            if observed_qpos is None or observed_mom is None:
                print(
                    "[Sampling] Prefix completion requested but observed_qpos/observed_mom were not provided; "
                    "falling back to generic full-trajectory sampling."
                )
                resolved_sample_mode = "generic_full_trajectory"
            elif self.unconditional_tau_in_state and observed_torque is None:
                print(
                    "[Sampling] Prefix completion in concat-state mode requires observed_torque; "
                    "falling back to generic full-trajectory sampling."
                )
                resolved_sample_mode = "generic_full_trajectory"
            else:
                prefix_len = (
                    max(1, min(trajectory_length - 1, int(round(context_fraction * trajectory_length))))
                    if prefix_len is None
                    else int(prefix_len)
                )
                prefix_len = max(1, min(prefix_len, trajectory_length - 1))
        else:
            prefix_len = 0
        
        # Generate or use provided torque conditioning unless torque is concatenated into state.
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
            if observed_torque is not None:
                torque = observed_torque

        # Start with pure noise for state (qpos, mom)
        if initial_noise is not None:
            x = initial_noise.to(device)
        else:
            x = torch.randn(num_samples, trajectory_length, self.state_dim, device=device)

        observed_prefix_state_norm = None
        if resolved_sample_mode == "observed_prefix_completion":
            observed_state = self._build_sampling_state(
                observed_qpos,
                observed_mom,
                torque=observed_torque if self.unconditional_tau_in_state else None,
            )
            observed_prefix_state_norm = self.normalize_state(observed_state)
            x = self._apply_observed_prefix_constraint(x, observed_prefix_state_norm, prefix_len)
        
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
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
        
        if self.unconditional_tau_in_state:
            print("[Sampling] Concat-state mode: torque is part of state; CFG/conditioning disabled.")
        else:
            print(f"[Sampling] CFG guidance_scale={guidance_scale}")
        
        context_idx = None
        if self.backbone != "transformer":
            # PREFIX context: use first num_context timesteps (not random)
            # IMPORTANT: Cap context length to training max to avoid OOD encoder behavior when extending
            if resolved_sample_mode == "observed_prefix_completion":
                num_context = prefix_len
                context_idx = self._build_sampling_context_indices(
                    trajectory_length,
                    num_context,
                    device=device,
                    strategy="query_subset",
                )
                print(
                    f"[Sampling] Prefix completion mode with observed prefix_len={prefix_len} "
                    f"and query-subset context size={num_context}"
                )
            else:
                max_context_train = int(self.max_timesteps * context_fraction)
                num_context = max(1, min(trajectory_length - 1, max_context_train))
                context_idx = self._build_sampling_context_indices(
                    trajectory_length,
                    num_context,
                    device=device,
                    strategy="prefix",
                )
                print(f"[Sampling] Context length: {num_context} (capped at {max_context_train} from training length {self.max_timesteps})")

        # Strategy1-only cached views reused across diffusion steps.
        strategy1_num_cands = max(2, int(guidance_num_candidates))
        strategy1_cond_flat = None
        strategy1_cond_uncond_flat = None
        strategy1_tau_flat = None
        strategy1_uniform = None
        if hnn is not None and guidance_method == "strategy1" and (not self.unconditional_tau_in_state):
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
            x_in = self._apply_observed_prefix_constraint(x_in, observed_prefix_state_norm, prefix_len)
            if self.unconditional_tau_in_state:
                tokens = self.build_tokens(
                    x_in,
                    timestep_int + 1,
                    skip_normalize=True,
                    time_indices=time_indices,
                )
                with torch.no_grad():
                    if self.backbone == "transformer":
                        return self.model(tokens)
                    contexts_local = tokens.index_select(dim=1, index=context_idx)
                    return self.model(contexts_local, tokens, torque=None)
            if self.backbone == "transformer":
                cond_tokens = self.build_tokens(
                    x_in,
                    timestep_int + 1,
                    skip_normalize=True,
                    torque=cond_in,
                    include_torque=True,
                    time_indices=time_indices,
                )
                with torch.no_grad():
                    eps_cond_local = self.model(cond_tokens)
                if guidance_scale != 1.0:
                    uncond_tokens = self.build_tokens(
                        x_in,
                        timestep_int + 1,
                        skip_normalize=True,
                        torque=cond_uncond_in,
                        include_torque=True,
                        time_indices=time_indices,
                    )
                    with torch.no_grad():
                        eps_uncond_local = self.model(uncond_tokens)
                    return eps_uncond_local + guidance_scale * (eps_cond_local - eps_uncond_local)
                return eps_cond_local

            queries_local = self.build_tokens(
                x_in,
                timestep_int + 1,
                skip_normalize=True,
                time_indices=time_indices,
            )
            contexts_local = queries_local.index_select(dim=1, index=context_idx)
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

            q_prev = x0_prev_phys[:, :, :self.qpos_dim]
            p_prev = x0_prev_phys[:, :, self.qpos_dim:self.qpos_dim + self.mom_dim]

            residual = compute_hnn_robust_hamres_energy(
                q_prev,
                p_prev,
                tau_prev,
                hnn,
                self.data_dt,
                qpos_representation=self.qpos_representation,
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

            return x_prev_cands[bidx, chosen]

        for i, t in enumerate(tqdm(ts, total=len(ts), desc="Sampling")):
            x = self._apply_observed_prefix_constraint(x, observed_prefix_state_norm, prefix_len)
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
                    if guidance_method == "strategy1":
                        if is_last:
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
                        x = self._apply_observed_prefix_constraint(x, observed_prefix_state_norm, prefix_len)
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
                            qpos_representation=self.qpos_representation,
                            alpha_q=alpha_q * noise_step_scale,
                            alpha_p=alpha_p * noise_step_scale,
                            guidance_trust_lambda=guidance_trust_lambda,
                            guidance_normalize_grad=guidance_normalize_grad,
                            guidance_joint_update=guidance_joint_update,
                        )
                    else:
                        raise ValueError(
                            f"Unknown guidance_method={guidance_method}. "
                            "Valid: {'strategy1', 'strategy2'}"
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
                x = self._apply_observed_prefix_constraint(x, observed_prefix_state_norm, prefix_len)

        # Denormalize state
        x = self._apply_observed_prefix_constraint(x, observed_prefix_state_norm, prefix_len)
        state = self.denormalize_state(x)
        if self.unconditional_tau_in_state:
            torque_tokens = state[:, :, self.qpos_dim + self.mom_dim:self.qpos_dim + self.mom_dim + self.torque_dim]
            torque = (
                self._shifted_token_torque_to_rollout_torque(torque_tokens)
                if getattr(self, "use_shifted_tau_tokens", False)
                else torque_tokens
            )

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
    
