"""
Model utilities for Trajectory DPF.

This module contains utility classes and functions used in the model implementation:
- EMA: Exponential Moving Average for improved sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import os
import mujoco

from src.qpos_representation import (
    REACHER_Q0Q1_SINCOS,
    decode_qpos_array,
    decode_qpos_tensor,
)

class EMA:
    """Exponential Moving Average (EMA) helper (not an nn.Module).

    - Keeps a moving average "shadow" of model parameters for evaluation/sampling.
    - Not registered as a submodule; does NOT affect model state_dict or trainer summaries.
    - API compatible with previous usage: update(model=None), store(model=None), copy_to(model=None), restore(model=None).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.model = model
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self._create_shadow()

    def _create_shadow(self) -> None:
        """Initialize shadow parameters for all trainable parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model=None) -> None:
        """Update EMA shadow parameters after each training batch."""
        _ = model  # Kept for backward-compatible call sites.
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone().detach()
                else:
                    shadow = self.shadow[name]
                    if shadow.device != param.data.device or shadow.dtype != param.data.dtype:
                        shadow = shadow.to(device=param.data.device, dtype=param.data.dtype)
                        self.shadow[name] = shadow
                    new_average = (1.0 - self.decay) * param.data + self.decay * shadow
                    self.shadow[name] = new_average.clone().detach()

    def apply_shadow(self) -> None:
        """Apply EMA shadow weights to the model for inference; backs up originals."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.backup:
                    self.backup[name] = param.data.clone().detach()
                shadow = self.shadow[name]
                if shadow.device != param.data.device or shadow.dtype != param.data.dtype:
                    shadow = shadow.to(device=param.data.device, dtype=param.data.dtype)
                    self.shadow[name] = shadow
                param.data = shadow.clone().detach()

    def store(self, model=None) -> None:
        """Alias for apply_shadow (kept for API compatibility)."""
        _ = model  # Kept for backward-compatible call sites.
        self.apply_shadow()

    def copy_to(self, model=None) -> None:
        """Alias for apply_shadow (kept for API compatibility)."""
        _ = model  # Kept for backward-compatible call sites.
        self.apply_shadow()

    def restore(self, model=None) -> None:
        """Restore original model weights after inference."""
        _ = model  # Kept for backward-compatible call sites.
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone().detach()
        self.backup = {}


def visualize_trajectory(trajectory: dict, save_path: str, name: str = 'trajectory'):

    keys = list(trajectory.keys())
    values = list(trajectory.values())

    nrows = len(keys)
    ncols = max([value.shape[-1] for value in values])
    
    if isinstance(values[0], torch.Tensor):
        values = [value.detach().cpu().numpy() for value in values]
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(30, 8))

    # Use actual trajectory length
    trajectory_length = values[0].shape[0]
    t = np.arange(trajectory_length)

    # Define distinct colors for each key
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i in range(nrows):
        num_dims = values[i].shape[-1]
        color = colors[i % len(colors)]  # Same color for all dimensions of this key
        for j in range(ncols):
            if j < num_dims:
                data = values[i][:, j]
                axes[i, j].scatter(t, data, label=f'Dimension {j} of {keys[i]}', s=1, color=color)
                axes[i, j].legend()
                # Auto-scale y-axis with some padding
                data_min, data_max = data.min(), data.max()
                margin = (data_max - data_min) * 0.1 + 0.1  # 10% margin + small constant
                axes[i, j].set_ylim(data_min - margin, data_max + margin)
            else:
                axes[i, j].set_visible(False)  # Hide unused subplots

    fig.savefig(os.path.join(save_path, f'{name}.jpg'))
    plt.close(fig)


def _unwrap_qpos_for_plotting(qpos: np.ndarray, qpos_representation: str) -> np.ndarray:
    """Unwrap periodic angle trajectories for visualization only."""
    if qpos_representation != REACHER_Q0Q1_SINCOS:
        return qpos
    return np.unwrap(qpos, axis=0)


def central_difference(seq: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Compute time derivative using central difference.
    
    Args:
        seq: [B, T, dim] sequence
        dt: timestep
        
    Returns:
        seq_dot: [B, T, dim] time derivative
    """
    seq_dot = torch.zeros_like(seq)
    # Central difference for interior points
    seq_dot[:, 1:-1] = (seq[:, 2:] - seq[:, :-2]) / (2 * dt)
    # Forward difference for first point (second-order accurate)
    seq_dot[:, 0] = (-3*seq[:, 0] + 4*seq[:, 1] - seq[:, 2]) / (2*dt)
    # Backward difference for last point (second-order accurate)
    seq_dot[:, -1] = (3*seq[:, -1] - 4*seq[:, -2] + seq[:, -3]) / (2*dt)
    return seq_dot


def forward_difference(seq: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Compute time derivative using forward difference.

    More aligned with Euler integration: (seq[i+1] - seq[i]) / dt
    Avoids mixing future/past information that central difference uses.

    Args:
        seq: [B, T, dim] sequence
        dt: timestep

    Returns:
        seq_dot: [B, T, dim] time derivative
    """
    seq_dot = torch.zeros_like(seq)
    # Forward difference for all but last point
    seq_dot[:, :-1] = (seq[:, 1:] - seq[:, :-1]) / dt
    # Copy last valid derivative for the final point
    seq_dot[:, -1] = seq_dot[:, -2]
    return seq_dot


def _gaussian_smooth_time_3d(seq: torch.Tensor, sigma: float) -> torch.Tensor:
    """Depthwise Gaussian smoothing over time for [B, T, D] tensors."""
    if sigma <= 0:
        return seq

    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=seq.device, dtype=seq.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # [B, T, D] -> [B, D, T] for grouped conv over time.
    seq_bdt = seq.transpose(1, 2)
    seq_pad = F.pad(seq_bdt, (radius, radius), mode="reflect")
    weight = kernel.view(1, 1, -1).repeat(seq.shape[-1], 1, 1)
    smoothed = F.conv1d(seq_pad, weight, groups=seq.shape[-1])
    return smoothed.transpose(1, 2)


def _pseudo_huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    return (delta ** 2) * (torch.sqrt(1.0 + (x / delta) ** 2) - 1.0)


def _expand_var_to_dim(var_like, dim: int, device, dtype):
    """Expand variance-like scalar/vector to per-dimension std; return None if unavailable."""
    if var_like is None:
        return None
    var_t = torch.as_tensor(var_like, device=device, dtype=dtype).flatten()
    if var_t.numel() == 1:
        return torch.sqrt(torch.clamp(var_t.repeat(dim), min=0.0))
    if var_t.numel() >= dim:
        return torch.sqrt(torch.clamp(var_t[:dim], min=0.0))
    return None


def compute_hnn_physics_energy(
    seq_qpos: torch.Tensor,
    seq_mom: torch.Tensor,
    seq_torque: torch.Tensor,
    hnn: nn.Module,
    dt: float,
    qpos_representation: str = "raw",
    use_forward_diff: bool = False,
) -> torch.Tensor:
    """
    Compute HNN-based physics consistency energy via 1-step prediction.

    Uses HNN to predict (q_{t+1}, p_{t+1}) from (q_t, p_t) via symplectic Euler,
    then compares with the actual next state in the trajectory.

    E = (1/(T-1)) sum_t |q_{t+1} - q̂_{t+1}|² + |p_{t+1} - p̂_{t+1}|²

    where:
        q̂_{t+1} = q_t + dt * dH/dp(q_t, p_t)
        p̂_{t+1} = p_t + dt * (-dH/dq(q_t, p_t) + τ_t)

    Args:
        seq_qpos: [B, T, qpos_dim] position trajectory
        seq_mom: [B, T, mom_dim] momentum trajectory
        seq_torque: [B, T, torque_dim] torque sequence
        hnn: Trained Hamiltonian Neural Network
        dt: timestep between trajectory points
        use_forward_diff: unused (kept for API compatibility)

    Returns:
        energy: scalar energy value
    """
    B, T, _ = seq_mom.shape

    seq_qpos_phys = decode_qpos_tensor(seq_qpos, qpos_representation)

    # Use t=0..T-2 as "current" states
    q_t = seq_qpos_phys[:, :-1]   # [B, T-1, qpos_dim]
    p_t = seq_mom[:, :-1]    # [B, T-1, mom_dim]
    tau_t = seq_torque[:, :-1]  # [B, T-1, torque_dim]

    # Actual next states
    q_next = seq_qpos_phys[:, 1:]  # [B, T-1, qpos_dim]
    p_next = seq_mom[:, 1:]   # [B, T-1, mom_dim]

    # Compute HNN gradients (flatten for StructuredHNN compatibility)
    # Detach from upstream graph; no clone needed for value-only guidance.
    p_flat = p_t.reshape(-1, p_t.shape[-1]).detach().requires_grad_(True)
    q_flat = q_t.reshape(-1, q_t.shape[-1]).detach().requires_grad_(True)

    H = hnn(p_flat, q_flat)  # [B*(T-1), 1]

    dH_dp, dH_dq = torch.autograd.grad(
        H.sum(),
        (p_flat, q_flat),
        create_graph=True
    )

    # Reshape back to [B, T-1, dim]
    dH_dp = dH_dp.reshape(B, T - 1, -1)
    dH_dq = dH_dq.reshape(B, T - 1, -1)

    # 1-step symplectic Euler prediction
    q_pred = q_t + dt * dH_dp                    # q̂_{t+1}
    p_pred = p_t + dt * (-dH_dq + tau_t)         # p̂_{t+1}

    # MSE between predicted and actual next state
    energy = ((q_next - q_pred) ** 2).sum(dim=-1) + ((p_next - p_pred) ** 2).sum(dim=-1)
    return energy.mean()  # mean over B × (T-1)


def compute_hnn_robust_hamres_energy(
    seq_qpos: torch.Tensor,
    seq_mom: torch.Tensor,
    seq_torque: torch.Tensor,
    hnn: nn.Module,
    dt: float,
    qpos_representation: str = "raw",
    smooth_sigma: float = 1.0,
    delta: float = 1.0,
    min_scale_q: float = 1e-3,
    min_scale_p: float = 1e-3,
    reduction: str = "mean",
    create_graph: bool = True,
) -> torch.Tensor:
    """
    Robust HamRes-style differentiable energy for guidance.

    Uses central-difference residuals with optional Gaussian smoothing, per-dimension
    normalization, and pseudo-Huber penalty. Aggregates with mean over time for stable
    gradients (evaluation can still use median aggregation).
    """
    B, T, _ = seq_qpos.shape
    if T < 3:
        # Keep graph-connected scalar.
        return seq_qpos.new_zeros(())

    seq_qpos_phys = decode_qpos_tensor(seq_qpos, qpos_representation)
    q_use = _gaussian_smooth_time_3d(seq_qpos_phys, smooth_sigma)
    p_use = _gaussian_smooth_time_3d(seq_mom, smooth_sigma)

    qdot = (q_use[:, 2:] - q_use[:, :-2]) / (2 * dt)
    pdot = (p_use[:, 2:] - p_use[:, :-2]) / (2 * dt)
    q_mid = q_use[:, 1:-1]
    p_mid = p_use[:, 1:-1]
    tau_mid = seq_torque[:, 1:-1]

    # Same stop-grad convention as one-step guidance energy.
    p_flat = p_mid.reshape(-1, p_mid.shape[-1]).detach().requires_grad_(True)
    q_flat = q_mid.reshape(-1, q_mid.shape[-1]).detach().requires_grad_(True)
    H = hnn(p_flat, q_flat)
    dH_dp, dH_dq = torch.autograd.grad(H.sum(), (p_flat, q_flat), create_graph=create_graph)
    dH_dp = dH_dp.reshape(B, T - 2, -1)
    dH_dq = dH_dq.reshape(B, T - 2, -1)

    r_q = qdot - dH_dp
    r_p = pdot - (-dH_dq + tau_mid)

    q_dim, p_dim = r_q.shape[-1], r_p.shape[-1]
    var_dq = getattr(hnn, "qvel_var", None)
    var_dp = getattr(hnn, "mom_dot_var", None)
    scale_q = _expand_var_to_dim(var_dq, q_dim, r_q.device, r_q.dtype)
    scale_p = _expand_var_to_dim(var_dp, p_dim, r_p.device, r_p.dtype)
    if scale_q is None:
        scale_q = torch.sqrt(torch.clamp(qdot.var(dim=(0, 1), unbiased=False), min=0.0))
    if scale_p is None:
        scale_p = torch.sqrt(torch.clamp(pdot.var(dim=(0, 1), unbiased=False), min=0.0))
    scale_q = torch.clamp(scale_q, min=min_scale_q)
    scale_p = torch.clamp(scale_p, min=min_scale_p)

    r_q_norm = r_q / scale_q.view(1, 1, -1)
    r_p_norm = r_p / scale_p.view(1, 1, -1)

    # Mean over dimensions then time; keep per-batch values for particle sampling strategy.
    per_t = _pseudo_huber(r_q_norm, delta).mean(dim=-1) + _pseudo_huber(r_p_norm, delta).mean(dim=-1)
    per_sample = per_t.mean(dim=1)
    if reduction == "none_batch":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


def _add_trust_regularizer(
    energy: torch.Tensor,
    seq_qpos: torch.Tensor,
    seq_mom: torch.Tensor,
    q_ref: torch.Tensor,
    p_ref: torch.Tensor,
    trust_lambda: float,
) -> torch.Tensor:
    if trust_lambda <= 0:
        return energy
    return energy + trust_lambda * (((seq_qpos - q_ref) ** 2).mean() + ((seq_mom - p_ref) ** 2).mean())


def run_one_step_guidance_hnn(
    x: torch.Tensor,
    seq_torque: torch.Tensor,
    qpos_dim: int,
    mom_dim: int,
    dt: float,
    hnn: nn.Module,
    qpos_representation: str = "raw",
    alpha_q: float = 1e-4,
    alpha_p: float = 1e-4,
    guidance_trust_lambda: float = 0.0,
    guidance_normalize_grad: bool = True,
    guidance_joint_update: bool = False,
) -> torch.Tensor:
    """
    Strategy 2:
    Apply one normalized gradient step (one-step energy) to the current trajectory.
    """
    seq_qpos = x[:, :, :qpos_dim].clone().requires_grad_(True)
    seq_mom = x[:, :, qpos_dim:qpos_dim + mom_dim].clone().requires_grad_(True)
    q_ref = x[:, :, :qpos_dim].detach()
    p_ref = x[:, :, qpos_dim:qpos_dim + mom_dim].detach()

    energy = compute_hnn_physics_energy(
        seq_qpos,
        seq_mom,
        seq_torque,
        hnn,
        dt,
        qpos_representation=qpos_representation,
        use_forward_diff=False,
    )
    energy = _add_trust_regularizer(energy, seq_qpos, seq_mom, q_ref, p_ref, guidance_trust_lambda)
    grad_q, grad_p = torch.autograd.grad(energy, [seq_qpos, seq_mom])

    eps = 1e-12
    if guidance_joint_update:
        # Joint update: share one per-sample scale across q and p.
        if guidance_normalize_grad:
            grad_joint = torch.cat([grad_q, grad_p], dim=-1)
            norm_joint = grad_joint.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1).clamp_min(eps)
            grad_q_use = grad_q / norm_joint
            grad_p_use = grad_p / norm_joint
        else:
            grad_q_use = grad_q
            grad_p_use = grad_p
    else:
        # Separate update: q and p each use their own scale.
        if guidance_normalize_grad:
            norm_q = grad_q.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1).clamp_min(eps)
            norm_p = grad_p.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1).clamp_min(eps)
            grad_q_use = grad_q / norm_q
            grad_p_use = grad_p / norm_p
        else:
            grad_q_use = grad_q
            grad_p_use = grad_p

    q_new = seq_qpos - alpha_q * grad_q_use
    p_new = seq_mom - alpha_p * grad_p_use
    return torch.cat([q_new.detach(), p_new.detach()], dim=-1)


def _align_generated_and_reconstructed(
    generated: dict[str, np.ndarray],
    reconstructed: dict[str, np.ndarray],
    alignment: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if alignment == 'pre_step':
        gen_qpos = generated['seq_qpos']
        gen_mom = generated['seq_mom']
        gen_tau = generated['seq_torque']
        recon_qpos = reconstructed['seq_qpos']
        recon_mom = reconstructed['seq_mom']
        recon_tau = reconstructed['seq_torque']
    elif alignment == 'post_step':
        gen_qpos = generated['seq_qpos'][1:]
        gen_mom = generated['seq_mom'][1:]
        gen_tau = generated['seq_torque'][1:]
        recon_qpos = reconstructed['seq_qpos']
        recon_mom = reconstructed['seq_mom']
        recon_tau = reconstructed['seq_torque']
    else:
        raise ValueError(f"Unknown trajectory alignment: {alignment}")

    t_min = min(
        len(gen_qpos), len(gen_mom), len(gen_tau),
        len(recon_qpos), len(recon_mom), len(recon_tau),
    )
    return (
        gen_qpos[:t_min],
        gen_mom[:t_min],
        gen_tau[:t_min],
        recon_qpos[:t_min],
        recon_mom[:t_min],
        recon_tau[:t_min],
    )


def compare_generated_with_reconstructed(
    generated: dict, mujoco_model_path: str, save_path: str, dt: float = 0.0005, data_dt: float = None,
    name: str = None, trajectory_alignment: str = 'pre_step', return_series: bool = False,
    prefix_len: int = 0,
    qpos_representation: str = "raw",
    highlight_indices: dict[str, int] | None = None,
    plot_dpi: int = 300,
) -> dict:
    """
    Compare generated trajectory with physics-reconstructed trajectory.
    
    Works with new data format: generated = {'seq_qpos', 'seq_mom', 'seq_torque'}
    Computes initial velocity from momentum using MuJoCo mass matrix.
    
    Args:
        generated: dict with 'seq_qpos', 'seq_mom', 'seq_torque' at data_dt resolution
        mujoco_model_path: Path to MuJoCo XML model
        save_path: Directory to save comparison plot
        dt: Fine simulation timestep
        data_dt: Data collection timestep (default: dt for backwards compatibility)
        name: Name for the output file (if None, skip plotting)
        trajectory_alignment: 'pre_step' for synchronized datasets, 'post_step' for legacy datasets
    
    Returns:
        dict with MSE values: {'mse_qpos': float, 'mse_mom': float, 'mse_total': float}
        If return_series=True, also returns aligned generated/reconstructed arrays.
    """
    import mujoco
    
    # Default data_dt to dt for backwards compatibility
    if data_dt is None:
        data_dt = dt
    
    print(f"[Compare] Starting compare_generated_with_reconstructed(name={name}, alignment={trajectory_alignment})")

    # Convert to numpy if needed
    gen = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in generated.items()}
    
    # Load MuJoCo model
    print(f"[Compare] Loading MuJoCo model from {mujoco_model_path}")
    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    data = mujoco.MjData(model)
    
    gen_qpos_raw = decode_qpos_array(gen['seq_qpos'], qpos_representation)
    qpos_dim = int(gen_qpos_raw.shape[-1])
    mom_dim = int(gen['seq_mom'].shape[-1])

    # Compute initial velocity from initial momentum: v = M^{-1} @ p
    print("[Compare] Computing initial velocity from momentum")
    data.qpos[:] = 0.0
    data.qpos[:qpos_dim] = gen_qpos_raw[0]
    data.qvel[:] = 0  # Temporary
    mujoco.mj_forward(model, data)
    
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    initial_qvel = np.zeros(model.nv, dtype=np.float64)
    initial_qvel[:mom_dim] = np.linalg.solve(M[:mom_dim, :mom_dim], gen['seq_mom'][0])
    
    # Reconstruct using MuJoCo physics (with data_dt support)
    print(f"[Compare] Reconstructing trajectory for {len(gen['seq_qpos'])} steps")
    recon = reconstruct_traj_with_momentum(
        model, len(gen['seq_qpos']), dt,
        gen_qpos_raw[0], initial_qvel, gen['seq_torque'],
        data_dt=data_dt
    )
    print("[Compare] Reconstruction finished")
    
    recon_qpos = recon['seq_qpos'][..., :qpos_dim]
    recon_mom = recon['seq_mom'][..., :mom_dim]
    gen_qpos, gen_mom, gen_tau, recon_qpos, recon_mom, recon_tau = _align_generated_and_reconstructed(
        {
            'seq_qpos': gen_qpos_raw,
            'seq_mom': gen['seq_mom'],
            'seq_torque': gen['seq_torque'],
        },
        {
            'seq_qpos': recon_qpos,
            'seq_mom': recon_mom,
            'seq_torque': recon['seq_torque'],
        },
        trajectory_alignment,
    )
    mse_qpos = np.mean((gen_qpos - recon_qpos) ** 2)
    mse_mom = np.mean((gen_mom - recon_mom) ** 2)
    mse_total = mse_qpos + mse_mom
    gen_qpos_plot = _unwrap_qpos_for_plotting(gen_qpos, qpos_representation)
    recon_qpos_plot = _unwrap_qpos_for_plotting(recon_qpos, qpos_representation)
    
    # Only create plot if name is provided
    if name is not None:
        print("[Compare] Building matplotlib figure")
        keys = ['seq_qpos', 'seq_mom', 'seq_torque']
        nrows, ncols = len(keys), max(v.shape[-1] for v in gen.values())
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(30, 10),
            dpi=int(plot_dpi),
            constrained_layout=True,
        )
        
        # Add MSE info to the figure title
        fig.suptitle(f'MSE: qpos={mse_qpos:.6f}, mom={mse_mom:.6f}, total={mse_total:.6f}', fontsize=14, y=1.02)
        
        prefix_len = max(0, int(prefix_len))
        highlight_items = list(highlight_indices.items()) if highlight_indices else []
        highlight_colors = ['darkgreen', 'darkorange', 'purple', 'teal']
        for i, key in enumerate(keys):
            if key == 'seq_qpos':
                gen_data = gen_qpos_plot
                recon_data = recon_qpos_plot
            elif key == 'seq_mom':
                gen_data = gen_mom
                recon_data = recon_mom
            else:
                gen_data = gen_tau
                recon_data = recon_tau
            t = np.arange(len(gen_data))
            for j in range(gen_data.shape[-1]):
                if j < ncols:
                    prefix_end = min(prefix_len, len(gen_data))
                    if prefix_end > 0:
                        axes[i, j].scatter(
                            t[:prefix_end], gen_data[:prefix_end, j], s=1, c='black', label='Prefix', alpha=0.9
                        )
                    if key == 'seq_torque':
                        axes[i, j].scatter(
                            t[prefix_end:],
                            gen_data[prefix_end:, j],
                            s=1,
                            c='crimson',
                            label='Applied torque',
                            alpha=0.7,
                        )
                    else:
                        axes[i, j].scatter(
                            t[prefix_end:],
                            gen_data[prefix_end:, j],
                            s=1,
                            c='blue',
                            label='Generated',
                            alpha=0.7,
                        )
                        axes[i, j].scatter(
                            t[prefix_end:],
                            recon_data[prefix_end:, j],
                            s=1,
                            c='red',
                            label='Reconstructed',
                            alpha=0.7,
                        )
                    if prefix_end > 0:
                        axes[i, j].axvline(prefix_end - 1, color='gray', linestyle=':', linewidth=1.0)
                    if highlight_items:
                        for event_idx, (event_label, raw_index) in enumerate(highlight_items):
                            event_index = int(raw_index)
                            if 0 <= event_index < len(gen_data):
                                color = highlight_colors[event_idx % len(highlight_colors)]
                                axes[i, j].axvline(event_index, color=color, linestyle='--', linewidth=1.2, alpha=0.9)
                                y_min, y_max = axes[i, j].get_ylim()
                                axes[i, j].text(
                                    event_index,
                                    y_max,
                                    event_label,
                                    rotation=90,
                                    va='top',
                                    ha='right',
                                    fontsize=8,
                                    color=color,
                                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.65),
                                )
                    axes[i, j].set_title(f'{key}[{j}]')
                    axes[i, j].legend(markerscale=4, fontsize=9, loc='best')
            for j in range(gen_data.shape[-1], ncols):
                axes[i, j].set_visible(False)

        out_path = os.path.join(save_path, f'{name}.jpg')
        print(f"[Compare] Saving figure to {out_path}")
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print("[Compare] Figure saved and closed")
    
    print(f"[Compare] Done: mse_qpos={mse_qpos:.6f}, mse_mom={mse_mom:.6f}, mse_total={mse_total:.6f}")
    out = {'mse_qpos': mse_qpos, 'mse_mom': mse_mom, 'mse_total': mse_total}
    if return_series:
        out.update(
            {
                'generated_qpos': gen_qpos_plot,
                'generated_mom': gen_mom,
                'generated_torque': gen_tau,
                'reconstructed_qpos': recon_qpos_plot,
                'reconstructed_mom': recon_mom,
                'reconstructed_torque': recon_tau,
            }
        )
    return out


def compare_multiple_generated_with_reconstructed(
    generated_list: list[dict],
    mujoco_model_path: str,
    save_path: str,
    dt: float = 0.0005,
    data_dt: float = None,
    name: str = None,
    trajectory_alignment: str = 'pre_step',
    prefix_len: int = 0,
    qpos_representation: str = "raw",
    plot_dpi: int = 300,
) -> dict:
    """
    Overlay multiple sampled branches from the same history prefix in the same
    callback-style comparison plot.

    The plot uses the same grid format as compare_generated_with_reconstructed:
    blue = generated branches, red = MuJoCo reconstructions, black = shared prefix.
    """
    if len(generated_list) == 0:
        raise ValueError("generated_list must contain at least one trajectory")

    series_rows = [
        compare_generated_with_reconstructed(
            generated=generated,
            mujoco_model_path=mujoco_model_path,
            save_path=save_path,
            dt=dt,
            data_dt=data_dt,
            name=None,
            trajectory_alignment=trajectory_alignment,
            return_series=True,
            qpos_representation=qpos_representation,
        )
        for generated in generated_list
    ]

    nrows, ncols = 3, max(v.shape[-1] for v in generated_list[0].values())
    fig, axes = plt.subplots(nrows, ncols, figsize=(30, 10), dpi=int(plot_dpi))
    mean_mse_q = float(np.mean([row['mse_qpos'] for row in series_rows]))
    mean_mse_p = float(np.mean([row['mse_mom'] for row in series_rows]))
    mean_mse_total = float(np.mean([row['mse_total'] for row in series_rows]))
    fig.suptitle(
        f'Branches={len(series_rows)} mean MSE: qpos={mean_mse_q:.6f}, mom={mean_mse_p:.6f}, total={mean_mse_total:.6f}',
        fontsize=14,
        y=1.02,
    )

    keys = [
        ('seq_qpos', 'generated_qpos', 'reconstructed_qpos'),
        ('seq_mom', 'generated_mom', 'reconstructed_mom'),
        ('seq_torque', 'generated_torque', 'reconstructed_torque'),
    ]
    prefix_len = max(0, int(prefix_len))

    for row_idx, (title, gen_key, recon_key) in enumerate(keys):
        branch_dim = series_rows[0][gen_key].shape[-1]
        for dim in range(ncols):
            ax = axes[row_idx, dim]
            if dim >= branch_dim:
                ax.set_visible(False)
                continue

            prefix_plotted = False
            gen_plotted = False
            recon_plotted = False
            for branch in series_rows:
                gen_arr = branch[gen_key]
                recon_arr = branch[recon_key]
                t = np.arange(len(gen_arr))
                prefix_end = min(prefix_len, len(gen_arr))
                if prefix_end > 0 and not prefix_plotted:
                    ax.scatter(
                        t[:prefix_end],
                        gen_arr[:prefix_end, dim],
                        s=2,
                        c='black',
                        alpha=0.9,
                        label='Prefix',
                    )
                    prefix_plotted = True
                if title == 'seq_torque':
                    ax.scatter(
                        t[prefix_end:],
                        gen_arr[prefix_end:, dim],
                        s=2,
                        c='crimson',
                        alpha=0.18,
                        label='Applied torque branches' if not gen_plotted else None,
                    )
                    gen_plotted = True
                else:
                    ax.scatter(
                        t[prefix_end:],
                        gen_arr[prefix_end:, dim],
                        s=2,
                        c='blue',
                        alpha=0.18,
                        label='Generated branches' if not gen_plotted else None,
                    )
                    gen_plotted = True
                    ax.scatter(
                        t[prefix_end:],
                        recon_arr[prefix_end:, dim],
                        s=2,
                        c='red',
                        alpha=0.18,
                        label='Reconstructed branches' if not recon_plotted else None,
                    )
                    recon_plotted = True
            if prefix_len > 0:
                ax.axvline(prefix_len - 1, color='gray', linestyle=':', linewidth=1.0)
            ax.set_title(f'{title}[{dim}]')
            ax.legend(markerscale=4, fontsize=9, loc='best')

    out_path = os.path.join(save_path, f'{name}.jpg')
    print(f"[Compare] Saving multi-branch figure to {out_path}")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

    return {
        'mse_qpos': mean_mse_q,
        'mse_mom': mean_mse_p,
        'mse_total': mean_mse_total,
        'num_branches': int(len(series_rows)),
    }


def reconstruct_traj_with_momentum(
    model, num_steps: int, dt: float, initial_qpos: np.array, initial_qvel: np.array, seq_torque: np.array,
    data_dt: float = None, trajectory_alignment: str = 'pre_step'
):
    """
    Reconstruct trajectory from torque, returning (qpos, mom, torque).
    
    Supports separate simulation timestep (dt) and data collection timestep (data_dt).
    When data_dt > dt, the simulation runs at fine resolution but data is collected
    at coarser intervals.
    
    Args:
        model: MuJoCo model
        num_steps: Number of data points to collect
        dt: Fine simulation timestep
        initial_qpos: Initial position
        initial_qvel: Initial velocity
        seq_torque: Torque sequence at data_dt resolution (num_steps, torque_dim)
        data_dt: Data collection timestep (default: dt for backwards compatibility)
        trajectory_alignment: 'pre_step' for synchronized trajectories, 'post_step' for legacy reconstruction
    
    Returns:
        dict with 'seq_qpos', 'seq_mom', 'seq_torque' at data_dt resolution
    """
    # Default data_dt to dt for backwards compatibility
    if data_dt is None:
        data_dt = dt
    
    # Compute skip_steps
    skip_steps = int(round(data_dt / dt))
    if skip_steps < 1:
        skip_steps = 1
    
    model.opt.timestep = dt
    data = mujoco.MjData(model)

    qpos_dim = int(initial_qpos.shape[-1])
    qvel_dim = int(initial_qvel.shape[-1])
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:qpos_dim] = initial_qpos
    data.qvel[:qvel_dim] = initial_qvel
    mujoco.mj_forward(model, data)

    seq_qpos = []
    seq_mom = []
    M = np.zeros((model.nv, model.nv))
    
    num_sim_steps = max(num_steps - 1, 0) * skip_steps

    if trajectory_alignment == 'pre_step':
        for data_idx in range(num_steps):
            mujoco.mj_fullM(model, M, data.qM)
            seq_qpos.append(data.qpos[:qpos_dim].copy())
            seq_mom.append((M @ data.qvel)[:qvel_dim].copy())
            if data_idx == num_steps - 1:
                break
            tau_t = seq_torque[data_idx]
            for _ in range(skip_steps):
                data.ctrl[:] = tau_t
                mujoco.mj_step(model, data)
    elif trajectory_alignment == 'post_step':
        for i in range(num_sim_steps):
            torque_idx = i // skip_steps
            data.ctrl[:] = seq_torque[torque_idx]
            mujoco.mj_step(model, data)
            if (i + 1) % skip_steps == 0:
                mujoco.mj_fullM(model, M, data.qM)
                seq_qpos.append(data.qpos[:qpos_dim].copy())
                seq_mom.append((M @ data.qvel)[:qvel_dim].copy())
    else:
        raise ValueError(f"Unknown trajectory alignment: {trajectory_alignment}")

    traj_recon = {
        'seq_qpos': np.array(seq_qpos),
        'seq_mom': np.array(seq_mom),
        'seq_torque': np.array(seq_torque[:len(seq_qpos)])
    }

    return traj_recon
