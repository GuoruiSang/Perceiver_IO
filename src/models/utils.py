"""
Model utilities for Trajectory DPF.

This module contains utility classes and functions used in the model implementation:
- EMA: Exponential Moving Average for improved sampling
- OutputQueryProvider: Provider of output queries for PerceiverIO decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from perceiver.model.core import QueryProvider
from einops import rearrange
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import os
import mujoco

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


class OutputQueryProvider(nn.Module, QueryProvider):
    """Provider of learnable output queries for PerceiverIO decoder.
    
    This class provides the query vectors used by the PerceiverIO decoder
    to extract predictions from latent representations.
    
    Args:
        num_query_channels: Number of channels in the query vectors
        num_queries: Number of query vectors (default: 1)
        init_scale: Standard deviation for parameter initialization (default: 0.02)
    """
    
    def __init__(
        self,
        num_query_channels: int,
        num_queries: int = 1,
        init_scale: float = 0.02
    ):
        super().__init__()
        self._num_queries = num_queries
        self._num_query_channels = num_query_channels
        self._query = nn.Parameter(
            torch.empty(num_queries, num_query_channels)
        )
        self._init_parameters(init_scale)
    
    def _init_parameters(self, init_scale: float):
        """Initialize query parameters with small random values."""
        with torch.no_grad():
            self._query.normal_(0.0, init_scale)
    
    @property
    def num_query_channels(self) -> int:
        """Number of channels in each query vector."""
        return self._num_query_channels
    
    @property
    def num_queries(self) -> int:
        """Number of query vectors."""
        return self._num_queries
    
    def forward(self, x=None) -> torch.Tensor:
        """
        Generate query vectors.
        
        Args:
            x: Input tensor (unused, for compatibility with PerceiverIO)
        
        Returns:
            Query tensor of shape [1, num_queries, num_query_channels]
        """
        return rearrange(self._query, "... -> 1 ...")
    
    def __call__(self, x=None) -> torch.Tensor:
        """Alias for forward pass."""
        return self.forward(x)


def qpos_energy(qpos: torch.Tensor, n_joints: int):
    """
    Ensure each quaternion has unit norm by penalizing squared norm deviation from 1.
    
    Args:
        qpos: [B, n_steps, n_joints*4] - quaternion positions
        n_joints: number of joints
    
    Returns:
        MSE loss between squared quaternion norms and 1
    """
    # Reshape to [B, n_steps, n_joints, 4] and compute squared norm for each quaternion
    qpos_reshaped = rearrange(qpos, 'b t (n d) -> b t n d', n=n_joints)
    squared_norms = torch.sum(qpos_reshaped ** 2, dim=-1)  # [B, n_steps, n_joints]
    
    targets = torch.ones_like(squared_norms)  # [B, n_steps, n_joints]

    return torch.nn.functional.mse_loss(squared_norms, targets)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor):
    """
    Quaternion multiplication q1 ⊗ q2
    
    Args:
        q1: shape (..., 4) - quaternions in [w, x, y, z] format
        q2: shape (..., 4) - quaternions in [w, x, y, z] format
    
    Returns:
        shape (..., 4) - product quaternion
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([w, x, y, z], dim=-1)

def align_quaternion_signs_in_time(qpos_sequence: torch.Tensor, n_joints: int) -> torch.Tensor:
    """
    Make quaternion signs temporally consistent: ensure dot(q_t, q_{t-1}) >= 0.

    Args:
        qpos_sequence: (B, n_steps, n_joints*4)
        n_joints: number of joints

    Returns:
        Aligned sequence with same shape.
    """
    B, n_steps, _ = qpos_sequence.shape
    q = qpos_sequence.reshape(B, n_steps, n_joints, 4).clone()
    # Cumulative alignment along time
    for t in range(1, n_steps):
        prev = q[:, t-1, :, :]  # (B, J, 4)
        curr = q[:, t, :, :]
        dots = torch.sum(curr * prev, dim=-1, keepdim=True)  # (B, J, 1)
        flip = (dots < 0).to(curr.dtype)
        sign = 1.0 - 2.0 * flip  # +1 or -1
        q[:, t, :, :] = curr * sign
    return q.reshape(B, n_steps, n_joints * 4)

def qpos_qvel_consistency_energy(
    qpos_sequence: torch.Tensor,
    qvel_sequence: torch.Tensor,
    dt,
    omega_in_body_frame: bool = True,
):
    """
    Integrate-and-compare consistency between quaternions and angular velocities.

    Predict q_next by integrating ω over dt and composing with q_current, then
    penalize geodesic angle between predicted and actual orientations.

    Args:
        qpos_sequence: (B, n_steps, n_joints*4) unit quaternions [w,x,y,z]
        qvel_sequence: (B, n_steps, n_joints*3) angular velocities (rad/s)
        dt: scalar timestep (float)
        omega_in_body_frame: if True, use q_next = q_current ⊗ Δq(ω·dt);
                             else Δq(ω·dt) ⊗ q_current

    Returns:
        Mean squared geodesic angle between q_next and predicted q_next
    """
    B, n_steps, _ = qpos_sequence.shape
    n_joints = qpos_sequence.shape[2] // 4

    # Reshape and normalize quaternions
    qpos = qpos_sequence.reshape(B, n_steps, n_joints, 4)
    qpos = qpos / torch.norm(qpos, dim=-1, keepdim=True)
    qvel = qvel_sequence.reshape(B, n_steps, n_joints, 3)

    # Current state and controls
    q_current = qpos[:, :-1, :, :]  # (B, T-1, J, 4)
    q_next = qpos[:, 1:, :, :]      # (B, T-1, J, 4)
    omega = qvel[:, :-1, :, :]      # (B, T-1, J, 3)

    # Compute delta quaternion from axis-angle: theta_vec = ω * dt
    theta_vec = omega * dt
    theta = torch.norm(theta_vec, dim=-1, keepdim=True)  # (B, T-1, J, 1)
    half_theta = 0.5 * theta
    eps = 1e-8

    # Vector part scale: sin(half_theta)/theta with small-angle fallback ~ 0.5 - theta^2/48
    scale = torch.where(
        theta > eps,
        torch.sin(half_theta) / (theta + eps),
        0.5 - (theta * theta) / 48.0,
    )
    vec_part = theta_vec * scale  # (B, T-1, J, 3)
    scal_part = torch.cos(half_theta)  # (B, T-1, J, 1)
    delta_q = torch.cat([scal_part, vec_part], dim=-1)
    delta_q = delta_q / torch.norm(delta_q, dim=-1, keepdim=True)

    # Compose to predict next quaternion
    if omega_in_body_frame:
        q_next_pred = quaternion_multiply(q_current, delta_q)
    else:
        q_next_pred = quaternion_multiply(delta_q, q_current)
    q_next_pred = q_next_pred / torch.norm(q_next_pred, dim=-1, keepdim=True)

    # Geodesic distance on S^3 (sign-invariant): angle = 2*acos(|dot|)
    dots = torch.sum(q_next_pred * q_next, dim=-1)  # (B, T-1, J)
    dots = torch.clamp(torch.abs(dots), 0.0, 1.0 - 1e-6)
    angles = 2.0 * torch.acos(dots)
    loss = torch.mean(angles * angles)
    return loss

def torque_consistency_energy(qpos: torch.Tensor, torque: torch.Tensor, model: pl.LightningModule):
    """
    Predict torque based on qpos using trained model, and compute the error between predicted torque and ground-truth torque

    Args:
        qpos: shape [B, n_steps, n_joints*4]
        torque: shape [B, n_steps, n_joints*3]
        model: trained model for predicting torque based on qpos
    
    Returns:
        error: shape [1,]
    """
    predicted_torque = model(qpos)
    error = torch.nn.functional.mse_loss(predicted_torque, torque)
    return error

def qvel_smoothness_energy(qvel: torch.Tensor):
    """
    Ensure minimal jerk (second order derivative of velocity). Smooth qvel -> smooth qpos -> smooth torque.

    Args:
        qvel: shape [B, n_steps, n_joints*3]
    
    Returns:
        mean_jerk: shape [1,]
    """
    acc = qvel[:,1:] - qvel[:,:-1] # [B, n_steps-1, n_joints*3]
    jerk = acc[:,1:] - acc[:,:-1] # [B, n_steps-2, n_joints*3]

    mean_jerk = torch.mean(torch.pow(rearrange(jerk, 'b t h -> (b t h)'), 2)) # [1,]

    return mean_jerk

def calculate_energy(qpos: torch.Tensor, qvel: torch.Tensor, torque: torch.Tensor, model: pl.LightningModule, n_joints: int, timestep: int):
    """

    Args:
        qpos: shape [B, n_steps, n_joints*4]
        qvel: shape [B, n_steps, n_joints*3]
        torque: shape [B, n_steps, n_joints*3]
        model: trained model for predicting torque based on qpos
        n_joints: number of joints
        timestep: Timestep of the trajectory

    Returns:
        total_energy: [1,]
    """
    e1 = qpos_energy(qpos, n_joints)
    e2 = qvel_smoothness_energy(qvel)
    e3 = qpos_qvel_consistency_energy(qpos, qvel, timestep)

    # Weights: E1 (qpos norm), E2 (qvel smoothness), E3 (consistency)
    k1, k2, k3 = 0.1, 1, 0.1

    return k1*e1 + k2*e2 + k3*e3

def langevin_dynamics(
    qpos: torch.Tensor, 
    qvel: torch.Tensor, 
    torque: torch.Tensor, 
    model: pl.LightningModule, 
    n_joints: int, 
    timestep: int,
    n_iterations: int = 100,
    step_size: float = 0.01,
    noise_scale: float = 0.001,
    initial_step_size: float = 1e-6,
    min_step_size: float = 1e-7,
    step_growth: float = 2.0,
    adaptive: bool = True,
    max_grad_norm: float = 0.05,
):
    """
    Optimize qpos/qvel/torque trajectories toward low energy states using Langevin dynamics.
    
    Langevin dynamics combines gradient descent with stochastic noise to find low-energy
    configurations while avoiding local minima.
    
    Args:
        qpos: shape [B, n_steps, n_joints*4] - initial joint positions (quaternions)
        qvel: shape [B, n_steps, n_joints*3] - initial joint velocities
        torque: shape [B, n_steps, n_joints*3] - initial joint torques
        model: trained model for predicting torque based on qpos
        n_joints: number of joints
        timestep: Timestep of the trajectory
        n_iterations: number of optimization steps (default: 100)
        step_size: maximum gradient descent step size (default: 0.01)
        noise_scale: scale of stochastic noise (default: 0.001)
        initial_step_size: starting step size for adaptive updates
        min_step_size: minimum allowed step size during backtracking
        step_growth: multiplicative factor applied when updates are stable
        adaptive: enable adaptive step sizing with backtracking
        max_grad_norm: clip gradient norms to this threshold before each update

    Returns:
        qpos_opt: shape [B, n_steps, n_joints*4] - optimized positions
        qvel_opt: shape [B, n_steps, n_joints*3] - optimized velocities
        torque_opt: shape [B, n_steps, n_joints*3] - optimized torques
    """
    # Clone and enable gradients
    qpos_opt = qpos.clone().detach().requires_grad_(True)
    qvel_opt = qvel.clone().detach().requires_grad_(True)
    torque_opt = torque.clone().detach().requires_grad_(True)
    
    # Set model to eval mode
    model.eval()
    
    target_step_size = max(step_size, 0.0)
    current_step_size = min(target_step_size, initial_step_size) if adaptive else target_step_size
    min_step_size = max(min_step_size, 0.0)
    
    for i in range(n_iterations):
        # Compute energy
        energy = calculate_energy(qpos_opt, qvel_opt, torque_opt, model, n_joints, timestep)
        
        if not torch.isfinite(energy):
            print("[Langevin] Warning: Energy became non-finite; stopping refinement early.")
            break
        
        # Compute gradients
        energy.backward()
        
        # Abort if gradients explode
        if (qpos_opt.grad is not None and not torch.isfinite(qpos_opt.grad).all()) or \
           (qvel_opt.grad is not None and not torch.isfinite(qvel_opt.grad).all()) or \
           (torque_opt.grad is not None and not torch.isfinite(torque_opt.grad).all()):
            print("[Langevin] Warning: Non-finite gradient encountered; stopping refinement early.")
            break
        
        with torch.no_grad():
            attempt_step = current_step_size if adaptive else target_step_size
            update_successful = False
            while True:
                # Preserve current state for potential rollback
                qpos_prev = qpos_opt.clone()
                qvel_prev = qvel_opt.clone()
                torque_prev = torque_opt.clone()
                
                # Gradient descent step with clipping
                if qpos_opt.grad is not None and max_grad_norm > 0:
                    grad_norm = qpos_opt.grad.norm()
                    if torch.isfinite(grad_norm) and grad_norm > max_grad_norm:
                        qpos_grad = qpos_opt.grad * (max_grad_norm / (grad_norm + 1e-8))
                    else:
                        qpos_grad = qpos_opt.grad
                else:
                    qpos_grad = qpos_opt.grad
                if qvel_opt.grad is not None and max_grad_norm > 0:
                    grad_norm = qvel_opt.grad.norm()
                    if torch.isfinite(grad_norm) and grad_norm > max_grad_norm:
                        qvel_grad = qvel_opt.grad * (max_grad_norm / (grad_norm + 1e-8))
                    else:
                        qvel_grad = qvel_opt.grad
                else:
                    qvel_grad = qvel_opt.grad
                if torque_opt.grad is not None and max_grad_norm > 0:
                    grad_norm = torque_opt.grad.norm()
                    if torch.isfinite(grad_norm) and grad_norm > max_grad_norm:
                        torque_grad = torque_opt.grad * (max_grad_norm / (grad_norm + 1e-8))
                    else:
                        torque_grad = torque_opt.grad
                else:
                    torque_grad = torque_opt.grad
                
                if qpos_grad is not None:
                    qpos_opt -= attempt_step * qpos_grad
                if qvel_grad is not None:
                    qvel_opt -= attempt_step * qvel_grad
                if torque_grad is not None:
                    torque_opt -= attempt_step * torque_grad
                
                # Add stochastic noise
                if noise_scale > 0:
                    qpos_opt += noise_scale * torch.randn_like(qpos_opt)
                    qvel_opt += noise_scale * torch.randn_like(qvel_opt)
                    torque_opt += noise_scale * torch.randn_like(torque_opt)
                
                # Project quaternions onto unit sphere (hard constraint)
                B, n_steps, _ = qpos_opt.shape
                qpos_reshaped = qpos_opt.reshape(B, n_steps, n_joints, 4)
                qpos_normalized = qpos_reshaped / torch.norm(qpos_reshaped, dim=-1, keepdim=True)
                qpos_opt.copy_(qpos_normalized.reshape(B, n_steps, n_joints * 4))
                
                if torch.isfinite(qpos_opt).all() and torch.isfinite(qvel_opt).all() and torch.isfinite(torque_opt).all():
                    update_successful = True
                    break
                
                # Rollback and reduce step size
                qpos_opt.copy_(qpos_prev)
                qvel_opt.copy_(qvel_prev)
                torque_opt.copy_(torque_prev)
                
                if not adaptive or attempt_step <= min_step_size:
                    update_successful = False
                    break
                attempt_step = max(attempt_step * 0.5, min_step_size)
            
            if not update_successful:
                print("[Langevin] Warning: Unable to find stable step size; stopping refinement early.")
                break
            
            # Zero gradients for next iteration
            if qpos_opt.grad is not None:
                qpos_opt.grad.zero_()
            if qvel_opt.grad is not None:
                qvel_opt.grad.zero_()
            if torque_opt.grad is not None:
                torque_opt.grad.zero_()
            
            if adaptive:
                current_step_size = min(target_step_size, attempt_step * step_growth)
    
    # Detach from computation graph
    qpos_opt = qpos_opt.detach()
    # Align quaternion signs over time for smoother trajectories
    with torch.no_grad():
        qpos_opt = align_quaternion_signs_in_time(qpos_opt, n_joints)
    qvel_opt = qvel_opt.detach()
    torque_opt = torque_opt.detach()
    
    return qpos_opt, qvel_opt, torque_opt


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


def compute_qpos_qvel_qacc_consistency_energy(
    qpos: torch.Tensor, 
    qvel: torch.Tensor, 
    qacc: torch.Tensor, 
    dt: float
) -> torch.Tensor:
    """
        Args:
            qpos: [B, timesteps, nv]
            qvel: [B, timesteps, nv]
            qacc: [B, timesteps, nv]
        Returns:
            energy: [1,]
    """
    # Use shared central-difference helper to avoid duplicate formulas.
    qpos_dot = central_difference(qpos, dt)
    qvel_dot = central_difference(qvel, dt)

    e1 = nn.functional.mse_loss(qpos_dot, qvel)
    e2 = nn.functional.mse_loss(qvel_dot, qacc)

    return e1+e2


def compute_torque_consistency_energy(
    qpos: torch.Tensor, 
    qvel:torch.Tensor, 
    qacc: torch.Tensor, 
    torque: torch.Tensor, 
    torque_predictor: nn.Module
) -> torch.Tensor:
    """
    """

    predicted_torque = torque_predictor(qpos, qvel, qacc)

    e = nn.functional.mse_loss(predicted_torque, torque)

    return e


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

    # Use t=0..T-2 as "current" states
    q_t = seq_qpos[:, :-1]   # [B, T-1, qpos_dim]
    p_t = seq_mom[:, :-1]    # [B, T-1, mom_dim]
    tau_t = seq_torque[:, :-1]  # [B, T-1, torque_dim]

    # Actual next states
    q_next = seq_qpos[:, 1:]  # [B, T-1, qpos_dim]
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

    q_use = _gaussian_smooth_time_3d(seq_qpos, smooth_sigma)
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


def compute_hnn_guidance_energy(
    seq_qpos: torch.Tensor,
    seq_mom: torch.Tensor,
    seq_torque: torch.Tensor,
    hnn: nn.Module,
    dt: float,
    mode: str = "one_step",
    use_forward_diff: bool = False,
    hamres_smooth_sigma: float = 1.0,
    hamres_delta: float = 1.0,
    hamres_min_scale_q: float = 1e-3,
    hamres_min_scale_p: float = 1e-3,
) -> torch.Tensor:
    if mode == "one_step":
        return compute_hnn_physics_energy(
            seq_qpos, seq_mom, seq_torque, hnn, dt, use_forward_diff=use_forward_diff
        )
    if mode == "robust_hamres":
        return compute_hnn_robust_hamres_energy(
            seq_qpos,
            seq_mom,
            seq_torque,
            hnn,
            dt,
            smooth_sigma=hamres_smooth_sigma,
            delta=hamres_delta,
            min_scale_q=hamres_min_scale_q,
            min_scale_p=hamres_min_scale_p,
        )
    raise ValueError(f"Unknown guidance energy mode: {mode}")


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

    energy = compute_hnn_physics_energy(seq_qpos, seq_mom, seq_torque, hnn, dt, use_forward_diff=False)
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


def run_residual_sigmoid_sampling_hnn(
    x: torch.Tensor,
    seq_torque: torch.Tensor,
    qpos_dim: int,
    mom_dim: int,
    dt: float,
    hnn: nn.Module,
    num_candidates: int = 16,
    alpha_q: float = 1e-4,
    alpha_p: float = 1e-4,
    guidance_hamres_smooth_sigma: float = 1.0,
    guidance_hamres_delta: float = 1.0,
    guidance_hamres_min_scale_q: float = 1e-3,
    guidance_hamres_min_scale_p: float = 1e-3,
    guidance_trust_lambda: float = 0.0,
) -> torch.Tensor:
    """
    Deprecated helper kept for compatibility.

    NOTE:
    The active Strategy-1 path is implemented in TrajectoryDPF.sample_trajectories
    (candidate generation from x_t -> x_{t-1}, then x0 scoring). This helper is
    not used by the current sampling pipeline.
    """
    if num_candidates < 2:
        return x.detach()

    seq_qpos = x[:, :, :qpos_dim].detach()
    seq_mom = x[:, :, qpos_dim:qpos_dim + mom_dim].detach()
    q_ref = seq_qpos
    p_ref = seq_mom

    eps = 1e-12
    bsz, tlen, _ = x.shape
    # Candidate perturbation scales in [0.25, 2.0], sampled independently per sample/candidate.
    scales_q = 0.25 + 1.75 * torch.rand(bsz, num_candidates, 1, 1, device=x.device, dtype=x.dtype)
    scales_p = 0.25 + 1.75 * torch.rand(bsz, num_candidates, 1, 1, device=x.device, dtype=x.dtype)
    noise_q = torch.randn(bsz, num_candidates, tlen, qpos_dim, device=x.device, dtype=x.dtype)
    noise_p = torch.randn(bsz, num_candidates, tlen, mom_dim, device=x.device, dtype=x.dtype)
    q_cands = seq_qpos.unsqueeze(1) + scales_q * alpha_q * noise_q
    p_cands = seq_mom.unsqueeze(1) + scales_p * alpha_p * noise_p

    # Keep candidate-0 as the unperturbed x0 so strategy1 can choose "no-op" when best.
    q_cands[:, 0] = seq_qpos
    p_cands[:, 0] = seq_mom

    q_flat = q_cands.reshape(bsz * num_candidates, tlen, qpos_dim)
    p_flat = p_cands.reshape(bsz * num_candidates, tlen, mom_dim)
    tau_flat = seq_torque.unsqueeze(1).expand(-1, num_candidates, -1, -1).reshape(bsz * num_candidates, tlen, -1)
    residual = compute_hnn_robust_hamres_energy(
        q_flat,
        p_flat,
        tau_flat,
        hnn,
        dt,
        smooth_sigma=guidance_hamres_smooth_sigma,
        delta=guidance_hamres_delta,
        min_scale_q=guidance_hamres_min_scale_q,
        min_scale_p=guidance_hamres_min_scale_p,
        reduction="none_batch",
        create_graph=False,
    ).reshape(bsz, num_candidates)

    if guidance_trust_lambda > 0:
        trust = ((q_cands - q_ref.unsqueeze(1)) ** 2).mean(dim=(2, 3))
        trust = trust + ((p_cands - p_ref.unsqueeze(1)) ** 2).mean(dim=(2, 3))
        residual = residual + guidance_trust_lambda * trust

    # Normalize residuals per sample -> sigmoid weights (lower residual gets higher probability).
    mean_r = residual.mean(dim=1, keepdim=True)
    std_r = residual.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    z = (residual - mean_r) / std_r
    w = torch.sigmoid(-z)
    probs = w / w.sum(dim=1, keepdim=True).clamp_min(eps)
    chosen = torch.multinomial(probs, num_samples=1).squeeze(1)
    bidx = torch.arange(bsz, device=x.device)
    q_new = q_cands[bidx, chosen]
    p_new = p_cands[bidx, chosen]
    return torch.cat([q_new, p_new], dim=-1).detach()

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
    name: str = None, trajectory_alignment: str = 'pre_step', return_series: bool = False
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
    
    qpos_dim = int(gen['seq_qpos'].shape[-1])
    mom_dim = int(gen['seq_mom'].shape[-1])

    # Compute initial velocity from initial momentum: v = M^{-1} @ p
    print("[Compare] Computing initial velocity from momentum")
    data.qpos[:] = 0.0
    data.qpos[:qpos_dim] = gen['seq_qpos'][0]
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
        gen['seq_qpos'][0], initial_qvel, gen['seq_torque'],
        data_dt=data_dt
    )
    print("[Compare] Reconstruction finished")
    
    recon_qpos = recon['seq_qpos'][..., :qpos_dim]
    recon_mom = recon['seq_mom'][..., :mom_dim]
    gen_qpos, gen_mom, gen_tau, recon_qpos, recon_mom, recon_tau = _align_generated_and_reconstructed(
        {
            'seq_qpos': gen['seq_qpos'],
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
    
    # Only create plot if name is provided
    if name is not None:
        print("[Compare] Building matplotlib figure")
        keys = ['seq_qpos', 'seq_mom', 'seq_torque']
        nrows, ncols = len(keys), max(v.shape[-1] for v in gen.values())
        fig, axes = plt.subplots(nrows, ncols, figsize=(30, 10))
        
        # Add MSE info to the figure title
        fig.suptitle(f'MSE: qpos={mse_qpos:.6f}, mom={mse_mom:.6f}, total={mse_total:.6f}', fontsize=14, y=1.02)
        
        for i, key in enumerate(keys):
            if key == 'seq_qpos':
                gen_data = gen_qpos
                recon_data = recon_qpos
            elif key == 'seq_mom':
                gen_data = gen_mom
                recon_data = recon_mom
            else:
                gen_data = gen_tau
                recon_data = recon_tau
            t = np.arange(len(gen_data))
            for j in range(gen_data.shape[-1]):
                if j < ncols:
                    axes[i, j].scatter(t, gen_data[:, j], s=1, c='blue', label='Generated', alpha=0.7)
                    axes[i, j].scatter(t, recon_data[:, j], s=1, c='red', label='Reconstructed', alpha=0.7)
                    axes[i, j].set_title(f'{key}[{j}]')
                    axes[i, j].legend(markerscale=5)
            for j in range(gen_data.shape[-1], ncols):
                axes[i, j].set_visible(False)
        
        fig.tight_layout()
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
                'generated_qpos': gen_qpos,
                'generated_mom': gen_mom,
                'generated_torque': gen_tau,
                'reconstructed_qpos': recon_qpos,
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
        )
        for generated in generated_list
    ]

    nrows, ncols = 3, max(v.shape[-1] for v in generated_list[0].values())
    fig, axes = plt.subplots(nrows, ncols, figsize=(30, 10))
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
            ax.legend(markerscale=5)

    fig.tight_layout()
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


def reconstruct_traj_using_torque(
    model, num_steps: int, dt: float, initial_qpos: np.array, initial_qvel: np.array, seq_torque: np.array,
    data_dt: float = None, trajectory_alignment: str = 'pre_step'
):
    """
    Legacy reconstruction function returning (qpos, qvel, qacc, torque).
    
    Supports separate simulation timestep (dt) and data collection timestep (data_dt).
    
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
        dict with 'seq_qpos', 'seq_qvel', 'seq_qacc', 'seq_torque' at data_dt resolution
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
    seq_qvel = []
    seq_qacc = []
    
    num_sim_steps = max(num_steps - 1, 0) * skip_steps

    if trajectory_alignment == 'pre_step':
        for data_idx in range(num_steps):
            seq_qpos.append(data.qpos.copy())
            seq_qvel.append(data.qvel.copy())
            seq_qacc.append(data.qacc.copy())
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
                seq_qpos.append(data.qpos.copy())
                seq_qvel.append(data.qvel.copy())
                seq_qacc.append(data.qacc.copy())
    else:
        raise ValueError(f"Unknown trajectory alignment: {trajectory_alignment}")

    seq_qpos = np.array(seq_qpos)
    seq_qvel = np.array(seq_qvel)
    seq_qacc = np.array(seq_qacc)

    traj_recon = {
        'seq_torque': np.array(seq_torque[:len(seq_qpos)]),
        'seq_qacc': seq_qacc,
        'seq_qvel': seq_qvel,
        'seq_qpos': seq_qpos
    }

    return traj_recon


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
