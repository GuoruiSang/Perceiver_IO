"""
Model utilities for Trajectory DPF.

This module contains utility classes and functions used in the model implementation:
- EMA: Exponential Moving Average for improved sampling
- OutputQueryProvider: Provider of output queries for PerceiverIO decoder
"""

import torch
import torch.nn as nn
from perceiver.model.core import QueryProvider
from einops import rearrange
import pytorch_lightning as pl
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
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone().detach()
                else:
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone().detach()

    def apply_shadow(self) -> None:
        """Apply EMA shadow weights to the model for inference; backs up originals."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.backup:
                    self.backup[name] = param.data.clone().detach()
                param.data = self.shadow[name].clone().detach()

    def store(self, model=None) -> None:
        """Alias for apply_shadow (kept for API compatibility)."""
        self.apply_shadow()

    def copy_to(self, model=None) -> None:
        """Alias for apply_shadow (kept for API compatibility)."""
        self.apply_shadow()

    def restore(self, model=None) -> None:
        """Restore original model weights after inference."""
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
    # Central difference: f'(t) = (f(t+1) - f(t-1)) / (2*dt)
    qpos_dot = torch.zeros_like(qvel)
    qpos_dot[:, 1:-1] = (qpos[:, 2:] - qpos[:, :-2]) / (2 * dt)
    qpos_dot[:, 0] = (-3*qpos[:, 0] + 4*qpos[:, 1] - qpos[:, 2]) / (2*dt)
    qpos_dot[:, -1] = (3*qpos[:, -1] - 4*qpos[:, -2] + qpos[:, -3]) / (2*dt)

    qvel_dot = torch.zeros_like(qacc)
    qvel_dot[:, 1:-1] = (qvel[:, 2:] - qvel[:, :-2]) / (2 * dt)
    qvel_dot[:, 0] = (-3*qvel[:, 0] + 4*qvel[:, 1] - qvel[:, 2]) / (2*dt)
    qvel_dot[:, -1] = (3*qvel[:, -1] - 4*qvel[:, -2] + qvel[:, -3]) / (2*dt)

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


def compute_hnn_physics_energy(
    seq_qpos: torch.Tensor,
    seq_mom: torch.Tensor,
    seq_torque: torch.Tensor,
    hnn: nn.Module,
    dt: float,
    lambda_init: float = 1.0,
    seq_qpos_init: torch.Tensor = None,
    seq_mom_init: torch.Tensor = None,
    return_components: bool = False,
) -> torch.Tensor:
    """
    Compute HNN-based physics consistency energy for trajectory refinement.
    
    Energy = mse(dot_qpos, dot_qpos_pred) + mse(dot_mom, dot_mom_pred) 
             + lambda_init * mse(current_trajectory, initial_trajectory)
    
    Where:
        - dot_qpos, dot_mom: derivatives computed from trajectory using central difference
        - dot_qpos_pred = dH/dp (from HNN)
        - dot_mom_pred = -dH/dq + torque (Hamilton's equation with external torque)
        - initial_trajectory: the trajectory before optimization started (regularization)
    
    Args:
        seq_qpos: [B, T, qpos_dim] position trajectory (current, being optimized)
        seq_mom: [B, T, mom_dim] momentum trajectory (current, being optimized)
        seq_torque: [B, T, torque_dim] torque sequence (conditioning)
        hnn: Trained Hamiltonian Neural Network
        dt: timestep for finite differences
        lambda_init: weight for regularization term (keeping trajectory close to initial)
        seq_qpos_init: [B, T, qpos_dim] initial position trajectory (before optimization)
        seq_mom_init: [B, T, mom_dim] initial momentum trajectory (before optimization)
    
    Returns:
        If return_components is False:
            energy: scalar energy value
        If return_components is True:
            (energy, e1, e2, e3): all scalars
    """
    # Compute derivatives from trajectory using central difference
    dot_qpos = central_difference(seq_qpos, dt)  # dq/dt from trajectory
    dot_mom = central_difference(seq_mom, dt)    # dp/dt from trajectory
    
    # Compute HNN predictions
    # HNN takes (p, q) and returns H, then we compute gradients
    # dq/dt = dH/dp, dp/dt = -dH/dq + torque
    seq_mom_grad = seq_mom.detach().clone().requires_grad_(True)
    seq_qpos_grad = seq_qpos.detach().clone().requires_grad_(True)
    
    H = hnn(seq_mom_grad, seq_qpos_grad)  # [B, T, 1]
    
    # Compute gradients of H w.r.t. p and q
    dH_dp, dH_dq = torch.autograd.grad(
        H.sum(), 
        (seq_mom_grad, seq_qpos_grad),
        create_graph=True
    )
    
    # HNN predictions for dynamics
    dot_qpos_pred = dH_dp                    # dq/dt = dH/dp
    dot_mom_pred = -dH_dq + seq_torque       # dp/dt = -dH/dq + torque
    
    # Energy term 1: position derivative consistency
    e1 = nn.functional.mse_loss(dot_qpos, dot_qpos_pred)
    
    # Energy term 2: momentum derivative consistency
    e2 = nn.functional.mse_loss(dot_mom, dot_mom_pred)
    
    # Energy term 3: regularization - keep trajectory close to initial (diffusion prediction)
    # This prevents the trajectory from drifting too far from the diffusion model's output
    e3 = torch.tensor(0.0, device=seq_qpos.device)
    if seq_qpos_init is not None and seq_mom_init is not None:
        e3_qpos = nn.functional.mse_loss(seq_qpos, seq_qpos_init)
        e3_mom = nn.functional.mse_loss(seq_mom, seq_mom_init)
        e3 = e3_qpos + e3_mom
    
    energy = e1 + e2 + lambda_init * e3
    if return_components:
        return energy, e1.detach(), e2.detach(), e3.detach()
    return energy


from tqdm import trange

def run_langevin_dynamics_hnn(
    x: torch.Tensor, 
    seq_torque: torch.Tensor,
    qpos_dim: int,
    mom_dim: int,
    dt: float, 
    hnn: nn.Module, 
    num_steps: int,
    step_size: float,
    noise_scale: float,
    lambda_init: float = 1.0
) -> torch.Tensor:
    """
    Refine trajectory using Langevin dynamics with HNN physics energy.
    
    Args:
        x: State tensor [B, T, qpos_dim + mom_dim] with structure [qpos | mom]
        seq_torque: Torque conditioning [B, T, torque_dim] (fixed, not optimized)
        qpos_dim: Dimension of position
        mom_dim: Dimension of momentum
        dt: Timestep for finite differences
        hnn: Trained Hamiltonian Neural Network
        num_steps: Number of Langevin steps
        step_size: Step size for gradient descent
        noise_scale: Scale of injected noise
        lambda_init: Weight for regularization term (keeping trajectory close to initial)
    
    Returns:
        Refined state tensor [B, T, qpos_dim + mom_dim]
    """
    # Store initial trajectory for regularization (before optimization)
    seq_qpos_init = x[:, :, :qpos_dim].detach().clone()
    seq_mom_init = x[:, :, qpos_dim:qpos_dim + mom_dim].detach().clone()
    
    seq_qpos = x[:, :, :qpos_dim].clone().requires_grad_(True)
    seq_mom = x[:, :, qpos_dim:qpos_dim + mom_dim].clone().requires_grad_(True)

    for i in trange(num_steps, desc='Running Langevin Dynamics'):
        energy, e1, e2, e3 = compute_hnn_physics_energy(
            seq_qpos, seq_mom, seq_torque, hnn, dt, lambda_init,
            seq_qpos_init=seq_qpos_init, seq_mom_init=seq_mom_init,
            return_components=True,
        )
        
        if i == 0 or i == num_steps - 1:
            print(f'HNN Energy: {energy.item():.6f} (e1={e1.item():.6f}, e2={e2.item():.6f}, e3={e3.item():.6f}, lambda={lambda_init})')
        grad_qpos, grad_mom = torch.autograd.grad(energy, [seq_qpos, seq_mom])

        noise_std = (2 * step_size * noise_scale) ** 0.5
        seq_qpos = seq_qpos - step_size * grad_qpos + noise_std * torch.randn_like(seq_qpos)
        seq_mom = seq_mom - step_size * grad_mom + noise_std * torch.randn_like(seq_mom)

        seq_qpos = seq_qpos.detach().requires_grad_(True)
        seq_mom = seq_mom.detach().requires_grad_(True)

    new_x = torch.cat([seq_qpos, seq_mom], dim=-1)
    return new_x


def run_adam_optimization_hnn(
    x: torch.Tensor,
    seq_torque: torch.Tensor,
    qpos_dim: int,
    mom_dim: int,
    dt: float,
    hnn: nn.Module,
    num_steps: int,
    lr: float = 1e-3,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    lambda_init: float = 1.0
) -> torch.Tensor:
    """
    Optimize trajectory using Adam with HNN physics consistency energy.
    
    Args:
        x: State tensor [B, T, qpos_dim + mom_dim] with structure [qpos | mom]
        seq_torque: Torque conditioning [B, T, torque_dim] (fixed, not optimized)
        qpos_dim: Dimension of position
        mom_dim: Dimension of momentum
        dt: Timestep for finite differences
        hnn: Trained Hamiltonian Neural Network
        num_steps: Number of optimization steps
        lr: Learning rate for Adam optimizer
        betas: Coefficients for running averages
        eps: Numerical stability term
        lambda_init: Weight for regularization term (keeping trajectory close to initial)
    
    Returns:
        Optimized state tensor [B, T, qpos_dim + mom_dim]
    """
    # Store initial trajectory for regularization (before optimization)
    seq_qpos_init = x[:, :, :qpos_dim].detach().clone()
    seq_mom_init = x[:, :, qpos_dim:qpos_dim + mom_dim].detach().clone()
    
    seq_qpos = nn.Parameter(x[:, :, :qpos_dim].clone())
    seq_mom = nn.Parameter(x[:, :, qpos_dim:qpos_dim + mom_dim].clone())

    optimizer = torch.optim.Adam(
        [seq_qpos, seq_mom],
        lr=lr,
        betas=betas,
        eps=eps
    )

    for i in trange(num_steps, desc='Running Adam Optimization'):
        optimizer.zero_grad()
        
        energy, e1, e2, e3 = compute_hnn_physics_energy(
            seq_qpos, seq_mom, seq_torque, hnn, dt, lambda_init,
            seq_qpos_init=seq_qpos_init, seq_mom_init=seq_mom_init,
            return_components=True,
        )
        if i == 0 or i == num_steps - 1:
            print(f'HNN Energy: {energy.item():.6f} (e1={e1.item():.6f}, e2={e2.item():.6f}, e3={e3.item():.6f}, lambda={lambda_init})')
        
        energy.backward()
        optimizer.step()

    new_x = torch.cat([seq_qpos.data, seq_mom.data], dim=-1)

    return new_x

def compare_generated_with_reconstructed(
    generated: dict, mujoco_model_path: str, save_path: str, dt: float = 0.0005, data_dt: float = None, name: str = 'comparison'
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
        name: Name for the output file
    
    Returns:
        dict with MSE values: {'mse_qpos': float, 'mse_mom': float, 'mse_total': float}
    """
    import mujoco
    
    # Default data_dt to dt for backwards compatibility
    if data_dt is None:
        data_dt = dt
    
    # Convert to numpy if needed
    gen = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in generated.items()}
    
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    data = mujoco.MjData(model)
    
    # Compute initial velocity from initial momentum: v = M^{-1} @ p
    data.qpos[:] = gen['seq_qpos'][0]
    data.qvel[:] = 0  # Temporary
    mujoco.mj_forward(model, data)
    
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    initial_qvel = np.linalg.solve(M, gen['seq_mom'][0])
    
    # Reconstruct using MuJoCo physics (with data_dt support)
    recon = reconstruct_traj_with_momentum(
        model, len(gen['seq_qpos']), dt,
        gen['seq_qpos'][0], initial_qvel, gen['seq_torque'],
        data_dt=data_dt
    )
    
    # Compute MSE between generated and reconstructed trajectories
    # Align: generated[1:] vs reconstructed (both have length T-1)
    mse_qpos = np.mean((gen['seq_qpos'][1:] - recon['seq_qpos']) ** 2)
    mse_mom = np.mean((gen['seq_mom'][1:] - recon['seq_mom']) ** 2)
    mse_total = mse_qpos + mse_mom
    
    # Plot: generated[1:] vs reconstructed (both have length T-1)
    keys = ['seq_qpos', 'seq_mom', 'seq_torque']
    nrows, ncols = len(keys), max(v.shape[-1] for v in gen.values())
    fig, axes = plt.subplots(nrows, ncols, figsize=(30, 10))
    
    # Add MSE info to the figure title
    fig.suptitle(f'MSE: qpos={mse_qpos:.6f}, mom={mse_mom:.6f}, total={mse_total:.6f}', fontsize=14, y=1.02)
    
    for i, key in enumerate(keys):
        gen_data, recon_data = gen[key][1:], recon[key]  # Align: generated[1:] vs reconstructed
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
    fig.savefig(os.path.join(save_path, f'{name}.jpg'), bbox_inches='tight')
    plt.close(fig)
    print(f"[Comparison] Saved to {os.path.join(save_path, name + '.jpg')}")
    
    return {'mse_qpos': mse_qpos, 'mse_mom': mse_mom, 'mse_total': mse_total}


def reconstruct_traj_using_torque(model, num_steps: int, dt: float, initial_qpos: np.array, initial_qvel: np.array, seq_torque: np.array, data_dt: float = None):
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

    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel

    seq_qpos = []
    seq_qvel = []
    seq_qacc = []
    
    num_sim_steps = (num_steps - 1) * skip_steps

    # Run simulation at fine timestep, collect at coarse intervals
    for i in range(num_sim_steps):
        # Apply torque: each torque in seq_torque is held for skip_steps iterations
        torque_idx = i // skip_steps
        data.ctrl[:] = seq_torque[torque_idx]
        
        mujoco.mj_step(model, data)

        # Collect data at data_dt intervals (every skip_steps simulation steps)
        if (i + 1) % skip_steps == 0:
            seq_qpos.append(data.qpos.copy())
            seq_qvel.append(data.qvel.copy())
            seq_qacc.append(data.qacc.copy())

    seq_qpos = np.array(seq_qpos)
    seq_qvel = np.array(seq_qvel)
    seq_qacc = np.array(seq_qacc)

    traj_recon = {
        'seq_torque': seq_torque[:-1],
        'seq_qacc': seq_qacc,
        'seq_qvel': seq_qvel,
        'seq_qpos': seq_qpos
    }

    return traj_recon


def reconstruct_traj_with_momentum(model, num_steps: int, dt: float, initial_qpos: np.array, initial_qvel: np.array, seq_torque: np.array, data_dt: float = None):
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

    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel

    seq_qpos = []
    seq_mom = []
    M = np.zeros((model.nv, model.nv))
    
    num_sim_steps = (num_steps - 1) * skip_steps

    # Run simulation at fine timestep, collect at coarse intervals
    for i in range(num_sim_steps):
        # Apply torque: each torque in seq_torque is held for skip_steps iterations
        torque_idx = i // skip_steps
        data.ctrl[:] = seq_torque[torque_idx]
        mujoco.mj_step(model, data)
        
        # Collect data at data_dt intervals (every skip_steps simulation steps)
        if (i + 1) % skip_steps == 0:
            # Compute mass matrix and momentum
            mujoco.mj_fullM(model, M, data.qM)
            
            seq_qpos.append(data.qpos.copy())
            seq_mom.append((M @ data.qvel).copy())

    traj_recon = {
        'seq_qpos': np.array(seq_qpos),
        'seq_mom': np.array(seq_mom),
        'seq_torque': seq_torque[:-1]
    }

    return traj_recon

def compute_chunked_integration_energy(
    seq_qpos: torch.Tensor,
    seq_mom: torch.Tensor,
    seq_torque: torch.Tensor,
    hnn: nn.Module,
    dt: float,
    chunk_length: int,
) -> torch.Tensor:
    """
    Compute integration energy over random chunks of the trajectory.
    
    This evaluates how well the HNN integrator can predict the trajectory over
    short horizons (chunk_length), which is more stable than full-trajectory integration
    and avoids boundary artifacts by using random chunks.
    
    Args:
        seq_qpos: [B, T, qpos_dim] position trajectory
        seq_mom: [B, T, mom_dim] momentum trajectory
        seq_torque: [B, T, torque_dim] torque sequence
        hnn: HNNWrapper model
        dt: timestep
        chunk_length: length of integration chunks
        
    Returns:
        energy: scalar mean squared error between integrated and actual chunks
    """
    B, T, _ = seq_qpos.shape
    
    # Ensure we can fit at least one chunk
    if T <= chunk_length:
        raise ValueError(f"Trajectory length {T} must be greater than chunk_length {chunk_length}")
    
    # Select random start indices for each batch element
    # Valid start range: [0, T - chunk_length - 1]
    # We need T-chunk_length-1 because integration produces chunk_length+1 states (including t=0)
    max_start = T - chunk_length - 1
    start_indices = torch.randint(0, max_start + 1, (B,), device=seq_qpos.device)
    
    # Gather initial conditions and ground truth chunks
    # We need to extract [start:start+chunk_length+1] for comparison
    
    # Helper to gather chunks: [B, chunk_len+1, dim]
    def gather_chunks(tensor, starts, length):
        batch_indices = torch.arange(B, device=tensor.device).unsqueeze(1)
        time_indices = starts.unsqueeze(1) + torch.arange(length + 1, device=tensor.device).unsqueeze(0)
        return tensor[batch_indices, time_indices]
    
    qpos_chunk_gt = gather_chunks(seq_qpos, start_indices, chunk_length)
    mom_chunk_gt = gather_chunks(seq_mom, start_indices, chunk_length)
    torque_chunk = gather_chunks(seq_torque, start_indices, chunk_length) # Torque needs to cover integration steps
    
    # Initial state for integration
    q0 = qpos_chunk_gt[:, 0, :]
    p0 = mom_chunk_gt[:, 0, :]
    
    # Torque sequence for integration: [chunk_length, B, dim]
    # We take the first chunk_length torques (t=0 to t=chunk_length-1)
    tau_seq = torque_chunk[:, :-1, :].permute(1, 0, 2)
    
    # Integrate forward
    # Returns trajectories of shape [chunk_length+1, B, dim]
    p_traj, q_traj, _ = hnn.integrate_trajectory(p0, q0, tau_seq, dt, chunk_length)
    
    # Permute back to [B, chunk_length+1, dim] for comparison
    p_traj = p_traj.permute(1, 0, 2)
    q_traj = q_traj.permute(1, 0, 2)
    
    # Compute MSE loss (energy)
    # We compare the whole chunk including t=0 (which should be 0 error) and t=chunk_length
    loss_q = nn.functional.mse_loss(q_traj, qpos_chunk_gt)
    loss_p = nn.functional.mse_loss(p_traj, mom_chunk_gt)
    
    return loss_q + loss_p


def run_adam_optimization_hnn_integration(
    x: torch.Tensor,
    seq_torque: torch.Tensor,
    qpos_dim: int,
    mom_dim: int,
    dt: float,
    hnn: nn.Module,
    num_steps: int,
    lr: float = 1e-3,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    chunk_length: int = 15,
    lambda_init: float = 1.0,
) -> torch.Tensor:
    """
    Optimize trajectory using Adam with HNN integration-based energy (shooting method).
    
    This uses chunked integration (k-step shooting) instead of derivative matching,
    which enforces causal consistency over short horizons. Each optimization step
    uses randomized chunk positions to avoid boundary artifacts.
    
    Args:
        x: State tensor [B, T, qpos_dim + mom_dim] with structure [qpos | mom]
        seq_torque: Torque conditioning [B, T, torque_dim] (fixed, not optimized)
        qpos_dim: Dimension of position
        mom_dim: Dimension of momentum
        dt: Timestep for integration
        hnn: Trained Hamiltonian Neural Network (HNNWrapper)
        num_steps: Number of optimization steps
        lr: Learning rate for Adam optimizer
        betas: Coefficients for running averages
        eps: Numerical stability term
        chunk_length: Length of integration chunks (default: 15)
        lambda_init: Weight for regularization term (keeping trajectory close to initial)
    
    Returns:
        Optimized state tensor [B, T, qpos_dim + mom_dim]
    """
    B, T, _ = x.shape
    
    # Validate chunk length
    if T <= chunk_length:
        print(f"[Warning] Trajectory length {T} <= chunk_length {chunk_length}, falling back to derivative matching")
        # Fall back to derivative-based method
        return run_adam_optimization_hnn(
            x, seq_torque, qpos_dim, mom_dim, dt, hnn, num_steps, lr, betas, eps, lambda_init
        )
    
    # Store initial trajectory for regularization (before optimization)
    seq_qpos_init = x[:, :, :qpos_dim].detach().clone()
    seq_mom_init = x[:, :, qpos_dim:qpos_dim + mom_dim].detach().clone()
    
    seq_qpos = nn.Parameter(x[:, :, :qpos_dim].clone())
    seq_mom = nn.Parameter(x[:, :, qpos_dim:qpos_dim + mom_dim].clone())

    optimizer = torch.optim.Adam(
        [seq_qpos, seq_mom],
        lr=lr,
        betas=betas,
        eps=eps
    )

    for i in trange(num_steps, desc='Running Adam Integration Optimization'):
        optimizer.zero_grad()
        
        # Compute integration energy with random chunks
        # Each step uses different random chunk positions (sliding window effect)
        integration_energy = compute_chunked_integration_energy(
            seq_qpos, seq_mom, seq_torque, hnn, dt, chunk_length
        )
        
        # Regularization: keep trajectory close to initial prediction
        reg_qpos = nn.functional.mse_loss(seq_qpos, seq_qpos_init)
        reg_mom = nn.functional.mse_loss(seq_mom, seq_mom_init)
        reg_energy = reg_qpos + reg_mom
        
        # Total energy
        energy = integration_energy + lambda_init * reg_energy
        
        if i == 0 or i == num_steps - 1:
            print(f'Integration Energy: {energy.item():.6f} (shooting={integration_energy.item():.6f}, reg={reg_energy.item():.6f}, lambda={lambda_init})')
        
        energy.backward()
        optimizer.step()

    new_x = torch.cat([seq_qpos.data, seq_mom.data], dim=-1)

    return new_x
