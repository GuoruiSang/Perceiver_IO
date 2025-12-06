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
    # e4 = torque_consistency_energy(qpos, torque, model)  # Commented out

    # Rebalanced weights to ensure qvel is optimized
    # E1 (qpos norm)
    # E2 (qvel smoothness)
    # E3 (consistency)
    k1, k2, k3 = 0.1, 1, 0.1

    return k1*e1 + k2*e2 + k3*e3  # + k4*e4 commented out

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


# trajectory = {
#     'seq_qpos': torch.zeros(500, 3),
#     'seq_qvel': torch.zeros(500, 4)
# }

# visualize_trajectory(trajectory, '/home/gsang/Projects/Perceiver_IO/plots')

def compute_qpos_qvel_qacc_consistency_energy(qpos: torch.Tensor, qvel: torch.Tensor, qacc: torch.Tensor, dt: float) -> torch.Tensor:
    """
        Args:
            qpos: [B, timesteps, nv]
            qvel: [B, timesteps, nv]
            qacc: [B, timesteps, nv]
        Returns:
            energy: [1,]
    """
    qpos_dot = torch.zeros_like(qvel)
    qpos_dot[:, 1: -1] = (qpos[:, :-2] - qpos[:, 2:]) / (2 * dt)
    qpos_dot[:, 0] = (-3*qpos[:, 0] + 4*qpos[:, 1] - qpos[:, 2]) / (2*dt)
    qpos_dot[:, -1] = (3*qpos[:, -1] - 4*qpos[:, -2] + qpos[:, -3]) / (2*dt)

    qvel_dot = torch.zeros_like(qacc)
    qvel_dot[:, 1: -1] = (qvel[:, :-2] - qvel[:, 2:]) / (2 * dt)
    qvel_dot[:, 0] = (-3*qvel[:, 0] + 4*qvel[:, 1] - qvel[:, 2]) / (2*dt)
    qvel_dot[:, -1] = (3*qvel[:, -1] - 4*qvel[:, -2] + qvel[:, -3]) / (2*dt)

    e1 = nn.functional.mse_loss(qpos_dot, qvel)
    e2 = nn.functional.mse_loss(qvel_dot, qacc)

    return e1+e2


def compute_torque_consistency_energy(qpos: torch.Tensor, qvel:torch.Tensor, qacc: torch.Tensor, torque: torch.Tensor, torque_predictor: nn.Module) -> torch.Tensor:
    """
    """

    predicted_torque = torque_predictor(qpos, qvel, qacc)

    e = nn.functional.mse_loss(predicted_torque, torque)

    return e


from tqdm import trange

def run_langevin_dynamics(x: torch.Tensor, torque_dim: int, qacc_dim: int, qvel_dim: int, dt: int, torque_predictor: nn.Module, num_steps: int, step_size: float, noise_scale: float) -> torch.Tensor:
    seq_torque = x[:, :, :torque_dim].clone().requires_grad_(True)
    seq_qacc = x[:, :, torque_dim:torque_dim+qacc_dim].clone().requires_grad_(True)
    seq_qvel = x[:, :, torque_dim+qacc_dim:torque_dim+qacc_dim+qvel_dim].clone().requires_grad_(True)
    seq_qpos = x[:, :, torque_dim+qacc_dim+qvel_dim:].clone().requires_grad_(True)

    for i in trange(num_steps, desc='Runing Langevin Dynamics'):
        e1 = compute_qpos_qvel_qacc_consistency_energy(seq_qpos, seq_qvel, seq_qacc, dt)
        e2 = compute_torque_consistency_energy(seq_qpos, seq_qvel, seq_qacc, seq_torque, torque_predictor)

        e = e1 + e2
        print(f'Total Energy: {e}---Energy 1: {e1}---Energy 2: {e2}')
        grad_torque, grad_qacc, grad_qvel, grad_qpos = torch.autograd.grad(e, [seq_torque, seq_qacc, seq_qvel, seq_qpos], )

        noise_std = (2 * step_size * noise_scale) ** 0.5
        seq_torque = seq_torque - step_size * grad_torque + noise_std * torch.randn_like(seq_torque)
        seq_qacc = seq_qacc - step_size * grad_qacc + noise_std * torch.randn_like(seq_qacc)
        seq_qvel = seq_qvel - step_size * grad_qvel + noise_std * torch.randn_like(seq_qvel)
        seq_qpos = seq_qpos - step_size * grad_qpos + noise_std * torch.randn_like(seq_qpos)

        seq_torque = seq_torque.detach().requires_grad_(True)
        seq_qacc = seq_qacc.detach().requires_grad_(True)
        seq_qvel = seq_qvel.detach().requires_grad_(True)
        seq_qpos = seq_qpos.detach().requires_grad_(True)

    new_x = torch.cat([seq_torque, seq_qacc, seq_qvel, seq_qpos], dim=-1)

    return new_x


def run_adam_optimization(
    x: torch.Tensor,
    torque_dim: int,
    qacc_dim: int,
    qvel_dim: int,
    dt: float,
    torque_predictor: nn.Module,
    num_steps: int,
    lr: float = 1e-3,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Optimize trajectory using Adam optimizer to minimize physics consistency energy.
    
    Args:
        x: Input tensor of shape (batch, seq_len, torque_dim + qacc_dim + qvel_dim + qpos_dim)
        torque_dim: Dimension of torque
        qacc_dim: Dimension of acceleration
        qvel_dim: Dimension of velocity
        dt: Time step
        torque_predictor: Module to predict torque from (qpos, qvel, qacc)
        num_steps: Number of optimization steps
        lr: Learning rate for Adam optimizer
        betas: Coefficients for computing running averages of gradient and its square
        eps: Term added to denominator for numerical stability
    
    Returns:
        Optimized trajectory tensor
    """
    seq_torque = nn.Parameter(x[:, :, :torque_dim].clone())
    seq_qacc = nn.Parameter(x[:, :, torque_dim:torque_dim+qacc_dim].clone())
    seq_qvel = nn.Parameter(x[:, :, torque_dim+qacc_dim:torque_dim+qacc_dim+qvel_dim].clone())
    seq_qpos = nn.Parameter(x[:, :, torque_dim+qacc_dim+qvel_dim:].clone())

    optimizer = torch.optim.Adam(
        [seq_torque, seq_qacc, seq_qvel, seq_qpos],
        lr=lr,
        betas=betas,
        eps=eps
    )

    for i in trange(num_steps, desc='Running Adam Optimization'):
        optimizer.zero_grad()
        
        e1 = compute_qpos_qvel_qacc_consistency_energy(seq_qpos, seq_qvel, seq_qacc, dt)
        e2 = compute_torque_consistency_energy(seq_qpos, seq_qvel, seq_qacc, seq_torque, torque_predictor)

        total_energy = e1 + e2
        print(f'Total Energy: {total_energy.item():.6f}---Energy 1: {e1.item():.6f}---Energy 2: {e2.item():.6f}')
        
        total_energy.backward()
        optimizer.step()

    new_x = torch.cat([
        seq_torque.data,
        seq_qacc.data,
        seq_qvel.data,
        seq_qpos.data
    ], dim=-1)

    return new_x

def reconstruct_traj_using_torque(model, num_steps: int, dt: float, initial_qpos: np.array, initial_qvel: np.array, seq_torque: np.array):
    model.opt.timestep = dt

    data = mujoco.MjData(model)

    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel

    seq_qpos = []
    seq_qvel = []
    seq_qacc = []

    for i in range(num_steps-1):
        data.ctrl[:] = seq_torque[i+1]
        
        mujoco.mj_step(model, data)

        seq_qpos.append(data.qpos.copy())
        seq_qvel.append(data.qvel.copy())
        seq_qacc.append(data.qacc.copy())

    seq_qpos = np.array(seq_qpos)
    seq_qvel = np.array(seq_qvel)
    seq_qacc = np.array(seq_qacc)

    traj_recon = {
        'seq_torque': seq_torque[1:],
        'seq_qacc': seq_qacc,
        'seq_qvel': seq_qvel,
        'seq_qpos': seq_qpos
    }

    return traj_recon
    
from HNN import TorquePredictor
import mujoco

device = 'cuda:0'
hnn_checkpoint_path = '/home/gsang/Projects/Perceiver_IO/checkpoints/HNN-epoch-epoch=99.ckpt'
generated_h5_file_path = '/home/gsang/Projects/Perceiver_IO/data/traj_40000-steps_500.h5'
mujoco_model_path = '/home/gsang/Projects/Perceiver_IO/configs/rigid_arm_hinge.xml'

num_steps = 500
dt = 0.0005
model = mujoco.MjModel.from_xml_path(mujoco_model_path)

hnn_checkpoint = torch.load(hnn_checkpoint_path)
hnn_state_dict = hnn_checkpoint['state_dict']

# Filter keys for torque_predictor and remove the prefix
torque_predictor_state_dict = {
    k.replace('torque_predictor.', ''): v 
    for k, v in hnn_state_dict.items() 
    if k.startswith('torque_predictor.')
}


torque_predictor = TorquePredictor(coordinate_dim=3).to(device)

# Load into torque_predictor
torque_predictor.load_state_dict(torque_predictor_state_dict)
print(f"Loaded {len(torque_predictor_state_dict)} parameters into torque_predictor")

import h5py

with h5py.File(generated_h5_file_path, 'r') as f:
    seq_qpos = torch.Tensor(f['traj_2']['seq_qpos'][:]).unsqueeze(0).to(device)
    seq_qvel = torch.Tensor(f['traj_2']['seq_qvel'][:]).unsqueeze(0).to(device)
    seq_qacc = torch.Tensor(f['traj_2']['seq_qacc'][:]).unsqueeze(0).to(device)
    seq_torque = torch.Tensor(f['traj_2']['seq_torque'][:]).unsqueeze(0).to(device)

# seq_qpos = torch.randn((1, 500, 3)).to(device)
# seq_qvel = torch.randn((1, 500, 3)).to(device)
# seq_qacc = torch.randn((1, 500, 3)).to(device)
# seq_torque = torch.randn((1, 500, 3)).to(device)

x = torch.cat([seq_torque, seq_qacc, seq_qvel, seq_qpos], dim=-1)
traj_before = {
    'seq_torque': seq_torque[0].clone(),
    'seq_qacc': seq_qacc[0].clone(),
    'seq_qvel': seq_qvel[0].clone(),
    'seq_qpos': seq_qpos[0].clone()
}

visualize_trajectory(traj_before, '/home/gsang/Projects/Perceiver_IO/plots', 'traj_before_langevin')

# new_x = run_langevin_dynamics(x, 3, 3, 3, 0.0005, torque_predictor, 20000, 1e-5, 1e-6)
# new_x = run_adam_optimization(x, 3, 3, 3, 0.0005, torque_predictor, 10000)
new_x = x.clone()
new_x[0, :, :3] = torque_predictor(traj_before['seq_qpos'], traj_before['seq_qvel'], traj_before['seq_qacc'])

torque_mse = nn.functional.mse_loss(new_x[0, :, :3], traj_before['seq_torque'])
print(f'Torque MSE: {torque_mse.item()}')

traj_after = {
    'seq_torque': new_x[0, :, :3].clone(),
    'seq_qacc': new_x[0, :, 3:6].clone(),
    'seq_qvel': new_x[0, :, 6:9].clone(),
    'seq_qpos': new_x[0, :, 9:].clone()
}

visualize_trajectory(traj_after, '/home/gsang/Projects/Perceiver_IO/plots', 'traj_after_langevin')

initial_qpos = traj_before['seq_qpos'][0].cpu().numpy()
initial_qvel = traj_before['seq_qvel'][0].cpu().numpy()
seq_torque = traj_before['seq_torque'].cpu().numpy()
traj_recon = reconstruct_traj_using_torque(model, 500, dt, initial_qpos, initial_qvel, seq_torque)
visualize_trajectory(traj_recon, '/home/gsang/Projects/Perceiver_IO/plots', 'traj_after_recon')
