import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
import mujoco
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.dataset import TrajectoryHNNCached


# -----------------------------------------------------------------------------
# 1. Verification Callback
# -----------------------------------------------------------------------------
class PhysicsCheckCallback(pl.Callback):
    def __init__(self, check_every_n_epochs=5, dt=0.00025):
        self.check_every_n_epochs = check_every_n_epochs
        self.dt = dt  # Must match training data timestep!

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        if trainer.current_epoch % self.check_every_n_epochs != 0:
            return
        
        device = pl_module.device
        
        model = mujoco.MjModel.from_xml_path('/home/gsang/Projects/Perceiver_IO/configs/rigid_arm_hinge.xml')
        model.opt.timestep = self.dt
        data = mujoco.MjData(model)
        
        # Use same initial range as training data
        data.qpos[:] = np.random.uniform(-0.5, 0.5, size=(model.nq,))
        data.qvel[:] = np.random.uniform(-0.5, 0.5, size=(model.nv,))

        M = np.zeros((model.nv, model.nv))
        mujoco.mj_forward(model, data)
        # Use mj_fullM instead of obsolete MjModel functions
        mujoco.mj_fullM(model, M, data.qM)

        seq_qpos = [data.qpos.copy()]
        seq_qvel = [data.qvel.copy()]
        seq_mom = [(M @ data.qvel).copy()]
        seq_true_energy = [data.energy[0] + data.energy[1]]  # KE + PE from MuJoCo

        # Match training trajectory length (1000 steps)
        num_steps = 500
        for _ in range(num_steps):
            mujoco.mj_step(model, data)
            mujoco.mj_fullM(model, M, data.qM)
            seq_qpos.append(data.qpos.copy())
            seq_qvel.append(data.qvel.copy())
            seq_mom.append((M @ data.qvel).copy())
            seq_true_energy.append(data.energy[0] + data.energy[1])

        # Convert to arrays for analysis
        seq_qpos = np.array(seq_qpos)
        seq_mom = np.array(seq_mom)
        seq_true_energy = np.array(seq_true_energy)
        
        # Compute learned H - Vectorized for speed!
        with torch.no_grad():
            p_tensor = torch.from_numpy(seq_mom).float().to(device)
            q_tensor = torch.from_numpy(seq_qpos).float().to(device)
            # Use pl_module() instead of pl_module.model() to ensure scaling is applied
            seq_H = pl_module(p_tensor, q_tensor).cpu().numpy().flatten()
        
        # Debug: Check if trajectory stays in training distribution
        q_min, q_max = seq_qpos.min(), seq_qpos.max()
        p_min, p_max = seq_mom.min(), seq_mom.max()
        print(f"\n[Validation Debug] q range: [{q_min:.3f}, {q_max:.3f}], p range: [{p_min:.3f}, {p_max:.3f}]")
        print(f"[Validation Debug] True energy range: {seq_true_energy.max() - seq_true_energy.min():.6f}")
        print(f"[Validation Debug] Learned H range: {seq_H.max() - seq_H.min():.6f}")
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        timesteps = np.arange(len(seq_H))
        
        # Left: Learned H
        axes[0].scatter(timesteps, seq_H, s=5, alpha=0.6, c='blue')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Learned H')
        axes[0].set_ylim(-50, 50)
        axes[0].set_title(f'Learned H (range: {seq_H.max() - seq_H.min():.4f})')
        axes[0].grid(True, alpha=0.3)
        
        # Right: True Energy from MuJoCo
        axes[1].scatter(timesteps, seq_true_energy, s=5, alpha=0.6, c='green')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('True Energy (KE + PE)')
        axes[1].set_title(f'MuJoCo True Energy (range: {seq_true_energy.max() - seq_true_energy.min():.6f})')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_image("Energy Conservation", images=[fig])
        plt.savefig('/home/gsang/Projects/Perceiver_IO/plots/Energy Conservation.jpg')
        plt.close()

# -----------------------------------------------------------------------------
# 2. MLP Model
# -----------------------------------------------------------------------------
class HNN(nn.Module):
    def __init__(self, coordinate_dim, momenta_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(coordinate_dim + momenta_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, p, q):
        x = torch.cat([p, q], dim=-1)
        return self.mlp(x)

class SeperableHNN(nn.Module):
    def __init__(self, coordinate_dim, momenta_dim):
        super().__init__()
        # Increased capacity (1024 units) and depth for 160M samples
        # Using CELU for potentially faster convergence and smoother gradients
        self.kinetic = nn.Sequential(
            nn.Linear(coordinate_dim + momenta_dim, 1024),
            nn.CELU(),
            nn.Linear(1024, 1024),
            nn.CELU(),
            nn.Linear(1024, 1024),
            nn.CELU(),
            nn.Linear(1024, 1024),
            nn.CELU(),
            nn.Linear(1024, 1),
        )
        
        # Potential Energy V(q)
        self.potential = nn.Sequential(
            nn.Linear(coordinate_dim, 1024),
            nn.CELU(),
            nn.Linear(1024, 1024),
            nn.CELU(),
            nn.Linear(1024, 1024),
            nn.CELU(),
            nn.Linear(1024, 1024),
            nn.CELU(),
            nn.Linear(1024, 1),
        )

    def forward(self, p, q):
        T = self.kinetic(torch.cat([p, q], dim=-1))
        V = self.potential(q)
        return T + V

class TorquePredictor(nn.Module):
    def __init__(self, coordinate_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3*coordinate_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 3),
        )
    
    def forward(self, pos, vel, acc):
        x = torch.cat([pos, vel, acc], dim=-1)
        return self.mlp(x)

# -----------------------------------------------------------------------------
# 3. HNN Wrapper
# -----------------------------------------------------------------------------
class HNNWrapper(pl.LightningModule):
    def __init__(self, coordinate_dim, momenta_dim, use_torque=True, predict_torque=True, 
                 qvel_var=1.0, mom_dot_var=1.0, q_std=1.0, p_std=1.0, eps: float = 1e-8):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpoint loading
        self.model = SeperableHNN(coordinate_dim, momenta_dim)
        self.use_torque = use_torque
        self.predict_torque = predict_torque
        if self.use_torque and self.predict_torque:
            self.torque_predictor = TorquePredictor(coordinate_dim)
        
        # Numerics
        self.register_buffer('eps', torch.tensor(float(eps)))

        # Variances for loss normalization (prefer per-dimension tensors; fall back to scalar)
        # Using register_buffer so they are moved to the correct device but not trained.
        self.register_buffer('qvel_var', torch.as_tensor(qvel_var, dtype=torch.float32))
        self.register_buffer('mom_dot_var', torch.as_tensor(mom_dot_var, dtype=torch.float32))
        
        # Input scaling statistics (prefer per-dimension tensors; fall back to scalar)
        self.register_buffer('q_std', torch.as_tensor(q_std, dtype=torch.float32))
        self.register_buffer('p_std', torch.as_tensor(p_std, dtype=torch.float32))

    def forward(self, p, q): 
        # Apply scaling during inference if needed
        p_scaled = p / (self.p_std + self.eps)
        q_scaled = q / (self.q_std + self.eps)
        return self.model(p_scaled, q_scaled)

    def compute_gradients(self, p, q):
        """Compute Hamiltonian gradients ∂H/∂p and ∂H/∂q."""
        p_grad = p.detach().clone().requires_grad_(True)
        q_grad = q.detach().clone().requires_grad_(True)
        H = self(p_grad, q_grad)
        dH_dp, dH_dq = torch.autograd.grad(
            H.sum(), (p_grad, q_grad), create_graph=False
        )
        return dH_dp, dH_dq, H

    def symplectic_euler_step(self, p, q, tau, dt):
        """
        One step of symplectic Euler integration using learned Hamiltonian.
        
        Symplectic Euler (semi-implicit):
            p_new = p + dt * (-∂H/∂q + τ)
            q_new = q + dt * ∂H/∂p(p_new, q)  # Use updated p
        
        This preserves the symplectic structure better than explicit Euler.
        """
        with torch.inference_mode(False):
            with torch.enable_grad():
                # First half: update momentum
                _, dH_dq, _ = self.compute_gradients(p, q)
                p_new = p + dt * (-dH_dq + tau)
                
                # Second half: update position using new momentum
                dH_dp_new, _, _ = self.compute_gradients(p_new, q)
                q_new = q + dt * dH_dp_new
        
        return p_new, q_new

    def integrate_trajectory(self, p0, q0, tau_seq, dt, num_steps):
        """
        Integrate trajectory from initial conditions using learned Hamiltonian.
        
        Args:
            p0: Initial momentum [B, dim]
            q0: Initial position [B, dim]
            tau_seq: Torque sequence [num_steps, B, dim] or [B, dim] (constant)
            dt: Integration timestep
            num_steps: Number of integration steps
            
        Returns:
            p_traj: Momentum trajectory [num_steps+1, B, dim]
            q_traj: Position trajectory [num_steps+1, B, dim]
            H_traj: Hamiltonian trajectory [num_steps+1, B, 1]
        """
        B, dim = p0.shape
        device = p0.device
        
        # Allocate trajectory storage
        p_traj = torch.zeros(num_steps + 1, B, dim, device=device)
        q_traj = torch.zeros(num_steps + 1, B, dim, device=device)
        H_traj = torch.zeros(num_steps + 1, B, 1, device=device)
        
        # Initial conditions
        p_traj[0] = p0
        q_traj[0] = q0
        H_traj[0] = self(p0, q0)
        
        p, q = p0.clone(), q0.clone()
        
        for t in range(num_steps):
            # Get torque for this step
            if tau_seq.dim() == 3:
                tau = tau_seq[t]
            else:
                tau = tau_seq  # Constant torque
            
            p, q = self.symplectic_euler_step(p, q, tau, dt)
            
            p_traj[t + 1] = p
            q_traj[t + 1] = q
            H_traj[t + 1] = self(p, q)
        
        return p_traj, q_traj, H_traj

    def configure_optimizers(self):
        # HNN-friendly optimizer: NO weight decay (hurts gradient fidelity)
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=0.0, fused=True)
        
        # CosineAnnealingLR: smooth decay from start, NO warm-up spike
        # This avoids the "loss down then up" pattern caused by OneCycleLR's LR ramp.
        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(total_steps, 1),
                eta_min=1e-6,  # Anneal to near-zero
            ),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def on_train_start(self):
        # Log hyperparameters to wandb
        if isinstance(self.logger, WandbLogger):
            self.logger.log_hyperparams(self.hparams)

    def calculate_loss(self, p, q, dqdt_target, dpdt_target, torque_target=None, qacc_target=None, create_graph=True):
        loss_torque = None
        # Physically-consistent input scaling:
        # Scale inputs, then adjust gradients by the same factor (chain rule)
        with torch.inference_mode(False):
            with torch.set_grad_enabled(True):
                p_raw = p.detach().requires_grad_(True)
                q_raw = q.detach().requires_grad_(True)
                
                # Scaled inputs for the network
                p_scaled = p_raw / (self.p_std + self.eps)
                q_scaled = q_raw / (self.q_std + self.eps)
                
                H = self.model(p_scaled, q_scaled)
                
                # Compute gradients ∂H/∂p and ∂H/∂q
                # create_graph=True is only needed during training for second-order derivatives
                grads = torch.autograd.grad(H.sum(), (p_raw, q_raw), create_graph=create_graph)
                dqdt_pred, dpdt_pred = grads[0], -grads[1]
                
                # Use ground-truth torque
                if self.use_torque and not self.predict_torque:
                    dpdt_pred = dpdt_pred + torque_target
                # Use predicted torque
                elif self.use_torque and self.predict_torque:
                    torque_pred = self.torque_predictor(q_raw, dqdt_target, qacc_target)
                    dpdt_pred = dpdt_pred + torque_pred
                    loss_torque = nn.functional.mse_loss(torque_target, torque_pred)
                

        # Normalized MSE loss: per-dimension MSE / per-dimension variance (or scalar variance fallback)
        # This prevents a single high-variance DOF from dominating training.
        qvel_var = self.qvel_var + self.eps
        mom_dot_var = self.mom_dot_var + self.eps
        loss_dqdt = torch.mean((dqdt_target - dqdt_pred) ** 2 / qvel_var)
        loss_dpdt = torch.mean((dpdt_target - dpdt_pred) ** 2 / mom_dot_var)

        # Balanced loss
        loss = loss_dqdt + loss_dpdt
        # Component logging (helps diagnose OneCycleLR spikes / imbalance)
        if getattr(self, "trainer", None) is not None:
            self.log('loss_dqdt', loss_dqdt, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
            self.log('loss_dpdt', loss_dpdt, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if loss_torque is not None:
            loss += 0.1 * loss_torque # Weight torque loss if needed
            self.log('loss_torque', loss_torque.item(), prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch['mom'], batch['qpos'], 
                                   batch['qvel'], batch['mom_dot'], 
                                   batch.get('torque', None), batch.get('qacc', None))
        self.log('train_loss', loss, prog_bar=True)
        # Log LR to confirm whether loss increases correlate with LR peaks (OneCycle behavior).
        if getattr(self, "trainer", None) is not None and self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0].get('lr', None)
            if lr is not None:
                self.log('lr', float(lr), prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Disable create_graph in validation to speed up and save memory
        loss = self.calculate_loss(batch['mom'], batch['qpos'], 
                                   batch['qvel'], batch['mom_dot'], 
                                   batch.get('torque', None), batch.get('qacc', None),
                                   create_graph=False)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        if self.use_torque and self.predict_torque:
            predicted_torque = self.torque_predictor(batch['qpos'], batch['qvel'], batch['qacc'])
            self.log('torque loss', nn.functional.mse_loss(batch['torque'], predicted_torque), on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        qpos = batch['qpos']
        qvel = batch['qvel']

        mom = batch['mom']
        mom_dot = batch['mom_dot']
        
        qacc = batch['qacc']
        torque = batch['torque']

        # Use self() instead of self.model() to apply scaling
        H = self(mom, qpos)

        # Use predicted torque if available, else use ground truth torque for physics check
        if self.use_torque and self.predict_torque:
            predicted_torque = self.torque_predictor(qpos, qvel, qacc)
            print(f"torque mse={nn.functional.mse_loss(torque, predicted_torque)}")
        else:
            predicted_torque = torque
        
        # Plot torque comparison for each dimension
        num_dims = torque.shape[1]
        timesteps = np.arange(len(torque))
        
        fig, axes = plt.subplots(1, num_dims, figsize=(5 * num_dims, 4))
        if num_dims == 1:
            axes = [axes]
        
        torque_np = torque.detach().cpu().numpy()
        predicted_torque_np = predicted_torque.detach().cpu().numpy()
        
        for i in range(num_dims):
            axes[i].scatter(timesteps, torque_np[:, i], s=2, alpha=0.6, label="ground truth")
            axes[i].scatter(timesteps, predicted_torque_np[:, i], s=2, alpha=0.6, label="predicted")
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel(f'Torque[{i}]')
            axes[i].set_ylim(-5, 5)
            axes[i].set_title(f'Torque Dim {i}, MSE: {np.mean((torque_np[:, i] - predicted_torque_np[:, i])**2):.6f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/gsang/Projects/Perceiver_IO/data/torque_comparison.jpg')
        plt.close()
        
        # Compute dH/dt analytically using chain rule:
        # dH/dt = (∂H/∂q)^T * q_dot + (∂H/∂p)^T * p_dot
        # where p_dot = -∂H/∂q + torque (Hamilton's equation with external torque)
        
        # IMPORTANT: PyTorch Lightning runs test/val under inference_mode/no_grad by default.
        # torch.set_grad_enabled(True) does NOT override inference_mode, so we must disable it here.
        with torch.inference_mode(False):
            with torch.enable_grad():
                mom_grad = mom.detach().clone().requires_grad_(True)
                qpos_grad = qpos.detach().clone().requires_grad_(True)

                # Recompute H with gradients enabled using the wrapper to handle scaling
                H_for_grad = self(mom_grad, qpos_grad)

                # Compute gradients ∂H/∂p and ∂H/∂q
                dH_dp, dH_dq = torch.autograd.grad(
                    H_for_grad.sum(),
                    (mom_grad, qpos_grad),
                    create_graph=False,
                )

                # Compute p_dot using Hamilton's equation: p_dot = -∂H/∂q + torque
                p_dot_model = -dH_dq + predicted_torque

                # NOTE:
                # The identity dH/dt = qdot^T * tau holds when qdot = dH/dp and pdot = -dH/dq + tau
                # are both taken from the *same* Hamiltonian H. If we mix dataset qvel with model dH/dp,
                # the residual term (dH/dq)^T (qvel - dH/dp) can dominate and make the "energy MSE" large.

                # Self-consistency (model) energy rate:
                # dH/dt = (∂H/∂q)^T * qdot_model + (∂H/∂p)^T * pdot_model
                qdot_model = dH_dp
                dHdt_model_chain = (
                    torch.sum(dH_dq * qdot_model, dim=-1, keepdim=True)
                    + torch.sum(dH_dp * p_dot_model, dim=-1, keepdim=True)
                )  # [B, 1]

                # Equivalent power form (model): qdot_model^T * tau
                dHdt_model_power = torch.einsum('b i, b i -> b', qdot_model, predicted_torque).unsqueeze(-1)

                # Data-consistency energy rate: derivative of learned H along the *data* trajectory
                # dH/dt = (∂H/∂q)^T * qvel_data + (∂H/∂p)^T * mom_dot_data
                dHdt_data_chain = (
                    torch.sum(dH_dq * qvel, dim=-1, keepdim=True)
                    + torch.sum(dH_dp * mom_dot, dim=-1, keepdim=True)
                )  # [B, 1]

                # Power from data velocity: qvel_data^T * tau
                dHdt_data_power = torch.einsum('b i, b i -> b', qvel, predicted_torque).unsqueeze(-1)

        # MSE checks for energy identity (model self-consistency) and data consistency
        mse_energy_model = nn.functional.mse_loss(dHdt_model_chain, dHdt_model_power)
        mse_energy_data = nn.functional.mse_loss(dHdt_data_chain, dHdt_data_power)

        # MSE checks for Hamilton's equations (raw + normalized)
        # Use ground truth torque for physics check to isolate HNN performance
        err_qvel = (dH_dp - qvel)
        err_mom_dot = (-dH_dq + torque - mom_dot)
        mse_qvel = torch.mean(err_qvel ** 2)
        mse_mom_dot = torch.mean(err_mom_dot ** 2)
        nmse_qvel = torch.mean((err_qvel ** 2) / (self.qvel_var + self.eps))
        nmse_mom_dot = torch.mean((err_mom_dot ** 2) / (self.mom_dot_var + self.eps))

        # Reduce log/plot spam in DDP: print/plot once from rank 0, first batch.
        if getattr(self, "trainer", None) is not None and self.trainer.is_global_zero and batch_idx == 0:
            # =====================================================================
            # TEST 1: Hamilton's Equations (what the model is trained on)
            # =====================================================================
            print(f"\n{'='*60}")
            print(f"TEST 1: Hamilton's Equations (Primary Training Objective)")
            print(f"{'='*60}")
            print(f"MSE(qvel_data, dH/dp):              {mse_qvel.item():.6e}")
            print(f"MSE(mom_dot_data, -dH/dq + torque): {mse_mom_dot.item():.6e}")
            print(f"nMSE(qvel_data, dH/dp):             {nmse_qvel.item():.6e}")
            print(f"nMSE(mom_dot_data, -dH/dq + torque):{nmse_mom_dot.item():.6e}")
            
            # =====================================================================
            # TEST 2: Model Self-Consistency (algebraic identity, should be ~0)
            # =====================================================================
            print(f"\n{'='*60}")
            print(f"TEST 2: Model Self-Consistency (Algebraic Identity)")
            print(f"{'='*60}")
            print(f"MSE(dH/dt chain rule, qdot^T*tau): {mse_energy_model.item():.6e}")
            print(f"  -> This should be ~0 (machine precision)")
            
            # =====================================================================
            # TEST 3: Energy Conservation (τ=0) - The Hamiltonian Structure Test
            # =====================================================================
            print(f"\n{'='*60}")
            print(f"TEST 3: Energy Conservation (τ=0)")
            print(f"{'='*60}")
            
            # Sample a subset of initial conditions for trajectory integration
            num_test_samples = min(100, mom.shape[0])
            test_indices = torch.randperm(mom.shape[0])[:num_test_samples]
            p0 = mom[test_indices]
            q0 = qpos[test_indices]
            
            # Integration parameters
            dt = 0.00025  # Same as data collection timestep
            num_integration_steps = 500
            zero_torque = torch.zeros_like(p0)
            
            # Integrate with τ=0
            with torch.inference_mode(False):
                p_traj, q_traj, H_traj = self.integrate_trajectory(
                    p0, q0, zero_torque, dt, num_integration_steps
                )
            
            # Compute energy drift statistics
            H_initial = H_traj[0]  # [B, 1]
            H_final = H_traj[-1]   # [B, 1]
            H_drift = (H_final - H_initial).abs()
            H_drift_relative = H_drift / (H_initial.abs() + 1e-8)
            
            print(f"Integration: {num_integration_steps} steps × dt={dt} = {num_integration_steps*dt:.4f}s")
            print(f"H(t=0) stats: Mean={H_initial.mean().item():.4e}, Std={H_initial.std().item():.4e}")
            print(f"H(t=T) stats: Mean={H_final.mean().item():.4e}, Std={H_final.std().item():.4e}")
            print(f"Absolute drift |H(T)-H(0)|: Mean={H_drift.mean().item():.4e}, Max={H_drift.max().item():.4e}")
            print(f"Relative drift |H(T)-H(0)|/|H(0)|: Mean={H_drift_relative.mean().item():.4e}, Max={H_drift_relative.max().item():.4e}")
            print(f"  -> Small drift = good Hamiltonian structure")
            
            # Plot energy conservation
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            t_steps = np.arange(num_integration_steps + 1) * dt
            
            # Left: H trajectory for a few samples
            H_traj_np = H_traj[:, :min(10, num_test_samples), 0].detach().cpu().numpy()
            for i in range(H_traj_np.shape[1]):
                axes[0].plot(t_steps, H_traj_np[:, i], alpha=0.7, linewidth=0.5)
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('H (Hamiltonian)')
            axes[0].set_title(f'Energy Conservation (τ=0): Mean drift = {H_drift.mean().item():.4e}')
            axes[0].grid(True, alpha=0.3)
            
            # Right: Histogram of relative drift
            axes[1].hist(H_drift_relative.detach().cpu().numpy().flatten(), bins=50, edgecolor='black')
            axes[1].set_xlabel('Relative Energy Drift |H(T)-H(0)|/|H(0)|')
            axes[1].set_ylabel('Count')
            axes[1].set_title(f'Distribution of Energy Drift (N={num_test_samples})')
            axes[1].axvline(H_drift_relative.mean().item(), color='r', linestyle='--', label=f'Mean={H_drift_relative.mean().item():.4e}')
            axes[1].legend()
            
            plt.tight_layout()
            plt.savefig('/home/gsang/Projects/Perceiver_IO/data/energy_conservation_test.jpg')
            plt.close()
            
            # =====================================================================
            # TEST 4: One-Step Prediction Accuracy
            # =====================================================================
            print(f"\n{'='*60}")
            print(f"TEST 4: One-Step Prediction Accuracy")
            print(f"{'='*60}")
            
            # Predicted next state using learned dynamics
            q_next_pred = qpos + dt * dH_dp
            p_next_pred = mom + dt * (-dH_dq + predicted_torque)
            
            # Ground truth next state (Euler approximation from data)
            q_next_gt = qpos + dt * qvel
            p_next_gt = mom + dt * mom_dot
            
            # MSE for one-step prediction
            mse_q_onestep = nn.functional.mse_loss(q_next_pred, q_next_gt)
            mse_p_onestep = nn.functional.mse_loss(p_next_pred, p_next_gt)
            
            print(f"MSE(q_pred, q_gt) one-step: {mse_q_onestep.item():.6e}")
            print(f"MSE(p_pred, p_gt) one-step: {mse_p_onestep.item():.6e}")
            print(f"  -> These are dt²-scaled versions of Hamilton equation MSEs")
            
            # =====================================================================
            # TEST 5: Multi-Step Trajectory Rollout (with torque)
            # =====================================================================
            print(f"\n{'='*60}")
            print(f"TEST 5: Multi-Step Trajectory Rollout (with τ)")
            print(f"{'='*60}")
            
            # Use first few samples and integrate with constant torque
            num_rollout_samples = min(50, mom.shape[0])
            p0_rollout = mom[:num_rollout_samples]
            q0_rollout = qpos[:num_rollout_samples]
            tau_rollout = predicted_torque[:num_rollout_samples]  # Constant torque
            
            rollout_steps = 100
            with torch.inference_mode(False):
                p_rollout, q_rollout, H_rollout = self.integrate_trajectory(
                    p0_rollout, q0_rollout, tau_rollout, dt, rollout_steps
                )
            
            # Check if trajectory stays bounded (not exploding)
            q_max = q_rollout.abs().max().item()
            p_max = p_rollout.abs().max().item()
            
            print(f"Rollout: {rollout_steps} steps with constant τ")
            print(f"Max |q| during rollout: {q_max:.4f}")
            print(f"Max |p| during rollout: {p_max:.4f}")
            print(f"  -> Bounded = stable dynamics")
            
            # =====================================================================
            # Summary Statistics
            # =====================================================================
            print(f"\n{'='*60}")
            print(f"SUMMARY")
            print(f"{'='*60}")
            print(f"Gradient stats:")
            print(f"  dH/dp: Mean={dH_dp.mean().item():.4e}, Std={dH_dp.std().item():.4e}")
            print(f"  dH/dq: Mean={dH_dq.mean().item():.4e}, Std={dH_dq.std().item():.4e}")
            print(f"Data stats:")
            print(f"  qvel:  Mean={qvel.mean().item():.4e}, Std={qvel.std().item():.4e}")
            print(f"  mom_dot: Mean={mom_dot.mean().item():.4e}, Std={mom_dot.std().item():.4e}")
            print(f"{'='*60}\n")

# -----------------------------------------------------------------------------
# 5. Main
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Optimize matmul performance for NVIDIA A100 GPUs
    torch.set_float32_matmul_precision('high')
    
    mode = 'test' # 'train' or 'test'
    predict_torque = False
    use_torque = True

    train_file = "/home/gsang/Projects/Perceiver_IO/data/traj_40000-steps_4000.h5"
    test_file = "/home/gsang/Projects/Perceiver_IO/data/traj_4000-steps_4000.h5"

    test_checkpoint_file = "/home/gsang/Projects/Perceiver_IO/Projects/Perceiver_IO/checkpoints/SeperableHNN(dim1024)-CELU-epoch-epoch=999.ckpt"
    if mode == 'train':
        print("-"*60)
        print(" "*25+"Start Training")
        print("-"*60)
        print(f"Loading dataset from {train_file}...")

        full_dataset = TrajectoryHNNCached(train_file, trajectory_length=1000)
        
        # Split into train/val (e.g., 90/10 split)
        # If the file has 2000 samples, this gives 1800 train, 200 val
        # train_len = int(0.9 * len(full_dataset))
        # val_len = len(full_dataset) - train_len
        # train_data, val_data = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

        train_data = full_dataset
        val_data = TrajectoryHNNCached(test_file)
        
        # Balanced batch size for both throughput and convergence
        batch_size_per_gpu = 8192 # Increased batch size
        
        train_loader = DataLoader(
            train_data, 
            batch_size=batch_size_per_gpu, 
            shuffle=True, 
            num_workers=8, # Increased workers
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_data, 
            batch_size=batch_size_per_gpu, 
            shuffle=False, 
            num_workers=8, # Increased workers
            pin_memory=True,
            persistent_workers=True
        )


        checkpoint_callback = ModelCheckpoint(
            dirpath='Projects/Perceiver_IO/checkpoints',
            filename='SeperableHNN(dim1024)-CELU-epoch-{epoch}',
            every_n_epochs=50,  # Save every 50 epochs to reduce I/O
            save_top_k=-1)     # Keep all checkpoints (don't delete old ones)

        verify_callback = PhysicsCheckCallback(check_every_n_epochs=50, dt=0.00025)  # Run less frequently
        # Only refresh progress bar every 100 batches - prevents SSH lag!
        progress_bar = TQDMProgressBar(refresh_rate=100)
        wandb_logger = WandbLogger(project='HNN_Hinge', name='SeperableHNN(dim1024)-3D-Hinge-CELU', save_dir='Projects/Perceiver_IO/wandb')
        
    elif mode == 'test':
        print("-"*60)
        print(" "*25+"Start Testing")
        print("-"*60)
        test_data = TrajectoryHNNCached(test_file)
        
        test_loader = DataLoader(test_data, batch_size=8192*2, shuffle=False, num_workers=4)
        
    # Detect dimension and statistics from dataset
    test_data = None
    if mode == 'test':
        test_data = TrajectoryHNNCached(test_file)
        
    sample = train_data[0] if mode == 'train' else test_data[0]
    dim = sample['qpos'].shape[0]
    print(f"Detected dataset dimension: {dim}")
    
    qvel_var, mom_dot_var = 1.0, 1.0
    q_std, p_std = 1.0, 1.0
    
    # Calculate statistics from whichever dataset we are actually using
    stats_data = train_data if mode == 'train' else test_data
    if stats_data is not None:
        qvel_var = stats_data.qvel.var(dim=0, unbiased=False)
        mom_dot_var = stats_data.mom_dot.var(dim=0, unbiased=False)
        q_std = stats_data.qpos.std(dim=0, unbiased=False)
        p_std = stats_data.mom.std(dim=0, unbiased=False)
        print(f"Calculated dataset statistics ({mode} set):")
        print(f"  - qvel_var (mean): {qvel_var.mean().item():.4f}, mom_dot_var (mean): {mom_dot_var.mean().item():.4f}")
        print(f"  - q_std (mean): {q_std.mean().item():.4f}, p_std (mean): {p_std.mean().item():.4f}")

    pl_model = HNNWrapper(dim, dim, use_torque=use_torque, predict_torque=predict_torque,
                          qvel_var=qvel_var, mom_dot_var=mom_dot_var,
                          q_std=q_std, p_std=p_std)
    
    trainer = pl.Trainer(
        max_epochs=1000, 
        accelerator='gpu', 
        devices=[1,2], 
        strategy='ddp_find_unused_parameters_true',
        # Revert to float32 for high-precision gradients required by HNNs
        precision=32,
        callbacks=[checkpoint_callback, verify_callback, progress_bar] if mode == 'train' else [], 
        logger=wandb_logger if mode == 'train' else None,
        enable_progress_bar=True,
        log_every_n_steps=50,
        # Gradient clipping prevents mom_dot/torque outliers from destabilizing training
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        # Speed optimizations for huge datasets:
        limit_train_batches=2000,   # Increased for larger model capacity
        limit_val_batches=0.05,      # Validate only on 5% of the val set
        val_check_interval=1.0,      # Check val at the end of every "virtual" epoch
        )

    if mode == 'train':
        trainer.fit(pl_model, train_loader, val_loader)
    elif mode == 'test':
        trainer.test(pl_model, test_loader, ckpt_path=test_checkpoint_file)

