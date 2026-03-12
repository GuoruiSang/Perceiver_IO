import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import mujoco
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.data.dataset import TrajectoryHNNCached


def infer_torque_gain_from_xml(xml_path: str, coordinate_dim: int, default_gain: float = 1.0):
    """Infer per-joint motor gear gains from MuJoCo XML actuator motors.

    Falls back to `default_gain` when XML is missing/unreadable or actuator parsing fails.
    """
    gains = np.full((coordinate_dim,), float(default_gain), dtype=np.float32)
    if not xml_path or not os.path.exists(xml_path):
        return gains
    try:
        root = ET.parse(xml_path).getroot()
        motors = root.findall(".//actuator/motor")
        if not motors:
            return gains
        for i, motor in enumerate(motors[:coordinate_dim]):
            gear_attr = motor.get("gear", None)
            if gear_attr is None:
                continue
            # MuJoCo allows vector gear; use first component for hinge motors.
            first = float(gear_attr.strip().split()[0])
            gains[i] = first
    except Exception:
        return gains
    return gains


# -----------------------------------------------------------------------------
# 1. Verification Callback
# -----------------------------------------------------------------------------
class PhysicsCheckCallback(pl.Callback):
    def __init__(self, check_every_n_epochs=5, dt=0.0002, xml_path=str(project_root / "configs" / "rigid_arm_hinge.xml")):
        self.check_every_n_epochs = check_every_n_epochs
        self.dt = dt  # Must match training data timestep!
        self.xml_path = xml_path

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        # Sanity validation runs before training starts; heavy MuJoCo/W&B work here
        # can stall the handoff into epoch 0 and produces no useful signal.
        if getattr(trainer, "sanity_checking", False):
            return
        if trainer.current_epoch % self.check_every_n_epochs != 0:
            return
        
        device = pl_module.device
        
        model = mujoco.MjModel.from_xml_path(self.xml_path)
        model.opt.timestep = self.dt
        data = mujoco.MjData(model)
        
        # The Reacher HNN dataset may drop target states and keep only arm DoFs.
        # Match callback inputs to the model's learned state dimension to avoid
        # feeding extra MuJoCo coordinates that were not used in training.
        state_dim = int(pl_module.q_std.numel()) if torch.is_tensor(pl_module.q_std) else int(np.size(pl_module.q_std))
        state_dim = min(state_dim, model.nq, model.nv)

        # Initialize full MuJoCo state, but only randomize the learned arm subspace.
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[:state_dim] = np.random.uniform(-0.5, 0.5, size=(state_dim,))
        data.qvel[:state_dim] = np.random.uniform(-0.5, 0.5, size=(state_dim,))

        M = np.zeros((model.nv, model.nv))
        mujoco.mj_forward(model, data)
        # Use mj_fullM instead of obsolete MjModel functions
        mujoco.mj_fullM(model, M, data.qM)

        seq_qpos = [data.qpos[:state_dim].copy()]
        seq_qvel = [data.qvel.copy()]
        seq_mom = [(M @ data.qvel)[:state_dim].copy()]
        seq_true_energy = [data.energy[0] + data.energy[1]]  # KE + PE from MuJoCo

        # Match training trajectory length (1000 steps)
        num_steps = 500
        for _ in range(num_steps):
            mujoco.mj_step(model, data)
            mujoco.mj_fullM(model, M, data.qM)
            seq_qpos.append(data.qpos[:state_dim].copy())
            seq_qvel.append(data.qvel.copy())
            seq_mom.append((M @ data.qvel)[:state_dim].copy())
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
        out_dir = str(project_root / 'plots' / 'training_debug')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f'{out_dir}/Energy_Conservation.jpg')
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


class StructuredHNN(nn.Module):
    """Physics-structured HNN: enforces T(q,p) = 0.5 * p^T M^{-1}(q) p.

    For mechanical systems, kinetic energy is ALWAYS quadratic in momentum.
    Instead of learning an arbitrary T(p,q), we learn the inverse mass matrix
    M^{-1}(q) via its Cholesky decomposition (guaranteed SPD).

    V(q) uses [q, sin(q), cos(q)] features since gravity potential depends
    on trigonometric functions of joint angles.
    """
    def __init__(self, coordinate_dim, momenta_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        self.dim = coordinate_dim
        # Number of free parameters in lower-triangular L: dim*(dim+1)/2
        self.num_chol_params = coordinate_dim * (coordinate_dim + 1) // 2

        # Input features: [q, sin(q), cos(q)]
        trig_input_dim = 3 * coordinate_dim

        # Network to predict Cholesky factor L(q) of M^{-1}(q)
        chol_layers = []
        chol_layers.append(nn.Linear(trig_input_dim, hidden_dim))
        chol_layers.append(nn.SiLU())
        for _ in range(num_layers - 2):
            chol_layers.append(nn.Linear(hidden_dim, hidden_dim))
            chol_layers.append(nn.SiLU())
        chol_layers.append(nn.Linear(hidden_dim, self.num_chol_params))
        self.cholesky_net = nn.Sequential(*chol_layers)

        # Network to predict V(q) with trig features
        v_layers = []
        v_layers.append(nn.Linear(trig_input_dim, hidden_dim))
        v_layers.append(nn.SiLU())
        for _ in range(num_layers - 2):
            v_layers.append(nn.Linear(hidden_dim, hidden_dim))
            v_layers.append(nn.SiLU())
        v_layers.append(nn.Linear(hidden_dim, 1))
        self.potential_net = nn.Sequential(*v_layers)

        # Indices for filling lower triangular matrix
        self.register_buffer('tril_rows', torch.tril_indices(coordinate_dim, coordinate_dim)[0])
        self.register_buffer('tril_cols', torch.tril_indices(coordinate_dim, coordinate_dim)[1])
        self.register_buffer('diag_idx', torch.arange(coordinate_dim))

    def _trig_features(self, q):
        return torch.cat([q, torch.sin(q), torch.cos(q)], dim=-1)

    def _get_cholesky(self, q):
        """Predict Cholesky factor L(q) such that M^{-1}(q) = L @ L^T (SPD)."""
        features = self._trig_features(q)
        raw = self.cholesky_net(features)  # [B, num_chol_params]

        B = q.shape[0]
        L = torch.zeros(B, self.dim, self.dim, device=q.device, dtype=q.dtype)
        L[:, self.tril_rows, self.tril_cols] = raw

        # Softplus on diagonal ensures positive definiteness
        L[:, self.diag_idx, self.diag_idx] = nn.functional.softplus(
            L[:, self.diag_idx, self.diag_idx]
        ) + 1e-4

        return L

    def forward(self, p, q):
        # Kinetic energy: T = 0.5 * p^T M^{-1}(q) p = 0.5 * ||L(q)^T p||^2
        L = self._get_cholesky(q)                                    # [B, dim, dim]
        Ltp = torch.bmm(L.transpose(1, 2), p.unsqueeze(-1))         # [B, dim, 1]
        T = 0.5 * (Ltp.squeeze(-1) ** 2).sum(dim=-1, keepdim=True)  # [B, 1]

        # Potential energy: V(q)
        features = self._trig_features(q)
        V = self.potential_net(features)  # [B, 1]

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
                 qvel_var=1.0, mom_dot_var=1.0, q_std=1.0, p_std=1.0, eps: float = 1e-8,
                 model_type='structured', hidden_dim=256, num_layers=4, lr=3e-4,
                 torque_gain=1.0):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpoint loading
        if model_type == 'structured':
            self.model = StructuredHNN(coordinate_dim, momenta_dim,
                                       hidden_dim=hidden_dim, num_layers=num_layers)
        elif model_type == 'separable':
            self.model = SeperableHNN(coordinate_dim, momenta_dim)
        else:
            self.model = HNN(coordinate_dim, momenta_dim)
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
        self.register_buffer('torque_gain', torch.as_tensor(torque_gain, dtype=torch.float32))

    def _scale_torque(self, torque):
        """Convert control-space torque to generalized force-space via gear gain."""
        if torque is None:
            return None
        return torque * self.torque_gain

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
        lr = self.hparams.get('lr', 3e-4)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.0, fused=True)
        
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
                    dpdt_pred = dpdt_pred + self._scale_torque(torque_target)
                # Use predicted torque
                elif self.use_torque and self.predict_torque:
                    torque_pred = self.torque_predictor(q_raw, dqdt_target, qacc_target)
                    dpdt_pred = dpdt_pred + self._scale_torque(torque_pred)
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

# -----------------------------------------------------------------------------
# 5. Main
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the active structured HNN.")
    parser.add_argument("--mode", type=str, default="train", choices=["train"],
                        help="Only train mode is part of the active experiment contract.")
    parser.add_argument("--train_file", type=str, default=str(project_root / "data" / "3dof" / "traj_40000-steps_4000.h5"))
    parser.add_argument("--test_file", type=str, default=str(project_root / "data" / "3dof" / "traj_2000-steps_4000.h5"),
                        help="Validation dataset path. Kept as --test_file for CLI compatibility.")
    parser.add_argument("--checkpoint_dir", type=str, default=str(project_root / "checkpoints" / "3dof" / "hnn"))
    parser.add_argument("--checkpoint_prefix", type=str, default="StructuredHNN-dim256-traj40000")
    parser.add_argument("--wandb_name", type=str, default="StructuredHNN-3D-Hinge-traj40000")
    parser.add_argument("--xml_path", type=str, default=str(project_root / "configs" / "rigid_arm_hinge.xml"),
                        help="Path to MuJoCo XML file for physics verification callback")
    parser.add_argument("--model_type", type=str, default="structured",
                        choices=["structured", "separable", "hnn"],
                        help="Model architecture: structured (quadratic T), separable (MLP T+V), hnn (single MLP)")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for structured model (default: 256)")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of layers for structured model (default: 4)")
    parser.add_argument("--torque_gain", type=float, default=1.0,
                        help="Fallback scalar gain to map control input to generalized torque.")
    parser.add_argument("--auto_torque_gain_from_xml", action="store_true", default=False,
                        help="Infer per-joint motor gear gain from XML actuator motors.")
    parser.add_argument(
        "--torque_alignment",
        type=str,
        default="auto",
        choices=["auto", "legacy", "transition_next"],
        help="How to align saved torque samples with states in the HNN dataset loader.",
    )
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker processes for cached HNN datasets.")
    parser.add_argument("--disable_verify_callback", action="store_true", default=False,
                        help="Disable the MuJoCo/W&B physics verification callback.")
    args = parser.parse_args()

    # Optimize matmul performance for NVIDIA A100 GPUs
    torch.set_float32_matmul_precision('high')

    predict_torque = False
    use_torque = True

    train_file = args.train_file
    val_file = args.test_file

    print("-"*60)
    print(" "*25+"Start Training")
    print("-"*60)
    print(f"Loading dataset from {train_file}...")

    train_data = TrajectoryHNNCached(
        train_file,
        trajectory_length=1000,
        torque_alignment=args.torque_alignment,
    )
    val_data = TrajectoryHNNCached(
        val_file,
        torque_alignment=args.torque_alignment,
    )

    batch_size_per_gpu = 2048

    use_persistent_workers = False
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f'{args.checkpoint_prefix}-epoch-{{epoch}}',
        every_n_epochs=50,
        save_top_k=-1)

    verify_callback = None
    if not args.disable_verify_callback:
        verify_callback = PhysicsCheckCallback(check_every_n_epochs=50, dt=0.0002, xml_path=args.xml_path)
    progress_bar = TQDMProgressBar(refresh_rate=100)
    wandb_logger = WandbLogger(project='HNN_Hinge', name=args.wandb_name, save_dir=str(project_root / 'wandb'))

    sample = train_data[0]
    dim = sample['qpos'].shape[0]
    print(f"Detected dataset dimension: {dim}")
    
    qvel_var, mom_dot_var = 1.0, 1.0
    q_std, p_std = 1.0, 1.0
    
    qvel_var = train_data.qvel.var(dim=0, unbiased=False)
    mom_dot_var = train_data.mom_dot.var(dim=0, unbiased=False)
    q_std = train_data.qpos.std(dim=0, unbiased=False)
    p_std = train_data.mom.std(dim=0, unbiased=False)
    print("Calculated dataset statistics (train set):")
    print(f"  - qvel_var (mean): {qvel_var.mean().item():.4f}, mom_dot_var (mean): {mom_dot_var.mean().item():.4f}")
    print(f"  - q_std (mean): {q_std.mean().item():.4f}, p_std (mean): {p_std.mean().item():.4f}")

    # LR sqrt scaling: lr = 3e-4 * sqrt(batch_size / 8192)
    lr = 3e-4 * (batch_size_per_gpu / 8192) ** 0.5
    print(f"Learning rate: {lr:.2e} (sqrt-scaled from 3e-4 at bs=8192 to bs={batch_size_per_gpu})")

    torque_gain = np.full((dim,), float(args.torque_gain), dtype=np.float32)
    if args.auto_torque_gain_from_xml:
        torque_gain = infer_torque_gain_from_xml(args.xml_path, dim, default_gain=args.torque_gain)
    print(f"Torque gain used (per dim): {torque_gain.tolist()}")

    pl_model = HNNWrapper(dim, dim, use_torque=use_torque, predict_torque=predict_torque,
                          qvel_var=qvel_var, mom_dot_var=mom_dot_var,
                          q_std=q_std, p_std=p_std,
                          model_type=args.model_type, hidden_dim=args.hidden_dim,
                          num_layers=args.num_layers, lr=lr,
                          torque_gain=torque_gain)
    
    trainer_devices = [0]
    trainer_strategy = 'auto'
    if len(trainer_devices) > 1:
        trainer_strategy = 'ddp_find_unused_parameters_true'

    trainer = pl.Trainer(
        max_epochs=1000, 
        accelerator='gpu', 
        devices=trainer_devices,  # When CUDA_VISIBLE_DEVICES is set, device 0 maps to the selected physical GPU
        strategy=trainer_strategy,
        # Revert to float32 for high-precision gradients required by HNNs
        precision=32,
        callbacks=[cb for cb in [checkpoint_callback, verify_callback, progress_bar] if cb is not None],
        logger=wandb_logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
        # Gradient clipping prevents mom_dot/torque outliers from destabilizing training
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        # Speed optimizations for huge datasets:
        limit_train_batches=2000,
        limit_val_batches=0.05,      # Validate only on 5% of the val set
        val_check_interval=1.0,      # Check val at the end of every "virtual" epoch
        )

    trainer.fit(pl_model, train_loader, val_loader)
