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
    def __init__(self, check_every_n_epochs=5, dt=0.0001):
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
        # Kinetic Energy T(p, q) - depends on both (mass matrix can vary with q)
        # Increased depth and switched to Tanh for better gradient flow
        self.kinetic = nn.Sequential(
            nn.Linear(coordinate_dim + momenta_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        
        # Potential Energy V(q) - strictly only depends on q
        self.potential = nn.Sequential(
            nn.Linear(coordinate_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
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
                 qvel_var=1.0, mom_dot_var=1.0, q_std=1.0, p_std=1.0):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpoint loading
        self.model = SeperableHNN(coordinate_dim, momenta_dim)
        self.use_torque = use_torque
        self.predict_torque = predict_torque
        if self.use_torque and self.predict_torque:
            self.torque_predictor = TorquePredictor(coordinate_dim)
        
        # Variances for loss normalization (calculated from dataset statistics)
        # Using register_buffer so they are moved to the correct device but not trained
        self.register_buffer('qvel_var', torch.tensor(float(qvel_var)))
        self.register_buffer('mom_dot_var', torch.tensor(float(mom_dot_var)))
        
        # Input scaling statistics
        self.register_buffer('q_std', torch.tensor(float(q_std)))
        self.register_buffer('p_std', torch.tensor(float(p_std)))

    def forward(self, p, q): 
        # Apply scaling during inference if needed
        p_scaled = p / self.p_std
        q_scaled = q / self.q_std
        return self.model(p_scaled, q_scaled)

    def configure_optimizers(self):
        # Reduced LR and added weight decay as per plan
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=1e-4, fused=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

    def calculate_loss(self, p, q, dqdt_target, dpdt_target, torque_target=None, qacc_target=None):
        loss_torque = None
        # Physically-consistent input scaling:
        # Scale inputs, then adjust gradients by the same factor (chain rule)
        with torch.inference_mode(False):
            with torch.set_grad_enabled(True):
                p_raw = p.detach().requires_grad_(True)
                q_raw = q.detach().requires_grad_(True)
                
                # Scaled inputs for the network
                p_scaled = p_raw / self.p_std
                q_scaled = q_raw / self.q_std
                
                H = self.model(p_scaled, q_scaled)
                
                # dH/dp_raw = (dH/dp_scaled) * (dp_scaled/dp_raw) = (dH/dp_scaled) / p_std
                # However, autograd.grad with respect to p_raw handles this automatically
                # if we define H in terms of p_raw.
                grads = torch.autograd.grad(H.sum(), (p_raw, q_raw), create_graph=True)
                dqdt_pred, dpdt_pred = grads[0], -grads[1]
                
                # Use ground-truth torque
                if self.use_torque and not self.predict_torque:
                    dpdt_pred = dpdt_pred + torque_target
                # Use predicted torque
                elif self.use_torque and self.predict_torque:
                    torque_pred = self.torque_predictor(q_raw, dqdt_target, qacc_target)
                    dpdt_pred = dpdt_pred + torque_pred
                    loss_torque = nn.functional.mse_loss(torque_target, torque_pred)
                

        # Normalized MSE loss: MSE / Variance
        loss_dqdt = nn.functional.mse_loss(dqdt_target, dqdt_pred) / self.qvel_var
        loss_dpdt = nn.functional.mse_loss(dpdt_target, dpdt_pred) / self.mom_dot_var

        # Balanced loss
        loss = loss_dqdt + loss_dpdt
        if loss_torque is not None:
            loss += loss_torque
            self.log('loss_torque', loss_torque.item(), prog_bar=True, on_epoch=True, sync_dist=False)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch['mom'], batch['qpos'], 
                                   batch['qvel'], batch['mom_dot'], 
                                   batch.get('torque', None), batch.get('qacc', None))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch['mom'], batch['qpos'], 
                                   batch['qvel'], batch['mom_dot'], 
                                   batch.get('torque', None), batch.get('qacc', None))
        self.log('val_loss', loss, prog_bar=True, sync_dist=False)

        if self.use_torque and self.predict_torque:
            predicted_torque = self.torque_predictor(batch['qpos'], batch['qvel'], batch['qacc'])
            self.log('torque loss', nn.functional.mse_loss(batch['torque'], predicted_torque), on_epoch=True, sync_dist=False)
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


        predicted_torque = self.torque_predictor(qpos, qvel, qacc)

        print(f"torque mse={nn.functional.mse_loss(torque, predicted_torque)}")
        
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

        dt = 0.0001
        
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
                p_dot = -dH_dq + predicted_torque

                # dH/dt = (∂H/∂q)^T * qvel + (∂H/∂p)^T * p_dot
                dHdt = (
                    torch.sum(dH_dq * qvel, dim=-1, keepdim=True)
                    + torch.sum(dH_dp * p_dot, dim=-1, keepdim=True)
                )  # [B, 1]

        # Power: v^T * predicted_torque (expected dH/dt for energy conservation)
        dHdt_pred = torch.einsum('b i, b i -> b', qvel, predicted_torque).unsqueeze(-1)

        t = np.arange(1, dHdt.shape[0]+1)

        # Debugging Statistics
        print(f"\n--- Debugging Statistics (Analytical dH/dt) ---")
        print(f"H shape: {H.shape}")
        print(f"H stats: Mean={H.mean().item():.4e}, Std={H.std().item():.4e}, Min={H.min().item():.4e}, Max={H.max().item():.4e}")
        print(f"dH/dp (velocity) stats: Mean={dH_dp.mean().item():.4e}, Std={dH_dp.std().item():.4e}")
        print(f"dH/dq stats: Mean={dH_dq.mean().item():.4e}, Std={dH_dq.std().item():.4e}")
        print(f"dH/dt (analytical) stats: Mean={dHdt.mean().item():.4e}, Std={dHdt.std().item():.4e}, Min={dHdt.min().item():.4e}, Max={dHdt.max().item():.4e}")
        print(f"Power (v^T * predicted_torque) stats: Mean={dHdt_pred.mean().item():.4e}, Std={dHdt_pred.std().item():.4e}, Min={dHdt_pred.min().item():.4e}, Max={dHdt_pred.max().item():.4e}")

        
        mse = nn.functional.mse_loss(dHdt, dHdt_pred)
        print(f"MSE between dH/dt and v^T * predicted_torque: {mse.item():.6e}")
        print(f"----------------------------\n")
        plt.figure(figsize=(10, 5))
        plt.scatter(t, dHdt.detach().cpu().numpy(), label='dH/dt (analytical)', s=1)
        plt.scatter(t, dHdt_pred.detach().cpu().numpy(), label='v^T * predicted_torque', s=1)
        plt.title(f'dH/dt = (∂H/∂q)ᵀq̇ + (∂H/∂p)ᵀṗ vs v^T*τ. MSE: {mse.item():.4e}')
        plt.xlabel('Time Step')
        plt.ylabel('Energy Rate')
        plt.legend()
        plt.savefig('/home/gsang/Projects/Perceiver_IO/data/validation_with_predicted_torque_predictor.jpg')
        print(f"MSE(dH/dt, v^T * torque): {mse.item():.6e}")

# -----------------------------------------------------------------------------
# 5. Main
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Optimize matmul performance for NVIDIA A100 GPUs
    torch.set_float32_matmul_precision('high')
    
    mode = 'train' # 'train' or 'test'
    predict_torque = False
    use_torque = True

    train_file = "/home/gsang/Projects/Perceiver_IO/data/traj_40000-steps_4000.h5"
    test_file = "/home/gsang/Projects/Perceiver_IO/data/traj_4000-steps_4000.h5"

    test_checkpoint_file = "/home/gsang/Projects/Perceiver_IO/checkpoints/SeperableHNN-Tanh-epoch-epoch=459.ckpt"
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
        
        # Reduced batch size for faster convergence
        batch_size_per_gpu = 4096
        
        train_loader = DataLoader(
            train_data, 
            batch_size=batch_size_per_gpu, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_data, 
            batch_size=batch_size_per_gpu, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True,
            persistent_workers=True
        )


        checkpoint_callback = ModelCheckpoint(
            dirpath='Projects/Perceiver_IO/checkpoints',
            filename='SeperableHNN(dim128)-Weighted-CELU-LN-epoch-{epoch}',
            every_n_epochs=10,  # Save every 20 epochs to reduce I/O
            save_top_k=-1)     # Keep all checkpoints (don't delete old ones)

        verify_callback = PhysicsCheckCallback(check_every_n_epochs=10, dt=0.0005)  # Run less frequently
        # Only refresh progress bar every 100 batches - prevents SSH lag!
        progress_bar = TQDMProgressBar(refresh_rate=100)
        wandb_logger = WandbLogger(project='HNN_Hinge', name='SeperableHNN(dim256)-3D-Hinge-ScaledInputs-Tanh', save_dir='Projects/Perceiver_IO/wandb')
        
    elif mode == 'test':
        print("-"*60)
        print(" "*25+"Start Testing")
        print("-"*60)
        test_data = TrajectoryHNNCached(test_file)
        
        test_loader = DataLoader(test_data, batch_size=8192*2, shuffle=False, num_workers=4)
        
    # Detect dimension and statistics from dataset
    sample = train_data[0] if mode == 'train' else test_data[0]
    dim = sample['qpos'].shape[0]
    print(f"Detected dataset dimension: {dim}")
    
    qvel_var, mom_dot_var = 1.0, 1.0
    q_std, p_std = 1.0, 1.0
    if mode == 'train':
        qvel_var = train_data.qvel.var().item()
        mom_dot_var = train_data.mom_dot.var().item()
        q_std = train_data.qpos.std().item()
        p_std = train_data.mom.std().item()
        print(f"Calculated dataset statistics:")
        print(f"  - qvel_var: {qvel_var:.4f}, mom_dot_var: {mom_dot_var:.4f}")
        print(f"  - q_std: {q_std:.4f}, p_std: {p_std:.4f}")

    pl_model = HNNWrapper(dim, dim, use_torque=use_torque, predict_torque=predict_torque,
                          qvel_var=qvel_var, mom_dot_var=mom_dot_var,
                          q_std=q_std, p_std=p_std)
    
    trainer = pl.Trainer(
        max_epochs=1000, 
        accelerator='gpu', 
        devices=[0,1], 
        strategy='ddp_find_unused_parameters_true',
        # Use float32 for better gradient stability and precision in HNNs
        precision=32,
        callbacks=[checkpoint_callback, verify_callback, progress_bar] if mode == 'train' else [], 
        logger=wandb_logger if mode == 'train' else None,
        enable_progress_bar=True,
        log_every_n_steps=200, # Reduce logging frequency
        )

    if mode == 'train':
        trainer.fit(pl_model, train_loader, val_loader)
    elif mode == 'test':
        trainer.test(pl_model, test_loader, ckpt_path=test_checkpoint_file)

