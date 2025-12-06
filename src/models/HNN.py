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
        
        # Compute learned H
        seq_H = []
        for i in range(len(seq_qpos)):
            p_tensor = torch.from_numpy(seq_mom[i]).float().to(device)
            q_tensor = torch.from_numpy(seq_qpos[i]).float().to(device)
            H_val = pl_module.model(p_tensor, q_tensor).detach().cpu().numpy()
            seq_H.append(H_val)
        seq_H = np.array(seq_H).flatten()
        
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
            nn.Linear(coordinate_dim + momenta_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, p, q):
        x = torch.cat([p, q], dim=-1)
        return self.mlp(x)

class TorquePredictor(nn.Module):
    def __init__(self, coordinate_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3*coordinate_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 3),
        )
    
    def forward(self, pos, vel, acc):
        x = torch.cat([pos, vel, acc], dim=-1)
        return self.mlp(x)

# -----------------------------------------------------------------------------
# 3. HNN Wrapper
# -----------------------------------------------------------------------------
class HNNWrapper(pl.LightningModule):
    def __init__(self, coordinate_dim, momenta_dim, use_torque=True, predict_torque=True):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpoint loading
        self.model = HNN(coordinate_dim, momenta_dim)
        self.use_torque = use_torque
        self.predict_torque = predict_torque
        if self.use_torque and self.predict_torque:
            self.torque_predictor = TorquePredictor(coordinate_dim)

    def forward(self, p, q): return self.model(p, q)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def calculate_loss(self, p, q, dqdt_target, dpdt_target, torque_target=None, qaccarget=None):
        with torch.set_grad_enabled(True):
            p = p.requires_grad_(True); q = q.requires_grad_(True)
            H = self.model(p, q)
            grads = torch.autograd.grad(H.sum(), (p, q), create_graph=True)
            dqdt_pred, dpdt_pred = grads[0], -grads[1]
            # Use ground-truth torque
            if self.use_torque and not self.predict_torque:
                dpdt_pred = dpdt_pred + torque_target
                loss_torque = None
            # Use predicted torque
            elif self.use_torque and self.predict_torque:
                # Use dqdt_target or dqdt_pred?
                torque_pred = self.torque_predictor(q, dqdt_target, qaccarget)
                dpdt_pred = dpdt_pred + torque_pred
                loss_torque = nn.functional.mse_loss(torque_target, torque_pred)
                

        loss_dqdt = nn.functional.mse_loss(dqdt_target, dqdt_pred)
        loss_dpdt = nn.functional.mse_loss(dpdt_target, dpdt_pred)

        loss = loss_dqdt + loss_dpdt
        if loss_torque is not None:
            loss += loss_torque
            self.log('loss_torque', loss_torque.item(), prog_bar=True, on_epoch=True)
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
        self.log('val_loss', loss, prog_bar=True)

        predicted_torque = self.torque_predictor(batch['qpos'], batch['qvel'], batch['qacc'])
        self.log('torque loss', nn.functional.mse_loss(batch['torque'], predicted_torque), on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        qpos = batch['qpos']
        qvel = batch['qvel']

        mom = batch['mom']
        mom_dot = batch['mom_dot']
        
        qacc = batch['qacc']
        torque = batch['torque']

        H = self.model(mom, qpos)


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
            axes[i].set_ylim(-10, 10)
            axes[i].set_title(f'Torque Dim {i}, MSE: {np.mean((torque_np[:, i] - predicted_torque_np[:, i])**2):.6f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/gsang/Projects/Perceiver_IO/data/torque_comparison.jpg')
        plt.close()

        dt = 0.0005
        dHdt = torch.zeros_like(H)
        dHdt[1:-1] = (H[2:] - H[:-2]) / (2*dt)
        dHdt[0] = (-3*H[0] + 4*H[1] - H[2]) / (2.0*dt)
        dHdt[-1] = (3*H[-1] - 4*H[-2] + H[-3]) / (2.0*dt)

        dHdt_pred = torch.einsum('b i, b i -> b', predicted_torque, qvel).unsqueeze(-1)

        t = np.arange(1, dHdt.shape[0]+1)

        # Debugging Statistics
        print(f"\n--- Debugging Statistics ---")
        print(f"H shape: {H.shape}")
        print(f"H stats: Mean={H.mean().item():.4e}, Std={H.std().item():.4e}, Min={H.min().item():.4e}, Max={H.max().item():.4e}")
        print(f"dHdt stats: Mean={dHdt.mean().item():.4e}, Std={dHdt.std().item():.4e}, Min={dHdt.min().item():.4e}, Max={dHdt.max().item():.4e}")
        print(f"Power (tau*qvel) stats: Mean={dHdt_pred.mean().item():.4e}, Std={dHdt_pred.std().item():.4e}, Min={dHdt_pred.min().item():.4e}, Max={dHdt_pred.max().item():.4e}")

        
        mse = nn.functional.mse_loss(dHdt, dHdt_pred)
        print(f"Mean of ||dHdt-(predicted_torque.T * qvel)||^2: {mse}")
        print(f"Explanation: The large MSE in dH/dt is likely due to noise amplification. \nSmall fluctuations in H (std={H.std().item():.2e}) divided by dt ({dt}) cause large dH/dt values.")
        print(f"----------------------------\n")
        plt.figure(figsize=(10, 5))
        plt.scatter(t, dHdt.detach().cpu().numpy(), label='dH/dt', s=1)
        plt.scatter(t, dHdt_pred.detach().cpu().numpy(), label='predicted_torque.T * qvel', s=1)
        plt.title(f'Comparison of dHdt and predicted_torque.T * qvel. MSE: {mse.item()}')
        plt.legend()
        plt.savefig('/home/gsang/Projects/Perceiver_IO/data/validation_with_torque_predictor.jpg')
        print(f"Mean of ||dHdt-(predicted_torque.T * qvel)||^2: {nn.functional.mse_loss(dHdt, dHdt_pred)}")

# -----------------------------------------------------------------------------
# 5. Main
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    mode = 'test' # 'train' or 'test'
    predict_torque = True
    use_torque = True

    train_file = "/home/gsang/Projects/Perceiver_IO/data/traj_40000-steps_500.h5"
    test_file = "/home/gsang/Projects/Perceiver_IO/data/traj_2000-steps_500.h5"

    test_checkpoint_file = "/home/gsang/Projects/Perceiver_IO/checkpoints/HNN-epoch-epoch=99.ckpt"
    if mode == 'train':
        print("-"*60)
        print(" "*25+"Start Training")
        print("-"*60)
        print(f"Loading dataset from {train_file}...")

        full_dataset = TrajectoryHNNCached(train_file)
        
        # Split into train/val (e.g., 90/10 split)
        # If the file has 2000 samples, this gives 1800 train, 200 val
        # train_len = int(0.9 * len(full_dataset))
        # val_len = len(full_dataset) - train_len
        # train_data, val_data = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

        train_data = full_dataset
        val_data = TrajectoryHNNCached(test_file)
        
        train_loader = DataLoader(train_data, batch_size=8192, shuffle=True, num_workers=20, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=8192, shuffle=False, num_workers=20, pin_memory=True)


        checkpoint_callback = ModelCheckpoint(
            dirpath='Projects/Perceiver_IO/checkpoints',
            filename='HNN-epoch-{epoch}',
            every_n_epochs=100,  # Save every 5 epochs
            save_top_k=-1)     # Keep all checkpoints (don't delete old ones)

        verify_callback = PhysicsCheckCallback(check_every_n_epochs=1, dt=0.0005)  # Match training data dt!
        # Only refresh progress bar every 100 batches - prevents SSH lag!
        progress_bar = TQDMProgressBar(refresh_rate=100)
        wandb_logger = WandbLogger(project='HNN_Hinge', name='HNN-3D-Hinge-With-PredictedTorque', save_dir='Projects/Perceiver_IO/wandb')
        
    elif mode == 'test':
        print("-"*60)
        print(" "*25+"Start Testing")
        print("-"*60)
        test_data = TrajectoryHNNCached(test_file)
        
        test_loader = DataLoader(test_data, batch_size=500, shuffle=False, num_workers=4)
        
    # Detect dimension from dataset
    sample = train_data[0] if mode == 'train' else test_data[0]
    dim = sample['qpos'].shape[0]
    print(f"Detected dataset dimension: {dim}")

    pl_model = HNNWrapper(dim, dim, use_torque=use_torque, predict_torque=predict_torque)

    trainer = pl.Trainer(
        max_epochs=1000, 
        accelerator='gpu', 
        devices=[5], 
        callbacks=[checkpoint_callback, verify_callback, progress_bar] if mode == 'train' else [], 
        logger=wandb_logger if mode == 'train' else None,
        enable_progress_bar=True,
        log_every_n_steps=50,
        )

    if mode == 'train':
        trainer.fit(pl_model, train_loader, val_loader)
    elif mode == 'test':
        trainer.test(pl_model, test_loader, ckpt_path=test_checkpoint_file)

