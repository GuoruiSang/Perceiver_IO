"""
Exact replica of wandb callback sampling logic for 2DoF.
Same as WandbTrajectoryCallback.on_validation_epoch_end().
"""
import sys, os, warnings, tempfile
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import torch
import matplotlib.pyplot as plt

DPF_CKPT = str(project_root / 'checkpoints' / '2dof' /
    'trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding'
    '&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions'
    ':epoch=2999_val_loss:val_loss=0.0008.ckpt')

if __name__ == '__main__':
    device = torch.device('cuda:0')

    # Load model exactly as in training
    from src.models.trajectory_dpf import TrajectoryDPF
    from src.models.utils import compare_generated_with_reconstructed

    print("Loading model...")
    dpf = TrajectoryDPF.load_from_checkpoint(DPF_CKPT, map_location=device, strict=False)
    ckpt = torch.load(DPF_CKPT, map_location=device, weights_only=False)
    if 'ema_state_dict' in ckpt:
        dpf.ema.load_state_dict(ckpt['ema_state_dict'])
    dpf.eval()
    del ckpt
    torch.cuda.empty_cache()

    # ─── EXACT wandb callback logic ───
    pl_module = dpf
    num_samples = 1  # wandb callback default

    print(f"Sampling (exact wandb logic): num_samples={num_samples}, "
          f"trajectory_length={min(1000, pl_module.max_timesteps)}, "
          f"num_diffusion_steps=100, context_fraction=0.5, use_ema=True, sampler='ddim'")

    # This is line-for-line the wandb callback call
    state, torque = pl_module.sample_trajectories(
        num_samples=num_samples,
        trajectory_length=min(1000, pl_module.max_timesteps),
        num_diffusion_steps=100,
        context_fraction=0.5,
        use_ema=True,
        sampler='ddim'
    )

    # Split state into components (same as callback)
    qpos_dim = pl_module.qpos_dim
    mom_dim = pl_module.mom_dim

    state_traj = state[0]  # Take first sample [T, state_dim]
    torque_traj = torque[0]  # [T, torque_dim]

    trajectory_dict = {
        'seq_qpos': state_traj[:, :qpos_dim],
        'seq_mom': state_traj[:, qpos_dim:],
        'seq_torque': torque_traj,
    }

    # Write XML to temp file (same as callback)
    save_dir = str(project_root / 'plots')
    xml_path = os.path.join(save_dir, '_tmp_model.xml')
    with open(xml_path, 'w') as f:
        f.write(pl_module.xml_content)

    # Call the exact same comparison function
    print("Running compare_generated_with_reconstructed (same as wandb)...")
    mse = compare_generated_with_reconstructed(
        trajectory_dict, xml_path, save_dir,
        dt=pl_module.dt, data_dt=pl_module.data_dt,
        name='wandb_style_comparison'
    )

    os.remove(xml_path)

    print(f"\n{'='*50}")
    print(f"MSE (exact wandb logic):")
    print(f"  qpos: {mse['mse_qpos']:.6f}")
    print(f"  mom:  {mse['mse_mom']:.6f}")
    print(f"  total: {mse['mse_total']:.6f}")
    print(f"{'='*50}")

    # Also run a few more samples to get statistics
    print(f"\nRunning 10 more samples for statistics...")
    mse_list = []
    for i in range(10):
        state_i, torque_i = pl_module.sample_trajectories(
            num_samples=1,
            trajectory_length=min(1000, pl_module.max_timesteps),
            num_diffusion_steps=100,
            context_fraction=0.5,
            use_ema=True,
            sampler='ddim'
        )
        td = {
            'seq_qpos': state_i[0, :, :qpos_dim],
            'seq_mom': state_i[0, :, qpos_dim:],
            'seq_torque': torque_i[0],
        }
        m = compare_generated_with_reconstructed(
            td, str(project_root / 'configs' / 'rigid_arm_hinge_2dof.xml'),
            save_dir, dt=pl_module.dt, data_dt=pl_module.data_dt, name=None
        )
        mse_list.append(m['mse_total'])
        print(f"  Sample {i}: mse_total={m['mse_total']:.6f}")

    print(f"\n{'='*50}")
    print(f"10-sample statistics:")
    print(f"  Mean MSE: {np.mean(mse_list):.6f}")
    print(f"  Std MSE:  {np.std(mse_list):.6f}")
    print(f"  Min MSE:  {np.min(mse_list):.6f}")
    print(f"  Max MSE:  {np.max(mse_list):.6f}")
    print(f"{'='*50}")

    print(f"\nPlot saved: {save_dir}/wandb_style_comparison.jpg")
