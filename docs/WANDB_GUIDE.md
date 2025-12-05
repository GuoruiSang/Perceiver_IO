# Weights & Biases Integration Guide

Complete guide for using Weights & Biases (W&B) logging with Trajectory DPF.

## Table of Contents
- [Installation & Setup](#installation--setup)
- [Quick Reference](#quick-reference)
- [Usage Examples](#usage-examples)
- [What Gets Logged](#what-gets-logged)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Installation & Setup

### Install W&B

```bash
pip install wandb
```

### Login to W&B

```bash
# Interactive login
wandb login

# Or use API key directly
export WANDB_API_KEY=your_key_here
```

## Quick Reference

### Basic Commands

```bash
# Enable W&B Logging
python src/training/train.py --h5_path data/data.h5 --wandb --epochs 1000

# Name Your Experiment
python src/training/train.py --wandb --wandb_run_name "my-experiment" ...

# Specify Project
python src/training/train.py --wandb --wandb_project "my-project" ...
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--wandb` | Enable W&B (required) | False |
| `--wandb_project` | Project name | trajectory-dpf |
| `--wandb_entity` | Your username/team | Current user |
| `--wandb_run_name` | Custom run name | Auto-generated |
| `--log_samples_every_n_epochs` | Sample logging frequency | 50 |

## Usage Examples

### Full Training with W&B

```bash
python src/training/train.py \
    --h5_path data/mujoco_dataset.h5 \
    --batch_size 128 \
    --epochs 1000 \
    --lr 1e-4 \
    --num_latents 256 \
    --num_latent_channels 256 \
    --diffusion_steps 1000 \
    --checkpoint_dir checkpoints \
    --wandb \
    --wandb_project trajectory-dpf \
    --wandb_run_name "experiment-large-model" \
    --log_samples_every_n_epochs 50
```

### Quick Test (10 epochs)

```bash
python src/training/train.py \
    --h5_path data/mujoco_dataset.h5 \
    --epochs 10 \
    --wandb \
    --log_samples_every_n_epochs 5
```

### Disable W&B (backward compatible)

```bash
python src/training/train.py --h5_path data/mujoco_dataset.h5 --epochs 1000
# No --wandb flag = no W&B logging
```

## What Gets Logged

### Metrics (Every Step/Epoch)

| Metric | Description | Frequency |
|--------|-------------|-----------|
| `train_loss` | Training MSE loss | Every step |
| `val_loss` | Validation MSE loss | Every epoch |
| `val_loss_qpos` | Position loss | Every epoch |
| `val_loss_qvel` | Velocity loss | Every epoch |
| `val_loss_torque` | Torque loss | Every epoch |
| `train_noise_std` | Noise statistics | Every epoch |
| `train_pred_std` | Prediction statistics | Every epoch |
| `train_diffusion_t` | Average diffusion timestep | Every epoch |
| `train_context_fraction` | Context/query split ratio | Every epoch |

### Sample Trajectories (Periodic)

- Generated trajectory visualizations showing qpos, qvel, and torque
- Logged every N epochs (configurable with `--log_samples_every_n_epochs`)

### Hyperparameters

All hyperparameters are automatically logged:
- Dataset information (path, size, dimensions)
- Model architecture (latents, channels)
- Training configuration (batch size, learning rate, diffusion steps)

### Model Checkpoints

- Best model checkpoints are automatically uploaded to W&B
- Can be downloaded for inference later

## Viewing Results

### Web Dashboard

Visit: `https://wandb.ai/YOUR_USERNAME/trajectory-dpf`

### Compare Runs

In W&B UI:
1. Select multiple runs
2. Click "Compare"
3. View side-by-side metrics

### Download Best Model

```bash
python scripts/analyze_wandb_runs.py \
    --project trajectory-dpf \
    --download_best \
    --output_dir ./models
```

## Troubleshooting

### Common Issues

#### 1. "Fatal error while uploading data" Warning

**Error Message:**
```
wandb: WARNING Fatal error while uploading data. Some run data will not be synced, 
but it will still be written to disk. Use `wandb sync` at the end of the run to try uploading.
```

**Cause:** Network bandwidth limitations, firewall issues, or heavy data logging.

**Solutions:**

##### Option A: Continue Training and Sync Later

```bash
# After training finishes
cd /home/gsang/Projects/Perceiver_IO/checkpoints/wandb/

# Sync the specific run
wandb sync run-<TIMESTAMP>-<RUN_ID>
```

##### Option B: Use Offline Mode

```bash
export WANDB_MODE=offline
python src/training/train.py --wandb --wandb_project DPF-trajectory

# After training, sync all runs
wandb sync checkpoints/wandb/
```

##### Option C: Reduce Logging Frequency

```bash
python src/training/train.py \
    --wandb \
    --log_samples_every_n_epochs 100 \
    --wandb_project DPF-trajectory
```

##### Option D: Increase Network Timeout

```bash
export WANDB_HTTP_TIMEOUT=300  # 5 minutes
python src/training/train.py --wandb --wandb_project DPF-trajectory
```

#### 2. W&B Not Installed

```bash
pip install wandb
wandb login
```

#### 3. Authentication Issues

```bash
wandb login --relogin
```

#### 4. Network Connectivity Issues

**Check W&B Status:**
```bash
wandb status
```

**Test Connection:**
```bash
wandb online
```

**Use Local Logging Only:**
```bash
export WANDB_MODE=dryrun  # Logs locally but doesn't sync
python src/training/train.py --wandb
```

#### 5. Storage Space Issues

**Check Storage:**
```bash
# Check disk space
df -h /home/gsang/Projects/Perceiver_IO/checkpoints

# Check W&B cache size
du -sh ~/.cache/wandb
```

**Clean Up:**
```bash
# Remove old W&B runs
wandb artifact cache cleanup

# Or manually remove old runs
rm -rf checkpoints/wandb/run-*
```

#### 6. Firewall/Proxy Issues

**Set Proxy:**
```bash
export WANDB_HTTP_PROXY=http://proxy.example.com:8080
export WANDB_HTTPS_PROXY=https://proxy.example.com:8080
```

**Use Self-Hosted W&B Server:**
```bash
export WANDB_BASE_URL=https://your-wandb-server.com
wandb login --host=https://your-wandb-server.com
```

### Quick Fix Reference

| Issue | Quick Fix |
|-------|-----------|
| Sync error during training | Let it continue, sync manually later with `wandb sync` |
| Can't connect | Use `export WANDB_MODE=offline` |
| Too much disk space | `wandb artifact cache cleanup` |
| Training too slow | `--log_samples_every_n_epochs 200` |
| Need to disable | `export WANDB_MODE=disabled` |

## Best Practices

### 1. Use Offline Mode for Long Training Runs

```bash
# Train offline
export WANDB_MODE=offline
python src/training/train.py --wandb

# Sync later when convenient
wandb sync --sync-all checkpoints/wandb/
```

### 2. Reduce Logging Frequency for Very Long Runs

```bash
python src/training/train.py \
    --wandb \
    --log_samples_every_n_epochs 200  # Instead of default 50
```

### 3. Organize Experiments with Naming Conventions

```bash
--wandb_run_name "exp1-lr1e4"  # Use clear naming convention
--wandb_run_name "exp1-lr5e4"
--wandb_run_name "exp1-lr1e3"
```

### 4. Batch Sync Multiple Runs

```bash
cd checkpoints/wandb/
for run in run-*/; do
    wandb sync "$run"
done
```

### 5. Use Silent Mode

```bash
export WANDB_SILENT=true
python src/training/train.py --wandb
```

## Environment Variables Reference

```bash
# Disable W&B
export WANDB_MODE=disabled

# Offline mode (log locally, sync later)
export WANDB_MODE=offline

# Dry run (no cloud sync at all)
export WANDB_MODE=dryrun

# Silent mode
export WANDB_SILENT=true

# Increase timeout
export WANDB_HTTP_TIMEOUT=300

# Change cache directory
export WANDB_CACHE_DIR=/path/to/cache

# Change data directory
export WANDB_DIR=/path/to/data
```

## Performance Tips

### For High-Bandwidth Environments

Default settings work fine:
```bash
python src/training/train.py --wandb
```

### For Low-Bandwidth Environments

Use offline mode + manual sync:
```bash
export WANDB_MODE=offline
python src/training/train.py --wandb
# Sync during off-peak hours
wandb sync --sync-all checkpoints/wandb/
```

### For Shared/Cluster Environments

Use project-specific directory:
```bash
export WANDB_DIR=/scratch/user/wandb
python src/training/train.py --wandb
```

## Key Metrics to Watch

1. **val_loss** - Overall model performance
2. **val_loss_qpos** - Position prediction quality
3. **val_loss_torque** - Torque prediction quality
4. **Generated trajectories** - Visual quality check

## Advanced Features

### Hyperparameter Sweeps

Create `sweep.yaml`:
```yaml
program: src/training/train.py
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  lr:
    values: [1e-5, 5e-5, 1e-4, 5e-4]
  num_latents:
    values: [128, 256, 512]
  batch_size:
    values: [64, 128, 256]
```

Run sweep:
```bash
wandb sweep sweep.yaml
wandb agent SWEEP_ID
```

### Analysis Scripts

Analyze all runs:
```bash
python scripts/analyze_wandb_runs.py \
    --project trajectory-dpf \
    --entity YOUR_USERNAME
```

Compare metrics:
```bash
python scripts/analyze_wandb_runs.py \
    --project trajectory-dpf \
    --metric val_loss \
    --output_dir ./analysis
```

## Emergency: Disable W&B Mid-Training

If W&B is causing issues during training:

1. **Stop training gracefully** (Ctrl+C once, wait for checkpoint)
2. **Disable W&B:**
   ```bash
   export WANDB_MODE=disabled
   ```
3. **Resume training without W&B**

## Getting Help

1. **Check W&B status page:** https://status.wandb.ai/
2. **W&B documentation:** https://docs.wandb.ai/
3. **Community forum:** https://community.wandb.ai/
4. **GitHub issues:** https://github.com/wandb/wandb/issues

