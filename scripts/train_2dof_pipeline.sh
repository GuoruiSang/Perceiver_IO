#!/bin/bash
# 2DOF Training Pipeline - Generate data then train DPF + HNN in parallel
set -e
cd /home/gsang/Projects/Perceiver_IO

echo "=== 2DOF Training Pipeline ==="
echo "Start time: $(date)"

# Step 1: Generate 40k training + 2k validation trajectories
echo ""
echo "=== Step 1: Generating 40k trajectories ==="
python scripts/generate_dataset_forward.py \
  --torque_policies "sinusoidal:1.0" \
  --xml_path configs/rigid_arm_hinge_2dof.xml \
  --num_trajectories 40000 \
  --num_val 2000 \
  --num_steps 4000 \
  --save_path data/2dof \
  --num_workers 24

echo "Dataset generation complete: $(date)"

# Step 2: Kill any existing training processes
echo ""
echo "=== Step 2: Killing existing training processes ==="
pkill -9 -f "trajectory_dpf.py.*--mode train" 2>/dev/null || true
pkill -9 -f "HNN.py" 2>/dev/null || true
sleep 3

# Step 3: Start DPF training (GPU 0,1,2,3) and HNN training (GPU 4) in parallel
echo ""
echo "=== Step 3: Starting DPF + HNN training in parallel ==="
mkdir -p logs checkpoints/2dof

# DPF Training (background, GPU 0,1,2) - matches ablation study "original" config
nohup python src/models/trajectory_dpf.py \
  --mode train \
  --h5_path data/2dof/traj_40000-steps_4000.h5 \
  --checkpoint_dir checkpoints/2dof \
  --devices 0 1 2 \
  --epochs 3000 \
  --batch_size 128 \
  --lr 1e-4 \
  --num_decoder_blocks 4 \
  --num_latents 256 \
  --num_latent_channels 256 \
  --diffusion_steps 1000 \
  --wandb true \
  --wandb_project "Trajectory-DPF-2DOF" \
  --resume_from_checkpoint "" \
  > logs/dpf_2dof_training.log 2>&1 &
DPF_PID=$!
echo "DPF training started (PID: $DPF_PID)"

# HNN Training (background, GPU 3)
CUDA_VISIBLE_DEVICES=3 nohup python src/models/HNN.py \
  --mode train \
  --train_file data/2dof/traj_40000-steps_4000.h5 \
  --test_file data/2dof/traj_2000-steps_4000.h5 \
  --checkpoint_dir checkpoints/2dof \
  --checkpoint_prefix "SeperableHNN-2DOF" \
  --wandb_name "SeperableHNN-2DOF-Hinge" \
  --xml_path configs/rigid_arm_hinge_2dof.xml \
  > logs/hnn_2dof_training.log 2>&1 &
HNN_PID=$!
echo "HNN training started (PID: $HNN_PID)"

echo ""
echo "=== Training launched! ==="
echo "DPF log: tail -f logs/dpf_2dof_training.log"
echo "HNN log: tail -f logs/hnn_2dof_training.log"
echo "Monitor: watch nvidia-smi"
echo ""
echo "Pipeline start time complete: $(date)"
