#!/bin/bash
# Launch ablation trajectory generation on 4 GPUs in parallel.
#
# GPU 0: original     — all Exp A + Exp B
# GPU 1: global_cond  — all Exp A + Exp B
# GPU 2: torque_concat — Exp A [training, sinusoidal] + Exp B
# GPU 3: torque_concat — Exp A [gp, zero]
#
# Usage:
#     bash scripts/launch_ablation_parallel.sh

set -e
cd "$(dirname "$0")/.."

CKPT_ORIGINAL="checkpoints/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt"
CKPT_GLOBAL_COND="checkpoints/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions_ablation-global_cond:epoch=2999_val_loss:val_loss=0.0008.ckpt"
CKPT_TORQUE_CONCAT="checkpoints/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions_ablation-torque_concat:epoch=2999_val_loss:val_loss=0.0010.ckpt"

BATCH_UNGUIDED=200
BATCH_GUIDED=200
OUTPUT_DIR="output_ablation/trajectories"
LOG_DIR="output_ablation"

mkdir -p "$LOG_DIR"

echo "Starting ablation trajectory generation on 4 GPUs..."
echo "Logs: $LOG_DIR/log_*.txt"
echo ""

# GPU 0: original — all Exp A + Exp B
python scripts/generate_ablation_trajectories.py \
    --model_name original \
    --checkpoint "$CKPT_ORIGINAL" \
    --device cuda:0 \
    --batch_size_unguided $BATCH_UNGUIDED \
    --batch_size_guided $BATCH_GUIDED \
    --output_dir "$OUTPUT_DIR" \
    > "$LOG_DIR/log_original.txt" 2>&1 &
PID_0=$!
echo "GPU 0: original (PID=$PID_0)"

# GPU 1: global_cond — all Exp A + Exp B
python scripts/generate_ablation_trajectories.py \
    --model_name global_cond \
    --checkpoint "$CKPT_GLOBAL_COND" \
    --device cuda:1 \
    --batch_size_unguided $BATCH_UNGUIDED \
    --batch_size_guided $BATCH_GUIDED \
    --output_dir "$OUTPUT_DIR" \
    > "$LOG_DIR/log_global_cond.txt" 2>&1 &
PID_1=$!
echo "GPU 1: global_cond (PID=$PID_1)"

# GPU 2: torque_concat — Exp A [training, sinusoidal] + Exp B
python scripts/generate_ablation_trajectories.py \
    --model_name torque_concat \
    --checkpoint "$CKPT_TORQUE_CONCAT" \
    --device cuda:2 \
    --batch_size_unguided $BATCH_UNGUIDED \
    --batch_size_guided $BATCH_GUIDED \
    --torque_conditions training,sinusoidal \
    --output_dir "$OUTPUT_DIR" \
    > "$LOG_DIR/log_torque_concat_gpu2.txt" 2>&1 &
PID_2=$!
echo "GPU 2: torque_concat [training,sinusoidal+ExpB] (PID=$PID_2)"

# GPU 3: torque_concat — Exp A [gp, zero] (no Exp B)
python scripts/generate_ablation_trajectories.py \
    --model_name torque_concat \
    --checkpoint "$CKPT_TORQUE_CONCAT" \
    --device cuda:3 \
    --batch_size_unguided $BATCH_UNGUIDED \
    --batch_size_guided $BATCH_GUIDED \
    --torque_conditions gp,zero \
    --no_exp_b \
    --output_dir "$OUTPUT_DIR" \
    > "$LOG_DIR/log_torque_concat_gpu3.txt" 2>&1 &
PID_3=$!
echo "GPU 3: torque_concat [gp,zero] (PID=$PID_3)"

echo ""
echo "All 4 processes launched. Monitor with:"
echo "  tail -f $LOG_DIR/log_original.txt"
echo "  tail -f $LOG_DIR/log_global_cond.txt"
echo "  tail -f $LOG_DIR/log_torque_concat_gpu2.txt"
echo "  tail -f $LOG_DIR/log_torque_concat_gpu3.txt"
echo ""
echo "Waiting for all processes to complete..."

FAIL=0
wait $PID_0 || { echo "GPU 0 (original) FAILED with exit code $?"; FAIL=1; }
wait $PID_1 || { echo "GPU 1 (global_cond) FAILED with exit code $?"; FAIL=1; }
wait $PID_2 || { echo "GPU 2 (torque_concat gpu2) FAILED with exit code $?"; FAIL=1; }
wait $PID_3 || { echo "GPU 3 (torque_concat gpu3) FAILED with exit code $?"; FAIL=1; }

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "All 4 processes completed successfully!"
else
    echo ""
    echo "Some processes failed. Check logs for details."
fi
