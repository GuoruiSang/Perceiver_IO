#!/bin/bash
# Launch HNN and DPF training after dataset generation completes.
# Waits for the training data file to appear, then starts both in parallel with nohup.

set -e

PROJECT=/home/gsang/Projects/Perceiver_IO
TRAIN_FILE="$PROJECT/data/traj_80000-steps_4000.h5"
VAL_FILE="$PROJECT/data/traj_2000-steps_4000.h5"
LOG_DIR="$PROJECT/logs"

mkdir -p "$LOG_DIR"

echo "$(date): Waiting for dataset generation to complete..."
echo "  Expecting: $TRAIN_FILE"
echo "  Expecting: $VAL_FILE"

# Poll until both files exist and the datagen tmux session is gone (generation finished)
while true; do
    # Check if the datagen tmux session is still running
    if tmux has-session -t datagen 2>/dev/null; then
        echo "$(date): datagen session still running, waiting..."
        sleep 60
    else
        # Session ended - check if files exist
        if [[ -f "$TRAIN_FILE" && -f "$VAL_FILE" ]]; then
            echo "$(date): Generation complete! Both files found."
            break
        else
            echo "$(date): ERROR - datagen session ended but files not found!"
            echo "  Train file exists: $(test -f "$TRAIN_FILE" && echo yes || echo no)"
            echo "  Val file exists: $(test -f "$VAL_FILE" && echo yes || echo no)"
            exit 1
        fi
    fi
done

# Print file sizes
echo ""
echo "Dataset files:"
ls -lh "$TRAIN_FILE"
ls -lh "$VAL_FILE"
echo ""

# Kill any existing DPF training processes to avoid conflicts
echo "$(date): Checking for existing DPF training processes..."
DPF_PIDS=$(pgrep -f "trajectory_dpf.py.*--mode train" 2>/dev/null || true)
if [[ -n "$DPF_PIDS" ]]; then
    echo "  Killing existing DPF processes: $DPF_PIDS"
    kill $DPF_PIDS 2>/dev/null || true
    sleep 5
    # Force kill if still running
    kill -9 $DPF_PIDS 2>/dev/null || true
    echo "  Done."
else
    echo "  No existing DPF processes found."
fi

# Kill any existing HNN training processes to avoid conflicts
echo "$(date): Checking for existing HNN training processes..."
HNN_PIDS=$(pgrep -f "src/models/HNN.py" 2>/dev/null || true)
if [[ -n "$HNN_PIDS" ]]; then
    echo "  Killing existing HNN processes: $HNN_PIDS"
    kill $HNN_PIDS 2>/dev/null || true
    sleep 5
    kill -9 $HNN_PIDS 2>/dev/null || true
    echo "  Done."
else
    echo "  No existing HNN processes found."
fi

# Launch HNN training on GPU 4
echo "$(date): Starting HNN training on GPU 4..."
cd "$PROJECT"
nohup python src/models/HNN.py \
    > "$LOG_DIR/hnn_training.log" 2>&1 &
HNN_PID=$!
echo "  HNN PID: $HNN_PID (log: $LOG_DIR/hnn_training.log)"

# Launch DPF training on GPUs 0,1,2,3 (fresh, no checkpoint resume)
echo "$(date): Starting DPF training on GPUs 0,1,2,3..."
nohup python src/models/trajectory_dpf.py \
    --mode train \
    --h5_path "$TRAIN_FILE" \
    --devices 0 1 2 3 \
    --resume_from_checkpoint "" \
    > "$LOG_DIR/dpf_training.log" 2>&1 &
DPF_PID=$!
echo "  DPF PID: $DPF_PID (log: $LOG_DIR/dpf_training.log)"

echo ""
echo "$(date): Both trainings launched!"
echo "  HNN (GPU 4):       tail -f $LOG_DIR/hnn_training.log"
echo "  DPF (GPU 0,1,2,3): tail -f $LOG_DIR/dpf_training.log"
echo ""
echo "Monitor GPU usage: watch nvidia-smi"

# Both trainings are now running independently via nohup.
# They will survive even if this script or its parent session exits.
echo "$(date): Launcher done. Trainings running in background via nohup."
