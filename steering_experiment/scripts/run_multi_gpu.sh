#!/bin/bash
# =============================================================================
# Multi-GPU Parallel Experiment Runner
# =============================================================================
#
# Runs a steering experiment with TRUE parallelism across 8 GPUs.
# Each GPU loads its own model and processes a subset of positions.
#
# Usage:
#   ./run_multi_gpu.sh <problem_idx> <behavior> [n_gpus]
#
# Example:
#   ./run_multi_gpu.sh 0 backtracking 8
#
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

PROBLEM_IDX="${1:-0}"
BEHAVIOR="${2:-backtracking}"
N_GPUS="${3:-8}"

# Generation settings (per GPU)
N_ROLLOUTS=20
BATCH_SIZE=64                        # B200 with 183GB can handle larger batches with Flash Attention 2
MAX_NEW_TOKENS=8192
ALPHA_VALUES="0.0 -1.0 1.0"

# =============================================================================
# SETUP
# =============================================================================

echo "============================================================"
echo "MULTI-GPU PARALLEL STEERING EXPERIMENT"
echo "============================================================"
echo ""
echo "Problem index:    $PROBLEM_IDX"
echo "Steering vector:  $BEHAVIOR"
echo "GPUs:             $N_GPUS"
echo "Rollouts:         $N_ROLLOUTS per position"
echo "Batch size:       $BATCH_SIZE per GPU"
echo "Max tokens:       $MAX_NEW_TOKENS"
echo "Alpha values:     $ALPHA_VALUES"
echo ""

# Detect GPUs
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

cd "$(dirname "$0")/../.."  # Go to project root

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment: .venv"
fi

# Create directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_ID="prob${PROBLEM_IDX}_${BEHAVIOR}_${TIMESTAMP}"
OUTPUT_DIR="steering_experiment/results/experiments/${EXPERIMENT_ID}"
LOG_DIR="logs/${EXPERIMENT_ID}"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo "Log directory:    $LOG_DIR"
echo ""

# =============================================================================
# LAUNCH PARALLEL WORKERS
# =============================================================================

echo "Launching $N_GPUS parallel workers..."
echo ""

PIDS=()

for GPU_ID in $(seq 0 $((N_GPUS - 1))); do
    echo "  GPU $GPU_ID: Starting worker..."
    
    nohup env CUDA_VISIBLE_DEVICES=$GPU_ID python steering_experiment/scripts/1_generate_rollouts.py \
        --problem_idx $PROBLEM_IDX \
        --behavior $BEHAVIOR \
        --alpha $ALPHA_VALUES \
        --n_rollouts $N_ROLLOUTS \
        --batch_size $BATCH_SIZE \
        --max_new_tokens $MAX_NEW_TOKENS \
        --skip_verification \
        --gpu_id $GPU_ID \
        --n_gpus $N_GPUS \
        --experiment_id "${EXPERIMENT_ID}" \
        > "$LOG_DIR/gpu_${GPU_ID}.log" 2>&1 &
    
    PIDS+=($!)
done

echo ""
echo "All workers launched. PIDs: ${PIDS[*]}"
echo ""
echo "Waiting for completion... (monitor with: tail -f $LOG_DIR/gpu_*.log)"
echo ""

# =============================================================================
# WAIT FOR COMPLETION
# =============================================================================

FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait $PID; then
        echo "  GPU $i: Complete ✓"
    else
        echo "  GPU $i: FAILED ✗ (check $LOG_DIR/gpu_${i}.log)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""

if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED workers failed!"
    echo "Check logs in $LOG_DIR/"
    exit 1
fi

# =============================================================================
# MERGE RESULTS
# =============================================================================

echo "Merging results from all GPUs..."
python steering_experiment/scripts/merge_gpu_results.py \
    --experiment_dir "$OUTPUT_DIR" \
    --n_gpus $N_GPUS

echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Logs saved to:    $LOG_DIR"
echo ""
echo "Next steps:"
echo "  1. Compute metrics:"
echo "     python steering_experiment/scripts/2_compute_metrics.py --results_dir $OUTPUT_DIR"
echo ""

