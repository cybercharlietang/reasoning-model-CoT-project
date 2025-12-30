#!/bin/bash
# Run a single experiment across multiple GPUs with maximum parallelism
#
# Usage:
#   ./run_parallel_experiments.sh <problem_idx> <behavior>
#
# Example:
#   ./run_parallel_experiments.sh 0 backtracking
#   ./run_parallel_experiments.sh 1 uncertainty_estimation
#
# This script runs ONE problem with ONE steering vector, using all available GPUs
# to parallelize batch generation.

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

PROBLEM_IDX="${1:-0}"
BEHAVIOR="${2:-backtracking}"

# Generation settings (optimized for B200)
N_ROLLOUTS=50
BATCH_SIZE=48                        # B200 has 192GB VRAM - can handle large batches
MAX_NEW_TOKENS=8192
ALPHA_VALUES="0.0 -1.0 1.0"         # Baseline, suppression, amplification

# =============================================================================
# SETUP
# =============================================================================

echo "============================================================"
echo "STEERING EXPERIMENT"
echo "============================================================"
echo ""
echo "Problem index: $PROBLEM_IDX"
echo "Steering vector: $BEHAVIOR"
echo "Rollouts: $N_ROLLOUTS"
echo "Batch size: $BATCH_SIZE"
echo "Max tokens: $MAX_NEW_TOKENS"
echo "Alpha values: $ALPHA_VALUES"
echo ""

# Detect GPUs
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# =============================================================================
# RUN EXPERIMENT
# =============================================================================

cd "$(dirname "$0")/../.."  # Go to project root

# Create logs directory if needed
mkdir -p logs

# Run the experiment
# Note: The model will automatically use all available GPUs with device_map="auto"
python steering_experiment/scripts/1_generate_rollouts.py \
    --problem_idx $PROBLEM_IDX \
    --behavior $BEHAVIOR \
    --alpha $ALPHA_VALUES \
    --n_rollouts $N_ROLLOUTS \
    --batch_size $BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --skip_verification \
    2>&1 | tee "logs/exp_prob${PROBLEM_IDX}_${BEHAVIOR}.log"

echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
echo ""
echo "Log saved to: logs/exp_prob${PROBLEM_IDX}_${BEHAVIOR}.log"
echo ""

# Show results directory
echo "Results:"
ls -la steering_experiment/results/experiments/ | grep "prob${PROBLEM_IDX}" | tail -1

echo ""
echo "Next steps:"
echo "  1. Compute metrics:"
echo "     python steering_experiment/scripts/2_compute_metrics.py --results_dir <exp_dir>"
echo "  2. Run another experiment:"
echo "     ./steering_experiment/scripts/run_parallel_experiments.sh <problem_idx> <behavior>"

