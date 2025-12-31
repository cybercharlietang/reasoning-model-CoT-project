#!/bin/bash
# Overnight run: Backtracking vector on Problems 1-5
# 40 rollouts each, 8 GPUs
# Estimated runtime: 10-12 hours

set -e

cd /root/reasoning-model-CoT-project
source .venv/bin/activate

PROBLEMS=(1 2 3 4 5)
VECTOR="backtracking"
N_ROLLOUTS=40
N_GPUS=8

echo "============================================================"
echo "OVERNIGHT RUN: BACKTRACKING VECTOR"
echo "============================================================"
echo "Problems: ${PROBLEMS[*]}"
echo "Vector: $VECTOR"
echo "Rollouts: $N_ROLLOUTS"
echo "GPUs: $N_GPUS"
echo "Start time: $(date)"
echo "============================================================"
echo ""

for PROB in "${PROBLEMS[@]}"; do
    echo ""
    echo "============================================================"
    echo "STARTING PROBLEM $PROB at $(date)"
    echo "============================================================"
    
    # Run multi-GPU experiment
    ./steering_experiment/scripts/run_multi_gpu.sh $PROB $VECTOR $N_GPUS $N_ROLLOUTS
    
    # Wait for completion
    while [ $(ps aux | grep "1_generate_rollouts" | grep -v grep | wc -l) -gt 0 ]; do
        sleep 60
    done
    
    echo "Problem $PROB completed at $(date)"
    
    # Merge results
    LATEST_DIR=$(ls -td steering_experiment/results/experiments/prob${PROB}_${VECTOR}_* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        echo "Merging results from $LATEST_DIR..."
        python steering_experiment/scripts/merge_gpu_results.py --experiment_dir "$LATEST_DIR"
        
        echo "Computing metrics..."
        python steering_experiment/scripts/2_compute_metrics.py --results_dir "$LATEST_DIR"
    fi
    
    echo ""
done

echo ""
echo "============================================================"
echo "OVERNIGHT RUN COMPLETE"
echo "End time: $(date)"
echo "============================================================"

# Summary
echo ""
echo "=== EXPERIMENT SUMMARY ==="
for PROB in "${PROBLEMS[@]}"; do
    LATEST_DIR=$(ls -td steering_experiment/results/experiments/prob${PROB}_${VECTOR}_* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ] && [ -f "$LATEST_DIR/metrics.json" ]; then
        echo "✓ Problem $PROB: $LATEST_DIR"
    else
        echo "✗ Problem $PROB: INCOMPLETE"
    fi
done

