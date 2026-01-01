#!/bin/bash
# Run uncertainty_estimation experiments on problems 0, 2, and 3
# Problem 0: Add 20 rollouts (already has 20)
# Problems 2, 3: New runs with 40 rollouts each

set -e
cd /root/reasoning-model-CoT-project

# Activate virtual environment
source .venv/bin/activate

VECTOR="uncertainty_estimation"
N_GPUS=8
BATCH_SIZE=64

echo "============================================================"
echo "UNCERTAINTY ESTIMATION EXPERIMENTS"
echo "============================================================"
echo "Start time: $(date)"
echo ""

# Problem 0: Add 20 more rollouts
echo "============================================================"
echo "Problem 0: Adding 20 rollouts (existing has 20)"
echo "============================================================"
./steering_experiment/scripts/run_multi_gpu.sh 0 $VECTOR $N_GPUS 20
echo "Problem 0 completed at $(date)"
echo ""

# Problem 2: New run with 40 rollouts
echo "============================================================"
echo "Problem 2: New run with 40 rollouts"
echo "============================================================"
./steering_experiment/scripts/run_multi_gpu.sh 2 $VECTOR $N_GPUS 40
echo "Problem 2 completed at $(date)"
echo ""

# Problem 3: New run with 40 rollouts
echo "============================================================"
echo "Problem 3: New run with 40 rollouts"
echo "============================================================"
./steering_experiment/scripts/run_multi_gpu.sh 3 $VECTOR $N_GPUS 40
echo "Problem 3 completed at $(date)"
echo ""

echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo "End time: $(date)"

# Run metrics computation for all
echo ""
echo "Computing metrics..."
for exp_dir in steering_experiment/results/experiments/prob*_uncertainty_estimation_*; do
    if [ -d "$exp_dir" ]; then
        echo "Processing $exp_dir..."
        python steering_experiment/scripts/2_compute_metrics.py --results_dir "$exp_dir"
    fi
done

echo "Done!"

