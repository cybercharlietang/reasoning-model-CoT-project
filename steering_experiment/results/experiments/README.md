# Steering Vectors Experiment Registry

## Experiment Naming Convention

```
exp{NNN}_{problem}_{vector}_{alphas}_{rollouts}
```

- `NNN`: Sequential experiment number (001, 002, ...)
- `problem`: Problem identifier (prob0, prob1, ... or multi)
- `vector`: Steering vector used (backtracking, uncertainty)
- `alphas`: Alpha values tested (alpha0_neg1 = 0.0 and -1.0)
- `rollouts`: Number of rollouts per alpha (n30 = 30 rollouts)

---

## Completed Experiments

### exp001_prob0_backtracking_alpha0_neg1_n30

| Field | Value |
|-------|-------|
| **Date** | 2025-12-30 |
| **Problem** | problem_2238 (Counting & Probability) |
| **Problem Index** | 0 (in selected_problems.json) |
| **Steering Vector** | backtracking |
| **Alpha Values** | 0.0 (baseline), -1.0 (suppression) |
| **Rollouts per Alpha** | 30 |
| **Positions Tested** | 60 (all chunks) |
| **Total Generations** | 3,600 |
| **Runtime** | 8h 31m |
| **Model** | DeepSeek-R1-Distill-Qwen-14B |

**Key Results:**
- CI_Accuracy ↔ Accuracy_Delta: r=0.287, p=0.026 (**significant**)
- CI_KL ↔ Steering_JS: r=0.124, p=0.346 (trend, not significant)
- Uncertainty management positions showed 3.2× higher KL divergence

**Files:**
- `config.json`: Experiment configuration
- `summary.json`: Generation statistics
- `metrics.json`: Computed metrics and correlations
- `rollouts/`: Per-position rollout data

---

## Planned Experiments

| ID | Problem | Vector | Alphas | Notes |
|----|---------|--------|--------|-------|
| exp002 | prob1 | backtracking | 0, -1 | Second problem, same setup |
| exp003 | prob0 | uncertainty_estimation | 0, -1 | Different steering vector |
| exp004 | prob0 | backtracking | 0, -1, -2 | Test stronger suppression |

**Run commands:**
```bash
# exp002: Second problem
python steering_experiment/scripts/1_generate_rollouts.py --problem_idx 1 --n_rollouts 30 --batch_size 12

# exp003: Different steering vector
python steering_experiment/scripts/1_generate_rollouts.py --problem_idx 0 --behavior uncertainty_estimation --n_rollouts 30 --batch_size 12

# Manual experiment ID (e.g., to redo exp002)
python steering_experiment/scripts/1_generate_rollouts.py --problem_idx 1 --experiment_id 2 --n_rollouts 30
```

---

## Analysis Scripts

```bash
# Compute metrics for an experiment
python steering_experiment/scripts/2_compute_metrics.py \
  --results_dir steering_experiment/results/experiments/exp001_prob0_backtracking_alpha0_neg1_n30

# Compute metrics for latest experiment
python steering_experiment/scripts/2_compute_metrics.py \
  --results_dir "$(ls -td steering_experiment/results/experiments/exp* | head -1)"

# Compare multiple experiments (TODO)
python steering_experiment/scripts/compare_experiments.py \
  --experiments exp001 exp002 exp003
```

## Quick Analysis

```python
# Load and analyze an experiment
import json
from pathlib import Path

exp_dir = Path("steering_experiment/results/experiments/exp001_prob0_backtracking_alpha0_neg1_n30")
with open(exp_dir / "metrics.json") as f:
    metrics = json.load(f)

# Filter target positions
targets = [m for m in metrics if m["is_uncertainty_management"]]
print(f"Target positions: {len(targets)}")
```

