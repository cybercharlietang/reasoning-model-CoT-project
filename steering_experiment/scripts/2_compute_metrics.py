#!/usr/bin/env python3
"""
Script 2: Compute Metrics from Generated Rollouts

This script loads the generated rollouts and computes:
1. Answer distributions for each condition
2. KL divergence between baseline and steered distributions
3. JS divergence (symmetric version)
4. Accuracy delta

Results are saved for statistical analysis.

Usage:
    python scripts/2_compute_metrics.py --results_dir path/to/results
"""

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class PositionMetrics:
    """Metrics for a single position."""
    position_idx: int
    function_tags: List[str]
    chunk_text: str
    
    # Existing CI scores from Thought Anchors
    ci_kl: float
    ci_accuracy: float
    
    # Computed metrics
    metrics_by_alpha: Dict[float, "AlphaMetrics"]
    
    # Is this a target position?
    is_uncertainty_management: bool


@dataclass  
class AlphaMetrics:
    """Metrics for a specific alpha value compared to baseline."""
    alpha: float
    
    # Distributions
    baseline_distribution: Dict[str, float]  # answer -> probability
    steered_distribution: Dict[str, float]
    
    # Divergences
    kl_divergence: float  # KL(steered || baseline)
    js_divergence: float  # JS(baseline, steered)
    
    # Accuracy
    baseline_accuracy: float
    steered_accuracy: float
    accuracy_delta: float
    
    # Sample sizes
    n_baseline: int
    n_steered: int
    
    # Answer extraction rate
    baseline_extraction_rate: float
    steered_extraction_rate: float


def compute_distribution(answers: List[Optional[str]]) -> Dict[str, float]:
    """
    Compute probability distribution over answers.
    
    Args:
        answers: List of extracted answers (None for extraction failures)
        
    Returns:
        Dictionary mapping answers to probabilities
    """
    # Filter out None values for distribution
    valid_answers = [a for a in answers if a is not None]
    
    if not valid_answers:
        return {}
    
    counter = Counter(valid_answers)
    total = len(valid_answers)
    
    return {answer: count / total for answer, count in counter.items()}


def kl_divergence(p: Dict[str, float], q: Dict[str, float], epsilon: float = 1e-10) -> float:
    """
    Compute KL(P || Q) = sum_x P(x) * log(P(x) / Q(x))
    
    Direction: KL(steered || baseline) measures "how surprised baseline would be 
    by steered samples" - i.e., how much steering changed the distribution.
    
    Uses smoothing to handle missing keys.
    
    Args:
        p: Target distribution (steered)
        q: Reference distribution (baseline)
        epsilon: Smoothing factor for zero probabilities
        
    Returns:
        KL divergence value (non-negative, can be infinite)
    """
    if not p or not q:
        return float('inf')
    
    # Get all keys
    all_keys = set(p.keys()) | set(q.keys())
    
    # Compute KL divergence with smoothing
    kl = 0.0
    for key in all_keys:
        p_x = p.get(key, epsilon)
        q_x = q.get(key, epsilon)
        
        if p_x > epsilon:
            kl += p_x * math.log(p_x / q_x)
    
    return max(0.0, kl)  # Ensure non-negative due to numerical issues


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Compute Jensen-Shannon divergence: JS(P, Q) = (KL(P||M) + KL(Q||M)) / 2
    where M = (P + Q) / 2
    
    Args:
        p: First distribution
        q: Second distribution
        
    Returns:
        JS divergence value (between 0 and log(2))
    """
    if not p or not q:
        return 1.0  # Maximum divergence
    
    # Get all keys
    all_keys = set(p.keys()) | set(q.keys())
    
    # Compute mixture distribution M
    m = {}
    for key in all_keys:
        m[key] = (p.get(key, 0.0) + q.get(key, 0.0)) / 2
    
    # JS = (KL(P||M) + KL(Q||M)) / 2
    kl_pm = kl_divergence(p, m)
    kl_qm = kl_divergence(q, m)
    
    return (kl_pm + kl_qm) / 2


def compute_accuracy(answers: List[Optional[str]], correctness: List[bool]) -> float:
    """
    Compute accuracy from correctness flags.
    
    Args:
        answers: List of extracted answers
        correctness: List of whether each answer is correct
        
    Returns:
        Accuracy (fraction of correct answers)
    """
    if not correctness:
        return 0.0
    
    # Only count generations where answer was extracted
    valid_pairs = [(a, c) for a, c in zip(answers, correctness) if a is not None]
    
    if not valid_pairs:
        return 0.0
    
    return sum(c for _, c in valid_pairs) / len(valid_pairs)


def load_rollouts(results_dir: Path) -> List[dict]:
    """
    Load rollout data from results directory.
    
    The rollout structure is:
        rollouts/
        ├── position_0/
        │   ├── metadata.json
        │   ├── alpha_0p0.json
        │   └── alpha_neg1p0.json
        └── position_1/
            └── ...
    """
    rollouts_dir = results_dir / "rollouts"
    
    if not rollouts_dir.exists():
        raise FileNotFoundError(f"Rollouts directory not found: {rollouts_dir}")
    
    rollouts = []
    for pos_dir in sorted(rollouts_dir.glob("position_*")):
        if not pos_dir.is_dir():
            continue
        
        # Load metadata
        metadata_file = pos_dir / "metadata.json"
        if not metadata_file.exists():
            continue
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Load results for each alpha
        results_by_alpha = {}
        for alpha_file in pos_dir.glob("alpha_*.json"):
            # Parse alpha from filename: alpha_0p0 -> "0.0", alpha_neg1p0 -> "-1.0"
            alpha_str = alpha_file.stem.replace("alpha_", "")
            alpha_str = alpha_str.replace("neg", "-").replace("p", ".")
            
            with open(alpha_file, "r") as f:
                results_by_alpha[alpha_str] = json.load(f)
        
        # Combine into single dict
        rollout_data = {
            "position_idx": metadata["position_idx"],
            "function_tags": metadata["function_tags"],
            "chunk_text": metadata["chunk_text"],
            "ci_kl": metadata.get("ci_kl", 0.0),
            "ci_accuracy": metadata.get("ci_accuracy", 0.0),
            "steer_range": metadata.get("steer_range"),
            "results_by_alpha": results_by_alpha,
        }
        rollouts.append(rollout_data)
    
    if not rollouts:
        raise ValueError(f"No rollout directories found in {rollouts_dir}")
    
    return rollouts


def compute_metrics_for_position(rollout_data: dict) -> PositionMetrics:
    """
    Compute all metrics for a single position.
    
    Args:
        rollout_data: Dictionary containing rollout results for one position
        
    Returns:
        PositionMetrics dataclass with computed metrics
    """
    position_idx = rollout_data["position_idx"]
    function_tags = rollout_data["function_tags"]
    chunk_text = rollout_data.get("chunk_text", "")
    ci_kl = rollout_data.get("ci_kl", 0.0)
    ci_accuracy = rollout_data.get("ci_accuracy", 0.0)
    
    is_uncertainty = "uncertainty_management" in function_tags
    
    # Get alpha values
    results_by_alpha = rollout_data["results_by_alpha"]
    alpha_values = [float(a) for a in results_by_alpha.keys()]
    
    # Find baseline (alpha=0)
    baseline_alpha = "0" if "0" in results_by_alpha else "0.0"
    if baseline_alpha not in results_by_alpha:
        raise ValueError(f"Position {position_idx}: No baseline (alpha=0) results found")
    
    baseline_results = results_by_alpha[baseline_alpha]
    baseline_answers = [r["answer"] for r in baseline_results]
    baseline_correctness = [r["is_correct"] for r in baseline_results]
    baseline_dist = compute_distribution(baseline_answers)
    baseline_acc = compute_accuracy(baseline_answers, baseline_correctness)
    baseline_extraction = sum(1 for a in baseline_answers if a is not None) / len(baseline_answers)
    
    # Compute metrics for each non-baseline alpha
    metrics_by_alpha = {}
    
    for alpha_str, results in results_by_alpha.items():
        alpha = float(alpha_str)
        
        if alpha == 0.0:
            continue
        
        steered_answers = [r["answer"] for r in results]
        steered_correctness = [r["is_correct"] for r in results]
        steered_dist = compute_distribution(steered_answers)
        steered_acc = compute_accuracy(steered_answers, steered_correctness)
        steered_extraction = sum(1 for a in steered_answers if a is not None) / len(steered_answers)
        
        # Compute divergences
        kl = kl_divergence(steered_dist, baseline_dist)
        js = js_divergence(baseline_dist, steered_dist)
        
        metrics_by_alpha[alpha] = AlphaMetrics(
            alpha=alpha,
            baseline_distribution=baseline_dist,
            steered_distribution=steered_dist,
            kl_divergence=kl,
            js_divergence=js,
            baseline_accuracy=baseline_acc,
            steered_accuracy=steered_acc,
            accuracy_delta=steered_acc - baseline_acc,
            n_baseline=len(baseline_answers),
            n_steered=len(steered_answers),
            baseline_extraction_rate=baseline_extraction,
            steered_extraction_rate=steered_extraction,
        )
    
    return PositionMetrics(
        position_idx=position_idx,
        function_tags=function_tags,
        chunk_text=chunk_text,
        ci_kl=ci_kl,
        ci_accuracy=ci_accuracy,
        metrics_by_alpha=metrics_by_alpha,
        is_uncertainty_management=is_uncertainty,
    )


def save_metrics(
    metrics: List[PositionMetrics],
    output_dir: Path,
) -> None:
    """Save computed metrics to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    metrics_data = []
    for m in metrics:
        m_dict = {
            "position_idx": m.position_idx,
            "function_tags": m.function_tags,
            "chunk_text": m.chunk_text,
            "ci_kl": m.ci_kl,
            "ci_accuracy": m.ci_accuracy,
            "is_uncertainty_management": m.is_uncertainty_management,
            "alpha_metrics": {},
        }
        
        for alpha, am in m.metrics_by_alpha.items():
            m_dict["alpha_metrics"][str(alpha)] = {
                "alpha": am.alpha,
                "kl_divergence": am.kl_divergence if not math.isinf(am.kl_divergence) else "inf",
                "js_divergence": am.js_divergence,
                "baseline_accuracy": am.baseline_accuracy,
                "steered_accuracy": am.steered_accuracy,
                "accuracy_delta": am.accuracy_delta,
                "n_baseline": am.n_baseline,
                "n_steered": am.n_steered,
                "baseline_extraction_rate": am.baseline_extraction_rate,
                "steered_extraction_rate": am.steered_extraction_rate,
                "baseline_distribution": am.baseline_distribution,
                "steered_distribution": am.steered_distribution,
            }
        
        metrics_data.append(m_dict)
    
    # Save
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"Saved metrics to {metrics_file}")


def print_summary(metrics: List[PositionMetrics]) -> None:
    """Print summary statistics."""
    print()
    print("=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print()
    
    # Group by uncertainty vs non-uncertainty
    uncertainty_positions = [m for m in metrics if m.is_uncertainty_management]
    other_positions = [m for m in metrics if not m.is_uncertainty_management]
    
    print(f"Total positions: {len(metrics)}")
    print(f"  Uncertainty management: {len(uncertainty_positions)}")
    print(f"  Other: {len(other_positions)}")
    print()
    
    # Get first alpha (assuming all positions have same alpha values)
    if not metrics or not metrics[0].metrics_by_alpha:
        print("No metrics computed.")
        return
    
    alpha = list(metrics[0].metrics_by_alpha.keys())[0]
    print(f"Analysis for alpha={alpha}:")
    print()
    
    # KL divergence stats
    all_kl = [m.metrics_by_alpha[alpha].kl_divergence for m in metrics 
              if not math.isinf(m.metrics_by_alpha[alpha].kl_divergence)]
    unc_kl = [m.metrics_by_alpha[alpha].kl_divergence for m in uncertainty_positions
              if not math.isinf(m.metrics_by_alpha[alpha].kl_divergence)]
    other_kl = [m.metrics_by_alpha[alpha].kl_divergence for m in other_positions
                if not math.isinf(m.metrics_by_alpha[alpha].kl_divergence)]
    
    print("KL Divergence (steered || baseline):")
    if all_kl:
        print(f"  All positions: mean={np.mean(all_kl):.4f}, std={np.std(all_kl):.4f}")
    if unc_kl:
        print(f"  Uncertainty mgmt: mean={np.mean(unc_kl):.4f}, std={np.std(unc_kl):.4f}")
    if other_kl:
        print(f"  Other: mean={np.mean(other_kl):.4f}, std={np.std(other_kl):.4f}")
    print()
    
    # JS divergence stats
    all_js = [m.metrics_by_alpha[alpha].js_divergence for m in metrics]
    unc_js = [m.metrics_by_alpha[alpha].js_divergence for m in uncertainty_positions]
    other_js = [m.metrics_by_alpha[alpha].js_divergence for m in other_positions]
    
    print("JS Divergence:")
    if all_js:
        print(f"  All positions: mean={np.mean(all_js):.4f}, std={np.std(all_js):.4f}")
    if unc_js:
        print(f"  Uncertainty mgmt: mean={np.mean(unc_js):.4f}, std={np.std(unc_js):.4f}")
    if other_js:
        print(f"  Other: mean={np.mean(other_js):.4f}, std={np.std(other_js):.4f}")
    print()
    
    # Accuracy delta stats
    all_acc = [m.metrics_by_alpha[alpha].accuracy_delta for m in metrics]
    unc_acc = [m.metrics_by_alpha[alpha].accuracy_delta for m in uncertainty_positions]
    other_acc = [m.metrics_by_alpha[alpha].accuracy_delta for m in other_positions]
    
    print("Accuracy Delta:")
    if all_acc:
        print(f"  All positions: mean={np.mean(all_acc):.4f}, std={np.std(all_acc):.4f}")
    if unc_acc:
        print(f"  Uncertainty mgmt: mean={np.mean(unc_acc):.4f}, std={np.std(unc_acc):.4f}")
    if other_acc:
        print(f"  Other: mean={np.mean(other_acc):.4f}, std={np.std(other_acc):.4f}")
    print()
    
    # CI correlation analysis
    # Note: We use JS divergence for correlation (bounded [0, log(2)], no infinities)
    # KL direction: KL(steered || baseline) - measures how steered differs from baseline
    print("-" * 40)
    print("Correlation Analysis (CI vs Steering Effect):")
    print("-" * 40)
    
    # Filter both lists together to handle any infinities
    valid_kl_pairs = [
        (m.ci_kl, m.metrics_by_alpha[alpha].kl_divergence) 
        for m in metrics 
        if not math.isinf(m.metrics_by_alpha[alpha].kl_divergence)
    ]
    
    if len(valid_kl_pairs) > 2:
        ci_vals, kl_vals = zip(*valid_kl_pairs)
        corr, p_value = stats.pearsonr(ci_vals, kl_vals)
        print(f"  CI_KL vs Steering_KL: r={corr:.4f}, p={p_value:.4f} (n={len(valid_kl_pairs)})")
    else:
        print(f"  CI_KL vs Steering_KL: insufficient data (n={len(valid_kl_pairs)})")
    
    # JS divergence (preferred - always finite)
    valid_js_pairs = [(m.ci_kl, m.metrics_by_alpha[alpha].js_divergence) for m in metrics]
    if len(valid_js_pairs) > 2:
        ci_vals, js_vals = zip(*valid_js_pairs)
        corr, p_value = stats.pearsonr(ci_vals, js_vals)
        print(f"  CI_KL vs Steering_JS: r={corr:.4f}, p={p_value:.4f} (n={len(valid_js_pairs)})")
        
        # This is the main hypothesis test
        if corr > 0 and p_value < 0.05:
            print(f"  → SIGNIFICANT positive correlation (supports hypothesis)")
        elif corr > 0:
            print(f"  → Positive trend but not significant (need more data)")
        else:
            print(f"  → No positive correlation found")
    
    # Accuracy correlation
    valid_acc_pairs = [(m.ci_accuracy, m.metrics_by_alpha[alpha].accuracy_delta) for m in metrics]
    if len(valid_acc_pairs) > 2:
        ci_acc_vals, acc_delta_vals = zip(*valid_acc_pairs)
        corr, p_value = stats.pearsonr(ci_acc_vals, acc_delta_vals)
        print(f"  CI_Accuracy vs Accuracy_Delta: r={corr:.4f}, p={p_value:.4f} (n={len(valid_acc_pairs)})")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute metrics from generated rollouts"
    )
    
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Directory containing rollout results"
    )
    
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for metrics (default: same as results_dir)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    
    print("=" * 60)
    print("STEERING VECTORS EXPERIMENT - METRICS COMPUTATION")
    print("=" * 60)
    print()
    
    # Load rollouts
    print(f"Loading rollouts from {results_dir}...")
    rollouts = load_rollouts(results_dir)
    print(f"Loaded {len(rollouts)} positions")
    print()
    
    # Compute metrics for each position
    print("Computing metrics...")
    metrics = []
    for rollout_data in rollouts:
        position_metrics = compute_metrics_for_position(rollout_data)
        metrics.append(position_metrics)
    
    # Save metrics
    save_metrics(metrics, output_dir)
    
    # Print summary
    print_summary(metrics)
    
    print()
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())

