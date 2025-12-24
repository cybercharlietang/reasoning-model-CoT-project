#!/usr/bin/env python3
"""
Visualize Thought Anchors Analysis Results

This script loads pre-computed importance metrics from chunks_labeled.json
and creates visualizations showing which sentence types are "thought anchors"
(disproportionately important for final answer accuracy).

Usage:
    python visualize_analysis.py --problem problem_2238
    python visualize_analysis.py --problem problem_2238 --output_dir results/
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# Optional: matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualizations will be skipped.")


def load_data(problem_dir: Path) -> list:
    """Load pre-computed labeled chunks with importance metrics."""
    chunks_file = problem_dir / "chunks_labeled.json"
    if not chunks_file.exists():
        raise FileNotFoundError(f"No chunks_labeled.json found in {problem_dir}")
    
    with open(chunks_file) as f:
        return json.load(f)


def print_methodology():
    """Print explanation of what the metrics mean."""
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ WHAT analyze_rollouts.py COMPUTES:                                  │
├─────────────────────────────────────────────────────────────────────┤
│ For each sentence in the Chain-of-Thought:                          │
│                                                                     │
│ 1. RESAMPLING IMPORTANCE (accuracy):                                │
│    → Remove sentence, generate 100 continuations                    │
│    → Measure: How much does accuracy drop?                          │
│    → Higher = more important (thought anchor!)                      │
│                                                                     │
│ 2. COUNTERFACTUAL IMPORTANCE (KL divergence):                       │
│    → Compare "similar" vs "dissimilar" resampled sentences          │
│    → Measure: How different are the answer distributions?           │
│    → Higher KL = removing sentence changes answer distribution      │
│                                                                     │
│ 3. TRAJECTORY DIVERGENCE:                                           │
│    → When we resample, how often is the new sentence DIFFERENT?     │
│    → High = sentence is a "branching point" (thought anchor!)       │
│                                                                     │
│ 4. OVERDETERMINEDNESS:                                              │
│    → How often do different rollouts produce the SAME continuation? │
│    → High = reasoning is "locked in" at this point                  │
└─────────────────────────────────────────────────────────────────────┘
""")


def compute_tag_metrics(chunks: list) -> dict:
    """Group metrics by function tag."""
    tag_metrics = defaultdict(lambda: {
        'count': 0,
        'resampling_acc': [],
        'counterfactual_kl': [],
        'trajectory_div': [],
        'overdeterminedness': []
    })
    
    for c in chunks:
        tags = c.get('function_tags', ['unknown'])
        for tag in tags:
            tag_metrics[tag]['count'] += 1
            tag_metrics[tag]['resampling_acc'].append(
                c.get('resampling_importance_accuracy', 0))
            tag_metrics[tag]['counterfactual_kl'].append(
                c.get('counterfactual_importance_kl', 0))
            tag_metrics[tag]['trajectory_div'].append(
                c.get('different_trajectories_fraction', 0))
            tag_metrics[tag]['overdeterminedness'].append(
                c.get('overdeterminedness', 0))
    
    return dict(tag_metrics)


def print_results_table(tag_metrics: dict):
    """Print importance metrics grouped by sentence type."""
    print("\n" + "=" * 75)
    print("IMPORTANCE BY SENTENCE TYPE (Paper's Key Finding)")
    print("=" * 75)
    print(f"{'Tag':<28} | {'N':>4} | {'Resamp%':>8} | {'CF KL':>8} | {'TrajDiv%':>8}")
    print("-" * 75)
    
    # Order tags by expected importance
    tag_order = [
        'plan_generation', 
        'uncertainty_management', 
        'self_checking',
        'fact_retrieval', 
        'active_computation', 
        'result_consolidation',
        'problem_setup', 
        'final_answer_emission',
        'unknown'
    ]
    
    for tag in tag_order:
        if tag in tag_metrics:
            m = tag_metrics[tag]
            avg_acc = np.mean(m['resampling_acc']) * 100
            avg_kl = np.mean(m['counterfactual_kl'])
            avg_traj = np.mean(m['trajectory_div']) * 100
            
            # Mark thought anchors
            is_anchor = avg_acc > 2 or avg_traj > 5
            marker = "⚓" if is_anchor else " "
            
            print(f"{tag:<28} | {m['count']:>4} | {avg_acc:>7.2f}% | {avg_kl:>8.4f} | {avg_traj:>7.2f}% {marker}")


def print_top_anchors(chunks: list, n: int = 5):
    """Print the top N most important sentences."""
    print("\n" + "=" * 75)
    print(f"TOP {n} MOST IMPORTANT SENTENCES (Thought Anchors)")
    print("=" * 75)
    
    sorted_chunks = sorted(
        chunks, 
        key=lambda x: x.get('resampling_importance_accuracy', 0), 
        reverse=True
    )
    
    for i, c in enumerate(sorted_chunks[:n]):
        print(f"\n[{i+1}] Chunk {c['chunk_idx']} - Tags: {c['function_tags']}")
        print(f"    Resampling Importance: {c.get('resampling_importance_accuracy', 0)*100:.1f}%")
        print(f"    Counterfactual KL: {c.get('counterfactual_importance_kl', 0):.4f}")
        print(f"    Trajectory Divergence: {c.get('different_trajectories_fraction', 0)*100:.1f}%")
        text = c.get('chunk', '')[:100].replace('\n', ' ')
        print(f"    Text: {text}...")


def create_visualization(chunks: list, tag_metrics: dict, output_path: Path):
    """Create matplotlib visualization of the analysis."""
    if not HAS_MATPLOTLIB:
        print("Skipping visualization (matplotlib not available)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Color scheme for sentence types
    anchor_tags = ['plan_generation', 'uncertainty_management', 'self_checking']
    
    # Plot 1: Importance by position
    ax1 = axes[0, 0]
    positions = [c['chunk_idx'] for c in chunks]
    importance = [c.get('resampling_importance_accuracy', 0) * 100 for c in chunks]
    colors = ['red' if c.get('function_tags', [''])[0] in anchor_tags else 'steelblue' 
              for c in chunks]
    ax1.bar(positions, importance, color=colors, alpha=0.7)
    ax1.set_xlabel('Sentence Position in CoT')
    ax1.set_ylabel('Resampling Importance (%)')
    ax1.set_title('Importance by Position\n(Red = Planning/Uncertainty/Checking)')
    ax1.axhline(y=np.mean(importance), color='green', linestyle='--', 
                label=f'Mean: {np.mean(importance):.1f}%')
    ax1.legend()
    
    # Plot 2: Importance by tag (bar chart)
    ax2 = axes[0, 1]
    tags = list(tag_metrics.keys())
    avg_importance = [np.mean(tag_metrics[t]['resampling_acc']) * 100 for t in tags]
    colors = ['red' if t in anchor_tags else 'steelblue' for t in tags]
    ax2.barh(tags, avg_importance, color=colors)
    ax2.set_xlabel('Avg Resampling Importance (%)')
    ax2.set_title('Importance by Sentence Type')
    
    # Plot 3: Trajectory divergence by tag
    ax3 = axes[1, 0]
    traj_div = [np.mean(tag_metrics[t]['trajectory_div']) * 100 for t in tags]
    colors = ['red' if t in anchor_tags else 'steelblue' for t in tags]
    ax3.barh(tags, traj_div, color=colors)
    ax3.set_xlabel('Trajectory Divergence (%)')
    ax3.set_title('How Often Does Resampling Change Direction?')
    
    # Plot 4: Scatter of importance vs overdeterminedness
    ax4 = axes[1, 1]
    importance = [c.get('resampling_importance_accuracy', 0) * 100 for c in chunks]
    overdet = [c.get('overdeterminedness', 0) * 100 for c in chunks]
    
    tag_colors = {
        'plan_generation': 'red', 
        'uncertainty_management': 'orange',
        'active_computation': 'blue', 
        'fact_retrieval': 'green',
        'result_consolidation': 'purple', 
        'problem_setup': 'gray',
        'self_checking': 'pink', 
        'final_answer_emission': 'black'
    }
    colors = [tag_colors.get(c.get('function_tags', ['unknown'])[0], 'gray') 
              for c in chunks]
    
    ax4.scatter(overdet, importance, c=colors, alpha=0.6, s=50)
    ax4.set_xlabel('Overdeterminedness (%)')
    ax4.set_ylabel('Resampling Importance (%)')
    ax4.set_title('Importance vs Overdeterminedness')
    
    # Add legend for scatter plot
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=c, markersize=8, label=t)
                       for t, c in tag_colors.items() if t in tag_metrics]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Thought Anchors analysis results"
    )
    parser.add_argument(
        '--problem', '-p',
        type=str,
        default='problem_2238',
        help='Problem ID to analyze (default: problem_2238)'
    )
    parser.add_argument(
        '--rollouts_dir', '-r',
        type=str,
        default='math_rollouts',
        help='Directory containing rollout data (default: math_rollouts)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Directory to save visualization (default: same as problem dir)'
    )
    parser.add_argument(
        '--no_plot',
        action='store_true',
        help='Skip visualization, only print text output'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    problem_dir = Path(args.rollouts_dir) / args.problem
    output_dir = Path(args.output_dir) if args.output_dir else problem_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 75)
    print(f"THOUGHT ANCHORS ANALYSIS: {args.problem}")
    print("=" * 75)
    
    # Load data
    print(f"\nLoading data from {problem_dir}...")
    chunks = load_data(problem_dir)
    print(f"Loaded {len(chunks)} labeled chunks with pre-computed metrics")
    
    # Print methodology
    print_methodology()
    
    # Compute grouped metrics
    tag_metrics = compute_tag_metrics(chunks)
    
    # Print results
    print_results_table(tag_metrics)
    print_top_anchors(chunks)
    
    # Create visualization
    if not args.no_plot:
        output_path = output_dir / "analysis_plot.png"
        create_visualization(chunks, tag_metrics, output_path)
    
    print("\n" + "=" * 75)
    print("KEY INSIGHT FROM THE PAPER:")
    print("=" * 75)
    print("""
Planning and uncertainty management sentences are THOUGHT ANCHORS:
- They have HIGH importance (removing them hurts accuracy)
- They have HIGH trajectory divergence (they determine which path the model takes)

Computation sentences are LESS important:
- The model can redo computations from different starting points
- Removing them often doesn't change the final answer
""")


if __name__ == "__main__":
    main()

