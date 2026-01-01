#!/usr/bin/env python3
"""
Quick helper to check problem statistics before running experiments.

Usage:
    python check_problem_stats.py                    # Show top 20 shortest
    python check_problem_stats.py 0 1 2 3 4 5       # Show specific problems
    python check_problem_stats.py --all              # Show all 106 problems
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_problems():
    with open(PROJECT_ROOT / "selected_problems.json") as f:
        return json.load(f)

def print_problem_table(problems, indices=None):
    """Print a table of problems."""
    print(f"{'Idx':>3} | {'Problem ID':>12} | {'Chunks':>7} | {'Tokens':>7} | {'Acc':>5} | {'Type':>25} | Question")
    print("-" * 110)
    
    if indices is None:
        # Sort by chunks and show all
        sorted_probs = sorted(enumerate(problems), key=lambda x: x[1].get('approximate_mean_chunks', 999))
        for idx, p in sorted_probs:
            print_row(idx, p)
    else:
        for idx in indices:
            if 0 <= idx < len(problems):
                print_row(idx, problems[idx])

def print_row(idx, p):
    chunks = p.get('approximate_mean_chunks', 0)
    tokens = p.get('approximate_mean_tokens', 0)
    acc = p.get('accuracy', 0)
    ptype = p.get('type', 'N/A')[:25]
    question = p.get('problem', '')[:45].replace('\n', ' ') + "..."
    print(f"{idx:3d} | {p['problem_idx']:>12} | {chunks:>7.1f} | {tokens:>7.1f} | {acc:>5.2f} | {ptype:>25} | {question}")

def main():
    problems = load_problems()
    
    if len(sys.argv) == 1:
        # Show top 20 shortest
        print(f"\n{'='*110}")
        print("TOP 20 SHORTEST PROBLEMS (sorted by approximate_mean_chunks)")
        print(f"{'='*110}\n")
        sorted_probs = sorted(enumerate(problems), key=lambda x: x[1].get('approximate_mean_chunks', 999))
        print_problem_table(problems, [idx for idx, _ in sorted_probs[:20]])
    elif sys.argv[1] == '--all':
        print(f"\n{'='*110}")
        print("ALL 106 PROBLEMS (sorted by approximate_mean_chunks)")
        print(f"{'='*110}\n")
        print_problem_table(problems)
    else:
        # Show specific problems
        indices = [int(x) for x in sys.argv[1:] if x.isdigit()]
        print(f"\n{'='*110}")
        print(f"PROBLEMS: {indices}")
        print(f"{'='*110}\n")
        print_problem_table(problems, indices)
    
    print(f"\nTotal problems: {len(problems)}")
    print("\nNOTE: 'Chunks' is approximate_mean_chunks from the dataset.")
    print("      Actual chunk count may differ. Max token hit rate is the main runtime factor!")

if __name__ == "__main__":
    main()


