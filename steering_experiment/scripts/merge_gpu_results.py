#!/usr/bin/env python3
"""
Merge results from multiple GPU workers into a single experiment directory.

This script combines rollouts from parallel GPU runs into a unified result set.
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime


def merge_results(experiment_dir: Path, n_gpus: int):
    """Merge results from all GPU workers."""
    
    print(f"Merging results from {n_gpus} GPUs...")
    
    rollouts_dir = experiment_dir / "rollouts"
    rollouts_dir.mkdir(exist_ok=True)
    
    total_positions = 0
    all_stats = {
        "total_generations": 0,
        "hit_max_tokens": 0,
        "extraction_failures": 0,
        "steering_warnings": 0,
    }
    
    # The rollouts are already saved to the same directory by each worker
    # Just need to combine checkpoints and stats
    
    # Look for position directories (already merged since all GPUs write to same dir)
    position_dirs = list(rollouts_dir.glob("position_*"))
    total_positions = len(position_dirs)
    
    print(f"  Found {total_positions} position directories")
    
    # Combine checkpoints from all GPUs
    all_completed = set()
    for gpu_id in range(n_gpus):
        checkpoint_file = experiment_dir / f"checkpoint_gpu{gpu_id}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            all_completed.update(checkpoint.get("completed_positions", []))
            
            # Aggregate stats
            gpu_stats = checkpoint.get("stats", {})
            for key in all_stats:
                all_stats[key] += gpu_stats.get(key, 0)
    
    # Also check main checkpoint
    main_checkpoint = experiment_dir / "checkpoint.json"
    if main_checkpoint.exists():
        with open(main_checkpoint, "r") as f:
            checkpoint = json.load(f)
        all_completed.update(checkpoint.get("completed_positions", []))
    
    print(f"  Total positions completed: {len(all_completed)}")
    print(f"  Total generations: {all_stats['total_generations']}")
    
    # Write merged checkpoint
    merged_checkpoint = {
        "completed_positions": sorted(list(all_completed)),
        "last_updated": datetime.now().isoformat(),
        "stats": all_stats,
        "merged_from_gpus": n_gpus,
    }
    
    with open(experiment_dir / "checkpoint.json", "w") as f:
        json.dump(merged_checkpoint, f, indent=2)
    
    # Update summary if exists
    summary_file = experiment_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
        
        summary["positions_tested"] = len(all_completed)
        summary["generation_stats"] = all_stats
        summary["merged_from_gpus"] = n_gpus
        summary["merge_timestamp"] = datetime.now().isoformat()
        
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
    
    print(f"\nMerge complete!")
    print(f"  Positions: {len(all_completed)}")
    print(f"  Results: {experiment_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge multi-GPU experiment results")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to experiment directory")
    parser.add_argument("--n_gpus", type=int, default=8,
                        help="Number of GPUs used")
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"Error: Directory not found: {experiment_dir}")
        return 1
    
    merge_results(experiment_dir, args.n_gpus)
    return 0


if __name__ == "__main__":
    exit(main())


