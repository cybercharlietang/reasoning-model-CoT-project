#!/usr/bin/env python3
"""
Script 1: Generate Steered Rollouts

This script generates rollouts for a single problem with steering vectors applied
at each position (chunk) in the reasoning trace.

For each position:
- Generate N baseline rollouts (alpha=0)
- Generate N steered rollouts (alpha=-1.0 or as configured)

Results are saved incrementally with checkpointing.

Usage:
    python scripts/1_generate_rollouts.py [--config CONFIG_OVERRIDE]
    
    # Or with specific overrides:
    python scripts/1_generate_rollouts.py --problem_idx 0 --n_rollouts 10 --batch_size 4
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from steering_experiment.config import (
    ExperimentConfig,
    get_pilot_config,
    validate_config,
    STEERING_LAYERS,
    PROMPT_TEMPLATE,
    EXPERIMENTS_DIR,
    generate_experiment_name,
)
from steering_experiment.src.data_loading import (
    load_problem_by_index,
    load_trace,
    get_steering_vector,
    validate_trace,
    get_problem_statistics,
)
from steering_experiment.src.steering import (
    SteeringManager,
    verify_steering_hook,
)
from steering_experiment.src.generation import (
    load_model_and_tokenizer,
    generate_rollouts_for_position,
    compute_generation_stats,
    save_rollouts,
    PositionRollouts,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate steered rollouts for steering vectors experiment"
    )
    
    # Problem selection
    parser.add_argument(
        "--problem_idx", type=int, default=0,
        help="Index into selected_problems.json (0-105)"
    )
    
    # Steering configuration
    parser.add_argument(
        "--behavior", type=str, default="backtracking",
        choices=["backtracking", "uncertainty_estimation", "initializing", 
                 "deduction", "adding_knowledge", "example_testing"],
        help="Steering behavior to use"
    )
    parser.add_argument(
        "--alpha", type=float, nargs="+", default=[0.0, -1.0],
        help="Steering strengths (space-separated, e.g., 0.0 -1.0 1.0)"
    )
    
    # Generation settings
    parser.add_argument(
        "--n_rollouts", type=int, default=30,
        help="Number of rollouts per condition per position"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=4096,
        help="Maximum tokens to generate per rollout"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Nucleus sampling parameter"
    )
    
    # Model settings
    parser.add_argument(
        "--quantize", action="store_true",
        help="Use 4-bit quantization (for low VRAM)"
    )
    
    # Position selection
    parser.add_argument(
        "--position_strategy", type=str, default="all",
        choices=["all", "balanced", "sample"],
        help="How to select positions to steer"
    )
    parser.add_argument(
        "--max_positions", type=int, default=None,
        help="Maximum number of positions to test (None for all)"
    )
    
    # Output settings
    parser.add_argument(
        "--experiment_id", type=str, default=None,
        help="Experiment ID (auto-incremented if not specified). "
             "Can be a number (e.g., 5 creates exp005_...) or a full name."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (overrides auto-generated name if specified)"
    )
    parser.add_argument(
        "--no_save_cot", action="store_true",
        help="Don't save full chain-of-thought (only save answers)"
    )
    
    # Execution control
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--skip_verification", action="store_true",
        help="Skip steering hook verification (faster startup)"
    )
    
    # Multi-GPU parallelism
    parser.add_argument(
        "--gpu_id", type=int, default=None,
        help="GPU ID for this worker (0-7). If set, positions are sharded across GPUs."
    )
    parser.add_argument(
        "--n_gpus", type=int, default=1,
        help="Total number of GPUs for position sharding"
    )
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Create experiment config from command line arguments."""
    
    # Determine output directory
    if args.output_dir:
        # Manual override takes precedence
        output_dir = args.output_dir
    elif args.experiment_id and not args.experiment_id.isdigit():
        # Full experiment ID provided (e.g., from multi-GPU script)
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = str(EXPERIMENTS_DIR / args.experiment_id)
    else:
        # Generate experiment name (with optional manual ID)
        exp_id = int(args.experiment_id) if args.experiment_id else None
        exp_name = generate_experiment_name(
            problem_idx=args.problem_idx,
            behavior=args.behavior,
            alpha_values=args.alpha,
            n_rollouts=args.n_rollouts,
            experiment_id=exp_id,  # None = auto-increment
        )
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = str(EXPERIMENTS_DIR / exp_name)
    
    # Determine device map - use specific GPU if running in multi-GPU mode
    if args.gpu_id is not None:
        device_map = f"cuda:0"  # CUDA_VISIBLE_DEVICES handles the actual GPU
    else:
        device_map = "auto"  # Spread across available GPUs
    
    config = ExperimentConfig(
        problem_idx=args.problem_idx,
        steering_behavior=args.behavior,
        alpha_values=args.alpha,
        n_rollouts=args.n_rollouts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_quantization=args.quantize,
        save_full_cot=not args.no_save_cot,
        output_dir=output_dir,
        device_map=device_map,
    )
    
    return config


def load_checkpoint(output_dir: Path) -> set:
    """Load checkpoint to find completed positions."""
    checkpoint_file = output_dir / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
        return set(checkpoint.get("completed_positions", []))
    return set()


def save_checkpoint(output_dir: Path, completed_positions: set, stats: dict, gpu_id: int = None) -> None:
    """Save checkpoint with completed positions."""
    # Use GPU-specific checkpoint file if running in multi-GPU mode
    if gpu_id is not None:
        checkpoint_file = output_dir / f"checkpoint_gpu{gpu_id}.json"
    else:
        checkpoint_file = output_dir / "checkpoint.json"
    
    checkpoint = {
        "completed_positions": sorted(list(completed_positions)),
        "last_updated": datetime.now().isoformat(),
        "stats": stats,
        "gpu_id": gpu_id,
    }
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f, indent=2)


def main():
    """Main function."""
    args = parse_args()
    config = create_config_from_args(args)
    
    print("=" * 60)
    print("STEERING VECTORS EXPERIMENT - ROLLOUT GENERATION")
    print("=" * 60)
    print()
    
    # Validate configuration
    issues = validate_config(config)
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        if any("Error" in issue for issue in issues):
            print("\nAborting due to configuration errors.")
            return 1
        print()
    
    # Print configuration
    print("Configuration:")
    print(f"  Problem index: {config.problem_idx}")
    print(f"  Steering behavior: {config.steering_behavior}")
    print(f"  Alpha values: {config.alpha_values}")
    print(f"  Rollouts per condition: {config.n_rollouts}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max new tokens: {config.max_new_tokens}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Position strategy: {args.position_strategy}")
    print(f"  Output directory: {config.output_dir}")
    print()
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump({
            "problem_idx": config.problem_idx,
            "steering_behavior": config.steering_behavior,
            "alpha_values": config.alpha_values,
            "n_rollouts": config.n_rollouts,
            "batch_size": config.batch_size,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "position_strategy": args.position_strategy,
            "max_positions": args.max_positions,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    print("Loading data...")
    
    # Load problem
    problem = load_problem_by_index(config.problem_idx)
    print(f"  Problem: {problem.problem_idx}")
    print(f"  Type: {problem.type}")
    print(f"  Ground truth: {problem.gt_answer}")
    
    # Load trace
    trace = load_trace(problem)
    print(f"  Loaded {trace.num_chunks} chunks")
    
    # Validate trace
    trace_issues = validate_trace(trace)
    if trace_issues:
        print(f"  Trace issues: {trace_issues}")
    
    # Get problem statistics
    stats = get_problem_statistics(problem, trace)
    print(f"  Uncertainty management positions: {stats['uncertainty_management_positions']}")
    print()
    
    # Select positions to test
    target_positions, control_positions = trace.select_positions(
        strategy=args.position_strategy,
        max_positions=args.max_positions,
        target_tag="uncertainty_management",
    )
    all_positions = sorted(set(target_positions + control_positions))
    
    # Multi-GPU position sharding
    if args.gpu_id is not None and args.n_gpus > 1:
        # Shard positions across GPUs: GPU 0 gets [0, 8, 16...], GPU 1 gets [1, 9, 17...], etc.
        all_positions = [p for i, p in enumerate(all_positions) if i % args.n_gpus == args.gpu_id]
        target_positions = [p for p in target_positions if p in all_positions]
        control_positions = [p for p in control_positions if p in all_positions]
        print(f"GPU {args.gpu_id}/{args.n_gpus}: Processing {len(all_positions)} positions (sharded)")
    
    print(f"Positions to test: {len(all_positions)}")
    print(f"  Target (uncertainty_management): {len(target_positions)}")
    print(f"  Control (other): {len(control_positions)}")
    print()
    
    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model_name,
        device_map=config.device_map,
        torch_dtype=config.torch_dtype,
        use_quantization=config.use_quantization,
    )
    print()
    
    # =========================================================================
    # SET UP STEERING
    # =========================================================================
    
    print("Setting up steering...")
    
    # Load steering vector
    steering_vec = get_steering_vector(config.steering_behavior)
    print(f"  Loaded {config.steering_behavior} vector")
    print(f"  Layer: {steering_vec.layer}")
    print(f"  Vector norm: {steering_vec.normalized.norm().item():.4f}")
    
    # Create steering manager
    steering_manager = SteeringManager(
        model=model,
        steering_vector=steering_vec.normalized,
        layer=steering_vec.layer,
    )
    steering_manager.register_hook()
    
    # Verify steering works
    if not args.skip_verification:
        print("  Verifying steering hook...")
        verify_results = verify_steering_hook(model, steering_manager, tokenizer)
        print(f"  Verification: activations_differ={verify_results['activations_differ']}, "
              f"diff={verify_results.get('mean_activation_diff', 0):.6f}")
    print()
    
    # =========================================================================
    # CHECK FOR RESUME
    # =========================================================================
    
    completed_positions = set()
    if args.resume:
        completed_positions = load_checkpoint(output_dir)
        if completed_positions:
            print(f"Resuming from checkpoint: {len(completed_positions)} positions already complete")
            print()
    
    # =========================================================================
    # GENERATE ROLLOUTS
    # =========================================================================
    
    print("Generating rollouts...")
    start_time = time.time()
    
    all_rollouts = []
    generation_stats = {
        "total_generations": 0,
        "hit_max_tokens": 0,
        "extraction_failures": 0,
        "steering_warnings": 0,
    }
    
    positions_to_process = [p for p in all_positions if p not in completed_positions]
    
    with tqdm(total=len(positions_to_process), desc="Positions") as pbar:
        for position_idx in positions_to_process:
            # Generate rollouts for this position
            position_rollouts = generate_rollouts_for_position(
                model=model,
                tokenizer=tokenizer,
                trace=trace,
                position_idx=position_idx,
                steering_manager=steering_manager,
                alpha_values=config.alpha_values,
                n_rollouts=config.n_rollouts,
                batch_size=config.batch_size,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                prompt_template=PROMPT_TEMPLATE,
            )
            
            all_rollouts.append(position_rollouts)
            
            # Update stats and compute baseline accuracy
            baseline_correct = 0
            baseline_total = 0
            for alpha, results in position_rollouts.results_by_alpha.items():
                for result in results:
                    generation_stats["total_generations"] += 1
                    if result.hit_max_tokens:
                        generation_stats["hit_max_tokens"] += 1
                    if result.answer is None:
                        generation_stats["extraction_failures"] += 1
                    # Track baseline accuracy
                    if alpha == 0.0:
                        baseline_total += 1
                        if result.is_correct:
                            baseline_correct += 1
            
            # Verify steering was applied for non-zero alphas
            if any(a != 0.0 for a in config.alpha_values):
                if not steering_manager.steering_applied:
                    print(f"\nWARNING: Steering not applied at position {position_idx}!")
                    print(f"  Steer range: {position_rollouts.steer_range}")
                    generation_stats["steering_warnings"] += 1
                steering_manager.reset_stats()
            
            # Save checkpoint
            completed_positions.add(position_idx)
            save_checkpoint(output_dir, completed_positions, generation_stats, gpu_id=args.gpu_id)
            
            # Save this position's rollouts
            save_rollouts(
                [position_rollouts],
                output_dir / "rollouts",
                save_full_cot=config.save_full_cot,
            )
            
            # Update progress bar with accuracy
            baseline_acc = baseline_correct / baseline_total if baseline_total > 0 else 0
            pbar.update(1)
            pbar.set_postfix({
                "gen": generation_stats["total_generations"],
                "max_tok": generation_stats["hit_max_tokens"],
                "acc": f"{baseline_acc:.0%}",
            })
    
    elapsed_time = time.time() - start_time
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print()
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print()
    print(f"Time elapsed: {elapsed_time / 60:.1f} minutes")
    print(f"Positions processed: {len(all_rollouts)}")
    print(f"Total generations: {generation_stats['total_generations']}")
    print()
    
    # Check max_tokens warning
    if generation_stats["total_generations"] > 0:
        max_tokens_rate = generation_stats["hit_max_tokens"] / generation_stats["total_generations"]
        print(f"Max tokens hit rate: {max_tokens_rate:.1%}")
        if max_tokens_rate > config.warn_max_tokens_threshold:
            print(f"  WARNING: {max_tokens_rate:.1%} > {config.warn_max_tokens_threshold:.1%} threshold!")
            print(f"  Consider increasing max_new_tokens to 8192")
        
        extraction_rate = 1.0 - (generation_stats["extraction_failures"] / generation_stats["total_generations"])
        print(f"Answer extraction rate: {extraction_rate:.1%}")
    
    print()
    print(f"Results saved to: {output_dir}")
    print()
    
    # Save final summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "problem": {
                "idx": problem.problem_idx,
                "type": problem.type,
                "gt_answer": problem.gt_answer,
                "num_chunks": trace.num_chunks,
            },
            "positions_tested": len(all_rollouts),
            "target_positions": target_positions,
            "control_positions": control_positions,
            "generation_stats": generation_stats,
            "elapsed_time_seconds": elapsed_time,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    # Cleanup
    steering_manager.remove_hook()
    
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())

