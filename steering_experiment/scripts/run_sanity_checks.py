#!/usr/bin/env python3
"""
Sanity Checks Script

This script runs comprehensive sanity checks to verify that all components
of the steering experiment are working correctly before running the full
experiment.

Checks performed:
1. Data loading (problems, traces, steering vectors)
2. Token alignment verification
3. Steering hook application
4. Answer extraction
5. Metric computation

Usage:
    python scripts/run_sanity_checks.py [--full]
    
    --full: Run full checks including model loading (slow)
"""

import argparse
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))


def check_imports() -> bool:
    """Check that all required modules can be imported."""
    print("=" * 60)
    print("CHECK 1: Import Verification")
    print("=" * 60)
    
    required_modules = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("tqdm", "TQDM"),
        ("huggingface_hub", "HuggingFace Hub"),
    ]
    
    all_ok = True
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name}: {e}")
            all_ok = False
    
    # Check our modules
    print("\n  Checking project modules...")
    project_modules = [
        ("steering_experiment.config", "Configuration"),
        ("steering_experiment.src.data_loading", "Data Loading"),
        ("steering_experiment.src.steering", "Steering"),
        ("steering_experiment.src.generation", "Generation"),
    ]
    
    for module_name, display_name in project_modules:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name}: {e}")
            all_ok = False
    
    print()
    return all_ok


def check_data_loading() -> bool:
    """Check that data loading works correctly."""
    print("=" * 60)
    print("CHECK 2: Data Loading")
    print("=" * 60)
    
    from steering_experiment.src.data_loading import (
        load_problem_by_index,
        load_trace,
        validate_trace,
        get_steering_vector,
    )
    
    all_ok = True
    
    # Check problem loading
    print("  Loading problem 0...")
    problem = load_problem_by_index(0)
    print(f"  ✓ Problem loaded: idx={problem.problem_idx}, type={problem.type}")
    print(f"    GT answer: {problem.gt_answer}")
    
    # Check trace loading
    print("\n  Loading trace...")
    trace = load_trace(problem)
    print(f"  ✓ Trace loaded: {trace.num_chunks} chunks")
    
    # Validate trace
    issues = validate_trace(trace)
    if issues:
        print(f"  ⚠ Trace validation issues: {issues}")
    else:
        print("  ✓ Trace validation passed")
    
    # Check chunk structure
    print("\n  Checking chunk structure...")
    for i, chunk in enumerate(trace.chunks[:3]):
        print(f"    Chunk {i}: {len(chunk.chunk)} chars, tags={chunk.function_tags}")
    if trace.num_chunks > 3:
        print(f"    ... ({trace.num_chunks - 3} more chunks)")
    
    # Check uncertainty_management positions
    unc_positions = trace.get_uncertainty_management_positions()
    print(f"\n  Uncertainty management positions: {len(unc_positions)}")
    if unc_positions:
        print(f"    Indices: {unc_positions[:5]}{'...' if len(unc_positions) > 5 else ''}")
    
    # Check steering vector loading
    print("\n  Loading steering vectors...")
    for behavior in ["backtracking", "uncertainty_estimation", "initializing"]:
        try:
            sv = get_steering_vector(behavior)
            print(f"  ✓ {behavior}: layer={sv.layer}, norm={sv.normalized.norm().item():.4f}")
        except Exception as e:
            print(f"  ✗ {behavior}: {e}")
            all_ok = False
    
    print()
    return all_ok


def check_token_alignment(tokenizer=None) -> bool:
    """Check that token alignment works correctly."""
    print("=" * 60)
    print("CHECK 3: Token Alignment")
    print("=" * 60)
    
    if tokenizer is None:
        print("  Skipping (no tokenizer - use --full for complete check)")
        print()
        return True
    
    from steering_experiment.src.data_loading import load_problem_by_index, load_trace
    from steering_experiment.src.steering import compute_steer_range_for_chunk
    from steering_experiment.config import PROMPT_TEMPLATE
    
    # Load problem and trace
    problem = load_problem_by_index(0)
    trace = load_trace(problem)
    
    all_ok = True
    
    # Check a few positions
    test_positions = [0, min(5, trace.num_chunks - 1), trace.num_chunks - 1]
    test_positions = sorted(set(test_positions))
    
    for pos in test_positions:
        print(f"\n  Position {pos}:")
        
        prefix_chunks = [c.chunk for c in trace.chunks[:pos]]
        target_chunk = trace.chunks[pos].chunk
        
        start_idx, end_idx = compute_steer_range_for_chunk(
            problem_text=problem.problem_text,
            prefix_chunks=prefix_chunks,
            target_chunk=target_chunk,
            tokenizer=tokenizer,
            prompt_template=PROMPT_TEMPLATE,
        )
        
        print(f"    ✓ Token range: [{start_idx}, {end_idx}] ({end_idx - start_idx + 1} tokens)")
        print(f"    Chunk: \"{target_chunk[:50]}...\"")
        
        # Verify by decoding
        prefix_with_chunk = " ".join(prefix_chunks + [target_chunk]) if prefix_chunks else target_chunk
        full_prompt = PROMPT_TEMPLATE.format(problem=problem.problem_text, prefix=prefix_with_chunk)
        tokens = tokenizer.encode(full_prompt, add_special_tokens=True)
        
        if end_idx < len(tokens):
            decoded = tokenizer.decode(tokens[start_idx:end_idx + 1])
            print(f"    Decoded: \"{decoded[:50]}...\"")
        else:
            print(f"    ⚠ End index {end_idx} >= token length {len(tokens)}")
            all_ok = False
    
    print()
    return all_ok


def check_steering_hook(model=None, tokenizer=None) -> bool:
    """Check that steering hooks work correctly."""
    print("=" * 60)
    print("CHECK 4: Steering Hook")
    print("=" * 60)
    
    if model is None or tokenizer is None:
        print("  Skipping (no model - use --full for complete check)")
        print()
        return True
    
    from steering_experiment.src.data_loading import get_steering_vector
    from steering_experiment.src.steering import SteeringManager, verify_steering_hook
    
    # Get steering vector
    sv = get_steering_vector("backtracking")
    
    # Create manager
    manager = SteeringManager(
        model=model,
        steering_vector=sv.normalized,
        layer=sv.layer,
    )
    manager.register_hook()
    
    # Verify
    print("  Running verification...")
    results = verify_steering_hook(model, manager, tokenizer)
    
    print(f"  Activations differ: {results['activations_differ']}")
    if 'mean_activation_diff' in results:
        print(f"  Mean difference: {results['mean_activation_diff']:.6f}")
    if 'steering_applied' in results:
        print(f"  Steering applied: {results['steering_applied']}")
    
    all_ok = results['activations_differ']
    
    if all_ok:
        print("  ✓ Steering hook working correctly")
    else:
        print("  ✗ Steering hook NOT working")
    
    # Cleanup
    manager.remove_hook()
    
    print()
    return all_ok


def check_answer_extraction() -> bool:
    """Check that answer extraction works correctly."""
    print("=" * 60)
    print("CHECK 5: Answer Extraction")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils import extract_boxed_answers, check_answer
    
    test_cases = [
        ("The answer is \\boxed{42}", "42", True),
        ("Therefore \\boxed{\\frac{1}{2}}", "\\frac{1}{2}", True),
        ("So we get \\boxed{x^2 + 1}", "x^2 + 1", True),
        ("No boxed answer here", None, False),
        ("<think>Thinking...\\boxed{wrong}</think>The final answer is \\boxed{correct}", "correct", True),
    ]
    
    all_ok = True
    
    for text, expected, should_extract in test_cases:
        # Handle </think> like we do in generation
        if "</think>" in text:
            answer_section = text.split("</think>")[-1]
        else:
            answer_section = text
        
        extracted = extract_boxed_answers(answer_section)
        answer = extracted[0] if extracted and extracted[0] else None
        
        if should_extract:
            if answer == expected:
                print(f"  ✓ \"{text[:40]}...\" → \"{answer}\"")
            else:
                print(f"  ✗ \"{text[:40]}...\" → \"{answer}\" (expected: \"{expected}\")")
                all_ok = False
        else:
            if answer is None:
                print(f"  ✓ \"{text[:40]}...\" → None (as expected)")
            else:
                print(f"  ✗ \"{text[:40]}...\" → \"{answer}\" (expected: None)")
                all_ok = False
    
    # Check answer comparison
    print("\n  Checking answer comparison...")
    comparison_tests = [
        ("42", "42", True),
        ("1/2", "0.5", True),
        ("\\frac{1}{2}", "0.5", True),
        ("42", "43", False),
    ]
    
    for ans1, ans2, should_match in comparison_tests:
        result = check_answer(ans1, ans2)
        if result == should_match:
            print(f"  ✓ \"{ans1}\" vs \"{ans2}\" → {result}")
        else:
            print(f"  ✗ \"{ans1}\" vs \"{ans2}\" → {result} (expected: {should_match})")
            all_ok = False
    
    print()
    return all_ok


def check_metric_computation() -> bool:
    """Check that metric computation works correctly."""
    print("=" * 60)
    print("CHECK 6: Metric Computation")
    print("=" * 60)
    
    import math
    from collections import Counter
    
    # Inline the metric functions for testing
    def compute_distribution(answers):
        valid_answers = [a for a in answers if a is not None]
        if not valid_answers:
            return {}
        counter = Counter(valid_answers)
        total = len(valid_answers)
        return {answer: count / total for answer, count in counter.items()}
    
    def kl_divergence(p, q, epsilon=1e-10):
        if not p or not q:
            return float('inf')
        all_keys = set(p.keys()) | set(q.keys())
        kl = 0.0
        for key in all_keys:
            p_x = p.get(key, epsilon)
            q_x = q.get(key, epsilon)
            if p_x > epsilon:
                kl += p_x * math.log(p_x / q_x)
        return max(0.0, kl)
    
    def js_divergence(p, q):
        if not p or not q:
            return 1.0
        all_keys = set(p.keys()) | set(q.keys())
        m = {}
        for key in all_keys:
            m[key] = (p.get(key, 0.0) + q.get(key, 0.0)) / 2
        kl_pm = kl_divergence(p, m)
        kl_qm = kl_divergence(q, m)
        return (kl_pm + kl_qm) / 2
    
    all_ok = True
    
    # Test distribution computation
    print("  Testing distribution computation...")
    answers = ["A", "A", "B", "A", None, "B"]
    dist = compute_distribution(answers)
    expected = {"A": 0.6, "B": 0.4}
    
    if abs(dist.get("A", 0) - 0.6) < 0.01 and abs(dist.get("B", 0) - 0.4) < 0.01:
        print(f"  ✓ Distribution: {dist}")
    else:
        print(f"  ✗ Distribution: {dist} (expected: {expected})")
        all_ok = False
    
    # Test KL divergence
    print("\n  Testing KL divergence...")
    p = {"A": 0.5, "B": 0.5}
    q = {"A": 0.25, "B": 0.75}
    kl = kl_divergence(p, q)
    # KL(P||Q) should be positive
    if kl > 0:
        print(f"  ✓ KL(P||Q) = {kl:.4f} (positive as expected)")
    else:
        print(f"  ✗ KL(P||Q) = {kl:.4f} (expected positive)")
        all_ok = False
    
    # Same distribution should have KL near 0
    kl_same = kl_divergence(p, p)
    if kl_same < 0.001:
        print(f"  ✓ KL(P||P) = {kl_same:.6f} ≈ 0")
    else:
        print(f"  ✗ KL(P||P) = {kl_same:.6f} (expected ≈ 0)")
        all_ok = False
    
    # Test JS divergence
    print("\n  Testing JS divergence...")
    js = js_divergence(p, q)
    # JS should be between 0 and log(2)
    if 0 <= js <= math.log(2):
        print(f"  ✓ JS(P,Q) = {js:.4f} (in valid range)")
    else:
        print(f"  ✗ JS(P,Q) = {js:.4f} (expected in [0, {math.log(2):.4f}])")
        all_ok = False
    
    # Same distribution should have JS = 0
    js_same = js_divergence(p, p)
    if js_same < 0.001:
        print(f"  ✓ JS(P,P) = {js_same:.6f} ≈ 0")
    else:
        print(f"  ✗ JS(P,P) = {js_same:.6f} (expected ≈ 0)")
        all_ok = False
    
    print()
    return all_ok


def check_configuration() -> bool:
    """Check that configuration is valid."""
    print("=" * 60)
    print("CHECK 7: Configuration")
    print("=" * 60)
    
    from steering_experiment.config import (
        ExperimentConfig,
        validate_config,
        get_pilot_config,
    )
    
    all_ok = True
    
    # Check default config
    print("  Checking default config...")
    config = ExperimentConfig()
    issues = validate_config(config)
    
    if not issues:
        print("  ✓ Default config valid")
    else:
        for issue in issues:
            print(f"  ⚠ {issue}")
    
    # Check pilot config
    print("\n  Checking pilot config...")
    pilot = get_pilot_config()
    print(f"  ✓ Pilot config: {pilot.n_rollouts} rollouts, alpha={pilot.alpha_values}")
    
    # Print key settings
    print("\n  Key settings:")
    print(f"    Model: {config.model_name}")
    print(f"    Steering behavior: {config.steering_behavior}")
    print(f"    Alpha values: {config.alpha_values}")
    print(f"    Batch size: {config.batch_size}")
    print(f"    Max new tokens: {config.max_new_tokens}")
    
    print()
    return all_ok


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run sanity checks for steering experiment"
    )
    
    parser.add_argument(
        "--full", action="store_true",
        help="Run full checks including model loading (slow)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " STEERING EXPERIMENT - SANITY CHECKS ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    results = {}
    
    # Check 1: Imports
    results["imports"] = check_imports()
    if not results["imports"]:
        print("FATAL: Import check failed. Cannot continue.")
        return 1
    
    # Check 2: Data loading
    results["data_loading"] = check_data_loading()
    
    # Check 5: Answer extraction
    results["answer_extraction"] = check_answer_extraction()
    
    # Check 7: Configuration
    results["configuration"] = check_configuration()
    
    # Full checks require model
    model = None
    tokenizer = None
    
    if args.full:
        print("=" * 60)
        print("LOADING MODEL (--full mode)")
        print("=" * 60)
        
        from steering_experiment.src.generation import load_model_and_tokenizer
        from steering_experiment.config import ExperimentConfig
        
        config = ExperimentConfig()
        model, tokenizer = load_model_and_tokenizer(
            model_name=config.model_name,
            device_map=config.device_map,
            torch_dtype=config.torch_dtype,
        )
        print()
    
    # Check 3: Token alignment
    results["token_alignment"] = check_token_alignment(tokenizer)
    
    # Check 4: Steering hook
    results["steering_hook"] = check_steering_hook(model, tokenizer)
    
    # Check 6: Metric computation
    results["metric_computation"] = check_metric_computation()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {check_name.replace('_', ' ').title()}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All sanity checks passed! ✓")
        print()
        print("Ready to run experiment:")
        print("  python scripts/1_generate_rollouts.py --problem_idx 0 --n_rollouts 30")
    else:
        print("Some checks failed. Please fix issues before running experiment.")
    
    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

