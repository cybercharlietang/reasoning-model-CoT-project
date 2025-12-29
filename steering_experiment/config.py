"""
Configuration for the Steering Vectors × Thought Anchors experiment.

This module contains all configurable parameters for the experiment.
Modify values here to change experiment settings without editing scripts.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


# =============================================================================
# PATHS
# =============================================================================

# Base paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent  # reasoning-model-CoT-project/
STEERING_VECTORS_PATH = PROJECT_ROOT / "steering vectors" / "venhoff_steering_vectors_14b.pt"
SELECTED_PROBLEMS_PATH = PROJECT_ROOT / "selected_problems.json"
RESULTS_DIR = Path(__file__).parent / "results"

# HuggingFace dataset
HF_DATASET_REPO = "uzaymacar/math-rollouts"

# Prompt template (must match what steering vectors were trained on)
PROMPT_TEMPLATE = "Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem} Solution: \n<think>\n{prefix}"


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    
    # --- Problem Selection ---
    problem_idx: int = 0  # Index into selected_problems.json (0-19)
    
    # --- Steering ---
    steering_behavior: str = "backtracking"  # Options: initializing, deduction, adding_knowledge, example_testing, uncertainty_estimation, backtracking
    alpha_values: List[float] = field(default_factory=lambda: [0.0, -1.0])  # 0.0 = baseline, -1.0 = negative steering
    
    # --- Rollouts ---
    n_rollouts: int = 30  # Rollouts per condition per position
    batch_size: int = 8  # Batch size for generation
    
    # --- Generation ---
    max_new_tokens: int = 4096  # Increase to 8192 if >5% hit limit
    temperature: float = 0.6
    top_p: float = 0.95
    
    # --- Model ---
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    use_quantization: bool = False  # Set True if OOM
    device_map: str = "auto"  # "auto" spreads across available GPUs
    torch_dtype: str = "float16"
    
    # --- Output ---
    output_dir: Optional[str] = None  # Auto-generated if None
    save_full_cot: bool = True  # Save full chain of thought (not just answers)
    checkpoint_every: int = 1  # Save checkpoint every N positions
    
    # --- Sanity Checks ---
    warn_max_tokens_threshold: float = 0.05  # Warn if >5% hit max_tokens
    min_baseline_diversity: int = 2  # Minimum unique answers in baseline
    
    def __post_init__(self):
        """Generate output directory name if not specified."""
        if self.output_dir is None:
            alpha_str = "_".join([f"a{a}" for a in self.alpha_values])
            self.output_dir = str(
                RESULTS_DIR / "pilot" / 
                f"problem_{self.problem_idx}_{self.steering_behavior}_{alpha_str}_n{self.n_rollouts}"
            )


@dataclass
class ModelConfig:
    """Model-specific configuration."""
    
    name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    num_layers: int = 48
    hidden_dim: int = 5120
    
    # Generation settings
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 4096
    do_sample: bool = True
    
    # Prompt format (matches generate_rollouts.py from Thought Anchors repo)
    prompt_template: str = "Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem} Solution: \n<think>\n{prefix}"
    

# =============================================================================
# STEERING VECTOR CONFIGURATION
# =============================================================================

# Layer assignments for each behavior (from Venhoff et al.)
STEERING_LAYERS = {
    "initializing": 29,
    "deduction": 29,
    "adding_knowledge": 24,  # Different layer!
    "example_testing": 29,
    "uncertainty_estimation": 29,
    "backtracking": 29,
}

# Mapping from steering behaviors to target anchor types (for analysis)
BEHAVIOR_TO_ANCHOR = {
    "backtracking": "uncertainty_management",
    "uncertainty_estimation": "uncertainty_management",
    # Add more mappings as needed
}


# =============================================================================
# ANCHOR TYPES (from Thought Anchors paper)
# =============================================================================

ANCHOR_TYPES = [
    "problem_setup",
    "plan_generation", 
    "fact_retrieval",
    "active_computation",
    "result_consolidation",
    "uncertainty_management",  # Target for backtracking steering
    "final_answer_emission",
    "self_checking",
    "unknown",
]


# =============================================================================
# DEFAULT CONFIGURATION INSTANCE
# =============================================================================

def get_default_config() -> ExperimentConfig:
    """Get the default experiment configuration."""
    return ExperimentConfig()


def get_pilot_config() -> ExperimentConfig:
    """Get configuration for pilot experiment (1 problem, backtracking, negative steering)."""
    return ExperimentConfig(
        problem_idx=0,
        steering_behavior="backtracking",
        alpha_values=[0.0, -1.0],  # baseline + negative
        n_rollouts=30,
        batch_size=8,
        max_new_tokens=4096,
        save_full_cot=True,
    )


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config(config: ExperimentConfig) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.
    
    Returns:
        List of warning/error messages (empty if all valid)
    """
    issues = []
    
    # Check problem index
    if config.problem_idx < 0 or config.problem_idx >= 20:
        issues.append(f"problem_idx {config.problem_idx} out of range [0, 19]")
    
    # Check steering behavior
    if config.steering_behavior not in STEERING_LAYERS:
        issues.append(f"Unknown steering behavior: {config.steering_behavior}")
    
    # Check alpha values
    if 0.0 not in config.alpha_values:
        issues.append("Warning: alpha_values should include 0.0 for baseline comparison")
    
    # Check batch size
    if config.batch_size < 1 or config.batch_size > 32:
        issues.append(f"batch_size {config.batch_size} may be suboptimal (recommended: 4-16)")
    
    # Check paths
    if not STEERING_VECTORS_PATH.exists():
        issues.append(f"Steering vectors not found: {STEERING_VECTORS_PATH}")
    
    if not SELECTED_PROBLEMS_PATH.exists():
        issues.append(f"Selected problems not found: {SELECTED_PROBLEMS_PATH}")
    
    return issues


if __name__ == "__main__":
    # Print configuration for verification
    config = get_pilot_config()
    print("=== Pilot Configuration ===")
    print(f"Problem index: {config.problem_idx}")
    print(f"Steering behavior: {config.steering_behavior}")
    print(f"Alpha values: {config.alpha_values}")
    print(f"Rollouts per condition: {config.n_rollouts}")
    print(f"Batch size: {config.batch_size}")
    print(f"Max new tokens: {config.max_new_tokens}")
    print(f"Output directory: {config.output_dir}")
    print()
    
    # Validate
    issues = validate_config(config)
    if issues:
        print("=== Validation Issues ===")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("=== Configuration Valid ===")

