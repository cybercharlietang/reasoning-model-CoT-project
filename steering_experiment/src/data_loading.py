"""
Data Loading Module for Steering Experiment

This module handles:
1. Loading problem metadata from selected_problems.json
2. Downloading traces (chunks_labeled.json) from HuggingFace
3. Loading pre-computed steering vectors
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Try to import huggingface_hub for selective downloads
try:
    from huggingface_hub import hf_hub_download, HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Run: pip install huggingface_hub")

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from steering_experiment.config import (
    STEERING_VECTORS_PATH,
    SELECTED_PROBLEMS_PATH,
    HF_DATASET_REPO,
    STEERING_LAYERS,
    RESULTS_DIR,
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Problem:
    """Represents a math problem with metadata."""
    problem_idx: str  # e.g., "problem_2238"
    problem_text: str
    gt_answer: str
    level: str
    type: str
    accuracy: float
    approximate_mean_chunks: float
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Problem":
        return cls(
            problem_idx=d["problem_idx"],
            problem_text=d["problem"],
            gt_answer=d["gt_answer"],
            level=d["level"],
            type=d["type"],
            accuracy=d["accuracy"],
            approximate_mean_chunks=d.get("approximate_mean_chunks", 100),
        )


@dataclass
class Chunk:
    """Represents a labeled chunk/sentence from a reasoning trace."""
    chunk: str  # The text content
    chunk_idx: int  # Position in trace
    function_tags: List[str]  # e.g., ["uncertainty_management"]
    depends_on: List[int]
    
    # Importance metrics (from Thought Anchors paper)
    accuracy: float  # Baseline accuracy at this position
    counterfactual_importance_kl: float  # CI based on KL divergence
    counterfactual_importance_accuracy: float  # CI based on accuracy change
    resampling_importance_kl: float  # Resampling-based importance (KL)
    resampling_importance_accuracy: float  # Resampling-based importance (accuracy)
    forced_importance_kl: float  # Forced answer importance (KL)
    forced_importance_accuracy: float  # Forced answer importance (accuracy)
    
    # Additional metrics
    different_trajectories_fraction: float  # Fraction of different trajectories
    overdeterminedness: float  # How overdetermined this step is
    summary: str  # Short summary of the chunk
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Chunk":
        return cls(
            chunk=d["chunk"],
            chunk_idx=d["chunk_idx"],
            function_tags=d.get("function_tags", ["unknown"]),
            depends_on=d.get("depends_on", []),
            accuracy=d.get("accuracy", 0.0),
            counterfactual_importance_kl=d.get("counterfactual_importance_kl", 0.0),
            counterfactual_importance_accuracy=d.get("counterfactual_importance_accuracy", 0.0),
            resampling_importance_kl=d.get("resampling_importance_kl", 0.0),
            resampling_importance_accuracy=d.get("resampling_importance_accuracy", 0.0),
            forced_importance_kl=d.get("forced_importance_kl", 0.0),
            forced_importance_accuracy=d.get("forced_importance_accuracy", 0.0),
            different_trajectories_fraction=d.get("different_trajectories_fraction", 0.0),
            overdeterminedness=d.get("overdeterminedness", 0.0),
            summary=d.get("summary", ""),
        )
    
    @property
    def is_uncertainty_management(self) -> bool:
        """Check if this chunk is tagged as uncertainty_management."""
        return "uncertainty_management" in self.function_tags
    
    @property
    def ci_kl(self) -> float:
        """Shorthand for counterfactual_importance_kl."""
        return self.counterfactual_importance_kl
    
    @property
    def ci_accuracy(self) -> float:
        """Shorthand for counterfactual_importance_accuracy."""
        return self.counterfactual_importance_accuracy


@dataclass
class Trace:
    """Represents a full reasoning trace with labeled chunks."""
    problem: Problem
    chunks: List[Chunk]
    
    @property
    def num_chunks(self) -> int:
        return len(self.chunks)
    
    def get_prefix(self, position_idx: int, include_position: bool = True) -> str:
        """
        Get text prefix up to (and optionally including) the given position.
        
        Args:
            position_idx: The chunk index
            include_position: If True, include chunk at position_idx
            
        Returns:
            Concatenated chunk text
        """
        end_idx = position_idx + 1 if include_position else position_idx
        chunks_text = [c.chunk for c in self.chunks[:end_idx]]
        return " ".join(chunks_text)
    
    def get_chunk_token_range(self, position_idx: int, tokenizer) -> Tuple[int, int]:
        """
        Get the token index range for a specific chunk.
        
        Args:
            position_idx: The chunk index
            tokenizer: HuggingFace tokenizer
            
        Returns:
            (start_token_idx, end_token_idx) for the chunk
        """
        # Text before this chunk
        prefix_before = self.get_prefix(position_idx, include_position=False)
        # Text including this chunk  
        prefix_including = self.get_prefix(position_idx, include_position=True)
        
        # We need to add the problem context for accurate tokenization
        # This will be handled by the caller who knows the full prompt format
        
        tokens_before = len(tokenizer.encode(prefix_before, add_special_tokens=False))
        tokens_including = len(tokenizer.encode(prefix_including, add_special_tokens=False))
        
        return tokens_before, tokens_including - 1  # -1 because end is inclusive
    
    def get_uncertainty_management_positions(self) -> List[int]:
        """Get indices of all uncertainty_management chunks."""
        return [i for i, c in enumerate(self.chunks) if c.is_uncertainty_management]
    
    def get_non_uncertainty_positions(self) -> List[int]:
        """Get indices of non-uncertainty_management chunks."""
        return [i for i, c in enumerate(self.chunks) if not c.is_uncertainty_management]
    
    def get_positions_by_tag(self, tag: str) -> List[int]:
        """Get indices of chunks with a specific function tag."""
        return [i for i, c in enumerate(self.chunks) if tag in c.function_tags]
    
    def select_positions(
        self,
        strategy: str = "all",
        max_positions: Optional[int] = None,
        target_tag: str = "uncertainty_management",
        n_target: Optional[int] = None,
        n_control: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Select positions to test based on strategy.
        
        Args:
            strategy: Selection strategy:
                - "all": All positions (split into target/control by tag)
                - "balanced": Equal number of target and control positions
                - "sample": Random sample of positions
            max_positions: Maximum total positions to return
            target_tag: Tag to identify target positions (default: uncertainty_management)
            n_target: Number of target positions (for balanced/sample strategies)
            n_control: Number of control positions (for balanced/sample strategies)
            
        Returns:
            (target_positions, control_positions) - Lists of chunk indices
        """
        import random
        
        # Get all target and control positions
        target_positions = self.get_positions_by_tag(target_tag)
        control_positions = [i for i in range(len(self.chunks)) if i not in target_positions]
        
        if strategy == "all":
            # Return all positions
            if max_positions:
                # Proportionally sample from both, but ensure at least 1 target if any exist
                total = len(target_positions) + len(control_positions)
                if target_positions:
                    n_t = max(1, int(max_positions * len(target_positions) / total))
                else:
                    n_t = 0
                n_c = max_positions - n_t
                target_positions = target_positions[:n_t]
                control_positions = control_positions[:n_c]
                
        elif strategy == "balanced":
            # Equal number from each group
            n = n_target or min(len(target_positions), len(control_positions))
            target_positions = target_positions[:n]
            control_positions = control_positions[:n]
            
        elif strategy == "sample":
            # Random sample
            if n_target and len(target_positions) > n_target:
                target_positions = random.sample(target_positions, n_target)
            if n_control and len(control_positions) > n_control:
                control_positions = random.sample(control_positions, n_control)
        
        return sorted(target_positions), sorted(control_positions)


@dataclass  
class SteeringVector:
    """Represents a loaded steering vector."""
    behavior: str
    raw: torch.Tensor
    normalized: torch.Tensor
    layer: int
    sample_count: int
    
    def get_vector(self, use_normalized: bool = True) -> torch.Tensor:
        """Get the steering vector (normalized by default)."""
        return self.normalized if use_normalized else self.raw


# =============================================================================
# PROBLEM LOADING
# =============================================================================

def load_selected_problems() -> List[Problem]:
    """
    Load all problems from selected_problems.json.
    
    Returns:
        List of Problem objects
    """
    if not SELECTED_PROBLEMS_PATH.exists():
        raise FileNotFoundError(f"Selected problems file not found: {SELECTED_PROBLEMS_PATH}")
    
    with open(SELECTED_PROBLEMS_PATH, 'r') as f:
        problems_data = json.load(f)
    
    return [Problem.from_dict(p) for p in problems_data]


def load_problem_by_index(index: int) -> Problem:
    """
    Load a specific problem by its index in selected_problems.json.
    
    Args:
        index: Index into the list (0-19)
        
    Returns:
        Problem object
    """
    problems = load_selected_problems()
    
    if index < 0 or index >= len(problems):
        raise ValueError(f"Problem index {index} out of range [0, {len(problems)-1}]")
    
    return problems[index]


# =============================================================================
# TRACE DOWNLOADING
# =============================================================================

def _get_trace_path_in_hf(problem_idx: str, model: str = "qwen-14b") -> str:
    """
    Get the path to chunks_labeled.json for a problem in the HF dataset.
    
    The dataset structure is:
    uzaymacar/math-rollouts/
    ├── deepseek-r1-distill-qwen-14b/
    │   └── temperature_0.6_top_p_0.95/
    │       └── correct_base_solution/
    │           └── problem_XXXX/
    │               ├── chunks_labeled.json  ← Contains labeled chunks with CI scores
    │               ├── chunks.json
    │               ├── base_solution.json
    │               └── problem.json
    
    Args:
        problem_idx: Problem ID like "problem_2238"
        model: Model variant ("qwen-14b" or "llama-8b")
        
    Returns:
        Path to chunks_labeled.json in the HF repo
    """
    model_dir = {
        "qwen-14b": "deepseek-r1-distill-qwen-14b",
        "llama-8b": "deepseek-r1-distill-llama-8b",
    }.get(model, "deepseek-r1-distill-qwen-14b")
    
    return f"{model_dir}/temperature_0.6_top_p_0.95/correct_base_solution/{problem_idx}/chunks_labeled.json"


def download_trace_for_problem(
    problem: Problem,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    model: str = "qwen-14b",
) -> List[Chunk]:
    """
    Download chunks_labeled.json for a specific problem from HuggingFace.
    
    Args:
        problem: Problem object with problem_idx
        cache_dir: Directory to cache downloaded files
        force_download: If True, re-download even if cached
        model: Model variant ("qwen-14b" or "llama-8b")
        
    Returns:
        List of Chunk objects
    """
    if not HF_AVAILABLE:
        raise ImportError("huggingface_hub required. Install with: pip install huggingface_hub")
    
    if cache_dir is None:
        cache_dir = RESULTS_DIR / "cache" / "traces"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check cache first
    cached_file = cache_dir / f"{problem.problem_idx}_{model}_chunks_labeled.json"
    
    if cached_file.exists() and not force_download:
        print(f"Loading cached trace for {problem.problem_idx} ({model})")
        with open(cached_file, 'r') as f:
            chunks_data = json.load(f)
        return [Chunk.from_dict(c) for c in chunks_data]
    
    # Download from HuggingFace
    print(f"Downloading trace for {problem.problem_idx} ({model}) from HuggingFace...")
    
    # Get the path in the dataset
    file_path = _get_trace_path_in_hf(problem.problem_idx, model)
    
    try:
        # Disable hf_transfer if not installed
        import os
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        
        local_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=file_path,
            repo_type="dataset",
        )
        
        # Load and cache
        with open(local_path, 'r') as f:
            chunks_data = json.load(f)
        
        # Save to our cache with simpler name
        with open(cached_file, 'w') as f:
            json.dump(chunks_data, f, indent=2)
        
        print(f"Downloaded and cached {len(chunks_data)} chunks for {problem.problem_idx}")
        return [Chunk.from_dict(c) for c in chunks_data]
        
    except Exception as e:
        raise RuntimeError(
            f"Could not download trace for {problem.problem_idx} ({model}).\n"
            f"Attempted path: {file_path}\n"
            f"Error: {e}\n"
            f"Check https://huggingface.co/datasets/uzaymacar/math-rollouts"
        )


def load_trace(problem: Problem, cache_dir: Optional[Path] = None) -> Trace:
    """
    Load a complete trace for a problem.
    
    Args:
        problem: Problem object
        cache_dir: Cache directory for downloaded traces
        
    Returns:
        Trace object with problem and chunks
    """
    chunks = download_trace_for_problem(problem, cache_dir)
    return Trace(problem=problem, chunks=chunks)


# =============================================================================
# STEERING VECTOR LOADING
# =============================================================================

def load_steering_vectors() -> Dict[str, SteeringVector]:
    """
    Load all steering vectors from the pre-computed file.
    
    Returns:
        Dictionary mapping behavior names to SteeringVector objects
    """
    if not STEERING_VECTORS_PATH.exists():
        raise FileNotFoundError(f"Steering vectors not found: {STEERING_VECTORS_PATH}")
    
    data = torch.load(STEERING_VECTORS_PATH, map_location='cpu')
    
    # Verify metadata
    expected_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    if data.get('model') != expected_model:
        print(f"Warning: Steering vectors were computed for {data.get('model')}, expected {expected_model}")
    
    vectors = {}
    for behavior, vec_data in data['vectors'].items():
        vectors[behavior] = SteeringVector(
            behavior=behavior,
            raw=vec_data['raw'],
            normalized=vec_data['normalized'],
            layer=vec_data['layer'],
            sample_count=vec_data['sample_count'],
        )
    
    return vectors


def get_steering_vector(behavior: str) -> SteeringVector:
    """
    Get a specific steering vector by behavior name.
    
    Args:
        behavior: One of: initializing, deduction, adding_knowledge, 
                  example_testing, uncertainty_estimation, backtracking
                  
    Returns:
        SteeringVector object
    """
    vectors = load_steering_vectors()
    
    if behavior not in vectors:
        available = list(vectors.keys())
        raise ValueError(f"Unknown behavior: {behavior}. Available: {available}")
    
    return vectors[behavior]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_problem_statistics(problem: Problem, trace: Trace) -> Dict[str, Any]:
    """
    Get statistics about a problem and its trace for logging.
    
    Returns:
        Dictionary with problem statistics
    """
    um_positions = trace.get_uncertainty_management_positions()
    
    return {
        "problem_idx": problem.problem_idx,
        "problem_type": problem.type,
        "problem_level": problem.level,
        "gt_answer": problem.gt_answer,
        "expected_accuracy": problem.accuracy,
        "num_chunks": trace.num_chunks,
        "num_uncertainty_management": len(um_positions),
        "uncertainty_management_positions": um_positions,
        "ci_scores": {
            i: trace.chunks[i].counterfactual_importance_kl 
            for i in um_positions
        },
    }


def validate_trace(trace: Trace) -> List[str]:
    """
    Validate a loaded trace and return any issues found.
    
    Returns:
        List of warning/error messages
    """
    issues = []
    
    # Check chunk indices are sequential
    expected_indices = list(range(len(trace.chunks)))
    actual_indices = [c.chunk_idx for c in trace.chunks]
    if actual_indices != expected_indices:
        issues.append(f"Non-sequential chunk indices: {actual_indices[:5]}...")
    
    # Check for required fields
    for i, chunk in enumerate(trace.chunks):
        if not chunk.chunk.strip():
            issues.append(f"Chunk {i} is empty")
        if not chunk.function_tags:
            issues.append(f"Chunk {i} has no function_tags")
    
    # Check for uncertainty_management chunks
    um_count = len(trace.get_uncertainty_management_positions())
    if um_count == 0:
        issues.append("No uncertainty_management chunks found in trace")
    
    return issues


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=== Testing Data Loading Module ===\n")
    
    # Test 1: Load selected problems
    print("1. Loading selected problems...")
    problems = load_selected_problems()
    print(f"   Loaded {len(problems)} problems")
    print(f"   First problem: {problems[0].problem_idx}")
    
    # Test 2: Load specific problem
    print("\n2. Loading problem by index (0)...")
    problem = load_problem_by_index(0)
    print(f"   Problem: {problem.problem_idx}")
    print(f"   Type: {problem.type}")
    print(f"   Answer: {problem.gt_answer}")
    print(f"   Accuracy: {problem.accuracy}")
    
    # Test 3: Load steering vectors
    print("\n3. Loading steering vectors...")
    vectors = load_steering_vectors()
    print(f"   Loaded {len(vectors)} behaviors: {list(vectors.keys())}")
    
    sv = get_steering_vector("backtracking")
    print(f"   Backtracking vector: layer={sv.layer}, shape={sv.normalized.shape}")
    
    # Test 4: Download trace
    print("\n4. Downloading trace for problem...")
    assert HF_AVAILABLE, "huggingface_hub is required for trace download!"
    
    trace = load_trace(problem)
    print(f"   Loaded {trace.num_chunks} chunks")
    um_positions = trace.get_uncertainty_management_positions()
    print(f"   Uncertainty management positions: {um_positions[:5]}..." if len(um_positions) > 5 else f"   Uncertainty management positions: {um_positions}")
    
    # Validate
    issues = validate_trace(trace)
    if issues:
        print(f"   Validation issues: {issues}")
    else:
        print("   Trace validation passed!")
    
    print("\n=== Data Loading Tests Complete ===")

