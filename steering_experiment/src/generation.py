"""
Generation Module for Steering Vectors Experiment

This module handles:
1. Loading the model and tokenizer
2. Batched generation with optional steering
3. Answer extraction from generated text
4. Tracking generation statistics (length, max_tokens hits, etc.)
"""

import torch
from torch import nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from steering_experiment.config import ModelConfig, ExperimentConfig, PROMPT_TEMPLATE
from steering_experiment.src.steering import SteeringManager, compute_steer_range_for_chunk
from steering_experiment.src.data_loading import Trace, Chunk, SteeringVector

# Import answer extraction from the main project
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import extract_boxed_answers, check_answer


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GenerationResult:
    """Result from a single generation."""
    text: str  # Full generated text
    answer: Optional[str]  # Extracted answer (if found)
    is_correct: Optional[bool]  # Whether answer matches ground truth
    num_tokens: int  # Number of tokens generated
    hit_max_tokens: bool  # Whether generation hit the max_tokens limit
    

@dataclass
class PositionRollouts:
    """All rollouts for a single position."""
    position_idx: int
    chunk: Chunk
    prefix: str  # The prompt prefix used
    steer_range: Tuple[int, int]  # Token positions that were steered
    
    # Results by alpha value
    results_by_alpha: Dict[float, List[GenerationResult]] = field(default_factory=dict)
    
    def get_answers(self, alpha: float) -> List[Optional[str]]:
        """Get all extracted answers for a given alpha."""
        return [r.answer for r in self.results_by_alpha.get(alpha, [])]
    
    def get_accuracy(self, alpha: float, ground_truth: str) -> float:
        """Compute accuracy for a given alpha."""
        answers = self.get_answers(alpha)
        if not answers:
            return 0.0
        correct = sum(1 for a in answers if a is not None and check_answer(a, ground_truth))
        return correct / len(answers)
    
    def get_answer_distribution(self, alpha: float) -> Dict[str, float]:
        """Get probability distribution over answers for a given alpha."""
        answers = self.get_answers(alpha)
        if not answers:
            return {}
        
        # Count occurrences (treat None as "NO_ANSWER")
        counts: Dict[str, int] = {}
        for a in answers:
            key = a if a is not None else "NO_ANSWER"
            counts[key] = counts.get(key, 0) + 1
        
        # Normalize to probabilities
        total = len(answers)
        return {k: v / total for k, v in counts.items()}


@dataclass
class GenerationStats:
    """Statistics from generation run."""
    total_generations: int = 0
    hit_max_tokens: int = 0
    extraction_failures: int = 0
    total_tokens_generated: int = 0
    
    @property
    def max_tokens_rate(self) -> float:
        """Fraction of generations that hit max_tokens."""
        if self.total_generations == 0:
            return 0.0
        return self.hit_max_tokens / self.total_generations
    
    @property
    def extraction_success_rate(self) -> float:
        """Fraction of generations with successful answer extraction."""
        if self.total_generations == 0:
            return 0.0
        return 1.0 - (self.extraction_failures / self.total_generations)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_and_tokenizer(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    device_map: str = "auto",
    torch_dtype: str = "float16",
    use_quantization: bool = False,
    use_flash_attention: bool = True,
) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Load the model and tokenizer.
    
    Args:
        model_name: HuggingFace model name
        device_map: Device mapping strategy ("auto", "cuda:0", etc.)
        torch_dtype: Data type ("float16", "bfloat16", "float32")
        use_quantization: Whether to use 4-bit quantization
        
    Returns:
        (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    print(f"  Device map: {device_map}")
    print(f"  Dtype: {torch_dtype}")
    print(f"  Quantization: {use_quantization}")
    print(f"  Flash Attention 2: {use_flash_attention}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)
    
    # Load model
    if use_quantization:
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
    else:
        # Build kwargs for model loading
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        
        # Add Flash Attention 2 if requested
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
    
    model.eval()
    print(f"Model loaded successfully")
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    print(f"  Model device: {next(model.parameters()).device}")
    
    return model, tokenizer


# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================

def generate_single(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    ground_truth: Optional[str] = None,
) -> GenerationResult:
    """
    Generate a single completion.
    
    Args:
        model: The model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        ground_truth: Ground truth answer for correctness check
        
    Returns:
        GenerationResult
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode
    output_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    num_tokens = len(output_tokens)
    hit_max = num_tokens >= max_new_tokens - 1
    
    # Extract answer - look after </think> tag if present
    # DeepSeek-R1 outputs </think> before the final answer
    if "</think>" in generated_text:
        answer_section = generated_text.split("</think>")[-1]
    else:
        answer_section = generated_text
    
    extracted = extract_boxed_answers(answer_section)
    answer = extracted[0] if extracted and extracted[0] else None
    
    # Check correctness
    is_correct = None
    if answer is not None and ground_truth is not None:
        is_correct = check_answer(answer, ground_truth)
    
    return GenerationResult(
        text=generated_text,
        answer=answer,
        is_correct=is_correct,
        num_tokens=num_tokens,
        hit_max_tokens=hit_max,
    )


def generate_batch(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    ground_truth: Optional[str] = None,
) -> List[GenerationResult]:
    """
    Generate completions for a batch of prompts.
    
    Args:
        model: The model
        tokenizer: Tokenizer
        prompts: List of input prompts (should all be the same for rollouts)
        max_new_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        ground_truth: Ground truth answer for correctness check
        
    Returns:
        List of GenerationResult
    """
    # Tokenize all prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_lengths = [
        (inputs["attention_mask"][i] == 1).sum().item()
        for i in range(len(prompts))
    ]
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Process each output
    results = []
    for i, (output_ids, input_len) in enumerate(zip(outputs, input_lengths)):
        # Get generated tokens (after input)
        output_tokens = output_ids[input_len:]
        generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        num_tokens = len(output_tokens)
        hit_max = num_tokens >= max_new_tokens - 1
        
        # Extract answer - look after </think> tag if present
        if "</think>" in generated_text:
            answer_section = generated_text.split("</think>")[-1]
        else:
            answer_section = generated_text
        
        extracted = extract_boxed_answers(answer_section)
        answer = extracted[0] if extracted and extracted[0] else None
        
        # Check correctness
        is_correct = None
        if answer is not None and ground_truth is not None:
            is_correct = check_answer(answer, ground_truth)
        
        results.append(GenerationResult(
            text=generated_text,
            answer=answer,
            is_correct=is_correct,
            num_tokens=num_tokens,
            hit_max_tokens=hit_max,
        ))
    
    return results


def generate_rollouts_for_position(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    trace: Trace,
    position_idx: int,
    steering_manager: Optional[SteeringManager],
    alpha_values: List[float],
    n_rollouts: int,
    batch_size: int,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    prompt_template: str = PROMPT_TEMPLATE,
) -> PositionRollouts:
    """
    Generate rollouts for a single position with different steering strengths.
    
    Args:
        model: The model
        tokenizer: Tokenizer
        trace: The reasoning trace
        position_idx: Which chunk position to steer at
        steering_manager: SteeringManager (can be None if all alphas are 0)
        alpha_values: List of steering strengths to test (e.g., [0.0, -1.0])
        n_rollouts: Number of rollouts per alpha value
        batch_size: Batch size for generation
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        prompt_template: Template for building prompts
        
    Returns:
        PositionRollouts with results for all alpha values
    """
    chunk = trace.chunks[position_idx]
    problem_text = trace.problem.problem_text
    ground_truth = trace.problem.gt_answer
    
    # Build prefix (includes the target chunk)
    prefix_chunks = [c.chunk for c in trace.chunks[:position_idx]]
    target_chunk = chunk.chunk
    prefix_with_chunk = " ".join(prefix_chunks + [target_chunk]) if prefix_chunks else target_chunk
    
    # Build the full prompt
    prompt = prompt_template.format(problem=problem_text, prefix=prefix_with_chunk)
    
    # Compute steer range (token positions of the target chunk)
    steer_range = compute_steer_range_for_chunk(
        problem_text, prefix_chunks, target_chunk, tokenizer, prompt_template
    )
    
    # Initialize result
    position_rollouts = PositionRollouts(
        position_idx=position_idx,
        chunk=chunk,
        prefix=prompt,
        steer_range=steer_range,
    )
    
    # Generate rollouts for each alpha
    for alpha in alpha_values:
        results = []
        
        # Reset steering stats and activate if needed
        if steering_manager is not None:
            steering_manager.reset_stats()
            if alpha != 0.0:
                steering_manager.activate(alpha=alpha, steer_range=steer_range)
        
        # Generate in batches
        for batch_start in range(0, n_rollouts, batch_size):
            current_batch_size = min(batch_size, n_rollouts - batch_start)
            batch_prompts = [prompt] * current_batch_size
            
            batch_results = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                ground_truth=ground_truth,
            )
            results.extend(batch_results)
        
        # Verify steering was applied (for non-zero alpha)
        if steering_manager is not None:
            if alpha != 0.0 and not steering_manager.steering_applied:
                print(f"WARNING: Steering was NOT applied for position {position_idx}, alpha={alpha}")
                print(f"  Steer range: {steer_range}, Hook call count: {steering_manager.hook_call_count}")
            steering_manager.deactivate()
        
        position_rollouts.results_by_alpha[alpha] = results
    
    return position_rollouts


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_generation_stats(rollouts: List[PositionRollouts]) -> GenerationStats:
    """Compute statistics across all rollouts."""
    stats = GenerationStats()
    
    for position in rollouts:
        for alpha, results in position.results_by_alpha.items():
            for result in results:
                stats.total_generations += 1
                stats.total_tokens_generated += result.num_tokens
                if result.hit_max_tokens:
                    stats.hit_max_tokens += 1
                if result.answer is None:
                    stats.extraction_failures += 1
    
    return stats


def save_rollouts(rollouts: List[PositionRollouts], output_path: Path, save_full_cot: bool = True) -> None:
    """
    Save rollouts to disk.
    
    Args:
        rollouts: List of PositionRollouts
        output_path: Directory to save to
        save_full_cot: Whether to save full generated text
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    for position in rollouts:
        pos_dir = output_path / f"position_{position.position_idx}"
        pos_dir.mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            "position_idx": position.position_idx,
            "chunk_text": position.chunk.chunk,
            "function_tags": position.chunk.function_tags,
            "ci_kl": position.chunk.ci_kl,
            "ci_accuracy": position.chunk.ci_accuracy,
            "steer_range": position.steer_range,
        }
        with open(pos_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save results for each alpha
        for alpha, results in position.results_by_alpha.items():
            alpha_data = []
            for i, result in enumerate(results):
                item = {
                    "rollout_idx": i,
                    "answer": result.answer,
                    "is_correct": result.is_correct,
                    "num_tokens": result.num_tokens,
                    "hit_max_tokens": result.hit_max_tokens,
                }
                if save_full_cot:
                    item["text"] = result.text
                alpha_data.append(item)
            
            alpha_str = f"alpha_{alpha}".replace(".", "p").replace("-", "neg")
            with open(pos_dir / f"{alpha_str}.json", "w") as f:
                json.dump(alpha_data, f, indent=2)


def load_rollouts(input_path: Path) -> List[PositionRollouts]:
    """
    Load rollouts from disk.
    
    Args:
        input_path: Directory containing saved rollouts
        
    Returns:
        List of PositionRollouts
    """
    rollouts = []
    
    for pos_dir in sorted(input_path.glob("position_*")):
        # Load metadata
        with open(pos_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Create chunk object (partial - only has what we saved)
        chunk = Chunk(
            chunk=metadata["chunk_text"],
            chunk_idx=metadata["position_idx"],
            function_tags=metadata["function_tags"],
            depends_on=[],
            accuracy=0.0,
            counterfactual_importance_kl=metadata["ci_kl"],
            counterfactual_importance_accuracy=metadata["ci_accuracy"],
            resampling_importance_kl=0.0,
            resampling_importance_accuracy=0.0,
            forced_importance_kl=0.0,
            forced_importance_accuracy=0.0,
            different_trajectories_fraction=0.0,
            overdeterminedness=0.0,
            summary="",
        )
        
        position = PositionRollouts(
            position_idx=metadata["position_idx"],
            chunk=chunk,
            prefix="",  # Not saved
            steer_range=tuple(metadata["steer_range"]),
        )
        
        # Load results for each alpha
        for alpha_file in pos_dir.glob("alpha_*.json"):
            # Parse alpha from filename
            alpha_str = alpha_file.stem.replace("alpha_", "")
            alpha_str = alpha_str.replace("neg", "-").replace("p", ".")
            alpha = float(alpha_str)
            
            with open(alpha_file, "r") as f:
                alpha_data = json.load(f)
            
            results = []
            for item in alpha_data:
                results.append(GenerationResult(
                    text=item.get("text", ""),
                    answer=item.get("answer"),
                    is_correct=item.get("is_correct"),
                    num_tokens=item.get("num_tokens", 0),
                    hit_max_tokens=item.get("hit_max_tokens", False),
                ))
            
            position.results_by_alpha[alpha] = results
        
        rollouts.append(position)
    
    return rollouts


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=== Testing Generation Module ===\n")
    
    # Test imports
    print("1. Testing imports...")
    print("   All imports successful")
    
    # Test GenerationResult dataclass
    print("\n2. Testing GenerationResult...")
    result = GenerationResult(
        text="The answer is \\boxed{42}",
        answer="42",
        is_correct=True,
        num_tokens=10,
        hit_max_tokens=False,
    )
    print(f"   Created: answer={result.answer}, correct={result.is_correct}")
    
    # Test PositionRollouts
    print("\n3. Testing PositionRollouts...")
    from steering_experiment.src.data_loading import Chunk
    
    dummy_chunk = Chunk(
        chunk="Let me think about this.",
        chunk_idx=0,
        function_tags=["problem_setup"],
        depends_on=[],
        accuracy=0.7,
        counterfactual_importance_kl=0.1,
        counterfactual_importance_accuracy=0.05,
        resampling_importance_kl=0.0,
        resampling_importance_accuracy=0.0,
        forced_importance_kl=0.0,
        forced_importance_accuracy=0.0,
        different_trajectories_fraction=0.0,
        overdeterminedness=0.0,
        summary="thinking",
    )
    
    pos_rollouts = PositionRollouts(
        position_idx=0,
        chunk=dummy_chunk,
        prefix="Problem: ...",
        steer_range=(10, 20),
    )
    pos_rollouts.results_by_alpha[0.0] = [result]
    
    print(f"   Position: {pos_rollouts.position_idx}")
    print(f"   Answers for alpha=0.0: {pos_rollouts.get_answers(0.0)}")
    
    # Test GenerationStats
    print("\n4. Testing GenerationStats...")
    stats = compute_generation_stats([pos_rollouts])
    print(f"   Total generations: {stats.total_generations}")
    print(f"   Max tokens rate: {stats.max_tokens_rate:.2%}")
    
    print("\n=== Generation Module Tests Complete ===")
    print("\nNote: Full generation test requires loading model (skipped).")

