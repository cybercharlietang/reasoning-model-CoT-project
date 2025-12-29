"""
Steering Module for Steering Vectors Experiment

This module implements hook-based activation steering for transformer models.
The SteeringManager class allows injecting steering vectors at specific layers
and token positions during model forward passes.

Key concepts:
- Steering vectors are added to the residual stream at a target layer
- We steer at ALL tokens of a target chunk (not just the last token)
- Steering only happens on the first forward pass (full prefix), not during
  autoregressive generation of new tokens
"""

import torch
from torch import nn
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class SteeringConfig:
    """Configuration for a steering intervention."""
    behavior: str  # Name of behavior (e.g., "backtracking")
    layer: int  # Layer to intervene at
    alpha: float  # Steering strength (can be negative)
    steer_start: int  # Start token position (inclusive)
    steer_end: int  # End token position (inclusive)
    
    @property
    def steer_range(self) -> Tuple[int, int]:
        return (self.steer_start, self.steer_end)


class SteeringManager:
    """
    Manages activation steering via forward hooks.
    
    This class registers hooks on transformer layers to inject steering vectors
    into the residual stream during forward passes.
    
    Usage:
        steering_manager = SteeringManager(model, steering_vector, layer=29)
        steering_manager.register_hook()
        
        # For generation with steering:
        steering_manager.activate(alpha=-1.0, steer_range=(10, 25))
        outputs = model.generate(...)
        steering_manager.deactivate()
        
        # Cleanup when done:
        steering_manager.remove_hook()
    """
    
    def __init__(
        self,
        model: nn.Module,
        steering_vector: torch.Tensor,
        layer: int,
        use_normalized: bool = True,
    ):
        """
        Initialize the SteeringManager.
        
        Args:
            model: HuggingFace transformer model
            steering_vector: The steering vector tensor (hidden_dim,)
            layer: Layer index to apply steering at
            use_normalized: Whether to use normalized vector (recommended)
        """
        self.model = model
        self.steering_vector = steering_vector.clone()
        self.layer = layer
        self.use_normalized = use_normalized
        
        # Hook state
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.is_active = False
        self.alpha = 0.0
        self.steer_start = 0
        self.steer_end = 0
        
        # For debugging/sanity checks
        self.hook_call_count = 0
        self.last_steered_positions: List[int] = []
        self.steering_applied = False
        
    def _get_layer_module(self) -> nn.Module:
        """Get the layer module to hook."""
        # Handle different model architectures
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Standard HuggingFace format (e.g., LlamaForCausalLM, Qwen2ForCausalLM)
            return self.model.model.layers[self.layer]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2 style
            return self.model.transformer.h[self.layer]
        else:
            raise ValueError(
                f"Unknown model architecture. Cannot find layer {self.layer}. "
                f"Model type: {type(self.model)}"
            )
    
    def _create_hook(self) -> Callable:
        """
        Create the forward hook function.
        
        The hook adds the steering vector to the residual stream at specified
        token positions. It only steers when:
        1. Steering is active (self.is_active = True)
        2. The sequence length is > 1 (first forward pass with full prefix)
        
        During autoregressive generation (seq_len == 1), no steering is applied.
        """
        def hook_fn(module, input, output):
            self.hook_call_count += 1
            
            if not self.is_active:
                return output
            
            # Output is typically a tuple: (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None
            
            # hidden_states shape: (batch_size, seq_len, hidden_dim)
            seq_len = hidden_states.shape[1]
            
            # Only steer on the first forward pass (when we have the full prefix)
            # During autoregressive generation, seq_len == 1
            if seq_len == 1:
                # Autoregressive step - don't steer
                return output
            
            # Determine which positions to steer
            # Clamp to valid range
            start = max(0, self.steer_start)
            end = min(seq_len - 1, self.steer_end)
            
            if start > end:
                # Invalid range, skip steering
                return output
            
            # Clone hidden states to avoid in-place modification issues
            hidden_states = hidden_states.clone()
            
            # Move steering vector to correct device and dtype
            steering_vec = self.steering_vector.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            
            # Apply steering to all positions in range at once (more efficient)
            hidden_states[:, start:end+1, :] += self.alpha * steering_vec
            
            # Record for debugging
            self.last_steered_positions = list(range(start, end + 1))
            self.steering_applied = True
            
            # Return modified output
            if rest is not None:
                return (hidden_states,) + rest
            else:
                return hidden_states
        
        return hook_fn
    
    def register_hook(self) -> None:
        """Register the forward hook on the target layer."""
        if self.hook_handle is not None:
            print("Warning: Hook already registered. Removing old hook first.")
            self.remove_hook()
        
        layer_module = self._get_layer_module()
        self.hook_handle = layer_module.register_forward_hook(self._create_hook())
        print(f"Registered steering hook on layer {self.layer}")
    
    def remove_hook(self) -> None:
        """Remove the forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            print(f"Removed steering hook from layer {self.layer}")
    
    def activate(
        self,
        alpha: float,
        steer_range: Tuple[int, int],
    ) -> None:
        """
        Activate steering for the next forward pass(es).
        
        Args:
            alpha: Steering strength (positive = amplify, negative = suppress)
            steer_range: (start, end) token positions to steer (inclusive)
        """
        self.is_active = True
        self.alpha = alpha
        self.steer_start, self.steer_end = steer_range
        self.steering_applied = False
        self.last_steered_positions = []
    
    def deactivate(self) -> None:
        """Deactivate steering."""
        self.is_active = False
        self.alpha = 0.0
    
    @contextmanager
    def steer(self, alpha: float, steer_range: Tuple[int, int]):
        """
        Context manager for steering.
        
        Usage:
            with steering_manager.steer(alpha=-1.0, steer_range=(10, 25)):
                outputs = model.generate(...)
        """
        self.activate(alpha, steer_range)
        try:
            yield
        finally:
            self.deactivate()
    
    def get_stats(self) -> dict:
        """Get debugging statistics."""
        return {
            "hook_registered": self.hook_handle is not None,
            "is_active": self.is_active,
            "alpha": self.alpha,
            "steer_range": (self.steer_start, self.steer_end),
            "hook_call_count": self.hook_call_count,
            "last_steered_positions": self.last_steered_positions,
            "steering_applied": self.steering_applied,
            "layer": self.layer,
            "vector_norm": self.steering_vector.norm().item(),
        }
    
    def reset_stats(self) -> None:
        """Reset debugging statistics."""
        self.hook_call_count = 0
        self.last_steered_positions = []
        self.steering_applied = False


def compute_steer_range_for_chunk(
    problem_text: str,
    prefix_chunks: List[str],
    target_chunk: str,
    tokenizer,
    prompt_template: str = "Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem} Solution: \n<think>\n{prefix}",
) -> Tuple[int, int]:
    """
    Compute the token range for a target chunk within a full prompt.
    
    This function determines which token positions correspond to the target chunk,
    accounting for the problem text and prompt template.
    
    NOTE: We use the raw prompt format (not chat template) for consistency with
    the original Thought Anchors codebase and the steering vector training data.
    
    Args:
        problem_text: The math problem text
        prefix_chunks: List of chunks BEFORE the target chunk
        target_chunk: The chunk to steer at
        tokenizer: HuggingFace tokenizer
        prompt_template: Template for the full prompt (raw format, not chat template)
        
    Returns:
        (start_token_idx, end_token_idx) - Token positions of the target chunk (inclusive)
    """
    # Build the prefix text (chunks before target)
    prefix_text = " ".join(prefix_chunks) if prefix_chunks else ""
    
    # Build the full prefix including target chunk
    prefix_with_target = prefix_text + (" " if prefix_text else "") + target_chunk
    
    # Format prompts
    prompt_before = prompt_template.format(problem=problem_text, prefix=prefix_text)
    prompt_with_target = prompt_template.format(problem=problem_text, prefix=prefix_with_target)
    
    # Tokenize
    tokens_before = tokenizer.encode(prompt_before, add_special_tokens=True)
    tokens_with_target = tokenizer.encode(prompt_with_target, add_special_tokens=True)
    
    # SANITY CHECK: target chunk should add tokens
    assert len(tokens_with_target) > len(tokens_before), (
        f"Target chunk added no tokens! "
        f"Before={len(tokens_before)}, After={len(tokens_with_target)}, "
        f"Target chunk='{target_chunk[:50]}...'"
    )
    
    # The target chunk tokens start after the prefix
    start_idx = len(tokens_before)
    end_idx = len(tokens_with_target) - 1  # Inclusive
    
    # SANITY CHECK: verify the decoded tokens match the target chunk
    # (allowing for tokenization artifacts like leading/trailing spaces)
    target_tokens = tokens_with_target[start_idx:end_idx + 1]
    decoded = tokenizer.decode(target_tokens)
    
    # The decoded text should contain the target chunk (with possible space prefix)
    assert target_chunk.strip() in decoded or decoded.strip() in target_chunk, (
        f"Token range mismatch! "
        f"Expected chunk to contain '{target_chunk[:30]}...', "
        f"but got '{decoded[:50]}...'"
    )
    
    return start_idx, end_idx


def verify_steering_hook(
    model: nn.Module,
    steering_manager: SteeringManager,
    tokenizer,
    test_prompt: str = "Test prompt for verification.",
) -> dict:
    """
    Verify that the steering hook is working correctly.
    
    This function runs a forward pass with and without steering and verifies
    that activations differ as expected.
    
    Args:
        model: The model
        steering_manager: SteeringManager instance (with hook registered)
        tokenizer: Tokenizer
        test_prompt: A test prompt
        
    Returns:
        Dictionary with verification results
        
    Raises:
        AssertionError: If hook is not registered
        RuntimeError: If steering is not working as expected
    """
    assert steering_manager.hook_handle is not None, "Hook not registered!"
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    seq_len = inputs["input_ids"].shape[1]
    
    # Baseline forward pass (no steering)
    steering_manager.deactivate()
    with torch.no_grad():
        baseline_output = model(**inputs, output_hidden_states=True)
    baseline_hidden = baseline_output.hidden_states[steering_manager.layer + 1]
    
    # Steered forward pass
    steering_manager.activate(alpha=1.0, steer_range=(0, seq_len - 1))
    with torch.no_grad():
        steered_output = model(**inputs, output_hidden_states=True)
    steered_hidden = steered_output.hidden_states[steering_manager.layer + 1]
    steering_manager.deactivate()
    
    # Check that activations differ
    diff = (steered_hidden - baseline_hidden).abs().mean().item()
    activations_differ = diff > 1e-6
    
    assert activations_differ, f"Steering did not change activations! Diff={diff}"
    assert steering_manager.steering_applied, "Steering was not applied!"
    
    return {
        "hook_registered": True,
        "baseline_pass": True,
        "steered_pass": True,
        "activations_differ": True,
        "mean_activation_diff": diff,
        "steering_applied": True,
    }


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=== Testing Steering Module ===\n")
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from steering_experiment.src.data_loading import get_steering_vector
    from transformers import AutoTokenizer
    
    # Test 1: Load steering vector
    print("1. Loading steering vector...")
    sv = get_steering_vector("backtracking")
    print(f"   Loaded: behavior={sv.behavior}, layer={sv.layer}, shape={sv.normalized.shape}")
    print(f"   Vector norm: {sv.normalized.norm().item():.4f}")
    
    # Test 2: Create SteeringManager (without model - just test instantiation)
    print("\n2. Testing SteeringManager class...")
    dummy_vector = torch.randn(5120)
    print("   SteeringManager class imports correctly")
    print("   (Full test requires loading model - skipped in unit test)")
    
    # Test 3: Test compute_steer_range_for_chunk
    print("\n3. Testing compute_steer_range_for_chunk...")
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        trust_remote_code=True
    )
    
    problem = "What is 2 + 2?"
    prefix_chunks = ["Let me think about this.", "First, I'll add the numbers."]
    target_chunk = "So 2 + 2 = 4."
    
    start, end = compute_steer_range_for_chunk(
        problem, prefix_chunks, target_chunk, tokenizer
    )
    
    print(f"   Problem: {problem}")
    print(f"   Prefix chunks: {len(prefix_chunks)}")
    print(f"   Target chunk: '{target_chunk}'")
    print(f"   Steer range: ({start}, {end}) - {end - start + 1} tokens")
    
    print("\n=== Steering Module Tests Complete ===")

