"""
Steering Experiment Source Modules

This package contains the core modules for the steering vectors experiment:
- data_loading: Load problems, traces, and steering vectors
- steering: SteeringManager for hook-based interventions
- generation: Batched rollout generation
- metrics: KL, JS divergence, accuracy computation
"""

from .data_loading import (
    load_selected_problems,
    load_problem_by_index,
    download_trace_for_problem,
    load_steering_vectors,
    get_steering_vector,
    load_trace,
    Problem,
    Chunk,
    Trace,
    SteeringVector,
)

from .steering import (
    SteeringManager,
    SteeringConfig,
    compute_steer_range_for_chunk,
    verify_steering_hook,
)

from .generation import (
    load_model_and_tokenizer,
    generate_single,
    generate_batch,
    generate_rollouts_for_position,
    compute_generation_stats,
    save_rollouts,
    load_rollouts,
    GenerationResult,
    PositionRollouts,
    GenerationStats,
)

