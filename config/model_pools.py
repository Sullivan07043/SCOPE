"""
Model Pool Configurations for SCOPE Router

This module defines various model pools for different use cases:
- DEFAULT_POOL: General purpose routing (7 models, balanced cost/performance)
- REASONING_POOL: Optimized for complex reasoning tasks
- HIGH_BUDGET_POOL: Premium models for maximum accuracy
- LOW_BUDGET_POOL: Cost-efficient models for budget-conscious users
- FULL_POOL: All 13 in-distribution models

Each pool contains OpenRouter model IDs and their pricing information.
"""

# OpenRouter pricing table (USD per 1M tokens)
# Source: OpenRouter API pricing as of 2025
PRICING = {
    # High-cost models
    "tngtech/deepseek-r1t2-chimera": {"input": 0.3, "output": 1.2},
    "amazon/nova-2-lite-v1": {"input": 0.3, "output": 2.5},
    "qwen/qwen3-235b-a22b": {"input": 0.18, "output": 0.54},
    
    # Medium-cost models
    "qwen/qwen3-14b": {"input": 0.05, "output": 0.22},
    
    # Low-cost models
    "openai/gpt-oss-20b": {"input": 0.03, "output": 0.14},
    "meta-llama/llama-3.3-70b-instruct": {"input": 0.1, "output": 0.32},
    "google/gemma-3-27b-it": {"input": 0.04, "output": 0.15},
    "meta-llama/llama-3.1-8b-instruct": {"input": 0.02, "output": 0.03},
    "google/gemma-3-12b-it": {"input": 0.03, "output": 0.1},
    "nvidia/nemotron-nano-9b-v2": {"input": 0.04, "output": 0.16},
    "mistralai/ministral-8b": {"input": 0.1, "output": 0.1},
    "google/gemma-3-4b-it": {"input": 0.017, "output": 0.068},
    "mistralai/ministral-3b": {"input": 0.04, "output": 0.04},
}

# Mapping from short names (used in dataset) to OpenRouter IDs
MODEL_NAME_TO_OPENROUTER_ID = {
    "deepseek-r1t2-chimera": "tngtech/deepseek-r1t2-chimera",
    "nova-2-lite-v1": "amazon/nova-2-lite-v1",
    "qwen3-235b-a22b": "qwen/qwen3-235b-a22b",
    "qwen3-14b": "qwen/qwen3-14b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "llama-3-3-70b": "meta-llama/llama-3.3-70b-instruct",
    "gemma-3-27b": "google/gemma-3-27b-it",
    "llama-3-1-8b": "meta-llama/llama-3.1-8b-instruct",
    "gemma-3-12b": "google/gemma-3-12b-it",
    "nemotron-nano-9b-v2": "nvidia/nemotron-nano-9b-v2",
    "ministral-8b": "mistralai/ministral-8b",
    "gemma-3-4b": "google/gemma-3-4b-it",
    "ministral-3b": "mistralai/ministral-3b",
}

# Reverse mapping
OPENROUTER_ID_TO_MODEL_NAME = {v: k for k, v in MODEL_NAME_TO_OPENROUTER_ID.items()}


# ============================================================================
# MODEL POOLS
# ============================================================================

# Default General Purpose Pool (7 models)
# Balanced selection across cost tiers for general routing tasks
DEFAULT_POOL = {
    # High-cost, high-performance (3 models)
    "tngtech/deepseek-r1t2-chimera",    # 73.60% accuracy, $0.719 avg cost
    "amazon/nova-2-lite-v1",             # 60.71% accuracy, $2.386 avg cost
    "qwen/qwen3-235b-a22b",              # 58.96% accuracy, $0.619 avg cost
    # Medium-cost (1 model)
    "qwen/qwen3-14b",                    # 58.73% accuracy, $0.294 avg cost
    # Low-cost (3 models)
    "openai/gpt-oss-20b",                # 59.13% accuracy, $0.082 avg cost
    "meta-llama/llama-3.3-70b-instruct", # 54.76% accuracy, $0.050 avg cost
    "google/gemma-3-27b-it",             # 53.17% accuracy, $0.027 avg cost
}

# Reasoning Task Pool (5 models)
# Optimized for complex reasoning, math, and logic tasks
REASONING_POOL = {
    "tngtech/deepseek-r1t2-chimera",    # Best for complex reasoning
    "qwen/qwen3-235b-a22b",              # Strong reasoning capabilities
    "qwen/qwen3-14b",                    # Good balance for reasoning
    "openai/gpt-oss-20b",                # Solid reasoning at low cost
    "meta-llama/llama-3.3-70b-instruct", # Large context, good reasoning
}

# High Budget Pool (4 models)
# Premium models for maximum accuracy when cost is not a concern
HIGH_BUDGET_POOL = {
    "tngtech/deepseek-r1t2-chimera",    # Highest accuracy
    "amazon/nova-2-lite-v1",             # Premium performance
    "qwen/qwen3-235b-a22b",              # Large model, high quality
    "qwen/qwen3-14b",                    # Medium-high tier
}

# Low Budget Pool (5 models)
# Cost-efficient models for budget-conscious applications
LOW_BUDGET_POOL = {
    "openai/gpt-oss-20b",                # Best cost-efficiency ratio
    "meta-llama/llama-3.3-70b-instruct", # Low cost, decent performance
    "google/gemma-3-27b-it",             # Very low cost
    "google/gemma-3-12b-it",             # Ultra low cost
    "google/gemma-3-4b-it",              # Minimal cost option
}

# Full Pool (13 models)
# All in-distribution models from SCOPE-60K dataset
FULL_POOL = {
    "tngtech/deepseek-r1t2-chimera",
    "amazon/nova-2-lite-v1",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen3-14b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemma-3-27b-it",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemma-3-12b-it",
    "nvidia/nemotron-nano-9b-v2",
    "mistralai/ministral-8b",
    "google/gemma-3-4b-it",
    "mistralai/ministral-3b",
}


# Pool registry for easy access
AVAILABLE_POOLS = {
    "default": DEFAULT_POOL,
    "reasoning": REASONING_POOL,
    "high_budget": HIGH_BUDGET_POOL,
    "low_budget": LOW_BUDGET_POOL,
    "full": FULL_POOL,
}


def get_pool(pool_name: str) -> set:
    """
    Get a model pool by name.
    
    Args:
        pool_name: One of 'default', 'reasoning', 'high_budget', 'low_budget', 'full'
    
    Returns:
        Set of OpenRouter model IDs
    """
    if pool_name not in AVAILABLE_POOLS:
        raise ValueError(f"Unknown pool: {pool_name}. Available: {list(AVAILABLE_POOLS.keys())}")
    return AVAILABLE_POOLS[pool_name]


def load_custom_pool(filepath: str) -> set:
    """
    Load a custom model pool from a text file.
    
    The file should contain one OpenRouter model ID per line.
    Lines starting with '#' are treated as comments.
    
    Args:
        filepath: Path to the text file
    
    Returns:
        Set of OpenRouter model IDs
    
    Example file content:
        # My custom model pool
        qwen/qwen3-14b
        openai/gpt-oss-20b
        google/gemma-3-27b-it
    """
    models = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                models.add(line)
    
    if not models:
        raise ValueError(f"No models found in {filepath}")
    
    return models


def get_model_pricing(model_id: str) -> dict:
    """
    Get pricing information for a model.
    
    Args:
        model_id: OpenRouter model ID
    
    Returns:
        Dict with 'input' and 'output' prices per 1M tokens
    """
    return PRICING.get(model_id, {"input": 0.0, "output": 0.0})


def calculate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate the cost of a single API call.
    
    Args:
        model_id: OpenRouter model ID
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
    
    Returns:
        Cost in USD
    """
    pricing = get_model_pricing(model_id)
    cost = (prompt_tokens / 1_000_000) * pricing["input"] + \
           (completion_tokens / 1_000_000) * pricing["output"]
    return cost


def print_pool_info(pool_name: str = None):
    """Print information about model pools."""
    if pool_name:
        pools = {pool_name: AVAILABLE_POOLS[pool_name]}
    else:
        pools = AVAILABLE_POOLS
    
    for name, models in pools.items():
        print(f"\n{'='*60}")
        print(f"Pool: {name.upper()} ({len(models)} models)")
        print('='*60)
        for model_id in sorted(models):
            pricing = get_model_pricing(model_id)
            print(f"  {model_id}")
            print(f"    Input: ${pricing['input']}/1M tokens, Output: ${pricing['output']}/1M tokens")


if __name__ == "__main__":
    print_pool_info()
