"""
SCOPE Router Configuration Module
"""

from .model_pools import (
    PRICING,
    MODEL_NAME_TO_OPENROUTER_ID,
    OPENROUTER_ID_TO_MODEL_NAME,
    DEFAULT_POOL,
    REASONING_POOL,
    HIGH_BUDGET_POOL,
    LOW_BUDGET_POOL,
    FULL_POOL,
    AVAILABLE_POOLS,
    get_pool,
    load_custom_pool,
    get_model_pricing,
    calculate_cost,
)

__all__ = [
    "PRICING",
    "MODEL_NAME_TO_OPENROUTER_ID",
    "OPENROUTER_ID_TO_MODEL_NAME",
    "DEFAULT_POOL",
    "REASONING_POOL",
    "HIGH_BUDGET_POOL",
    "LOW_BUDGET_POOL",
    "FULL_POOL",
    "AVAILABLE_POOLS",
    "get_pool",
    "load_custom_pool",
    "get_model_pricing",
    "calculate_cost",
]
