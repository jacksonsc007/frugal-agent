"""
Configuration package for the frugal-agent project.
"""

from .training_config import (
    PROXY_CONFIG,
    MODEL_CONFIG,
    LORA_CONFIG,
    TRAINING_CONFIG,
    DATASET_CONFIG,
    get_experiment_config,
    get_grpo_config,
    get_model_config,
    get_peft_config,
)

__all__ = [
    "PROXY_CONFIG",
    "MODEL_CONFIG",
    "LORA_CONFIG",
    "TRAINING_CONFIG",
    "DATASET_CONFIG",
    "get_experiment_config",
    "get_grpo_config",
    "get_model_config",
    "get_peft_config",
] 