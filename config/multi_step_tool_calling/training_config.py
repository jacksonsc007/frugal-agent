"""
Training configuration for multi-step tool calling with Unsloth.
This file contains all the hyperparameters and configuration settings
extracted from multi_step_tool_calling-unsloth.py
"""

import os
from typing import List

# ========== Environment Configuration ==========
# Proxy settings (if needed)
PROXY_CONFIG = {
    "https_proxy": "http://192.168.1.22:7890"
}

# ========== Model Configuration ==========
MODEL_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "model_size": "7B",
    "max_seq_length": 4096 + 1024,  # 5120
    "max_prompt_length": 2048,
    "max_completion_length": (4096 + 1024) - 2048,  # 3072
    "load_in_4bit": True,
    "fast_inference": True,  # vLLM support
    "gpu_memory_utilization": 0.7,
}

# ========== LoRA Configuration ==========
LORA_CONFIG = {
    "lora_rank": 32,
    "lora_alpha": 32,  # adjust the learning rate of lora module, in effect
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "use_gradient_checkpointing": "unsloth",
    "random_state": 3407
}

# ========== Training Configuration ==========
TRAINING_CONFIG = {
    "learning_rate": 5e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "optim": "paged_adamw_8bit",
    "logging_steps": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "num_generations": 6,
    "num_train_epochs": 1,
    "save_steps": 250,
    "max_grad_norm": 0.1,
    "logging_strategy": "steps",
    "log_level": "info",
    "log_completions": True,
}

# ========== Dataset Configuration ==========
DATASET_CONFIG = {
    # pre-processed datasets
    "dataset_files": [
        "datasets/alpaca/alpaca-markdown.json",
        # "datasets/alpaca/alpaca-naive.json",
        "datasets/natural_thinking/facebook_natural_reasoning-markhdown.json",
        "datasets/help_steer3/HelpSteer3-markdown.json"
    ],
    "alpaca_dataset": {
        "name": "yahma/alpaca-cleaned",
        "split": "train",
        "min_output_length": 500,
        "max_output_length": 2000
    },
    "natural_thinking_dataset": {
        "url": "facebook/natural_reasoning",
        "split": "train",
        "output_file": "dataset.json",
        "required_num_data": 3500,
        "max_samples": 50000,
        "min_response_length": 50,
        "max_response_length": 2000
    }
}

# ========== Experiment Configuration ==========
def get_experiment_config():
    """Get experiment configuration based on environment variable."""
    if os.environ.get("EXP_NAME") is None:
        raise ValueError("Environment Variable EXP_NAME must be set")
    
    exp_name = os.environ.get("EXP_NAME")
    model_size = MODEL_CONFIG["model_size"]
    lora_rank = LORA_CONFIG["lora_rank"]
    
    run_name = f"Qwen2.5-{model_size}-GRPO-lora_{lora_rank}-{exp_name}-multi_step"
    output_dir = f"outputs/{run_name}"
    
    return {
        "exp_name": exp_name,
        "run_name": run_name,
        "output_dir": output_dir
    }

# ========== Utility Functions ==========
def get_grpo_config():
    """Get GRPO configuration for the trainer."""
    exp_config = get_experiment_config()
    
    return {
        **TRAINING_CONFIG,
        "max_prompt_length": MODEL_CONFIG["max_prompt_length"],
        "max_completion_length": MODEL_CONFIG["max_completion_length"],
        "output_dir": exp_config["output_dir"],
        "run_name": exp_config["run_name"],
        "logging_dir": exp_config["output_dir"],
    }

def get_model_config():
    """Get model configuration for FastLanguageModel.from_pretrained."""
    return {
        "model_name": MODEL_CONFIG["model_name"],
        "max_seq_length": MODEL_CONFIG["max_seq_length"],
        "load_in_4bit": MODEL_CONFIG["load_in_4bit"],
        "fast_inference": MODEL_CONFIG["fast_inference"],
        "max_lora_rank": LORA_CONFIG["lora_rank"],
        "gpu_memory_utilization": MODEL_CONFIG["gpu_memory_utilization"],
    }

def get_peft_config():
    """Get PEFT configuration for FastLanguageModel.get_peft_model."""
    return {
        "r": LORA_CONFIG["lora_rank"],
        "target_modules": LORA_CONFIG["target_modules"],
        "lora_alpha": LORA_CONFIG["lora_alpha"],
        "use_gradient_checkpointing": LORA_CONFIG["use_gradient_checkpointing"],
        "random_state": LORA_CONFIG["random_state"]
    }
