# Configuration Directory

This directory contains all configuration files for the frugal-agent project.

## Files

### `training_config.py`
Contains all configuration parameters for the multi-step tool calling training script.

#### Configuration Sections:

1. **Environment Configuration** (`PROXY_CONFIG`, `DEBUG_CONFIG`)
   - Proxy settings for network access
   - Debug configuration for development

2. **Model Configuration** (`MODEL_CONFIG`)
   - Model name and size
   - Sequence length settings
   - GPU memory utilization

3. **LoRA Configuration** (`LORA_CONFIG`)
   - LoRA rank and alpha
   - Target modules for fine-tuning
   - Gradient checkpointing settings

4. **Training Configuration** (`TRAINING_CONFIG`)
   - Learning rate and optimizer settings
   - Batch size and gradient accumulation
   - Logging and saving parameters

5. **Dataset Configuration** (`DATASET_CONFIG`)
   - Dataset file paths
   - Alpaca dataset settings
   - Natural thinking dataset settings

6. **Reward Functions Configuration** (`REWARD_FUNCTIONS_CONFIG`)
   - Enabled and disabled reward functions
   - Configurable reward function selection

7. **System Configuration** (`SYSTEM_CONFIG`)
   - System prompts and tools
   - Trainer class selection

#### Utility Functions:

- `get_experiment_config()`: Generates experiment-specific configuration
- `get_grpo_config()`: Returns GRPO trainer configuration
- `get_model_config()`: Returns model loading configuration
- `get_peft_config()`: Returns PEFT model configuration

## Usage

```python
from config.training_config import (
    MODEL_CONFIG, 
    get_experiment_config,
    get_grpo_config
)

# Use configuration in your training script
exp_config = get_experiment_config()
grpo_config = get_grpo_config()
```

## Environment Variables

- `EXP_NAME`: Required environment variable for experiment naming
- `OUTPUT_DIR`: Automatically set based on experiment configuration

## Modifying Configuration

To modify training parameters, edit the corresponding configuration dictionaries in `training_config.py`. The main training script will automatically use these updated values. 