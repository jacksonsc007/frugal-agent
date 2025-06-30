"""
Alpaca dataset processing utilities.
"""

from datasets import load_dataset, Dataset
from config.multi_step_tool_calling.training_config import DATASET_CONFIG
from utils.sys_prompts import MASTERMIND_SYS_PROMPT as SYSTEM_PROMPT


def get_alpaca_instruction_response_pairs(split="train") -> Dataset:
    """
    Load and process Alpaca dataset for instruction-response pairs.
    
    Args:
        split: Dataset split to load ("train", "test", "validation")
        
    Returns:
        Processed dataset with prompt and response format
    """
    alpaca_config = DATASET_CONFIG["alpaca_dataset"]
    dataset = load_dataset(alpaca_config["name"], split=split)
    
    # Filter out examples where the output length is less than 100 characters
    dataset = dataset.filter(
        lambda example: len(example["output"]) >= alpaca_config["min_output_length"] 
        and len(example["output"]) <= alpaca_config["max_output_length"]
    )
    
    def formatting_prompts_func(examples):
        instruction = examples["instruction"]
        input_text = examples["input"]
        output = examples["output"]
        return {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"{instruction}\n{input_text}"}
            ],
            'response': output
        }
    
    dataset = dataset.map(formatting_prompts_func, batched=False)
    return dataset 