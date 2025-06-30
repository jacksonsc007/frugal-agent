"""
Natural Thinking dataset processing utilities.
"""

import json
from datasets import load_dataset, Dataset
from config.multi_step_tool_calling.training_config import DATASET_CONFIG
from utils.sys_prompts import MASTERMIND_SYS_PROMPT as SYSTEM_PROMPT


def get_natural_thinking_dataset(url, split="train", output_file="dataset.json", required_num_data=3500):
    """
    Load and process Natural Thinking dataset.
    
    Args:
        url: Dataset URL or name
        split: Dataset split to load
        output_file: Output file name (unused in current implementation)
        required_num_data: Number of data samples to include
        
    Returns:
        Processed dataset with prompt and response format
    """
    nt_config = DATASET_CONFIG["natural_thinking_dataset"]
    dataset = load_dataset(url, split=split)
    dataset = dataset.shuffle(seed=42).select(
        range(min(nt_config["max_samples"], len(dataset)))
    )
    
    def check_length(text):
        text_length = len(text["responses"][0]['response'])
        return (text_length > nt_config["min_response_length"]) and (text_length < nt_config["max_response_length"])
    
    dataset = dataset.filter(check_length)
    
    def formatting_prompts_func(examples):
        instruction = examples["question"]
        output = examples["responses"][0]["response"]
        
        return {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': instruction},
            ],
            'response': output
        }
    
    dataset = dataset.map(formatting_prompts_func, batched=False)
    print("\033[92m number of data: \033[0m", len(dataset))
    
    return dataset 