"""
Custom dataset processing utilities.
"""

import json
from datasets import Dataset


def get_training_dataset(json_files: list[str]):
    """
    Load and process custom training dataset from JSON files.
    
    Args:
        json_files: List of JSON file paths containing training data
        
    Returns:
        Processed dataset with shuffled data
    """
    all_data = []
    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)  # Merge datasets
    
    dataset = Dataset.from_list(all_data)
    # shuffle
    dataset = dataset.shuffle(seed=42)
    
    # Note: Length filtering is commented out in original code
    # def check_length(text):
    #     if len(text["prompt"][-1]['content']) < 3000:
    #         return True
    #     else:
    #         return False
    # dataset = dataset.filter(
    #     check_length,
    # )
    
    print("\033[92m Dataset size: \033[0m", len(dataset))
    return dataset 