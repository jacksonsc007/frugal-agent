import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from unsloth import FastLanguageModel
import torch
import re
import json
import argparse
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer

from utils.UnslothGRPOTrainer_modified import UnslothGRPOTrainer

from utils.reward_functions import *
from utils.env import FormatterEnv

# Import formatter config
from config.formatter.training_config import (
    get_experiment_config, get_grpo_config, get_model_config, get_peft_config, DATASET_CONFIG
)

debug = False

if debug:
    # improve torch tensor printing
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

    # debug with debugpy
    import debugpy
    # Listen on a specific port (choose any available port, e.g., 61074)
    debugpy.listen(("0.0.0.0", 61074))
    print("Waiting for debugger to attach...")
    # Optional: Wait for the debugger to attach before continuing execution
    debugpy.wait_for_client()


def get_training_dataset(json_files: list[str]):
    all_data = []
    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)  # Merge datasets
    
    dataset = Dataset.from_list(all_data)
    # shuffle
    dataset = dataset.shuffle(seed=42)
    # limit the input length to avoid OOM
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="resume training", default=False)
    parser.add_argument("--checkpoint", type=str, help="checkpoint to resume from", default=None)
    args = parser.parse_args()
    
    if (args.resume and not args.checkpoint):
        raise ValueError("--checkpoint must be specified when --resume is used")
    
    # prepare dataset
    dataset_files = DATASET_CONFIG["dataset_files"]
    dataset = get_training_dataset(dataset_files)

    # load model and tokenizer
    model_config = get_model_config()
    model, tokenizer = FastLanguageModel.from_pretrained(**model_config)
    
    peft_config = get_peft_config()
    model = FastLanguageModel.get_peft_model(
        model,
        **peft_config
    )

    custom_env = FormatterEnv()

    # prepare trainer
    train_args = GRPOConfig(**get_grpo_config())
    trainer = UnslothGRPOTrainer(
        model=model,
        env=custom_env,
        processing_class=tokenizer,
        reward_funcs=[
            # yaml_format_reward_func,
            # markdown_format_reward_func,
            # placeholder_reward_func,
            # logic_heading_hierarchy_func,
            # overall_format_reward_func,
            # more_tags_reward_func,
            # log_func
        ],
        args=train_args,
        train_dataset=dataset,
    )
    trainer.train(
        resume_from_checkpoint=args.checkpoint if args.resume else None
    )