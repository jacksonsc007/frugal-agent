"""
The code was modified from https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb
"""
import torch
import re
import json
import argparse
import os
import sys
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# unsloth needs to be import before importing trl
from unsloth import FastLanguageModel
from trl import GRPOConfig

# Import configuration
from config.multi_step_tool_calling.training_config import (
    get_experiment_config, get_grpo_config, 
    get_model_config, get_peft_config
)

from utils.reward_functions import *
from utils.env import ToolCallingEnv
from utils.sys_prompts import MASTERMIND_SYS_PROMPT as SYSTEM_PROMPT
from utils.arsenal import TOOLS
from utils.datasets import get_alpaca_instruction_response_pairs, get_natural_thinking_dataset, get_training_dataset
from utils.UnslothGRPOTrainer_modified import UnslothGRPOTrainer
from utils.logger.logger import mylogger as logger


# ========== Get Experiment Configuration ==========
exp_config = get_experiment_config()
exp_name = exp_config["exp_name"]
run_name = exp_config["run_name"]
output_dir = exp_config["output_dir"]
os.environ["OUTPUT_DIR"] = output_dir

# ========== Environments ==========


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="resume training", default=False)
    parser.add_argument("--checkpoint", type=str, help="checkpoint to resume from", default=None)
    args = parser.parse_args()
    
    if (args.resume and not args.checkpoint):
        raise ValueError("--checkpoint must be specified when --resume is used")
    
    # load model and tokenizer
    model_config = get_model_config()
    model, tokenizer = FastLanguageModel.from_pretrained(**model_config)
    # override the original chat template
    # tokenizer.chat_template = Ink_QWEN_DEPENDENT_TOOL_CALL_TEMPLATE
    
    peft_config = get_peft_config()
    model = FastLanguageModel.get_peft_model(
        model,
        **peft_config
    )
    
    # prepare dataset
    dataset = get_alpaca_instruction_response_pairs()

    # prepare trainer
    grpo_config = get_grpo_config()
    train_args = GRPOConfig(**grpo_config)
    
    # specify the environment for the agent
    custom_env = ToolCallingEnv(TOOLS)
    
    # reward functions are bound to environments. No need to be specified explicitly.
    reward_funcs = []
    
    trainer = UnslothGRPOTrainer(
        model=model,
        env=custom_env,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=train_args,
        train_dataset=dataset,
    )
    trainer.train(
        resume_from_checkpoint=args.checkpoint if args.resume else None
    )