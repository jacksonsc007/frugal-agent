"""
This script serves to save the trained lora model for future usage.
"""

from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
import sys
sys.path.append(".")
import argparse
import os
from logger.logger import mylogger as logger

# ========== Hyperparameters ==========
max_seq_length = (4096 + 1024)
max_prompt_length = 2048
# NOTE: ?
max_completion_length = max_seq_length - max_prompt_length
lora_rank = 32
# ========== Hyperparameters ==========

if __name__ == "__main__":
    
    # read model checkpoint, output path, output name from argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--if_merge", type=bool, default=False)
    args = parser.parse_args()
    model_checkpoint = args.model_checkpoint
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model_name = args.model_name

    
    
    # print(f"\033[92m[INFO] Loading checkpoints...\033[0m")
    logger.info(f"Loading checkpoints...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_checkpoint,
        max_seq_length = max_seq_length,
        load_in_4bit=True,
        fast_inference=True, # vLLM support
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_rank, # adjust the learning rate of lora module, in effect
        use_gradient_checkpointing= "unsloth",
        random_state= 3407
    )
    # print(f"\033[92m[INFO] Saving lora model {model_name}... to {output_path}\033[0m")
    logger.info(f"Saving lora model {model_name}... to {output_path}")
    save_path = os.path.join(output_path, model_name)
    model.save_lora(save_path)
    tokenizer.save_pretrained(save_path)
    # lora = model.load_lora(save_path)

    # merge lora with base model and save
    if args.if_merge:
        merged_model_name = model_name + "-merged"
        merged_save_path = os.path.join(output_path, merged_model_name)
        model.save_pretrained_merged(merged_save_path, tokenizer, save_method = "merged_16bit",)

    # another way to save lora
    # # save lora to fp16 
    # save_path = os.path.join(output_path, model_name)
    # model.save_pretrained_merged(save_path, tokenizer, save_method = "lora",)