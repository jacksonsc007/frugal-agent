from unsloth import FastLanguageModel
import torch
import re
import json
import argparse
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer

import sys
sys.path.append(".")
from utils.reward_functions import *
import coredumpy
# Create a dump in "./dumps" when there's an unhandled exception
coredumpy.patch_except(directory='./dumps')

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

# ========== Hyperparameters ==========
exp_name = "ink_curated_data-only_markdown_struture_data-concise_sys_prompt_2-resume-add_logic_hieraychy_bonus_reward"
max_seq_length = (4096 + 1024)
max_prompt_length = 2048
# NOTE: ?
max_completion_length = max_seq_length - max_prompt_length
lora_rank = 32

MODEL_SIZE = "7B"
output_dir=f"outputs/Qwen2.5-{MODEL_SIZE}-GRPO-formatter-lora_{lora_rank}-{exp_name}"
run_name=f"Qwen2.5-{MODEL_SIZE}-GRPO-formatter-lora_{lora_rank}-{exp_name}"

# ========== Hyperparameters ==========

dataset_files = [
    "datasets/alpaca/alpaca-markdown.json",
    # "datasets/alpaca/alpaca-naive.json",
    "datasets/natural_thinking/facebook_natural_reasoning-markhdown.json",
    "datasets/help_steer3/HelpSteer3-markdown.json"
]
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
    
    # load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen2.5-7B-Instruct",
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
    
    # prepare dataset
    dataset = get_training_dataset(dataset_files)

    # prepare trainer
    train_args = GRPOConfig(
        learning_rate= 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        num_generations = 6,
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        num_train_epochs = 10,
        # or
        # max_steps=500,
        save_steps = 250,
        max_grad_norm = 0.1,
        # report_to = "wandb",
        output_dir = output_dir,
        run_name = run_name,
        logging_dir= output_dir,
        logging_strategy= "steps",
        log_level="info",
    ) 
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            yaml_format_reward_func,
            markdown_format_reward_func,
            placeholder_reward_func,
            logic_heading_hierarchy_func,
            overall_format_reward_func,
            more_tags_reward_func,
            log_func
        ],
        args=train_args,
        train_dataset=dataset,
    )
    trainer.train(
        resume_from_checkpoint=args.checkpoint if args.resume else None
    )