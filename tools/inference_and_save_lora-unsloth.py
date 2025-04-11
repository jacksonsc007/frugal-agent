from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
import sys
sys.path.append(".")

# ========== Hyperparameters ==========
max_seq_length = (4096 + 1024)
max_prompt_length = 2048
# NOTE: ?
max_completion_length = max_seq_length - max_prompt_length
lora_rank = 32
# ========== Hyperparameters ==========

if __name__ == "__main__":
    # load model and tokenizer
    model_checkpoint = "outputs/Qwen2.5-7B-GRPO-formatter-lora_32-fix_lora_bug-saver_format_with_concrete_sys_prompt-multi_step/checkpoint-4500"
    print(f"\033[92m[INFO] Loading checkpoints...\033[0m")
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
    # 2. sampling with lora
    model.save_lora("grpo_saved_lora")
    lora = model.load_lora("grpo_saved_lora")

    # merge lora with base model and save
    model.save_pretrained_merged("qwen2.5-7B-instruct-lora-grpo-enhanced_saver_response", tokenizer, save_method = "merged_16bit",)