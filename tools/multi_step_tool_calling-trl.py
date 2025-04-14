# train_grpo.py

#

# See https://github.com/willccbb/verifiers for ongoing developments

#

import coredumpy
# Create a dump in "./dumps" when there's an unhandled exception
coredumpy.patch_except(directory='./dumps')
import re

import torch

from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import LoraConfig

from trl import GRPOConfig, GRPOTrainer
from typing import List, Dict, Any, Tuple, Sequence
from vllm import LLM, SamplingParams, RequestOutput
import json

import sys
sys.path.append(".")
sys.path.append("..")
from utils.env import ToolCallingEnv
from utils.sys_prompts import MASTERMIND_SYS_PROMPT as SYSTEM_PROMPT
from utils.arsenal import TOOLS
from utils.reward_functions import *

debug = True

import os
os.environ["https_proxy"] = "http://192.168.1.12:7891"

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



XML_COT_FORMAT = """\

<reasoning>

{reasoning}

</reasoning>

<answer>

{answer}

</answer>

"""



def extract_xml_answer(text: str) -> str:

    answer = text.split("<answer>")[-1]

    answer = answer.split("</answer>")[0]

    return answer.strip()



def extract_hash_answer(text: str) -> str | None:

    if "####" not in text:

        return None

    return text.split("####")[1].strip().replace(",", "").replace("$", "")



# uncomment middle messages for 1-shot prompting

def get_gsm8k_questions(split = "train") -> Dataset:

    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore

    data = data.map(lambda x: { # type: ignore

        'prompt': [

            {'role': 'system', 'content': SYSTEM_PROMPT},

            #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},

            #{'role': 'assistant', 'content': XML_COT_FORMAT.format(

            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",

            #    answer="7"

            #)},

            {'role': 'user', 'content': x['question']}

        ],

        'answer': extract_hash_answer(x['answer'])

    }) # type: ignore

    return data # type: ignore

def get_alpaca_instruction_response_pairs(split = "train")->Dataset:
    dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
    # Filter out examples where the output length is less than 100 characters
    dataset = dataset.filter(lambda example: len(example["output"]) >= 500 and len(example["output"]) <= 2000)
    def formatting_prompts_func(examples):
        instruction = examples["instruction"]
        input       = examples["input"]
        output      = examples["output"]
        return {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"{instruction}\n{input}"}
            ],
            'response': output
            # 'answer': extract_hash_answer(output)
        }
    dataset = dataset.map(formatting_prompts_func, batched = False)
    return dataset

dataset = get_alpaca_instruction_response_pairs()

# Reward functions

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:

    responses = [completion[0]['content'] for completion in completions]
    q = ''.join(
        [
            f"=>{p['role']}:\n{p['content']}\n" if p.get( 'content' ) else f"{p['role']}\n{p['tool_calls']}\n" for p in prompts[0] ]
        )
    # q = prompts[0][-1]['content']

    extracted_responses = [extract_xml_answer(r) for r in responses]

    # print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")

    print(  f"\n\033[92mQuestion\033[0m:\n{q}",
            f"\n\033[91mAnswer\033[0m:\n{answer[0]}",
            f"\n\033[93mResponse\033[m:\n{responses[0]}",
            f"\n\033[94mExtracted\033[0m:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]



def int_reward_func(completions, **kwargs) -> list[float]:

    responses = [completion[0]['content'] for completion in completions]

    extracted_responses = [extract_xml_answer(r) for r in responses]

    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]



def strict_format_reward_func(completions, **kwargs) -> list[float]:

    """Reward function that checks if the completion has a specific format."""

    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"

    responses = [completion[0]["content"] for completion in completions]

    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]

    return [0.5 if match else 0.0 for match in matches]



def soft_format_reward_func(completions, **kwargs) -> list[float]:

    """Reward function that checks if the completion has a specific format."""

    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"

    responses = [completion[0]["content"] for completion in completions]

    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]

    return [0.5 if match else 0.0 for match in matches]



def count_xml(text) -> float:

    count = 0.0

    if text.count("<reasoning>\n") == 1:

        count += 0.125

    if text.count("\n</reasoning>\n") == 1:

        count += 0.125

    if text.count("\n<answer>\n") == 1:

        count += 0.125

        count -= len(text.split("\n</answer>\n")[-1])*0.001

    if text.count("\n</answer>") == 1:

        count += 0.125

        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001

    return count



def xmlcount_reward_func(completions, **kwargs) -> list[float]:

    contents = [completion[0]["content"] for completion in completions]

    return [count_xml(c) for c in contents]



#model_name = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_SIZE = "0.5B"
# MODEL_SIZE = "7B"
model_name = f"Qwen/Qwen2.5-{MODEL_SIZE}-Instruct"



if "Llama" in model_name:

    output_dir = "outputs/Llama-1B-GRPO"

    run_name = "Llama-1B-GRPO-gsm8k"

else:

    output_dir=f"outputs/Qwen-{MODEL_SIZE}-GRPO"

    run_name=f"Qwen-{MODEL_SIZE}-GRPO-gsm8k"



training_args = GRPOConfig(

    output_dir=output_dir,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.3,

    run_name=run_name,

    learning_rate=5e-6,

    adam_beta1 = 0.9,

    adam_beta2 = 0.99,

    weight_decay = 0.1,

    warmup_ratio = 0.1,

    lr_scheduler_type='cosine',

    logging_steps=1,

    bf16=True,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=4,

    num_generations=2,

    max_prompt_length=1024,

    max_completion_length=2048,

    num_train_epochs=1,

    save_steps=100,

    max_grad_norm=0.1,

    # report_to="none",
    report_to="wandb",

    log_on_each_node=False,
    log_completions = True,

)

peft_config = LoraConfig(

    r=16,

    lora_alpha=64,

    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],

    task_type="CAUSAL_LM",

    lora_dropout=0.05,

)

model = AutoModelForCausalLM.from_pretrained(

    model_name,

    torch_dtype=torch.bfloat16,

   attn_implementation="flash_attention_2",

    device_map=None,
    use_cache=False

).to("cuda")



tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token



# use peft at your own risk; not working for me with multi-GPU training

custom_env = ToolCallingEnv(TOOLS)

trainer = GRPOTrainer(
    model=model,
    env = custom_env,

    processing_class=tokenizer,

    reward_funcs=[

        # xmlcount_reward_func,

        # soft_format_reward_func,

        # strict_format_reward_func,

        # int_reward_func,

        # correctness_reward_func,

        ordinary_tool_calling_reward_func,
        
        log_func_multi_step,

        saver_content_reward_func,
        
        saver_filetype_reward_func,
        
        ],

    args=training_args,

    train_dataset=dataset,

    peft_config=peft_config

)

trainer.train()
