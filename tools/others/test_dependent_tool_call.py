"""
Tutorial from Qwen official documentation
https://qwen.readthedocs.io/en/latest/framework/function_call.html#hugging-face-transformers

It aims to shed light on the tool-calling function supported by Qwen series models. Most importantly,
it demonstrates how is tool-calling capability is implemented.

2025-03-15
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import sys
# sys.path.append("..")
# sys.path.append(".")
# from utils.arsenal import try_parse_tool_calls, try_invoke_tool_calls, try_parse_intermediate_representation, TOOLS

# import importlib
# arsenal = importlib.import_module("utils.arsenal")
# try_parse_tool_calls = arsenal.try_parse_tool_calls
# try_invoke_tool_calls = arsenal.try_invoke_tool_calls
# try_parse_intermediate_representation = arsenal.try_parse_intermediate_representation
# TOOLS = arsenal.TOOLS


import importlib.util
import sys
from pathlib import Path

# Path to the file
file_path = Path("./utils/arsenal.py").resolve()
module_name = "arsenal_dynamic"

spec = importlib.util.spec_from_file_location(module_name, file_path)
arsenal = importlib.util.module_from_spec(spec)
sys.modules[module_name] = arsenal
spec.loader.exec_module(arsenal)

# Now access functions/variables
try_parse_tool_calls = arsenal.try_parse_tool_calls
try_invoke_tool_calls = arsenal.try_invoke_tool_calls
try_parse_intermediate_representation = arsenal.try_parse_intermediate_representation
TOOLS = arsenal.TOOLS

debug = True

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
                  

MESSAGES = [
    {"role": "system", "content": "Imagine you are a mastermind overseeing a suite of advanced AI tools designed to assist users with various tasks. You should consider the user's request and decide which tool you would call upon to provide the best response."},
#     {"role": "system", "content": MASTERMIND_SYS_PROMPT_DEPENDENT_TOOL_CALLS_3},
#     {"role": "user", "content": "call tool_1 and then call tool_2 with the output of tool_1"},
#     {"role": "assistant", "content": """<tool_call>
# {"name": "tool_1", "arguments": {"arg1": "value1"}, "call_sequence_id": 1}
# </tool_call>
# <tool_call>
# {"name": "tool_2", "arguments": {"arg1": "{1.output}"}, "call_sequence_id": 2}
# </tool_call>
# """
# },
    # {
    #     "role": "tool",
    #     "content": "tool_1 is called"
    # },
#     {"role": "user",  "content": "Please format and save the following content: what is avx? AVX (Advanced Vector Extensions) is an x86 CPU instruction set boosting floating-point and SIMD performance for tasks like multimedia/scientific computing. AVX2 adds integer ops; AVX-512 doubles vector width to 512 bits."},
#     {"role": "assistant", 
#      "content": """<tool_call>
# {"name": "format_organizer", "arguments": {"instruction": "What is AVX?", "response": "AVX (Advanced Vector Extensions) is an x86 CPU instruction set boosting floating-point and SIMD performance for tasks like multimedia/scientific computing. AVX2 adds integer ops; AVX-512 doubles vector width to 512 bits."}, "call_sequence_id": 1}
# </tool_call>
# <tool_call>
# {"name": "save_file", "arguments": {"file_name": "avx_explanation.txt", "content": "{1.output}"}, "call_sequence_id": 2}
# </tool_call>"""
#     },
#     {"role": "user", "content": "What is the main theme of the short story 'The Most Dangerous Game'?"},
#     {
#         "role": "assistant",
#         "content":"""\
# <tool_call>
# {"name": "question_answer_expert", "arguments": {"query": "What is the main theme of the short story 'The Most Dangerous Game'?"}, "call_sequence_id": 1}
# </tool_call> 
# """
#     },
#     {
#         "role": "tool", 
#         "content": "The main theme of *'The Most Dangerous Game'* by Richard Connell is **the thin line between civilization and savagery**. The story explores how quickly societal morals can be stripped away when survival is at stake. It questions the ethics of hunting for sport by turning the tables—making a human the prey—and shows how even the most refined individuals can become savage when pushed to the brink. It also delves into themes of **power, violence, and the instinct for self-preservation**."
#     },
#     {
#         "role": "user", 
#         "content": "please format and then save the formatted answer"
#     },
]


model_name = "Qwen/Qwen2.5-7B-Instruct" # use 0.5B to accommodate with 4090
# the chekcpoint for grpo-trained models
# checkpoint_path = "outputs/Qwen-0.5B-GRPO/checkpoint-1868"
# checkpoint_path = model_name
checkpoint_path = "qwen2.5-7B-instruct-lora-grpo-dependent_tool_call"

print("\033[91m[INFO]loading model \033[0m")
model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        # model_name,
        torch_dtype="auto",
        device_map="auto"
)

print("\033[91m[INFO]loading tokenizer\033[0m ")
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_path,
        # model_name,
)


tools = TOOLS
messages = MESSAGES[:]
TOOL_RESPONSE_RECORDS = {}

# Load your Jinja2 template as a string
from pathlib import Path
template_str = Path("ink_qwen_dependent_tool_call_template.jinja").read_text()

while (user_input := input("Press Enter to continue or 'q' to quit: ")) != 'q':
    if user_input != "c":
        messages.append({"role": "user", "content": user_input})

    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        # chat_template=modified_template
        chat_template=template_str
    )

    print("\033[92m[INFO] Generated Prompt:\033[0m ")
    print(text)
    # exit()
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512
    )
    output_text = tokenizer.batch_decode(outputs,)[0][len(text):]
    print("\033[92m[INFO] Generated Response:\033[0m ")
    print(output_text)


    parsed_output = try_parse_tool_calls(output_text)
    print("\033[92m[INFO] Parsed Response Message:\033[0m \n")
                
    import pprint
    pprint.pprint(parsed_output)
    messages.append(parsed_output)
                
    _, tool_names, _, tool_call_ids, tool_reponses = try_invoke_tool_calls(parsed_output, TOOL_RESPONSE_RECORDS)
    for fn_name, call_id, fn_res in zip(tool_names, tool_call_ids, tool_reponses):
        print("\033[92m[INFO] Tool Responses:\033[0m \n")
        tool_msg =         {
                "role": "tool",
                # "name": fn_name,
                "content": fn_res,
                # "call_sequence_id": call_id
            }
        pprint.pprint(tool_msg)
        messages.append(tool_msg)