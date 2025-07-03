"""
Before running this script, run vLLM service:
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes  --ma
x_model_len 8192
"""
import json

from utils.arsenal import TOOLS, get_function_by_name
from utils.sys_prompts import MASTERMIND_SYS_PROMPT
from openai import OpenAI

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

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model_name = "Qwen/Qwen2.5-7B-Instruct"

tools = TOOLS

messages = [
    {
        "role": "system",
        "content": MASTERMIND_SYS_PROMPT,
    },
    {
        "role": "user",
        "content": "what is avx?"
    },
    {
        "role": "assistant",
        "content": "AVX stands for Advanced Vector Extensions. It is a set of CPU instructions that improve the performance of floating-point operations."
    },
    {
        "role": "user",
        "content": "please format the answer and save the  answer to a file named avx_formatted.md"
    }
]

response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
        "chat_template_kwargs": {"enable_thinking": False}  # default to True
    },
)

response_message = response.choices[0].message.model_dump()
tool_calls = response_message.get( 'tool_calls', None)
if tool_calls:
    for t in tool_calls:
        import pprint
        pprint.pprint(t)
        print('\n---\n')
        

# messages.append(response_message)

# if tool_calls := messages[-1].get("tool_calls", None):
#     for tool_call in tool_calls:
#         call_id: str = tool_call["id"]
#         if fn_call := tool_call.get("function"):
#             fn_name: str = fn_call["name"]
#             fn_args: dict = json.loads(fn_call["arguments"])
        
#             fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))

#             messages.append({
#                 "role": "tool",
#                 "content": fn_res,
#                 "tool_call_id": call_id,
#             })