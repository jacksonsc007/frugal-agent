import json
import os
import sys
import gradio as gr
import openai
import re
from typing import Any, Dict, Generator, List, Optional

sys.path.append("/root/workspace/ink_agent-multi_step")
from utils.sys_prompts import MASTERMIND_SYS_PROMPT_TOOL_SPECIFIC as MASTERMIND_SYS_PROMPT
from openai import OpenAI
from tool_caller import process_message

# Configuration and environment setup
os.environ['https_proxy'] = 'http://192.168.1.12:7891'

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

def chat_interface(
    message: str,
    history: List[Any],
    messages_state: List[Dict[str, Any]],
    dependent_tool_output_record: Dict[str, Any]
) -> Generator[str, None, None]:
    """
    Gradio chat interface callback. Handles user input and yields assistant responses.
    """
    response_stream = process_message(message, messages_state, dependent_tool_output_record)
    for chunk in response_stream:
        yield chunk

# Gradio UI setup
def init_messages() -> List[Dict[str, str]]:
    """Initialize the conversation with a system prompt."""
    return [{"role": "system", "content": MASTERMIND_SYS_PROMPT}]

def main() -> None:
    # Initialize vLLM client
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    models = client.models.list()
    print(f"\033[96m[INFO] Available models:\033[0m")
    import pprint
    pprint.pprint(models.data)
    # base_model = models.data[0].id

    """Main entry point to set up and launch the Gradio app."""
    with gr.Blocks(theme='soft') as demo:
        messages_state = gr.State(init_messages)
        dependent_tool_output_record = gr.State({})
        gr.ChatInterface(
            chat_interface,
            chatbot=gr.Chatbot(height=1100, resizable=True, type='messages'),
            additional_inputs=[messages_state, dependent_tool_output_record],
            title="vLLM Agent Assistant",
            description="A knowledgeable AI assistant with tool integration",
        )
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
