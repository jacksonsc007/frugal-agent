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

# Global message type prefixes
PREFIX_ASSISTANT   = "---\n🤖 **Assistant**     : \n\n"
PREFIX_TOOL_START  = "---\n🔧 **Tool Execution**: \n\n"
PREFIX_TOOL_OUTPUT = "---\n📊 **Tool Output**\n\n"
PREFIX_TOOL_ERROR  = "---\n❌ **Tool Error**\n\n"
PREFIX_UNKNOWN     = "---\n❓ **Unknown**        : \n\n"

def format_message_for_display(response_dict: Dict[str, Any]) -> str:
    """
    Format structured response for display in Gradio chat interface.
    Adds a leading newline before each message type for separation.
    """
    msg_type = response_dict.get("type", "unknown")
    content = response_dict.get("content", "")
    tool_name = response_dict.get("tool_name", "")
    
    if msg_type == "assistant":
        return f"{PREFIX_ASSISTANT}\n{content}"
    elif msg_type == "tool_start":
        return f"{PREFIX_TOOL_START}\n{content}"
    elif msg_type == "tool_output":
        return f"{PREFIX_TOOL_OUTPUT} ({tool_name}):\n{content}"
    elif msg_type == "tool_error":
        return f"{PREFIX_TOOL_ERROR} ({tool_name}):\n{content}"
    else:
        return f"{PREFIX_UNKNOWN} {content}"

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
        
        
        gr.Markdown("# vLLM Agent Assistant")
        gr.Markdown("A knowledgeable AI assistant with tool integration - Messages are now separated by type (🤖 Assistant, 🔧 Tool Execution, 📊 Tool Output, ❌ Tool Error)")
        
        chatbot = gr.Chatbot(height=1100, resizable=True)
        msg = gr.Textbox(label="Input", placeholder="Type your message here...")
        clear = gr.Button("Clear")
        
        def user(user_message, history, messages_state, dependent_tool_output_record):
            """Handle user input"""
            return "", history + [[user_message, None]]
        
        def bot(history, messages_state, dependent_tool_output_record):
            """Handle bot response with multiple message types"""
            if not history:
                return history
            
            user_message = history[-1][0]
            response_stream = process_message(user_message, messages_state, dependent_tool_output_record)
            
            # Initialize with current history
            current_history = history.copy()
            
            # Track the current assistant response
            current_assistant_response = ""
            assistant_message_added = False
            last_tool_name = None
            
            # Define message prefixes for consistency
            def get_prefix(msg_type, tool_name=None):
                if msg_type == "assistant":
                    return PREFIX_ASSISTANT
                elif msg_type == "tool_start":
                    return PREFIX_TOOL_START
                elif msg_type == "tool_output":
                    return f"{PREFIX_TOOL_OUTPUT} ({tool_name}):"
                elif msg_type == "tool_error":
                    return f"{PREFIX_TOOL_ERROR} ({tool_name}):"
                else:
                    return PREFIX_UNKNOWN
            
            for response_dict in response_stream:
                if isinstance(response_dict, dict):
                    msg_type = response_dict.get("type", "unknown")
                    content = response_dict.get("content", "")
                    tool_name = response_dict.get("tool_name", "")
                    prefix = get_prefix(msg_type, tool_name)
                    
                    if msg_type == "assistant":
                        current_assistant_response = content
                        if not assistant_message_added:
                            current_history[-1][1] = f"{prefix}\n{content}"
                            assistant_message_added = True
                        else:
                            current_history[-1][1] = f"{prefix}\n{content}"
                    elif msg_type == "tool_start":
                        formatted_message = f"{prefix}\n{content}"
                        current_history.append([None, formatted_message])
                        last_tool_name = tool_name
                    elif msg_type == "tool_output":
                        tool_output_prefix = get_prefix("tool_output", tool_name)
                        if current_history and current_history[-1][1].startswith(tool_output_prefix):
                            prev = current_history[-1][1]
                            current_history[-1][1] = prev + content
                        else:
                            formatted_message = f"{tool_output_prefix}\n{content}"
                            current_history.append([None, formatted_message])
                        last_tool_name = tool_name
                    elif msg_type == "tool_error":
                        formatted_message = f"{prefix}\n{content}"
                        current_history.append([None, formatted_message])
                        last_tool_name = None
                else:
                    if not assistant_message_added:
                        current_history[-1][1] = str(response_dict)
                        assistant_message_added = True
                    else:
                        current_history[-1][1] = str(response_dict)
                yield current_history
        
        def clear_history():
            """Clear chat history"""
            return [], init_messages(), {}
        
        # Set up event handlers
        msg.submit(user, [msg, chatbot, messages_state, dependent_tool_output_record], [msg, chatbot], queue=False).then(
            bot, [chatbot, messages_state, dependent_tool_output_record], [chatbot]
        )
        clear.click(clear_history, outputs=[chatbot, messages_state, dependent_tool_output_record])
        
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
