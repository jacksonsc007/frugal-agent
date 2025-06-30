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

# HTML templates for different message types
def create_message_html(prefix: str, content: str, message_type: str) -> str:
    """Create HTML for a message with appropriate styling based on type."""
    type_classes = {
        "assistant": "assistant-message",
        "tool_start": "tool-start-message",
        "tool_output": "tool-output-message",
        "tool_error": "tool-error-message",
        "unknown": "unknown-message"
    }
    
    icon_map = {
        "assistant": "🤖",
        "tool_start": "🔧",
        "tool_output": "📊",
        "tool_error": "❌",
        "unknown": "❓"
    }
    
    return f"""
    <div class="message-container {type_classes.get(message_type, '')}">
        <div class="message-header">
            <span class="message-icon">{icon_map.get(message_type, '')}</span>
            <span class="message-type">{prefix.strip()}</span>
        </div>
        <div class="message-content">
            {content}
        </div>
    </div>
    """

# Gradio UI setup with custom CSS
css = """
body, .message-container {
    font-family: Arial, sans-serif;
}

.message-container {
    border-radius: 10px;
    padding: 4px 6px;
    margin: 4px 0;
    border: 1px solid #e0e0e0;
    width: 1200px;
}

.message-header {
    font-weight: bold;
    margin-bottom: 2px;
    display: flex;
    align-items: center;
    gap: 4px;
}

.message-icon {
    font-size: 1.2em;
}

.message-content {
    padding-left: 0px;
    font-size: 1.15em;
    width: 100%;
    margin: 0px 0px 0px 10px;
}

.assistant-message {
    background-color: #f5f7ff;
    border-left: 4px solid #4a6bdf;
}

.tool-start-message {
    background-color: #fff9f0;
    border-left: 4px solid #ffb74d;
}

.tool-output-message {
    background-color: #f0fff4;
    border-left: 4px solid #66bb6a;
}

.tool-error-message {
    background-color: #ffebee;
    border-left: 4px solid #f44336;
}

.unknown-message {
    background-color: #f5f5f5;
    border-left: 4px solid #9e9e9e;
}

.dark .assistant-message {
    background-color: #1a237e;
}

.dark .tool-start-message {
    background-color: #ff6f00;
}

.dark .tool-output-message {
    background-color: #1b5e20;
}

.dark .tool-error-message {
    background-color: #b71c1c;
}

.dark .unknown-message {
    background-color: #212121;
}

.user-message {
    padding: 4px;
    margin: 0px;
}
"""

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
    with gr.Blocks(theme='soft', css=css) as demo:
        messages_state = gr.State(init_messages)
        dependent_tool_output_record = gr.State({})
        
        gr.Markdown("# vLLM Agent Assistant")
        gr.Markdown("A knowledgeable AI assistant with tool integration - Messages are styled by type")
        
        chatbot = gr.Chatbot(height=1100, resizable=True, bubble_full_width=True, render="html", render_markdown=False)
        msg = gr.Textbox(label="Input", placeholder="Type your message here...")
        clear = gr.Button("Clear")
        
        def user(user_message, history, messages_state, dependent_tool_output_record):
            """Handle user input"""
            user_html = f"""
            <div class="message-container user-message">
                <div class="message-header">
                    <span class="message-icon">＜(´⌯  ̫⌯`)＞</span>
                    <span class="message-type">User</span>
                </div>
                <div class="message-content">
                    {user_message}
                </div>
            </div>
            """
            return "", history + [[user_html, None]]
        
        def bot(history, messages_state, dependent_tool_output_record):
            """Handle bot response with multiple message types"""
            if not history:
                return history
            
            user_message = history[-1][0]
            # Extract just the text content from the user HTML message
            user_content = re.search(r'<div class="message-content">(.*?)</div>', user_message, re.DOTALL)
            if user_content:
                user_message = user_content.group(1).strip()
            
            response_stream = process_message(user_message, messages_state, dependent_tool_output_record)
            
            # Initialize with current history
            current_history = history.copy()
            last_msg_type = ""
            for response_dict in response_stream:
                if isinstance(response_dict, dict):
                    msg_type = response_dict.get("type", "unknown")
                    content = response_dict.get("content", "")
                    tool_name = response_dict.get("tool_name", "")
                    
                    # Create appropriate prefix based on message type
                    prefix_map = {
                        "assistant": "Assistant",
                        "tool_start": "Tool Execution",
                        "tool_output": f"Tool Output ({tool_name})",
                        "tool_error": f"Tool Error ({tool_name})",
                        "unknown": "Unknown"
                    }
                    prefix = prefix_map.get(msg_type, "Unknown")
                    
                    # Format the message as HTML
                    message_html = create_message_html(prefix, content, msg_type)
                    
                    if msg_type != last_msg_type:
                        current_history.append([None, message_html])
                        last_msg_type = msg_type
                    else:
                        current_history[-1][1] = message_html
                else:
                    # Fallback for non-dict responses
                    message_html = create_message_html("Unknown", str(response_dict), "unknown")
                    if current_history[-1][1] is None:
                        current_history[-1][1] = message_html
                    else:
                        current_history.append([None, message_html])
                
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