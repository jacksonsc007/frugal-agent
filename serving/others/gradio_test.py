import gradio as gr
from typing import List, Dict, Any, Generator
from openai import OpenAI

# Initialize vLLM client
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

def process_message(message: str, history: List, state: Dict) -> Dict:
    """Simulate processing a message and generating multiple responses."""
    # Replace with your actual logic (e.g., LLM calls, tool usage)
    return {
        "main_response": f"Main answer to: {message}",
        "analysis": f"Analysis of '{message}': This is a deeper breakdown...",
        "tools_used": f"Tools used: search, calculator"
    }

def chat_interface(
    message: str,
    history: List,
    main_chat_history: List,
    analysis_chat_history: List,
    tools_chat_history: List,
    state: gr.State
) -> Generator[Dict, None, None]:
    """
    Handles user input and streams responses to multiple chatbots.
    """
    # Process the message (replace with your actual logic)
    responses = process_message(message, history, state)
    
    # Update each chatbot history separately
    yield {
        main_chat_history: [(message, responses["main_response"])],
        analysis_chat_history: [(None, responses["analysis"])],  # None hides user input
        tools_chat_history: [(None, responses["tools_used"])]
    }

def main():
    with gr.Blocks(theme="soft", title="Multi-Chatbox Assistant") as demo:
        # State to persist data across interactions
        state = gr.State({"conversation": []})
        
        # Three separate chatbots in a column layout
        with gr.Column():
            main_chatbot = gr.Chatbot(
                label="Main Response",
                height=300,
                bubble_full_width=False
            )
            analysis_chatbot = gr.Chatbot(
                label="Detailed Analysis",
                height=200,
                bubble_full_width=False
            )
            tools_chatbot = gr.Chatbot(
                label="Tools Used",
                height=100,
                bubble_full_width=False
            )
        
        # Input and controls
        msg = gr.Textbox(placeholder="Ask me anything...")
        clear_btn = gr.Button("Clear Chat")
        
        # Submit handler updates all chatbots
        msg.submit(
            chat_interface,
            inputs=[msg, state, main_chatbot, analysis_chatbot, tools_chatbot, state],
            outputs=[main_chatbot, analysis_chatbot, tools_chatbot]
        )
        
        clear_btn.click(
            lambda: ([], [], [], {}),  # Reset all histories and state
            outputs=[main_chatbot, analysis_chatbot, tools_chatbot, state],
            queue=False
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()