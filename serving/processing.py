import json
from typing import Any, Dict, Generator, List, Callable, Union
from serving.tools import get_function_by_name, try_parse_intermediate_representation
from openai import OpenAI
from openai.lib.streaming.chat._completions import ChatCompletionStreamState
import openai
from utils.arsenal import TOOLS
from utils.logger.logger import mylogger as logger

# You may need to set up the client here or pass it as an argument
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)


def process_message(
    message: str,
    messages_state: List[Dict[str, Any]],
    dependent_tool_output_dict: Dict[str, Any],
    use_streaming: bool = True
) -> Generator[Dict[str, Any], None, None]:
    """
    Process a user message, interact with the LLM, and delegate tool call handling to handle_tool_calls.
    Yields structured responses to distinguish between assistant responses and tool outputs.
    """
    messages_state.append({"role": "user", "content": message})
    # model_name = "Qwen/Qwen3-8B"
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_name = "dependent_tool_caller"
    if use_streaming:
        stream = client.chat.completions.create(
            messages=messages_state,
            model=model_name,
            tools=TOOLS,
            stream=True,
            max_completion_tokens=4096,
            temperature=0.7,
        )
        state = ChatCompletionStreamState(
            input_tools=openai.NOT_GIVEN,
            response_format=openai.NOT_GIVEN,
        )
        content_buffer = ""
        tool_calls_buffer = {}
        
        for chunk in stream:
            state.handle_chunk(chunk)
            # Process content
            content_delta = chunk.choices[0].delta.content or ""
            content_buffer += content_delta
            
            # Process tool calls
            tool_call_deltas = chunk.choices[0].delta.tool_calls or []
            
            for delta in tool_call_deltas:
                idx = delta.index
                if idx not in tool_calls_buffer:
                    tool_calls_buffer[idx] = {
                        'name': '',
                        'args': '',
                        'id': ''
                    }
                
                if delta.function:
                    tool_calls_buffer[idx]['name'] += delta.function.name or ""
                    tool_calls_buffer[idx]['args'] += delta.function.arguments or ""
                    if hasattr(delta.function, "call_sequence_id"):
                        tool_calls_buffer[idx]['id'] = str(delta.function.call_sequence_id)
            
            # Format output cleanly
            output_lines = []
            if content_buffer.strip():
                output_lines.append(f"💬: {content_buffer.strip()}")
            
            for idx, tool in sorted(tool_calls_buffer.items()):
                tool_output = []
                if tool['name']:
                    tool_output.append(f"🛠️ Calling {tool['name']}")
                if tool['id']:
                    tool_output.append(f"    • call_sequence_id #{tool['id']}")
                if tool['args'].strip():
                    try:
                        args = json.loads(tool['args'])
                        pretty_args = "\n".join(f"    • {k}: {v}" for k,v in args.items())
                        tool_output.append(f"{pretty_args}")
                    except:
                        tool_output.append(f"{tool['args']}")
                
                if tool_output:
                    output_lines.append("\n".join(tool_output))
            
            if output_lines:
                yield {
                    "type": "assistant",
                    "content": "\n\n".join(output_lines)
                }
        final_response = state.get_final_completion()
        assistant_message = final_response.choices[0].message.model_dump()
    else:
        response = client.chat.completions.create(
            messages=messages_state,
            model=model_name,
            tools=TOOLS,
            stream=False,
            max_completion_tokens=2048,
        )
        assistant_message = response.choices[0].message.model_dump()
        # Yield complete assistant response for non-streaming mode
        assistant_content = assistant_message.get("content", "")
        if assistant_content:
            yield {"type": "assistant", "content": assistant_content}

    # Ensure tool call arguments are in the correct format
    try:
        tool_calls = assistant_message.get('tool_calls')
        if tool_calls is not None:
            for tool_call in tool_calls:
                if isinstance(tool_call['function']['arguments'], dict):
                    tool_call['function']['arguments'] = json.dumps(tool_call['function']['arguments'])
    except Exception as e:
        logger.warning("Tool calls have incorrect function field")

    messages_state.append(assistant_message)

    # Delegate tool call handling to the new module
    yield from handle_tool_calls(assistant_message, messages_state, dependent_tool_output_dict)


def handle_tool_calls(
    assistant_message: Dict[str, Any],
    messages_state: List[Dict[str, Any]],
    dependent_tool_output_dict: Dict[str, Any]
) -> Generator[Dict[str, Any], None, None]:
    """
    Handle tool calls from the assistant message, execute the corresponding functions,
    and update the conversation state.

    Args:
        assistant_message: The message from the assistant containing tool calls.
        messages_state: The conversation state to update.
        dependent_tool_output_dict: Dictionary to store outputs of dependent tool calls.

    Yields:
        Dict[str, Any]: Structured tool execution results with type and content.
    """
    tool_calls = assistant_message.get("tool_calls")
    if tool_calls is not None:
        for tool_call in tool_calls:
            call_id = tool_call["id"]
            fn_name = tool_call["function"]["name"]
            
            # Yield tool call start notification
            yield {"type": "tool_start", "content": f"Executing tool: {fn_name}", "tool_name": fn_name}
            
            try:
                fn_args = tool_call["function"]["arguments"]
                if isinstance(fn_args, str):
                    fn_args = json.loads(tool_call["function"]["arguments"], strict=False)
                    logger.warning("\033[96m[INFO] Tool call arguments are in string format, please make sure the type is dict \033[0m")
                tool_call_sequence_id = tool_call["function"].get("call_sequence_id", None)
                fn_args = try_parse_intermediate_representation(fn_args, dependent_tool_output_dict)
            except Exception as e:
                error_message = f"Error parsing tool call arguments: {e}"
                logger.warning(error_message)
                yield {"type": "tool_error", "content": error_message, "tool_name": fn_name}
                messages_state.pop()
                break
            try:
                fn = get_function_by_name(fn_name)
                tool_msgs = ""
                for tool_msg_chunk in fn(**fn_args):
                    tool_msgs += tool_msg_chunk
                    # Yield only the new chunk, not the full accumulated output
                    yield {"type": "tool_output", "content": tool_msgs, "tool_name": fn_name}
                fn_result = tool_msgs
                call_sequence_id = tool_call_sequence_id
                if call_sequence_id is not None:
                    if call_sequence_id in dependent_tool_output_dict:
                        raise ValueError(f"Duplicate call_sequence_id: {call_sequence_id}")
                    dependent_tool_output_dict[tool_call_sequence_id] = fn_result
                    logger.info(f"Tool calling records updated: {dependent_tool_output_dict.keys()} \033[0m")
            except Exception as e:
                fn_result = json.dumps(f"Error: {str(e)}")
                yield {"type": "tool_error", "content": fn_result, "tool_name": fn_name}
            
            messages_state.append({
                "role": "tool",
                "content": fn_result,
                "tool_call_id": call_id
            }) 