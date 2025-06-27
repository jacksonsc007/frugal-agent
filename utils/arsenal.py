from typing import Annotated
from openai import OpenAI
from openai.lib.streaming.chat._completions import ChatCompletionStreamState
import openai
import json
import re
import os

def question_answer_expert(
        query: str
):
    """ An expert excels at providing detailed explanations and answers to questions.
    Args:
        query: The question users have ask.
    
    Returns:
        str: The detailed answer to the question.
    """
    # print("\033[96m[TOOL] question_answer_expert tool is called \033[0m")
    # return "question_answer_expert is not initialized"
    # return "The main theme of *'The Most Dangerous Game'* by Richard Connell is **the thin line between civilization and savagery**. The story explores how quickly societal morals can be stripped away when survival is at stake. It questions the ethics of hunting for sport by turning the tables—making a human the prey—and shows how even the most refined individuals can become savage when pushed to the brink. It also delves into themes of **power, violence, and the instinct for self-preservation**."
    return """

Dispatch width refers to the number of instructions a processor can issue (send to execution units) in a single clock cycle. It's a crucial factor determining a processor's instruction-level parallelism (ILP) capabilities and, consequently, its performance.

Here's a breakdown:

    Issuing Instructions: Modern processors don't simply execute instructions in the exact order they appear in the program. They try to find independent instructions and execute them concurrently to improve speed. This process of selecting and sending instructions to available execution units (like ALUs, FPUs, load/store units) is called "issuing" or "dispatching."

    Dispatch Width as a Limit: The dispatch width defines the maximum number of instructions that can be issued simultaneously. A processor with a dispatch width of 4 can, at most, issue four instructions in one clock cycle. It might issue fewer if there aren't enough independent instructions ready for execution or if there are resource conflicts (e.g., multiple instructions needing the same execution unit).

    Impact on Performance: A wider dispatch width generally leads to higher performance because the processor can exploit more ILP. It can keep more execution units busy and complete more work per clock cycle. However, simply increasing dispatch width isn't a magic bullet. Other factors like instruction dependencies, branch prediction accuracy, and memory access latency also play significant roles.

    Relationship to Other Terms:
        Superscalar Architecture: Processors that can dispatch more than one instruction per cycle are called superscalar. Dispatch width is a key characteristic of superscalar processors.
        Instruction Pipeline: Instructions are processed in a pipeline, going through stages like fetch, decode, issue, execute, and write-back. The dispatch stage is where the processor decides which instructions are ready to move forward in the pipeline.
        Issue Width vs. Retire Width: Issue width (dispatch width) refers to how many instructions enter the execution phase per cycle. Retire width refers to how many instructions complete and write back their results per cycle. Ideally, these should be similar, but they don't have to be identical.

In summary, dispatch width is a measure of a processor's ability to exploit instruction-level parallelism by simultaneously issuing multiple instructions to execution units. A wider dispatch width generally indicates a more powerful processor, although overall performance depends on various other architectural features and program characteristics.
"""

def format_organizer(
    instruction: Annotated[str, "The instruction users input"],
    response: Annotated[str, "The corresponding response from the LLM"],
):
    """ An Organizer which organize instruction and response pairs into specific format.
    Args:
        instruction: The instruction users input.
        response: The corresponding response from the LLM.

    Returns:
        str: The organized instruction and response pairs.
    """
    # print("\033[96m[TOOL] format_organizer tool is called \033[0m")
    # return "format organizer is not initialized"
    # return "The formatted content is: The main theme of *'The Most Dangerous Game'* by Richard Connell is **the thin line between civilization and savagery**. The story explores how quickly societal morals can be stripped away when survival is at stake. It questions the ethics of hunting for sport by turning the tables—making a human the prey—and shows how even the most refined individuals can become savage when pushed to the brink. It also delves into themes of **power, violence, and the instinct for self-preservation**."
    return """
---
Dispatch width refers to the number of instructions a processor can issue (send to execution units) in a single clock cycle. It's a crucial factor determining a processor's instruction-level parallelism (ILP) capabilities and, consequently, its performance.

Here's a breakdown:

    Issuing Instructions: Modern processors don't simply execute instructions in the exact order they appear in the program. They try to find independent instructions and execute them concurrently to improve speed. This process of selecting and sending instructions to available execution units (like ALUs, FPUs, load/store units) is called "issuing" or "dispatching."

    Dispatch Width as a Limit: The dispatch width defines the maximum number of instructions that can be issued simultaneously. A processor with a dispatch width of 4 can, at most, issue four instructions in one clock cycle. It might issue fewer if there aren't enough independent instructions ready for execution or if there are resource conflicts (e.g., multiple instructions needing the same execution unit).

    Impact on Performance: A wider dispatch width generally leads to higher performance because the processor can exploit more ILP. It can keep more execution units busy and complete more work per clock cycle. However, simply increasing dispatch width isn't a magic bullet. Other factors like instruction dependencies, branch prediction accuracy, and memory access latency also play significant roles.

    Relationship to Other Terms:
        Superscalar Architecture: Processors that can dispatch more than one instruction per cycle are called superscalar. Dispatch width is a key characteristic of superscalar processors.
        Instruction Pipeline: Instructions are processed in a pipeline, going through stages like fetch, decode, issue, execute, and write-back. The dispatch stage is where the processor decides which instructions are ready to move forward in the pipeline.
        Issue Width vs. Retire Width: Issue width (dispatch width) refers to how many instructions enter the execution phase per cycle. Retire width refers to how many instructions complete and write back their results per cycle. Ideally, these should be similar, but they don't have to be identical.

In summary, dispatch width is a measure of a processor's ability to exploit instruction-level parallelism by simultaneously issuing multiple instructions to execution units. A wider dispatch width generally indicates a more powerful processor, although overall performance depends on various other architectural features and program characteristics.
"""



def save_file(
    file_name: Annotated[str, "The name of file to be saved"],
    content: Annotated[str, "The content of the file to be saved"]
):
    """Save a file
    Args:
        file_name: The name of file to be saved.
        content: The content of the file to be saved.
    
    Returns:
        str: The result of saving the file
    """
    # print("\033[96m[TOOL] save_file tool is called \033[0m")
    try:
        if not(output_dir := os.getenv("OUTPUT_DIR")):
            output_dir = "outputs"
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H")
        output_dir = os.path.join(output_dir, "saver", timestamp)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, file_name), 'w') as file:
            file.write(content)
        # print(f"File {file_name} saved successfully")
        res = f"File {file_name} saved successfully"
    except Exception as e:
        res = f"An error happened when saving the file: {e}"
        print(res)
    return res

TOOLS_NAMES = ["question_answer_expert", "format_organizer", "save_file"]
TOOLS = [
{
    "type": "function",
    "function": {
        "name": "question_answer_expert",
        "description": "An expert excels at providing detailed explanations and answers to questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question users have ask."
                }
            },
            "required": [
                "query"
            ]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "format_organizer",
        "description": "An Organizer which organize instruction and response pairs into specific format.",
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "The instruction users input."
                },
                "response": {
                    "type": "string",
                    "description": "The corresponding response from the LLM."
                }
            },
            "required": [
                "instruction",
                "response"
            ]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "save_file",
        "description": "Save a file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "The name of file to be saved"
                },
                "content": {
                    "type": "string",
                    "description": "The content of the file to be saved"
                }
            },
            "required": [
                "file_name",
                "content"
            ]
        }
    }
}
]

# Existing tool functions (question_answer_expert, format_organizer, save_file)
# Keep all your existing tool function definitions here

def get_function_by_name(name):
    return {
        "question_answer_expert": question_answer_expert,
        "format_organizer": format_organizer,
        "save_file": save_file
    }.get(name)

def try_parse_tool_calls(content: str):
    """Try parse the tool calls. This function is modifed to be compatible with vllm, which require func["arguments"] to be string instead of dict"""
    tool_calls = []
    offset = 0
    pattern = r"<tool_call>\n(.+?)\n</tool_call>"
    for i, m in enumerate(re.finditer(pattern, content, flags=re.DOTALL)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1), strict=False)
            tool_calls.append({"type": "function", "function": func})
            # if isinstance(func.get("arguments"), dict):
            #     func["arguments"] = json.dumps(func["arguments"])
        except Exception as e:
            print(f"\033[96m[Error]  [Tool Parsing Error]: Failed to parse tool calls: the content is:\n {m.group(1)}.\n{e}\033[0m")
    if tool_calls:
        # sort dependent tool calls
        # if all(tool_call.get("call_sequence_id") for tool_call in tool_calls):
        if all(tool_call.get("function").get("call_sequence_id") for tool_call in tool_calls):
            # print("\033[96m[INFO]  \033[0m")
            try:
                tool_calls = sorted(tool_calls, key=lambda x: x.get("function", {}).get("call_sequence_id", 0))
            except Exception as e:
                print(f"\033[96m[Error]  [Tool Parsing Error]: Failed to sort tool calls: {e}\033[0m")
        if offset > 0 and content[:offset].strip():
            c = content[:offset]
        else: 
            c = ""
        # NOTE: The messages should only contain string, we need to convert dict back to string
        for tool_call in tool_calls:
            func = tool_call.get("function")
            if isinstance(func.get("arguments"), dict):
                func["arguments"] = json.dumps(func["arguments"])
        return {"role": "assistant", "content": c, "tool_calls": tool_calls}
    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}

def try_parse_intermediate_representation(args, dependent_tool_output_dict):
    # Replace placeholders (e.g., {1.output}) with actual outputs
    for key, value in args.items():
        if isinstance(value, str):
            if match := re.search(r'\{(\d+)\.output\}', value):
                call_sequence_id = int(match.group(1))
                call_output = dependent_tool_output_dict.get(call_sequence_id, None)
                if call_output is None:
                    Warning(f"Output for call id {call_sequence_id} missing")
                else:
                    Warning(f"Found output for call id {call_sequence_id} in the record.")
                args[key] = re.sub(
                    r'\{(\d+)\.output\}',
                    call_output,
                    value
                )
    return args

def try_invoke_tool_calls(assistant_message: dict, dependent_tool_output_dict: dict = {}):
    tool_call_valid = []
    tool_name = []
    tool_args = []
    tool_responses = []
    tool_call_ids = []
    if "tool_calls" in assistant_message:
        for tool_call in assistant_message["tool_calls"]:
            try:
                fn_name = tool_call["function"]["name"]
                if isinstance(tool_call["function"]["arguments"], dict):
                    fn_args = tool_call["function"]["arguments"]
                else:
                    DeprecationWarning(f"In this version, tool call {tool_call} is using dict for arguments rather than string")
                    fn_args = json.loads(tool_call["function"]["arguments"], strict=False)
                call_sequence_id = tool_call["function"].get("call_sequence_id", None)
            except Exception as e:
                print(f"Error parsing tool call {tool_call}\n {e}")
                tool_call_valid.append(False)
                continue
            try:
                # Execute tool
                fn = get_function_by_name(fn_name)
                fn_args = try_parse_intermediate_representation(fn_args, dependent_tool_output_dict)
                # fn_result = json.dumps(fn(**fn_args))
                # NOTE: the output is string for now
                fn_result = (fn(**fn_args))
                tool_call_valid.append(True)
                tool_name.append(fn_name)
                tool_args.append(fn_args)
                tool_responses.append(fn_result)
                tool_call_ids.append(call_sequence_id)
                if call_sequence_id is not None:
                    if call_sequence_id in dependent_tool_output_dict:
                        raise ValueError(f"Duplicate call_sequence_id: {call_sequence_id}")
                    dependent_tool_output_dict[call_sequence_id] = fn_result
                # Append tool response to state
            except Exception as e:
                print(f"Error: Failed to invoke tool {fn_name} with arguments {fn_args}: {e}")
                tool_call_valid.append(False)
                continue
    return tool_call_valid, tool_name, tool_args, tool_call_ids, tool_responses

if __name__ == "__main__":
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
        debugpy.listen(("0.0.0.0", 61075))
        print("Waiting for debugger to attach...")
        # Optional: Wait for the debugger to attach before continuing execution
        debugpy.wait_for_client()
    response ="""<tool_call>
{"name": "save_file", "arguments": {"file_name": "discussion_on_middle_aged_mental_health.md", "content": "---
tags:
  - {health} health/mental_health
  - {age} age/middle_aged
---

# Instruction
Discuss recent news article about study finding that middle-aged people are more likely to suffer from mental health issues.

# Summary
A conversation about a recent study indicating that middle-aged individuals are more likely to experience mental health issues, discussing potential reasons and the importance of mental health awareness.

## Details
Person 1: Hey, did you read that article in the news today about mental health?  
Person 2: No, I didn't. What was it about?  
Person 1: It was a study that found that middle-aged people are more likely to suffer from mental health issues.  
Person 2: Really? That's interesting. Is there a specific reason why middle-aged people are more prone to these issues?  
Person 1: Well, the article mentioned that one of the factors could be the amount of responsibility and stress that usually comes with that age. You know, things like taking care of a family, a demanding job, and financial burdens.  
Person 2: Yeah, that makes sense. It's important for people to take care of their mental health and seek help if needed.  
Person 1: I completely agree. It's important to have these conversations and raise awareness about mental health."}}
</tool_call>"""
    output = try_parse_tool_calls(response)
    invoke_result, tool_names, tool_args = try_invoke_tool_calls(output)
    print(output)
