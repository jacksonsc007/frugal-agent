import re
import os
import openai
from openai import OpenAI
from openai.lib.streaming.chat._completions import ChatCompletionStreamState
from typing import Annotated
from utils.sys_prompts import IT_SYS_PROMPT_deepseek_concise_2 as SYSTEM_PROMPT_FORMATTER

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

def get_function_by_name(name):
    return {
        "question_answer_expert": question_answer_expert,
        "format_organizer": format_organizer,
        "save_file": save_file
    }.get(name)

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

def question_answer_expert(
        query: str
):
    """ An expert excels at providing detailed explanations and answers to questions.
    Args:
        query: The question users have ask.
    
    Returns:
        str: The detailed answer to the question.
    """
    print("\033[96m[TOOL] question_answer_expert tool is called \033[0m")
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=os.environ['GOOGLE_API_KEY'],
        base_url="https://generativelanguage.googleapis.com/v1beta/"
    )

    # Define the model and query
    model = 'gemini-2.0-flash'
    system_prompt = "You're an expert in the field of computer science. Please answer the question in a concise manner. Control your answer within 500 words."
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": query
        }
    ]
    stream = client.chat.completions.create(
        model=model,  # Model name to use
        messages=messages,  # Chat history
        stream=True,  # Stream response
    )
    # response = "memory coalescing is a technique used in computer programming to optimize memory access by merging multiple small memory accesses into larger ones."
    partial_message = ""
    for chunk in stream:
        m = (chunk.choices[0].delta.content or "")
        print(m, end= '', flush=True)
        partial_message += m
        yield m
    print("\n", flush=True)
    # return partial_message

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
    print("\033[96m[TOOL] format_organizer tool is called \033[0m")
    try:
        structured_instruction_response = f"""\
<instruction>
{instruction}
</instruction>
<response>
{response}
</response>/
"""        
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_FORMATTER
            },
            {
                "role": "user",
                "content": structured_instruction_response
            }
        ]

        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        model_name = "formatter"
        # Create a chat completion request and send it to the API server
        stream = client.chat.completions.create(
            model=model_name,  # Model name to use
            messages=messages,  # Chat history
            stream=True,  # Stream response
            max_completion_tokens=2048,
            temperature=0.7,
            top_p=0.8,
            extra_body={
                "repetition_penalty": 1.05,
            },
        )

        # Read and return generated text from response stream
        partial_message = ""
        for chunk in stream:
            m = (chunk.choices[0].delta.content or "")
            print(m, end= '', flush=True)
            partial_message += m
            yield m
        print("\n", flush=True)
        # return partial_message
    except Exception as e:
        res = f"An error happened when saving the file: {e}"
        yield res
    # return res


def save_file(
    file_name: Annotated[str, "The name of file to be saved"],
    content: Annotated[str, "The content of the file to be saved"]
):
    """Save a file
    Args:
        file_name: The name of file to be saved
        content: The content of the file to be saved
    
    Returns:
        str: The result of saving the file
    """
    print("\033[96m[TOOL] save_file tool is called \033[0m")
    output_dir = "tmp_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        with open(os.path.join(output_dir, file_name), 'w') as file:
            file.write(content)
        # print(f"File {file_name} saved successfully")
        res = f"File {file_name} saved successfully"
    except Exception as e:
        res = f"An error happened when saving the file: {e}"
    print(res)
    # return res
    yield res
