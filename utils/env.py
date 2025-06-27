import copy
from typing import List, Dict, Any, Tuple, Sequence
from functools import partial
import json
from vllm import LLM, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest
from .arsenal import get_function_by_name, try_parse_tool_calls, try_parse_intermediate_representation
from collections import defaultdict
import sys
sys.path.append(".")
from utils.sys_prompts import IT_SYS_PROMPT_deepseek_concise_2 as SYSTEM_PROMPT_FORMATTER
from utils.sys_prompts import MASTERMIND_SYS_PROMPT_DEPENDENT_TOOL_CALLS_3, MASTERMIND_SYS_PROMPT_DEPENDENT_TOOL_CALLS_GENERAL_FEWSHOT
from utils.chat_template import Ink_QWEN_DEPENDENT_TOOL_CALL_TEMPLATE as vllm_chat_template


class ToolCallingEnv:
    """
    Example Environment that:
      1) Sends an initial user prompt to the LLM.
      2) Appends the assistant's reply and a follow-up user query: "Are you sure?".
      3) Sends everything again to the LLM for a final response.
      4) Returns just the completion tokens for each prompt.
    """
    def __init__(self, tools) -> None:
        self.tools = tools
    def step(
        self,
        states: List[Dict[str, Any]],
        llm: LLM,
        sampling_params: SamplingParams,
        data: List[Dict],
        lora_request: LoRARequest
    ) -> Tuple[List[Dict[str, Any]], List[RequestOutput]]:
        chat_with_tools = partial(llm.chat, sampling_params=sampling_params, tools=self.tools, lora_request=lora_request, chat_template=vllm_chat_template)
        # the initial prompt as the first tool_calling request
        # the prompt must be same
        if not all([d == data[0] for d in data]):
            raise ValueError("The inputdata must be same among generations")
        prompt = data[0]['prompt']
        sys_prompt = prompt[0]
        sys_prompt_for_dependent_tool_calls ={
                "role": "system",
                "content": MASTERMIND_SYS_PROMPT_DEPENDENT_TOOL_CALLS_3
        }
        init_user_prompt = prompt[1]
        # init system prompt
        for s in states:
            s['messages'].append(sys_prompt)
            # s['messages'].append(sys_prompt_for_dependent_tool_calls)
            # here is few-shot examples
            use_few_shot = False
            if use_few_shot:
                s['messages'].extend(MASTERMIND_SYS_PROMPT_DEPENDENT_TOOL_CALLS_GENERAL_FEWSHOT)
#                 s['messages'].extend(
#                     [
#                         {
#                             "role": "user",  
#                             "content": "Please format and save the following content: what is avx? AVX (Advanced Vector Extensions) is an x86 CPU instruction set boosting floating-point and SIMD performance for tasks like multimedia/scientific computing. AVX2 adds integer ops; AVX-512 doubles vector width to 512 bits."},
#                         {
#                             "role": "assistant", 
#                             "content": """<tool_call>
# {"name": "format_organizer", "arguments": {"instruction": "What is AVX?", "response": "AVX (Advanced Vector Extensions) is an x86 CPU instruction set boosting floating-point and SIMD performance for tasks like multimedia/scientific computing. AVX2 adds integer ops; AVX-512 doubles vector width to 512 bits."}, "call_sequence_id": 1}
# </tool_call>
# <tool_call>
# {"name": "save_file", "arguments": {"file_name": "avx_explanation.txt", "content": "{1.output}"}, "call_sequence_id": 2}
# </tool_call>"""
#                         },
#                     ]
#                 )
        tool_calling_instructions = [
            # should call 'expert'
            init_user_prompt,
            # should call 'formatter'
            {
                "role": "user", 
                "content": "please format and save the formatted answer"
            },
            # should call 'saver'
            # {
            #     "role": "user", 
                # choice 1: concrete instruction
                # "content": "please save the formatted answer into a file named training_multi_step_tool_calling.md",
                # choice 2: instruction from a slothful user
                # "content": "save it to a file named training_multi_step_tool_calling.md",
                # choice 3: instruction from an inveterately slothful user who is too lazy to be saved
                # "content": "save it",
            # },
        ]
        assert len(tool_calling_instructions) == 2, "In this exp, we only consider two tool chain"
        for stage_id, ins in enumerate(tool_calling_instructions): # ins: instructions
            for s in states:
                s['messages'].append(ins)
            input_messages = [s["messages"] for s in states]
            # only process unfinished states
            outputs = chat_with_tools(copy.deepcopy(input_messages))
            for idx, output in enumerate(outputs):
                # Track prompt_tokens to later slice out the completion part
                if output.outputs[0].finish_reason == "length":
                    # raise ValueError("The generation is too long, please increase the max length")
                    print("\033[96m[Truncation Warning]  The generation is too long, please increase the max length\033[0m")
                state = states[idx]
                state["prompt_token_ids"].append(output.prompt_token_ids)
                state["completion_token_ids"].append(output.outputs[0].token_ids)
                state["prompt_texts"].append(output.prompt)
                state["prompt_messages"].append(copy.deepcopy(state["messages"]))
                state["completion"].append(output.outputs[0].text)
                if stage_id == 0:
                    if_use_tool, tool_name = self.try_parse_invoke_tool_calls(state["messages"], output, data[idx], llm, sampling_params, stage_id)
                    state["tool_call"].append(if_use_tool)
            if stage_id == 0:
                # we manually align the input messages for next tool_calling even if therer is one generation getting the correct tool_calling format.
                # This is a hack to make sure the input prompts are identical among generations.
                # since applying grpo requires the same input but distinct output
                correct_state_idx = None
                for idx, state in enumerate(states):
                    if state['tool_call'][-1]:
                        correct_state_idx = idx
                        break
                if correct_state_idx is None:
                    # stop starting next tool_calling, when previous tool_calling generation all failed
                    return states, stage_id
                # sync the input messages among all states before starting next tool call
                # it make sure the prompt is same for each new tool call
                for idx, state in enumerate(states):
                    if idx != correct_state_idx:
                        states[idx]["messages"] = copy.deepcopy(states[correct_state_idx]["messages"])
        return states, stage_id


    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm: LLM,
        sampling_params: SamplingParams,
        data: List[Dict],
        lora_request: LoRARequest
    ) -> List[Sequence[int]]:
        # to avoid in-place modification of the original prompts
        prompts = copy.deepcopy(prompts)
        # Setup conversation states
        states = [ ]
        for p in prompts:
            state = defaultdict(list)
            states.append(state)
        states,stage_idx = self.step(states, llm, sampling_params, data, lora_request)
        # gather completions reaching the same stage
        aligned_prompt_completion_pairs = []
        for stage_idx in range(stage_idx+1):
            aligned_prompt_completion_pairs.append(
                # we've made sure the prompts are identical
                {
                    "prompt_token_ids"    : states[0] [ "prompt_token_ids"     ][stage_idx] ,
                    "prompt_texts"        : states[0] [ "prompt_texts"         ][stage_idx],
                    "prompt_messages"     : [ state [ "prompt_messages" ] [ stage_idx ] for state in states ] ,
                    "completion_token_ids": [ state [ "completion_token_ids" ] [ stage_idx ] for state in states ] ,
                    "completion"          : [ state [ "completion"           ] [ stage_idx ] for state in states ] ,
                }
            )
        
        return aligned_prompt_completion_pairs

    def convert_to_vllm_format(self, states):
        """vLLM requires the arguement of tool_calling to be string,
        and it then transform it to dict in place. Thus, if we pass the
        orginal states into llm.chat, the states will be modfied to dict.
        This function is used to recover it back to string.
        """
        for state in states:
            for m in state['messages']:
                if 'tool_calls' in m:
                    if isinstance(m['tool_calls'][0]['function']['arguments'], dict):
                        m['tool_calls'][0]['function']['arguments'] = json.dumps(m['tool_calls'][0]['function']['arguments'])

    def try_parse_invoke_tool_calls(self, messages, final_response, data, llm, sampling_params, stage_id) -> bool:
        """parse and invoke tools"""
        response = final_response.outputs[0].text
        assistant_message = try_parse_tool_calls(response)

        # assistant_message = final_response.choices[0].message.model_dump()
        messages.append(assistant_message)

        if_use_tool = False
        fn_name = None
        dependent_tool_output_dict = {}
        # invoke tools
        assert stage_id == 0, "In this exp, we only consider two tool chain. Only question_answer_expert get a chance to be invoked"
        if "tool_calls" in assistant_message:
            if len(assistant_message["tool_calls"]) != 1:
                # currently this function only support to invoke single call of question_answer_expert
                return if_use_tool, fn_name
            for tool_call in assistant_message["tool_calls"]:
                try:
                    fn_name = tool_call["function"]["name"]
                    fn_args = json.loads(tool_call["function"]["arguments"], strict=False)
                    call_sequence_id =tool_call["function"].get("call_sequence_id")
                    if call_sequence_id is not None and not isinstance(call_sequence_id, int):
                        print(f"\033[96m[Error] tool call id should be of type int, but we got {type(call_sequence_id)} Please check the parsing logic. \033[0m")
                        print(call_sequence_id)
                        
                except Exception as e:
                    print(f"Error parsing tool call {tool_call}\n {e}")
                    continue
                try:
                    # Execute tool
                    fn = get_function_by_name(fn_name)
                    
                    if call_sequence_id is not None:
                        fn_args = try_parse_intermediate_representation(fn_args, dependent_tool_output_dict)
                    
                    # NOTE: the output is string for now
                    # fn_result = json.dumps(fn(**fn_args))
                    fn_result = (fn(**fn_args))
                    """ NOTE:The following is a hack.
                    For question_answer_expert, we use the gt answer as a pseudo answer from expert
                    For formatter, We use the llm to generate formatted result.
                    
                    The above fn_result = fn(**fn_args) make sure the argument is passed correctly, or the exception is captured.
                    """
                    # NOTE:Since we intend to call formatter and saver both, they are in the same stage. In this case, we # only have two stages and we do not need the response from formatter. 
                    if fn_name == "question_answer_expert":
                        fn_result = data['output']
                        if_use_tool = True
                        if call_sequence_id is not None:
                            dependent_tool_output_dict[call_sequence_id] = fn_result
                        # Append tool response to state
                        messages.append({
                            "role": "tool",
                            "content": fn_result,
                            # "tool_call_id": call_id
                        })
                except Exception as e:
                    print(f"Error: Failed to invoke tool {fn_name} with arguments {fn_args}: {e}") 
        return if_use_tool, fn_name


class DoubleCheckEnv:
    """
    Example Environment that:
      1) Sends an initial user prompt to the LLM.
      2) Appends the assistant's reply and a follow-up user query: "Are you sure?".
      3) Sends everything again to the LLM for a final response.
      4) Returns just the completion tokens for each prompt.
    """

    def step(
        self,
        states: List[Dict[str, Any]],
        llm: LLM,
        sampling_params: SamplingParams
    ) -> Tuple[List[Dict[str, Any]], List[RequestOutput]]:
        # First LLM call for each state's messages
        outputs = llm.chat([s["messages"] for s in states], sampling_params=sampling_params)
        for i, state in enumerate(states):
            state["messages"].append({
                "role": "assistant", 
                "content": outputs[i].outputs[0].text
            })
            state["messages"].append({
                "role": "user", 
                "content": "Are you sure?"
            })
            # Track prompt_tokens to later slice out the completion part
            state["prompt_tokens"] = len(outputs[i].prompt_token_ids)

        # Second LLM call after "Are you sure?" is appended
        outputs = llm.chat([s["messages"] for s in states], sampling_params=sampling_params)
        for i, state in enumerate(states):
            state["messages"].append({
                "role": "assistant", 
                "content": outputs[i].outputs[0].text
            })
            state["completed"] = True

        return states, outputs

    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm: LLM,
        sampling_params: SamplingParams
    ) -> List[Sequence[int]]:
        # to avoid in-place modification of the original prompts
        prompts = copy.deepcopy(prompts)
        # Setup conversation states
        states = [{"messages": p, "completed": False, "prompt_tokens": -1} for p in prompts]
        outputs = [None] * len(prompts)

        # Keep stepping until each conversation is marked complete
        while not all(s["completed"] for s in states):
            states, outputs = self.step(states, llm, sampling_params)

        # Gather prompt+completion IDs, then slice out the prompt portion
        all_ids = [
            list(o.prompt_token_ids) + list(o.outputs[0].token_ids) 
            for o in outputs
        ]
        completion_ids = [
            all_ids[i][states[i]["prompt_tokens"]:] 
            for i in range(len(prompts))
        ]
        return completion_ids