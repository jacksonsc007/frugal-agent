import copy
from typing import List, Dict, Any, Tuple, Sequence
from functools import partial
import json
from vllm import LLM, SamplingParams, RequestOutput
from .arsenal import get_function_by_name, try_parse_tool_calls


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
        sampling_params: SamplingParams
    ) -> Tuple[List[Dict[str, Any]], List[RequestOutput]]:
        chat_with_tools = partial(llm.chat, sampling_params=sampling_params, tools=self.tools)
        tool_calling_instructions = [
            {
                "role": "user", 
                "content": "please format your answer"
            },
            {
                "role": "user", 
                # choice 1: concrete instruction
                "content": "please save the formatted answer into a file named training_multi_step_tool_calling.md",
                # choice 2: instruction from a slothful user
                # "content": "save it to a file named training_multi_step_tool_calling.md",
                # choice 3: instruction from an inveterately slothful user who is too lazy to be saved
                # "content": "save it",
            }
        ]
        for i, ins in enumerate(tool_calling_instructions): # ins: instructions
            unfinished_idx = []
            unfinished_states = []
            for idx, state in enumerate(states):
                if state["completed"] is False:
                    unfinished_idx.append(idx)
                    unfinished_states.append(state)
            input_messages = [s["messages"] for s in unfinished_states]
            # if former tool_calling failed, no need to proceed
            if len(input_messages) == 0:
                break
            # only process unfinished states
            outputs = chat_with_tools(copy.deepcopy(input_messages))
            for idx, output in zip(unfinished_idx, outputs):
                # Track prompt_tokens to later slice out the completion part
                state = states[idx]
                if (state["prompt_tokens"] == -1):
                    state["prompt_tokens"] = len(output.prompt_token_ids)
                if_use_tool = self.try_parse_invoke_tool_calls(state["messages"], output)
                # if tool is called, we proceed with next tool_calling instruction
                if if_use_tool:
                    state["messages"].append(ins)
                else:
                    state["completed"] = True
        outputs = chat_with_tools([s["messages"] for s in copy.deepcopy(states)])
        for i, (state, output) in enumerate(zip(states, outputs)):
            if not state["completed"]:
                if_use_tool = self.try_parse_invoke_tool_calls(state["messages"], output)
                state["completed"] = True
        return states, outputs

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

    def try_parse_invoke_tool_calls(self, messages, final_response) -> bool:
        """parse and invoke tools"""
        response = final_response.outputs[0].text
        assistant_message = try_parse_tool_calls(response)

        # assistant_message = final_response.choices[0].message.model_dump()
        messages.append(assistant_message)

        if_use_tool = False
        # invoke tools
        if "tool_calls" in assistant_message:
            for tool_call in assistant_message["tool_calls"]:
                # call_id = tool_call["id"]
                fn_name = tool_call["function"]["name"]
                try:
                    fn_args = json.loads(tool_call["function"]["arguments"], strict=False)
                except Exception as e:
                    print(f"Error parsing tool call arguments: {e}")
                    print(f'tool arguements:\n{tool_call["function"]["arguments"]}')
                
                # Execute tool
                fn = get_function_by_name(fn_name)
                try:
                    fn_result = json.dumps(fn(**fn_args))
                except Exception as e:
                    fn_result = json.dumps(f"Error: {str(e)}")
                
                # Append tool response to state
                messages.append({
                    "role": "tool",
                    "content": fn_result,
                    # "tool_call_id": call_id
                })
                if_use_tool = True
        return if_use_tool

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