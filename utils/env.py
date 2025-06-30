import copy
import random
from typing import List, Dict, Any, Tuple, Sequence
from functools import partial
import json
import torch
import re
from pathlib import Path
from vllm import LLM, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest
from .arsenal import get_function_by_name, try_parse_tool_calls, try_invoke_tool, try_parse_intermediate_representation 
from collections import defaultdict
import sys
sys.path.append(".")
from utils.sys_prompts import SYS_PROMPT_formatter_deepseek_concise_2 as SYSTEM_PROMPT_FORMATTER
from utils.sys_prompts import MASTERMIND_SYS_PROMPT_DEPENDENT_TOOL_CALLS_3, MASTERMIND_SYS_PROMPT_DEPENDENT_TOOL_CALLS_GENERAL_FEWSHOT
from utils.logger.logger import mylogger as logger
from utils.arsenal import try_invoke_tool_calls
from utils.reward_functions import yaml_format_reward_func, markdown_format_reward_func, placeholder_reward_func, logic_heading_hierarchy_func, overall_format_reward_func, more_tags_reward_func, dependent_tool_calling_reward_func, ordinary_tool_calling_reward_func



class ToolCallingEnv:
    """
    An environment favors tool calling with ir (intermediate representation) and <call_sequence_id>.
    """
    def __init__(self, tools) -> None:
        self.tools = tools

        self.num_stages = 2
        self.reward_functions = self.get_reward_functions()
        assert len(self.reward_functions) == self.num_stages
        # the prompts for second stage
        self.chat_template_path = Path("chat_templates/ink_qwen_dependent_tool_call_template.jinja")
        # Load chat template from file
        try:
            with open(self.chat_template_path, 'r', encoding='utf-8') as f:
                self.chat_template = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Chat template file not found at {self.chat_template_path}, using default template")
        except Exception as e:
            raise RuntimeError(f"Error loading chat template from {self.chat_template_path}: {e}")
        self.stage_prompts: List = self.get_stage_prompts()

        self.tool_call_record = {}
        
    """
    Get the reward functions for all stages.
    """
    def get_reward_functions(self) -> List:
        reward_func_stage1 = [ordinary_tool_calling_reward_func]
        reward_func_stage2 = [dependent_tool_calling_reward_func]
        return [reward_func_stage1, reward_func_stage2]
        

    def log_prompt_response_pair(self, messages: Dict, prompt_text: str, responses: List, stage_id: int, friendly_output: bool = False):
        if friendly_output:
            q = ''.join(
                [
                    f"\n=> {p['role']}:\n{p['content']}\n" if p.get( 'content' ) else f"\n=> {p['role']} tool_calling:\n{p['tool_calls']}\n" for p in messages
                ]
            )
            logger.debug("="*100)
            logger.debug(  f"\n\033[92mQuestion stage-{stage_id}\033[0m:\n{q}")
            for idx, r in enumerate(responses):
                logger.debug(
                    f"\n\033[93mResponse {idx} stage-{stage_id}\033[m:\n{r}",
                )
        else:
            logger.debug("="*100)
            logger.debug(  f"\n\033[92mQuestion stage-{stage_id}\033[0m:\n{prompt_text}")
            for idx, r in enumerate(responses):
                logger.debug(
                    f"\n\033[93mResponse {idx} stage-{stage_id}\033[m:\n{r}",
                )
            
        
    """
    Prepare the input prompts for all stages.
    """
    def get_stage_prompts(self) -> List:
        # possible prompts for formatter and saver

        first_stage_prompts = None
        # multiple prompts are prepared
        second_stage_prompts = [
            "please format and save the formatted answer",
            "format and save"
        ]
        prompts = [first_stage_prompts, second_stage_prompts]
        return prompts

    def sample_dev(
        self,
        messages: List[Dict[str, Any]],
        llm: LLM,
        sampling_params: SamplingParams,
        data: List[Dict],
        lora_request: LoRARequest
    ) -> Tuple[List[Dict[str, Any]], List[RequestOutput]]:
        # wrap vllm sampling function
        vllm_sample = partial(
            llm.chat, sampling_params=sampling_params, tools=self.tools, lora_request=lora_request, chat_template=self.chat_template
        )
        # the prompt must be same
        if not all([p == messages[0] for p in messages]):
            raise ValueError("The input prompts must be same")
        states_all_stages = []
        num_generations = len(messages)
        for stage_id in range(self.num_stages):
            valid_generation_idx = -1 # record the index of a correct generation 
            stage_prompt = self.stage_prompts[stage_id]
            if stage_prompt is not None:
                assert isinstance(stage_prompt, list)
                extra_stage_message = {
                    "role": "user",
                    "content": random.choice(stage_prompt)
                }
                for out_idx in range(num_generations):
                    messages[out_idx].append(extra_stage_message)
            # sampling through vLLM
            outputs = vllm_sample(copy.deepcopy(messages))
            completions = []
            # prompts text after applying chat template
            prompt_texts = []
            state = defaultdict(list)
            for out_idx, output in enumerate(outputs):
                if output.outputs[0].finish_reason == "length":
                    logger.error("The generation is truncated, please increase the max length")
                state["prompt_token_ids"].append(output.prompt_token_ids)
                state["completion_token_ids"].append(output.outputs[0].token_ids)
                prompt_text = output.prompt
                prompt_texts.append(prompt_text)
                state["prompt_texts"].append(prompt_text)
                state["prompt_messages"].append(copy.deepcopy(messages[out_idx]))
                comp: str = output.outputs[0].text
                state["completion"].append(comp)
                completions.append(comp)
                """
                Validate if the generation in the first stage is correct. If so, update prompts for next stage with tool response.
                """
                if stage_id == 0:
                    is_valid, tool_name = self.try_parse_invoke_tool_calls(messages[out_idx], comp, data[out_idx], stage_id)
                    # state["tool_call"].append(is_valid)
                    if is_valid:
                        valid_generation_idx = out_idx
            self.log_prompt_response_pair(messages[0], prompt_texts[0], completions, stage_id)
            reward_func_list: list  = self.reward_functions[stage_id]
            num_r_funcs = len(reward_func_list)
            rewards: torch.Tensor = torch.zeros(num_generations, num_r_funcs, device="cuda")
            for i, func in enumerate(reward_func_list):
                reward: list[float] = func(
                    completions=completions, stage_id=stage_id
                )
                rewards[:, i] = torch.tensor(reward, dtype=torch.float32, device="cuda")
            state["rewards"] = rewards
            state["reward_functions"] = reward_func_list
            states_all_stages.append(state)

            """
            Make sure the input prompts are same before proceeding to the second stage.
            We simply duplicate the correct generation in the first stage.
            This is a hack to make sure the input prompts are identical among generations, since applying grpo requires exploration under the same input
            """
            if stage_id == 0:
                if valid_generation_idx == -1:
                    # stop starting next tool_calling, when previous tool_calling generation all failed
                    return states_all_stages
                # sync the input messages among all states before starting next tool call
                # it make sure the prompt is same for each new tool call
                for out_idx in range(num_generations):
                    if out_idx != valid_generation_idx:
                        messages[out_idx] = copy.deepcopy(messages[valid_generation_idx])
        return states_all_stages

    """
    The exlopration phase in RL.
    The model generates a bunch of responses, given the same input prompt.
    vLLM backend is empolyed to speed up generation.
    """
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
        states= self.sample_dev(prompts, llm, sampling_params, data, lora_request)
        return states

    def try_parse_invoke_tool_calls(self, messages, response, data, stage_id) -> bool:
        """parse and invoke tools"""
        assistant_message = try_parse_tool_calls(response)
        # assistant_message = final_response.choices[0].message.model_dump()
        messages.append(assistant_message)

        is_correct = False
        fn_name = None
        # invoke tools
        assert stage_id == 0, "In this exp, we only consider two tool chain. Only question_answer_expert get a chance to be invoked"
        if "tool_calls" in assistant_message:
            if len(assistant_message["tool_calls"]) != 1:
                # currently this function only support to invoke single call of question_answer_expert
                return False, None
            for tool_call in assistant_message["tool_calls"]:
                is_valid, fn_name, fn_args, fn_result = try_invoke_tool(tool_call)
                """ NOTE The following is a hack.
                For question_answer_expert, we use the gt answer as a pseudo answer from expert
                For formatter, We use the llm to generate formatted result.
                The above fn_result = fn(**fn_args) make sure the argument is passed correctly, or the exception is captured.
                """
                if is_valid:
                    # NOTE:Since we intend to call formatter and saver both, they are in the same stage. In this case, we 
                    # only have two stages and we do not need the response from formatter. 
                    if fn_name == "question_answer_expert":
                        fn_result = data['output']
                        is_correct = True
                        # Append tool response to state
                        messages.append({
                            "role": "tool",
                            "content": fn_result,
                        })
        return is_correct, fn_name


class FormatterEnv:
    """
    An environment that favors a well-defined format.
    """
    def __init__(self) -> None:
        self.num_stages = 1
        self.reward_functions = self.get_reward_functions()
        assert len(self.reward_functions) == self.num_stages
        # the prompts for second stage

        self.chat_template_path = Path("chat_templates/ink_qwen_dependent_tool_call_template.jinja")

        # Load chat template from file
        try:
            with open(self.chat_template_path, 'r', encoding='utf-8') as f:
                self.chat_template = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Chat template file not found at {self.chat_template_path}, using default template")
        except Exception as e:
            raise RuntimeError(f"Error loading chat template from {self.chat_template_path}: {e}")

        self.stage_prompts: List = self.get_stage_prompts()

        self.tool_call_record = {}
        
    """
    Get the reward functions for all stages.
    """
    def get_reward_functions(self) -> List:
        reward_func_stage1 = [
            yaml_format_reward_func,
            markdown_format_reward_func,
            placeholder_reward_func,
            logic_heading_hierarchy_func,
            overall_format_reward_func,
            more_tags_reward_func
        ]
        return [reward_func_stage1]
        
    def log_prompt_response_pair(self, messages: Dict, prompt_text: str, responses: List, stage_id: int, friendly_output: bool = False):
        if friendly_output:
            q = ''.join(
                [
                    f"\n=> {p['role']}:\n{p['content']}\n" if p.get( 'content' ) else f"\n=> {p['role']} tool_calling:\n{p['tool_calls']}\n" for p in messages
                ]
            )
            logger.debug("="*100)
            logger.debug(  f"\n\033[92mQuestion stage-{stage_id}\033[0m:\n{q}")
            for idx, r in enumerate(responses):
                logger.debug(
                    f"\n\033[93mResponse {idx} stage-{stage_id}\033[m:\n{r}",
                )
        else:
            logger.debug("="*100)
            logger.debug(  f"\n\033[92mQuestion stage-{stage_id}\033[0m:\n{prompt_text}")
            for idx, r in enumerate(responses):
                logger.debug(
                    f"\n\033[93mResponse {idx} stage-{stage_id}\033[m:\n{r}",
                )
            
        
    """
    Prepare the input prompts for all stages.
    """
    def get_stage_prompts(self) -> List:
        first_stage_prompts = None
        prompts = [first_stage_prompts]
        return prompts

    def sample_dev(
        self,
        messages: List[Dict[str, Any]],
        llm: LLM,
        sampling_params: SamplingParams,
        data: List[Dict],
        lora_request: LoRARequest
    ) -> Tuple[List[Dict[str, Any]], List[RequestOutput]]:
        # wrap vllm sampling function
        vllm_sample = partial(
            llm.chat, sampling_params=sampling_params, lora_request=lora_request, chat_template=self.chat_template
        )
        # the prompt must be same
        if not all([p == messages[0] for p in messages]):
            raise ValueError("The input prompts must be same")
        states_all_stages = []
        num_generations = len(messages)
        for stage_id in range(self.num_stages):
            valid_generation_idx = -1 # record the index of a correct generation 
            stage_prompt = self.stage_prompts[stage_id]
            if stage_prompt is not None:
                assert isinstance(stage_prompt, list)
                extra_stage_message = {
                    "role": "user",
                    "content": random.choice(stage_prompt)
                }
                for out_idx in range(num_generations):
                    messages[out_idx].append(extra_stage_message)
            # sampling through vLLM
            outputs = vllm_sample(copy.deepcopy(messages))
            completions = []
            # prompts text after applying chat template
            prompt_texts = []
            state = defaultdict(list)
            for out_idx, output in enumerate(outputs):
                if output.outputs[0].finish_reason == "length":
                    logger.error("The generation is truncated, please increase the max length")
                state["prompt_token_ids"].append(output.prompt_token_ids)
                state["completion_token_ids"].append(output.outputs[0].token_ids)
                prompt_text = output.prompt
                prompt_texts.append(prompt_text)
                state["prompt_texts"].append(prompt_text)
                state["prompt_messages"].append(copy.deepcopy(messages[out_idx]))
                comp: str = output.outputs[0].text
                state["completion"].append(comp)
                completions.append(comp)
            self.log_prompt_response_pair(messages[0], prompt_texts[0], completions, stage_id)
            reward_func_list: list  = self.reward_functions[stage_id]
            num_r_funcs = len(reward_func_list)
            rewards: torch.Tensor = torch.zeros(num_generations, num_r_funcs, device="cuda")
            for i, func in enumerate(reward_func_list):
                reward: list[float] = func(
                    completions, stage_id=stage_id
                )
                rewards[:, i] = torch.tensor(reward, dtype=torch.float32, device="cuda")
            state["rewards"] = rewards
            state["reward_functions"] = reward_func_list
            states_all_stages.append(state)
        return states_all_stages

    """
    The exlopration phase in RL.
    The model generates a bunch of responses, given the same input prompt.
    vLLM backend is empolyed to speed up generation.
    """
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
        states= self.sample_dev(prompts, llm, sampling_params, data, lora_request)
        return states

    def try_parse_invoke_tool_calls(self, messages, response, data, stage_id) -> bool:
        """parse and invoke tools"""
        assistant_message = try_parse_tool_calls(response)
        # assistant_message = final_response.choices[0].message.model_dump()
        messages.append(assistant_message)

        is_correct = False
        fn_name = None
        # invoke tools
        assert stage_id == 0, "In this exp, we only consider two tool chain. Only question_answer_expert get a chance to be invoked"
        if "tool_calls" in assistant_message:
            if len(assistant_message["tool_calls"]) != 1:
                # currently this function only support to invoke single call of question_answer_expert
                return False, None
            for tool_call in assistant_message["tool_calls"]:
                is_valid, fn_name, fn_args, fn_result = try_invoke_tool(tool_call)
                """ NOTE The following is a hack.
                For question_answer_expert, we use the gt answer as a pseudo answer from expert
                For formatter, We use the llm to generate formatted result.
                The above fn_result = fn(**fn_args) make sure the argument is passed correctly, or the exception is captured.
                """
                if is_valid:
                    # NOTE:Since we intend to call formatter and saver both, they are in the same stage. In this case, we 
                    # only have two stages and we do not need the response from formatter. 
                    if fn_name == "question_answer_expert":
                        fn_result = data['output']
                        is_correct = True
                        # Append tool response to state
                        messages.append({
                            "role": "tool",
                            "content": fn_result,
                        })
        return is_correct, fn_name

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