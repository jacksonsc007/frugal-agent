import re
import json
from utils.logger.logger import mylogger as logger

# Reward functions
# =============================
# 1. formatter reward functions
# =============================
# YAML_REGREX = r"---\ntags:\n\s\s-\s(\w+)\n(?:\s\s-\s(\1\/\w+)\n)+---"
YAML_REGREX = r"^---\ntags:\n(?:  - (\w+)(?:[ ]+\1\/\w+)+\n)+---"
YAML_REGREX_1 = r"^---\n(.*\n)*---"
# YAML_REGREX_2 = r"tags:"
HEADING_LEVEL_REGEX = r"^(#{1,6})\s(.*)$"  # Matches markdown headings and their levels
HEADING_3LEVEL_REGEX = r"^(#{3,6})\s(.*)$"  # Matches markdown headings and their levels
SECTION_REGEX = r"# Instruction|# Summary|## Details"
STRICT_SECTION_REGREX = r"# Instruction\n(.*\n)+# Summary\n(.*\n)+## Details"
OVERALL_REGREX = r"^---\ntags:\n(?:  - (\S+)(?:[ ]+\1\/\S+)+\n)+---(\s)+# Instruction\n(.*\n)+# Summary\n(.*\n)+## Details\n(?!(.*\n)*(## |# ))(.*\n)+"

def yaml_format_reward_func(responses, **kwargs) -> list[float]:
    def strict_yaml_format_reward_func_(text: str):
        match = re.search(YAML_REGREX, text)
        if match:
            return 0.5
        else:
            return 0.0
    def yaml_loose_format_reward_func(text: str):
        match = re.search(YAML_REGREX_1, text)
        if match:
            return 0.25
        else:
            return 0.0
    # def tag_awareness_reward_func(text: str):
    #     match = re.search(YAML_REGREX_2, text)
    #     # show all groups in match
    #     if match:
    #         return 0.125
    #     else:
    #         return 0
    rewards = []
    for r in responses:
        reward = strict_yaml_format_reward_func_(r)
        if (reward == 0):
            reward += yaml_loose_format_reward_func(r)
            # reward += tag_awareness_reward_func(r)
        if reward > 0.5:
            raise ValueError("format reward function should not be greater than 0.5")
        rewards.append(reward)
    logger.info(f"yaml_format reward: {rewards}")
    return rewards

# help model identify general_tag and sub_tag are just placeholders
def placeholder_reward_func(responses, **kwargs) -> list[float]:
    # pattern = r"([\{,\},\S]+)\s+\1/([\{, \},\S]+)"
    tag_pattern = r"(\S+)\s+\1/(\S+)"
    extract_pattern = r"---\ntags:\n(?:  - (\S+)(?:[ ]+\1\/\S+)+\n)+---"

    def func_(text):
        reward = 0
        # first make sure the yaml format is right
        if match := re.search(extract_pattern, text):
          target = match.group()
          # print("yaml content:\n", target)
        #   print("yaml content:")
        #   print(target)
          tag_pairs = re.findall(tag_pattern, target)
          num_pairs = len(tag_pairs)
          if num_pairs == 0:
              return 0
          r_unit = 0.5 / num_pairs
        #   print("extracted tags: ",tag_pairs)
          for (gt, st) in tag_pairs:
              if ('general_tag' not in gt):
                  reward += r_unit
              if ('sub_tag' not in st):
                  reward += r_unit
        return reward
    rewards = []
    for r in responses:
        reward = func_(r)
        rewards.append(reward)
    logger.info(f"placeholder reward: {rewards}")
    return rewards

# incentivize model to generate more tag pair (at most three)
def more_tags_reward_func(responses, **kwargs) -> list[float]:
    # pattern = r"([\{,\},\S]+)\s+\1/([\{, \},\S]+)"
    tag_pattern = r"- (\w+)(?=[\n/ ])"
    # understand the meaning of []
    # \w+ (matches only alphanumeric and underscores) is safer for extracting tags.
    extract_pattern = r"---\ntags:\n(  - .*\n)+---"

    def func_(text):
        true_reward = 0
        # first make sure the yaml format is right
        if match := re.search(extract_pattern, text):
            target = match.group()
            # print("yaml content:\n", target)
            #   print("yaml content:")
            #   print(target)
            tag_pairs = re.findall(tag_pattern, target)
            num_pairs = len(tag_pairs)
            # r_unit = 0.5 / num_pairs
            # print("extracted tags: ",tag_pairs)
            # if all tag pairs are valid, and no repetition
            # more pairs more reward
            general_tags = tag_pairs
            has_repetition = (len(general_tags) != len(set(general_tags)))
            if not has_repetition:
                true_reward = min(num_pairs, 3) * 0.25
        return true_reward
    rewards = []
    for r in responses:
        reward = func_(r)
        rewards.append(reward)
    logger.info(f"more tags reward: {rewards}")
    return rewards



# yaml and markdown requirements are both met
def overall_format_reward_func(responses, **kwargs) -> list[float]:
    # OVERALL_REGREX = r"^---\ntags:\n(?:  - (\S+)(?:[ ]+\1\/\S+)+\n)+---(\s)+# Instruction\n(.*\n)+# Summary\n(.*\n)+## Details\n(?!(.*\n)+(## |# ))(.*\n)+"
    # heading level1 and level2 are not allowed after ##details,
    # unless they appear in codeblocks

    def replace_code_block_content(text):
        # Define a regex pattern to match code blocks
        # This pattern assumes code blocks are enclosed in triple backticks (```)
        code_block_pattern = r'```.*?```'

        # Replace the content within code blocks with empty content
        # The `re.DOTALL` flag allows the `.` to match newline characters as well
        replaced_text = re.sub(code_block_pattern, '', text, flags=re.DOTALL)

        return replaced_text

    def func_(r):
        if re.search(OVERALL_REGREX, r):
            return 0.5
        else:
            return 0
    rewards = []
    for r in responses:
        # we first eliminate codeblocks
        r = replace_code_block_content(r)
        reward = func_(r)
        rewards.append(reward)
    logger.info(f"overall format reward: {rewards}")
    return rewards


# TODO: apply this only when overall format is correct
def logic_heading_hierarchy_func(responses, **kwargs) -> list[float]:
    def heading_hierarchy_func_(text):
        """
        Evaluate the logical hierarchy of heading levels.
        """
        headings = re.findall(HEADING_LEVEL_REGEX, text, re.MULTILINE)
        if not headings:
            return 0  # No headings found

        prev_level = 0
        for level, _ in headings:
            current_level = len(level)  # Number of '#' indicates heading level
            if current_level > prev_level + 1:
                return 0  # Penalize if heading levels are skipped
            prev_level = current_level
        return 0.5
    def bonus_func_(text):
        """
        If heading level 3 or higher appears, and the format is correct,
        return 0.5, otherwise return 0
        """
        # if there are heading level 3 or higher, this avoid model
        # getting rid of the original heading levels
        if (re.findall(HEADING_3LEVEL_REGEX, text, re.MULTILINE)):
            headings = re.findall(HEADING_LEVEL_REGEX, text, re.MULTILINE)
            if not headings:
                return 0  # No headings found

            prev_level = 0
            for level, _ in headings:
                current_level = len(level)  # Number of '#' indicates heading level
                if current_level > prev_level + 1:
                    return 0  # Penalize if heading levels are skipped
                prev_level = current_level
            return 0.5
        # print("overall format is not correct")
        return 0.0
    def replace_code_block_content(text):
        # Define a regex pattern to match code blocks
        # This pattern assumes code blocks are enclosed in triple backticks (```)
        code_block_pattern = r'```.*?```'

        # Replace the content within code blocks with empty content
        # The `re.DOTALL` flag allows the `.` to match newline characters as well
        replaced_text = re.sub(code_block_pattern, '', text, flags=re.DOTALL)

        return replaced_text

    rewards = []
    for r in responses:
        # we should eliminate codeblocks first
        r = replace_code_block_content(r)
        reward = heading_hierarchy_func_(r)
        # Model tends to eliminate headling levels to achieve this reward,
        # we encourage model to keep the headling levels by introducing bonus reward
        bonus = bonus_func_(r)
        reward += bonus
        rewards.append(reward)
    logger.info(f"logic_heading_hierarchy_func reward: {rewards}")
    return rewards

def markdown_format_reward_func(responses, **kwargs) -> list[float]:
    def section_reward_func(text: str):
        match = re.findall(SECTION_REGEX, text, flags=re.MULTILINE)
        num_match = len(match)
        if (num_match > 0 and num_match <= 3):
            return (num_match) * (0.5) / 3
        return 0.0
    def strict_markdown_format_reward_func(text: str):
        match = re.search(STRICT_SECTION_REGREX, text)
        if match:
            return 0.5
        else:
            return 0.0
    rewards = []
    for r in responses:
        reward = strict_markdown_format_reward_func(r)
        rewards.append(reward)
    logger.info(f"markdown format reward: {rewards}")
    return rewards
        
def log_func(prompts, completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    # extracted_responses = [extract_xml_answer(r) for r in responses]
    print(  f"\n\033[92mQuestion\033[0m:\n{q}")
    for id in range(len(responses)):
        print(f"\n\033[93mResponse {id}:\033[0m:\n{responses[id]}")
    return [0.0 for _ in range(len(responses))]


# =============================
# 2. reasoning reward functions
# =============================
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(  f"\n\033[92mQuestion\033[0m:\n{q}",
            f"\n\033[91mAnswer\033[0m:\n{answer[0]}",
            f"\n\033[93mResponse\033[m:\n{responses[0]}",
            f"\n\033[94mExtracted\033[0m:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
                     

# =============================
# 3. multi-step tool calling
# =============================
from .arsenal import try_invoke_tool_calls, try_parse_tool_calls

def dependent_tool_calling_reward_func(prompts, completions, stage_id, **kwargs) -> list[float]:
    """
    0. Only target at calling formatter and saver, for now.
    1. Basic tool calling
       json_format_reward : json format is correct;
       valid_invoke_reward: tool calling is valid.
    2. Dependent tool calling format
        - call_sequence_id_awareness_reward: 
            existence of call_seqnce_id
        - tool_number_reward: 
            awareness of the need to call two tools at once
        - ir_awareness_reward: 
            awareness of intermediate representation
    3. formatter should not contain ir, saver should contain ir
        tool_specific_reward
        
    """
    if stage_id != 1:
        return [0.] * len(completions)

    responses = [completion[0]['content'] for completion in completions]
    def func_(text):
        call_sequence_id_awareness_reward = 0
        ir_awareness_reward = 0
        tool_number_reward = 0
        json_format_reward = 0
        valid_invoke_reward = 0
        call_id_correctness = 0
        ir_content_correctness = 0
        
        output = try_parse_tool_calls(text)
        invoke_result, tool_names, tool_args, _, _ = try_invoke_tool_calls(output, {})
        # print(invoke_result)
        if len(invoke_result) > 0 and (all(invoke_result)):
            valid_invoke_reward = 0.5
        if tool_calls :=output.get("tool_calls"):
            json_format_reward = 0.5
            if len(tool_calls) == 2:
                tool_number_reward = 0.5
            if all(t.get('function', {}).get('call_sequence_id', None) for t in tool_calls):
                call_sequence_id_awareness_reward = 0.5
        # NOTE: we find the model hard to understand the intermediate representation. To encourage 
        # the model to use as more intermediate representation as possible, we offer reward aggressively
        all_match = re.findall(r"\{(\d+)\.output\}", text)
        if len(all_match) > 0:
            ir_awareness_reward = 0.5 * min(len(all_match), 2)
        # print("\033[96m[Congrats] intermediate representation appears! \033[0m")
        
        # ensure call both format_organizer and save_file
        if tool_names == ["format_organizer", "save_file"] or tool_names == ["save_file", "format_organizer"]:
            for name, arg, t in zip(tool_names, tool_args, tool_calls):
                arg_str = json.dumps(arg, indent=2)
                if name == "format_organizer":
                    if t.get('function', {}).get('call_sequence_id', -1) == 2:
                        call_id_correctness += 0.5
                    # if the tool is formatter, we should not see ir
                    # NOTE: to discourage formatter to use {tool_reponse} to slack off
                    match = re.search(r"\{(.+?)\}", arg_str)
                    if match and match.group(1) == "1.output":
                        ir_content_correctness += 0.5
                    # NOTE: we find formatter sometimes use "1.response" to refer to the intermediate representation
                    # Despite the format is not correct, we should encourage this behavior
                    elif match and match.group(1) == "1.response":
                        ir_content_correctness += 0.35
                    # if the content of response is less than 88 characters, we assume the model adopts another way
                    # to use the intermediate representation, we also encourage this behavior 
                    elif arg.get('response') and len(arg['response']) < 88:
                        ir_content_correctness += 0.15
                elif name == "save_file":
                    if t.get('function', {}).get('call_sequence_id', -1) == 3:
                        call_id_correctness += 0.5
                    match = re.search(r"\{(.+?)\}", arg_str)
                    if match and match.group(1) == "2.output":
                        ir_content_correctness += 0.5
                    elif match and match.group(1) == "1.response":
                        ir_content_correctness += 0.25
                    elif arg.get('content') and len(arg['content']) < 88:
                        ir_content_correctness += 0.25
        reward = call_sequence_id_awareness_reward + ir_awareness_reward + tool_number_reward + json_format_reward + valid_invoke_reward + call_id_correctness + ir_content_correctness
        return reward

    rewards = []
    for r in responses:
        reward = func_(r)
        rewards.append(reward)
    return rewards

def ordinary_tool_calling_reward_func(prompts, completions, stage_id, **kwargs) -> list[float]:
    """
    0. Only target at question_answer_expert for now.
    1. If the tools called are correct
    2. tool call should contain call_sequence_id
    """
    responses = [completion[0]['content'] for completion in completions]
    # this reward function does not function for dependent tool calling
    if stage_id == 1:
        return [0.0 for _ in range(len(responses))]
    def func_(text):
        reward = 0
        json_format_reward =0
        valid_invoke_reward = 0
        correct_tool_reward = 0
        tool_call_id_awareness = 0
        tool_call_id_correctness = 0

        output = try_parse_tool_calls(text)
        if output.get("tool_calls"):
            json_format_reward = 0.5
            if all(tool_call.get("function").get("call_sequence_id") for tool_call in output.get("tool_calls")):
                tool_call_id_awareness = 0.5
                if all(tool_call.get("function").get("call_sequence_id") == 1 for tool_call in output.get("tool_calls")):
                    tool_call_id_correctness = 0.5
            invoke_result, tool_names, _, _, _ = try_invoke_tool_calls(output, {})
            if (all(invoke_result)):
                valid_invoke_reward = 0.5
                if all(tool_name == "question_answer_expert" for tool_name in tool_names):
                    correct_tool_reward = 0.5
        reward = json_format_reward + valid_invoke_reward + correct_tool_reward + tool_call_id_awareness + tool_call_id_correctness
        return reward

    rewards = []
    for r in responses:
        reward = func_(r)
        rewards.append(reward)
    return rewards

def saver_filetype_reward_func(prompts, completions, stage_id, **kwargs) -> list[float]:
    """
    1. Ensure that the filetype is markdown when the user does not specify the filetype or filename
    """
    # This reward func targets at calling saver
    responses = [completion[0]['content'] for completion in completions]
    # For dependent tool call, whose stage_id== 1
    if stage_id != 1:
        return [0.0 for _ in range(len(responses))]
    def func_(prompts, text):
        markdown_filetype_reward = 0
        output = try_parse_tool_calls(text)
        invoke_result, tool_names, tool_args, _, _ = try_invoke_tool_calls(output, {})
        for idx, tool_name in enumerate(tool_names):
            if tool_name == "save_file":
                file_name = tool_args[idx]["file_name"]
                file_type = file_name.split(".")[-1]
                if file_type == "md":
                    markdown_filetype_reward = 0.5
        reward = markdown_filetype_reward
        return reward
    rewards = []
    for idx,r in enumerate(responses):
        reward = func_(prompts, r)
        rewards.append(reward)
    return rewards

def saver_content_reward_func(prompts, completions, stage_id, **kwargs) -> list[float]:
    """
    1. content is same as return of formatter
    """
    # This reward func targets at calling saver
    responses = [completion[0]['content'] for completion in completions]
    if stage_id != 2:
        return [0.0 for _ in range(len(responses))]
    tool_stage_id_dict = { "question_answer_expert": 0, "format_organizer": 1, "save_file": 2 }
    def func_(prompts, text):
        content_integrity_reward = 0
        output = try_parse_tool_calls(text)
        invoke_result, tool_names, tool_args = try_invoke_tool_calls(output, {})
        if len(tool_names) != 1 or not invoke_result[0] or tool_names[0] != "save_file":
            return 0.0
        assert prompts[0][-2]['role'] == 'tool', "In this experiment, the second last prompt"
        "before saver should be the output of formatter"
        target_content = prompts[0][-2]['content'].strip()
        arg_content = tool_args[0]["content"].strip()
        if target_content == arg_content:
            content_integrity_reward = 0.5
        # else:
        #     content_integrity_reward = min(0.5 * (1 - abs(len(arg_content) - len(target_content)) / len(target_content)), 0.45)
        reward = content_integrity_reward
        return reward

    rewards = []
    for r in responses:
        reward = func_(prompts, r)
        rewards.append(reward)
    return rewards


def log_func_multi_step(prompts, completions, stage_id, **kwargs) -> list[float]:

    responses = [completion[0]['content'] for completion in completions]
    q = ''.join(
        [
            f"\n=> {p['role']}:\n{p['content']}\n" if p.get( 'content' ) else f"\n=> {p['role']} tool_calling:\n{p['tool_calls']}\n" for p in prompts[0] ]
        )
    print("="*100)
    print(  f"\n\033[92mQuestion stage-{stage_id}\033[0m:\n{q}")
    for idx, r in enumerate(responses):
        print(
            f"\n\033[93mResponse {idx} stage-{stage_id}\033[m:\n{r}",
        )
    # print(
    #     f"\n\033[93mResponse 0 stage-{stage_id}\033[m:\n{responses[0]}",
    # )
    print("="*50 + "reward summary" + "="*50)
    # saver_content_rewards = saver_content_reward_func(prompts, completions, stage_id)
    # saver_filetype_rewards = saver_filetype_reward_func(prompts, completions, stage_id)
    tool_calling_rewards = ordinary_tool_calling_reward_func(prompts, completions, stage_id)
    dependent_tool_calling_rewards = dependent_tool_calling_reward_func(prompts, completions, stage_id)
    
    # print(f"\n\033[94mSAVER CONTENT REWARD\033[0m:\n{saver_content_rewards}")
    # print(f"\n\033[94mSAVER FILETYPE REWARD\033[0m:\n{saver_filetype_rewards}")
    print(f"\n\033[94mTOOL CALLING REWARD\033[0m:\n{tool_calling_rewards}")
    print(f"\n\033[94mDependent TOOL CALLING REWARD\033[0m:\n{dependent_tool_calling_rewards}")
    print("="*100)
    return [0.] * len(responses)