vllm serve Qwen/Qwen2.5-7B-Instruct  \
    --enable-lora --enable-auto-tool-choice \
    --tool-parser-plugin utils/ink_dependent_tool_parser.py \
    --tool-call-parser ink_dependent_tool_parser  \
    --chat-template chat_templates/ink_qwen_dependent_tool_call_template.jinja \
    --lora-modules dependent_tool_caller=lora_models/dependent_tool_caller/ \
    formatter=lora_models/formatter/  --max-lora-rank 32 