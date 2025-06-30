# Ink's Frugal Agent

**Ink's Frugal Agent** is a cost-effective agent designed to optimize LLM-based interactions while minimizing token usage. It supports:

1. **Multi-tool calling** within a single response
2. **Intermediate representation (IR)** generation to reduce redundant token usage
3. **Instruction refinement**, including polishing user prompts and requesting help from online-serving LLMs
4. **Structured output formatting**, summarizing and organizing instruction-answer pairs into **Obsidian-compatible Markdown**

---

## 🛠️ Build

```fish
bash build.sh
```

> ⚠️ **Note**: When installing `vLLM`, ensure you are on the correct branch (`ink-branch-based-on-0.7.3`).  
> If not, you may encounter an error related to `setuptools-scm`:
> ```
> Failed to get the base commit in the main branch. Using the nightly wheel. The libraries in this wheel may not be compatible with your dev branch: Command '['git', 'merge-base', 'main', 'ink-branch-based-on-0.7.3']' returned non-zero exit status 128.
> ```

---

## 🧪 Training

Training is powered by [`Unsloth`](https://github.com/unslothai/unsloth) for fast fine-tuning.

> ⚠️ Currently, only **1 GPU** is supported. If you run into OOM issues, adjust the training configuration in the `configs` directory.  
> The default config has been tested on a single **RTX 4090**.

### Train the Formatter Model

Trains a model that:
- Summarizes and organizes question-response pairs
- Generates Obsidian-compatible YAML metadata

```fish
python tools/train_formatter.py
```

### Train the Frugal Agent

Trains an agent capable of generating tool calls using intermediate representations (IR):

```fish
export EXP_NAME="test"
python tools/multi_step_tool_calling-unsloth.py
```

---

## 💾 Save LoRA Weights

To export the trained LoRA weights for inference or deployment:

```fish
set CHECKPOINT outputs/exp1
python tools/inference_and_save_lora-unsloth.py --model_checkpoint $CHECKPOINT --output_path lora_models/ --model_name dependent_tool_caller
```

---

## 🚀 Serving with vLLM

Start the vLLM server:

```fish
bash serve_vllm.sh
```

### Custom Modifications for vLLM

To support the **Frugal Agent** in production with **vLLM**, the following customizations were made:

1. **Custom chat template & tool parser** to handle the additional field `call_sequence_id`
   - Located at: `chat_templates/ink_qwen_dependent_tool_call_template.jinja`

2. **New streaming response classes** added to:
   - `third_party/vllm/vllm/entrypoints/openai/protocol.py`:
   ```python
   class DeltaDependentFunctionCall(BaseModel):
       ...

   class DeltaDependentToolCall(OpenAIBaseModel):
       ...

   class DeltaDependentMessage(DeltaMessage):
       ...

   class ChatCompletionDependentResponseStreamChoice(OpenAIBaseModel):
       ...

   class ChatCompletionDependentStreamResponse(OpenAIBaseModel):
       ...
   ```

---

## 💬 Chat via Web Interface

Use Gradio to interact with the deployed model through a web interface:

```fish
python serving/vllm_api-ink_agent-gradio.py
```

---

## 🙌 Acknowledgments

This project builds upon the following open-source tools and frameworks:

- [verifiers](https://github.com/willccbb/verifiers)
- [Unsloth](https://github.com/unslothai/unsloth)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [Cursor](https://cursor.sh/)
