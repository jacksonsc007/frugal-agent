# Frugal Agent

**Frugal-Agent** is a cost-effective agent designed to optimize tool calling and reduce token usage. It supports:

1. **Dependent tool calling** within a single response  
2. **Intermediate Representation (IR)** generation to minimize redundant token consumption  
3. **Instruction refinement**, including polishing user prompts and querying online-serving LLMs for assistance  
4. **Structured output formatting**, organizing instruction-answer pairs into **Obsidian-compatible Markdown**

---

## Core Files Overview

| Folder/File | Description |
|-------------|-------------|
| `tools/` | Training and model saving scripts |
| `serving/` | vLLM serving-related files |
| `utils/env.py` | Core environment class that provides rewards to models |
| `utils/ink_dependent_tool_parser.py` | Custom tool parser identifying `call_sequence_id` |
| `utils/reward_functions.py` | Reward functions used during training |
| `utils/UnslothGRPOTrainer_modified.py` | Custom trainer tailored for this project |
| `utils/arsenal.py` | Helper functions for parsing and invoking tool calls |

---

## 🛠️ Build

To build the project, run:

```fish
bash build.sh
```

> ⚠️ **Note**: When installing `vLLM`, ensure you're on the correct branch (`ink-branch-based-on-0.7.3`).  
> Otherwise, you may encounter a `setuptools-scm` error:
>
> ```
> Failed to get the base commit in the main branch. Using the nightly wheel. The libraries in this wheel may not be compatible with your dev branch: Command '['git', 'merge-base', 'main', 'ink-branch-based-on-0.7.3']' returned non-zero exit status 128.
> ```

---

## 🧪 Training

Training is powered by [`Unsloth`](https://github.com/unslothai/unsloth) for fast fine-tuning.

> ⚠️ **Note**: Currently, only **1 GPU** is supported. If you encounter OOM issues, adjust the training configuration in the `configs` directory.  
> The default config has been tested on a single **RTX 4090**.

### Train the Formatter Model

Activate the virtual environment before running training scripts:

```fish
source .venv/bin/activate.fish
```

This model will:
- Summarize and organize question-response pairs  
- Generate Obsidian-compatible YAML metadata  

### Prepare Datasets

Run the notebook `datasets/alpaca/load_save_data.ipynb` to process the `alpaca` dataset.  
Customize dataset settings in `config/formatter/training_config.py`.

### Start Training

```fish
python tools/train_formatter-unsloth.py
```

### Train the Frugal Agent

Train an agent capable of generating tool calls using intermediate representations (IR):

```fish
export EXP_NAME="test"
python tools/train_agent-unsloth.py
```

---

## 💾 Save LoRA Weights

Export trained LoRA weights for deployment:

```fish
python tools/inference_and_save_lora-unsloth.py --model_checkpoint <checkpoint_path> --output_path lora_models/ --model_name <model_name>
```

---

## 🚀 Serving with vLLM

Launch the vLLM server:

```fish
bash serve_vllm.sh
```

### Custom Modifications for vLLM

To support the **Frugal Agent** in production, the following customizations were made:

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

Interact with the deployed model through a web interface using Gradio:

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
- [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2)
- [nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3)
- [Atsunori/HelpSteer2-DPO](https://huggingface.co/datasets/Atsunori/HelpSteer2-DPO)
- [facebook/natural_reasoning](https://huggingface.co/datasets/facebook/natural_reasoning)