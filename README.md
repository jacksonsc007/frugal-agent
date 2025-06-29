# Introduction

Ink's Frugal Agent


# Build
```fish
bash build.sh
```

> An error occurs when installing vLLM without explicitly checking out to branch `ink-branch-based-on-0.7.3`:
> Failed to get the base commit in the main branch. Using the nightly wheel. The libraries in this wheel may not be compatible with your dev branch: Command '['git', 'merge-base', 'main', 'ink-branch-based-on-0.7.3']' returned non-zero exit status 128.
> This issue is likely caused by the use of the library `setuptools-scm`, which manages building through git meta data.


# Training
The training is powered by `Unsloth`.
- To train a custom formatter which summarize and organize question-response pairs, along with generating obsidian-compatible yaml meta info:
```fish
```


- To train a frugal agent capable of generate tool calls with ir (intermediate representations):
```fish
export EXP_NAME="test"
python tools/multi_step_tool_calling-unsloth.py
```