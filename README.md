

Ink's Frugal Agent


# Build
```fish
export PROJECT_DIR=$PWD
cd third_party
# `main` branch is required when installing trl
git clone https://github.com/jacksonsc007/trl.git
cd trl
git checkout agent
cd $PROJECT_DIR

uv add third_party/trl --editable -v --no-build-isolation
```


```
git submodule update --init --recursive

# install vLLM
pushd third_party/vllm
# Need to manually check branch, or build failed., check the note below.
git checkout ink-branch-based-on-0.7.3
popd
VLLM_USE_PRECOMPILED=1  uv pip install --no-build-isolation -e third_party/vllm/ -v

# install flash_attn
uv pip install flash-attn --no-build-isolation -v
```

> An error occurs when installing vLLM without explicitly checking out to branch `ink-branch-based-on-0.7.3`:
> Failed to get the base commit in the main branch. Using the nightly wheel. The libraries in this wheel may not be compatible with your dev branch: Command '['git', 'merge-base', 'main', 'ink-branch-based-on-0.7.3']' returned non-zero exit status 128.
> This issue is likely caused by the use of the library `setuptools-scm`, which manages building through git meta data.