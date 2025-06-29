set -e

# clone sub-modules
git submodule update --init --recursive

# Intall basic environment. Flash attention and vLLM is explicitly isolated as they are error-prone.
uv sync

# Install flash_attn
uv pip install flash-attn --no-build-isolation -v

# Install vLLM
# refer to https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#build-wheel-from-source for more details
pushd third_party/vllm
# Need to manually check branch, or build would faile, check the note below.
git checkout ink-branch-based-on-0.7.3
popd
VLLM_USE_PRECOMPILED=1  uv pip install --no-build-isolation -e third_party/vllm/ -v
