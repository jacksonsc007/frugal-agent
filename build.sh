set -e

# clone sub-modules
git submodule update --init --recursive

# install flash_attn
uv pip install flash-attn --no-build-isolation -v

# install vLLM
pushd third_party/vllm
# Need to manually check branch, or build would faile, check the note below.
git checkout ink-branch-based-on-0.7.3
popd
VLLM_USE_PRECOMPILED=1  uv pip install --no-build-isolation -e third_party/vllm/ -v
