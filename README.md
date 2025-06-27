

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
```