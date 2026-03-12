#!/bin/bash

# Simple helper script to install and start vLLM with Qwen2.5-14B-Instruct
# on a Vast.ai instance. Paste this into the Vast terminal or run as a script.

pip install vllm huggingface_hub

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --quantization awq \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000

