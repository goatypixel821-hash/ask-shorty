#!/bin/bash
set -e

echo '=== Installing vLLM ==='
pip install vllm -q

echo '=== Starting Qwen2.5-14B-Instruct ==='
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --quantization awq_marlin \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 36396 \
  --gpu-memory-utilization 0.90

