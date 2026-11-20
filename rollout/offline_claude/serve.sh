#!/usr/bin/env bash
set -eo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate kvcache
export HF_HUB_OFFLINE=1 ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
exec python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 18764 \
  --model /home/liyu.468/models/Qwen3-4B --served-model-name qwen3-4b \
  --enforce-eager --max-model-len 32768 --gpu-memory-utilization 0.48 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --default-chat-template-kwargs '{"enable_thinking":false}'
