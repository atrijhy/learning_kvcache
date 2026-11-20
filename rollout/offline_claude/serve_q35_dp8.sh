#!/usr/bin/env bash
set -eo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate kvcache
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
exec python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 18764 \
 --model /home/liyu.468/models/Qwen3.5-4B --served-model-name qwen3.5-4b --dtype bfloat16 \
 --language-model-only --enforce-eager --tensor-parallel-size 1 --data-parallel-size 8 \
 --data-parallel-size-local 8 --data-parallel-backend mp --data-parallel-rpc-port 18784 \
 --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization 0.60 \
 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
 --default-chat-template-kwargs '{"enable_thinking":true}'
