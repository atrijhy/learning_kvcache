#!/usr/bin/env bash
# One local listener per immutable checkpoint. Run each in its own terminal.
set -eo pipefail
cd "$(dirname "$0")/../.."
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate kvcache
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH="$PWD:${PYTHONPATH:-}"
base=/home/liyu.468/models/Qwen3-4B
case "$1" in
 router) device=0; port=18764; model="$base"; name=qwen-router ;;
 100|500)
    if [[ $1 = 100 ]]; then device=1; port=18765; else device=2; port=18766; fi
    model="$PWD/outputs/session_harness/exports/step$1"; name="prefix-$1"
    if [[ ! -d "$model" ]]; then
        python -m prefix_tuning.virtual_prefix.export_qwen3_residual_kv_prefix_checkpoint \
          --checkpoint "outputs/envscaler_qwen3_sft_20261102/full_native_lr1e-5/checkpoint-$1" \
          --base-model "$base" --output "$model" --max-model-len 32768
    fi ;;
 *) echo "Usage: bash serve.sh router|100|500" >&2; exit 2 ;;
esac
export ASCEND_RT_VISIBLE_DEVICES="$device"
exec python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port "$port" \
 --model "$model" --served-model-name "$name" --enforce-eager --max-model-len 32768 \
 --gpu-memory-utilization 0.48 --enable-auto-tool-choice --tool-call-parser hermes \
 --default-chat-template-kwargs '{"enable_thinking":false}'
