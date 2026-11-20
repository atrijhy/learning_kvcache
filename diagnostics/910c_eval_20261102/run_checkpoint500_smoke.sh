#!/usr/bin/env bash
set -uo pipefail
cd /home/liyu.468/learning_kvcache
OUT="$PWD/outputs/eval_qwen3_prefix/checkpoint-500/smoke"
MODEL_DIR="$PWD/outputs/envscaler_qwen3_sft_20261102/full_native_lr1e-5/vllm_exports/checkpoint-500-ctx65536"
curl --noproxy '*' -fsS http://127.0.0.1:18160/v1/models > "$OUT/served_model.json" || exit 1

bash benchmark/run.sh --benchmark bfcl_v3 \
  --openai-base-url http://127.0.0.1:18160/v1 --model qwen3-kv-prefix \
  --bfcl-model-path "$MODEL_DIR" --bfcl-categories multi_turn \
  --bfcl-data-dir benchmark/bfcl_v3/data/envscaler_v3 \
  --bfcl-run-ids-file "$OUT/bfcl_ids.json" --temperature 0.7 --concurrency 1 \
  --output-dir "$OUT/bfcl_v3" > "$OUT/bfcl.log" 2>&1
bfcl_code=$?
printf 'bfcl_exit=%s\n' "$bfcl_code" | tee "$OUT/exit_codes.txt"

for domain in retail airline; do
  bash benchmark/run.sh --benchmark tau2 --tau2-domain "$domain" --task-ids 0 \
    --openai-base-url http://127.0.0.1:18160/v1 --model qwen3-kv-prefix \
    --tau2-agent-thinking --temperature 0.7 --top-p 0.95 \
    --llm-max-completion-tokens 10000 --tau2-max-steps 30 --concurrency 1 \
    --tau2-require-external-user --tau2-require-external-judge \
    --output-dir "$OUT/tau_$domain" > "$OUT/tau_$domain.log" 2>&1
  printf 'tau_%s_exit=%s\n' "$domain" "$?" | tee -a "$OUT/exit_codes.txt"
done
