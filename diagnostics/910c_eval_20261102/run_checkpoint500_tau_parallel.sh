#!/usr/bin/env bash
set -uo pipefail
cd /home/liyu.468/learning_kvcache
OUT="$PWD/outputs/eval_qwen3_prefix/checkpoint-500/smoke"
pids=()
domains=(retail airline)
for domain in "${domains[@]}"; do
  bash benchmark/run.sh --benchmark tau2 --tau2-domain "$domain" --task-ids 0 \
    --openai-base-url http://127.0.0.1:18160/v1 --model qwen3-kv-prefix \
    --tau2-agent-thinking --temperature 0.7 --top-p 0.95 \
    --llm-max-completion-tokens 10000 --tau2-max-steps 30 --concurrency 1 \
    --tau2-require-external-user --tau2-require-external-judge \
    --output-dir "$OUT/tau_$domain" > "$OUT/tau_$domain.log" 2>&1 &
  pids+=("$!")
done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"
  printf 'tau_%s_exit=%s\n' "${domains[$i]}" "$?" >> "$OUT/exit_codes.txt"
done
