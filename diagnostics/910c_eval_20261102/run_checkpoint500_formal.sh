#!/usr/bin/env bash
# Run record: orchestration only; all generation and scoring use benchmark/run.sh.
set -uo pipefail
cd /home/liyu.468/learning_kvcache
OUT=${1:?absolute output directory required}
[[ "$OUT" = /* && -d "$OUT" && ! -e "$OUT/evaluation_state.json" ]] || exit 2
MODEL_DIR="$PWD/outputs/envscaler_qwen3_sft_20261102/full_native_lr1e-5/vllm_exports/checkpoint-500-ctx65536"
ulimit -n 65536

state() {
  /usr/local/miniconda3/bin/python - "$OUT/evaluation_state.json" "$1" "${2:-null}" <<'PY'
import datetime, json, sys
from pathlib import Path
p, phase, exit_code = sys.argv[1:]
Path(p).write_text(json.dumps({
    "phase": phase, "exit_code": json.loads(exit_code),
    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}, indent=2) + "\n")
PY
}

curl --noproxy '*' -fsS http://127.0.0.1:18160/v1/models > "$OUT/served_model.json" || exit 1
state bfcl_running
bash benchmark/run.sh --benchmark bfcl_v3 \
  --openai-base-url http://127.0.0.1:18160/v1 --model qwen3-kv-prefix \
  --bfcl-model-path "$MODEL_DIR" --bfcl-categories multi_turn \
  --bfcl-data-dir benchmark/bfcl_v3/data/envscaler_v3 \
  --temperature 0.7 --concurrency 128 \
  --output-dir "$OUT/bfcl_v3" > "$OUT/logs/bfcl.log" 2>&1
bfcl_code=$?
printf 'bfcl_exit=%s\n' "$bfcl_code" >> "$OUT/exit_codes.txt"
# Do not spend simulator API calls if the inference service has died.
if ! curl --noproxy '*' -fsS http://127.0.0.1:18160/health > /dev/null; then
  state service_failed 1
  exit 1
fi

state tau_running
pids=()
domains=(retail airline)
for domain in "${domains[@]}"; do
  bash benchmark/run.sh --benchmark tau2 --tau2-domain "$domain" \
    --openai-base-url http://127.0.0.1:18160/v1 --model qwen3-kv-prefix \
    --tau2-agent-thinking --temperature 0.7 --top-p 0.95 \
    --llm-max-completion-tokens 16384 --tau2-max-steps 200 --concurrency 8 \
    --tau2-require-external-user --tau2-require-external-judge \
    --output-dir "$OUT/tau_${domain}" > "$OUT/logs/tau_${domain}.log" 2>&1 &
  pids+=("$!")
done
failed=0
[[ "$bfcl_code" == 0 ]] || failed=1
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"
  code=$?
  printf 'tau_%s_exit=%s\n' "${domains[$i]}" "$code" >> "$OUT/exit_codes.txt"
  [[ "$code" == 0 ]] || failed=1
done
if [[ "$failed" == 0 ]]; then state completed 0; else state failed 1; fi
exit "$failed"
