#!/usr/bin/env bash
# Author-table protocol for one model (greedy, as in the AWM/EnvScaler/TACO screenshot): BFCL v3 all, HumanEval /
# MBPP (EvalPlus, base and plus), LCB v6 (n=1, T=0, 16K) + v5 from the same samples. All generation runs concurrently
# against one vLLM server, which is stopped as soon as generation ends; LCB judging continues in the background.
#   greedy_suite.sh <model_dir> <gpus> <port>     (results in /root/eval_local/author_greedy/<basename>)
root=/efs/kadocjiang/learning_kvcache_upstream m=$1 gpus=$2 port=$3 name=$(basename $1)
out=/root/eval_local/author_greedy/$name; mkdir -p $out; cd $root
export PATH=/root/.local/bin:$PATH
CUDA_VISIBLE_DEVICES=$gpus nohup .venv/bin/vllm serve "$m" --served-model-name qwen3.5-4b --host 127.0.0.1 --port $port \
  --trust-remote-code --model-impl vllm --max-model-len 262144 --gpu-memory-utilization 0.85 --dtype bfloat16 \
  --data-parallel-size "$(awk -F, "{print NF}" <<< "$gpus")" --tensor-parallel-size 1 \
  --enable-auto-tool-choice --reasoning-parser qwen3 --tool-call-parser qwen3_coder > $out/vllm.log 2>&1 < /dev/null &
server=$!
until curl -sf -m 5 -X POST "localhost:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.5-4b","messages":[{"role":"user","content":"Reply with OK"}],"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}' >/dev/null; do
  kill -0 $server 2>/dev/null || { echo "vLLM died; see $out/vllm.log"; exit 1; }; sleep 15
done
ev() { .venv/bin/python -m benchmark.eval.run_eval --openai-base-url "http://127.0.0.1:$port/v1" --model qwen3.5-4b \
  --api-key EMPTY --temperature 0 --top-p 1 "$@"; }
ev --benchmark bfcl_v3 --bfcl-model-path "$m" --bfcl-categories all --llm-max-completion-tokens 4096 --concurrency 64 \
  --output-dir $out/bfcl_v3/all > $out/bfcl_v3.log 2>&1 &
for b in humaneval+ mbpp+; do
  ev --benchmark $b --llm-max-completion-tokens 16384 --step-timeout 7200 --concurrency 128 --code-n-samples 1 \
    --code-pass-k 1 --code-eval-workers 24 --code-memory 32g --code-sandbox local --output-dir $out/coding/$b > $out/$b.log 2>&1 &
done
ev --benchmark livecodebench --lcb-release v6 --llm-max-completion-tokens 16384 --step-timeout 7200 --concurrency 128 \
  --code-n-samples 1 --code-pass-k 1 --code-generate-only --code-sandbox local --output-dir $out/coding/lcb_v6_gen \
  > $out/lcb_v6_gen.log 2>&1 &
wait $(jobs -p | grep -v "^$server$")
kill $server; echo "[$(date +%T)] generation done $m"
(
  for r in v6 v5; do
    ev --benchmark livecodebench --lcb-release $r --code-samples $out/coding/lcb_v6_gen/samples.jsonl --code-n-samples 1 \
      --code-pass-k 1 --code-eval-workers 8 --code-memory 32g --code-eval-timeout 14400 --code-sandbox local \
      --output-dir $out/coding/lcb_$r > $out/lcb_$r.log 2>&1 &
  done; wait; echo "[$(date +%T)] judged $m"
) >> /root/local_runs/greedy/judged.log 2>&1 &
