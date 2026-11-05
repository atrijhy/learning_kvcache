#!/usr/bin/env bash
# greedy 评测一个模型：BFCL v3 all、HumanEval/MBPP（EvalPlus，表里报 base 测试的 pass@1）、LCB v6（n=1，T=0）。
#   greedy_suite.sh <模型目录> <GPU列表> <端口>        结果：$EVAL_OUT/<模型目录名>[_<MAX_TOKENS>]/
# MAX_TOKENS（默认 16384）只改代码题的生成上限；改成 32768 用来检查"截断导致的 LCB 掉点"。
# 所有生成共用一个 vLLM（DP=GPU 数），生成完立刻关掉；LCB 判分在后台继续，结果写进 lcb_v6.log。
source "$(dirname "$0")/env.sh"
m=$1 gpus=$2 port=$3 mt=${MAX_TOKENS:-16384}
name=$(basename "$m"); [[ $mt != 16384 ]] && name=${name}_$mt
out=$EVAL_OUT/$name; mkdir -p "$out"; cd "$REPO_ROOT"
export OPSD_PLUGIN=1 PYTHONPATH=$PLUGIN:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
CUDA_VISIBLE_DEVICES=$gpus setsid nohup "$VENV_BIN/vllm" serve "$m" --served-model-name qwen3.5-4b --host 127.0.0.1 \
  --port $port --trust-remote-code --model-impl vllm --max-model-len 262144 --gpu-memory-utilization 0.85 --dtype bfloat16 \
  --data-parallel-size "$(awk -F, '{print NF}' <<< "$gpus")" --tensor-parallel-size 1 \
  --enable-auto-tool-choice --reasoning-parser qwen3 --tool-call-parser qwen3_coder > "$out/vllm.log" 2>&1 < /dev/null &
server=$!
until curl -sf -m 5 -X POST "localhost:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.5-4b","messages":[{"role":"user","content":"Reply with OK"}],"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}' >/dev/null; do
  kill -0 $server 2>/dev/null || { echo "vLLM 起不来，看 $out/vllm.log"; exit 1; }; sleep 15
done
ev() { "$KV_PY" -m benchmark.eval.run_eval --openai-base-url "http://127.0.0.1:$port/v1" --model qwen3.5-4b \
  --api-key EMPTY --temperature 0 --top-p 1 "$@"; }
pids=()
if [[ $mt == 16384 ]]; then
  ev --benchmark bfcl_v3 --bfcl-model-path "$m" --bfcl-categories all --llm-max-completion-tokens 4096 --concurrency 64 \
    --output-dir "$out/bfcl_v3/all" > "$out/bfcl_v3.log" 2>&1 & pids+=($!)
fi
for b in humaneval+ mbpp+; do
  ev --benchmark $b --llm-max-completion-tokens $mt --step-timeout 7200 --concurrency 128 --code-n-samples 1 \
    --code-pass-k 1 --code-eval-workers 24 --code-memory 32g --code-sandbox local --output-dir "$out/coding/$b" > "$out/$b.log" 2>&1 & pids+=($!)
done
ev --benchmark livecodebench --lcb-release v6 --llm-max-completion-tokens $mt --step-timeout 14400 --concurrency 128 \
  --code-n-samples 1 --code-pass-k 1 --code-generate-only --code-sandbox local --output-dir "$out/coding/lcb_v6_gen" \
  > "$out/lcb_v6_gen.log" 2>&1 & pids+=($!)
wait "${pids[@]}"
kill -- -$(ps -o pgid= $server | tr -d ' ') 2>/dev/null || kill $server; echo "[$(date +%T)] 生成完成 $m"
( ev --benchmark livecodebench --lcb-release v6 --code-samples "$out/coding/lcb_v6_gen/samples.jsonl" --code-n-samples 1 \
    --code-pass-k 1 --code-eval-workers 8 --code-memory 32g --code-eval-timeout 14400 --code-sandbox local \
    --output-dir "$out/coding/lcb_v6" > "$out/lcb_v6.log" 2>&1; echo "[$(date +%T)] LCB 判分完成 $m" ) &
