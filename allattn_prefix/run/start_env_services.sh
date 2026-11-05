#!/usr/bin/env bash
# 启动训练用的三个工具环境：AWM :8899，EnvScaler :8900，TACO 判题 :8901（端口写死在 opsd_mixed 数据里，不能改）。
source "$(dirname "$0")/env.sh"
logs=$WORK/service_logs; mkdir -p "$logs"; cd "$REPO_ROOT/OpenEnv"
export PATH=$VENV_BIN:$PATH
AWM_DATA_DIR=$AWM_DATA_DIR AWM_READY_TIMEOUT=600 PYTHONPATH=src:envs \
  setsid nohup "$KV_PY" -m uvicorn envs.agent_world_model_env.server.app:app --host 0.0.0.0 --port 8899 \
  --ws-ping-interval 20 --ws-ping-timeout 300 > "$logs/awm.log" 2>&1 < /dev/null &
ENVSCALER_DATA_DIR=$ENVSCALER_DATA_DIR PYTHONPATH=src:envs \
  setsid nohup "$KV_PY" -m uvicorn envs.env_scaler_env.server.app:app --host 0.0.0.0 --port 8900 \
  --ws-ping-interval 20 --ws-ping-timeout 300 > "$logs/envscaler.log" 2>&1 < /dev/null &
cd "$REPO_ROOT"
DEEPCODER_TACO_DATA_DIR=$DEEPCODER_TACO_DATA_DIR PYTHONPATH=$REPO_ROOT/OpenEnv/src:$REPO_ROOT/OpenEnv/envs \
  setsid nohup "$KV_PY" -m code_judge_env.server.app > "$logs/code_judge.log" 2>&1 < /dev/null &
for port in 8899 8900 8901; do
  for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 5; done
  curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && echo "port $port up" || echo "port $port NOT up（看 $logs）"
done
