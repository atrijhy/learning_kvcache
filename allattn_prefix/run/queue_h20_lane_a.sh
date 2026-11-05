#!/usr/bin/env bash
# H20 第一组（GPU 0-3，评测端口 8181）：E0 base 自检 → E1 32k 截断检查 → E2 换 seed 重复 EnvScaler ×0.01
# 启动：nohup setsid bash allattn_prefix/run/queue_h20_lane_a.sh > $WORK/lane_a.log 2>&1 &
D=$(cd "$(dirname "$0")" && pwd); source "$D/env.sh"
G=0,1,2,3 P=8181
log() { echo "[$(date '+%F %T')] $*"; }
# E0：base 在本机的 16k 评测，必须复现 BFCL 67.93 / HumanEval 90.24 / MBPP 82.01 / LCB 42.75（±0.5）
log "E0 base 16k"; bash "$D/greedy_suite.sh" "$BASE_MODEL" $G $P
# E1：主机器训好的 EnvScaler ×0.01 prefix（assets.tar 里的 trained/）先在 16k 复现，再和 base 一起放宽到 32k
bash "$D/build_model.sh" 0.01 vs01
T=$MODELS/prefix_full_vs01_envscaler_lr1e-5_step67
if [[ ! -d $T ]]; then
  cp -al "$MODELS/Qwen3.5-4B-AllAttn-G512A64-vs01" "$T"; rm "$T/residual_attention_prefix.safetensors"
  cp "$TRAINED/prefix_full_vs01_envscaler_lr1e-5_step67/residual_attention_prefix.safetensors" "$T/"
fi
log "E1 EnvScaler x0.01 16k"; bash "$D/greedy_suite.sh" "$T" $G $P
log "E1 base 32k"; MAX_TOKENS=32768 bash "$D/greedy_suite.sh" "$BASE_MODEL" $G $P
log "E1 EnvScaler x0.01 32k"; MAX_TOKENS=32768 bash "$D/greedy_suite.sh" "$T" $G $P
# E2：同配置换 seed 再训一次，量训练本身的波动（BFCL +1.0 是否超出噪声）
log "E2 EnvScaler x0.01 seed 11"
SEEDS="11 4321 777" bash "$D/train_full.sh" $G envscaler $P "$MODELS/Qwen3.5-4B-AllAttn-G512A64-vs01" vs01s11
log "LANE A DONE"
