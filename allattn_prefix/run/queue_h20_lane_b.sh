#!/usr/bin/env bash
# H20 第二组（GPU 4-7，评测端口 8182）：E3 GDN 缩放扫描，EnvScaler ×0.1 → ×0.003（主机器跑 ×0.01、×0.03）
# 启动：nohup setsid bash allattn_prefix/run/queue_h20_lane_b.sh > $WORK/lane_b.log 2>&1 &
D=$(cd "$(dirname "$0")" && pwd); source "$D/env.sh"
G=4,5,6,7 P=8182
log() { echo "[$(date '+%F %T')] $*"; }
for s in 0.1 0.003; do
  tag=vs$(echo $s | tr . p)   # 0.1 → vs0p1，0.003 → vs0p003（vs01 专指 0.01，和主机器同名）
  bash "$D/build_model.sh" $s $tag
  log "E3 EnvScaler x$s"; bash "$D/train_full.sh" $G envscaler $P "$MODELS/Qwen3.5-4B-AllAttn-G512A64-$tag" $tag
done
log "LANE B DONE"
