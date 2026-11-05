#!/usr/bin/env bash
# 构建 AllAttn-G512A64 初始化目录（base 权重硬链接，prefix: K~N(0,0.02)，V=0，初始化与 base 完全等价）。
#   build_model.sh <gdn_value_scale> [tag]   例：build_model.sh 0.01 vs01 → $MODELS/Qwen3.5-4B-AllAttn-G512A64-vs01（不给 tag 时 0.1 → vs0p1）
# 同一 seed 下得到的初始 prefix 与主机器上 vs01 逐字节一致。MODELS 必须和 BASE_MODEL 在同一文件系统（硬链接）。
source "$(dirname "$0")/env.sh"
scale=$1 tag=${2:-vs$(echo "$1" | tr . p)}
out=$MODELS/Qwen3.5-4B-AllAttn-G512A64-$tag
[[ -d $out ]] && { echo "已存在 $out"; exit 0; }
"$KV_PY" "$REPO_ROOT/allattn_prefix/tools/build_allattn_model.py" --base "$BASE_MODEL" --out "$out" \
  --gdn 512 --attn 64 --gdn-value-scale "$scale"
