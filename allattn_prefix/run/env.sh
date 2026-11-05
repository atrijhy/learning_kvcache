# AllAttn-G512A64 全流程的路径配置。换机器只改这个文件；其余脚本都 source 它。
# 用法：先编辑下面的变量，再 `source allattn_prefix/run/env.sh` 检查无报错。
export REPO_ROOT=${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}   # 本仓库根目录
export KV_PY=${KV_PY:-/root/kv_venv/bin/python}   # 训练/导出/评测共用的 python（kv_env.tar 解压到 / 后就是这个路径）
export BASE_MODEL=${BASE_MODEL:-/data/models/Qwen3.5-4B}               # 原版 Qwen3.5-4B（也是 teacher）
export MODELS=${MODELS:-/data/kvmodels}                                # 构建出来的初始化目录、导出的 prefix 模型
export WORK=${WORK:-/data/allattn_runs}                                # 日志、checkpoint、rollout
export EVAL_OUT=${EVAL_OUT:-$WORK/eval}                                # 评测结果（每个模型一个子目录）
# 三个工具环境服务的数据（训练时 rollout 要调用；端口固定 8899/8900/8901，写死在训练数据里）
export AWM_DATA_DIR=${AWM_DATA_DIR:-$REPO_ROOT/awm_data}
export ENVSCALER_DATA_DIR=${ENVSCALER_DATA_DIR:-$REPO_ROOT/envscaler_data}
export DEEPCODER_TACO_DATA_DIR=${DEEPCODER_TACO_DATA_DIR:-$REPO_ROOT/data/deepcoder_taco/177913a7bd43791646ef6a43645caa3c871ab3db}

export VENV_BIN=$(dirname "$KV_PY")
export PLUGIN=$REPO_ROOT/allattn_prefix/opsd_plugin
export DATA=$REPO_ROOT/traj_data/opsd_mixed_1625_v1
export TRAINED=${TRAINED:-/data/transfer/trained}                       # assets.tar 里主机器训好的 prefix
mkdir -p "$MODELS" "$WORK" "$EVAL_OUT"
for p in "$KV_PY" "$BASE_MODEL/config.json" "$DATA/train.parquet" "$PLUGIN/sitecustomize.py"; do
  [[ -e $p ]] || echo "[env.sh] 缺少 $p" >&2
done
