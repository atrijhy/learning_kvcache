#!/usr/bin/env bash
# kv_env.tar、eval.tar 解压后运行一次，把原机器写死的绝对路径改成本机路径：
#  1) kv_venv 里 verl / OpenEnv / prefix_tuning(vLLM 插件) / ms-swift 是 editable 安装，指向 /efs/kadocjiang/learning_kvcache
#  2) 代码评测环境 benchmark/coding/.venv 指向原机器的 benchmark/runtime/python
source "$(dirname "$0")/env.sh"
SP=$(dirname "$KV_PY")/../lib/python3.11/site-packages
sed -i "s|/efs/kadocjiang/learning_kvcache/|$REPO_ROOT/|g" "$SP"/__editable__*.pth "$SP"/__editable___*finder.py
grep -h "$REPO_ROOT" "$SP"/__editable__*.pth | head -3
R=$REPO_ROOT/benchmark/runtime/python; PYD=$R/cpython-3.11.16-linux-x86_64-gnu
[[ -x $PYD/bin/python3.11 ]] || { echo "缺少 $PYD（eval.tar 没解压到 $REPO_ROOT？）"; exit 1; }
ln -sfn "$PYD" "$R/cpython-3.11-linux-x86_64-gnu"
V=$REPO_ROOT/benchmark/coding/.venv
sed -i "s|^home = .*|home = $R/cpython-3.11-linux-x86_64-gnu/bin|" "$V/pyvenv.cfg"
for n in python python3 python3.11; do [[ -L $V/bin/$n ]] && ln -sfn "$R/cpython-3.11-linux-x86_64-gnu/bin/python3.11" "$V/bin/$n"; done
"$V/bin/python" -c "from evalplus.evaluate import check_correctness; import sys; print('coding venv ok', sys.version.split()[0])"
"$KV_PY" -c "import vllm, verl, torch, prefix_tuning.virtual_prefix.vllm_plugin; print('kv env ok: vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
