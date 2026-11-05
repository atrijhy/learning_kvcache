# AllAttn-G512A64 全流程：在一台新机器上从零跑到出结果

本文档面向在另一台 8 卡机器（如 8×H20）上接手实验的人：照着做，就能复现主机器的训练和评测，并跑完第 7 节列出的补充实验。背景和已有结论见同目录的 [README.md](README.md)。

所有命令都在仓库根目录执行。路径只在 `allattn_prefix/run/env.sh` 一个地方配置。

---

## 0. 流程总览

```
拷贝 3 个 tar  →  clone 代码  →  解压 + 修路径  →  起 3 个工具服务  →  base 自检评测
      →  构建初始化模型  →  正式训练（自动导出 + 评测）  →  collect_results.py 汇总
```

- **一个正式实验占 4 张卡**：3 张 actor，1 张 teacher；评测时 4 张卡做数据并行。8 卡机器同时跑两组：GPU 0-3 是第一组，GPU 4-7 是第二组。
- **主机器（8×H100）上的耗时**：

  | 任务 | 训练 | 评测 |
  |---|---|---|
  | EnvScaler | 约 3 h | 约 45 min，另有 LCB 判分 20–30 min 在后台 |
  | AWM | 约 2.3 h | 同上 |
  | TACO | 约 4.5 h | 同上 |

  H20 算力低，粗估慢 1.5–2.5 倍。

## 1. 需要从主机器拷过来的东西

主机器打包好的文件在 `/root/transfer/`，传过来后先核对 `sha256sum -c SHA256SUMS`。

| 文件 | 大小 | 内容 | 解压到 |
|---|---|---|---|
| `kv_env.tar` | ~13 GB | 训练、导出、评测共用的 Python 环境 `/root/kv_venv`（Python 3.11.16、torch 2.10.0+cu128、vLLM 0.18.0），以及它依赖的 uv Python 解释器 | `/`（还原成 `/root/kv_venv` 和 `/root/.local/share/uv/python`） |
| `assets.tar` | ~10 GB | `Qwen3.5-4B/`（base，也是 teacher）、`awm_data/`、`envscaler_data/`、`data/deepcoder_taco/`（三个工具环境的数据）、`trained/`（主机器训好的 EnvScaler ×0.01 prefix） | 任意目录，比如 `/data/transfer` |
| `eval.tar` | ~5 GB | `benchmark/coding`（代码评测的独立环境、官方 LCB/EvalPlus 代码、LCB v6 数据）、`benchmark/bfcl_v3`（BFCL 官方代码）、`benchmark/runtime/python`、`root/.cache/evalplus` | 见 2.2 |

机器要求：

- NVIDIA 驱动支持 CUDA 12.8（≥ 570）。
- 用 root 跑。代码评测的 `--code-sandbox local` 要以 uid 65534 在无网络的 namespace 里执行。
- 每张卡显存 ≥ 80 GB。
- 本地盘约 300 GB 空闲：checkpoint 每 20 步存一次，每个实验约 20 GB。

为什么直接拷环境而不是重装：

- 仓库自带的 `setup_kvcache.sh` 要求 vLLM ≥ 0.22，但本实验全部在 vLLM 0.18.0 上跑。
- 新结构的 vLLM 插件 `vllm_all_attention_prefix.py` 继承了 0.18 的 `Qwen3_5GatedDeltaNet.forward`，换版本可能直接报错。

## 2. 安装

### 2.1 代码

```bash
git clone -b new-structure-allattn-prefix git@github.com:atrijhy/learning_kvcache.git
cd learning_kvcache
```

### 2.2 解压

```bash
tar -C / -xf kv_env.tar                     # → /root/kv_venv、/root/.local/share/uv/python
mkdir -p /data/transfer && tar -C /data/transfer -xf assets.tar
tar -xf eval.tar benchmark/                 # 在仓库根目录解压，补齐 benchmark/coding、bfcl_v3、runtime
tar -C / -xf eval.tar root/.cache/evalplus  # EvalPlus 数据缓存
# 工具环境数据放到仓库里（也可以在 env.sh 里把路径改成 /data/transfer/...）
cp -r /data/transfer/awm_data/. awm_data/
cp -r /data/transfer/envscaler_data envscaler_data
mkdir -p data && cp -r /data/transfer/data/deepcoder_taco data/
```

### 2.3 配置路径

编辑 `allattn_prefix/run/env.sh`，一般只需改这几项：

| 变量 | 含义 | 示例 |
|---|---|---|
| `BASE_MODEL` | base 模型目录 | `/data/transfer/Qwen3.5-4B` |
| `MODELS` | 构建的初始化目录和导出的 prefix 放这里。**必须和 `BASE_MODEL` 在同一个文件系统**，因为 base 权重是硬链接的 | `/data/kvmodels` |
| `WORK` | 日志、checkpoint、rollout | `/data/allattn_runs` |
| `EVAL_OUT` | 评测结果 | 默认 `$WORK/eval` |
| `TRAINED` | 主机器训好的 prefix | `/data/transfer/trained` |

```bash
source allattn_prefix/run/env.sh     # 不报"缺少 ..."就对了
```

### 2.4 修路径（只需一次）

拷来的环境里写死了原机器的路径：

- kv_venv 里 verl、OpenEnv、`prefix_tuning`（vLLM 插件入口）、ms-swift 是 editable 安装，指向原机器的仓库；
- `benchmark/coding/.venv` 指向原机器的解释器。

```bash
bash allattn_prefix/run/fix_eval_env.sh
# 应该看到：coding venv ok 3.11.16 / kv env ok: vllm 0.18.0 torch 2.10.0+cu128 cuda True
```

### 2.5 启动三个工具环境服务

训练 rollout 时要调用这三个服务，端口写死在 opsd_mixed 数据里，不能改：

| 服务 | 端口 |
|---|---|
| AWM | 8899 |
| EnvScaler | 8900 |
| TACO 判题 | 8901 |

```bash
bash allattn_prefix/run/start_env_services.sh   # 三行 "port xxxx up"
```

机器重启后要重新起。训练脚本开始前会检查这三个服务的 `/health`，不通就直接退出。

## 3. 自检：base 评测必须复现

```bash
bash allattn_prefix/run/greedy_suite.sh $BASE_MODEL 0,1,2,3 8181
# 约 1 h 后（LCB 判分再 20-30 min）：
$KV_PY allattn_prefix/run/collect_results.py
```

主机器上 base 的 greedy 结果如下，H20 上应该一致（±0.5）：

| BFCL v3 | multi-turn | relevance | irrelevance | LCB v6 | LCB 截断 | HumanEval | MBPP |
|---|---|---|---|---|---|---|---|
| 67.93 | 45.13 | 66.67 | 82.56 | 42.75 | 584 | 90.24 | 82.01 |

对不上就先别训练。依次排查：

- `$EVAL_OUT/Qwen3.5-4B/vllm.log`：vLLM 是否正常启动；
- `bfcl_v3.log`：BFCL 是否报错；
- `humaneval+.log`：sandbox 是否报权限错误。

## 4. 正式训练

### 4.1 构建初始化模型

```bash
bash allattn_prefix/run/build_model.sh 0.01 vs01    # → $MODELS/Qwen3.5-4B-AllAttn-G512A64-vs01
```

- GDN 层 512 个、attention 层 64 个 soft token；K ~ N(0, 0.02)，V = 0，所以初始化时和 base 完全等价。
- seed 固定，构建结果和主机器的 vs01 逐字节一致。
- **第一个参数是 GDN value 缩放**：GDN 分支实际用的 V = 缩放 × V 参数。在 Adam 下它等价于这一支的 lr 乘了这个数。
  - 1.0 会让正式训练在第 2 步崩掉，见 README 和第 8 节。
  - 0.01 是目前验证过的稳定值。
- 不给 tag 时自动命名，比如 0.1 → `vs0p1`。`vs01` 专指 0.01，和主机器同名。

### 4.2 训练 + 导出 + 评测（一条命令）

```bash
nohup setsid bash allattn_prefix/run/train_full.sh 0,1,2,3 envscaler 8181 \
  $MODELS/Qwen3.5-4B-AllAttn-G512A64-vs01 vs01 > /dev/null 2>&1 &
```

参数依次是：4 张 GPU、域（`awm` / `envscaler` / `taco`）、评测端口（两组要用不同端口）、初始化目录、tag。

配置和主机器的正式训练完全一致：

- opsd_mixed 全量数据，1 个 epoch（AWM / EnvScaler 67 步，TACO 66 步）；
- lr 1e-5，batch 24，top-64 forward KL；
- V0 同步路径，3 actor + 1 teacher；
- 每 20 步存一次 checkpoint；
- 各域的 thinking 开关、轮数、工具输出长度沿用 `train_domain.bash` 的默认值，和 `reproduce_20260923.sh` 相同；
- 和 `reproduce_20260923.sh` 只有两处不同：关掉了训练中的验证，打开了 CUDA graph。两者都不影响训练本身。

跑完后自动做三件事：

1. 导出到 `$MODELS/prefix_full_<tag>_<dom>_step<N>`；
2. 跑 greedy 评测；
3. LCB 在后台判分。

| 日志 | 内容 |
|---|---|
| `$WORK/chain_full_<tag>_<dom>.log` | 里程碑 |
| `$WORK/train_full_<tag>_<dom>.log` | 训练明细 |
| `$EVAL_OUT/prefix_full_<tag>_<dom>_step<N>/` | 评测结果 |

**vLLM 卡死时自动续跑**：rollout 偶尔会在 `sample_tokens` 卡住，900 s 后超时报错。脚本会换一个 rollout seed，从最近的 checkpoint 续跑，seed 依次试 `default 4321 777`。要换 seed 做重复实验，用 `SEEDS="11 4321 777"`，并换一个 tag。

**一组卡上排多个实验**：写一个串行脚本，参考 `run/queue_h20_lane_a.sh`。每一步都是前一步结束后才开始。

### 4.3 训练时要盯的指标

```bash
grep -oE "step:[0-9]+ - .*" $WORK/train_full_vs01_envscaler.log | tail -3 | \
  grep -oE "step:[0-9]+|critic/score/mean:[0-9.]+|actor/distillation/loss:[0-9.e-]+|actor/grad_norm:[0-9.e-]+|response_length/mean:[0-9.]+|response_length/clip_ratio:[0-9.]+"
```

| 指标 | 正常（×0.01） | 崩了的样子（×1.0，第 2–3 步） |
|---|---|---|
| `critic/score/mean`（reward） | EnvScaler 0.7–0.86；TACO 0.4–0.75 | EnvScaler 0.19；TACO 0.00 |
| `actor/distillation/loss`（KL） | EnvScaler ~0.02；TACO ~0.015 | 比第 1 步翻倍以上（0.048 / 0.07） |
| `actor/grad_norm` | ~0.01 | ~1–5 |
| `response_length/clip_ratio`（16k 截断比例） | TACO 0.2–0.5；EnvScaler ~0 | TACO 0.92 |

## 5. 评测

`greedy_suite.sh <模型目录> <GPU> <端口>`：一个 vLLM 服务，DP 等于卡数，同时跑下面这些评测；生成完立刻关掉服务。

- BFCL v3 all；
- HumanEval / MBPP（EvalPlus，表里报原版测试的 pass@1，不是加强版）；
- LCB v6（1055 题）。

全部 greedy：T=0，每题生成 1 次，代码题生成上限 16k。

- `MAX_TOKENS=32768 bash .../greedy_suite.sh ...`：代码题上限放宽到 32k，不跑 BFCL，结果目录加后缀 `_32768`。用来区分"做错了"和"想太久被截断"。
- 汇总：`$KV_PY allattn_prefix/run/collect_results.py`。它会打印所有评测过的模型，括号里是相对 base 的差。`LCB 截断`列是生成到上限还没结束的题数。

## 6. 结果回传

需要发回主机器的东西：

- `collect_results.py` 打印的表；
- 每个训练的 `$WORK/train_full_*.log`（画训练曲线用）；
- `$WORK/chain_full_*.log`。

checkpoint 和 rollout 不用传。

## 7. 本机要跑的补充实验（H20）

已经写成两个串行脚本，各占一组卡，启动后不用管：

```bash
nohup setsid bash allattn_prefix/run/queue_h20_lane_a.sh > $WORK/lane_a.log 2>&1 &
nohup setsid bash allattn_prefix/run/queue_h20_lane_b.sh > $WORK/lane_b.log 2>&1 &
```

| 编号 | 卡 | 内容 | 回答什么问题 | 判断方法 |
|---|---|---|---|---|
| E0 | 0-3 | base 16k 评测 | H20 上的评测和主机器一致吗 | 第 3 节的表，±0.5 |
| E1 | 0-3 | 主机器训好的 EnvScaler ×0.01 prefix：先 16k 复现，再和 base 一起在 32k 上评测 | EnvScaler prefix 的 LCB -7.6 是不是只因为想得更长、被 16k 截断 | 16k 应复现 BFCL 68.98 / LCB 35.17 / HumanEval 89.63 / MBPP 84.66。32k 下 LCB 差距明显缩小，就说明只是变啰嗦，不是能力下降 |
| E2 | 0-3 | EnvScaler ×0.01 换 rollout seed 重训一次（tag `vs01s11`） | BFCL +1.0 是否超出训练本身的波动 | 和主机器 vs01 的 68.98 比；两次差距大于 1 的话，+1.0 就不可信 |
| E3 | 4-7 | EnvScaler GDN 缩放 ×0.1（`vs0p1`）、×0.003（`vs0p003`） | 0.01 是不是太保守，或者还不够保守 | 和主机器的 ×0.01（68.98）、×0.03 放在一起，看 BFCL 和 HumanEval / MBPP 随缩放的变化。×0.1 先看前 3 步有没有崩（4.3 节） |

预计耗时（按 H20 比 H100 慢 2 倍估）：第一组约 4 h 评测 + 6 h 训练；第二组约 2 × 7.5 h。

## 8. 常见问题

- **端口占用 / 残留进程**：
  - vLLM 一定要按进程组杀：`kill -- -<PGID>`。只杀 python 父进程的话，EngineCore 会成为孤儿，继续占约 70 GB 显存。
  - 训练残留：`pkill -9 -f "hydra.run.dir=$WORK/hydra/<name>"`。
- **不要编辑正在运行的 bash 脚本**：bash 是边执行边读文件的，改了会让正在跑的流程跳步。要改就复制一份新文件。
- **AWM 训练 OOM**：`train_full.sh` 已经开了 `entropy_from_logits_with_chunking=True`，不要去掉。
- **GDN 状态精度**：`opsd_plugin/sitecustomize.py` 会把新架构登记进 vLLM 的 `MODELS_CONFIG_MAP`，让 GDN 递归状态按 fp32 缓存（vLLM 默认对自定义架构用 bf16，序列越长误差越大）。前提是 `OPSD_PLUGIN=1`，并且 `PYTHONPATH` 里有 `opsd_plugin`。`train_full.sh` 和 `greedy_suite.sh` 都已经设置好了，自己写命令时别漏。
- **第 2 步崩溃的原理**：
  - Adam 第一步时每个参数都移动约 lr·sign(grad)，和梯度大小无关。
  - GDN 分支 512 个 value token 的梯度方向高度一致，梯度范数约是 A64 的 1000 倍（194 vs 0.19）。
  - 在 CPU 上模拟第一步，GDN 分支让模型输出偏离 base 的 KL 是 0.85，A64 约 0。
  - 复现脚本：`experiments/analysis/first_step_sensitivity.py <初始化目录> <rollout.jsonl>`。
  - 训完后各分支各自贡献多少：`experiments/analysis/branch_ablate.py <导出目录> <rollout.jsonl>`。主机器上 EnvScaler ×0.01 训完后，GDN 分支贡献 0.028，A64 贡献 0.000。
