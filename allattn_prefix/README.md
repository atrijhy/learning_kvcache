# 新结构：全层残差注意力 prefix（AllAttn-G512A64）

这个目录在作者的 Hybrid G512-A64 prefix 基础上，把 24 个 Gated DeltaNet（GDN）层的 prefix 改成残差注意力分支。8 个 attention 层的 A64 分支不变，OPSD 训练配方也不变（数据、lr、loss、配置都和 `reproduce_20260923.sh` 一致）。

## 1. 为什么改：GDN 层的 prefix 会衰减到 0

在作者的结构里，GDN 层的 512 个 prefix token 只是给递归状态一个初值 S₀。之后每读一个 token，状态都会被衰减：

    S_t = α_t · S_{t-1} · (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ

`experiments/analysis/gdn_decay.py` 在 base 模型、TACO rollout 上测了 S₀ 还剩多少（这是上界，没有算 delta 规则的擦除项）。按 24 层的中位数 head 算：

| 读过的 token 数 | 128 | 512 | 1k |
|---|---|---|---|
| S₀ 剩余比例 | 8.6e-3 | 1.8e-7 | ≈ 0 |

作者的 OPSD prompt 往往有几千个 token（题目加上一次失败尝试），所以模型开始回答时，24 层 GDN 里的 prefix 已经基本没有作用了。prefix 自己也没法学着绕开：衰减系数 α、β 是由用户 token 经过冻结的权重算出来的，prefix 改变不了它们。

## 2. 新结构

**GDN 层**：在 DeltaNet 的输出上加一个不衰减的 softmax 注意力分支，加在 gated RMSNorm 之前：

    o_t = DeltaNet(x)_t + softmax(q_t · K_pᵀ / √d_k) · V_p

- q_t 用的是这一层 conv 之前的 query 投影。
- K_p、V_p 是 512 个可训练的 soft token，经过这一层冻结的 `in_proj_qkv` 中 k、v 那两段权重投影得到。

**attention 层**：保留作者的 `Qwen3_5ResidualSoftPrefixAttention`（A64），没有改动。

**初始化**：V 设为 0，K 从 N(0, 0.02) 采样。V=0 时这个分支的输出就是 0，所以初始化时模型和 base 完全等价。这一点在 CPU 和 vLLM 上都验证过。

**不再需要 prefix_beta_logits 和 prefix_a**：新分支不往递归状态里写东西，也就不存在写入强度 β 和衰减 α 的问题。

| | 作者 Hybrid G512-A64 | 新结构 AllAttn-G512A64 |
|---|---|---|
| GDN 层 prefix | 512 token，写进递归初始状态（会衰减） | 512 token，残差注意力分支（不衰减） |
| attention 层 prefix | A64 | A64（不变） |
| 可训练张量 | 112 | 64 |
| 可训练参数 | 78.9M | 65.5M |

## 3. 文件

| 文件 | 作用 |
|---|---|
| `opsd_plugin/allattn/modeling_qwen3_5_all_attention_prefix.py` | HF / verl 训练端模型 `Qwen3_5AllAttentionPrefixForConditionalGeneration`，也会被复制进模型目录 |
| `opsd_plugin/allattn/vllm_all_attention_prefix.py` | vLLM 端实现（TP=1）。GDN 层分支在 vLLM forward 里计算，attention 层复用作者的 hybrid 实现；`load_weights` 会检查所有 prefix 张量都已加载 |
| `opsd_plugin/allattn/export_allattn_prefix.py` | 把 FSDP checkpoint 导出成 `residual_attention_prefix.safetensors` |
| `opsd_plugin/sitecustomize.py` | `OPSD_PLUGIN=1` 时，在每个进程里注册 vLLM 架构，**并登记进 MODELS_CONFIG_MAP，让 GDN 状态按 fp32 缓存**（和 vllm_plugin.py 的修复相同），另外注册额外的 loss |
| `opsd_plugin/opsd_losses.py` | 可选 loss `forward_kl_topk_focus`：只对每条序列中 KL 最大的前 5% token 计算 loss。正式实验没有用它 |
| `opsd_plugin/internal/train_domain.bash` | 作者 launcher 的副本，改了 3 处：① 不再写死 `trainable_tensor_count == 112`，改成从模型 config 读张量数和参数量，传给 verl 校验；② 可训练参数的正则加上 `linear_attn.prefix_key_tokens/prefix_value_tokens`；③ REPO_ROOT 写死成本机路径 |
| `opsd_plugin/internal/teacher_ceiling_slots.sh` | 评测 pipeline 的副本，只多了一个 `TRAIN_SCRIPT` 覆盖参数 |
| `tools/build_allattn_model.py` | 从 base 模型构建初始化目录：`--base Qwen3.5-4B --out DIR --gdn 512 --attn 64` |
| `experiments/ovf/` | 过拟合（吸收）测试：筛选题目、训练 20 个 epoch、不带 hint 测通过率 |
| `experiments/full/` | 正式训练（作者全量数据，1 个 epoch，lr 1e-5）、导出和评测的 chain，包括 vLLM `sample_tokens` 卡死后自动续跑 |
| `experiments/eval/greedy_suite.sh` | 作者表格口径的 greedy 评测：BFCL v3 all、HumanEval+、MBPP+、LCB v6 |
| `experiments/analysis/` | GDN 衰减测量、KL 的 token 分布、HF 和 vLLM 的 logprob 一致性检查 |

使用：`export OPSD_PLUGIN=1 PYTHONPATH=<本目录>/opsd_plugin:<仓库>/verl:<仓库>`，然后按作者的方式训练，`MODEL` 指向 build 出来的目录。脚本里的 `/efs/...`、`/root/...` 是本机路径，换机器时需要改。

## 4. 结果（2026-11-05）

**过拟合 / 吸收测试**
- 每个域挑 48 道题：student 做不对、teacher 带上 hint 能做对。
- 训练 20 个 epoch，lr 用作者的 1e-5。
- 测不带 hint 时的通过率，每题采样 4 次。下表 5/10/20 ep 对应 step 10/20/40。

| 域 | base | 新结构 5 / 10 / 20 ep | 作者结构 @1e-5 | 作者结构 @2e-3 (20 ep) | teacher |
|---|---|---|---|---|---|
| EnvScaler | 39.6 | 49.5 / **60.9** / 54.2 | ≈ +0 | 60.9 | 81.8 |
| AWM | 53.6 | 51.0 / 54.7 / **59.9** | ≈ +0 | 63.0 | 75.0 |
| TACO | 19.8 | 32.8 / 35.4 / **36.5** | ≈ +0 | 35.4 | 92.2 |

在作者的 lr 下，作者的结构基本学不进去，新结构在三个域都学进去了。

**正式训练**（作者配置）：EnvScaler 的 prefix 在 greedy 评测上低于 base。

| | BFCL v3 | Relevance | Irrelevance | Multi-turn | HumanEval | MBPP |
|---|---|---|---|---|---|---|
| base | 67.93 | 66.67 | 82.56 | 45.13 | 90.24 | 82.01 |
| 新结构 EnvScaler | 65.46 | 94.44 | 76.97 | 40.38 | 82.93 | 78.84 |

模型变得更倾向于调用工具。我们正在做一个检查：用同一条评测路径评测没训练过的模型。它理论上必须等于 base，用来排除评测实现本身的问题。TACO 和 AWM 的正式结果还没出来。
