# 新结构：全层残差注意力 prefix（AllAttn-G512A64）

这个目录在原 Hybrid G512-A64 prefix 基础上，把 24 个 Gated DeltaNet（GDN）层的 prefix 改成残差注意力分支。8 个 attention 层的 A64 分支不变，OPSD 训练配方也不变（数据、lr、loss、配置都和 `reproduce_20260923.sh` 一致）。

## 1. 为什么改：GDN 层的 prefix 会衰减到 0

在原结构里，GDN 层的 512 个 prefix token 只是给递归状态一个初值 S₀。之后每读一个 token，状态都会被衰减：

    S_t = α_t · S_{t-1} · (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ

`experiments/analysis/gdn_decay.py` 在 base 模型、TACO rollout 上测了 S₀ 还剩多少（这是上界，没有算 delta 规则的擦除项）。按 24 层的中位数 head 算：

| 读过的 token 数 | 128 | 512 | 1k |
|---|---|---|---|
| S₀ 剩余比例 | 8.6e-3 | 1.8e-7 | ≈ 0 |

opsd_mixed 的 prompt 往往有几千个 token（题目加上一次失败尝试），所以模型开始回答时，24 层 GDN 里的 prefix 已经基本没有作用了。prefix 自己也没法学着绕开：衰减系数 α、β 是由用户 token 经过冻结的权重算出来的，prefix 改变不了它们。

## 2. 新结构

**GDN 层**：在 DeltaNet 的输出上加一个不衰减的 softmax 注意力分支，加在 gated RMSNorm 之前：

    o_t = DeltaNet(x)_t + softmax(q_t · K_pᵀ / √d_k) · V_p

- q_t 用的是这一层 conv 之前的 query 投影。
- K_p、V_p 是 512 个可训练的 soft token，经过这一层冻结的 `in_proj_qkv` 中 k、v 那两段权重投影得到。

**attention 层**：保留原来的 `Qwen3_5ResidualSoftPrefixAttention`（A64），没有改动。

**初始化**：V 设为 0，K 从 N(0, 0.02) 采样。V=0 时这个分支的输出就是 0，所以初始化时模型和 base 完全等价。这一点在 CPU 和 vLLM 上都验证过。

**不再需要 prefix_beta_logits 和 prefix_a**：新分支不往递归状态里写东西，也就不存在写入强度 β 和衰减 α 的问题。

| | 原结构 Hybrid G512-A64 | 新结构 AllAttn-G512A64 |
|---|---|---|
| GDN 层 prefix | 512 token，写进递归初始状态（会衰减） | 512 token，残差注意力分支（不衰减） |
| attention 层 prefix | A64 | A64（不变） |
| 可训练张量 | 112 | 64 |
| 可训练参数 | 78.9M | 65.5M |

## 3. 文件

| 文件 | 作用 |
|---|---|
| `opsd_plugin/allattn/modeling_qwen3_5_all_attention_prefix.py` | HF / verl 训练端模型 `Qwen3_5AllAttentionPrefixForConditionalGeneration`，也会被复制进模型目录 |
| `opsd_plugin/allattn/vllm_all_attention_prefix.py` | vLLM 端实现（TP=1）。GDN 层分支在 vLLM forward 里计算，attention 层复用原来的 hybrid 实现；`load_weights` 会检查所有 prefix 张量都已加载 |
| `opsd_plugin/allattn/export_allattn_prefix.py` | 把 FSDP checkpoint 导出成 `residual_attention_prefix.safetensors` |
| `opsd_plugin/sitecustomize.py` | `OPSD_PLUGIN=1` 时，在每个进程里注册 vLLM 架构，**并登记进 MODELS_CONFIG_MAP，让 GDN 状态按 fp32 缓存**（和 vllm_plugin.py 的修复相同），另外注册额外的 loss |
| `opsd_plugin/opsd_losses.py` | 可选 loss `forward_kl_topk_focus`：只对每条序列中 KL 最大的前 5% token 计算 loss。正式实验没有用它 |
| `opsd_plugin/internal/train_domain.bash` | 原 launcher 的副本，改了 3 处：① 不再写死 `trainable_tensor_count == 112`，改成从模型 config 读张量数和参数量，传给 verl 校验；② 可训练参数的正则加上 `linear_attn.prefix_key_tokens/prefix_value_tokens`；③ REPO_ROOT 写死成本机路径 |
| `opsd_plugin/internal/teacher_ceiling_slots.sh` | 评测 pipeline 的副本，只多了一个 `TRAIN_SCRIPT` 覆盖参数 |
| `tools/build_allattn_model.py` | 从 base 模型构建初始化目录：`--base Qwen3.5-4B --out DIR --gdn 512 --attn 64 --gdn-value-scale 0.01` |
| `run/` | 可移植的一键脚本：路径配置 `env.sh`、起工具服务、构建模型、正式训练+导出+评测 `train_full.sh`、评测 `greedy_suite.sh`、汇总 `collect_results.py`。用法见 [RUNBOOK.md](RUNBOOK.md) |
| `experiments/ovf/` | 过拟合（吸收）测试：筛选题目、训练 20 个 epoch、不带 hint 测通过率 |
| `experiments/full/` | 正式训练（opsd_mixed 全量数据，1 个 epoch，lr 1e-5）、导出和评测的 chain，包括 vLLM `sample_tokens` 卡死后自动续跑 |
| `experiments/eval/greedy_suite.sh` | greedy 评测：BFCL v3 all、HumanEval+、MBPP+、LCB v6 |
| `experiments/analysis/` | GDN 衰减测量、KL 的 token 分布、HF 和 vLLM 的 logprob 一致性检查、第一步 Adam 敏感度（`first_step_sensitivity.py`）、训练后各分支贡献（`branch_ablate.py`） |

使用：`export OPSD_PLUGIN=1 PYTHONPATH=<本目录>/opsd_plugin:<仓库>/verl:<仓库>`，然后按原来的方式训练，`MODEL` 指向 build 出来的目录。脚本里的 `/efs/...`、`/root/...` 是本机路径，换机器时需要改。

## 4. 结果（2026-11-05）

**过拟合 / 吸收测试**
- 每个域挑 48 道题：student 做不对、teacher 带上 hint 能做对。
- 训练 20 个 epoch，lr 用原配方的 1e-5。
- 测不带 hint 时的通过率，每题采样 4 次。下表 5/10/20 ep 对应 step 10/20/40。

| 域 | base | 新结构 5 / 10 / 20 ep | 原结构 @1e-5 | 原结构 @2e-3 (20 ep) | teacher |
|---|---|---|---|---|---|
| EnvScaler | 39.6 | 49.5 / **60.9** / 54.2 | ≈ +0 | 60.9 | 81.8 |
| AWM | 53.6 | 51.0 / 54.7 / **59.9** | ≈ +0 | 63.0 | 75.0 |
| TACO | 19.8 | 32.8 / 35.4 / **36.5** | ≈ +0 | 35.4 | 92.2 |

在原配方的 lr 下，原结构基本学不进去，新结构在三个域都学进去了。

**正式训练**（opsd_mixed 全量数据，原配方 lr 1e-5，1 个 epoch）

| | BFCL v3 | multi-turn | relevance | irrelevance | LCB v6（16k 截断题数） | HumanEval | MBPP |
|---|---|---|---|---|---|---|---|
| base | 67.93 | 45.13 | 66.67 | 82.56 | 42.75 (584) | 90.24 | 82.01 |
| EnvScaler，GDN value ×1（修复前） | 65.46 (-2.5) | 40.38 | 94.44 | 76.97 | 30.52 (721) | 82.93 (-7.3) | 78.84 (-3.2) |
| EnvScaler，GDN value ×0.01 | 68.98 (+1.0) | 47.50 (+2.4) | 72.22 | 81.88 | 35.17 (669) | 89.63 (-0.6) | 84.66 (+2.6) |
| TACO，GDN value ×1（修复前） | 66.87 (-1.1) | | | | 37.8 | 85.37 (-4.9) | 78.31 (-3.7) |

- **修复前全部低于 base**：三个域都在第 2 步崩溃，TACO reward 0.61 → 0.00，92% 的回答被 16k 截断，之后只恢复了一部分。
  - 原因：Adam 第一步每个参数都移动约 lr，和梯度大小无关；GDN 分支的梯度范数约是 A64 的 1000 倍。
  - 修复：GDN 分支的 value 乘 0.01（config 里的 `residual_gdn_prefix_value_scale`），等价于这一支的 lr 乘 0.01。
- **修复后不再崩溃**：BFCL +1.0，HumanEval / MBPP 不掉。
  - LCB 的 -7.6 几乎全部来自被 16k 截断的题变多（+85）；写出了代码的题，正确率是 96%，base 是 94%。
- **改变输出的几乎全是 GDN 分支**：在 rollout 上把各分支单独置零后测 KL(base‖model)，GDN 分支 0.028，A64 0.000。
- 没训练过的新结构走同一条评测路径，得分是 67.77 / 90.24 / 83.07，和 base 一致，说明评测实现没问题。
- TACO、AWM 的 ×0.01，以及 EnvScaler 的 ×0.03 正在跑。

## 5. 在新机器上跑

全流程（拷环境、起服务、自检、训练、评测、汇总，以及待跑的补充实验）见 [RUNBOOK.md](RUNBOOK.md)。一键脚本在 `run/`。
