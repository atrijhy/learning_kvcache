# Qwen3 residual KV memory SFT on Ascend

Use `experiment.sh sft` for offline EnvScaler trajectory training. This path
freezes Qwen3-4B and trains independent hidden-space K/V tables in each of its
36 full-attention layers. It uses the same separately normalized residual
attention memory as the full-attention layers in the existing Qwen3.5 hybrid
adapter. Qwen3 has no DeltaNet layers. It does not concatenate memory into the
native causal cache or shift real-token positions.

The 256-token default has 47,185,920 FP32 trainable parameters. Keys start with
normal standard deviation 0.02; values start at zero so initial outputs match
the base exactly. All base projections and norms remain frozen. This is a
prefix-only SFT experiment, rather than the EnvScaler paper's full-weight SFT.

The raw `XXHStudyHard/EnvScaler-SFT-Traj-9K` data is stored under
`traj_data/envscaler_sft_9k/`. The preparation command verifies its SHA256,
counts all trajectories, partitions by task ID, and uniformly samples a
bounded pilot across the source trajectories. A task's conversation and
non-conversation trajectories always stay in the same split. Each expanded
sample supervises only the final assistant reasoning/action; system, user,
tool results, and historical actions are masked, and historical reasoning is
removed. Oversized turns are skipped without truncation. This length-limited
pilot is not the full 32K training protocol in the paper.

```bash
./experiment.sh sft prepare \
  --source "$PWD/traj_data/envscaler_sft_9k/envscaler_sft_traj_9k_metadata.json" \
  --model /home/liyu.468/models/Qwen3-4B \
  --output-dir "$PWD/traj_data/envscaler_sft_9k/pilot_8192_v1" \
  --max-length 8192 --train-samples 128 --eval-samples 16

./experiment.sh sft train \
  --data-dir "$PWD/traj_data/envscaler_sft_9k/pilot_8192_v1" \
  --output-dir "$PWD/outputs/envscaler_qwen3_sft_20261102/smoke3" \
  --steps 3 --gradient-accumulation 2 --prefix-tokens 256 \
  --learning-rate 0.001 --eval-samples 4 --device npu:0
```

Output directories must be new. `--load-prefix PATH/prefix.safetensors` starts
a new optimizer from an existing prefix; it does not resume optimizer or data
order. Checkpoints contain `prefix.safetensors`, `prefix_config.json`, and
`optimizer.pt`; the original base model must also be available. To use the
memory, load the base, call `install_residual_kv_prefix`, and then
`load_prefix_state_dict`. It is not a standalone `AutoModel` model directory.

`report.json` records before/after teacher-forced losses, per-step gradients,
memory, and frozen-weight checks. `metrics.jsonl` contains optimizer updates.
Only supervised positions materialize vocabulary logits. Token-weighted
gradient accumulation, BF16 base weights, FP32 memory, and non-reentrant
gradient checkpointing keep the one-NPU pilot bounded. No tool environment,
online teacher, vLLM service, or external logging service is needed.

The launcher uses the existing `kvcache` Python when present; otherwise it
uses this image's `/usr/local/miniconda3/bin/python`, which already includes
the NPU training stack. It sources CANN 9.1 when available.

Regression checks cover neutral initialization, K/V gradients, frozen base
weights, causality, native-cache incremental inference, checkpoint reload,
masked SFT targets, and equivalence of sparse/dense causal cross entropy:

```bash
PYTHONPATH="$PWD" /usr/local/miniconda3/bin/python -m pytest -q \
  prefix_tuning/tests/test_qwen3_residual_kv_prefix.py
```

A few SFT steps verify that training works. Loss changes on a tiny sample
do not establish better tool-use task success or benchmark performance.

## Full-data run

The full run uses native PyTorch through Transformers Trainer/Accelerate and
HCCL on eight NPUs. It uses all 9,022 source trajectories and all 163,165
assistant turns. There is no holdout, per-trajectory turn cap, or length-based
sample filtering. CPU dataloader workers encode examples lazily. For examples
exceeding 32,768 tokens, the same last-turn-first source/target budget allocation
as LlamaFactory's `mask_history` processor retains the current supervision and
the most recent history. Historical reasoning is removed.

The agreed full settings are: learning rate **1e-5** (ten times the upstream
1e-6), 3 epochs, micro batch 1 per rank, gradient accumulation 32, effective
batch 256, cosine schedule, 10% warmup, BF16 base, and 256 FP32 K/V prefix
positions per layer. Checkpoints are saved every 100 optimizer updates.
Training starts from the frozen base, rather than a pilot checkpoint. There
are no online environments or extra validation runs. The parallel execution
uses native DDP rather than LlamaFactory/DeepSpeed; this is still a prefix-only
experiment rather than the paper's full-weight SFT.

```bash
./experiment.sh sft prepare-full \
  --source "$PWD/traj_data/envscaler_sft_9k/envscaler_sft_traj_9k_metadata.json" \
  --data-dir "$PWD/traj_data/envscaler_sft_9k/full_32768"

./experiment.sh sft train-full \
  --model /home/liyu.468/models/Qwen3-4B \
  --data-dir "$PWD/traj_data/envscaler_sft_9k/full_32768" \
  --output-dir "$PWD/outputs/envscaler_qwen3_sft_20261102/full_native_lr1e-5"
```

`run_config.json` records the exact settings and source hash; `run_status.json`
and `metrics.jsonl` track progress. Native checkpoints contain the prefix in
`model.safetensors`, its `prefix_config.json`, and normal Trainer optimizer,
scheduler, and per-rank RNG states. They require the original frozen base.
For interruption recovery, launch the same command with
`--resume-from-checkpoint /path/to/checkpoint-N`. Output paths cannot otherwise
be reused. This image's Transformers 5.15 represents the 10% warmup ratio as
`warmup_steps=0.1`; its scheduler computes `ceil(total_steps * 0.1)`.

# Evaluation on Ascend 910C

Use the existing BFCL/tau2 entry point and native vLLM-Ascend prefix plugin.
See [benchmark/910C.md](../../benchmark/910C.md) for export, serving, and evaluation commands.

