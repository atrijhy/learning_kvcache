"""Small offline SFT runs for Qwen3 residual KV memory on Ascend or CUDA.

Run through experiment.sh sft, which initializes the platform environment.
Checkpoints contain only the trained memory and its optimizer, never base weights.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM
from prefix_tuning.virtual_prefix.modeling_qwen3_residual_kv_prefix import (
    install_residual_kv_prefix, prefix_state_dict, load_prefix_state_dict,
)


def supervised_loss(model, row, device):
    ids = torch.tensor([row["input_ids"]], device=device, dtype=torch.long)
    labels = torch.tensor(row["labels"], device=device, dtype=torch.long)
    # Logit position i predicts label i+1. Do not materialize vocabulary logits
    # for the long, fully masked system/tools/history portion of each sample.
    positions = torch.nonzero(labels[1:] != -100).flatten()
    targets = labels[1:][positions]
    output = model(input_ids=ids, use_cache=False, logits_to_keep=positions)
    loss = torch.nn.functional.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]).float(), targets, reduction="sum")
    return loss, targets.numel()


@torch.no_grad()
def evaluate(model, rows, device):
    model.eval()
    loss_sum, tokens = 0.0, 0
    for row in rows:
        loss, count = supervised_loss(model, row, device)
        loss_sum += loss.item()
        tokens += count
    return {"loss": loss_sum/tokens, "target_tokens": tokens, "samples": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/liyu.468/models/Qwen3-4B")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--prefix-tokens", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--load-prefix")
    args = parser.parse_args()
    if min(args.steps, args.gradient_accumulation, args.prefix_tokens, args.eval_samples) <= 0:
        parser.error("Steps, accumulation, prefix length and evaluation size must be positive")
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive")
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2)+"\n")
    data = Path(args.data_dir)
    manifest = json.loads((data / "manifest.json").read_text())
    rows = {split: [json.loads(line) for line in (data / f"{split}.jsonl").open()]
            for split in ("train", "eval")}
    assert not {r["task_id"] for r in rows["train"]} & {r["task_id"] for r in rows["eval"]}
    for group in rows.values():
        for row in group:
            assert row["prompt_tokens"] > 0
            assert all(t == -100 for t in row["labels"][:row["prompt_tokens"]])
            assert row["labels"][row["prompt_tokens"]:] == row["input_ids"][row["prompt_tokens"]:]
            assert row["target_tokens"] == sum(x != -100 for x in row["labels"])
            assert len(row["input_ids"]) <= manifest["max_length"]
    print("Loading frozen base model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(args.device)
    model.eval()
    probe = torch.tensor([rows["train"][0]["input_ids"][:32]], device=args.device)
    with torch.no_grad():
        base_logits = model(probe, use_cache=False).logits.detach().cpu()
    parameters = install_residual_kv_prefix(model, args.prefix_tokens)
    with torch.no_grad():
        initial_logits = model(probe, use_cache=False).logits.detach().cpu()
    neutral_max_diff = (initial_logits.float()-base_logits.float()).abs().max().item()
    assert neutral_max_diff == 0, f"Zero-V initialization changed base logits: {neutral_max_diff}"
    del base_logits, initial_logits
    if args.load_prefix:
        load_prefix_state_dict(model, load_file(args.load_prefix))
    base_probes = {name: p.detach().flatten()[:64].cpu().clone()
                   for name, p in model.named_parameters() if not p.requires_grad}
    value_keys = [name for name in parameters if name.endswith("prefix_value_tokens")]
    key_keys = [name for name in parameters if name.endswith("prefix_key_tokens")]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=args.learning_rate, weight_decay=0.0, foreach=False)
    metadata = {"method": "qwen3_residual_kv_prefix_sft", "base_model": args.model,
                "base_weights_frozen": True, "layers": model.config.num_hidden_layers,
                "prefix_tokens": args.prefix_tokens, "prefix_parameter_dtype": "float32",
                "trainable_parameters": sum(p.numel() for p in parameters.values()),
                "base_parameters": sum(p.numel() for n,p in model.named_parameters() if n not in parameters),
                "neutral_initialization_max_logit_difference": neutral_max_diff,
                "data_manifest": manifest, "hardware": args.device,
                "torch": torch.__version__, "started_at_unix": time.time()}
    print(json.dumps({k:v for k,v in metadata.items() if k != "data_manifest"}), flush=True)
    eval_rows = rows["eval"][:args.eval_samples]
    train_probes = rows["train"][:args.eval_samples]
    before = {"eval": evaluate(model, eval_rows, args.device),
              "train_probe": evaluate(model, train_probes, args.device)}
    print(json.dumps({"before": before}), flush=True)
    order, cursor = list(range(len(rows["train"]))), 0
    rng.shuffle(order)
    metrics = []
    for step in range(1, args.steps+1):
        window = []
        for _ in range(args.gradient_accumulation):
            if cursor == len(order):
                rng.shuffle(order)
                cursor = 0
            window.append(rows["train"][order[cursor]])
            cursor += 1
        total_tokens = sum(row["target_tokens"] for row in window)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started, summed = time.monotonic(), 0.0
        for row in window:
            loss, count = supervised_loss(model, row, args.device)
            assert count == row["target_tokens"]
            assert torch.isfinite(loss).item(), "Non-finite training loss"
            summed += loss.item()
            (loss/total_tokens).backward()
        def active(names):
            return sum(parameters[n].grad is not None and parameters[n].grad.abs().max().item() > 0 for n in names)
        value_active, key_active = active(value_keys), active(key_keys)
        assert value_active == len(value_keys), f"Only {value_active} layers received value gradients"
        if step > 1 or args.load_prefix:
            assert key_active == len(key_keys), f"Only {key_active} layers received key gradients"
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0, foreach=False)
        assert torch.isfinite(norm).item(), "Non-finite prefix gradients"
        optimizer.step()
        record = {"step": step, "loss": summed/total_tokens, "target_tokens": total_tokens,
                  "input_tokens": sum(len(r["input_ids"]) for r in window),
                  "grad_norm": norm.item(), "value_layers_with_grad": value_active,
                  "key_layers_with_grad": key_active, "seconds": time.monotonic()-started}
        metrics.append(record)
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record)+"\n")
        print(json.dumps(record), flush=True)
    after = {"eval": evaluate(model, eval_rows, args.device),
             "train_probe": evaluate(model, train_probes, args.device)}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p.detach().flatten()[:64].cpu(), base_probes[name]), name
    prefix = prefix_state_dict(model)
    assert any(t.abs().max().item() > 0 for n,t in prefix.items() if n.endswith("prefix_value_tokens"))
    save_file(prefix, str(out / "prefix.safetensors"))
    (out / "prefix.safetensors").chmod(0o644)
    torch.save({"optimizer": optimizer.state_dict(), "step": args.steps}, out / "optimizer.pt")
    if args.device.startswith("npu"):
        peak_gb = torch.npu.max_memory_allocated()/1024**3
    elif args.device.startswith("cuda"):
        peak_gb = torch.cuda.max_memory_allocated()/1024**3
    else:
        peak_gb = None
    report = {**metadata, "status": "complete", "optimizer_steps": args.steps,
              "samples_seen": args.steps*args.gradient_accumulation,
              "before": before, "after": after, "peak_allocated_gib": peak_gb,
              "base_probe_tensors_unchanged": len(base_probes), "metrics": metrics,
              "checkpoint": str(out / "prefix.safetensors"),
              "limitation": "Small offline pilot; task success and benchmark gains are not evaluated."}
    (out / "prefix_config.json").write_text(json.dumps({
        "method": metadata["method"], "base_model": args.model, "prefix_tokens": args.prefix_tokens,
        "key_init_std": 0.02, "dtype": "float32", "model_type": "qwen3",
        "load": "Load base, install_residual_kv_prefix, then load_prefix_state_dict."}, indent=2)+"\n")
    (out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": "complete", "before": before, "after": after,
                      "peak_allocated_gib": peak_gb, "output_dir": str(out)}), flush=True)


if __name__ == "__main__":
    main()
