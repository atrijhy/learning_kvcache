"""Fast 1-GPU or Multi-GPU DDP trainer for Qwen3-4B nope_token_null_additive (exact o_t = d_t + p_t)."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from prefix_tuning.virtual_prefix.modeling_qwen3_residual_kv_prefix import (
    install_residual_kv_prefix,
    load_prefix_state_dict,
    prefix_state_dict,
)


def trajectory_loss(model, row):
    ids, target_positions = row[1], row[3]
    positions = torch.tensor(target_positions, device=model.device)
    inputs = torch.tensor([ids], device=model.device)
    labels = inputs[0, positions]
    hidden = model.model(input_ids=inputs, use_cache=False).last_hidden_state[0, positions - 1]

    def score(chunk, targets):
        return F.cross_entropy(model.lm_head(chunk).float(), targets, reduction="sum")

    return sum(
        (checkpoint(score, h, y, use_reentrant=False) if torch.is_grad_enabled() else score(h, y))
        for h, y in zip(hidden.split(256), labels.split(256), strict=True)
    )


def reset_caches(model):
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "invalidate_residual_attention_prefix_cache"):
            layer.self_attn.invalidate_residual_attention_prefix_cache()


@torch.no_grad()
def evaluate_rows(model, rows, rank, world, *, baseline=False):
    reset_caches(model)
    model.eval()
    names = sorted({row[0] for row in rows})
    sums = torch.zeros(len(names), 2, dtype=torch.float64, device=model.device)
    gate_stats = torch.zeros(3, dtype=torch.float64, device=model.device)
    for layer in model.model.layers:
        layer.self_attn.prefix_disabled = baseline
        if not baseline:
            def make_mon():
                def mon(lam, delta, native):
                    gate_stats[0] += lam.sum(dtype=torch.float64)
                    gate_stats[1] += (delta.norm(dim=-1) / (native.float().norm(dim=-1) + 1e-8)).sum(dtype=torch.float64)
                    gate_stats[2] += lam.numel()
                return mon
            layer.self_attn.prefix_monitor = make_mon()
    try:
        for row in rows[rank::world]:
            idx = names.index(row[0])
            sums[idx, 0] += trajectory_loss(model, row)
            sums[idx, 1] += len(row[2])
    finally:
        for layer in model.model.layers:
            layer.self_attn.prefix_disabled = False
            layer.self_attn.prefix_monitor = None
        model.train()
    if world > 1:
        dist.all_reduce(sums)
        dist.all_reduce(gate_stats)
    by_source = {
        name: {"nll": loss / count, "target_tokens": int(count)}
        for name, (loss, count) in zip(names, sums.tolist(), strict=True)
        if count > 0
    }
    tot_l, tot_c = sums.sum(dim=0).tolist()
    g_lam = (gate_stats[0] / max(1.0, gate_stats[2])).item() if not baseline else 0.0
    g_rat = (gate_stats[1] / max(1.0, gate_stats[2])).item() if not baseline else 0.0
    return (tot_l / tot_c if tot_c > 0 else float("nan")), by_source, g_lam, g_rat


def rank_batches(batch, rows, world):
    if world == 1:
        return [list(batch)]
    buckets = [[] for _ in range(world)]
    loads = [0] * world
    for idx in sorted(batch, key=lambda i: len(rows[i][1]), reverse=True):
        best = min(range(world), key=loads.__getitem__)
        buckets[best].append(idx)
        loads[best] += len(rows[idx][1])
    return buckets


def save_ckpt(ckpt_dir: Path, model, init_cfg: dict, epoch: int):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = prefix_state_dict(model)
    tmp = ckpt_dir / "model.safetensors.tmp"
    save_file(state, str(tmp), metadata={"format": "pt"})
    tmp.replace(ckpt_dir / "model.safetensors")
    cfg = dict(init_cfg)
    cfg["epoch"] = epoch
    (ckpt_dir / "prefix_config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--init-dir", type=Path, required=True)
    ap.add_argument("--data-pt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--attn-value-lr", type=float, default=2e-5)
    ap.add_argument("--attn-key-lr", type=float, default=2e-5)
    ap.add_argument("--attn-null-lr", type=float, default=2e-5)
    ap.add_argument("--attn-bias-lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261127)
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))
    torch.manual_seed(args.seed)

    init_cfg = json.loads((args.init_dir / "prefix_config.json").read_text())
    num_tokens = int(init_cfg["num_virtual_tokens"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    parameters = install_residual_kv_prefix(model, num_virtual_tokens=num_tokens, mode="nope_token_null_additive")
    load_prefix_state_dict(model, load_file(str(args.init_dir / "model.safetensors"), device="cpu"))
    for p in parameters.values():
        p.data = p.data.float().to(device)
        p.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    grouped = {}
    for name, p in parameters.items():
        if name.endswith("prefix_logit_bias"):
            lr_val = args.attn_bias_lr
        elif name.endswith("prefix_value_tokens"):
            lr_val = args.attn_value_lr
        elif name.endswith("prefix_key_tokens"):
            lr_val = args.attn_key_lr
        elif name.endswith("prefix_null_proj"):
            lr_val = args.attn_null_lr
        else:
            raise ValueError(name)
        grouped.setdefault(lr_val, []).append(p)
    groups = [{"params": ps, "lr": lr_val} for lr_val, ps in grouped.items()]
    trainable = list(parameters.values())

    data = torch.load(args.data_pt, map_location="cpu", weights_only=False)
    rows, val_rows = data["train"], data["validation"]

    optimizer = torch.optim.AdamW(groups, weight_decay=0.0, fused=True)
    group_rates = [g["lr"] for g in optimizer.param_groups]
    steps_per_epoch = math.ceil(len(rows) / args.batch_size)
    steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, math.ceil(steps * 0.1))

    if rank == 0:
        print(f"[tna-train] {args.out_dir.name}: P={num_tokens}, train={len(rows)}, val={len(val_rows)}, bs={args.batch_size}, ep={args.epochs}, steps={steps}, world={world}", flush=True)

    base_val, base_src, _, _ = evaluate_rows(model, val_rows, rank, world, baseline=True)
    history = [{"epoch": 0, "step": 0, "val_nll": base_val, "by_source": base_src, "is_base": True}]

    started = time.monotonic()
    best_val, best_ep, step = float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        indices = list(range(len(rows)))
        random.Random(args.seed + epoch).shuffle(indices)
        ep_loss, ep_toks = 0.0, 0
        for start in range(0, len(indices), args.batch_size):
            step += 1
            batch = indices[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            metrics = torch.zeros(2, device=device)
            for idx in rank_batches(batch, rows, world)[rank]:
                row = rows[idx]
                loss = trajectory_loss(model, row)
                w = row[4] if len(row) > 4 else 1.0
                (w * loss / len(row[2]) / len(batch)).backward()
                metrics[0] += loss.detach()
                metrics[1] += len(row[2])
            if world > 1:
                grads = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in trainable])
                dist.all_reduce(grads)
                dist.all_reduce(metrics)
                off = 0
                for p in trainable:
                    p.grad = grads[off:off + p.numel()].view_as(p)
                    off += p.numel()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            frac = (step / warmup_steps) if step <= warmup_steps else (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min((step - warmup_steps) / max(1, steps - warmup_steps), 1.0))))
            for g, r in zip(optimizer.param_groups, group_rates, strict=True):
                g["lr"] = r * frac
            optimizer.step()
            ep_loss += metrics[0].item()
            ep_toks += int(metrics[1].item())
            if rank == 0 and (step == 1 or step % 20 == 0):
                print(f"  ep={epoch} step={step}/{steps} tr={metrics[0].item()/metrics[1].item():.5f} gnorm={norm.item():.3f} t={time.monotonic()-started:.1f}s", flush=True)

        val_nll, by_src, g_lam, g_rat = evaluate_rows(model, val_rows, rank, world, baseline=False)
        tr_nll = ep_loss / max(1, ep_toks)
        rec = {"epoch": epoch, "step": step, "train_nll": tr_nll, "val_nll": val_nll, "by_source": by_src,
               "gate_lam": g_lam, "gate_ratio": g_rat, "elapsed_s": round(time.monotonic() - started, 1)}
        history.append(rec)
        if rank == 0:
            save_ckpt(args.out_dir / f"epoch-{epoch:02d}", model, init_cfg, epoch)
            if val_nll < best_val:
                best_val, best_ep = val_nll, epoch
                save_ckpt(args.out_dir / "best", model, init_cfg, epoch)
            print(f"  [ep {epoch}] tr={tr_nll:.5f} val={val_nll:.5f} (d={val_nll-base_val:+.5f}) lam={g_lam:.4f} rat={g_rat:.4f} t={rec['elapsed_s']}s", flush=True)

    if rank == 0:
        (args.out_dir / "metrics.json").write_text(json.dumps({
            "name": args.out_dir.name, "num_virtual_tokens": num_tokens, "train_rows": len(rows),
            "val_rows": len(val_rows), "epochs": args.epochs, "batch_size": args.batch_size,
            "base_val_nll": base_val, "best_epoch": best_ep, "best_val_nll": best_val,
            "wall_seconds": round(time.monotonic() - started, 1), "history": history,
        }, indent=2) + "\n")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
