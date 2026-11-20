"""Prepare a bounded SFT pilot, splitting entire task IDs before turn expansion.

No prompt/response is truncated. Oversized turns are skipped and counted.
Only the current assistant reasoning/action is supervised. All historical
reasoning is removed, matching EnvScaler's mask-history protocol.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import ijson
from transformers import AutoTokenizer


def task_key(item):
    info = item["task_info"]
    return str(info.get("task_id") or json.dumps(info, sort_keys=True, ensure_ascii=False))


def tokenize_turn(tokenizer, item, turn):
    messages = json.loads(item["messages"]) if isinstance(item["messages"], str) else item["messages"]
    tools = json.loads(item["tools"]) if isinstance(item["tools"], str) else item["tools"]
    context = copy.deepcopy(messages[:turn])
    for message in context:
        if message["role"] == "assistant":
            message.pop("reasoning_content", None)
            if "</think>" in (message.get("content") or ""):
                message["content"] = message["content"].split("</think>")[-1].lstrip("\n")
    prompt = tokenizer.apply_chat_template(context, tools=tools, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(context + [messages[turn]], tools=tools,
                                         tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(full, add_special_tokens=False)
    if not full.startswith(prompt) or ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Chat template boundary mismatch: {task_key(item)}, turn {turn}")
    labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
    if len(ids) <= len(prompt_ids):
        raise ValueError("No supervised tokens")
    return {"input_ids": ids, "labels": labels, "task_id": task_key(item),
            "traj_type": item["traj_type"], "assistant_message_index": turn,
            "prompt_tokens": len(prompt_ids), "target_tokens": len(ids)-len(prompt_ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(args.source)
    metadata_path = source.parent / ".hfd/repo_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    expected = next(x for x in metadata["siblings"] if x["rfilename"] == source.name)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    assert source.stat().st_size == expected["size"]
    assert digest.hexdigest() == expected["lfs"]["sha256"]
    rng = random.Random(args.seed)
    reservoirs, counts, task_sets = {"train": [], "eval": []}, Counter(), {"train": set(), "eval": set()}
    capacity = {"train": max(args.train_samples*4, 256), "eval": max(args.eval_samples*4, 64)}
    types = Counter()
    with source.open("rb") as stream:
        for item in ijson.items(stream, "item"):
            key = task_key(item)
            split = "eval" if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10 == 0 else "train"
            task_sets[split].add(key)
            counts[split] += 1
            types[item["traj_type"]] += 1
            pool = reservoirs[split]
            if len(pool) < capacity[split]:
                pool.append(item)
            else:
                index = rng.randrange(counts[split])
                if index < len(pool):
                    pool[index] = item
    assert not task_sets["train"] & task_sets["eval"]
    print(json.dumps({"source_trajectories": dict(counts), "trajectory_types": dict(types)}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    stats = {}
    selected_tasks = {}
    for split, limit in [("train", args.train_samples), ("eval", args.eval_samples)]:
        rows, oversized = [], 0
        rng.shuffle(reservoirs[split])
        for item_index, item in enumerate(reservoirs[split]):
            messages = json.loads(item["messages"])
            indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
            # The first prompt is a lower bound on every later prompt once
            # historical reasoning is removed. Skip schemas that cannot fit
            # rather than repeatedly tokenizing their entire long histories.
            first = tokenize_turn(tokenizer, item, indices[0])
            if first["prompt_tokens"] >= args.max_length:
                oversized += len(indices)
                continue
            accepted = 0
            for turn in indices:
                row = first if turn == indices[0] else tokenize_turn(tokenizer, item, turn)
                if row["prompt_tokens"] >= args.max_length:
                    oversized += sum(i >= turn for i in indices)
                    break
                if len(row["input_ids"]) > args.max_length:
                    oversized += 1
                    continue
                rows.append(row)
                accepted += 1
                if len(rows) >= limit or accepted >= 2:
                    break
            if len(rows) >= limit:
                break
            if item_index % 32 == 0:
                print(json.dumps({"split": split, "candidates": item_index+1,
                                  "accepted_samples": len(rows)}), flush=True)
        if len(rows) < limit:
            raise RuntimeError(f"Only {len(rows)} fitting {split} samples, wanted {limit}; increase max-length")
        rng.shuffle(rows)
        with (out / f"{split}.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
        selected_tasks[split] = {r["task_id"] for r in rows}
        stats[split] = {"samples": len(rows), "task_ids": sorted(selected_tasks[split]),
                        "min_length": min(len(r["input_ids"]) for r in rows),
                        "max_length": max(len(r["input_ids"]) for r in rows),
                        "target_tokens": sum(r["target_tokens"] for r in rows),
                        "oversized_candidates_skipped": oversized,
                        "traj_types": dict(Counter(r["traj_type"] for r in rows))}
        print(json.dumps({split: {k:v for k,v in stats[split].items() if k != "task_ids"}}), flush=True)
    assert not selected_tasks["train"] & selected_tasks["eval"]
    manifest = {"source": str(source), "source_revision": metadata["sha"],
                "source_sha256": digest.hexdigest(), "source_hash_verified": True,
                "source_trajectory_count": sum(counts.values()),
                "source_split_counts": dict(counts), "source_traj_types": dict(types),
                "model": args.model, "seed": args.seed, "max_length": args.max_length,
                "split_unit": "task_id", "history_reasoning": "removed",
                "loss_mask": "current assistant reasoning and action only",
                "truncation": "none; skip oversized turns", "pilot": True, "splits": stats}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
