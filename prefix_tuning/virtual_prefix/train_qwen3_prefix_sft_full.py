"""Full-data native PyTorch/Transformers SFT; the base model remains frozen.

Every assistant turn in every trajectory becomes a sample. CPU workers encode
samples on demand, avoiding a long preprocessing pass and retaining all data.
The reverse-turn cutoff allocation follows LlamaFactory's mask_history SFT
processor (Apache-2.0), including its source/target length allocation:
https://github.com/hiyouga/LlamaFactory/tree/main/src/llamafactory/data/processor
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ijson
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from prefix_tuning.virtual_prefix.modeling_qwen3_residual_kv_prefix import (
    install_residual_kv_prefix, load_prefix_state_dict, prefix_state_dict,
)


def prepare_index(source, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(source).resolve()
    metadata = json.loads((source.parent / ".hfd/repo_metadata.json").read_text())
    entry = next(x for x in metadata["siblings"] if x["rfilename"] == source.name)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    assert source.stat().st_size == entry["size"] and digest.hexdigest() == entry["lfs"]["sha256"]
    indices, types, trajectories = [], {}, 0
    with source.open("rb") as stream:
        for trajectory_id, item in enumerate(ijson.items(stream, "item")):
            messages = json.loads(item["messages"])
            indices.extend((trajectory_id, i) for i,m in enumerate(messages) if m["role"] == "assistant")
            types[item["traj_type"]] = types.get(item["traj_type"], 0)+1
            trajectories += 1
    np.save(output / "turn_index.npy", np.asarray(indices, dtype=np.int32))
    manifest = {"source": str(source), "source_revision": metadata["sha"],
                "source_sha256": digest.hexdigest(), "source_hash_verified": True,
                "source_trajectories": trajectories, "trajectory_types": types,
                "training_samples": len(indices), "all_assistant_turns": True,
                "validation_split": 0, "max_length": 32768, "mask_history": True,
                "history_reasoning": "removed", "encoding": "lazy_cpu_workers",
                "truncation": "last-turn-first budget allocation, matching LlamaFactory mask_history"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest), flush=True)


def allocate_lengths(source, target, budget):
    # Same allocation as LlamaFactory infer_seqlen; targets retain priority.
    if 2*target < budget:
        kept_target = min(target, budget)
    elif 2*source < budget:
        kept_target = min(target, budget-source)
    else:
        kept_target = min(target, int(budget*target/(source+target)))
    return min(source, budget-kept_target), kept_target


class FullTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, model):
        self.manifest = json.loads((Path(data_dir)/"manifest.json").read_text())
        self.index = np.load(Path(data_dir)/"turn_index.npy", mmap_mode="r")
        with open(self.manifest["source"]) as stream:
            raw = json.load(stream)
        self.records = [(item["tools"], item["messages"]) for item in raw]
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        assert len(raw) == self.manifest["source_trajectories"]
        assert len(self.index) == self.manifest["training_samples"]

    def __len__(self):
        return len(self.index)

    @lru_cache(maxsize=16)
    def trajectory(self, trajectory_id):
        tools, messages = self.records[trajectory_id]
        return json.loads(tools), json.loads(messages)

    def __getitem__(self, index):
        trajectory_id, turn = map(int, self.index[index])
        tools, original = self.trajectory(trajectory_id)
        messages = copy.deepcopy(original[:turn+1])
        for message in messages[:-1]:
            if message["role"] == "assistant":
                message.pop("reasoning_content", None)
                if "</think>" in (message.get("content") or ""):
                    message["content"] = message["content"].split("</think>")[-1].lstrip("\n")
        text = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False)
        # Qwen's native chat format is the format used by the author-provided
        # step1 conversion. A target ends at im_end; the following newline is
        # part of the next source, as in LlamaFactory's Qwen3 turn template.
        header, end = "<|im_start|>assistant\n", "<|im_end|>"
        pairs, previous_end, cursor = [], 0, 0
        while True:
            start = text.find(header, cursor)
            if start < 0:
                break
            target_start = start+len(header)
            target_end = text.find(end, target_start)
            if target_end < 0:
                raise ValueError(f"Missing assistant terminator in trajectory {trajectory_id}")
            target_end += len(end)
            pairs.append((text[previous_end:target_start], text[target_start:target_end]))
            previous_end, cursor = target_end, target_end
        expected = sum(m["role"] == "assistant" for m in messages)
        if len(pairs) != expected:
            raise ValueError(f"Assistant boundary mismatch in trajectory {trajectory_id}, turn {turn}")
        encoded, budget = [], self.manifest["max_length"]
        for i, (source, target) in enumerate(reversed(pairs)):
            if budget <= 0:
                break
            source_ids = self.tokenizer.encode(source, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            source_length, target_length = allocate_lengths(len(source_ids), len(target_ids), budget)
            ids = source_ids[:source_length]+target_ids[:target_length]
            labels = [-100]*source_length + (target_ids[:target_length] if i == 0 else [-100]*target_length)
            encoded.append((ids, labels))
            budget -= len(ids)
        ids, labels = [], []
        for turn_ids, turn_labels in reversed(encoded):
            ids.extend(turn_ids)
            labels.extend(turn_labels)
        if not any(x != -100 for x in labels[1:]):
            raise ValueError(f"No target labels in trajectory {trajectory_id}, turn {turn}")
        return {"input_ids": ids, "labels": labels}


def collate_one(rows):
    assert len(rows) == 1, "The agreed micro batch size is 1"
    ids = torch.tensor([rows[0]["input_ids"]], dtype=torch.long)
    return {"input_ids": ids, "labels": torch.tensor([rows[0]["labels"]], dtype=torch.long),
            "attention_mask": torch.ones_like(ids)}


class PrefixOnlyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Retain the preceding logit for the causal shift, then let the native
        # Transformers loss handle accumulation and distributed normalization.
        inputs = dict(inputs)
        labels = inputs["labels"]
        first = int((labels[0] != -100).nonzero()[0].item())
        keep = labels.shape[-1]-max(first-1, 0)
        inputs["labels"] = labels[:, -keep:]
        inputs["logits_to_keep"] = keep
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def _save(self, output_dir=None, state_dict=None):
        output = Path(output_dir or self.args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        state = prefix_state_dict(self.model)
        save_file(state, str(output/"model.safetensors"), metadata={"format": "pt"})
        (output/"model.safetensors").chmod(0o644)
        (output/"prefix_config.json").write_text(json.dumps({
            "method": "qwen3_residual_kv_prefix", "base_model": self.model.config._name_or_path,
            "num_virtual_tokens": 256, "key_init_std": 0.02, "prefix_dtype": "float32",
            "base_weights_frozen": True, "tensor_count": len(state)}, indent=2)+"\n")
        torch.save(self.args, output/"training_args.bin")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        load_prefix_state_dict(model or self.model, load_file(str(Path(resume_from_checkpoint)/"model.safetensors")))


class RunStatus(TrainerCallback):
    def __init__(self, output, config):
        self.output, self.config = Path(output), config

    def write(self, state, status, logs=None):
        if not state.is_world_process_zero:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        record = {"status": status, "global_step": state.global_step, "max_steps": state.max_steps,
                  "epoch": state.epoch, "updated_at_unix": time.time(), "logs": logs or {}}
        temp = self.output/"run_status.json.tmp"
        temp.write_text(json.dumps(record, indent=2)+"\n")
        temp.replace(self.output/"run_status.json")
        if logs:
            with (self.output/"metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record)+"\n")

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            (self.output/"run_config.json").write_text(json.dumps(self.config, indent=2)+"\n")
        self.write(state, "running")

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.write(state, "running", logs)

    def on_train_end(self, args, state, control, **kwargs):
        self.write(state, "complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--model", default="/home/liyu.468/models/Qwen3-4B")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    if args.prepare_only:
        if not args.source:
            parser.error("--prepare-only requires --source")
        prepare_index(args.source, args.data_dir)
        return
    import torch_npu  # noqa: F401
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 8:
        raise ValueError("Full run requires 8 ranks for batch 8 x 1 x 32 = 256")
    torch.npu.set_device(local_rank)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    output = Path(args.output_dir)
    if int(os.environ.get("RANK", "0")) == 0:
        if output.exists() and not args.resume_from_checkpoint:
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.mkdir(parents=True, exist_ok=True)
    # Initialize the framework before loading weights so every rank receives
    # its own NPU, sampler and optimizer state through Accelerate/HCCL.
    training = TrainingArguments(
        output_dir=str(output), num_train_epochs=3, per_device_train_batch_size=1,
        gradient_accumulation_steps=32, learning_rate=1e-5, lr_scheduler_type="cosine",
        warmup_steps=0.1, bf16=True, optim="adamw_torch", weight_decay=0.0,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_backend="hccl", ddp_find_unused_parameters=False, ddp_broadcast_buffers=False,
        ddp_timeout=180000000, dataloader_num_workers=2, dataloader_pin_memory=False,
        dataloader_persistent_workers=True, logging_steps=1, save_steps=100,
        save_strategy="steps", eval_strategy="no", report_to="none", seed=42, data_seed=42,
        remove_unused_columns=False, disable_tqdm=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    model.config.use_cache = False
    trainable = install_residual_kv_prefix(model, 256)
    assert len(trainable) == 72 and all("prefix_" in name for name in trainable)
    dataset = FullTrajectoryDataset(args.data_dir, args.model)
    config = {"framework": "native PyTorch + Transformers Trainer/Accelerate",
              "base_model": args.model, "base_weights_frozen": True,
              "trainable_parameters": sum(p.numel() for p in trainable.values()),
              "prefix_tokens_per_layer": 256, "attention_layers": 36,
              "learning_rate": 1e-5, "epochs": 3, "max_length": 32768,
              "world_size": world_size, "micro_batch_per_rank": 1, "gradient_accumulation": 32,
              "global_batch_size": 256, "scheduler": "cosine", "warmup_ratio": 0.1,
              "precision": "BF16 base / FP32 trainable prefix", "save_steps": 100,
              "initialization": "fresh base, not a pilot checkpoint",
              "data": dataset.manifest, "torch": torch.__version__,
              "checkpoint_contents": "prefix weights, native trainer/optimizer/scheduler/RNG state"}
    trainer = PrefixOnlyTrainer(model=model, args=training, train_dataset=dataset,
                               data_collator=collate_one, callbacks=[RunStatus(output, config)])
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
