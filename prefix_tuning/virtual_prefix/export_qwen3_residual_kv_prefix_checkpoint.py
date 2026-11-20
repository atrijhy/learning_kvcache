"""Export the trained FP32 memory alongside base shards for native vLLM loading."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "outputs/envscaler_qwen3_sft_20261102/full_native_lr1e-5"
ARCHITECTURE = "Qwen3ResidualKVPrefixForCausalLM"


def checkpoint_path(value="latest"):
    if value != "latest":
        path = Path(value).expanduser().resolve()
        if not (path / "model.safetensors").is_file() or not (path / "prefix_config.json").is_file():
            raise ValueError(f"Not a prefix checkpoint: {path}")
        return path
    if all((TRAINING / name).is_file() for name in
           ["model.safetensors", "prefix_config.json", "trainer_state.json", "run_status.json"]):
        status = json.loads((TRAINING / "run_status.json").read_text())
        state = json.loads((TRAINING / "trainer_state.json").read_text())
        if status.get("status") == "complete" and state.get("global_step") == status.get("global_step"):
            return TRAINING
    candidates = [p for p in TRAINING.glob("checkpoint-*")
                  if all((p / name).is_file() for name in
                         ["trainer_state.json", "model.safetensors", "prefix_config.json"])]
    if not candidates:
        raise ValueError(f"No completed checkpoint under {TRAINING}")
    return max(candidates, key=lambda p: int(p.name.split("-")[-1]))


def export(checkpoint, base_model, output, max_length=65536):
    checkpoint = checkpoint_path(str(checkpoint))
    base_model, output = Path(base_model).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    config = json.loads((base_model / "config.json").read_text())
    prefix_config = json.loads((checkpoint / "prefix_config.json").read_text())
    if config.get("model_type") != "qwen3" or prefix_config.get("method") != "qwen3_residual_kv_prefix":
        raise ValueError("Expected Qwen3 base model and residual KV prefix checkpoint")
    count = prefix_config["num_virtual_tokens"]
    if not isinstance(count, int) or count <= 0:
        raise ValueError("Invalid prefix length")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError("The trained negative-position prefix requires default RoPE")
    state = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    names = {f"model.layers.{i}.self_attn.prefix_{kind}_tokens"
             for i in range(config["num_hidden_layers"]) for kind in ["key", "value"]}
    if set(state) != names:
        raise ValueError("Checkpoint prefix names do not match the base model layers")
    for name, tensor in state.items():
        if list(tensor.shape) != [count, config["hidden_size"]] or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid prefix tensor: {name}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"Expected FP32 trained prefix tensor: {name}")
    index = json.loads((base_model / "model.safetensors.index.json").read_text())
    if set(index["weight_map"]) & names:
        raise ValueError("Base model already contains prefix tensors")
    original_context = config["max_position_embeddings"]
    config.update(architectures=[ARCHITECTURE], max_position_embeddings=max_length,
                  residual_attention_prefix_num_virtual_tokens=count,
                  residual_attention_prefix_key_init_std=prefix_config.get("key_init_std", 0.02))
    prefix_file = "residual_attention_prefix.safetensors"
    index["weight_map"].update({name: prefix_file for name in state})
    index.setdefault("metadata", {})["total_size"] = index.get("metadata", {}).get("total_size", 0) + sum(t.numel()*t.element_size() for t in state.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for source in base_model.iterdir():
            if source.name in ("config.json", "model.safetensors.index.json"):
                continue
            if source.is_symlink():
                raise ValueError(f"Refusing a symlink in base model: {source}")
            target = staging / source.name
            if source.is_dir():
                if any(p.is_symlink() for p in source.rglob("*")):
                    raise ValueError(f"Refusing nested symlink: {source}")
                shutil.copytree(source, target)
            elif source.suffix == ".safetensors":
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
            else:
                shutil.copy2(source, target)
        save_file(state, str(staging / prefix_file))
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staging / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        manifest = {"architecture": ARCHITECTURE, "base_model": str(base_model),
                    "source_checkpoint": str(checkpoint), "prefix_tensors": len(state), "prefix_dtype": "float32",
                    "native_max_position_embeddings": original_context, "max_model_len": max_length,
                    "rope_scaling_added": False, "weight_format": "base shards + independent prefix shard"}
        (staging / "prefix_merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"output": str(output), **manifest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--base-model", default="/home/liyu.468/models/Qwen3-4B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=65536)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.base_model, args.output, args.max_model_len), indent=2))


if __name__ == "__main__":
    main()
