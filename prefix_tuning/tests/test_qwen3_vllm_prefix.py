"""Compare the shared vLLM memory math to the training model on CPU."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
from prefix_tuning.virtual_prefix.modeling_qwen3_residual_kv_prefix import install_residual_kv_prefix
from prefix_tuning.virtual_prefix.vllm_residual_prefix_attention import ResidualPrefixAttentionMixin
from prefix_tuning.virtual_prefix.export_qwen3_residual_kv_prefix_checkpoint import export


def test_vllm_memory_matches_hf_and_isolates_requests():
    torch.manual_seed(29)
    config = Qwen3Config(vocab_size=128, hidden_size=48, intermediate_size=96,
                        num_hidden_layers=1, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, max_position_embeddings=128)
    hf = Qwen3ForCausalLM(config).eval()
    install_residual_kv_prefix(hf, 4)
    attention = hf.model.layers[0].self_attn
    with torch.no_grad():
        attention.prefix_value_tokens.normal_()

    class Packed(nn.Module):
        def forward(self, hidden):
            return torch.cat([attention.q_proj(hidden), attention.k_proj(hidden), attention.v_proj(hidden)], -1), None

    class NativeMemory(ResidualPrefixAttentionMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = Packed()
            self.k_norm = attention.k_norm
            self.prefix_key_tokens = attention.prefix_key_tokens
            self.prefix_value_tokens = attention.prefix_value_tokens
            self.num_prefix_tokens, self.num_heads, self.num_kv_heads, self.head_dim = 4, 4, 2, 16
            self.q_size, self.kv_size, self.scaling = 64, 32, 16**-0.5
            self.rotary_emb = SimpleNamespace(rotary_dim=16, base=config.rope_parameters["rope_theta"], is_neox_style=True)
            self._cached_prefix_projection_key = None
            self._prefix_projection_cache_misses = 0

        def _split_qkv(self, values):
            return values.split([64, 32, 32], dim=-1)

    native = NativeMemory()
    key = attention.k_norm(attention.k_proj(attention.prefix_key_tokens).view(1, 4, 2, 16)).transpose(1, 2)
    value = attention.v_proj(attention.prefix_value_tokens).view(1, 4, 2, 16).transpose(1, 2)
    cos, sin = attention.prefix_rotary_emb(key, torch.arange(-4, 0).unsqueeze(0))
    _, key = apply_rotary_pos_emb(key, key, cos, sin)
    query = torch.randn(2, 4, 5, 16)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, repeat_kv(key, 2).expand(2, -1, -1, -1), repeat_kv(value, 2).expand(2, -1, -1, -1),
        is_causal=False, scale=attention.scaling)
    flattened = query.transpose(1, 2).reshape(-1, 4, 16)
    result = native._residual_prefix_attention(flattened).view(2, 5, 4, 16).transpose(1, 2)
    torch.testing.assert_close(result, expected)
    whole = native._residual_prefix_attention(flattened)
    chunked = torch.cat([native._residual_prefix_attention(part) for part in flattened.split(3)])
    torch.testing.assert_close(whole, chunked)
    assert native._prefix_projection_cache_misses == 1
    with torch.no_grad():
        attention.prefix_value_tokens.add_(0.5)
    changed = native._residual_prefix_attention(flattened)
    assert native._prefix_projection_cache_misses == 2
    assert not torch.allclose(whole, changed)


def test_export_keeps_base_immutable_and_indexes_every_prefix(tmp_path):
    base, checkpoint, output = [tmp_path / name for name in ["base", "checkpoint-1", "export"]]
    base.mkdir(); checkpoint.mkdir()
    config = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
              "num_hidden_layers": 2, "hidden_size": 8, "max_position_embeddings": 40960}
    original = json.dumps(config)
    (base / "config.json").write_text(original)
    save_file({"model.embed_tokens.weight": torch.ones(8, 8)}, str(base / "base.safetensors"))
    (base / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 256}, "weight_map": {"model.embed_tokens.weight": "base.safetensors"}}))
    (base / "tokenizer_config.json").write_text("{}")
    state = {f"model.layers.{i}.self_attn.prefix_{kind}_tokens": torch.randn(4, 8)
             for i in range(2) for kind in ["key", "value"]}
    save_file(state, str(checkpoint / "model.safetensors"))
    (checkpoint / "prefix_config.json").write_text(json.dumps({"method": "qwen3_residual_kv_prefix", "num_virtual_tokens": 4}))
    report = export(checkpoint, base, output)
    assert report["prefix_tensors"] == 4
    assert (base / "config.json").read_text() == original
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(state) < set(index["weight_map"])
    loaded = load_file(str(output / "residual_attention_prefix.safetensors"))
    for name, value in state.items():
        torch.testing.assert_close(value, loaded[name], rtol=0, atol=0)
    assert not any(p.is_symlink() for p in output.rglob("*"))
    assert json.loads((output / "config.json").read_text())["max_position_embeddings"] == 65536
