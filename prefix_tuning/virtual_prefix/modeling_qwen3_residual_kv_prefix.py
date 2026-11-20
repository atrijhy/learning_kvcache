"""The project's residual attention memory, adapted to full-attention Qwen3.

Each layer owns independent hidden-space K/V tables. Frozen projections,
K normalization and negative-position RoPE turn them into a shared KV memory.
The memory has its own softmax; its output is added to causal self-attention.
Zero V initialization preserves the base model exactly at installation time.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3 import modeling_qwen3 as qwen3


class Qwen3ResidualKVPrefixAttention(qwen3.Qwen3Attention):
    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, **kwargs):
        shape = hidden_states.shape[:-1]
        head_shape = (*shape, -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(head_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)
        query, key = qwen3.apply_rotary_pos_emb(query, key, *position_embeddings)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        interface = qwen3.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, qwen3.eager_attention_forward)
        native, weights = interface(
            self, query, key, value, attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)

        count = self.prefix_key_tokens.shape[0]
        kv_shape = (1, count, self.config.num_key_value_heads, self.head_dim)
        prefix_key = self.k_norm(
            self.k_proj(self.prefix_key_tokens.to(key.dtype)).view(kv_shape)
        ).transpose(1, 2)
        prefix_value = self.v_proj(
            self.prefix_value_tokens.to(value.dtype)).view(kv_shape).transpose(1, 2)
        positions = torch.arange(-count, 0, device=key.device).unsqueeze(0)
        cos, sin = self.prefix_rotary_emb(prefix_key, positions)
        _, prefix_key = qwen3.apply_rotary_pos_emb(prefix_key, prefix_key, cos, sin)
        prefix_key = qwen3.repeat_kv(prefix_key, self.num_key_value_groups)
        prefix_value = qwen3.repeat_kv(prefix_value, self.num_key_value_groups)
        residual = F.scaled_dot_product_attention(
            query, prefix_key.expand(shape[0], -1, -1, -1),
            prefix_value.expand(shape[0], -1, -1, -1),
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False, scale=self.scaling)
        combined = native + residual.transpose(1, 2)
        return self.o_proj(combined.reshape(*shape, -1).contiguous()), weights


def install_residual_kv_prefix(model, num_virtual_tokens=256, key_init_std=0.02):
    """Freeze the base and add FP32 memory parameters without reloading weights."""
    if model.config.model_type != "qwen3":
        raise ValueError("This adapter requires the full-attention Qwen3 architecture")
    if num_virtual_tokens <= 0 or key_init_std <= 0:
        raise ValueError("Prefix length and key initialization std must be positive")
    if any(isinstance(m, Qwen3ResidualKVPrefixAttention) for m in model.modules()):
        raise ValueError("A residual KV prefix is already installed")
    model.requires_grad_(False)
    layers = 0
    for module in list(model.modules()):
        if not isinstance(module, qwen3.Qwen3Attention):
            continue
        module.__class__ = Qwen3ResidualKVPrefixAttention
        device = module.k_proj.weight.device
        size = (num_virtual_tokens, model.config.hidden_size)
        module.prefix_key_tokens = nn.Parameter(torch.empty(size, device=device, dtype=torch.float32))
        module.prefix_value_tokens = nn.Parameter(torch.zeros(size, device=device, dtype=torch.float32))
        nn.init.normal_(module.prefix_key_tokens, std=key_init_std)
        module.prefix_rotary_emb = qwen3.Qwen3RotaryEmbedding(model.config).to(device)
        layers += 1
    if layers != model.config.num_hidden_layers:
        raise ValueError(f"Expected {model.config.num_hidden_layers} attention layers, got {layers}")
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def prefix_state_dict(model):
    return {name: p.detach().cpu().contiguous().clone() for name, p in model.named_parameters()
            if name.endswith((".prefix_key_tokens", ".prefix_value_tokens"))}


def load_prefix_state_dict(model, state):
    expected = {name for name, p in model.named_parameters()
                if name.endswith((".prefix_key_tokens", ".prefix_value_tokens"))}
    if set(state) != expected:
        raise ValueError(f"Prefix names differ: missing={expected-set(state)}, extra={set(state)-expected}")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            parameters[name].copy_(tensor)
