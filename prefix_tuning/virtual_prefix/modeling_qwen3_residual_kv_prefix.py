"""The project's residual attention memory, adapted to full-attention Qwen3.

Each layer owns independent hidden-space K/V tables. Frozen projections,
K normalization and negative-position RoPE turn them into a shared KV memory.
The memory has its own softmax; its output is added to causal self-attention.
Zero V initialization preserves the base model exactly at installation time.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3 import modeling_qwen3 as qwen3

from .joint_prefix_attention import (
    causal_context_attention_with_lse,
    nope_token_joint_merge,
    token_null_attention,
)

PREFIX_SUFFIXES = (
    ".prefix_key_tokens",
    ".prefix_value_tokens",
    ".prefix_logit_bias",
    ".prefix_null_proj",
)


class Qwen3ResidualKVPrefixAttention(qwen3.Qwen3Attention):
    def invalidate_residual_attention_prefix_cache(self) -> None:
        self._cached_prefix_key = None
        self._cached_prefix_value = None
        self._cached_prefix_projection_key = None

    def train(self, mode: bool = True):
        if mode:
            self.invalidate_residual_attention_prefix_cache()
        return super().train(mode)

    def _compute_nope_prefix_kv(self, dtype: torch.dtype):
        count = self.prefix_key_tokens.shape[0]
        kv_shape = (1, count, self.config.num_key_value_heads, self.head_dim)
        prefix_key = self.k_norm(
            self.k_proj(self.prefix_key_tokens.to(dtype)).view(kv_shape)
        ).transpose(1, 2).contiguous()
        prefix_value = self.v_proj(
            self.prefix_value_tokens.to(dtype)
        ).view(kv_shape).transpose(1, 2).contiguous()
        return prefix_key, prefix_value

    def _get_nope_prefix_kv(self, dtype: torch.dtype):
        if self.training or torch.is_grad_enabled():
            return self._compute_nope_prefix_kv(dtype)
        cache_key = (
            self.prefix_key_tokens._version,
            self.prefix_value_tokens._version,
            self.prefix_key_tokens.device,
            dtype,
        )
        if getattr(self, "_cached_prefix_projection_key", None) != cache_key:
            pk, pv = self._compute_nope_prefix_kv(dtype)
            self._cached_prefix_key = pk.detach()
            self._cached_prefix_value = pv.detach()
            self._cached_prefix_projection_key = cache_key
        return self._cached_prefix_key, self._cached_prefix_value

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, **kwargs):
        shape = hidden_states.shape[:-1]
        head_shape = (*shape, -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(head_shape)).transpose(1, 2)
        mode = getattr(self, "prefix_mode", "residual")
        pre_rope_query = query if mode in ("nope_token_joint", "nope_token_null_additive") else None
        key = self.k_norm(self.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)
        query, key = qwen3.apply_rotary_pos_emb(query, key, *position_embeddings)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        enabled = not getattr(self, "prefix_disabled", False)
        if enabled and mode == "nope_token_joint":
            use_flex = (
                getattr(self, "_use_compiled_prefix", False)
                and self.training
                and query.shape[0] == 1
                and query.shape[-2] == key.shape[-2]
            )
            ctx_out, ctx_lse = causal_context_attention_with_lse(
                query, key, value,
                scale=self.scaling,
                attention_mask=None if use_flex else attention_mask,
                use_flex=use_flex,
            )
            prefix_key, prefix_value = self._get_nope_prefix_kv(pre_rope_query.dtype)
            null = F.linear(
                hidden_states.float(), self.prefix_null_proj.float()
            ).transpose(1, 2).unsqueeze(-1)
            out, lam, delta, rel_lse = nope_token_joint_merge(
                ctx_out, ctx_lse, pre_rope_query, prefix_key, prefix_value,
                self.prefix_logit_bias, null, scale=self.scaling,
            )
            cal = getattr(self, "prefix_calibration", None)
            if cal is not None:
                cal(rel_lse.detach())
            mon = getattr(self, "prefix_monitor", None)
            if mon is not None:
                mon(lam.detach(), delta.transpose(1, 2).detach(), ctx_out.transpose(1, 2).detach())
            return self.o_proj(out.transpose(1, 2).reshape(*shape, -1).contiguous()), None

        getter = getattr(qwen3.ALL_ATTENTION_FUNCTIONS, "get_interface", qwen3.ALL_ATTENTION_FUNCTIONS.get)
        interface = getter(self.config._attn_implementation, qwen3.eager_attention_forward)
        native, weights = interface(
            self, query, key, value, attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)

        if not enabled:
            return self.o_proj(native.reshape(*shape, -1).contiguous()), weights

        if mode == "nope_token_null_additive":
            prefix_key, prefix_value = self._get_nope_prefix_kv(pre_rope_query.dtype)
            branch_sizes = getattr(self, "prefix_branch_sizes", None) or [prefix_key.shape[-2]]
            num_branches = len(branch_sizes)
            total_delta = None
            offset = 0
            for bi, p_len in enumerate(branch_sizes):
                pk_b = prefix_key[:, :, offset:offset + p_len, :]
                pv_b = prefix_value[:, :, offset:offset + p_len, :]
                offset += p_len
                bias_b = self.prefix_logit_bias[bi] if num_branches > 1 else self.prefix_logit_bias
                null_w_b = self.prefix_null_proj[bi] if num_branches > 1 else self.prefix_null_proj
                null_b = F.linear(hidden_states.float(), null_w_b.float()).transpose(1, 2).unsqueeze(-1)
                delta_b, lam_b, raw_lse_b = token_null_attention(
                    pre_rope_query, pk_b, pv_b, bias_b, null_b, scale=self.scaling
                )
                total_delta = delta_b if total_delta is None else (total_delta + delta_b)
                cal = getattr(self, "prefix_calibration", None)
                if cal is not None and num_branches == 1:
                    cal(raw_lse_b.detach())
                mon = getattr(self, "prefix_monitor", None)
                if mon is not None:
                    mon(lam_b.detach(), delta_b.transpose(1, 2).detach(), native.detach())
            combined = native + total_delta.transpose(1, 2).to(native.dtype)
            return self.o_proj(combined.reshape(*shape, -1).contiguous()), weights

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


def install_residual_kv_prefix(model, num_virtual_tokens=256, key_init_std=0.02, mode="residual", branch_sizes=None):
    """Freeze the base and add FP32 memory parameters without reloading weights."""
    if model.config.model_type != "qwen3":
        raise ValueError("This adapter requires the full-attention Qwen3 architecture")
    if num_virtual_tokens <= 0 or key_init_std <= 0:
        raise ValueError("Prefix length and key initialization std must be positive")
    if mode not in ("residual", "nope_token_joint", "nope_token_null_additive"):
        raise ValueError(f"Unsupported Qwen3 prefix mode: {mode}")
    if any(isinstance(m, Qwen3ResidualKVPrefixAttention) for m in model.modules()):
        raise ValueError("A residual KV prefix is already installed")
    if branch_sizes is None:
        branch_sizes = [num_virtual_tokens]
    else:
        branch_sizes = [int(x) for x in branch_sizes]
        if sum(branch_sizes) != num_virtual_tokens:
            raise ValueError(f"sum(branch_sizes)={sum(branch_sizes)} != num_virtual_tokens={num_virtual_tokens}")
    num_branches = len(branch_sizes)
    model.requires_grad_(False)
    layers = 0
    num_heads = model.config.num_attention_heads
    init_bias = -math.log(num_virtual_tokens) - 8.0
    for module in list(model.modules()):
        if not isinstance(module, qwen3.Qwen3Attention):
            continue
        module.__class__ = Qwen3ResidualKVPrefixAttention
        module.prefix_mode = mode
        module.prefix_branch_sizes = list(branch_sizes)
        module.prefix_disabled = False
        module._use_compiled_prefix = False
        module.prefix_calibration = None
        module.prefix_monitor = None
        module._cached_prefix_key = None
        module._cached_prefix_value = None
        module._cached_prefix_projection_key = None
        device = module.k_proj.weight.device
        size = (num_virtual_tokens, model.config.hidden_size)
        module.prefix_key_tokens = nn.Parameter(torch.empty(size, device=device, dtype=torch.float32))
        module.prefix_value_tokens = nn.Parameter(torch.zeros(size, device=device, dtype=torch.float32))
        nn.init.normal_(module.prefix_key_tokens, std=key_init_std)
        if mode in ("nope_token_joint", "nope_token_null_additive"):
            b_shape = (num_branches, num_heads) if num_branches > 1 else (num_heads,)
            n_shape = (num_branches, num_heads, model.config.hidden_size) if num_branches > 1 else (num_heads, model.config.hidden_size)
            module.prefix_logit_bias = nn.Parameter(
                torch.full(b_shape, init_bias, device=device, dtype=torch.float32)
            )
            module.prefix_null_proj = nn.Parameter(
                torch.zeros(n_shape, device=device, dtype=torch.float32)
            )
        else:
            module.prefix_rotary_emb = qwen3.Qwen3RotaryEmbedding(model.config).to(device)
        layers += 1
    if layers != model.config.num_hidden_layers:
        raise ValueError(f"Expected {model.config.num_hidden_layers} attention layers, got {layers}")
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def prefix_state_dict(model):
    return {name: p.detach().cpu().contiguous().clone() for name, p in model.named_parameters()
            if name.endswith(PREFIX_SUFFIXES)}


def load_prefix_state_dict(model, state):
    expected = {name for name, p in model.named_parameters()
                if name.endswith(PREFIX_SUFFIXES)}
    if set(state) != expected:
        raise ValueError(f"Prefix names differ: missing={expected-set(state)}, extra={set(state)-expected}")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            parameters[name].copy_(tensor)
