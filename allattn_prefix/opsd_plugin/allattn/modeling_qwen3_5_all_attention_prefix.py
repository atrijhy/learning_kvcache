"""Qwen3.5 with a residual full-attention soft prefix on EVERY layer (kvcache OPSD, 2026-11-04).

Why: the Hybrid G512 prefix lives in the Gated DeltaNet recurrent state, and that state decays to ~0 within a few
hundred user tokens (ovf/decay/gdn_decay.py), so most of the prefix is gone before the response. Here every layer gets
a shared, non-causal softmax attention branch over its own hidden-space key/value soft tokens:
  * the 8 full-attention layers: the same residual branch as the hybrid model (Qwen3_5ResidualSoftPrefixAttention);
  * the 24 Gated DeltaNet layers: o_t = DeltaNet(...)_t + softmax(q_t K_p^T / sqrt(d_k)) V_p, added to the core output
    before the gated RMSNorm / out_proj. q_t is the layer's own (pre-conv) query projection, K_p / V_p are the prefix
    soft tokens through the layer's own (frozen) in_proj_qkv key / value slices, query heads repeated to the value heads.
The prefix never enters the recurrence, positions, input ids or caches, so it cannot decay. Values start at 0, so the
initial model equals the base model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5ForConditionalGeneration


class Qwen3_5ResidualSoftPrefixAttention(Qwen3_5Attention):
    """Native causal self-attention plus a shared residual soft-prefix branch."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.num_prefix_tokens = int(
            getattr(config, "residual_attention_prefix_num_virtual_tokens", 0)
        )
        if self.num_prefix_tokens <= 0:
            raise ValueError(
                "text_config.residual_attention_prefix_num_virtual_tokens must be positive"
            )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.prefix_key_tokens = torch.nn.Parameter(
            torch.empty(self.num_prefix_tokens, config.hidden_size)
        )
        self.prefix_value_tokens = torch.nn.Parameter(
            torch.zeros(self.num_prefix_tokens, config.hidden_size)
        )
        init_std = float(
            getattr(config, "residual_attention_prefix_key_init_std", 0.02)
        )
        if init_std <= 0:
            raise ValueError("residual_attention_prefix_key_init_std must be positive")
        torch.nn.init.normal_(self.prefix_key_tokens, mean=0.0, std=init_std)

        # Prefix positions are logically immediately before user position 0.
        # Keeping user positions unchanged preserves the base model's cache and
        # context semantics while RoPE sees the correct relative distance.
        self.prefix_rotary_emb = qwen3_5.Qwen3_5TextRotaryEmbedding(config)
        self.register_buffer("_cached_prefix_key", None, persistent=False)
        self.register_buffer("_cached_prefix_value", None, persistent=False)
        self._cached_prefix_projection_key = None
        self._prefix_projection_cache_misses = 0

    def invalidate_residual_attention_prefix_cache(self) -> None:
        self._cached_prefix_key = None
        self._cached_prefix_value = None
        self._cached_prefix_projection_key = None

    def _compute_prefix_key_value(
        self,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_tokens = self.prefix_key_tokens.to(dtype)
        value_tokens = self.prefix_value_tokens.to(dtype)
        prefix_key = self.k_norm(
            self.k_proj(key_tokens).view(
                1, self.num_prefix_tokens, self.num_key_value_heads, self.head_dim
            )
        ).transpose(1, 2)
        prefix_value = self.v_proj(value_tokens).view(
            1, self.num_prefix_tokens, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        prefix_positions = torch.arange(
            -self.num_prefix_tokens,
            0,
            device=key_tokens.device,
            dtype=torch.long,
        ).unsqueeze(0)
        prefix_cos, prefix_sin = self.prefix_rotary_emb(prefix_key, prefix_positions)
        _, prefix_key = qwen3_5.apply_rotary_pos_emb(
            prefix_key,
            prefix_key,
            prefix_cos,
            prefix_sin,
        )
        return prefix_key.contiguous(), prefix_value.contiguous()

    def _get_prefix_key_value(
        self,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Training and gradient-checkpoint recomputation must retain the graph.
        if self.training or torch.is_grad_enabled():
            return self._compute_prefix_key_value(dtype)

        cache_key = (
            self.prefix_key_tokens._version,
            self.prefix_value_tokens._version,
            self.prefix_key_tokens.device,
            dtype,
        )
        if cache_key != self._cached_prefix_projection_key:
            prefix_key, prefix_value = self._compute_prefix_key_value(dtype)
            self._cached_prefix_key = prefix_key.detach()
            self._cached_prefix_value = prefix_value.detach()
            self._cached_prefix_projection_key = cache_key
            self._prefix_projection_cache_misses += 1
        return self._cached_prefix_key, self._cached_prefix_value

    def _residual_prefix_attention(
        self,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        prefix_key, prefix_value = self._get_prefix_key_value(query_states.dtype)
        batch_size = query_states.shape[0]
        prefix_key = prefix_key.expand(batch_size, -1, -1, -1)
        prefix_value = prefix_value.expand(batch_size, -1, -1, -1)
        if self.num_key_value_groups > 1:
            prefix_key = qwen3_5.repeat_kv(prefix_key, self.num_key_value_groups)
            prefix_value = qwen3_5.repeat_kv(prefix_value, self.num_key_value_groups)
        dropout_p = self.attention_dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            query_states,
            prefix_key,
            prefix_value,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
            scale=self.scaling,
        )
        return output.transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = qwen3_5.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface = qwen3_5.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            qwen3_5.eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output + self._residual_prefix_attention(query_states)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), attn_weights


class Qwen3_5ResidualSoftPrefixGatedDeltaNet(qwen3_5.Qwen3_5GatedDeltaNet):
    """Stock Gated DeltaNet plus a residual softmax attention branch over hidden-space soft tokens."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # GDN-layer prefix size: residual_gdn_prefix_num_virtual_tokens if set, else the attention-layer size
        self.num_prefix_tokens = int(getattr(config, "residual_gdn_prefix_num_virtual_tokens", 0)
                                     or getattr(config, "residual_attention_prefix_num_virtual_tokens", 0))
        if self.num_prefix_tokens <= 0:
            raise ValueError("text_config.residual_(gdn|attention)_prefix_num_virtual_tokens must be positive")
        self.prefix_key_tokens = torch.nn.Parameter(torch.empty(self.num_prefix_tokens, self.hidden_size))
        self.prefix_value_tokens = torch.nn.Parameter(torch.zeros(self.num_prefix_tokens, self.hidden_size))
        torch.nn.init.normal_(self.prefix_key_tokens, mean=0.0,
                              std=float(getattr(config, "residual_attention_prefix_key_init_std", 0.02)))
        # effective value tokens = scale * parameter. Under Adam a scale s acts as a per-group lr of s * lr for the GDN
        # branch (Adam's first step moves every element by ~lr, and this branch is ~1000x more sensitive than A64)
        self.prefix_value_scale = float(getattr(config, "residual_gdn_prefix_value_scale", 1.0))
        self._prefix_branch_out = None
        norm = self.norm
        norm_forward = norm.forward
        owner = [self]

        def norm_with_prefix(hidden_states, gate=None):
            branch = owner[0]._prefix_branch_out
            if branch is not None:
                hidden_states = hidden_states + branch.to(hidden_states.dtype)
            return norm_forward(hidden_states, gate)

        norm.forward = norm_with_prefix

    def _prefix_key_value(self, dtype):
        weight = self.in_proj_qkv.weight
        kd, vd = self.key_dim, self.value_dim
        rep = self.num_v_heads // self.num_k_heads
        key = F.linear(self.prefix_key_tokens.to(dtype), weight[kd:2 * kd].to(dtype))
        value = F.linear((self.prefix_value_tokens * self.prefix_value_scale).to(dtype), weight[2 * kd:2 * kd + vd].to(dtype))
        key = key.view(self.num_prefix_tokens, self.num_k_heads, self.head_k_dim)
        if rep > 1:
            key = key.repeat_interleave(rep, dim=1)
        value = value.view(self.num_prefix_tokens, self.num_v_heads, self.head_v_dim)
        return key, value                                         # (M, Hv, dk), (M, Hv, dv)

    def _prefix_branch(self, hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        rep = self.num_v_heads // self.num_k_heads
        query = F.linear(hidden_states, self.in_proj_qkv.weight[: self.key_dim].to(dtype))
        query = query.view(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        if rep > 1:
            query = query.repeat_interleave(rep, dim=2)
        key, value = self._prefix_key_value(dtype)
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(0, 1).unsqueeze(0).expand(bsz, -1, -1, -1),
            value.transpose(0, 1).unsqueeze(0).expand(bsz, -1, -1, -1),
            dropout_p=0.0, is_causal=False, scale=self.head_k_dim ** -0.5)
        return out.transpose(1, 2).reshape(-1, self.head_v_dim)   # (B*T*Hv, dv), same order as core_attn_out

    def forward(self, hidden_states, *args, **kwargs):
        self._prefix_branch_out = self._prefix_branch(hidden_states)
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            self._prefix_branch_out = None


class Qwen3_5AllAttentionPrefixForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 with residual full-attention soft prefixes on all 32 layers."""

    def __init__(self, config):
        orig_attn, orig_gdn = qwen3_5.Qwen3_5Attention, qwen3_5.Qwen3_5GatedDeltaNet
        qwen3_5.Qwen3_5Attention = Qwen3_5ResidualSoftPrefixAttention
        qwen3_5.Qwen3_5GatedDeltaNet = Qwen3_5ResidualSoftPrefixGatedDeltaNet
        try:
            super().__init__(config)
        finally:
            qwen3_5.Qwen3_5Attention, qwen3_5.Qwen3_5GatedDeltaNet = orig_attn, orig_gdn

    def invalidate_residual_attention_prefix_caches(self) -> None:
        for module in self.modules():
            if isinstance(module, Qwen3_5ResidualSoftPrefixAttention):
                module.invalidate_residual_attention_prefix_cache()


__all__ = ["Qwen3_5AllAttentionPrefixForConditionalGeneration", "Qwen3_5ResidualSoftPrefixAttention",
           "Qwen3_5ResidualSoftPrefixGatedDeltaNet"]
