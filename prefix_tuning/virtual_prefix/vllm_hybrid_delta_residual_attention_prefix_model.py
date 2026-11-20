"""Native vLLM Qwen3.5 model with Delta and residual attention prefixes."""

from __future__ import annotations

import torch
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import qwen3_5
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.utils import set_weight_attrs

from .vllm_residual_prefix_attention import ResidualPrefixAttentionMixin

from .vllm_independent_delta_kv_prefix_model import (
    Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration,
)


class Qwen3_5ResidualSoftPrefixAttention(ResidualPrefixAttentionMixin, Qwen3NextAttention):
    """Native self-attention plus a shared non-causal prefix-only branch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_prefix_tokens = int(
            getattr(
                self.config,
                "residual_attention_prefix_num_virtual_tokens",
                0,
            )
        )
        if self.num_prefix_tokens <= 0:
            raise ValueError(
                "text_config.residual_attention_prefix_num_virtual_tokens "
                "must be positive"
            )

        self.prefix_key_tokens = torch.nn.Parameter(
            torch.empty(self.num_prefix_tokens, self.hidden_size)
        )
        self.prefix_value_tokens = torch.nn.Parameter(
            torch.zeros(self.num_prefix_tokens, self.hidden_size)
        )
        init_std = float(
            getattr(self.config, "residual_attention_prefix_key_init_std", 0.02)
        )
        if init_std <= 0:
            raise ValueError("residual_attention_prefix_key_init_std must be positive")
        torch.nn.init.normal_(self.prefix_key_tokens, mean=0.0, std=init_std)
        for parameter in (self.prefix_key_tokens, self.prefix_value_tokens):
            # Hidden-space soft tokens are replicated. TP=1 is used by the OPSD
            # launcher; the default loader also gives correct replicated TP loads.
            set_weight_attrs(parameter, {"weight_loader": default_weight_loader})

        self.register_buffer("_cached_prefix_key", None, persistent=False)
        self.register_buffer("_cached_prefix_value", None, persistent=False)
        self._cached_prefix_projection_key = None
        self._prefix_projection_cache_misses = 0

    def invalidate_residual_attention_prefix_cache(self) -> None:
        self._cached_prefix_key = None
        self._cached_prefix_value = None
        self._cached_prefix_projection_key = None

    def _split_qkv(self, qkv: torch.Tensor):
        if self.attn_output_gate:
            return qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        return qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn_output_gate:
            q_gate, k, v = self._split_qkv(qkv)
            q_gate = q_gate.view(-1, self.num_heads, self.head_dim * 2)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(-1, self.q_size)
            gate = gate.reshape(-1, self.q_size)
        else:
            q, k, v = self._split_qkv(qkv)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.q_size
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.kv_size
        )
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        query = q.view(-1, self.num_heads, self.head_dim)
        attn_output = attn_output + self._residual_prefix_attention(query)
        if self.attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate)
        output[:], _ = self.o_proj(attn_output)


class Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration(
    Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration
):
    """Native vLLM Qwen3.5 with both approved prefix mechanisms."""

    def __init__(self, *, vllm_config, prefix: str = "model"):
        original_cls = qwen3_5.Qwen3NextAttention
        qwen3_5.Qwen3NextAttention = Qwen3_5ResidualSoftPrefixAttention
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            qwen3_5.Qwen3NextAttention = original_cls

    def invalidate_residual_attention_prefix_caches(self) -> None:
        for module in self.modules():
            if isinstance(module, Qwen3_5ResidualSoftPrefixAttention):
                module.invalidate_residual_attention_prefix_cache()

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        expected = {
            name
            for name, _ in self.named_parameters()
            if name.endswith((".prefix_key_tokens", ".prefix_value_tokens"))
        }
        missing = expected.difference(loaded)
        if missing:
            raise RuntimeError(
                "vLLM did not load all residual attention prefix tensors: "
                + ", ".join(sorted(missing))
            )
        self.invalidate_residual_attention_prefix_caches()
        return loaded


__all__ = ["Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration"]
