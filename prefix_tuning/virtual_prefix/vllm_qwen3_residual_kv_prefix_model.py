"""Native vLLM Qwen3 attention with the project's shared residual KV memory."""
from __future__ import annotations
from functools import partial
import torch
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import qwen3
from vllm.model_executor.utils import set_weight_attrs
from .vllm_residual_prefix_attention import ResidualPrefixAttentionMixin


class Qwen3ResidualKVPrefixAttention(ResidualPrefixAttentionMixin, qwen3.Qwen3Attention):
    def __init__(self, *args, num_prefix_tokens, key_init_std=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_prefix_tokens = num_prefix_tokens
        # Preserve the FP32 checkpoint tables; cast only at the frozen projection,
        # exactly as in the training implementation.
        self.prefix_key_tokens = torch.nn.Parameter(
            torch.empty(num_prefix_tokens, self.hidden_size, dtype=torch.float32))
        self.prefix_value_tokens = torch.nn.Parameter(
            torch.zeros(num_prefix_tokens, self.hidden_size, dtype=torch.float32))
        torch.nn.init.normal_(self.prefix_key_tokens, std=key_init_std)
        for parameter in (self.prefix_key_tokens, self.prefix_value_tokens):
            set_weight_attrs(parameter, {"weight_loader": default_weight_loader})
        self.register_buffer("_cached_prefix_key", None, persistent=False)
        self.register_buffer("_cached_prefix_value", None, persistent=False)
        self._cached_prefix_projection_key = None
        self._prefix_projection_cache_misses = 0

    def _split_qkv(self, qkv):
        return qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = self._split_qkv(qkv)
        q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(-1, self.q_size)
        k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(-1, self.kv_size)
        q, k = self.rotary_emb(positions, q, k)
        # Native vLLM-Ascend owns request scheduling and the paged causal KV cache.
        attention = self.attn(q, k, v)
        residual = self._residual_prefix_attention(q.reshape(-1, self.num_heads, self.head_dim))
        output, _ = self.o_proj(attention + residual)
        return output


class Qwen3ResidualKVPrefixForCausalLM(qwen3.Qwen3ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_config
        count = getattr(config, "residual_attention_prefix_num_virtual_tokens", 0)
        if not isinstance(count, int) or count <= 0:
            raise ValueError("A positive residual_attention_prefix_num_virtual_tokens is required")
        rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("This trained prefix uses default RoPE; scaled RoPE is unsupported")
        original = qwen3.Qwen3Attention
        qwen3.Qwen3Attention = partial(
            Qwen3ResidualKVPrefixAttention, num_prefix_tokens=count,
            key_init_std=getattr(config, "residual_attention_prefix_key_init_std", 0.02))
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            qwen3.Qwen3Attention = original

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        expected = {name for name, _ in self.named_parameters()
                    if name.endswith((".prefix_key_tokens", ".prefix_value_tokens"))}
        missing = expected.difference(loaded)
        if missing:
            raise RuntimeError("vLLM did not load all Qwen3 prefix tensors: " + ", ".join(sorted(missing)))
        # Compute lazily after vLLM finishes the device-specific weight transforms.
        for module in self.modules():
            if isinstance(module, Qwen3ResidualKVPrefixAttention):
                module._cached_prefix_projection_key = None
                module._cached_prefix_key = None
                module._cached_prefix_value = None
        return loaded
