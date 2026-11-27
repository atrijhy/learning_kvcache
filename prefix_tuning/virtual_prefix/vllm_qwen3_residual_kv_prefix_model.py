"""Native vLLM Qwen3 attention with shared residual or nope_token_joint KV memory."""
from __future__ import annotations
from functools import partial
from weakref import ref

import torch
import torch.nn.functional as F
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import qwen3
from vllm.model_executor.utils import set_weight_attrs
from .joint_prefix_attention import token_null_attention
from .vllm_residual_prefix_attention import ResidualPrefixAttentionMixin

PREFIX_SUFFIXES = (
    ".prefix_key_tokens",
    ".prefix_value_tokens",
    ".prefix_logit_bias",
    ".prefix_null_proj",
)


class Qwen3ResidualKVPrefixAttention(ResidualPrefixAttentionMixin, qwen3.Qwen3Attention):
    def __init__(self, *args, hf_config, num_prefix_tokens, key_init_std=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = hf_config
        self.num_prefix_tokens = num_prefix_tokens
        self.joint_prefix = bool(getattr(hf_config, "residual_attention_prefix_joint_softmax", False))
        self.prefix_mode = getattr(hf_config, "residual_attention_prefix_mode", "residual")
        self.nope_token_joint = self.joint_prefix and self.prefix_mode == "nope_token_joint"
        self.nope_token_null_additive = self.prefix_mode == "nope_token_null_additive"
        branch_sizes = getattr(hf_config, "residual_attention_prefix_branch_sizes", None) or [num_prefix_tokens]
        self.prefix_branch_sizes = [int(x) for x in branch_sizes]
        self.num_prefix_branches = len(self.prefix_branch_sizes)
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
        self._prefix_cache_refreshed = False

        if self.nope_token_null_additive:
            b_shape = (self.num_prefix_branches, self.num_heads) if self.num_prefix_branches > 1 else (self.num_heads,)
            n_shape = (
                (self.num_prefix_branches, self.num_heads, self.hidden_size)
                if self.num_prefix_branches > 1 else (self.num_heads, self.hidden_size)
            )
            self.prefix_logit_bias = torch.nn.Parameter(torch.zeros(b_shape, dtype=torch.float32), requires_grad=False)
            self.prefix_null_proj = torch.nn.Parameter(torch.zeros(n_shape, dtype=torch.float32), requires_grad=False)
            set_weight_attrs(self.prefix_logit_bias, {"weight_loader": default_weight_loader})
            set_weight_attrs(self.prefix_null_proj, {"weight_loader": default_weight_loader})

        if self.joint_prefix:
            from .vllm_joint_prefix import FlashAttentionImpl, JointPrefixFlashAttention

            impl = self.attn.impl
            if (type(impl) is not FlashAttentionImpl or impl.kv_cache_dtype.startswith("fp8")
                    or impl.vllm_flash_attn_version != 3):
                raise ValueError("Joint prefix rollout requires FlashAttention 3 and an unquantized KV cache")
            self.prefix_logit_bias = torch.nn.Parameter(
                torch.zeros(self.num_heads, dtype=torch.float32)
            )
            set_weight_attrs(self.prefix_logit_bias, {"weight_loader": default_weight_loader})
            if self.nope_token_joint:
                self.prefix_null_proj = torch.nn.Parameter(
                    torch.zeros(self.num_heads, self.hidden_size, dtype=torch.float32),
                    requires_grad=False,
                )
                set_weight_attrs(self.prefix_null_proj, {"weight_loader": default_weight_loader})
            self.register_buffer("_joint_offsets", None, persistent=False)
            self.register_buffer("_joint_key_offsets", None, persistent=False)
            impl.__class__ = JointPrefixFlashAttention
            impl.prefix_layer = ref(self)

    def _compute_nope_prefix_kv_vllm(self, dtype: torch.dtype):
        key_tokens = self.prefix_key_tokens.to(dtype)
        value_tokens = self.prefix_value_tokens.to(dtype)
        # Project through vLLM's fused qkv_proj without RoPE (exact match to HF _compute_nope_prefix_kv!)
        k_qkv, _ = self.qkv_proj(key_tokens)
        _, pk, _ = self._split_qkv(k_qkv)
        pk = self.k_norm(pk.reshape(-1, self.num_kv_heads, self.head_dim)).transpose(0, 1).unsqueeze(0).contiguous()
        v_qkv, _ = self.qkv_proj(value_tokens)
        _, _, pv = self._split_qkv(v_qkv)
        pv = pv.reshape(-1, self.num_kv_heads, self.head_dim).transpose(0, 1).unsqueeze(0).contiguous()
        return pk, pv

    def _get_nope_prefix_kv_vllm(self, dtype: torch.dtype):
        cache_key = (
            self.prefix_key_tokens._version,
            self.prefix_value_tokens._version,
            self.prefix_key_tokens.device,
            dtype,
        )
        if self._cached_prefix_projection_key != cache_key or self._cached_prefix_key is None:
            pk, pv = self._compute_nope_prefix_kv_vllm(dtype)
            self._cached_prefix_key = pk.detach()
            self._cached_prefix_value = pv.detach()
            self._cached_prefix_projection_key = cache_key
            self._prefix_projection_cache_misses += 1
        return self._cached_prefix_key, self._cached_prefix_value

    @torch.no_grad()
    def refresh_residual_attention_prefix_cache(self) -> None:
        dtype = self.qkv_proj.weight.dtype
        prefix_key, prefix_value = self._compute_prefix_key_value(dtype)
        same_layout = (
            self._cached_prefix_key is not None
            and self._cached_prefix_key.shape == prefix_key.shape
            and self._cached_prefix_key.dtype == prefix_key.dtype
            and self._cached_prefix_key.device == prefix_key.device
        )
        if same_layout:
            self._cached_prefix_key.copy_(prefix_key)
            self._cached_prefix_value.copy_(prefix_value)
        else:
            self._cached_prefix_key = prefix_key.detach().contiguous()
            self._cached_prefix_value = prefix_value.detach().contiguous()
        if self.joint_prefix and self._joint_offsets is None:
            self._joint_offsets = torch.arange(2, device=prefix_key.device, dtype=torch.int32)
            self._joint_key_offsets = self._joint_offsets * self.num_prefix_tokens
        self._cached_prefix_projection_key = (
            self.prefix_key_tokens._version,
            self.prefix_value_tokens._version,
            self.prefix_key_tokens.device,
            dtype,
        )
        self._prefix_cache_refreshed = True
        self._prefix_projection_cache_misses += 1

    def _split_qkv(self, qkv):
        return qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = self._split_qkv(qkv)
        q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(-1, self.q_size)
        pre_rope_q = (
            q.view(-1, self.num_heads, self.head_dim).clone()
            if (self.nope_token_joint or self.nope_token_null_additive)
            else None
        )
        k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(-1, self.kv_size)
        q, k = self.rotary_emb(positions, q, k)
        if self.nope_token_null_additive:
            attention = self.attn(q, k, v)
            pk, pv = self._cached_prefix_key, self._cached_prefix_value
            # pre_rope_q: [T, H, D] -> [1, H, T, D]
            q_bh = pre_rope_q.transpose(0, 1).unsqueeze(0)
            total_delta = None
            offset = 0
            for bi, p_len in enumerate(self.prefix_branch_sizes):
                pk_b = pk[:, :, offset:offset + p_len, :]
                pv_b = pv[:, :, offset:offset + p_len, :]
                offset += p_len
                bias_b = self.prefix_logit_bias[bi] if self.num_prefix_branches > 1 else self.prefix_logit_bias
                null_w_b = self.prefix_null_proj[bi] if self.num_prefix_branches > 1 else self.prefix_null_proj
                null_b = F.linear(hidden_states.float(), null_w_b.float()).transpose(0, 1).unsqueeze(0).unsqueeze(-1)
                delta_b, _, _ = token_null_attention(
                    q_bh, pk_b, pv_b, bias_b, null_b, scale=self.scaling
                )
                total_delta = delta_b if total_delta is None else (total_delta + delta_b)
            residual = total_delta.squeeze(0).transpose(0, 1).reshape(-1, self.q_size).to(attention.dtype)
            output, _ = self.o_proj(attention + residual)
            return output
        if self.nope_token_joint:
            self._pre_rope_query = pre_rope_q
            self._null_projection = F.linear(hidden_states.float(), self.prefix_null_proj.float())
            try:
                attention = self.attn(q, k, v)
            finally:
                self._pre_rope_query = None
                self._null_projection = None
            output, _ = self.o_proj(attention)
            return output
        if self.joint_prefix:
            attention = self.attn(q, k, v)
            output, _ = self.o_proj(attention)
            return output
        # Native vLLM owns request scheduling and the paged causal KV cache.
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
        joint_prefix = bool(getattr(config, "residual_attention_prefix_joint_softmax", False))
        if joint_prefix and not vllm_config.model_config.disable_cascade_attn:
            raise ValueError("Set disable_cascade_attn=True for joint prefix rollout")
        rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("This trained prefix uses default RoPE; scaled RoPE is unsupported")
        original = qwen3.Qwen3Attention
        qwen3.Qwen3Attention = partial(
            Qwen3ResidualKVPrefixAttention,
            hf_config=config,
            num_prefix_tokens=count,
            key_init_std=getattr(config, "residual_attention_prefix_key_init_std", 0.02))
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            qwen3.Qwen3Attention = original

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        expected = {name for name, _ in self.named_parameters()
                    if name.endswith(PREFIX_SUFFIXES)}
        missing = expected.difference(loaded)
        if missing:
            raise RuntimeError("vLLM did not load all Qwen3 prefix tensors: " + ", ".join(sorted(missing)))
        for module in self.modules():
            if isinstance(module, Qwen3ResidualKVPrefixAttention):
                module._cached_prefix_projection_key = None
                module._cached_prefix_key = None
                module._cached_prefix_value = None
                module._prefix_cache_refreshed = False
                if module.joint_prefix:
                    module.refresh_residual_attention_prefix_cache()
                elif module.nope_token_null_additive:
                    module._get_nope_prefix_kv_vllm(module.qkv_proj.weight.dtype)
        return loaded
