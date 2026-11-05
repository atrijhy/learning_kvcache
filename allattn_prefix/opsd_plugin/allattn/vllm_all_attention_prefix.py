"""vLLM side of Qwen3_5AllAttentionPrefixForConditionalGeneration (see modeling_qwen3_5_all_attention_prefix.py):
residual full-attention soft prefix on every layer. Attention layers reuse the hybrid plugin's
Qwen3_5ResidualSoftPrefixAttention; Gated DeltaNet layers add softmax(q K_p^T / sqrt(d_k)) V_p to the core output
before the gated RMSNorm, with q = the layer's pre-conv query projection (same as HF). Prefix K/V are recomputed in
every forward from the soft tokens and the frozen in_proj_qkvz slices, so weight syncs and CUDA graphs need no cache
refresh. TP=1 only (the OPSD launcher's setting)."""
import torch
import torch.nn.functional as F
from einops import rearrange
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import qwen3_5
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.utils import set_weight_attrs
from prefix_tuning.virtual_prefix.vllm_hybrid_delta_residual_attention_prefix_model import (
    Qwen3_5ResidualSoftPrefixAttention,
)

ARCH = "Qwen3_5AllAttentionPrefixForConditionalGeneration"


class Qwen3_5ResidualSoftPrefixGatedDeltaNet(qwen3_5.Qwen3_5GatedDeltaNet):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if self.tp_size != 1:
            raise NotImplementedError("residual GDN prefix supports TP=1 only")
        self.num_prefix_tokens = int(getattr(config, "residual_gdn_prefix_num_virtual_tokens", 0)
                                     or getattr(config, "residual_attention_prefix_num_virtual_tokens", 0))
        if self.num_prefix_tokens <= 0:
            raise ValueError("residual_attention_prefix_num_virtual_tokens must be positive")
        hidden = config.hidden_size
        self.prefix_key_tokens = torch.nn.Parameter(torch.empty(self.num_prefix_tokens, hidden))
        self.prefix_value_tokens = torch.nn.Parameter(torch.zeros(self.num_prefix_tokens, hidden))
        torch.nn.init.normal_(self.prefix_key_tokens, mean=0.0, std=0.02)
        for p in (self.prefix_key_tokens, self.prefix_value_tokens):
            set_weight_attrs(p, {"weight_loader": default_weight_loader})

    def _prefix_branch(self, mixed_qkv: torch.Tensor) -> torch.Tensor:
        kd, vd = self.key_dim, self.value_dim
        hk, hv, dk, dv = self.num_k_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim
        rep = hv // hk
        dtype = mixed_qkv.dtype
        w = self.in_proj_qkvz.weight                     # rows: q | k | v | z
        key = F.linear(self.prefix_key_tokens.to(dtype), w[kd:2 * kd]).view(-1, hk, dk)
        value = F.linear(self.prefix_value_tokens.to(dtype), w[2 * kd:2 * kd + vd]).view(-1, hv, dv)
        query = mixed_qkv[:, :kd].reshape(-1, hk, dk)
        if rep > 1:
            query = query.repeat_interleave(rep, dim=1)
            key = key.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0), key.transpose(0, 1).unsqueeze(0), value.transpose(0, 1).unsqueeze(0),
            dropout_p=0.0, is_causal=False, scale=dk ** -0.5)
        return out.squeeze(0).transpose(0, 1)            # (T, Hv, dv)

    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor):
        num_tokens = hidden_states.size(0)
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        ba, _ = self.in_proj_ba(hidden_states)
        b, a = ba.chunk(2, dim=-1)
        b = b.contiguous()
        a = a.contiguous()
        branch = self._prefix_branch(mixed_qkv)          # computed before the core op (which may consume mixed_qkv)
        core_attn_out = torch.zeros((num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                                    dtype=hidden_states.dtype, device=hidden_states.device)
        torch.ops.vllm.gdn_attention_core(mixed_qkv, b, a, core_attn_out, self.prefix)
        core_attn_out = core_attn_out + branch.to(core_attn_out.dtype)
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)


class Qwen3_5AllAttentionPrefixForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = "model"):
        orig_attn, orig_gdn = qwen3_5.Qwen3NextAttention, qwen3_5.Qwen3_5GatedDeltaNet
        qwen3_5.Qwen3NextAttention = Qwen3_5ResidualSoftPrefixAttention
        qwen3_5.Qwen3_5GatedDeltaNet = Qwen3_5ResidualSoftPrefixGatedDeltaNet
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            qwen3_5.Qwen3NextAttention, qwen3_5.Qwen3_5GatedDeltaNet = orig_attn, orig_gdn

    def refresh_residual_attention_prefix_caches(self) -> None:
        for m in self.modules():
            if isinstance(m, Qwen3_5ResidualSoftPrefixAttention):
                m.refresh_residual_attention_prefix_cache()

    def invalidate_residual_attention_prefix_caches(self) -> None:
        for m in self.modules():
            if isinstance(m, Qwen3_5ResidualSoftPrefixAttention):
                m.invalidate_residual_attention_prefix_cache()

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        expected = {n for n, _ in self.named_parameters() if n.endswith((".prefix_key_tokens", ".prefix_value_tokens"))}
        missing = expected.difference(loaded)
        if missing and len(missing) == len(expected):
            raise RuntimeError("no residual prefix tensor was loaded")
        if missing:
            raise RuntimeError(f"residual prefix tensors not loaded: {sorted(missing)[:4]} ... ({len(missing)})")
        self.refresh_residual_attention_prefix_caches()
        return loaded
