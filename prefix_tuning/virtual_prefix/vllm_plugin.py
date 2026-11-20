"""Process-safe vLLM general-plugin registration."""

from vllm import ModelRegistry


INDEPENDENT_DELTA_KV_PREFIX_ARCHITECTURE = (
    "Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration"
)
INDEPENDENT_DELTA_KV_PREFIX_MODEL_CLASS = (
    "prefix_tuning.virtual_prefix.vllm_independent_delta_kv_prefix_model:"
    "Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration"
)
HYBRID_DELTA_RESIDUAL_ATTENTION_PREFIX_ARCHITECTURE = (
    "Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration"
)
HYBRID_DELTA_RESIDUAL_ATTENTION_PREFIX_MODEL_CLASS = (
    "prefix_tuning.virtual_prefix."
    "vllm_hybrid_delta_residual_attention_prefix_model:"
    "Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration"
)
QWEN3_RESIDUAL_KV_PREFIX_ARCHITECTURE = "Qwen3ResidualKVPrefixForCausalLM"
QWEN3_RESIDUAL_KV_PREFIX_MODEL_CLASS = (
    "prefix_tuning.virtual_prefix.vllm_qwen3_residual_kv_prefix_model:"
    "Qwen3ResidualKVPrefixForCausalLM"
)
def register() -> None:
    ModelRegistry.register_model(
        QWEN3_RESIDUAL_KV_PREFIX_ARCHITECTURE,
        QWEN3_RESIDUAL_KV_PREFIX_MODEL_CLASS,
    )
    ModelRegistry.register_model(
        INDEPENDENT_DELTA_KV_PREFIX_ARCHITECTURE,
        INDEPENDENT_DELTA_KV_PREFIX_MODEL_CLASS,
    )
    ModelRegistry.register_model(
        HYBRID_DELTA_RESIDUAL_ATTENTION_PREFIX_ARCHITECTURE,
        HYBRID_DELTA_RESIDUAL_ATTENTION_PREFIX_MODEL_CLASS,
    )
