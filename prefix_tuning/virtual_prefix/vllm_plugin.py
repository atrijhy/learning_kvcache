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


def _patch_partial_wake_up() -> None:
    """Keep the scheduler paused until every slept tag is awake again.

    vLLM 0.18 ``EngineCore.wake_up`` resumes the scheduler unconditionally.
    verl wakes ``weights`` first, syncs them, then wakes ``kv_cache``; with the
    async colocated trainer, requests queued during the pause get scheduled in
    between and touch the still-unmapped KV-cache pool (illegal memory access).
    """
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore.wake_up, "_partial_wake_safe", False):
        return

    def wake_up(self, tags=None):
        if tags is not None and "scheduling" in tags:
            tags = [t for t in tags if t != "scheduling"]
        if tags is None or tags:
            self.model_executor.wake_up(tags)
        if not self.model_executor.is_sleeping:
            self.resume_scheduler()

    wake_up._partial_wake_safe = True
    EngineCore.wake_up = wake_up


def _register_qwen3_5_config_hooks() -> None:
    """Give the prefix architectures Qwen3.5's config hook.

    vLLM looks up per-architecture config fixes by architecture name. Without
    an entry, the Qwen3.5 hook that sets mamba_ssm_cache_dtype from the HF
    config (mamba_ssm_dtype=float32) is skipped, the Gated DeltaNet recurrent
    state is cached in bfloat16, and decoding drifts with output length.
    """
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        Qwen3_5ForConditionalGenerationConfig,
    )

    for architecture in (
        INDEPENDENT_DELTA_KV_PREFIX_ARCHITECTURE,
        HYBRID_DELTA_RESIDUAL_ATTENTION_PREFIX_ARCHITECTURE,
    ):
        MODELS_CONFIG_MAP.setdefault(architecture, Qwen3_5ForConditionalGenerationConfig)


def register() -> None:
    _patch_partial_wake_up()
    _register_qwen3_5_config_hooks()
    ModelRegistry.register_model(
        INDEPENDENT_DELTA_KV_PREFIX_ARCHITECTURE,
        INDEPENDENT_DELTA_KV_PREFIX_MODEL_CLASS,
    )
    ModelRegistry.register_model(
        HYBRID_DELTA_RESIDUAL_ATTENTION_PREFIX_ARCHITECTURE,
        HYBRID_DELTA_RESIDUAL_ATTENTION_PREFIX_MODEL_CLASS,
    )
