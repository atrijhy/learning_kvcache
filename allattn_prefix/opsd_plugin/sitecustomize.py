# kvcache OPSD experiment plugin (2026-11-04). EFS (and so verl / prefix_tuning source) is read-only on this pod, so
# extras are registered at interpreter start in every process that has /root/opsd_plugin on PYTHONPATH (driver, Ray
# workers, vLLM workers). Only active when OPSD_PLUGIN=1.
#  * opsd_losses: extra distillation losses (forward_kl_topk_focus)
#  * allattn: Qwen3_5AllAttentionPrefixForConditionalGeneration (residual full-attention prefix on all 32 layers) for vLLM
import os
if os.environ.get("OPSD_PLUGIN") == "1":
    import sys
    try:
        from opsd_losses import register
        register()
    except Exception as e:
        print(f"[opsd_plugin] loss registration failed: {e!r}", file=sys.stderr)
    try:
        from vllm import ModelRegistry
        ModelRegistry.register_model("Qwen3_5AllAttentionPrefixForConditionalGeneration",
                                     "allattn.vllm_all_attention_prefix:Qwen3_5AllAttentionPrefixForConditionalGeneration")
        from vllm.model_executor.models.config import MODELS_CONFIG_MAP, Qwen3_5ForConditionalGenerationConfig
        MODELS_CONFIG_MAP.setdefault("Qwen3_5AllAttentionPrefixForConditionalGeneration", Qwen3_5ForConditionalGenerationConfig)
    except Exception as e:
        print(f"[opsd_plugin] vllm registration failed: {e!r}", file=sys.stderr)
