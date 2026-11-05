"""HF vs vLLM consistency for Qwen3_5AllAttentionPrefixForConditionalGeneration on a PERTURBED prefix (value tokens
N(0, 0.05), so every layer's branch is active): prompt log-probs of the same text under (a) HF, (b) vLLM, (c) vLLM base.
Pass if |vLLM - HF| is at bf16-noise level and much smaller than |vLLM base - HF|. Run with OPSD_PLUGIN=1 and
/root/opsd_plugin on PYTHONPATH. Usage: CUDA_VISIBLE_DEVICES=G verify_vllm.py"""
import os, shutil, json, torch


def main():
    from safetensors.torch import load_file, save_file
    SRC = "/root/kvmodels/Qwen3.5-4B-AllAttn-A512"; T = "/root/kvmodels/allattn_verify_tmp"
    if not os.path.exists(T):
        shutil.copytree(SRC, T, ignore=shutil.ignore_patterns("residual_attention_prefix.safetensors"),
                        copy_function=lambda s, d: os.link(s, d) if os.path.basename(s).startswith("model.safetensors-") else shutil.copy2(s, d))
        st = load_file(f"{SRC}/residual_attention_prefix.safetensors"); g = torch.Generator().manual_seed(0)
        st = {k: ((torch.randn(v.shape, generator=g) * 0.05).to(v.dtype) if k.endswith("prefix_value_tokens") else v) for k, v in st.items()}
        save_file(st, f"{T}/residual_attention_prefix.safetensors", metadata={"format": "pt"})
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(T)
    text = ("You are given an array of n integers. Find the length of the longest strictly increasing subsequence. "
            "Explain the O(n log n) patience-sorting idea, then write a Python function that reads n and the array from "
            "stdin and prints the answer. Make sure to handle n = 1 and arrays with repeated values correctly.") * 3
    ids = tok(text, add_special_tokens=False).input_ids
    hf = AutoModelForImageTextToText.from_pretrained(T, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        lp = torch.log_softmax(hf(input_ids=torch.tensor([ids]).cuda()).logits[0].float(), -1)
    hf_lp = [lp[i - 1, ids[i]].item() for i in range(1, len(ids))]
    del hf; torch.cuda.empty_cache()
    from vllm import LLM, SamplingParams
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    def vlp(model):
        llm = LLM(model, gpu_memory_utilization=0.3, max_model_len=4096, enforce_eager=True, trust_remote_code=True)
        o = llm.generate({"prompt_token_ids": ids}, sp)[0].prompt_logprobs
        out = [o[i][ids[i]].logprob for i in range(1, len(ids))]
        del llm; torch.cuda.empty_cache(); return out
    v = vlp(T)
    import statistics as S
    d = [abs(a - b) for a, b in zip(v, hf_lp)]
    print(f"tokens {len(ids)}: |vLLM-HF| mean {S.mean(d):.4f} max {max(d):.4f}", flush=True)
    vb = vlp("/root/kvmodels/Qwen3.5-4B")
    db = [abs(a - b) for a, b in zip(vb, hf_lp)]
    print(f"|vLLM base - HF prefix| mean {S.mean(db):.4f} max {max(db):.4f}")
    print("PASS" if S.mean(d) < 0.25 * S.mean(db) else "FAIL")


if __name__ == "__main__":
    main()
