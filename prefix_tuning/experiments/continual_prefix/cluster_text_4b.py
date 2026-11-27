"""Route A (Continual Learning Clustering via Pure Qwen3-4B Verbatim Self-Labeling):
- Direct 150-250 verbatim lesson discovery (domain-agnostic mechanisms, autonomous 1 <= K <= max_k).
- Explicit OTHER rejection option + single-letter token logprob margin OOD guard.
- Supports initial discovery, online incremental routing, and recursive OTHER buffer incubation.
"""
import argparse
import gc
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def extract_json(text: str):
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip() or text
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", clean, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))
    s, e = clean.find("{"), clean.rfind("}")
    if s >= 0 and e > s:
        return json.loads(clean[s : e + 1])
    raise ValueError(f"Unparsable JSON: {clean[:300]}")


def discover_categories(llm, tok, rows: list[dict], prefix: str = "C", max_k: int = 5, sample_n: int = 200, seed: int = 42, context_hint: str = "") -> list[dict]:
    """Let Qwen3-4B autonomously choose K (1 <= K <= max_k) from up to 200-250 verbatim lessons."""
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    sample = pool[: min(sample_n, len(pool))]
    body = "\n".join(f"[{i+1}] {r['experience'][:240].replace(chr(10), ' ')}" for i, r in enumerate(sample))

    msgs = [
        {
            "role": "system",
            "content": (
                "You are an autonomous self-evolving AI agent discovering clean, mutually-exclusive categories "
                "directly from raw failure lessons. Categories must be domain-agnostic failure mechanisms with "
                "STRICT positive admission rules so unrelated modalities or novel error types are rejected as OTHER."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{context_hint}Below are {len(sample)} verbatim failure lessons:\n{body}\n\n"
                f"Choose the natural number of categories K yourself (1 <= K <= {max_k}, NO forced minimum — only split when patterns are clearly distinct!).\n"
                "Rules:\n"
                "1. Abstract away specific story/entity/tool names (NEVER name categories after specific apps, hospitals, drugs, or IDs). Describe ONLY reusable failure mechanisms for this task modality (e.g., tool-parameter misuse, missing verification step, input format violation, syntax error, algorithmic logic bug).\n"
                "2. Specify the general task modality AND the general failure mechanism in each admission rule so genuinely different modalities or unseen failure mechanisms are rejected as OTHER.\n"
                "3. Do NOT create vague catch-all categories like 'General Error' or 'Other Failures' — unrelated/novel tasks MUST be left for OTHER.\n"
                "4. Output STRICT valid JSON:\n"
                "{\n"
                '  "categories": [\n'
                "    {\n"
                f'      "id": "{prefix}1",\n'
                '      "name": "<short English name>",\n'
                '      "definition": "<1-sentence concrete definition>",\n'
                '      "admission_rule": "<ONLY assign this category if ...>"\n'
                "    }\n"
                "  ]\n"
                "}\n"
            ),
        },
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=4096))[0].outputs[0].text
    try:
        cats = extract_json(out)["categories"][:max_k]
    except Exception:
        p_nt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out2 = llm.generate([p_nt], SamplingParams(temperature=0.0, max_tokens=2048))[0].outputs[0].text
        cats = extract_json(out2)["categories"][:max_k]
    for i, c in enumerate(cats):
        c["id"] = f"{prefix}{i+1}"
    return cats


def classify_with_other_and_logprobs(llm, tok, cats: list[dict], rows: list[dict], margin_threshold: float = 1.5, enable_thinking: bool = False):
    """Classify each item into {C1..Ck, OTHER} with explicit OTHER + logprob Top1-Top2 margin guard."""
    vids = [c["id"] for c in cats] + ["OTHER"]
    opt_letters = [chr(ord("A") + i) for i in range(len(cats))] + ["O"]
    letter_to_id = {l: vid for l, vid in zip(opt_letters, vids, strict=True)}
    cat_lines = "\n".join(
        f"- {l} ({c['id']} - {c['name']}): {c['definition']} Rule: {c.get('admission_rule', '')}"
        for l, c in zip(opt_letters[:-1], cats, strict=True)
    )
    cat_lines += "\n- O (OTHER / UNMATCHED): Does NOT clearly satisfy ANY admission rule above (different task modality or novel unseen failure mechanism)."

    sys_msg = (
        "You are a strict classifier with an explicit rejection option O (OTHER). "
        "ONLY pick A..Z if the lesson CLEARLY matches BOTH the task modality AND failure mechanism of that category. "
        "Otherwise output O. "
        f"Respond with ONLY a single letter from: {', '.join(opt_letters)}."
    )
    prompts = []
    for r in rows:
        ctx_snip = f"Tools: {str(r.get('tool_names', ''))[:120]}\n" if r.get("tool_names") else ""
        usr_msg = f"Categories:\n{cat_lines}\n\n{ctx_snip}Lesson:\n{r['experience'][:450]}\n\nAnswer (single letter ONLY):"
        prompts.append(
            tok.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=1024 if enable_thinking else 4,
        logprobs=5 if not enable_thinking else None,
    )
    gens = llm.generate(prompts, sp)
    valid_set = set(opt_letters)
    raw_preds, final_preds, margins = [], [], []
    for g in gens:
        o = g.outputs[0]
        raw = re.sub(r"<think>.*?</think>", "", o.text, flags=re.DOTALL).strip()
        chosen = "O"
        for ch in raw.upper():
            if ch in valid_set:
                chosen = ch
                break
        cid = letter_to_id[chosen]
        margin = 10.0
        if not enable_thinking and o.logprobs:
            for tid, lpd in zip(o.token_ids, o.logprobs):
                piece = tok.decode([tid]).strip().upper()
                if piece in valid_set and lpd:
                    lps = sorted((v.logprob for v in lpd.values()), reverse=True)
                    margin = (lps[0] - lps[1]) if len(lps) >= 2 else 10.0
                    break
        raw_preds.append(cid)
        final_preds.append("OTHER" if (cid == "OTHER" or (not enable_thinking and margin < margin_threshold)) else cid)
        margins.append(round(float(margin), 4))
    return raw_preds, final_preds, margins


def main():
    ap = argparse.ArgumentParser(description="Route A: Pure Qwen3-4B verbatim self-labeling with OTHER + logprob OOD guard")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--taxonomy-json", type=Path, default=None, help="Existing taxonomy JSON for incremental routing mode")
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--sample-n", type=int, default=200)
    ap.add_argument("--margin-threshold", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.input_jsonl.open() if l.strip()]
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.85, enable_prefix_caching=False)

    if args.taxonomy_json and args.taxonomy_json.exists():
        cats = json.loads(args.taxonomy_json.read_text())["categories"]
    else:
        cats = discover_categories(llm, tok, rows, max_k=args.max_k, sample_n=args.sample_n, seed=args.seed)

    raw_preds, final_preds, margins = classify_with_other_and_logprobs(
        llm, tok, cats, rows, margin_threshold=args.margin_threshold, enable_thinking=False
    )
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    out = {
        "categories": cats,
        "margin_threshold": args.margin_threshold,
        "counts_raw": dict(Counter(raw_preds)),
        "counts_guarded": dict(Counter(final_preds)),
        "assignments": [
            {"id": r.get("id", str(i)), "raw_label": rp, "label": fp, "margin": mg}
            for i, (r, rp, fp, mg) in enumerate(zip(rows, raw_preds, final_preds, margins, strict=True))
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved K={len(cats)} counts={out['counts_guarded']} -> {args.out_json}")


if __name__ == "__main__":
    main()
