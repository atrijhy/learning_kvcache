"""Zero-Retraining Multi-Branch Additive Composition (o_t = d_t + sum_k p_t^(k))
+ Pairwise Gate Overlap Monitoring Overlap(a, b) for Redundant Branch Merging."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from prefix_tuning.virtual_prefix.modeling_qwen3_residual_kv_prefix import (
    install_residual_kv_prefix,
    load_prefix_state_dict,
)


def compose_multibranch_additive(ckpt_dirs: list[Path], out_dir: Path) -> Path:
    """Exact zero-retraining multi-branch additive composition:
    - prefix_key_tokens / prefix_value_tokens: concatenated along dim=0 -> [sum(P_k), H]
    - prefix_logit_bias: stacked along dim=0 -> [K, num_heads]
    - prefix_null_proj: stacked along dim=0 -> [K, num_heads, H]
    Supports heterogeneous branch_sizes = [P_1, ..., P_K].
    """
    sts = [load_file(str(d / "model.safetensors"), device="cpu") for d in ckpt_dirs]
    cfgs = [json.loads((d / "prefix_config.json").read_text()) for d in ckpt_dirs]
    branch_sizes = [int(c["num_virtual_tokens"]) for c in cfgs]
    out = {}
    for k in sorted(sts[0]):
        ts = [s[k].float() for s in sts]
        if k.endswith(("prefix_key_tokens", "prefix_value_tokens")):
            out[k] = torch.cat(ts, dim=0).contiguous()
        elif k.endswith(("prefix_logit_bias", "prefix_null_proj")):
            out[k] = torch.stack(ts, dim=0).contiguous() if len(ts) > 1 else ts[0].contiguous()
        else:
            raise ValueError(f"Unexpected prefix tensor: {k}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    cfg = dict(cfgs[0])
    cfg["method"] = "qwen3_nope_token_null_additive"
    cfg["attention_mode"] = "nope_token_null_additive"
    cfg["num_virtual_tokens"] = sum(branch_sizes)
    cfg["branch_sizes"] = branch_sizes
    cfg["trainable_numel"] = sum(t.numel() for t in out.values())
    cfg["composed_from"] = [str(d) for d in ckpt_dirs]
    (out_dir / "prefix_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return out_dir


@torch.no_grad()
def measure_gate_overlap(base_model: Path, composed_dir: Path, val_pt: Path, gate_threshold: float = 0.5) -> dict:
    """Measure Val NLL, per-layer active branch counts (0 / 1 / >=2), and pairwise Jaccard Overlap(a, b)."""
    cfg = json.loads((composed_dir / "prefix_config.json").read_text())
    P = int(cfg["num_virtual_tokens"])
    bsizes = cfg.get("branch_sizes", [P])
    K = len(bsizes)

    val_rows = torch.load(val_pt, map_location="cpu", weights_only=False)["validation"]
    model = AutoModelForCausalLM.from_pretrained(
        base_model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    install_residual_kv_prefix(model, num_virtual_tokens=P, mode="nope_token_null_additive", branch_sizes=bsizes)
    load_prefix_state_dict(model, load_file(str(composed_dir / "model.safetensors"), device="cpu"))

    n_layers = model.config.num_hidden_layers
    tok_counts = torch.zeros(n_layers, 3, dtype=torch.int64, device="cuda")
    pair_inter = torch.zeros(K, K, dtype=torch.int64, device="cuda")
    pair_union = torch.zeros(K, K, dtype=torch.int64, device="cuda")
    lam_sum = torch.zeros(n_layers, dtype=torch.float64, device="cuda")
    rat_sum = torch.zeros(n_layers, dtype=torch.float64, device="cuda")
    n_elems = torch.zeros(n_layers, dtype=torch.float64, device="cuda")
    bufs = [[] for _ in range(n_layers)]

    for li, layer in enumerate(model.model.layers):
        def make_mon(idx):
            def mon(lam, delta, native):
                lam_sum[idx] += lam.sum(dtype=torch.float64)
                rat_sum[idx] += (delta.float().norm(dim=-1) / (native.float().norm(dim=-1) + 1e-8)).sum(dtype=torch.float64)
                n_elems[idx] += lam.numel()
                buf = bufs[idx]
                buf.append(lam.squeeze(0).squeeze(-1))  # [H, T]
                if len(buf) == K:
                    stacked = torch.stack(buf, dim=0)  # [K, H, T]
                    tok_open = stacked.mean(dim=1) > gate_threshold  # [K, T]
                    n_act = tok_open.sum(dim=0)
                    tok_counts[idx, 0] += (n_act == 0).sum()
                    tok_counts[idx, 1] += (n_act == 1).sum()
                    tok_counts[idx, 2] += (n_act >= 2).sum()
                    if K >= 2:
                        for a in range(K):
                            for b in range(a + 1, K):
                                pair_inter[a, b] += (tok_open[a] & tok_open[b]).sum()
                                pair_union[a, b] += (tok_open[a] | tok_open[b]).sum()
                    buf.clear()
            return mon
        layer.self_attn.prefix_monitor = make_mon(li)

    tot_loss, tot_toks = 0.0, 0
    with torch.inference_mode():
        for row in val_rows:
            ids, pos = row[1], row[3]
            x = torch.tensor([ids], device="cuda")
            p_t = torch.tensor(pos, device="cuda")
            y = x[0, p_t]
            h = model.model(input_ids=x, use_cache=False).last_hidden_state[0, p_t - 1]
            tot_loss += sum(F.cross_entropy(model.lm_head(hc).float(), yc, reduction="sum").item()
                            for hc, yc in zip(h.split(256), y.split(256), strict=True))
            tot_toks += len(pos)

    tok_tot = tok_counts.sum(dim=0).float()
    tok_tot = (tok_tot / tok_tot.sum().clamp_min(1)).cpu().tolist()
    pairwise_jaccard = {}
    for a in range(K):
        for b in range(a + 1, K):
            u = pair_union[a, b].item()
            pairwise_jaccard[f"{a}-{b}"] = round(pair_inter[a, b].item() / max(1, u), 5)

    return {
        "composed_dir": str(composed_dir),
        "K": K,
        "branch_sizes": bsizes,
        "val_nll": round(tot_loss / max(1, tot_toks), 5),
        "mean_gate_lam": round((lam_sum / n_elems.clamp_min(1)).mean().item(), 6),
        "mean_p_over_d": round((rat_sum / n_elems.clamp_min(1)).mean().item(), 6),
        "token_gt_thr_frac": {"zero": round(tok_tot[0], 5), "one": round(tok_tot[1], 5), "ge2": round(tok_tot[2], 5)},
        "pairwise_jaccard_overlap": pairwise_jaccard,
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_comp = sub.add_parser("compose")
    p_comp.add_argument("--out-dir", type=Path, required=True)
    p_comp.add_argument("ckpts", nargs="+", type=Path)

    p_ov = sub.add_parser("overlap")
    p_ov.add_argument("--base-model", type=Path, required=True)
    p_ov.add_argument("--composed-dir", type=Path, required=True)
    p_ov.add_argument("--val-pt", type=Path, required=True)
    p_ov.add_argument("--gate-threshold", type=float, default=0.5)
    p_ov.add_argument("--out-json", type=Path, default=None)

    args = ap.parse_args()
    if args.cmd == "compose":
        out = compose_multibranch_additive(args.ckpts, args.out_dir)
        print(f"Composed K={len(args.ckpts)} -> {out}")
    else:
        res = measure_gate_overlap(args.base_model, args.composed_dir, args.val_pt, args.gate_threshold)
        print(json.dumps(res, indent=2))
        if args.out_json:
            args.out_json.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
