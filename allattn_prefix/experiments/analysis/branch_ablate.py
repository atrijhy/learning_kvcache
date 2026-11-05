"""How much does each prefix branch of a TRAINED AllAttn model change the output? KL(base || model) on one rollout with:
full prefix / GDN branch only / attention (A64) only. CPU, fp32, ~5 min.
Usage: branch_ablate.py MODEL_DIR ROLLOUT_JSONL   (ROLLOUT_JSONL: any $WORK/rollouts/<run>/<step>.jsonl)"""
import json, sys, torch
M = sys.argv[1]; T = 2048
torch.set_num_threads(64)
from transformers.models.qwen3_5 import modeling_qwen3_5 as q35
q35.FusedRMSNormGated = None; q35.causal_conv1d_fn = None; q35.causal_conv1d_update = None
q35.chunk_gated_delta_rule = None; q35.fused_recurrent_gated_delta_rule = None
from transformers import AutoModelForImageTextToText, AutoTokenizer
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForImageTextToText.from_pretrained(M, trust_remote_code=True, dtype=torch.float32).eval()
for n, L in model.named_modules():
    if type(L).__name__.endswith("GatedDeltaNet"):
        L.causal_conv1d_fn = None; L.chunk_gated_delta_rule = q35.torch_chunk_gated_delta_rule; L.recurrent_gated_delta_rule = q35.torch_recurrent_gated_delta_rule
R = [json.loads(l) for l in open(sys.argv[2])]
r = sorted(R, key=lambda x: len(x["output"]))[len(R) // 2]
pid = tok(r["input"], add_special_tokens=False).input_ids; rid = tok(r["output"], add_special_tokens=False).input_ids
P = min(len(pid), T // 2); ids = pid[-P:] + rid[:T - P]; resp = slice(P, len(ids) - 1)
x = torch.tensor([ids])
V = {n: p for n, p in model.named_parameters() if n.endswith("prefix_value_tokens")}
saved = {n: p.detach().clone() for n, p in V.items()}
gdn = [n for n in V if "linear_attn" in n]; att = [n for n in V if "linear_attn" not in n]
def lp(keep):
    with torch.no_grad():
        for n in V: V[n].copy_(saved[n] if n in keep else torch.zeros_like(saved[n]))
        return torch.log_softmax(model(input_ids=x).logits[0][resp], -1)
base = lp([])
for name, keep in [("full prefix", gdn + att), ("GDN branch only", gdn), ("attention A64 only", att)]:
    l = lp(keep); kl = (base.exp() * (base - l)).sum(-1)
    flip = (l.argmax(-1) != base.argmax(-1)).float().mean()
    print(f"{M.split('/')[-1]:45s} {name:20s} KL(base||model) mean {kl.mean():.4f} p95 {kl.quantile(.95):.4f} | argmax changed {100*flip:.1f}%", flush=True)
