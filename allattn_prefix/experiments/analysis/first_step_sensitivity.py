"""Why does the new structure collapse after its FIRST update? Adam's first step moves every trainable element by
~lr * sign(grad) (m/sqrt(v) = +-1 at step 1), independent of the gradient size. Simulate that step on CPU and measure
how far the model's next-token distribution moves on a real EnvScaler rollout (step-1 rollout of the formal run).
Gradient direction: student NLL of its own response (proxy for the KL direction; also random signs as control).
Variants: GDN-branch value tokens only / attention-layer (A64) value tokens only, at several lr."""
import json, sys, torch, glob
import torch.nn.functional as F
M = sys.argv[1]; T = 2048   # Usage: first_step_sensitivity.py INIT_MODEL_DIR ROLLOUT_JSONL
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
print(f"prompt {len(pid)} tok (using last {P}), response {len(rid)} tok, total {len(ids)}", flush=True)
x = torch.tensor([ids])
V = {n: p for n, p in model.named_parameters() if n.endswith("prefix_value_tokens")}
gdn = [n for n in V if "linear_attn" in n]; att = [n for n in V if "self_attn" in n]
print(f"value tensors: GDN {len(gdn)} x {tuple(V[gdn[0]].shape)}, attention {len(att)} x {tuple(V[att[0]].shape)}", flush=True)
for p in model.parameters(): p.requires_grad_(False)
for p in V.values(): p.requires_grad_(True)
logits = model(input_ids=x).logits[0]
base_lp = torch.log_softmax(logits[resp].detach(), -1)
nll = F.cross_entropy(logits[resp], x[0, resp.start + 1:resp.stop + 1]); nll.backward()
G = {n: V[n].grad.detach().clone() for n in V}; del logits
print(f"NLL {nll.item():.3f}; grad norms GDN {sum(G[n].norm()**2 for n in gdn)**.5:.3e}  attn {sum(G[n].norm()**2 for n in att)**.5:.3e}", flush=True)
for p in V.values(): p.requires_grad_(False)
gen = torch.Generator().manual_seed(0)
def run(names, lr, how):
    with torch.no_grad():
        for n in V: V[n].zero_()
        for n in names:
            s = torch.sign(G[n]) if how == "grad" else torch.randint(0, 2, G[n].shape, generator=gen).float() * 2 - 1
            V[n].copy_(-lr * s)
        lp = torch.log_softmax(model(input_ids=x).logits[0][resp], -1)
        kl = (base_lp.exp() * (base_lp - lp)).sum(-1)
        flip = (lp.argmax(-1) != base_lp.argmax(-1)).float().mean()
    print(f"{how:6s} {('GDN' if names is gdn else 'attn' if names is att else 'all'):5s} lr {lr:.0e}: KL(base||new) mean {kl.mean():.4f} p95 {kl.quantile(.95):.4f} max {kl.max():.3f} | argmax changed {100*flip:.1f}%", flush=True)
for lr in (1e-5,):
    run(gdn, lr, "grad")
run(gdn + att, 1e-5, "grad")
