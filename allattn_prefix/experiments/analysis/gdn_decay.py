"""How fast does a GDN layer forget its initial recurrent state (where the G512 delta prefix lives)?
For a real TACO rollout (student prompt + response), hook every Gated DeltaNet layer of base Qwen3.5-4B, compute the
per-token gate g_t = -exp(A_log) * softplus(a_t + dt_bias) and the surviving fraction of the initial state after t
tokens, prod_{i<=t} exp(g_i) (the delta-rule erasure (I - beta k k^T) only removes more, so this is an upper bound).
Note the prefix state is built BEFORE the prompt, so the response starts ~prompt_len tokens in."""
import json, sys, torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer
M = "/root/kvmodels/Qwen3.5-4B"; dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; T = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
torch.set_num_threads(32)
tok = AutoTokenizer.from_pretrained(M)
from transformers.models.qwen3_5 import modeling_qwen3_5 as _q
if dev == "cpu":  # force the pure-torch GDN path (fla / causal_conv1d kernels are CUDA/Triton only)
    _q.FusedRMSNormGated = None; _q.causal_conv1d_fn = None; _q.causal_conv1d_update = None
    _q.chunk_gated_delta_rule = None; _q.fused_recurrent_gated_delta_rule = None
model = AutoModelForImageTextToText.from_pretrained(M, torch_dtype=torch.bfloat16 if dev != "cpu" else torch.float32).to(dev).eval()
import pandas as pd
rows = pd.read_parquet("/root/local_runs/ovf/data/taco_ab/sources/taco/train.parquet")
R = [json.loads(l) for l in open("/root/local_runs/ovf/rollouts/ovf_opsd_taco/1.jsonl")]
r = max(R, key=lambda x: len(x["output"])); t = int(r["env_task_idx"])
p = [dict(m) for m in rows[rows.task_idx == t].iloc[0].prompt]
ptxt = tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
pid = tok(ptxt, add_special_tokens=False).input_ids; rid = tok(r["output"], add_special_tokens=False).input_ids
ids = (pid + rid)[:T]; print(f"task {t}: prompt {len(pid)} tok, response {len(rid)} tok, using {len(ids)}", flush=True)
gates = {}
layers = [m for n, m in model.named_modules() if type(m).__name__.endswith("GatedDeltaNet")]
def hook(mod, inp, out, idx=None):
    a = out.float(); g = -mod_ref[idx].A_log.float().exp() * F.softplus(a + mod_ref[idx].dt_bias.float())
    gates[idx] = g[0].cpu()                                   # (T, heads)
mod_ref = {}
from transformers.models.qwen3_5 import modeling_qwen3_5 as q35
for L in layers:
    if dev == 'cpu':
        L.causal_conv1d_fn = None; L.chunk_gated_delta_rule = q35.torch_chunk_gated_delta_rule; L.recurrent_gated_delta_rule = q35.torch_recurrent_gated_delta_rule
for i, L in enumerate(layers):
    mod_ref[i] = L; L.in_proj_a.register_forward_hook(lambda m, inp, out, i=i: hook(m, inp, out, i))
with torch.no_grad():
    model.model(input_ids=torch.tensor(ids, device=dev)[None])
marks = [p for p in (128, 512, 1024, 2048, 4096, 8192) if p <= len(ids)]
print("surviving fraction of the initial (prefix) state after t tokens, per GDN layer: median over heads [max head]")
allmed = {m: [] for m in marks}
for i in sorted(gates):
    cs = gates[i].cumsum(0)                                   # (T, H) log survival
    row = []
    for m in marks:
        s = cs[m - 1].exp(); allmed[m].append(s.median().item()); row.append(f"t={m}: {s.median():.2e} [{s.max():.2e}]")
    print(f"GDN layer {i:2d}: " + "  ".join(row), flush=True)
print("ALL layers median: " + "  ".join(f"t={m}: {torch.tensor(allmed[m]).median():.2e}" for m in marks))
print(f"prompt length here = {len(pid)} tokens: the response's FIRST token already sits at t={len(pid)}")
