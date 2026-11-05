"""Where does the OPSD signal sit? For TACO OPSD-overfit rollouts (student = base / zero-init prefix, step 1), score each
response token with the base model twice: student context (author retry prompt) and teacher context (same + "\n\n" +
ground_truth), and compute per-token KL(teacher || student) over the full vocab. Reports how concentrated the KL mass is
(share in the top 1% / 5% tokens, share in the first 10% of the response) and the top tokens with context.
Usage: CUDA_VISIBLE_DEVICES=G kl_tokens.py [N_ROLLOUTS]"""
import glob, json, sys
import pandas as pd, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
O = "/root/local_runs/ovf"; M = "/root/kvmodels/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForImageTextToText.from_pretrained(M, torch_dtype=torch.bfloat16).cuda().eval()
rows = pd.read_parquet(f"{O}/data/taco_ab/sources/taco/train.parquet")
prompt = {int(t): [dict(m) for m in p] for t, p in zip(rows.task_idx, rows.prompt)}
gt = {int(t): r["ground_truth"] for t, r in zip(rows.task_idx, rows.reward_model)}
R = [json.loads(l) for l in open(f"{O}/rollouts/ovf_opsd_taco/1.jsonl")]
R = sorted([r for r in R if int(r["env_task_idx"]) in prompt], key=lambda r: len(r["output"]))[:N]

@torch.no_grad()
def logprobs(ctx_ids, resp_ids, chunk=1024):
    ids = torch.tensor(ctx_ids + resp_ids, device="cuda")[None]
    h = model.model(input_ids=ids).last_hidden_state[0, len(ctx_ids) - 1:-1]
    return [torch.log_softmax(model.lm_head(h[s:s + chunk]).float(), -1).cpu() for s in range(0, h.shape[0], chunk)]

allkl, report = [], []
for r in R:
    t = int(r["env_task_idx"]); p = prompt[t]
    pt = [dict(m) for m in p]; pt[-1]["content"] = pt[-1]["content"] + "\n\n" + gt[t]
    sc = tok(tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
    tc = tok(tok.apply_chat_template(pt, tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
    resp = tok(r["output"], add_special_tokens=False).input_ids
    kl = []
    for ls, lt in zip(logprobs(sc, resp), logprobs(tc, resp)):
        kl.append((lt.exp() * (lt - ls)).sum(-1))
    kl = torch.cat(kl); n = len(kl); tot = kl.sum().item()
    srt = kl.sort(descending=True).values
    top1 = srt[:max(1, n // 100)].sum().item() / tot; top5 = srt[:max(1, n // 20)].sum().item() / tot
    first10 = kl[:n // 10].sum().item() / tot
    idx = kl.topk(5).indices.tolist()
    ctx = [repr(tok.decode(resp[max(0, i - 8):i]) + "[[" + tok.decode(resp[i:i + 1]) + "]]")[:80] + f" kl={kl[i]:.2f} @{i}/{n}" for i in sorted(idx)]
    report.append((t, n, kl.mean().item(), top1, top5, first10, float(r["score"]), ctx)); allkl.append(kl)
    print(f"task {t}: {n} tok, mean KL {kl.mean():.4f}, top1% tokens hold {100*top1:.0f}% of KL, top5% {100*top5:.0f}%, first 10% of response {100*first10:.0f}%, student score {float(r['score']):.2f}", flush=True)
    for c in ctx: print("    ", c)
a = torch.cat(allkl); s = a.sort(descending=True).values
print(f"\nALL {len(a)} tokens: mean KL {a.mean():.4f}; top1% hold {100*s[:len(s)//100].sum()/s.sum():.0f}%, top5% {100*s[:len(s)//20].sum()/s.sum():.0f}%, "
      f"tokens with KL>1: {(a>1).float().mean()*100:.2f}%")
