"""OPSD overfit absorption table: student pass WITHOUT the hint on the picked training tasks (n=4) for the base (screen
student arm), the OPSD-overfit prefix at steps 10/20/40, and the teacher WITH the hint (screen teacher arm).
Usage: absorb_table.py [dom ...]"""
import glob, json, sys
O = "/root/local_runs/ovf"
for dom in sys.argv[1:] or ["taco", "awm", "envscaler"]:
    try: p = json.load(open(f"{O}/data/{dom}_ab/picked.json"))
    except FileNotFoundError: continue
    keys = {tuple(k) for k in p["tasks"]}
    base = sum(p["student"].values()) / len(p["student"]); teach = sum(p["teacher"].values()) / len(p["teacher"])
    row = [f"{dom:10s} tasks {len(keys)}  base(screen,biased) {100*base:5.1f}"]
    for s in ("base777", 10, 20, 40):
        d = f"{O}/out_base777" if s == "base777" else f"{O}/out_ab_step{s}"
        R = [json.loads(l) for f in glob.glob(f"{d}/val_{dom}_ab/*.jsonl") for l in open(f)]
        R = [r for r in R if not (r.get("env_error") or r.get("env_reward_type") == "transport_error")
             and (r["env_scenario"], str(int(r["env_task_idx"]))) in keys]
        row.append(("base(fresh) " if s == "base777" else f"step{s} ") + (f"{100*sum(float(r['score'])>=0.999 for r in R)/len(R):5.1f} (n={len(R)})" if R else "  -  "))
    row.append(f"teacher {100*teach:5.1f}")
    print(" | ".join(row))
