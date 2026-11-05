"""OPSD overfit test, step 2: from the screen (student vs teacher arm, n=4), pick the K tasks with student <= 1/4 and the
largest teacher - student gap (teacher >= 2/4), write the OPSD train set (the author's rows unchanged) and the
absorption eval set (student-arm rows). Usage: select_ovf.py SCREEN_DIR VAL_OUT_DIR OPSD_DATA_DIR {dom} [K]"""
import glob, json, os, shutil, sys
from collections import defaultdict
import pandas as pd, re
HINT = re.compile(r"\n\n\[(Reference plan|Expert advice|Reference solution)\]")
scr, val_out, opsd_dir, dom = sys.argv[1:5]; K = int(sys.argv[5]) if len(sys.argv) > 5 else 24
st, te = defaultdict(list), defaultdict(list)
for f in glob.glob(f"{val_out}/*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r.get("env_error") or r.get("env_reward_type") == "transport_error": continue
        k = (r.get("env_scenario"), int(r["env_task_idx"]))
        # teacher arm = ground_truth appended to the last user turn ("\n\n[Reference plan|Expert advice|Reference solution]");
        # AWM rows are 75% fresh first attempts (no "try again"), so the marker alone decides
        teacher = HINT.search(r["input"]) is not None
        (te if teacher else st)[k].append(float(r["score"]) >= 0.999)
keys = [k for k in st if k in te]
s = lambda k: sum(st[k]) / len(st[k]); t = lambda k: sum(te[k]) / len(te[k])
print(f"{dom}: {len(keys)} tasks; student pass {100*sum(map(s,keys))/len(keys):.1f}%  teacher pass {100*sum(map(t,keys))/len(keys):.1f}%")
cand = sorted([k for k in keys if s(k) <= 0.25 and t(k) >= 0.5], key=lambda k: s(k) - t(k))
print(f"candidates (student<=1/4, teacher>=2/4): {len(cand)}")
pick = cand[:K]
if len(pick) < K:  # fill with the next-largest positive gaps
    more = sorted([k for k in keys if k not in pick and t(k) >= s(k)], key=lambda k: s(k) - t(k))
    pick += more[:K - len(pick)]
pick = set(pick)
tr = pd.read_parquet(f"{scr}/picked_train_rows.parquet")
sel = tr[[(r.scenario, int(r.task_idx)) in pick for r in tr.itertuples()]].reset_index(drop=True)
os.makedirs(f"{opsd_dir}/sources/{dom}", exist_ok=True)
sel.to_parquet(f"{opsd_dir}/sources/{dom}/train.parquet")
v = pd.read_parquet(f"{scr}/val.parquet")
v = v[v.extra_info.map(lambda e: e["ovf_arm"]) == "student"]
v = v[[(sc, int(i)) in pick for i, sc in zip(v.task_idx, v.scenario)]].reset_index(drop=True)
v.to_parquet(f"{opsd_dir}/val.parquet")
json.dump({"tasks": sorted([list(map(str, k)) for k in pick]), "student": {str(k): s(k) for k in pick}, "teacher": {str(k): t(k) for k in pick}},
          open(f"{opsd_dir}/picked.json", "w"), indent=1)
print(f"picked {len(sel)} train rows; base student pass on them {100*sum(s(k) for k in pick)/len(pick):.1f}%, teacher {100*sum(t(k) for k in pick)/len(pick):.1f}%")
