"""OPSD overfit test, step 1 (author protocol): N random train rows per domain of opsd_mixed_1625_v1 (the author's
retry prompts: task + own failed first attempt + "Notice: ... try again"), each twice: student arm = the prompt as the
OPSD student sees it, teacher arm = same + "\n\n" + reward_model.ground_truth appended to the last user turn (as the OPSD
teacher sees it). Run n=4; the overfit set = tasks with student <= 1/4 and teacher >= 2/4.
Usage: build_ovf_screen.py OUT_DIR {awm|envscaler|taco} [N]"""
import os, shutil, sys
import numpy as np, pandas as pd
out, dom = sys.argv[1:3]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 128
U = f"/efs/kadocjiang/learning_kvcache_upstream/traj_data/opsd_mixed_1625_v1/sources/{dom}"
tr = pd.read_parquet(f"{U}/train.parquet").sample(n=N, random_state=0).reset_index(drop=True)
rows = []
for _, r in tr.iterrows():
    for arm in ("student", "teacher"):
        x = r.copy(); p = [dict(m) for m in r.prompt]
        if arm == "teacher":
            p[-1]["content"] = p[-1]["content"] + "\n\n" + r.reward_model["ground_truth"]
        x["prompt"] = np.array(p, dtype=object)
        x["extra_info"] = {**r.extra_info, "ovf_arm": arm}
        rows.append(x)
os.makedirs(f"{out}/sources/{dom}", exist_ok=True)
pd.DataFrame(rows).reset_index(drop=True).to_parquet(f"{out}/val.parquet")
shutil.copy(f"{U}/train.parquet", f"{out}/sources/{dom}/train.parquet")
tr.to_parquet(f"{out}/picked_train_rows.parquet")
kinds = tr.reward_model.map(lambda m: m["ground_truth"].split("]")[0] + "]").value_counts().to_dict()
print(dom, len(rows), "rows; hint kinds", kinds)
