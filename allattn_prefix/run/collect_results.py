"""汇总 $EVAL_OUT 下所有模型的 greedy 评测，打印带 base 对比的 markdown 表（括号里是相对 base 的差）。
用法：python collect_results.py [EVAL_OUT] [BASE_DIR_NAME=Qwen3.5-4B]
截断列 = LCB v6 里生成到上限还没结束的题数（finish_reason=length），用来区分"答错"和"想太久被截断"。"""
import json, os, re, sys
root = sys.argv[1] if len(sys.argv) > 1 else os.environ["EVAL_OUT"]
base_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen3.5-4B"
COLS = ["BFCL v3", "multi-turn", "relevance", "irrelevance", "LCB v6", "LCB 截断", "HumanEval", "MBPP"]

def first_pass1(path):
    try:
        m = re.search(r'"pass@1":\s*([0-9.]+)', open(path, errors="ignore").read())
        return 100 * float(m.group(1)) if m else None
    except FileNotFoundError:
        return None

def scores(d):
    r = {}
    try:
        t = open(f"{d}/bfcl_v3.log", errors="ignore").read()
        row = re.search(r"\n\s*1 \|[^\n]*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)%", t)
        if row:
            v = [float(x) for x in row.groups()]
            r.update({"BFCL v3": v[0], "multi-turn": v[3], "relevance": v[4], "irrelevance": v[5]})
    except FileNotFoundError:
        pass
    r["LCB v6"] = first_pass1(f"{d}/lcb_v6.log")
    r["HumanEval"] = first_pass1(f"{d}/humaneval+.log")
    r["MBPP"] = first_pass1(f"{d}/mbpp+.log")
    try:
        n = [json.loads(l).get("finish_reason") for l in open(f"{d}/coding/lcb_v6_gen/samples.jsonl")]
        r["LCB 截断"] = sum(x == "length" for x in n)
    except FileNotFoundError:
        r["LCB 截断"] = None
    return r

runs = sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)))
base = scores(os.path.join(root, base_name)) if base_name in runs else {}
print("| 模型 | " + " | ".join(COLS) + " |\n|" + "---|" * (len(COLS) + 1))
for name in ([base_name] if base else []) + [x for x in runs if x != base_name]:
    s = scores(os.path.join(root, name)); cells = []
    for c in COLS:
        v = s.get(c)
        if v is None: cells.append("-"); continue
        txt = f"{v}" if c == "LCB 截断" else f"{v:.2f}"
        if name != base_name and base.get(c) is not None:
            dv = v - base[c]; txt += f" ({dv:+d})" if c == "LCB 截断" else f" ({dv:+.1f})"
        cells.append(txt)
    print(f"| {name} | " + " | ".join(cells) + " |")
