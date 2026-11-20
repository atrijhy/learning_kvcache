"""Summarize the scored tasks and actual failed traces from an experience run."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
tasks = json.loads((root / "tasks.json").read_text())
totals = Counter(t[0] for t in tasks)
records = {tuple(r["key"]): r for r in map(json.loads, (root / "index.jsonl").read_text().splitlines())}
out = ["# Qwen3.5-4B · Claude Code 离线采集", "",
       f"题单 {len(tasks)} 道，已结束 {len(records)} 道。TP=1，DP=16，总任务并发 32，thinking 开启。",
       "", "| 数据集 | 已结束/题数 | 已评分 | 满分 | 部分分 | 零或负分 | 运行异常 | 满分率（已评分） |",
       "|---|---:|---:|---:|---:|---:|---:|---:|"]
examples = defaultdict(dict)
for domain, total in totals.items():
    rows = [r for k, r in records.items() if k[0] == domain]
    rows = [dict(r, reward=r.get("last_verified_reward", r.get("reward"))) for r in rows]
    scored = [r for r in rows if r["status"] == "scored"]
    full = sum(r["reward"] == 1 for r in scored)
    partial = sum(0 < r["reward"] < 1 for r in scored)
    rate = f"{full/len(scored):.1%}" if scored else "—"
    out.append(f"| {domain} | {len(rows)}/{total} | {len(scored)} | {full} | {partial} | {len(scored)-full-partial} | {len(rows)-len(scored)} | {rate} |")
    for r in rows:
        if r["status"] == "scored" and r["reward"] == 1:
            continue
        path = Path(r["trace"]) if r.get("trace") else None
        if not path or not (path / "episode.json").exists():
            continue
        ep = json.loads((path / "episode.json").read_text())
        verdict = ep.get("verifier", {})
        key = r["status"] if r["status"] != "scored" else (ep.get("finish"), verdict.get("reward_type"), r["reward"] > 0)
        if key in examples[domain] or len(examples[domain]) >= 6:
            continue
        checks = verdict.get("verify_result") or {}
        evidence = {k: checks[k] for k in ["result", "passed_checks", "total_checks", "failed_check_indices",
                                             "passed_tests", "total_tests", "wrong_answers", "runtime_errors",
                                             "timeouts", "compile_error"] if k in checks}
        calls = []
        if (path / "events.jsonl").exists():
            with (path / "events.jsonl").open() as stream:
                for line in stream:
                    message = json.loads(line)
                    calls.extend(c["function"]["name"] for c in message.get("tool_calls", []))
        examples[domain][key] = (ep["scenario"], ep["task_idx"], ep.get("task","")[:200],
                                 calls, r.get("reward"), evidence, r.get("error"), path)
out += ["", "## 失败样本", ""]
for domain, cases in examples.items():
    for scenario, i, task, calls, reward, evidence, error, path in cases.values():
        out += [f"- **{domain} / {scenario}:{i}**　得分 {reward}。{task.replace(chr(10),' ')}",
                f"  工具调用 {len(calls)} 次。{', '.join(calls[:8])}。评分记录 {json.dumps(evidence, ensure_ascii=False)}。"
                f" [完整记录]({path.relative_to(root)}/episode.json)"]
        if error:
            out.append("  运行异常：" + error[:250])
(root / "report.md").write_text("\n".join(out)+"\n")
