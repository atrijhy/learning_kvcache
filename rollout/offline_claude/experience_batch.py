"""Collect one independent episode per unique task in the mixed experience train set."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATE = ROOT / "outputs/q35_dp16_session_experience"
SOURCE = ROOT / "traj_data/opsd_mixed_1625_v1/train.parquet"
WORKERS = 32
TIMEOUT_SECONDS = 1200
DOMAINS = {"awm": "awm", "envscaler_rl": "envscaler", "deepcoder_taco": "taco"}


def task_list():
    groups = defaultdict(list)
    for row in pq.read_table(SOURCE, columns=["data_source", "scenario", "task_idx"]).to_pylist():
        domain = DOMAINS[row["data_source"]]
        groups[domain].append((domain, row["scenario"], row["task_idx"]))
    assert all(len(set(groups[d])) == len(groups[d]) == 1625 for d in DOMAINS.values())
    return [task for group in zip(*(groups[d] for d in DOMAINS.values())) for task in group]


def run(task):
    domain, scenario, index = task
    cmd = [sys.executable, str(HERE / "collect.py"), domain, scenario, str(index),
           "--state", str(STATE), "--model", "qwen3.5-4b", "--thinking", "--teacher-session", "--timeout", str(TIMEOUT_SECONDS),
           "--claude", "/root/.local/opt/kvcache-claude-2.1.273/package/claude"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    summary = None
    for line in reversed(proc.stdout.splitlines()):
        try:
            v = json.loads(line)
            if isinstance(v, dict) and "trace" in v:
                summary = v
                break
        except ValueError:
            pass
    result = {**(summary or {"status": "error", "error": proc.stdout[-1000:]}),
              "key": task, "ended": time.time()}
    if summary and summary.get("trace"):
        folder = Path(summary["trace"])
        reason_blocks = 0
        events = folder / "events.jsonl"
        if events.exists():
            for line in events.open():
                try:
                    evt = json.loads(line)
                except ValueError:
                    continue
                if evt.get("role") == "assistant" and (evt.get("reasoning_content") or
                        "</think>" in (evt.get("content") or "")):
                    reason_blocks += 1
        result["thinking_blocks"] = reason_blocks
    return result


def report(records):
    counters = defaultdict(Counter)
    failures = []
    for r in records.values():
        c = counters[r["key"][0]]
        c["finished"] += 1
        if r["status"] == "scored":
            c["scored"] += 1
            c["full_success"] += int(r.get("last_verified_reward", r["reward"]) == 1)
            c["partial"] += int(0 < (r.get("last_verified_reward", r["reward"]) or 0) < 1)
            c["thinking_observed"] += int(bool(r.get("thinking_blocks")))
        else:
            c["errors"] += 1
        c["first_round_success"] += int(r.get("reward") == 1)
        if r["status"] != "scored" or r.get("last_verified_reward", r.get("reward")) != 1:
            failures.append(r)
    for c in counters.values():
        c["full_success_over_scored"] = c["full_success"]/c["scored"] if c["scored"] else None
    for name, obj in [("progress", counters), ("failures", failures)]:
        tmp = STATE / (name+".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        tmp.replace(STATE / (name+".json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    lock = (STATE/"batch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = task_list()
    config = {"source": str(SOURCE.relative_to(ROOT)), "tasks": len(tasks),
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "domains": dict(Counter(t[0] for t in tasks)), "harness": "Claude Code",
              "model_path": "/home/liyu.468/models/Qwen3.5-4B", "thinking": True,
              "parallel_episodes": WORKERS, "data_parallel_size": 16, "tensor_parallel_size": 1,
              "episode_timeout_s": TIMEOUT_SECONDS, "turn_limit": None, "teacher_mode": "same_session", "event_format": "messages-v1"}
    (STATE/"run_config.json").write_text(json.dumps(config, indent=2))
    (STATE/"tasks.json").write_text(json.dumps(tasks))
    index = STATE/"index.jsonl"
    records = {}
    if index.exists():
        for line in index.open():
            v = json.loads(line)
            records[tuple(v["key"])] = v
    targets = [t for t in (tasks[:3] if args.pilot else tasks) if t not in records]
    with urlopen("http://127.0.0.1:18764/v1/models", timeout=10) as reply:
        assert "qwen3.5-4b" in [m["id"] for m in json.load(reply)["data"]]
    with index.open("a", buffering=1) as output, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        it = iter(targets)
        pending = {pool.submit(run, t) for t in [next(it, None) for _ in range(WORKERS)] if t}
        while pending:
            job = next(as_completed(pending))
            pending.remove(job)
            r = job.result()
            records[tuple(r["key"])] = r
            print(json.dumps(r), file=output)
            if len(records) <= 3 or len(records) % 16 == 0:
                report(records)
            next_task = next(it, None)
            if next_task:
                pending.add(pool.submit(run, next_task))
    report(records)
    print("Finished", len(records), flush=True)


if __name__ == "__main__":
    main()
