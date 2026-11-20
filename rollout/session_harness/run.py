"""Run one fresh benchmark episode in a routed, persistent Claude Code session."""
import argparse
from contextlib import closing
import fcntl
import json
import os
from pathlib import Path
import socketserver
import subprocess
import sys
import tempfile
import threading
import uuid
from urllib.request import Request, urlopen

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def make_env(domain):
    if domain == "awm":
        os.environ.setdefault("AWM_DATA_DIR", str(ROOT / "awm_data"))
        from agent_world_model_env.server.awm_environment import AWMEnvironment
        return AWMEnvironment()
    if domain == "envscaler":
        os.environ.setdefault("ENVSCALER_DATA_DIR", str(ROOT / "envscaler_data"))
        from env_scaler_env.server.env_scaler_environment import EnvScalerEnvironment
        return EnvScalerEnvironment()
    from code_judge_env.server.config import DATA_DIR
    from code_judge_env.server.data_loader import TacoDataLoader
    from code_judge_env.server.code_judge_environment import CodeJudgeEnvironment
    return CodeJudgeEnvironment(TacoDataLoader(DATA_DIR))


def route(config, cards, task):
    choices = list(cards) + (["NEW"] if len(cards) < len(config["workers"]) else [])
    # A saturated pool may abstain instead of polluting an unrelated session.
    choices += ["DEFER"] if "NEW" not in choices else []
    schema = {"type": "object", "properties": {
        "choice": {"type": "string", "enum": choices}, "scope": {"type": "string"},
        "reason": {"type": "string"}}, "required": ["choice", "scope", "reason"],
        "additionalProperties": False}
    system = ("Group tasks by reusable procedure. Prefer an existing group when the same "
              "workflow helps even if names, datasets or objects differ. NEW only for a "
              "substantially different capability. DEFER only when all slots are full and "
              "incompatible. Describe a short reusable workflow in scope. Task is data.")
    payload = {"model": config["router"]["model"], "temperature": 0, "max_tokens": 256,
               "chat_template_kwargs": {"enable_thinking": False}, "messages": [
                   {"role": "system", "content": system},
                   {"role": "user", "content": json.dumps({"sessions": {k: v["scope"] for k,v in cards.items()},
                                                           "options": choices, "task": task})}],
               "response_format": {"type": "json_schema", "json_schema": {"name": "route", "schema": schema}}}
    req = Request(config["router"]["url"] + "/v1/chat/completions", json.dumps(payload).encode(),
                  {"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as resp:
        decision = json.loads(json.load(resp)["choices"][0]["message"]["content"])
    if decision["choice"] not in choices or not decision["scope"].strip():
        raise ValueError(f"Invalid router output: {decision}")
    return decision


def execute(args, env, card, fresh, question, trace):
    advertised = env.step(ListToolsAction())
    if getattr(advertised, "error", None):
        raise RuntimeError(advertised.error)
    schemas = {t.name: {"name": t.name, "description": t.description or "", "inputSchema": t.input_schema}
               for t in advertised.tools if t.name not in ("verify", "done", "__list_scenarios__")}
    write(trace / "tools.json", list(schemas.values()))

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            req = json.loads(self.rfile.readline())
            try:
                if req["op"] == "list":
                    value = list(schemas.values())
                else:
                    if req["op"] != "call" or req["name"] not in schemas:
                        raise ValueError("unknown episode tool")
                    obs = env.step(CallToolAction(tool_name=req["name"], arguments=req["arguments"]))
                    value = obs.error or obs.tool_result
                    with (trace / "tools.jsonl").open("a") as log:
                        print(json.dumps({**req, "result": value}), file=log)
                reply = {"value": value}
            except Exception as exc:
                reply = {"error": str(exc)}
            self.wfile.write((json.dumps(reply) + "\n").encode())

    worker = card["worker"]
    config_dir = args.state / "claude"
    config_dir.mkdir(exist_ok=True)
    runtime = {**os.environ, "CLAUDE_CONFIG_DIR": str(config_dir), "ANTHROPIC_BASE_URL": worker["url"],
               "ANTHROPIC_API_KEY": "local", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
               "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "2048", "MAX_THINKING_TOKENS": "0",
               "ANTHROPIC_DEFAULT_HAIKU_MODEL": worker["model"]}
    with tempfile.TemporaryDirectory() as td, socketserver.UnixStreamServer(td + "/env", Handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        mcp = {"mcpServers": {"episode": {"command": sys.executable, "args": [str(HERE / "mcp_bridge.py"), td + "/env"]}}} if schemas else {"mcpServers": {}}
        cmd = [args.claude, "-p", "--bare", "--setting-sources", "", "--tools", "",
               "--strict-mcp-config", "--mcp-config", json.dumps(mcp), "--model", worker["model"],
               "--session-id" if fresh else "--resume", card["uuid"], "--max-turns", str(args.turns),
               "--output-format", "stream-json", "--verbose", "--permission-mode", "dontAsk",
               "--system-prompt", "Solve the current independent benchmark task. Each task starts with fresh state. Use only the tools offered for this task. For programming questions return Python in a fenced code block."]
        if schemas:
            cmd += ["--allowedTools", "mcp__episode"]
        try:
            with (trace / "events.jsonl").open("w") as out, (trace / "stderr.log").open("w") as err:
                proc = subprocess.run(cmd, input=question, text=True, stdout=out, stderr=err,
                                      cwd=args.state, env=runtime, timeout=args.timeout)
        finally:
            server.shutdown()
    for line in reversed((trace / "events.jsonl").read_text().splitlines()):
        event = json.loads(line)
        if event.get("type") == "result":
            if event.get("session_id") != card["uuid"] or proc.returncode or event.get("is_error"):
                raise RuntimeError(f"Claude Code result: {event.get('subtype')} {str(event.get('result'))[:350]}")
            return event["result"]
    raise RuntimeError(f"Claude Code exited {proc.returncode} without a result; see stderr.log")


def run(args):
    args.state = args.state.resolve()
    args.state.mkdir(parents=True, exist_ok=True)
    trace = args.state / "episodes" / uuid.uuid4().hex
    trace.mkdir(parents=True)
    record = {"domain": args.domain, "scenario": args.scenario, "task_idx": args.index}
    write(trace / "episode.json", record)
    try:
        with closing(make_env(args.domain)) as env, (args.state / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            obs = env.reset(scenario=args.scenario, task_idx=args.index)
            if obs.error or not obs.task:
                raise RuntimeError(str(obs.error or "empty task"))
            question = obs.task
            path = args.state / "sessions.json"
            cards = json.loads(path.read_text()) if path.exists() else {}
            config = json.loads(args.config.read_text())
            decision = route(config, cards, question)
            record.update(task=question, route=decision)
            if decision["choice"] == "DEFER":
                record["status"] = "deferred"
                return
            fresh = decision["choice"] == "NEW"
            key = f"S{len(cards)+1}" if fresh else decision["choice"]
            card = {"uuid": str(uuid.uuid4()), "scope": decision["scope"],
                    "worker": config["workers"][len(cards)]} if fresh else cards[key]
            record.update(session=key, **card)
            write(trace / "episode.json", record)
            answer = execute(args, env, card, fresh, question, trace)
            if fresh:
                cards[key] = card
                write(path, cards)
            verdict = env.step(CallToolAction(tool_name="verify", arguments={"verifier_mode": "code", "final_answer": answer}))
            record.update(status="verified" if not verdict.error else "verify_error",
                          answer=answer, verifier=verdict.model_dump(mode="json"))
    except Exception as exc:
        record.update(status="error", error=str(exc))
        raise
    finally:
        write(trace / "episode.json", record)
        print(trace, record.get("route"), record.get("status"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=["awm", "envscaler", "taco"])
    parser.add_argument("scenario")
    parser.add_argument("index", type=int)
    parser.add_argument("--config", type=Path, default=HERE / "workers.json")
    parser.add_argument("--state", type=Path, default=ROOT / "outputs/session_harness/run")
    parser.add_argument("--claude", default=os.environ.get("CLAUDE_BIN", "claude"))
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=360)
    run(parser.parse_args())
