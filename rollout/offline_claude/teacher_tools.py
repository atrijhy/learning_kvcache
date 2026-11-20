"""Public diagnostic tools for the offline teacher's isolated episode."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, closing, contextmanager
import json

QUERY_PREFIXES = ("get_", "list_", "search_", "find_", "lookup_", "count_", "describe_", "check_")
CONTROL_TOOLS = {"verify", "done", "__list_scenarios__"}


def schema(name, description, parameters):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": parameters}}


def canonical(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


@contextmanager
def _open_tools(record, saved_tools, messages):
    if record["domain"] == "taco":
        from code_judge_env.server.judge import extract_python_code
        name = "check_student_python_syntax"
        def syntax_check(tool, args):
            if tool != name or args:
                return {"error": "Unknown tool or arguments"}
            code = extract_python_code(record.get("answer") or "")
            if not code:
                return {"code_found": False}
            try:
                compile(code, "<student submission>", "exec")
            except (SyntaxError, ValueError, OverflowError) as exc:
                return {"valid": False, "message": str(exc), "line": getattr(exc, "lineno", None)}
            return {"code_found": True, "valid": True}
        yield [schema(name, "Check Python syntax in the student's submitted answer without executing it.",
                      {"type": "object", "properties": {}, "additionalProperties": False})], syntax_check, {}
        return

    from rollout.offline_claude.collect import make_env
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
    with closing(make_env(record["domain"])) as env:
        obs = env.reset(scenario=record["scenario"], task_idx=record["task_idx"])
        if obs.error or obs.task != record["task"]:
            raise ValueError("Could not reset teacher environment to the original task")
        listing = env.step(ListToolsAction())
        if listing.error:
            raise ValueError("Could not list teacher environment tools")
        names = {t["name"] for t in saved_tools} - CONTROL_TOOLS
        available = {t.name: t for t in listing.tools if t.name in names}
        queries = {n: t for n, t in available.items() if n.startswith(QUERY_PREFIXES)}

        def step(name, args):
            obs = env.step(CallToolAction(tool_name=name, arguments=args))
            return obs.error or obs.tool_result

        results = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
        replay = {"replayed": 0, "different_results": 0, "skipped": 0}
        for message in messages:
            for call in message.get("tool_calls", []):
                fn = call["function"]
                name = fn["name"].removeprefix("mcp__episode__")
                if call["id"] not in results or name not in available:
                    replay["skipped"] += 1
                    continue
                value = step(name, fn["arguments"])
                replay["replayed"] += 1
                if canonical(value) != canonical(results[call["id"]]):
                    replay["different_results"] += 1

        def query(name, args):
            if name not in queries:
                return {"error": "Not an advertised diagnostic tool"}
            return step(name, args)

        yield [schema(n, t.description or "", t.input_schema) for n, t in queries.items()], query, replay


@asynccontextmanager
async def diagnostic_tools(record, saved_tools, messages):
    # AWM's synchronous MCP client runs its own event loop. Keep its lifecycle on one thread.
    loop = asyncio.get_running_loop()
    session = _open_tools(record, saved_tools, messages)
    with ThreadPoolExecutor(max_workers=1) as worker:
        tools, call, replay = await loop.run_in_executor(worker, session.__enter__)
        async def query(name, args):
            return await loop.run_in_executor(worker, call, name, args)
        try:
            yield tools, query, replay
        finally:
            await loop.run_in_executor(worker, session.__exit__, None, None, None)
