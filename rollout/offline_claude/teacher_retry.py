"""Generate one refine.py advice block and link a clean retry to its failed episode."""
import asyncio
import json
import math
import os
from pathlib import Path
import sys


def public_feedback(record):
    """Only fixed labels, finite scores and nonnegative integer counters can leave the verifier."""
    verdict = record.get("verifier") or {}
    result = verdict.get("verify_result") or {}
    feedback = {}
    score = verdict.get("reward")
    if type(score) in (int, float) and math.isfinite(score):
        feedback["score"] = score
    if isinstance(result, dict):
        for key in ("passed_checks", "total_checks", "checker_errors", "passed_tests", "total_tests"):
            count = result.get(key)
            if type(count) is int and count >= 0:
                feedback[key] = count
    return feedback


async def teacher_advice(record, trace, in_session=False):
    # Standalone collect.py is also invoked from outside the repo.
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from rollout import refine
    from rollout.prompt import _render_student_conversation
    from rollout.offline_claude.teacher_tools import diagnostic_tools

    backend = "deepcoder-taco" if record["domain"] == "taco" else record["domain"]
    key = os.environ.get("JD_LLM_API_KEY")
    if not key:
        key = (Path.home() / ".config/kvcache-learning/jd-teacher.key").read_text().strip()
    options = refine._build_parser().parse_args([
        "--dataset", backend, "--no-only-infer", "--no-llm-judge",
        "--teacher-base-url", "http://llm-gw.jd.local/v1",
        "--teacher-model", "GLM-5.3-joybuilder", "--teacher-api-key", key,
    ])
    refine._resolve_backend_args(options)
    if not options.teacher_api_key:
        raise ValueError("Teacher credentials are missing")
    messages = []
    with (trace / "events.jsonl").open() as stream:
        for line in stream:
            message = json.loads(line)
            if "reasoning_content" in message:
                message["reasoning"] = message.pop("reasoning_content")
            messages.append(message)
    if not any(m["role"] == "assistant" for m in messages):
        raise ValueError("No student output to diagnose")
    schemas = json.loads((trace / "tools.json").read_text())
    task = record["task"]
    if schemas:
        signatures = []
        for tool in schemas:
            properties = tool["inputSchema"].get("properties", {})
            args = ", ".join(f"{name}: {spec.get('type', 'any')}" for name, spec in properties.items())
            signatures.append(f"mcp__episode__{tool['name']}({args})")
        task += ("\n\nAvailable student tools (complete list). Ground your advice in these signatures. "
                 "If a requested operation is unavailable, say so.\n" + "\n".join(signatures))
    class FinalAdviceClient(refine.RetryLLM):
        async def chat_message_stream(self, messages, **kwargs):
            response = await super().chat_message_stream(messages, **kwargs)
            if not (response.content or "").strip():
                raise ValueError("Teacher returned no final answer")
            return response

    client = FinalAdviceClient(base_url=options.teacher_base_url, api_key=options.teacher_api_key,
                               model=options.teacher_model, timeout=options.teacher_timeout)
    feedback = public_feedback(record)
    outcome = "timeout" if record["status"] == "timeout" else "incomplete"
    try:
        actions = []
        async with diagnostic_tools(record, schemas, messages) as (tools, query, replay):
            prompt = (f"Task\n{task}\n\nConversation so far\n"
                      f"{_render_student_conversation(messages)}\n\n"
                      f"Verification: {outcome}. Public feedback: {json.dumps(feedback)}.")
            if in_session:
                prompt += ("\nThe student is waiting in the same live session. Its actions still apply. "
                           "Use prior feedback and later actions as context for the next instruction.")
            if replay:
                prompt += ("\nThe diagnostic environment is a separate reset and replay of the student's calls. "
                           f"Replay statistics: {json.dumps(replay)}. If outputs differ, qualify your conclusions.")
            supervisor = ("You are supervising a student working on the user's task. "
                          "Use the conversation and public tool results to decide what the student should do next. "
                          "Use the available diagnostic tools if they help. Do not infer hidden grading rules.")
            dialog = [{"role": "system", "content": supervisor}, {"role": "user", "content": prompt}]
            while tools and len(actions) < 3:
                turn = await refine.llm_turn_native(
                    client, dialog, tools=tools, tool_choice="auto", temperature=options.teacher_temperature,
                    max_tokens=options.teacher_max_tokens)
                if not turn.tool_calls:
                    break
                for call in turn.tool_calls:
                    call_id, name, args = refine._parse_tool_call(call)
                    if len(actions) < 3:
                        value = await query(name, args)
                        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                        content = content[:4000]
                        actions.append({"name": name, "arguments": args, "result": content})
                    else:
                        content = "Diagnostic call budget exhausted."
                    dialog.append({"role": "tool", "tool_call_id": call_id, "content": content})
        continuity = ("You are speaking in the same ongoing session. Preserve completed work and "
                      "take previous instructions into account." if in_session else
                      "The student will start from the original environment again.")
        dialog[0] = {"role": "system", "content": supervisor + " " + continuity}
        dialog.append({"role": "user", "content": (
            "Speak directly to the student as a human collaborator. In one or two sentences, "
            "give the next concrete action, using the student's language. If the cause is uncertain, "
            "say so. Write only the message you want the student to receive.")})
        final = await refine.llm_turn_native(client, dialog, temperature=options.teacher_temperature,
                                             max_tokens=options.teacher_max_tokens, stream=True)
    finally:
        await client._client.close()
    return {"model": options.teacher_model, "advice": final.content.strip(),
            "source": "offline_claude.teacher_retry.teacher_advice", "tool_schema_count": len(schemas),
            "verifier_feedback": feedback, "replay": replay, "teacher_tool_calls": actions}



def refine_failure(args, index, trace, record, attempt, write):
    pair = {"status": "awaiting_teacher", "domain": record["domain"], "scenario": record["scenario"],
            "task_idx": index, "task_sha256": record["task_sha256"], "attempt_0": ".",
            "initial_session_id": record["session_id"], "initial_status": record["status"],
            "initial_reward": record.get("verifier", {}).get("reward")}
    path = trace / "pair.json"
    write(path, pair)
    try:
        advice = asyncio.run(teacher_advice(record, trace))
        write(trace / "advice.json", advice)
        pair.update(status="retrying", advice="advice.json", attempt_1="attempt_1")
        write(path, pair)
        _, retry = attempt(args, index, trace / "attempt_1", advice["advice"], record["task_sha256"])
        reward = retry.get("verifier", {}).get("reward")
        pair.update(status="paired" if retry["status"] == "scored" else "retry_failed",
                    retry_status=retry["status"], retry_session_id=retry["session_id"], retry_reward=reward)
        if pair["initial_reward"] is not None and reward is not None:
            pair["reward_delta"] = reward - pair["initial_reward"]
        if retry.get("error"):
            pair["error"] = retry["error"]
    except Exception as exc:
        pair.update(status="teacher_error" if pair["status"] == "awaiting_teacher" else "retry_failed",
                    error=str(exc)[:600])
    write(path, pair)
    return pair
