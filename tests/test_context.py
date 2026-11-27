"""Context management, with a scripted fake model: no API calls, no cost."""

import json
from pathlib import Path
from typing import Any

from anthropic.types import Message

from miniagent.agent import AgentConfig, StopReason
from miniagent.agent.context import (
    CONTEXT_EDITING_BETA,
    NOTES_TOOL,
    SUMMARY_PROMPT,
    ContextConfig,
    ContextManager,
)
from miniagent.agent.loop import run_agent
from miniagent.agent.pricing import cost_usd

OFF = ContextConfig(clear_tool_results_at=None, compact_at=None)


def message(content: list[dict[str, Any]], stop_reason: str, input_tokens: int = 100) -> Message:
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "fake",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_read_input_tokens": input_tokens - 10,
                "cache_creation_input_tokens": 0,
            },
        }
    )


def bash_call(call_id: str, command: str, input_tokens: int = 100) -> Message:
    thinking = {"type": "thinking", "thinking": "", "signature": f"sig-{call_id}"}
    use = {
        "type": "tool_use",
        "id": call_id,
        "name": "bash",
        "input": {"command": command, "timeout_s": None},
    }
    return message([thinking, use], "tool_use", input_tokens)


def done(text: str = "done", input_tokens: int = 100) -> Message:
    return message([{"type": "text", "text": text}], "end_turn", input_tokens)


class FakeMessages:
    def __init__(self, script: list[Message]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Message:
        self.requests.append(json.loads(json.dumps(request, default=_dump)))
        return self.script.pop(0)


class FakeClient:
    def __init__(self, script: list[Message], beta_script: list[Message] | None = None) -> None:
        self.messages = FakeMessages(script)
        self.beta = type("Beta", (), {"messages": FakeMessages(beta_script or [])})()


def _dump(obj: Any) -> Any:
    return obj.model_dump() if hasattr(obj, "model_dump") else str(obj)


def manager(config: ContextConfig, repo: Path | None = None) -> ContextManager:
    return ContextManager(
        "Fix the bug.", config, model="claude-haiku-5-5", system_prompt="SYS", tools=[],
        repo_path=repo,
    )  # fmt: skip


def fill(ctx: ContextManager, n: int, size: int = 4000) -> None:
    """n assistant tool calls, each answered by a large tool result."""
    for i in range(n):
        response = bash_call(f"t{i}", "cat big.txt")
        ctx.record_response(response.content, response.usage)
        ctx.add_tool_results(
            [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * size}]
        )


# --- Clearing old tool results ------------------------------------------------------


def test_clear_keeps_recent_results_and_strips_thinking():
    ctx = manager(ContextConfig(clear_tool_results_at=3000, keep_tool_results=2, compact_at=None))
    fill(ctx, 5)
    ctx.estimated_tokens = 20_000  # as if the last response reported a big prompt
    events = ctx.prepare(client=None)

    assert [e.kind for e in events] == ["clear"]
    assert events[0].detail["cleared_results"] == 3
    results = [m["content"][0]["content"] for m in ctx.wire()[1:] if m["role"] == "user"]
    assert all("cleared" in r for r in results[:3])
    assert results[3:] == ["x" * 4000, "x" * 4000]
    # Bookkeeping keys never reach the API, and no thinking block is left behind.
    assert "_cleared" not in json.dumps(ctx.wire(), default=_dump)
    assert "thinking" not in json.dumps(ctx.wire(), default=_dump)
    assert ctx.estimated_tokens < events[0].tokens_before


def test_clear_fires_in_batches_not_every_step():
    ctx = manager(ContextConfig(clear_tool_results_at=3000, keep_tool_results=2, compact_at=None))
    fill(ctx, 5)
    ctx.estimated_tokens = 20_000
    assert ctx.prepare(client=None)
    ctx.estimated_tokens = 5000  # still above the trigger, but nothing new is old
    assert ctx.prepare(client=None) == []


def test_below_threshold_nothing_changes():
    ctx = manager(ContextConfig(clear_tool_results_at=10**6, compact_at=10**6))
    fill(ctx, 5)
    before = json.dumps(ctx.wire(), default=_dump)
    assert ctx.prepare(client=None) == []
    assert json.dumps(ctx.wire(), default=_dump) == before


# --- Compaction ---------------------------------------------------------------------


def test_compaction_restarts_from_task_and_summary():
    ctx = manager(ContextConfig(clear_tool_results_at=None, compact_at=1000, notes=True))
    fill(ctx, 3)
    ctx.run_notes({"action": "append", "content": "bug is in parse() line 40"})
    client = FakeClient([done("SUMMARY: edited src/app.py, tests 3/4 pass")])

    events = ctx.prepare(client)

    assert [e.kind for e in events] == ["compact"]
    assert len(ctx.messages) == 1
    restart = ctx.messages[0]["content"]
    assert restart.startswith("Fix the bug.")  # the task, verbatim
    assert "SUMMARY: edited src/app.py" in restart
    assert "bug is in parse() line 40" in restart  # notes survive
    assert ctx.extra_usage  # the summary call is billed too
    # The summary request: same system and tools (a cache hit), no tool calls allowed,
    # and the ask appended after the last tool results, not as a new user turn.
    request = client.messages.requests[0]
    assert request["tool_choice"] == {"type": "none"}
    assert request["system"] == ctx.system
    last = request["messages"][-1]["content"]
    assert last[0]["type"] == "tool_result" and last[-1]["text"] == SUMMARY_PROMPT


def test_failed_summary_keeps_the_history():
    ctx = manager(ContextConfig(clear_tool_results_at=None, compact_at=1000))
    fill(ctx, 3)
    events = ctx.prepare(FakeClient([done("")]))  # empty summary
    assert [e.kind for e in events] == ["compact_failed"]
    assert len(ctx.messages) == 7


# --- Server-side clearing -----------------------------------------------------------


def test_server_clearing_is_request_params_and_uses_the_beta_endpoint(tmp_path: Path):
    config = AgentConfig(
        model="fake",
        context=ContextConfig(server_clear_at=50_000, clear_tool_results_at=None, compact_at=None),
    )
    client = FakeClient([], beta_script=[bash_call("t1", "echo hi"), done()])
    result = run_agent("t", "go", tmp_path, config, client=client)

    assert result.stop_reason == StopReason.FINISHED
    assert not client.messages.requests  # everything went to client.beta.messages
    request = client.beta.messages.requests[0]
    assert request["betas"] == [CONTEXT_EDITING_BETA]
    edit = request["context_management"]["edits"][0]
    assert edit["type"] == "clear_tool_uses_20250919"
    assert edit["trigger"] == {"type": "input_tokens", "value": 50_000}


# --- The loop end to end ------------------------------------------------------------


def test_history_is_append_only_until_a_strategy_fires(tmp_path: Path):
    """Each request must extend the previous one byte for byte: that is what keeps
    the prompt cache warm and keeps thinking blocks valid on newer models."""
    script = [bash_call(f"t{i}", "echo hi") for i in range(4)] + [done()]
    client = FakeClient(script)
    run_agent("t", "go", tmp_path, AgentConfig(model="fake", context=OFF), client=client)

    requests = client.messages.requests
    for earlier, later in zip(requests, requests[1:], strict=False):
        assert earlier["system"] == later["system"] and earlier["tools"] == later["tools"]
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]


def test_long_run_compacts_and_finishes(tmp_path: Path):
    big = 150_000
    script = [
        bash_call("t1", "echo a", input_tokens=1_000),
        bash_call("t2", "echo b", input_tokens=big),  # context jumps past the trigger
        done("SUMMARY of the run"),  # the summarizer's answer
        bash_call("t3", "echo c", input_tokens=2_000),
        done("fixed"),
    ]
    client = FakeClient(script)
    config = AgentConfig(
        model="claude-haiku-5-5",
        context=ContextConfig(clear_tool_results_at=None, compact_at=100_000),
    )
    result = run_agent(
        "t", "go", tmp_path, config, client=client, transcript_path=tmp_path / "t.jsonl"
    )

    assert result.stop_reason == StopReason.FINISHED and result.summary == "fixed"
    assert result.compactions == 1
    assert result.peak_context_tokens >= big
    after = client.messages.requests[3]["messages"]
    assert len(after) == 1 and "SUMMARY of the run" in after[0]["content"]
    events = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert any(e["event"] == "context" and e["kind"] == "compact" for e in events)
    assert all("context_estimate" in e for e in events if e["event"] == "step")


def test_cost_and_budget_stop(tmp_path: Path):
    client = FakeClient([bash_call(f"t{i}", "echo") for i in range(10)])
    config = AgentConfig(model="claude-haiku-5-5", context=OFF, max_cost_usd=0.00002)
    result = run_agent("t", "go", tmp_path, config, client=client)
    assert result.stop_reason == StopReason.BUDGET
    assert result.cost_usd is not None and result.cost_usd >= 0.00002
    assert result.steps < 10
    assert 0 < result.cache_hit_rate < 1


def test_notes_tool_is_offered_only_when_enabled(tmp_path: Path):
    off = manager(ContextConfig())
    on = manager(ContextConfig(notes=True))
    assert NOTES_TOOL not in off.tools and NOTES_TOOL in on.tools
    assert on.run_notes({"action": "read", "content": None}) == ("(no notes yet)", False)
    on.run_notes({"action": "write", "content": "plan: A"})
    on.run_notes({"action": "append", "content": "A failed"})
    assert on.run_notes({"action": "read", "content": None})[0] == "plan: A\nA failed"


def test_project_instructions_are_added_to_the_system_prompt(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("Use tabs. Run `make test`.")
    ctx = manager(ContextConfig(), repo=tmp_path)
    assert "Run `make test`." in ctx.system_prompt
    assert ctx.system[0]["cache_control"] == {"type": "ephemeral"}
    assert (
        "make test"
        not in manager(ContextConfig(project_instructions=False), tmp_path).system_prompt
    )


def test_exact_token_count_when_asked():
    class Counter:
        class messages:  # noqa: N801
            @staticmethod
            def count_tokens(**_: Any):
                return type("R", (), {"input_tokens": 4242})()

    ctx = manager(ContextConfig(count_tokens="api", clear_tool_results_at=None, compact_at=None))
    ctx.prepare(Counter())
    assert ctx.estimated_tokens == 4242


def test_pricing():
    # 1M of each kind on Haiku 5.5: 0.10 + 0.01 + 0.125 + 0.50
    assert round(cost_usd("claude-haiku-5-5", 10**6, 10**6, 10**6, 10**6), 4) == 0.735
    assert cost_usd("some-unknown-model", 1, 1, 1, 1) is None
