"""Agent loop tests with a scripted fake model: no API key, no cost, deterministic.

The fake returns real `anthropic.types.Message` objects, so the loop is exercised
against the same types it gets from the API.
"""

import json
from pathlib import Path
from typing import Any

from anthropic.types import Message

from miniagent.agent import AgentConfig, StopReason
from miniagent.agent.loop import run_agent
from miniagent.agent.tools import execute_tool
from miniagent.harness.grade import grade
from miniagent.harness.tasks import TASKS_DIR, load_task, prepare_workspace

FIXED_CHUNKS = '''def chunk(items: list, size: int) -> list[list]:
    """Split `items` into consecutive chunks of at most `size` items."""
    if size <= 0:
        raise ValueError("size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]
'''


def _message(content: list[dict[str, Any]], stop_reason: str) -> Message:
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
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 50,
                "cache_creation_input_tokens": 10,
            },
        }
    )


def _tool_call(call_id: str, name: str, tool_input: dict[str, Any]) -> Message:
    block = {"type": "tool_use", "id": call_id, "name": name, "input": tool_input}
    return _message([block], "tool_use")


class FakeMessages:
    def __init__(self, script: list[Message]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Message:
        # Copy the message list: the loop keeps appending to the same list object.
        self.requests.append({**request, "messages": list(request["messages"])})
        return self.script.pop(0)


class FakeClient:
    def __init__(self, script: list[Message]) -> None:
        self.messages = FakeMessages(script)


def _toy_repo(tmp_path: Path) -> tuple[Any, Path]:
    task = load_task(TASKS_DIR / "fix-off-by-one")
    return task, prepare_workspace(task, tmp_path)


def test_scripted_run_fixes_toy_task(tmp_path: Path):
    task, repo = _toy_repo(tmp_path)
    client = FakeClient(
        [
            _tool_call(
                "t1", "read_file", {"path": "src/chunks.py", "start_line": None, "end_line": None}
            ),
            _tool_call("t2", "write_file", {"path": "src/chunks.py", "content": FIXED_CHUNKS}),
            _tool_call("t3", "run_tests", {}),
            _message([{"type": "text", "text": "Fixed the range bound."}], "end_turn"),
        ]
    )
    transcript = tmp_path / "run.jsonl"

    result = run_agent(
        task.id,
        task.prompt,
        repo,
        AgentConfig(model="fake"),
        client=client,
        transcript_path=transcript,
    )

    assert result.stop_reason == StopReason.FINISHED
    assert result.steps == 4
    assert result.summary == "Fixed the range bound."
    assert result.input_tokens == 400
    assert result.cache_read_tokens == 200
    assert result.output_tokens == 80
    assert grade(task, repo).passed

    # Each request resends the whole history: user, then (assistant, tool results) pairs.
    roles = [m["role"] for m in client.messages.requests[-1]["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant", "user"]
    # The tool result answers the right tool_use id.
    last_results = client.messages.requests[-1]["messages"][-1]["content"]
    assert last_results[0]["tool_use_id"] == "t3"

    events = [json.loads(line)["event"] for line in transcript.read_text().splitlines()]
    assert events == ["start", "step", "step", "step", "step", "end"]


def test_step_limit_stops_the_loop(tmp_path: Path):
    task, repo = _toy_repo(tmp_path)
    client = FakeClient([_tool_call(f"t{i}", "run_tests", {}) for i in range(5)])

    result = run_agent(
        task.id, task.prompt, repo, AgentConfig(model="fake", max_steps=3), client=client
    )

    assert result.stop_reason == StopReason.STEP_LIMIT
    assert result.steps == 3


def test_max_tokens_is_reported_as_error(tmp_path: Path):
    task, repo = _toy_repo(tmp_path)
    client = FakeClient([_message([{"type": "text", "text": "partial"}], "max_tokens")])

    result = run_agent(task.id, task.prompt, repo, AgentConfig(model="fake"), client=client)

    assert result.stop_reason == StopReason.ERROR
    assert "max_tokens" in result.summary


def test_tools_return_errors_instead_of_raising(tmp_path: Path):
    no_range = {"start_line": None, "end_line": None}
    escape = execute_tool("read_file", {"path": "../../etc/passwd", **no_range}, tmp_path)
    missing = execute_tool("read_file", {"path": "nope.py", **no_range}, tmp_path)
    unknown = execute_tool("delete_everything", {}, tmp_path)

    assert escape.is_error and "escapes" in escape.text
    assert missing.is_error
    assert unknown.is_error


def test_failing_tests_are_not_a_tool_error(tmp_path: Path):
    _, repo = _toy_repo(tmp_path)
    (repo / "tests" / "test_fail.py").write_text("def test_fail():\n    assert False\n")

    output = execute_tool("run_tests", {}, repo)

    assert not output.is_error
    assert output.text.startswith("exit code: 1")


def test_scripted_run_uses_search_and_edit_on_multi_file_task(tmp_path: Path):
    task = load_task(TASKS_DIR / "fix-discount-multi-file")
    repo = prepare_workspace(task, tmp_path)
    client = FakeClient(
        [
            _tool_call("t1", "list_dir", {"path": "."}),
            _tool_call("t2", "grep", {"pattern": "def apply_discount", "path": None}),
            _tool_call(
                "t3",
                "edit_file",
                {
                    "path": "src/discounts.py",
                    "old_str": "percent / 1000",
                    "new_str": "percent / 100",
                },
            ),
            _tool_call("t4", "run_tests", {}),
            _message([{"type": "text", "text": "Fixed the divisor."}], "end_turn"),
        ]
    )

    result = run_agent(task.id, task.prompt, repo, AgentConfig(model="fake"), client=client)

    assert result.stop_reason == StopReason.FINISHED
    assert grade(task, repo).passed
    # The grep result the model saw pointed it at the right file.
    grep_result = client.messages.requests[2]["messages"][-1]["content"][0]["content"]
    assert grep_result.startswith("src/discounts.py:1:")
