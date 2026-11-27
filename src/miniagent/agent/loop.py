"""The agent loop: ask the model, run the tools it asks for, repeat.

This is a hand-rolled loop on the raw Messages API on purpose. The SDK's tool runner
would hide exactly the parts this project exists to learn: when to stop, what goes
back into the conversation, and what each step costs.
Read docs/milestone-1-walkthrough.md alongside this file.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import anthropic

from miniagent.agent.tools import CommandRunner, execute_tool, tool_schemas
from miniagent.agent.types import AgentConfig, RunResult, StopReason


class MessagesClient(Protocol):
    """The one slice of `anthropic.Anthropic` the loop uses. Tests pass a fake."""

    @property
    def messages(self) -> Any: ...


StepCallback = Callable[[int, Any, list[dict[str, Any]]], None]


def run_agent(
    task_id: str,
    task_prompt: str,
    repo_path: Path,
    config: AgentConfig,
    *,
    client: MessagesClient | None = None,
    transcript_path: Path | None = None,
    on_step: StepCallback | None = None,
    sandbox: CommandRunner | None = None,
) -> RunResult:
    """Run the agent on one task inside `repo_path` and return what happened.

    A pure function of its inputs: no globals, no working-directory changes. That is
    what lets the orchestrator later run many of these in parallel.

    With `sandbox`, shell commands and tests run there (a container with `repo_path`
    mounted) instead of on the host. The caller starts and stops it.
    """
    tools = tool_schemas(config.tools)
    messages: list[dict[str, Any]] = [{"role": "user", "content": task_prompt}]
    result = RunResult(
        task_id=task_id,
        repo_path=repo_path,
        stop_reason=StopReason.STEP_LIMIT,
        transcript_path=transcript_path,
    )
    started = time.monotonic()
    transcript = _Transcript(transcript_path)
    transcript.write({"event": "start", "task_id": task_id, "config": config.model_dump()})

    try:
        # Created inside the try: a missing API key surfaces as a clean ERROR result.
        client = client or anthropic.Anthropic()
        for step in range(1, config.max_steps + 1):
            if time.monotonic() - started > config.timeout_s:
                result.stop_reason = StopReason.TIMEOUT
                break

            response = client.messages.create(**_request(config, tools, messages))
            result.steps = step
            _add_usage(result, response.usage)

            # Append the assistant turn exactly as returned (all blocks, not just text).
            messages.append({"role": "assistant", "content": response.content})

            tool_results = _run_tool_calls(response.content, repo_path, config, sandbox)
            transcript.write(
                {
                    "event": "step",
                    "step": step,
                    "stop_reason": response.stop_reason,
                    "content": [b.model_dump() for b in response.content],
                    "tool_results": tool_results,
                    "usage": response.usage.model_dump(),
                }
            )
            if on_step:
                on_step(step, response, tool_results)

            if response.stop_reason == "tool_use":
                # All results from one assistant turn go back in ONE user message.
                messages.append({"role": "user", "content": tool_results})
                continue

            if response.stop_reason == "end_turn":
                result.stop_reason = StopReason.FINISHED
                result.summary = _text_of(response.content)
            else:
                # max_tokens, refusal, or anything new: stop and record why.
                result.stop_reason = StopReason.ERROR
                result.summary = f"model stopped with stop_reason={response.stop_reason}"
            break
    except anthropic.AnthropicError as exc:
        result.stop_reason = StopReason.ERROR
        result.summary = f"{type(exc).__name__}: {exc}"
    finally:
        result.wall_time_s = round(time.monotonic() - started, 2)
        transcript.write({"event": "end", "result": result.model_dump(mode="json")})

    return result


def _request(
    config: AgentConfig, tools: list[dict[str, Any]], messages: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build the Messages API arguments for one step."""
    request: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "system": config.system_prompt,
        "tools": tools,
        "messages": messages,
        # Automatic prompt caching: the API caches the longest stable prefix
        # (tools, system, earlier messages). Every step resends the whole history,
        # so from step 2 on most input tokens are cache reads at ~0.1x price.
        "cache_control": {"type": "ephemeral"},
    }
    if config.effort:
        request["output_config"] = {"effort": config.effort}
    return request


def _run_tool_calls(
    content: list[Any],
    repo_path: Path,
    config: AgentConfig,
    sandbox: CommandRunner | None = None,
) -> list[dict[str, Any]]:
    results = []
    for block in content:
        if block.type != "tool_use":
            continue
        output = execute_tool(
            block.name,
            block.input,
            repo_path,
            test_command=config.test_command,
            test_timeout_s=config.test_timeout_s,
            share_python_env=config.share_python_env,
            sandbox=sandbox,
        )
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,  # must match the tool_use block it answers
                "content": output.text,
                "is_error": output.is_error,
            }
        )
    return results


def _add_usage(result: RunResult, usage: Any) -> None:
    # Cached tokens are reported separately from `input_tokens`; count all of them
    # so cost reporting later can price each kind correctly.
    result.input_tokens += usage.input_tokens
    result.cache_read_tokens += usage.cache_read_input_tokens or 0
    result.cache_write_tokens += usage.cache_creation_input_tokens or 0
    result.output_tokens += usage.output_tokens


def _text_of(content: list[Any]) -> str:
    return "\n".join(b.text for b in content if b.type == "text").strip()


class _Transcript:
    """Append-only JSONL log: one line per event, flushed as it happens, so a crash
    mid-run still leaves everything up to the crash on disk."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    def write(self, record: dict[str, Any]) -> None:
        if not self.path:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
