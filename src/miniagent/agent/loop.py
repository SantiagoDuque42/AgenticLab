"""The agent loop: ask the model, run the tools it asks for, repeat.

This is a hand-rolled loop on the raw Messages API on purpose. The SDK's tool runner
would hide exactly the parts this project exists to learn: when to stop, what goes
back into the conversation, and what each step costs.
Read docs/milestone-1-walkthrough.md alongside this file, and
docs/stage-3-walkthrough.md for how the conversation is kept small (ContextManager).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import anthropic

from miniagent.agent.context import ContextEvent, ContextManager
from miniagent.agent.pricing import cost_usd
from miniagent.agent.tools import CommandRunner, ToolOutput, execute_tool, tool_schemas
from miniagent.agent.types import AgentConfig, RunResult, StopReason


class MessagesClient(Protocol):
    """The one slice of `anthropic.Anthropic` the loop uses. Tests pass a fake."""

    @property
    def messages(self) -> Any: ...


StepCallback = Callable[[int, Any, list[dict[str, Any]]], None]
ContextCallback = Callable[[int, ContextEvent], None]


def run_agent(
    task_id: str,
    task_prompt: str,
    repo_path: Path,
    config: AgentConfig,
    *,
    client: MessagesClient | None = None,
    transcript_path: Path | None = None,
    on_step: StepCallback | None = None,
    on_context: ContextCallback | None = None,
    sandbox: CommandRunner | None = None,
) -> RunResult:
    """Run the agent on one task inside `repo_path` and return what happened.

    A pure function of its inputs: no globals, no working-directory changes. That is
    what lets the orchestrator later run many of these in parallel.

    With `sandbox`, shell commands and tests run there (a container with `repo_path`
    mounted) instead of on the host. The caller starts and stops it.
    """
    ctx = ContextManager(
        task_prompt,
        config.context,
        model=config.model,
        system_prompt=config.system_prompt,
        tools=tool_schemas(config.tools),
        repo_path=repo_path,
    )
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
            if _over_budget(result, config):
                result.stop_reason = StopReason.BUDGET
                result.summary = f"stopped at the cost cap: ${result.cost_usd:.4f}"
                break

            # Context management runs before every request: it may clear old tool
            # output or compact the history (which itself costs a model call).
            for event in ctx.prepare(client):
                result.clears += event.kind == "clear"
                result.compactions += event.kind == "compact"
                transcript.write({"event": "context", "step": step, **vars(event)})
                if on_context:
                    on_context(step, event)
            for usage in ctx.extra_usage:
                _add_usage(result, usage, config.model)
            ctx.extra_usage.clear()
            estimated = ctx.estimated_tokens

            response = _create(client, _request(config, ctx))
            result.steps = step
            _add_usage(result, response.usage, config.model)

            # Append the assistant turn exactly as returned (all blocks, not just text).
            ctx.record_response(response.content, response.usage)

            tool_results = _run_tool_calls(response.content, repo_path, config, sandbox, ctx)
            transcript.write(
                {
                    "event": "step",
                    "step": step,
                    "stop_reason": response.stop_reason,
                    "content": [b.model_dump() for b in response.content],
                    "tool_results": tool_results,
                    "usage": response.usage.model_dump(),
                    # Estimated size before sending vs. what the API measured, to see
                    # how good the estimate is (see the walkthrough).
                    "context_estimate": estimated,
                    "context_actual": _request_size(response.usage),
                    "server_edits": _server_edits(response),
                }
            )
            if on_step:
                on_step(step, response, tool_results)

            if response.stop_reason == "tool_use":
                # All results from one assistant turn go back in ONE user message.
                ctx.add_tool_results(tool_results)
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
    except KeyboardInterrupt:
        # Ctrl+C is a normal way to end a run you've seen enough of. Return what it
        # cost so far instead of a traceback; the caller still stops the sandbox.
        result.stop_reason = StopReason.INTERRUPTED
        result.summary = "stopped with Ctrl+C"
    finally:
        result.peak_context_tokens = ctx.peak_tokens
        result.wall_time_s = round(time.monotonic() - started, 2)
        if ctx.notes:
            transcript.write({"event": "notes", "notes": ctx.notes})
        transcript.write({"event": "end", "result": result.model_dump(mode="json")})

    return result


def _request(config: AgentConfig, ctx: ContextManager) -> dict[str, Any]:
    """Build the Messages API arguments for one step."""
    request: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "system": ctx.system,
        "tools": ctx.tools,
        "messages": ctx.wire(),
        # Automatic prompt caching: the API caches the longest stable prefix
        # (tools, system, earlier messages). Every step resends the whole history,
        # so from step 2 on most input tokens are cache reads at ~0.1x price.
        "cache_control": {"type": "ephemeral"},
        **ctx.request_params(),
    }
    if config.effort:
        request["output_config"] = {"effort": config.effort}
    return request


def _create(client: MessagesClient, request: dict[str, Any]) -> Any:
    # Beta features (server-side context editing) live on the beta endpoint.
    if "betas" in request:
        return client.beta.messages.create(**request)
    return client.messages.create(**request)


def _request_size(usage: Any) -> int:
    return (
        usage.input_tokens
        + (usage.cache_read_input_tokens or 0)
        + (usage.cache_creation_input_tokens or 0)
    )


def _server_edits(response: Any) -> list[dict[str, Any]]:
    managed = getattr(response, "context_management", None)
    if not managed:
        return []
    return [edit.model_dump() for edit in managed.applied_edits]


def _over_budget(result: RunResult, config: AgentConfig) -> bool:
    return (
        config.max_cost_usd is not None
        and result.cost_usd is not None
        and result.cost_usd >= config.max_cost_usd
    )


def _run_tool_calls(
    content: list[Any],
    repo_path: Path,
    config: AgentConfig,
    sandbox: CommandRunner | None = None,
    ctx: ContextManager | None = None,
) -> list[dict[str, Any]]:
    results = []
    for block in content:
        if block.type != "tool_use":
            continue
        if block.name == "notes" and ctx is not None and ctx.config.notes:
            # Notes are harness state, not a file in the repo: they live in the
            # context manager so compaction can carry them over.
            output = ToolOutput(*ctx.run_notes(block.input))
        else:
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


def _add_usage(result: RunResult, usage: Any, model: str) -> None:
    # Cached tokens are reported separately from `input_tokens`; count all of them
    # so each kind is priced correctly.
    result.input_tokens += usage.input_tokens
    result.cache_read_tokens += usage.cache_read_input_tokens or 0
    result.cache_write_tokens += usage.cache_creation_input_tokens or 0
    result.output_tokens += usage.output_tokens
    result.cost_usd = cost_usd(
        model,
        result.input_tokens,
        result.cache_read_tokens,
        result.cache_write_tokens,
        result.output_tokens,
    )


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
