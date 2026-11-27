"""Context management: keep long runs inside a token budget.

Every step resends the whole conversation, so its size is the run's main cost and
its main risk: past some point a run gets slow, expensive, and finally doesn't fit.
`ContextManager` owns the message list for the loop and, before each request, lets
a list of strategies act on it:

- `ClearToolResults`: client-side. Old tool outputs (a 300-line file read twenty
  steps ago) are replaced with a one-line placeholder. Cheap, keeps every decision
  the model made, loses only raw output it can fetch again.
- `ServerClearToolResults`: the same idea done by the API (context editing, beta),
  sent as request parameters. Our copy of the history stays untouched.
- `Compact`: when the history is still too big, ask the model for a summary of the
  run so far and restart the conversation from the task plus that summary.

Two more pieces of context live here because they are about what the model sees:
the agent's notes (a scratchpad tool whose contents survive compaction) and the
repo's own instructions file (AGENTS.md / CLAUDE.md), added to the system prompt
once at the start.

Read docs/stage-3-walkthrough.md alongside this file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

# A rough rule for code and English: one token per ~3.5 characters. Only used to
# estimate what was appended since the last response; the API's usage numbers
# correct the estimate on every step.
CHARS_PER_TOKEN = 3.5

PROJECT_INSTRUCTION_FILES = ("AGENTS.md", "CLAUDE.md", "MINIAGENT.md")
MAX_INSTRUCTIONS_CHARS = 8000

CONTEXT_EDITING_BETA = "context-management-2025-06-27"

NOTES_TOOL: dict[str, Any] = {
    "name": "notes",
    "description": (
        "Your scratchpad for this task. It survives when older conversation is "
        "summarized away, so keep the facts you'll need later here: the plan, what you "
        "tried and the result, numbers, file names, open questions. action='read' "
        "returns the notes; 'append' adds content as a new line; 'write' replaces them."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["read", "append", "write"]},
            "content": {"type": ["string", "null"], "description": "Text for append/write."},
        },
        "required": ["action", "content"],
        "additionalProperties": False,
    },
}

SUMMARY_PROMPT = """\
Stop working for a moment and write a summary of this task so far. The conversation \
above will be deleted and replaced by your summary, so it is all you will have to \
continue from. Do not call any tools. Include:

1. What the task is and what "done" means.
2. What you have done: files read and changed (with paths), commands run, results.
3. The current state: what works, what fails, the latest test or metric numbers.
4. Decisions made and approaches tried that did not work, and why.
5. Exactly what to do next.

Be specific: exact paths, function names, numbers and error messages. Leave out \
anything you can easily look up again by reading a file."""


class ContextConfig(BaseModel):
    """Thresholds are in tokens of context (the size of the next request). None = off."""

    clear_tool_results_at: int | None = 60_000
    keep_tool_results: int = 3  # the most recent N tool results are never cleared
    server_clear_at: int | None = None  # use the API's context editing instead
    compact_at: int | None = 120_000
    notes: bool = False
    project_instructions: bool = True
    count_tokens: str = "estimate"  # "estimate" (free, local) or "api" (exact, 1 call)


@dataclass
class ContextEvent:
    """Something a strategy did, for the transcript and the run summary."""

    kind: str  # "clear", "compact", "compact_failed"
    tokens_before: int
    tokens_after: int
    detail: dict[str, Any] = field(default_factory=dict)


class Strategy(Protocol):
    name: str

    def request_params(self) -> dict[str, Any]:
        """Extra Messages API parameters (server-side strategies)."""
        ...

    def apply(self, ctx: ContextManager, client: Any) -> ContextEvent | None:
        """Change ctx.messages if needed (client-side strategies)."""
        ...


# --- Strategies ------------------------------------------------------------------


@dataclass
class ClearToolResults:
    """Replace old tool outputs with a placeholder once the context passes `trigger`.

    Clears everything older than the last `keep` results in one go, so it fires
    rarely: each firing changes an early part of the history, which costs one
    prompt-cache miss from that point on.
    """

    trigger: int
    keep: int = 3
    name: str = "clear_tool_results"

    def request_params(self) -> dict[str, Any]:
        return {}

    def apply(self, ctx: ContextManager, client: Any) -> ContextEvent | None:
        if ctx.estimated_tokens < self.trigger:
            return None
        results = [
            block
            for message in ctx.messages
            if message["role"] == "user" and isinstance(message["content"], list)
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        old = [b for b in results[: max(len(results) - self.keep, 0)] if not b.get("_cleared")]
        if not old:
            return None
        before = ctx.estimated_tokens
        removed = 0
        for block in old:
            text = block["content"] if isinstance(block["content"], str) else ""
            placeholder = (
                f"[output cleared to save context: {len(text)} characters. "
                "Run the tool again if you need it.]"
            )
            removed += len(text) - len(placeholder)
            block["content"] = placeholder
            block["_cleared"] = True  # stripped before sending, see ContextManager.wire()
        ctx.strip_thinking()
        ctx.estimated_tokens = max(before - int(removed / CHARS_PER_TOKEN), 0)
        return ContextEvent("clear", before, ctx.estimated_tokens, {"cleared_results": len(old)})


@dataclass
class ServerClearToolResults:
    """The API's own tool-result clearing (context editing, beta).

    The server edits its copy of the conversation before the model sees it; ours is
    sent unchanged, so this needs no thinking-block handling at all.
    """

    trigger: int
    keep: int = 3
    name: str = "server_clear_tool_results"

    def request_params(self) -> dict[str, Any]:
        edit = {
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": self.trigger},
            "keep": {"type": "tool_uses", "value": self.keep},
            # Only clear when it's worth a cache miss.
            "clear_at_least": {"type": "input_tokens", "value": min(10_000, self.trigger // 4)},
        }
        return {"betas": [CONTEXT_EDITING_BETA], "context_management": {"edits": [edit]}}

    def apply(self, ctx: ContextManager, client: Any) -> ContextEvent | None:
        return None


@dataclass
class Compact:
    """Summarize the run and restart from task + summary once the context passes
    `trigger`. "Simple compaction": nothing from the old history is replayed, so no
    old thinking block can be invalidated, and the next request starts clean."""

    trigger: int
    max_summary_tokens: int = 4000
    name: str = "compact"

    def request_params(self) -> dict[str, Any]:
        return {}

    def apply(self, ctx: ContextManager, client: Any) -> ContextEvent | None:
        if ctx.estimated_tokens < self.trigger or len(ctx.messages) < 3:
            return None
        before = ctx.estimated_tokens
        try:
            summary, usage = ctx.summarize(client, self.max_summary_tokens)
        except Exception as exc:  # a failed summary must not end the run
            return ContextEvent("compact_failed", before, before, {"error": repr(exc)})
        ctx.restart_from_summary(summary)
        return ContextEvent(
            "compact",
            before,
            ctx.estimated_tokens,
            {"summary": summary, "summary_usage": usage},
        )


# --- The manager -----------------------------------------------------------------


class ContextManager:
    """Owns the conversation the loop sends, and the strategies that keep it small."""

    def __init__(
        self,
        task_prompt: str,
        config: ContextConfig,
        *,
        model: str,
        system_prompt: str,
        tools: list[dict[str, Any]],
        repo_path: Path | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.task_prompt = task_prompt
        self.system_prompt = system_prompt + _project_instructions(repo_path, config)
        self.notes = ""
        self.tools = tools + ([NOTES_TOOL] if config.notes else [])
        self.messages: list[dict[str, Any]] = [{"role": "user", "content": task_prompt}]
        self.estimated_tokens = 0
        self.peak_tokens = 0
        self.extra_usage: list[Any] = []  # usage of summarizer calls, billed too
        self.strategies: list[Strategy] = []
        if config.server_clear_at:
            self.strategies.append(
                ServerClearToolResults(config.server_clear_at, config.keep_tool_results)
            )
        if config.clear_tool_results_at:
            self.strategies.append(
                ClearToolResults(config.clear_tool_results_at, config.keep_tool_results)
            )
        if config.compact_at:
            self.strategies.append(Compact(config.compact_at))

    # The request ---------------------------------------------------------------

    @property
    def system(self) -> list[dict[str, Any]]:
        # A cache breakpoint on the system prompt (tools render before it, so they are
        # included). The top-level automatic breakpoint moves with the conversation;
        # this one stays, so after a compaction tools + system are still cache hits.
        return [
            {"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}
        ]

    def request_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for strategy in self.strategies:
            for key, value in strategy.request_params().items():
                if key == "betas":
                    params.setdefault("betas", []).extend(value)
                else:
                    params[key] = value
        return params

    def wire(self) -> list[dict[str, Any]]:
        """The messages as sent: our bookkeeping keys (leading `_`) removed."""
        return [
            {
                "role": m["role"],
                "content": [
                    {k: v for k, v in b.items() if not k.startswith("_")}
                    if isinstance(b, dict)
                    else b
                    for b in m["content"]
                ]
                if isinstance(m["content"], list)
                else m["content"],
            }
            for m in self.messages
        ]

    def prepare(self, client: Any) -> list[ContextEvent]:
        """Run before each request: measure, then let each strategy act."""
        if self.config.count_tokens == "api":
            self.estimated_tokens = self.count_exact(client) or self.estimated_tokens
        events = []
        for strategy in self.strategies:
            event = strategy.apply(self, client)
            if event:
                events.append(event)
        self.peak_tokens = max(self.peak_tokens, self.estimated_tokens)
        return events

    # Bookkeeping after each response --------------------------------------------

    def record_response(self, content: list[Any], usage: Any) -> None:
        """The API reports how big the request was; that replaces our estimate."""
        self.messages.append({"role": "assistant", "content": content})
        self.estimated_tokens = (
            usage.input_tokens
            + (usage.cache_read_input_tokens or 0)
            + (usage.cache_creation_input_tokens or 0)
            + usage.output_tokens
        )
        self.peak_tokens = max(self.peak_tokens, self.estimated_tokens)

    def add_tool_results(self, results: list[dict[str, Any]]) -> None:
        self.messages.append({"role": "user", "content": results})
        chars = sum(len(json.dumps(r)) for r in results)
        self.estimated_tokens += int(chars / CHARS_PER_TOKEN)

    def count_exact(self, client: Any) -> int | None:
        """Exact size of the next request, from the free token-counting endpoint."""
        try:
            response = client.messages.count_tokens(
                model=self.model, system=self.system, tools=self.tools, messages=self.wire()
            )
        except Exception:  # fake clients, network trouble: keep the estimate
            return None
        return response.input_tokens

    # Editing helpers used by strategies -----------------------------------------

    def strip_thinking(self) -> None:
        """Remove every thinking block from the history.

        Newer models sign each thinking block against the exact conversation before
        it; edit an earlier turn and every later block is rejected (a 400 on newer
        accounts). Removing *all* thinking blocks is allowed, so a client-side edit
        does that at the same time. The model loses its old private reasoning, not
        its visible actions or results.
        """
        for message in self.messages:
            if message["role"] == "assistant" and isinstance(message["content"], list):
                message["content"] = [
                    b
                    for b in message["content"]
                    if _block_type(b) not in ("thinking", "redacted_thinking")
                ]

    def summarize(self, client: Any, max_tokens: int) -> tuple[str, dict[str, Any]]:
        """Ask the model to summarize the run. Same model, system and tools as the
        run itself, so the request is a near-complete prompt-cache hit."""
        messages = self.wire()
        last = messages[-1]
        ask = {"type": "text", "text": SUMMARY_PROMPT}
        if last["role"] == "user" and isinstance(last["content"], list):
            messages[-1] = {"role": "user", "content": [*last["content"], ask]}
        else:
            messages.append({"role": "user", "content": [ask]})
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system,
            tools=self.tools,
            tool_choice={"type": "none"},
            messages=messages,
            cache_control={"type": "ephemeral"},
        )
        self.extra_usage.append(response.usage)
        text = "\n".join(b.text for b in response.content if _block_type(b) == "text").strip()
        if not text:
            raise ValueError("empty summary")
        return text, response.usage.model_dump()

    def restart_from_summary(self, summary: str) -> None:
        parts = [
            self.task_prompt,
            "\n\n---\nYou have already worked on this task. The earlier conversation was "
            "summarized to save context; here is that summary:\n\n<summary>\n"
            f"{summary}\n</summary>",
        ]
        if self.notes:
            parts.append(f"\n\nYour notes so far:\n<notes>\n{self.notes}\n</notes>")
        parts.append("\n\nContinue the task from where you left off.")
        self.messages = [{"role": "user", "content": "".join(parts)}]
        self.estimated_tokens = int(
            (len(self.system_prompt) + len(json.dumps(self.tools)) + len("".join(parts)))
            / CHARS_PER_TOKEN
        )

    # The notes tool ----------------------------------------------------------------

    def run_notes(self, tool_input: dict[str, Any]) -> tuple[str, bool]:
        action, content = tool_input.get("action"), tool_input.get("content") or ""
        if action == "read":
            return (self.notes or "(no notes yet)"), False
        if action == "append":
            self.notes = f"{self.notes}\n{content}".strip()
            return f"Appended. Notes are {len(self.notes)} characters.", False
        if action == "write":
            self.notes = content.strip()
            return f"Notes replaced ({len(self.notes)} characters).", False
        return f"Unknown action {action!r}; use read, append or write.", True


def _block_type(block: Any) -> str | None:
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _project_instructions(repo_path: Path | None, config: ContextConfig) -> str:
    """The repo's own instructions for agents, if it has a file for them."""
    if not (repo_path and config.project_instructions):
        return ""
    for name in PROJECT_INSTRUCTION_FILES:
        path = repo_path / name
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > MAX_INSTRUCTIONS_CHARS:
                text = text[:MAX_INSTRUCTIONS_CHARS] + "\n[... truncated]"
            return (
                f"\n\nThe repository has instructions for agents in {name}. Follow them "
                f"unless the task says otherwise:\n<project_instructions>\n{text}\n"
                "</project_instructions>"
            )
    return ""
