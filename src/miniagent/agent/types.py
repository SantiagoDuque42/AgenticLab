"""Data types shared by the agent, the harness and (later) the orchestrator.

These are the seams the multi-agent orchestrator will build on, so keep them
free of global state: an agent run is fully described by its inputs and its
RunResult.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from miniagent.agent.context import ContextConfig
from miniagent.agent.tools import DEFAULT_TOOLS


class StopReason(StrEnum):
    FINISHED = "finished"  # the model called `finish` or stopped asking for tools
    STEP_LIMIT = "step_limit"
    TIMEOUT = "timeout"
    BUDGET = "budget"  # max_cost_usd reached
    INTERRUPTED = "interrupted"  # Ctrl+C
    ERROR = "error"


class AgentConfig(BaseModel):
    """Everything that varies between runs. A planner, coder and reviewer in the
    orchestrator are the same loop with different configs."""

    model: str
    role: str = "coder"
    system_prompt: str = (
        "You are a careful software engineer working in a Python repository. "
        "Get oriented first: list the repo, then use grep and glob to find the code "
        "the task is about, and read it before changing it. Make the smallest change "
        "that solves the task, using edit_file for existing files. Run the tests after "
        "each change. When the tests pass and the task is done, reply with a short "
        "summary of what you changed and stop calling tools."
    )
    tools: list[str] = Field(default_factory=lambda: list(DEFAULT_TOOLS))
    max_steps: int = 30
    max_tokens: int = 16000
    effort: str | None = None  # "low" | "medium" | "high" ...; None = the model's default
    timeout_s: int = 600
    # How run_tests runs. None = pytest under miniagent's interpreter (benchmark tasks).
    # For a real repo use its own command, e.g. "uv run pytest".
    test_command: str | None = None
    test_timeout_s: int = 120
    # True for benchmark tasks: their commands run in miniagent's own Python env
    # (where the task's libraries, e.g. numpy, are installed). False for real repos,
    # whose commands must use the repo's own environment.
    share_python_env: bool = False
    # How the conversation is kept small on long runs (see agent/context.py).
    context: ContextConfig = Field(default_factory=ContextConfig)
    # Stop before the next model call once the run has cost this much. None = no cap.
    max_cost_usd: float | None = None


class RunResult(BaseModel):
    """What one agent run produced. The harness grades it; the orchestrator
    merges or retries based on it."""

    task_id: str
    repo_path: Path
    stop_reason: StopReason
    steps: int = 0
    input_tokens: int = 0  # uncached input, full price
    cache_read_tokens: int = 0  # input served from the prompt cache, ~0.1x price
    cache_write_tokens: int = 0  # input written to the cache, ~1.25x price
    output_tokens: int = 0
    cost_usd: float | None = None  # None when the model isn't in the price table
    peak_context_tokens: int = 0  # largest request the run sent
    clears: int = 0  # times old tool results were cleared
    compactions: int = 0  # times the history was summarized
    wall_time_s: float = 0.0
    summary: str = ""
    transcript_path: Path | None = None

    @property
    def cache_hit_rate(self) -> float:
        """Share of input tokens served from the prompt cache."""
        total = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / total if total else 0.0
