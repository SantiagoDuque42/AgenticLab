"""Data types shared by the agent, the harness and (later) the orchestrator.

These are the seams the multi-agent orchestrator will build on, so keep them
free of global state: an agent run is fully described by its inputs and its
RunResult.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class StopReason(StrEnum):
    FINISHED = "finished"  # the model called `finish` or stopped asking for tools
    STEP_LIMIT = "step_limit"
    TIMEOUT = "timeout"
    ERROR = "error"


class AgentConfig(BaseModel):
    """Everything that varies between runs. A planner, coder and reviewer in the
    orchestrator are the same loop with different configs."""

    model: str
    role: str = "coder"
    system_prompt: str = (
        "You are a careful software engineer working in a Python repository. "
        "Read the relevant code before changing it, make the smallest change that "
        "solves the task, and run the tests after each change. When the tests pass "
        "and the task is done, reply with a short summary of what you changed and "
        "stop calling tools."
    )
    tools: list[str] = Field(default_factory=lambda: ["read_file", "write_file", "run_tests"])
    max_steps: int = 30
    max_tokens: int = 16000
    effort: str | None = None  # "low" | "medium" | "high" ...; None = the model's default
    timeout_s: int = 600


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
    wall_time_s: float = 0.0
    summary: str = ""
    transcript_path: Path | None = None
