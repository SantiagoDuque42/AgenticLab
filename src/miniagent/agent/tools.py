"""Tools the agent can call: the JSON schema the model sees, plus the Python that runs it.

Milestone 1 has three tools: read_file, write_file, run_tests. They run directly on the
host inside `repo_path`. Milestone 2 moves them into a Docker sandbox, which is why every
tool takes `repo_path` explicitly instead of using the current working directory.

Design rules (explained in docs/milestone-1-walkthrough.md):
- Every path is resolved inside `repo_path`; anything that escapes is refused.
- Tools never raise to the loop. Failures come back as a ToolOutput with is_error=True,
  so the model can read the error and try something else.
- Long outputs are truncated, and the truncation is announced, so one noisy test log
  cannot blow the context window or the budget.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_OUTPUT_CHARS = 10_000
TEST_TIMEOUT_S = 120


@dataclass(frozen=True)
class ToolOutput:
    text: str
    is_error: bool = False


# --- Schemas -------------------------------------------------------------------------
# `strict: True` makes the API guarantee `input` matches the schema, so the
# implementations below can index input["path"] without defensive checks.
# Strict mode requires `additionalProperties: false` and an explicit `required` list.

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "read_file": {
        "name": "read_file",
        "description": (
            "Read a UTF-8 text file from the repository and return its contents with "
            "line numbers. `path` is relative to the repository root, e.g. 'src/app.py'. "
            "Use it to inspect code before changing it."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the repo root."}
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    "write_file": {
        "name": "write_file",
        "description": (
            "Create or overwrite a file in the repository with exactly `content`. "
            "Send the complete new file, not a diff. Parent directories are created."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the repo root."},
                "content": {"type": "string", "description": "The full new file content."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    "run_tests": {
        "name": "run_tests",
        "description": (
            "Run the repository's pytest suite and return the exit code and output. "
            "Run it after every change and before you finish."
        ),
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}


def tool_schemas(names: list[str]) -> list[dict[str, Any]]:
    """Schemas for the enabled tools, in a fixed order so the prompt prefix stays cacheable."""
    return [TOOL_SCHEMAS[n] for n in sorted(names)]


# --- Helpers -------------------------------------------------------------------------


def safe_path(repo_path: Path, relative: str) -> Path:
    """Resolve `relative` inside `repo_path`, refusing paths that escape it."""
    root = repo_path.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"path escapes the repository: {relative}")
    return target


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the start and the end, since test failures summarise at the bottom."""
    if len(text) <= limit:
        return text
    half = limit // 2
    dropped = len(text) - limit
    return f"{text[:half]}\n\n[... {dropped} characters truncated ...]\n\n{text[-half:]}"


# --- Implementations -----------------------------------------------------------------


def read_file(repo_path: Path, path: str) -> ToolOutput:
    target = safe_path(repo_path, path)
    if not target.is_file():
        return ToolOutput(f"No such file: {path}", is_error=True)
    lines = target.read_text(encoding="utf-8").splitlines()
    numbered = "\n".join(f"{i:>4} | {line}" for i, line in enumerate(lines, start=1))
    return ToolOutput(truncate(numbered) or "(empty file)")


def write_file(repo_path: Path, path: str, content: str) -> ToolOutput:
    target = safe_path(repo_path, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return ToolOutput(f"Wrote {len(content)} characters to {path}")


def run_tests(repo_path: Path) -> ToolOutput:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return ToolOutput(f"Tests timed out after {TEST_TIMEOUT_S}s", is_error=True)
    output = truncate(proc.stdout + proc.stderr)
    # A failing test run is NOT a tool error: the tool worked and is reporting a result.
    return ToolOutput(f"exit code: {proc.returncode}\n{output}")


_IMPLEMENTATIONS: dict[str, Callable[..., ToolOutput]] = {
    "read_file": read_file,
    "write_file": write_file,
    "run_tests": run_tests,
}


def execute_tool(name: str, tool_input: dict[str, Any], repo_path: Path) -> ToolOutput:
    """Run one tool call. Never raises: every failure becomes an error ToolOutput."""
    impl = _IMPLEMENTATIONS.get(name)
    if impl is None:
        return ToolOutput(f"Unknown tool: {name}", is_error=True)
    try:
        return impl(repo_path, **tool_input)
    except (ValueError, OSError, UnicodeDecodeError, TypeError) as exc:
        return ToolOutput(f"{type(exc).__name__}: {exc}", is_error=True)
