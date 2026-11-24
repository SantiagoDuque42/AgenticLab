"""Tool schemas and implementations. Milestone 1: read_file, write_file, run_tests.

Each tool is a JSON schema the model sees plus a Python function that runs it.
In milestone 2 the implementations move into a Docker sandbox; keep every tool
taking `repo_path` explicitly so that move is mechanical.

Rules that save debugging time later:
- Resolve every path against `repo_path` and refuse anything that escapes it.
- Return errors as tool output text, not exceptions, so the model can recover.
- Truncate long outputs (test logs!) and say that you truncated them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "read_file": {
        "name": "read_file",
        "description": "Read a text file from the repository. Path is relative to the repo root.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "write_file": {
        "name": "write_file",
        "description": "Overwrite a file in the repository with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    "run_tests": {
        "name": "run_tests",
        "description": "Run the repository's pytest suite and return its output.",
        "input_schema": {"type": "object", "properties": {}},
    },
}


def tool_schemas(names: list[str]) -> list[dict[str, Any]]:
    return [TOOL_SCHEMAS[n] for n in names]


def safe_path(repo_path: Path, relative: str) -> Path:
    """Resolve `relative` inside `repo_path`, refusing paths that escape it."""
    root = repo_path.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"path escapes the repository: {relative}")
    return target
