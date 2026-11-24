"""Tools the agent can call: the JSON schema the model sees, plus the Python that runs it.

Stage 2 tool set:
    read_file   read a file, optionally a line range
    list_dir    list one directory level
    glob        find files by pattern
    grep        search file contents (ripgrep, with a pure-Python fallback)
    edit_file   replace one exact, unique snippet
    write_file  create or overwrite a whole file
    bash        run a shell command with a timeout and capped output
    run_tests   run pytest (a shortcut: bash could do it, but this is cheaper and uniform)

Everything still runs on the host inside `repo_path`; Stage 4 moves it into a sandbox.
Design rules are explained in docs/milestone-1-walkthrough.md and
docs/stage-2-walkthrough.md:
- Every path is resolved inside `repo_path`; anything that escapes is refused.
- Tools never raise to the loop. Failures come back as ToolOutput(is_error=True).
- Outputs are capped, and the cap is announced.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_OUTPUT_CHARS = 10_000
MAX_LISTING = 200  # max entries from list_dir / glob, max matching lines from grep
TEST_TIMEOUT_S = 120
BASH_DEFAULT_TIMEOUT_S = 60
BASH_MAX_TIMEOUT_S = 300
IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
# Commands bash refuses outright. A speed bump against obvious accidents while bash
# still runs on the host (until the Stage 4 sandbox) - NOT a security boundary: a
# determined or confused model can get around any deny-list.
BLOCKED_COMMANDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brm\s+(-\w*\s+)*(/|~|\$HOME)(\s|/?$|/\*)"), "rm on / or the home dir"),
    (re.compile(r"\b(sudo|su|doas)\b"), "privilege escalation"),
    (re.compile(r"\b(mkfs\w*|fdisk|parted|diskutil)\b"), "disk formatting"),
    (re.compile(r"\bdd\b.*\bof=/dev/"), "raw writes to a device"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"), "power commands"),
    (re.compile(r":\(\)\s*\{.*\};\s*:"), "fork bomb"),
    (re.compile(r"\b(curl|wget)\b[^|]*\|\s*(ba|z)?sh\b"), "piping a download into a shell"),
    (re.compile(r"\bgit\s+push\b"), "git push"),
    (re.compile(r"\bchmod\s+(-\w+\s+)*[0-7]*777\s+/"), "chmod 777 on system paths"),
    (re.compile(r">\s*/dev/sd[a-z]"), "writing to a disk device"),
]

# Environment variables the agent's shell must never see.
SECRET_ENV_PREFIXES = ("ANTHROPIC_", "OPENAI_", "AWS_", "GITHUB_TOKEN", "GH_TOKEN")


@dataclass(frozen=True)
class ToolOutput:
    text: str
    is_error: bool = False


# --- Schemas -------------------------------------------------------------------------
# All schemas are `strict`, so the API guarantees inputs match them. Optional
# arguments are declared as required-but-nullable (`["integer", "null"]`): the model
# must send the key, and sends null when it has nothing to say. That keeps every
# argument explicit and the implementations free of missing-key checks.


def _schema(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


_PATH = {"type": "string", "description": "Path relative to the repo root, e.g. 'src/app.py'."}

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "read_file": _schema(
        "read_file",
        "Read a UTF-8 text file with line numbers. For large files, pass start_line and "
        "end_line (1-based, inclusive) to read only part of it; pass null to read from "
        "the start or to the end.",
        {
            "path": _PATH,
            "start_line": {"type": ["integer", "null"]},
            "end_line": {"type": ["integer", "null"]},
        },
    ),
    "list_dir": _schema(
        "list_dir",
        "List the files and subdirectories directly inside a directory (one level). "
        "Directories end with '/'. Use '.' for the repo root. Start here to get oriented.",
        {"path": _PATH},
    ),
    "glob": _schema(
        "glob",
        "Find files whose path matches a glob pattern, relative to the repo root. "
        "'**' matches any number of directories, e.g. '**/*.py' or 'src/**/test_*.py'.",
        {"pattern": {"type": "string"}},
    ),
    "grep": _schema(
        "grep",
        "Search file contents with a regular expression and return matching lines as "
        "'path:line: text'. Use it to find where a function is defined or used. "
        "`path` limits the search to a file or directory; null searches the whole repo.",
        {
            "pattern": {"type": "string", "description": "Python/ripgrep regex."},
            "path": {"type": ["string", "null"]},
        },
    ),
    "edit_file": _schema(
        "edit_file",
        "Replace one exact snippet in a file. `old_str` must match the file exactly, "
        "including whitespace and indentation, and must occur exactly once; include a few "
        "surrounding lines to make it unique. Prefer this over write_file for changes to "
        "existing files.",
        {
            "path": _PATH,
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
        },
    ),
    "write_file": _schema(
        "write_file",
        "Create a new file, or overwrite an existing one, with exactly `content`. "
        "For small changes to an existing file, use edit_file instead.",
        {"path": _PATH, "content": {"type": "string", "description": "The full file."}},
    ),
    "bash": _schema(
        "bash",
        "Run a shell command in the repo root and return its exit code and output. "
        "Non-interactive only: no editors, no prompts, no servers that never exit. "
        "timeout_s defaults to 60 (max 300); pass null for the default.",
        {
            "command": {"type": "string"},
            "timeout_s": {"type": ["integer", "null"]},
        },
    ),
    "run_tests": _schema(
        "run_tests",
        "Run the repository's pytest suite and return the exit code and output. "
        "Run it after every change and before you finish.",
        {},
    ),
}

DEFAULT_TOOLS = sorted(TOOL_SCHEMAS)


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


def _rel(repo_path: Path, path: Path) -> str:
    return path.relative_to(repo_path.resolve()).as_posix()


def _is_ignored(repo_path: Path, path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.relative_to(repo_path.resolve()).parts)


def _numbered(lines: list[str], first_line_no: int) -> str:
    return "\n".join(f"{i:>4} | {line}" for i, line in enumerate(lines, start=first_line_no))


def _capped(items: list[str], what: str) -> str:
    if len(items) <= MAX_LISTING:
        return "\n".join(items)
    shown = "\n".join(items[:MAX_LISTING])
    return f"{shown}\n[... {len(items) - MAX_LISTING} more {what} not shown; narrow the search]"


# --- Read and find -------------------------------------------------------------------


def read_file(
    repo_path: Path, path: str, start_line: int | None, end_line: int | None
) -> ToolOutput:
    target = safe_path(repo_path, path)
    if not target.is_file():
        return ToolOutput(f"No such file: {path}", is_error=True)
    lines = target.read_text(encoding="utf-8").splitlines()
    if not lines:
        return ToolOutput("(empty file)")
    start = max(start_line or 1, 1)
    end = min(end_line or len(lines), len(lines))
    if start > end:
        return ToolOutput(
            f"Empty range {start}-{end}; the file has {len(lines)} lines.", is_error=True
        )
    body = _numbered(lines[start - 1 : end], start)
    header = f"{path} (lines {start}-{end} of {len(lines)})\n"
    return ToolOutput(header + truncate(body))


def list_dir(repo_path: Path, path: str) -> ToolOutput:
    target = safe_path(repo_path, path)
    if not target.is_dir():
        return ToolOutput(f"Not a directory: {path}", is_error=True)
    entries = sorted(
        f"{p.name}/" if p.is_dir() else p.name
        for p in target.iterdir()
        if p.name not in IGNORED_DIRS
    )
    return ToolOutput(_capped(entries, "entries") or "(empty directory)")


def glob(repo_path: Path, pattern: str) -> ToolOutput:
    if pattern.startswith("/") or ".." in Path(pattern).parts:
        return ToolOutput("Pattern must be relative to the repo root.", is_error=True)
    root = repo_path.resolve()
    matches = sorted(
        _rel(repo_path, p)
        for p in root.glob(pattern)
        if p.is_file() and not _is_ignored(repo_path, p)
    )
    return ToolOutput(_capped(matches, "files") or f"No files match {pattern}")


def grep(repo_path: Path, pattern: str, path: str | None) -> ToolOutput:
    target = safe_path(repo_path, path or ".")
    if not target.exists():
        return ToolOutput(f"No such path: {path}", is_error=True)
    try:
        re.compile(pattern)
    except re.error as exc:
        return ToolOutput(f"Invalid regex: {exc}", is_error=True)
    if shutil.which("rg"):
        lines = _grep_ripgrep(repo_path, pattern, target)
    else:
        lines = list(_grep_python(repo_path, pattern, target))
    return ToolOutput(_capped(lines, "matching lines") or f"No matches for {pattern!r}")


def _grep_ripgrep(repo_path: Path, pattern: str, target: Path) -> list[str]:
    # ripgrep is much faster on big repos and respects .gitignore. Output paths are
    # relative to the repo root because we run it from there.
    cmd = [
        "rg",
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color",
        "never",
        "--sort",
        "path",
    ]
    for d in sorted(IGNORED_DIRS):
        cmd += ["--glob", f"!{d}"]
    cmd += ["-e", pattern, "--", target.relative_to(repo_path.resolve()).as_posix() or "."]
    proc = subprocess.run(
        cmd, cwd=repo_path, capture_output=True, text=True, timeout=30, check=False
    )
    # rg exits 1 when nothing matched; that is not an error.
    return [_format_rg_line(line) for line in proc.stdout.splitlines()]


def _format_rg_line(line: str) -> str:
    path, line_no, text = line.split(":", 2)
    return f"{path.removeprefix('./')}:{line_no}: {text.strip()}"


def _grep_python(repo_path: Path, pattern: str, target: Path) -> Iterator[str]:
    regex = re.compile(pattern)
    files = [target] if target.is_file() else sorted(target.rglob("*"))
    for file in files:
        if not file.is_file() or _is_ignored(repo_path, file):
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: skip, like ripgrep does
        for no, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                yield f"{_rel(repo_path, file)}:{no}: {line.strip()}"


# --- Change --------------------------------------------------------------------------


def edit_file(repo_path: Path, path: str, old_str: str, new_str: str) -> ToolOutput:
    target = safe_path(repo_path, path)
    if not target.is_file():
        return ToolOutput(f"No such file: {path}", is_error=True)
    if not old_str:
        return ToolOutput("old_str is empty; use write_file to create files.", is_error=True)
    if old_str == new_str:
        return ToolOutput("old_str and new_str are identical; nothing to do.", is_error=True)
    text = target.read_text(encoding="utf-8")
    count = text.count(old_str)
    if count == 0:
        return ToolOutput(
            f"old_str not found in {path}. It must match exactly, including whitespace. "
            "Read the file again and copy the snippet.",
            is_error=True,
        )
    if count > 1:
        lines = [text[: m.start()].count("\n") + 1 for m in re.finditer(re.escape(old_str), text)]
        return ToolOutput(
            f"old_str occurs {count} times in {path} (lines {lines}). "
            "Include more surrounding lines so it matches exactly once.",
            is_error=True,
        )
    start = text.index(old_str)
    new_text = text[:start] + new_str + text[start + len(old_str) :]
    target.write_text(new_text, encoding="utf-8")
    # Show the edited region so the model can check its own change without a re-read.
    first = text[:start].count("\n") + 1
    new_lines = new_text.splitlines()
    lo = max(first - 3, 1)
    hi = min(first + new_str.count("\n") + 3, len(new_lines))
    snippet = _numbered(new_lines[lo - 1 : hi], lo)
    return ToolOutput(f"Edited {path}. Lines {lo}-{hi} now read:\n{snippet}")


def write_file(repo_path: Path, path: str, content: str) -> ToolOutput:
    target = safe_path(repo_path, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    target.write_text(content, encoding="utf-8")
    verb = "Overwrote" if existed else "Created"
    return ToolOutput(f"{verb} {path} ({len(content)} characters)")


# --- Run -----------------------------------------------------------------------------


def _clean_env() -> dict[str, str]:
    """The parent environment minus secrets and minus miniagent's own virtualenv.

    Secrets: the agent's commands never see API keys.
    VIRTUAL_ENV and PATH: without this, `uv run` or `python` inside a target repo would pick up
    miniagent's venv instead of the repo's own environment.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(SECRET_ENV_PREFIXES) and k != "VIRTUAL_ENV"
    }
    # `uv run miniagent` also puts miniagent's venv/bin first on PATH; drop it so a bare
    # `python` or `pytest` in the target repo resolves to that repo's tools.
    own_bin = str(Path(sys.prefix) / "bin")
    if sys.prefix != sys.base_prefix and "PATH" in env:
        env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep) if p != own_bin)
    return env


def _run(cmd: list[str] | str, repo_path: Path, timeout_s: int, shell: bool) -> ToolOutput:
    # start_new_session puts the command in its own process group, so on timeout we can
    # kill everything it spawned, not just the shell that started it.
    proc = subprocess.Popen(
        cmd,
        cwd=repo_path,
        shell=shell,
        env=_clean_env(),
        stdin=subprocess.DEVNULL,  # anything that waits for input fails fast instead
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # one stream, in the order things happened
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()
        return ToolOutput(
            f"Timed out after {timeout_s}s and was killed.\n{truncate(output or '')}",
            is_error=True,
        )
    return ToolOutput(f"exit code: {proc.returncode}\n{truncate(output)}")


def blocked_reason(command: str) -> str | None:
    for pattern, reason in BLOCKED_COMMANDS:
        if pattern.search(command):
            return reason
    return None


def bash(repo_path: Path, command: str, timeout_s: int | None) -> ToolOutput:
    reason = blocked_reason(command)
    if reason:
        return ToolOutput(
            f"Blocked by miniagent's command guard ({reason}). Work inside the repo with "
            "the file tools, or use a safer command.",
            is_error=True,
        )
    timeout = min(max(timeout_s or BASH_DEFAULT_TIMEOUT_S, 1), BASH_MAX_TIMEOUT_S)
    # A non-zero exit code is a result, not a tool error: the command ran.
    return _run(command, repo_path, timeout, shell=True)


def run_tests(
    repo_path: Path, test_command: str | None = None, timeout_s: int = TEST_TIMEOUT_S
) -> ToolOutput:
    """Run the test suite. With no `test_command`, use pytest under miniagent's own
    interpreter (right for benchmark tasks, which share miniagent's environment).
    For a real repo, pass that repo's own command, e.g. 'uv run pytest'."""
    if test_command:
        return _run(test_command, repo_path, timeout_s, shell=True)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    return _run(cmd, repo_path, timeout_s, shell=False)


_IMPLEMENTATIONS: dict[str, Callable[..., ToolOutput]] = {
    "read_file": read_file,
    "list_dir": list_dir,
    "glob": glob,
    "grep": grep,
    "edit_file": edit_file,
    "write_file": write_file,
    "bash": bash,
    "run_tests": run_tests,
}


def execute_tool(
    name: str,
    tool_input: dict[str, Any],
    repo_path: Path,
    *,
    test_command: str | None = None,
    test_timeout_s: int = TEST_TIMEOUT_S,
) -> ToolOutput:
    """Run one tool call. Never raises: every failure becomes an error ToolOutput.

    `test_command` / `test_timeout_s` are run settings, not model inputs: the model
    calls run_tests with no arguments, and the harness decides how tests run.
    """
    impl = _IMPLEMENTATIONS.get(name)
    if impl is None:
        return ToolOutput(f"Unknown tool: {name}", is_error=True)
    try:
        if name == "run_tests":
            return run_tests(repo_path, test_command, test_timeout_s)
        return impl(repo_path, **tool_input)
    except (ValueError, OSError, UnicodeDecodeError, TypeError, subprocess.SubprocessError) as exc:
        return ToolOutput(f"{type(exc).__name__}: {exc}", is_error=True)
