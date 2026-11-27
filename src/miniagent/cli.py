"""Command line entry point: `uv run miniagent --help`."""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from miniagent.agent import AgentConfig
from miniagent.agent.context import ContextConfig
from miniagent.agent.loop import run_agent
from miniagent.agent.tools import TOOL_SCHEMAS
from miniagent.harness.grade import grade
from miniagent.harness.tasks import RUNS_DIR, TASKS_DIR, load_task, load_tasks, prepare_workspace
from miniagent.sandbox import DockerSandbox, SandboxError, docker_available
from miniagent.workspace import create_workspace, default_test_command, write_patch

app = typer.Typer(no_args_is_help=True)
console = Console()

WSL_HINT = (
    "miniagent runs on Linux, macOS or WSL, not native Windows: its safety layers "
    "(the read-only git shim, process-group kill on timeout, a POSIX shell for `bash`) "
    "need a POSIX system. On Windows, run it inside WSL: `wsl --install -d Ubuntu`, "
    "then use it from the Ubuntu shell with your repo at /mnt/c/..."
)


def require_posix() -> None:
    """Refuse to start on native Windows rather than run with weaker safety guarantees."""
    if os.name != "posix":
        console.print(f"[red]{WSL_HINT}[/red]")
        raise typer.Exit(code=2)


@app.callback()
def _main() -> None:
    """miniagent: a small coding agent with its own eval harness."""
    require_posix()


@app.command()
def tasks() -> None:
    """List benchmark tasks."""
    table = Table("id", "tags", "max steps")
    for t in load_tasks():
        table.add_row(t.id, ", ".join(t.tags), str(t.max_steps))
    console.print(table)


@app.command("grade")
def grade_cmd(task_id: str, repo: Path) -> None:
    """Grade an existing repo directory against a task's hidden tests."""
    result = grade(load_task(TASKS_DIR / task_id), repo)
    console.print(result.output)
    for name, value in result.metrics.items():
        console.print(f"{name}: {value:.3f}")
    console.print("[green]PASS[/green]" if result.passed else "[red]FAIL[/red]")
    raise typer.Exit(0 if result.passed else 1)


_MODEL = typer.Option(None, envvar="MINIAGENT_MODEL", help="Claude model id.")
_EFFORT = typer.Option(None, help="low | medium | high; default: the model's.")
_TOOLS = typer.Option(
    None, help="Comma-separated tool names to enable, e.g. 'read_file,edit_file,bash'."
)
_SANDBOX = typer.Option(
    "docker",
    help="Where shell commands and tests run: 'docker' (a container that only sees the "
    "workspace) or 'none' (directly on this machine, with the guard rails only).",
)


# Context management (docs/stage-3-walkthrough.md). Thresholds are tokens of context;
# 0 turns a strategy off.
_CLEAR_AT = typer.Option(60_000, help="Clear old tool outputs above this many tokens.")
_SERVER_CLEAR_AT = typer.Option(
    0, help="Let the API clear old tool outputs (context editing beta) above this many tokens."
)
_COMPACT_AT = typer.Option(120_000, help="Summarize the history above this many tokens.")
_NOTES = typer.Option(False, help="Give the agent a notes tool that survives compaction.")
_COUNT_TOKENS = typer.Option(
    "estimate", help="'estimate' (free, from usage) or 'api' (exact count before each call)."
)
_MAX_COST = typer.Option(None, help="Stop the run once it has cost this many USD.")


def _context(
    clear_at: int, server_clear_at: int, compact_at: int, notes: bool, count_tokens: str
) -> ContextConfig:
    if count_tokens not in ("estimate", "api"):
        raise typer.BadParameter("--count-tokens must be 'estimate' or 'api'")
    return ContextConfig(
        clear_tool_results_at=clear_at or None,
        server_clear_at=server_clear_at or None,
        compact_at=compact_at or None,
        notes=notes,
        count_tokens=count_tokens,
    )


def _check_sandbox(sandbox: str) -> None:
    if sandbox not in ("docker", "none"):
        raise typer.BadParameter("--sandbox must be 'docker' or 'none'")
    if sandbox == "docker" and not docker_available():
        raise typer.BadParameter(
            "Docker isn't reachable (`docker info` failed). Start Docker, or run with "
            "--sandbox none to use the guard rails only."
        )


@contextlib.contextmanager
def _sandbox_for(kind: str, workspace: Path, network: bool) -> Iterator[DockerSandbox | None]:
    if kind == "none":
        console.print("[yellow]sandbox: none, commands run directly on this machine[/yellow]")
        yield None
        return
    try:
        box = DockerSandbox(workspace, network=network).start()
    except SandboxError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(
        f"sandbox: docker ({box.name}), sees only the workspace, "
        f"network {'on' if network else 'off'}"
    )
    try:
        yield box
    finally:
        box.stop()


@app.command()
def run(
    task_id: str,
    model: str = _MODEL,
    effort: str = _EFFORT,
    tools: str = _TOOLS,
    keep: bool = typer.Option(False, help="Print the workspace path to inspect it."),
    sandbox: str = _SANDBOX,
    clear_at: int = _CLEAR_AT,
    server_clear_at: int = _SERVER_CLEAR_AT,
    compact_at: int = _COMPACT_AT,
    notes: bool = _NOTES,
    count_tokens: str = _COUNT_TOKENS,
    max_cost: float = _MAX_COST,
) -> None:
    """Run the agent on one benchmark task in a fresh copy of its repo, then grade it."""
    task = load_task(TASKS_DIR / task_id)
    _check_sandbox(sandbox)
    context = _context(clear_at, server_clear_at, compact_at, notes, count_tokens)
    workdir = Path(tempfile.mkdtemp(prefix="miniagent-"))
    repo = prepare_workspace(task, workdir)
    config = _config(
        model,
        effort,
        tools,
        max_steps=task.max_steps,
        timeout_s=task.timeout_s,
        share_python_env=True,  # task repos use miniagent's env (numpy etc.)
        context=context,
        max_cost_usd=max_cost,
    )
    transcript = _transcript_path(task.id)

    # Benchmark runs are offline: the task has everything it needs, and no network
    # means nothing to leak to and nothing to download mid-run.
    with _sandbox_for(sandbox, repo, network=False) as box:
        result = run_agent(
            task.id,
            task.prompt,
            repo,
            config,
            transcript_path=transcript,
            on_step=_print_step,
            sandbox=box,
        )
    # Grading runs on the host, after the container is gone: the held-out data in
    # tasks/_shared was never mounted, so the agent could not have read it.
    graded = grade(task, repo)

    _print_result(result, transcript)
    for name, value in graded.metrics.items():
        console.print(f"{name}: {value:.3f}")
    if keep:
        console.print(f"workspace: {repo}")
    console.print("[green]PASS[/green]" if graded.passed else "[red]FAIL[/red]")
    raise typer.Exit(0 if graded.passed else 1)


@app.command("run-repo")
def run_repo(
    path: Annotated[Path, typer.Argument(help="The repository to work on. It is never modified.")],
    prompt: Annotated[str, typer.Argument(help="What the agent should do, in plain words.")],
    model: str = _MODEL,
    effort: str = _EFFORT,
    tools: str = _TOOLS,
    test_command: str = typer.Option(
        None,
        help="How to run this repo's tests. Default: 'uv run pytest -q' if it has a "
        "pyproject.toml, otherwise 'python -m pytest -q'.",
    ),
    max_steps: int = typer.Option(40, help="Hard cap on model calls."),
    timeout_s: int = typer.Option(1200, help="Hard cap on wall time for the whole run."),
    test_timeout_s: int = typer.Option(300, help="Timeout for one test run."),
    sandbox: str = _SANDBOX,
    network: bool = typer.Option(
        True,
        help="Network inside the sandbox. On by default so `uv run` can install your "
        "repo's dependencies; --no-network if they're already vendored.",
    ),
    clear_at: int = _CLEAR_AT,
    server_clear_at: int = _SERVER_CLEAR_AT,
    compact_at: int = _COMPACT_AT,
    notes: bool = _NOTES,
    count_tokens: str = _COUNT_TOKENS,
    max_cost: float = _MAX_COST,
) -> None:
    """Run the agent on a copy of your repository. Your repo and its git are never touched.

    You get a changes.patch to review, and apply yourself if you want it. No grading."""
    # Validate everything before creating the workspace, so a typo leaves nothing behind.
    _check_sandbox(sandbox)
    config = _config(
        model,
        effort,
        tools,
        max_steps=max_steps,
        timeout_s=timeout_s,
        test_timeout_s=test_timeout_s,
        context=_context(clear_at, server_clear_at, compact_at, notes, count_tokens),
        max_cost_usd=max_cost,
    )
    try:
        workspace = create_workspace(path)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[bold]workspace[/bold] (a copy): {workspace.path}")
    console.print(f"your repo is not modified, and its git gets nothing: {workspace.source}")

    config.test_command = test_command or default_test_command(
        workspace.path, in_sandbox=sandbox == "docker"
    )
    console.print(f"tests run with: {config.test_command}")
    transcript = _transcript_path(f"repo-{workspace.source.name}")

    with _sandbox_for(sandbox, workspace.path, network=network) as box:
        result = run_agent(
            workspace.source.name,
            prompt,
            workspace.path,
            config,
            transcript_path=transcript,
            on_step=_print_step,
            sandbox=box,
        )

    _print_result(result, transcript)
    console.rule("changes")
    console.print(write_patch(workspace))
    console.print(f"\npatch:    {workspace.patch_path}")
    console.print(f"files:    {workspace.path}   (open in VS Code: code {workspace.path})")
    console.print(
        f"apply it yourself, if you want it: git -C {workspace.source} apply {workspace.patch_path}"
    )
    console.print(f"clean up: {workspace.cleanup_hint()}")


def _config(model: str | None, effort: str | None, tools: str | None, **kwargs: Any) -> AgentConfig:
    if not model:
        raise typer.BadParameter("set --model or MINIAGENT_MODEL")
    has_profile = (Path.home() / ".config" / "anthropic").exists()
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")) and (
        not has_profile
    ):
        raise typer.BadParameter(
            "no Anthropic credentials found. Put ANTHROPIC_API_KEY in .env and run with "
            "`uv run --env-file .env miniagent ...`"
        )
    config = AgentConfig(model=model, effort=effort, **kwargs)
    if tools:
        names = [t.strip() for t in tools.split(",") if t.strip()]
        unknown = sorted(set(names) - set(TOOL_SCHEMAS))
        if unknown:
            raise typer.BadParameter(f"unknown tools: {unknown}; known: {sorted(TOOL_SCHEMAS)}")
        config.tools = names
    return config


def _transcript_path(name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return RUNS_DIR / name / f"{stamp}.jsonl"


def _print_result(result: Any, transcript: Path) -> None:
    console.rule("result")
    console.print(f"stop reason: {result.stop_reason}   steps: {result.steps}")
    console.print(
        f"tokens  in: {result.input_tokens}  cache read: {result.cache_read_tokens}  "
        f"cache write: {result.cache_write_tokens}  out: {result.output_tokens}"
    )
    cost = f"${result.cost_usd:.4f}" if result.cost_usd is not None else "n/a (model not priced)"
    console.print(
        f"cost: {cost}   cache hit rate: {result.cache_hit_rate:.0%}   "
        f"peak context: {result.peak_context_tokens} tokens   "
        f"clears: {result.clears}   compactions: {result.compactions}"
    )
    console.print(f"wall time: {result.wall_time_s}s   transcript: {transcript}")
    if result.summary:
        console.print(f"summary: {result.summary}")


def _print_step(step: int, response: Any, tool_results: list[dict[str, Any]]) -> None:
    console.rule(f"step {step}")
    results = iter(tool_results)
    for block in response.content:
        if block.type == "text" and block.text.strip():
            console.print(f"[cyan]model:[/cyan] {block.text.strip()}")
        elif block.type == "tool_use":
            args = {
                k: (v if len(str(v)) < 80 else f"<{len(str(v))} chars>")
                for k, v in block.input.items()
            }
            console.print(f"[yellow]tool:[/yellow] {block.name}({args})")
            r = next(results)
            first = r["content"].splitlines()[0] if r["content"] else ""
            style = "red" if r["is_error"] else "dim"
            console.print(f"  [{style}]-> {first}[/{style}]")


if __name__ == "__main__":
    app()
