"""Command line entry point: `uv run miniagent --help`."""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from miniagent.agent import AgentConfig
from miniagent.agent.loop import run_agent
from miniagent.agent.tools import TOOL_SCHEMAS
from miniagent.harness.grade import grade
from miniagent.harness.tasks import RUNS_DIR, TASKS_DIR, load_task, load_tasks, prepare_workspace
from miniagent.workspace import create_workspace, default_test_command, write_patch

app = typer.Typer(no_args_is_help=True)
console = Console()


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
    console.print("[green]PASS[/green]" if result.passed else "[red]FAIL[/red]")
    raise typer.Exit(0 if result.passed else 1)


_MODEL = typer.Option(None, envvar="MINIAGENT_MODEL", help="Claude model id.")
_EFFORT = typer.Option(None, help="low | medium | high; default: the model's.")
_TOOLS = typer.Option(
    None, help="Comma-separated tool names to enable, e.g. 'read_file,edit_file,bash'."
)


@app.command()
def run(
    task_id: str,
    model: str = _MODEL,
    effort: str = _EFFORT,
    tools: str = _TOOLS,
    keep: bool = typer.Option(False, help="Print the workspace path to inspect it."),
) -> None:
    """Run the agent on one benchmark task in a fresh copy of its repo, then grade it."""
    task = load_task(TASKS_DIR / task_id)
    workdir = Path(tempfile.mkdtemp(prefix="miniagent-"))
    repo = prepare_workspace(task, workdir)
    config = _config(model, effort, tools, max_steps=task.max_steps, timeout_s=task.timeout_s)
    transcript = _transcript_path(task.id)

    result = run_agent(
        task.id, task.prompt, repo, config, transcript_path=transcript, on_step=_print_step
    )
    graded = grade(task, repo)

    _print_result(result, transcript)
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
) -> None:
    """Run the agent on a copy of your repository. Your repo and its git are never touched.

    You get a changes.patch to review, and apply yourself if you want it. No grading."""
    # Validate everything before creating the workspace, so a typo leaves nothing behind.
    config = _config(
        model,
        effort,
        tools,
        max_steps=max_steps,
        timeout_s=timeout_s,
        test_timeout_s=test_timeout_s,
    )
    try:
        workspace = create_workspace(path)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[bold]workspace[/bold] (a copy): {workspace.path}")
    console.print(f"your repo is not modified, and its git gets nothing: {workspace.source}")

    config.test_command = test_command or default_test_command(workspace.path)
    console.print(f"tests run with: {config.test_command}")
    transcript = _transcript_path(f"repo-{workspace.source.name}")

    result = run_agent(
        workspace.source.name,
        prompt,
        workspace.path,
        config,
        transcript_path=transcript,
        on_step=_print_step,
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
