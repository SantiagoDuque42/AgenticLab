"""Command line entry point: `uv run miniagent --help`."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from miniagent.agent import AgentConfig
from miniagent.agent.loop import run_agent
from miniagent.harness.grade import grade
from miniagent.harness.tasks import RUNS_DIR, TASKS_DIR, load_task, load_tasks, prepare_workspace

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


@app.command()
def run(
    task_id: str,
    model: str = typer.Option(None, envvar="MINIAGENT_MODEL", help="Claude model id."),
    effort: str = typer.Option(None, help="low | medium | high; default: the model's."),
    keep: bool = typer.Option(False, help="Print the workspace path to inspect it."),
) -> None:
    """Run the agent on one task in a fresh copy of its repo, then grade it."""
    if not model:
        raise typer.BadParameter("set --model or MINIAGENT_MODEL")
    task = load_task(TASKS_DIR / task_id)
    workdir = Path(tempfile.mkdtemp(prefix="miniagent-"))
    repo = prepare_workspace(task, workdir)
    config = AgentConfig(
        model=model, effort=effort, max_steps=task.max_steps, timeout_s=task.timeout_s
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    transcript = RUNS_DIR / task.id / f"{stamp}.jsonl"

    result = run_agent(
        task.id, task.prompt, repo, config, transcript_path=transcript, on_step=_print_step
    )
    graded = grade(task, repo)

    console.rule("result")
    console.print(f"stop reason: {result.stop_reason}   steps: {result.steps}")
    console.print(
        f"tokens  in: {result.input_tokens}  cache read: {result.cache_read_tokens}  "
        f"cache write: {result.cache_write_tokens}  out: {result.output_tokens}"
    )
    console.print(f"wall time: {result.wall_time_s}s   transcript: {transcript}")
    if result.summary:
        console.print(f"summary: {result.summary}")
    if keep:
        console.print(f"workspace: {repo}")
    console.print("[green]PASS[/green]" if graded.passed else "[red]FAIL[/red]")
    raise typer.Exit(0 if graded.passed else 1)


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
