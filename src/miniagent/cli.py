"""Command line entry point: `uv run miniagent --help`."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from miniagent.agent import AgentConfig
from miniagent.agent.loop import run_agent
from miniagent.harness.grade import grade
from miniagent.harness.tasks import TASKS_DIR, load_task, load_tasks, prepare_workspace

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
def run(task_id: str, model: str = typer.Option(None, envvar="MINIAGENT_MODEL")) -> None:
    """Run the agent on one task in a fresh copy of its repo, then grade it."""
    if not model:
        raise typer.BadParameter("set --model or MINIAGENT_MODEL")
    task = load_task(TASKS_DIR / task_id)
    workdir = Path(tempfile.mkdtemp(prefix="miniagent-"))
    repo = prepare_workspace(task, workdir)
    config = AgentConfig(model=model, max_steps=task.max_steps, timeout_s=task.timeout_s)
    result = run_agent(task.id, task.prompt, repo, config)
    console.print(result)
    graded = grade(task, repo)
    console.print("[green]PASS[/green]" if graded.passed else "[red]FAIL[/red]")
    console.print(f"workspace kept at {repo}" if os.environ.get("MINIAGENT_KEEP") else "")


if __name__ == "__main__":
    app()
