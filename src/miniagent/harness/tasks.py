"""Benchmark task loading.

A task is a directory under `tasks/`:

    tasks/<task-id>/
      task.yaml       # prompt, limits, tags
      repo/           # starting code, copied fresh for every run
      hidden_tests/   # copied in only after the agent finishes
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
TASKS_DIR = REPO_ROOT / "tasks"
RUNS_DIR = REPO_ROOT / "runs"


class Task(BaseModel):
    id: str
    prompt: str
    max_steps: int = 30
    timeout_s: int = 600
    tags: list[str] = Field(default_factory=list)
    root: Path

    @property
    def repo_dir(self) -> Path:
        return self.root / "repo"

    @property
    def hidden_tests_dir(self) -> Path:
        return self.root / "hidden_tests"


def load_task(task_dir: Path) -> Task:
    data = yaml.safe_load((task_dir / "task.yaml").read_text())
    return Task(id=task_dir.name, root=task_dir, **data)


def load_tasks(tasks_dir: Path = TASKS_DIR) -> list[Task]:
    return [load_task(d) for d in sorted(tasks_dir.iterdir()) if (d / "task.yaml").exists()]


def prepare_workspace(task: Task, workdir: Path) -> Path:
    """Copy the task's starting repo into `workdir` and return the copy's path."""
    dest = workdir / task.id
    shutil.copytree(task.repo_dir, dest)
    return dest
