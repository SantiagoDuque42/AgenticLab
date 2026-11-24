"""Grading: copy the hidden tests into the agent's final repo and run pytest."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from miniagent.harness.tasks import Task


class GradeResult(BaseModel):
    task_id: str
    passed: bool
    output: str


def grade(task: Task, final_repo: Path, timeout_s: int = 120) -> GradeResult:
    hidden_dest = final_repo / "hidden_tests"
    if hidden_dest.exists():
        shutil.rmtree(hidden_dest)
    shutil.copytree(task.hidden_tests_dir, hidden_dest)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "hidden_tests"],
        cwd=final_repo,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return GradeResult(
        task_id=task.id, passed=proc.returncode == 0, output=proc.stdout + proc.stderr
    )
