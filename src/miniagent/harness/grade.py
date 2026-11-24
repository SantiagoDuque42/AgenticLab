"""Grading: copy the hidden tests into the agent's final repo and run pytest.

Hidden tests get two environment variables: MINIAGENT_SHARED_DIR, where held-out
data lives (e.g. the official test set the agent never sees), and
MINIAGENT_METRICS_FILE, where `record_metric` writes numbers such as the RMSE.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from miniagent.harness.metrics import METRICS_FILE_ENV, read_metrics
from miniagent.harness.tasks import SHARED_DIR, Task


class GradeResult(BaseModel):
    task_id: str
    passed: bool
    output: str
    metrics: dict[str, float] = Field(default_factory=dict)


def grade(task: Task, final_repo: Path, timeout_s: int = 300) -> GradeResult:
    hidden_dest = final_repo / "hidden_tests"
    if hidden_dest.exists():
        shutil.rmtree(hidden_dest)
    shutil.copytree(task.hidden_tests_dir, hidden_dest)
    with tempfile.TemporaryDirectory() as tmp:
        metrics_file = Path(tmp) / "metrics.jsonl"
        env = {
            **os.environ,
            "MINIAGENT_SHARED_DIR": str(SHARED_DIR),
            METRICS_FILE_ENV: str(metrics_file),
        }
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "hidden_tests"],
            cwd=final_repo,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        metrics = read_metrics(metrics_file)
    return GradeResult(
        task_id=task.id,
        passed=proc.returncode == 0,
        output=proc.stdout + proc.stderr,
        metrics=metrics,
    )
