"""Numbers a hidden test reports besides pass/fail, e.g. the RMSE a model reached.

Pass/fail says whether the agent cleared the bar; the metric says by how much, which
is what you compare across models, prompts and agents. Hidden tests run in a pytest
subprocess, so they hand metrics back through a JSON-lines file named in an
environment variable set by the grader.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

METRICS_FILE_ENV = "MINIAGENT_METRICS_FILE"


def record_metric(name: str, value: float) -> None:
    """Call from a hidden test. A no-op outside grading (e.g. a test run by hand)."""
    path = os.environ.get(METRICS_FILE_ENV)
    if path:
        with Path(path).open("a") as f:
            f.write(json.dumps({"name": name, "value": float(value)}) + "\n")


def read_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["name"]: row["value"] for row in rows}
