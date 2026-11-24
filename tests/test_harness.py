import shutil
from pathlib import Path

import pytest

from miniagent.agent.tools import run_tests, safe_path
from miniagent.harness.grade import grade
from miniagent.harness.tasks import TASKS_DIR, load_task, load_tasks, prepare_workspace

TOY = "fix-off-by-one"


def test_tasks_load():
    ids = [t.id for t in load_tasks()]
    assert TOY in ids


ALL_TASKS = [t.id for t in load_tasks()]


@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_task_fails_before_fix(tmp_path: Path, task_id: str):
    """A task whose hidden tests already pass measures nothing."""
    task = load_task(TASKS_DIR / task_id)
    repo = prepare_workspace(task, tmp_path)
    assert not grade(task, repo).passed


@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_task_passes_with_reference_solution(tmp_path: Path, task_id: str):
    """A task nobody can pass measures nothing either: prove a fix exists."""
    task = load_task(TASKS_DIR / task_id)
    repo = prepare_workspace(task, tmp_path)
    shutil.copytree(task.solution_dir, repo, dirs_exist_ok=True)
    result = grade(task, repo)
    assert result.passed, result.output


@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_visible_tests_pass_at_start(tmp_path: Path, task_id: str):
    """Visible tests are the agent's starting signal; they must run cleanly."""
    task = load_task(TASKS_DIR / task_id)
    repo = prepare_workspace(task, tmp_path)
    assert run_tests(repo).text.startswith("exit code: 0")


def test_starting_repo_is_untouched(tmp_path: Path):
    task = load_task(TASKS_DIR / TOY)
    repo = prepare_workspace(task, tmp_path)
    shutil.rmtree(repo / "src")
    assert (task.repo_dir / "src/chunks.py").exists()


def test_safe_path_blocks_escape(tmp_path: Path):
    assert safe_path(tmp_path, "a/b.py") == (tmp_path / "a/b.py").resolve()
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../outside.py")


def test_shared_files_are_copied_into_the_workspace(tmp_path: Path):
    task = load_task(TASKS_DIR / "cmapss-rul-target")
    repo = prepare_workspace(task, tmp_path)
    assert (repo / "data" / "train_FD001.txt.gz").exists()
    # The held-out test set stays out of the agent's workspace.
    assert not list(repo.rglob("test_FD001*")) and not list(repo.rglob("RUL_FD001*"))


def test_grade_collects_metrics(tmp_path: Path):
    task = load_task(TASKS_DIR / "cmapss-rul-target")
    repo = prepare_workspace(task, tmp_path)
    shutil.copytree(task.solution_dir, repo, dirs_exist_ok=True)
    result = grade(task, repo)
    assert result.passed, result.output
    assert 0 < result.metrics["test_rmse"] <= 19.0


def test_record_metric_is_a_noop_outside_grading(monkeypatch):
    from miniagent.harness.metrics import METRICS_FILE_ENV, record_metric

    monkeypatch.delenv(METRICS_FILE_ENV, raising=False)
    record_metric("x", 1.0)  # must not raise
