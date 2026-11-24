import shutil
from pathlib import Path

import pytest

from miniagent.agent.tools import safe_path
from miniagent.harness.grade import grade
from miniagent.harness.tasks import TASKS_DIR, load_task, load_tasks, prepare_workspace

TOY = "fix-off-by-one"


def test_tasks_load():
    ids = [t.id for t in load_tasks()]
    assert TOY in ids


def test_toy_task_fails_before_fix(tmp_path: Path):
    task = load_task(TASKS_DIR / TOY)
    repo = prepare_workspace(task, tmp_path)
    assert not grade(task, repo).passed


def test_toy_task_passes_after_fix(tmp_path: Path):
    task = load_task(TASKS_DIR / TOY)
    repo = prepare_workspace(task, tmp_path)
    fixed = (repo / "src/chunks.py").read_text().replace("len(items) - size + 1", "len(items)")
    (repo / "src/chunks.py").write_text(fixed)
    assert grade(task, repo).passed


def test_starting_repo_is_untouched(tmp_path: Path):
    task = load_task(TASKS_DIR / TOY)
    repo = prepare_workspace(task, tmp_path)
    shutil.rmtree(repo / "src")
    assert (task.repo_dir / "src/chunks.py").exists()


def test_safe_path_blocks_escape(tmp_path: Path):
    assert safe_path(tmp_path, "a/b.py") == (tmp_path / "a/b.py").resolve()
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../outside.py")
