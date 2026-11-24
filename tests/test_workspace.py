"""Workspaces for real repos: the original is never touched, and changes are diffable."""

import shutil
import subprocess
from pathlib import Path

import pytest

from miniagent.agent.tools import execute_tool
from miniagent.workspace import create_workspace, default_test_command, diff_stat

pytestmark = pytest.mark.skipif(not shutil.which("git"), reason="git not installed")


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "myproject"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("VALUE = 1\n")
    git("init", "-q", cwd=repo)
    git("add", "-A", cwd=repo)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=repo)
    return repo


def test_worktree_isolates_changes(git_repo: Path, tmp_path: Path):
    ws = create_workspace(git_repo, root=tmp_path / "ws")

    (ws.path / "src" / "app.py").write_text("VALUE = 2\n")
    (ws.path / "new.py").write_text("x = 1\n")

    assert ws.mode == "worktree" and ws.branch.startswith("miniagent/")
    assert (git_repo / "src" / "app.py").read_text() == "VALUE = 1\n"  # original untouched
    assert not (git_repo / "new.py").exists()
    stat = diff_stat(ws)
    assert "src/app.py" in stat and "new.py" in stat
    assert ws.branch in git("branch", "--list", "miniagent/*", cwd=git_repo)


def test_dirty_repo_warns(git_repo: Path, tmp_path: Path):
    (git_repo / "src" / "app.py").write_text("VALUE = 99\n")  # uncommitted
    ws = create_workspace(git_repo, root=tmp_path / "ws")
    assert ws.warning and "uncommitted" in ws.warning
    assert (ws.path / "src" / "app.py").read_text() == "VALUE = 1\n"


def test_copy_mode_skips_noise_and_has_a_baseline(tmp_path: Path):
    folder = tmp_path / "plain"
    (folder / ".venv" / "lib").mkdir(parents=True)
    (folder / "main.py").write_text("print('hi')\n")
    ws = create_workspace(folder, root=tmp_path / "ws")

    assert ws.mode == "copy"
    assert (ws.path / "main.py").exists() and not (ws.path / ".venv").exists()
    assert diff_stat(ws) == "(no changes)"
    (ws.path / "main.py").write_text("print('bye')\n")
    assert "main.py" in diff_stat(ws)
    assert (folder / "main.py").read_text() == "print('hi')\n"


def test_force_copy_on_git_repo_includes_uncommitted(git_repo: Path, tmp_path: Path):
    (git_repo / "src" / "app.py").write_text("VALUE = 99\n")
    ws = create_workspace(git_repo, force_copy=True, root=tmp_path / "ws")
    assert ws.mode == "copy"
    assert (ws.path / "src" / "app.py").read_text() == "VALUE = 99\n"


def test_default_test_command(tmp_path: Path):
    assert default_test_command(tmp_path) == "python -m pytest -q"
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    expected = "uv run pytest -q" if shutil.which("uv") else "python -m pytest -q"
    assert default_test_command(tmp_path) == expected


def test_custom_test_command_runs_in_repo_without_our_venv(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/miniagent/.venv")
    out = execute_tool(
        "run_tests", {}, tmp_path, test_command="echo venv=${VIRTUAL_ENV:-none} && pwd"
    )
    assert "venv=none" in out.text
    assert str(tmp_path.resolve()) in out.text
