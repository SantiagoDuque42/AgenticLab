"""run-repo workspaces: the source repo and its git are never touched, and nothing is
ever committed anywhere."""

import shutil
import subprocess
from pathlib import Path

import pytest

from miniagent.agent.tools import execute_tool
from miniagent.workspace import create_workspace, default_test_command, write_patch

needs_git = pytest.mark.skipif(not shutil.which("git"), reason="git not installed")


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


def snapshot(repo: Path) -> dict[str, bytes]:
    """Every file under the repo, .git included, byte for byte."""
    return {p.relative_to(repo).as_posix(): p.read_bytes() for p in repo.rglob("*") if p.is_file()}


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "myproject"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("VALUE = 1\n")
    (repo / "old.txt").write_text("bye\n")
    (repo / ".venv" / "lib").mkdir(parents=True)
    (repo / ".venv" / "lib" / "big.bin").write_bytes(b"\0" * 10)
    if shutil.which("git"):
        git("init", "-q", cwd=repo)
        git("add", "-A", cwd=repo)
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=repo)
    return repo


def test_copy_skips_git_and_noise(git_repo: Path, tmp_path: Path):
    ws = create_workspace(git_repo, root=tmp_path / "ws")
    assert (ws.path / "src" / "app.py").exists()
    assert not (ws.path / ".git").exists()
    assert not (ws.path / ".venv").exists()
    assert ws.baseline != ws.path


def test_uncommitted_changes_are_included(git_repo: Path, tmp_path: Path):
    (git_repo / "src" / "app.py").write_text("VALUE = 99\n")
    ws = create_workspace(git_repo, root=tmp_path / "ws")
    assert (ws.path / "src" / "app.py").read_text() == "VALUE = 99\n"


@needs_git
def test_source_repo_and_its_git_are_byte_for_byte_untouched(git_repo: Path, tmp_path: Path):
    before = snapshot(git_repo)
    ws = create_workspace(git_repo, root=tmp_path / "ws")

    # The agent edits, and even tries to commit, inside the workspace.
    execute_tool("edit_file", {"path": "src/app.py", "old_str": "1", "new_str": "2"}, ws.path)
    commit = execute_tool("bash", {"command": "git commit -am x", "timeout_s": None}, ws.path)
    write_patch(ws)

    assert commit.is_error  # refused by the read-only git guard
    # .git is part of the snapshot: no new objects, refs, branches or worktree entries.
    assert snapshot(git_repo) == before


def test_patch_lists_modified_added_deleted(git_repo: Path, tmp_path: Path):
    ws = create_workspace(git_repo, root=tmp_path / "ws")
    (ws.path / "src" / "app.py").write_text("VALUE = 2\n")
    (ws.path / "new.py").write_text("x = 1\n")
    (ws.path / "old.txt").unlink()

    summary = write_patch(ws)
    patch = ws.patch_path.read_text()

    assert "modified  src/app.py  +1 -1" in summary
    assert "added  new.py  +1 -0" in summary
    assert "deleted  old.txt  +0 -1" in summary
    assert "-VALUE = 1\n+VALUE = 2\n" in patch
    assert "--- /dev/null\n+++ b/new.py" in patch


@needs_git
def test_patch_applies_cleanly_with_git_apply(git_repo: Path, tmp_path: Path):
    ws = create_workspace(git_repo, root=tmp_path / "ws")
    (ws.path / "src" / "app.py").write_text("VALUE = 2\n")
    (ws.path / "new.py").write_text("x = 1\n")
    write_patch(ws)

    # The user's own, optional, step: apply to a clone, not the original.
    clone = tmp_path / "clone"
    shutil.copytree(git_repo, clone)
    git("apply", str(ws.patch_path), cwd=clone)
    assert (clone / "src" / "app.py").read_text() == "VALUE = 2\n"
    assert (clone / "new.py").read_text() == "x = 1\n"


def test_no_changes(git_repo: Path, tmp_path: Path):
    ws = create_workspace(git_repo, root=tmp_path / "ws")
    assert write_patch(ws) == "(no changes)"
    assert ws.patch_path.read_text() == ""


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
