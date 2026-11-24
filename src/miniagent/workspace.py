"""Workspaces for running the agent on a real repository without touching the original.

Two modes:
- git worktree (default for git repos): a second checkout of the same repository on a
  new branch `miniagent/<timestamp>`. Cheap, shares history, and the agent's changes
  show up in VS Code's Source Control and in `git diff` like any branch.
- copy (non-git folders, or --copy): a plain copy of the files, with a fresh git repo
  and one baseline commit so `git diff` still shows exactly what the agent changed.

Either way the agent never writes to the folder you pointed it at.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from miniagent.agent.tools import IGNORED_DIRS

DEFAULT_WORKSPACES_DIR = Path.home() / ".miniagent" / "workspaces"


@dataclass(frozen=True)
class Workspace:
    path: Path
    mode: str  # "worktree" | "copy"
    source: Path
    branch: str | None = None
    warning: str | None = None

    def cleanup_hint(self) -> str:
        if self.mode == "worktree":
            return (
                f"git -C {self.source} worktree remove --force {self.path} && "
                f"git -C {self.source} branch -D {self.branch}"
            )
        return f"rm -rf {self.path}"


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


def git_toplevel(path: Path) -> Path | None:
    if not shutil.which("git"):
        return None
    proc = _git("rev-parse", "--show-toplevel", cwd=path)
    return Path(proc.stdout.strip()) if proc.returncode == 0 else None


def create_workspace(
    source: Path, *, force_copy: bool = False, root: Path = DEFAULT_WORKSPACES_DIR
) -> Workspace:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"not a directory: {source}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    toplevel = git_toplevel(source)

    if toplevel and not force_copy:
        dest = root / f"{toplevel.name}-{stamp}"
        branch = f"miniagent/{stamp}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        proc = _git("worktree", "add", "-b", branch, str(dest), "HEAD", cwd=toplevel)
        if proc.returncode != 0:
            raise RuntimeError(f"git worktree add failed: {proc.stderr.strip()}")
        warning = None
        if _git("status", "--porcelain", cwd=toplevel).stdout.strip():
            warning = (
                "The source repo has uncommitted changes. A worktree starts from the last "
                "commit, so the agent will NOT see them. Commit them first, or use --copy."
            )
        return Workspace(dest, "worktree", toplevel, branch=branch, warning=warning)

    dest = root / f"{source.name}-{stamp}"
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns(*IGNORED_DIRS))
    if shutil.which("git"):
        _git("init", "-q", cwd=dest)
        _git("add", "-A", cwd=dest)
        _git(
            "-c", "user.name=miniagent", "-c", "user.email=miniagent@localhost",
            "commit", "-q", "--no-verify", "-m", "baseline before agent run",
            cwd=dest,
        )  # fmt: skip
    return Workspace(dest, "copy", source)


def diff_stat(workspace: Workspace) -> str:
    """What the agent changed, as `git diff --stat` including new files."""
    if not shutil.which("git"):
        return "(git not installed; compare the folders manually)"
    _git("add", "-A", "--intent-to-add", cwd=workspace.path)  # make new files show up
    out = _git("diff", "--stat", cwd=workspace.path).stdout.strip()
    return out or "(no changes)"


def default_test_command(repo: Path) -> str:
    """The repo's own way of running its tests, in its own environment."""
    if (repo / "pyproject.toml").exists() and shutil.which("uv"):
        return "uv run pytest -q"
    return "python -m pytest -q"
