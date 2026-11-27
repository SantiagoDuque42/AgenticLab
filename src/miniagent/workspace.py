"""Workspaces for running the agent on a real repository without touching the original.

The guarantee: miniagent never modifies your repository and never writes to git.

Your files (minus .git, virtualenvs and caches) are copied twice,
into `work/` where the agent edits, and `baseline/`, a read-only snapshot used to
compute the diff afterwards. No git repository is created, no commit is made, and
nothing is written under your repo's folder or its .git. You get a `changes.patch`
to review and apply yourself, with `git apply` or by hand, if you want the changes.

There is deliberately no git-worktree mode: a worktree shares your repo's .git, so it
would add a branch and worktree metadata there, and anything the agent committed would
land in your object store. Copying is slower, and it is the only option that leaves
your repo's git with nothing at all.

On top of that, the agent's own git use is read-only (READONLY_GIT_SUBCOMMANDS in
tools.py), and the copy has no .git, so there is no repo for git to write to anyway.
"""

from __future__ import annotations

import difflib
import filecmp
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from miniagent.agent.tools import IGNORED_DIRS

DEFAULT_WORKSPACES_DIR = Path.home() / ".miniagent" / "workspaces"


@dataclass(frozen=True)
class Workspace:
    path: Path  # where the agent works
    source: Path  # your repository: never written to
    baseline: Path  # read-only snapshot to diff against

    @property
    def patch_path(self) -> Path:
        return self.path.parent / "changes.patch"

    def cleanup_hint(self) -> str:
        return f"rm -rf {self.path.parent}"


def create_workspace(source: Path, *, root: Path = DEFAULT_WORKSPACES_DIR) -> Workspace:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"not a directory: {source}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    run_dir = root / f"{source.name}-{stamp}"
    ignore = shutil.ignore_patterns(*IGNORED_DIRS)
    shutil.copytree(source, run_dir / "baseline", ignore=ignore)
    shutil.copytree(source, run_dir / "work", ignore=ignore)
    return Workspace(run_dir / "work", source, run_dir / "baseline")


# --- What changed --------------------------------------------------------------------


def _files(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and not any(part in IGNORED_DIRS for part in p.relative_to(root).parts)
    }


def _lines(path: Path) -> list[str] | None:
    try:
        return path.read_text(encoding="utf-8").splitlines(keepends=True)
    except (UnicodeDecodeError, OSError):
        return None  # binary


def write_patch(workspace: Workspace) -> str:
    """Write `changes.patch` (unified diff) next to the workspace and return a summary.

    Diffs `work/` against the `baseline/` snapshot in pure Python: no git involved.
    """
    before, after = _files(workspace.baseline), _files(workspace.path)
    chunks: list[str] = []
    summary: list[str] = []
    for rel in sorted(before | after):
        old_path, new_path = workspace.baseline / rel, workspace.path / rel
        if rel in before and rel in after and filecmp.cmp(old_path, new_path, shallow=False):
            continue
        old = _lines(old_path) if rel in before else []
        new = _lines(new_path) if rel in after else []
        status = "added" if rel not in before else "deleted" if rel not in after else "modified"
        if old is None or new is None:
            summary.append(f"{status:>8}  {rel} (binary)")
            continue
        diff = list(
            difflib.unified_diff(
                old,
                new,
                fromfile=f"a/{rel}" if rel in before else "/dev/null",
                tofile=f"b/{rel}" if rel in after else "/dev/null",
            )
        )
        plus = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
        minus = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
        summary.append(f"{status:>8}  {rel}  +{plus} -{minus}")
        chunks.append("".join(line if line.endswith("\n") else line + "\n" for line in diff))
    workspace.patch_path.write_text("".join(chunks))
    return "\n".join(summary) or "(no changes)"


def default_test_command(repo: Path, *, in_sandbox: bool = False) -> str:
    """The repo's own way of running its tests, in its own environment.
    The sandbox image always has uv; on the host, only if it is installed."""
    if (repo / "pyproject.toml").exists() and (in_sandbox or shutil.which("uv")):
        return "uv run pytest -q"
    return "python -m pytest -q"
