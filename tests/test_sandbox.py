"""The Docker sandbox: what the agent's commands can and cannot reach.

Skipped when Docker isn't available. CI runs them (GitHub's Ubuntu runners have
Docker); the first run builds the image, which takes about a minute.
"""

import os
import shutil
from pathlib import Path

import pytest

from miniagent.agent.tools import execute_tool
from miniagent.harness.grade import grade
from miniagent.harness.tasks import TASKS_DIR, load_task, prepare_workspace
from miniagent.sandbox import DockerSandbox, docker_available, ensure_image

pytestmark = pytest.mark.skipif(not docker_available(), reason="Docker not available")


@pytest.fixture(scope="session")
def image() -> str:
    return ensure_image()


@pytest.fixture
def box(tmp_path: Path, image: str):
    work = tmp_path / "work"
    work.mkdir()
    (work / "app.py").write_text("VALUE = 1\n")
    with DockerSandbox(work, image=image) as sandbox:
        yield sandbox


def sh(box: DockerSandbox, command: str, timeout_s: int = 30) -> str:
    out = execute_tool("bash", {"command": command, "timeout_s": timeout_s}, box.workspace,
                       sandbox=box)  # fmt: skip
    return out.text


def test_commands_run_in_the_workspace(box: DockerSandbox):
    out = sh(box, "pwd; cat app.py; echo 'VALUE = 2' > app.py")
    assert "/work" in out and "VALUE = 1" in out
    # Same directory as the host workspace: file tools and shell see the same files.
    assert (box.workspace / "app.py").read_text() == "VALUE = 2\n"


def test_files_written_in_work_belong_to_the_host_user(box: DockerSandbox):
    sh(box, "touch made-in-sandbox.txt")
    assert (box.workspace / "made-in-sandbox.txt").stat().st_uid == os.getuid()


def test_host_files_outside_the_workspace_do_not_exist(box: DockerSandbox, tmp_path: Path):
    secret = tmp_path / "secret.txt"  # right next to the workspace on the host
    secret.write_text("do not read")
    out = sh(box, f"cat {secret}; ls {Path.home()} 2>&1; ls /home 2>&1")
    assert "do not read" not in out
    # Neither AgenticLab nor the held-out test data is mounted.
    out = sh(box, f"ls {TASKS_DIR / '_shared'} 2>&1 || echo missing")
    assert "missing" in out and "test_FD001" not in out


def test_root_filesystem_is_read_only(box: DockerSandbox):
    assert "Read-only file system" in sh(box, "touch /usr/evil 2>&1; touch /etc/evil 2>&1")


def test_tmp_is_private_and_disappears(box: DockerSandbox, tmp_path: Path, image: str):
    marker = f"miniagent-test-{os.getpid()}"
    sh(box, f"echo hi > /tmp/{marker}")
    assert "hi" in sh(box, f"cat /tmp/{marker}")  # state persists within one run
    assert not Path(f"/tmp/{marker}").exists()  # never on the host's /tmp
    other = tmp_path / "other"
    other.mkdir()
    with DockerSandbox(other, image=image) as fresh:  # nor in the next run
        assert "No such file" in sh(fresh, f"cat /tmp/{marker} 2>&1")


def test_no_network(box: DockerSandbox):
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3); print('CONNECTED')\n"
        "except OSError as e:\n"
        "    print('blocked', e)\n"
    )
    (box.workspace / "net.py").write_text(code)
    out = sh(box, "python net.py")
    assert "blocked" in out and "CONNECTED" not in out


def test_no_secrets_in_the_environment(box: DockerSandbox, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    assert "sk-should-not-leak" not in sh(box, "env")


def test_not_root_and_no_privileges(box: DockerSandbox):
    out = sh(box, "id -u; grep CapEff /proc/self/status").splitlines()
    assert out[1] == str(os.getuid())  # out[0] is "exit code: 0"
    assert out[2].split()[1] == "0000000000000000"  # no capabilities, even if uid is 0


def test_timeout_kills_the_command_and_the_sandbox_survives(box: DockerSandbox):
    out = execute_tool("bash", {"command": "sleep 60 & sleep 60", "timeout_s": 2},
                       box.workspace, sandbox=box)  # fmt: skip
    assert out.is_error and "Timed out" in out.text
    running = sh(box, "cat /proc/[0-9]*/cmdline | tr '\\0' ' '")
    assert "sleep 60" not in running and "sleep infinity" in running
    assert "exit code: 0" in sh(box, "echo still alive")


def test_git_writes_are_still_refused(box: DockerSandbox):
    out = execute_tool("bash", {"command": "git commit -am x", "timeout_s": None},
                       box.workspace, sandbox=box)  # fmt: skip
    assert out.is_error  # the command guard, before anything reaches the container
    # A script that calls git hits the read-only shim inside the image.
    (box.workspace / "sneaky.sh").write_text("git commit -am x\n")
    assert "disabled" in sh(box, "sh sneaky.sh 2>&1")


def test_benchmark_task_runs_and_grades_through_the_sandbox(tmp_path: Path, image: str):
    task = load_task(TASKS_DIR / "cmapss-debug-failed-run")
    repo = prepare_workspace(task, tmp_path)
    shutil.copytree(task.solution_dir, repo, dirs_exist_ok=True)
    with DockerSandbox(repo, image=image) as sandbox:
        tests = execute_tool("run_tests", {}, repo, sandbox=sandbox)
        nightly = execute_tool(
            "bash", {"command": "python -m rul.nightly", "timeout_s": 120}, repo, sandbox=sandbox
        )
    assert tests.text.startswith("exit code: 0"), tests.text
    assert nightly.text.startswith("exit code: 0"), nightly.text
    assert grade(task, repo).passed  # grading happens on the host, after the container
