"""Docker sandbox: the agent's shell commands and tests run in a container.

Before this, `bash` ran on your machine with your user's rights. The guard rails
(deny-list, read-only git shim, secrets stripped from the environment) only filter
what the agent asks for; a script could still read ~/.ssh, write to /tmp, or open
a network connection. A container changes what the agent *can* do, not just what
it is allowed to ask for:

- only the run's workspace is mounted, at /work. AgenticLab, tasks/_shared (the
  held-out test sets), your home folder and your other repos don't exist in there;
- the root filesystem is read-only, and /tmp is a throwaway tmpfs that disappears
  with the container;
- no network for benchmark runs (`network=False`);
- CPU, memory and process-count limits, no Linux capabilities, no privilege gain;
- no environment variables from your shell, so no API keys, by construction.

One container per run, started once and reused for every command (`docker exec`),
so commands share state the way a terminal does (files in /tmp, a built venv) and
there is no ~1 s container start per command. File tools (read_file, edit_file, ...)
still run on the host: they only touch the workspace, which is the same directory
mounted at /work, and `safe_path` already refuses anything outside it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from miniagent.agent.tools import (
    _GIT_SHIM,
    READONLY_GIT_SUBCOMMANDS,
    ToolOutput,
    truncate,
)

DOCKER_DIR = Path(__file__).resolve().parents[2] / "docker"
WORKDIR = "/work"


class SandboxError(RuntimeError):
    pass


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    probe = subprocess.run(["docker", "info"], capture_output=True, timeout=30)
    return probe.returncode == 0


def _git_shim_text() -> str:
    return _GIT_SHIM.format(allowed="|".join(READONLY_GIT_SUBCOMMANDS), real_git="/usr/bin/git")


def image_tag() -> str:
    """Content-addressed tag: a change to the Dockerfile or the shim means a new image."""
    digest = hashlib.sha256()
    digest.update((DOCKER_DIR / "Dockerfile").read_bytes())
    digest.update(_git_shim_text().encode())
    return f"miniagent-sandbox:{digest.hexdigest()[:12]}"


def ensure_image(log=print) -> str:
    """Build the sandbox image if it isn't there yet, and return its tag."""
    tag = image_tag()
    exists = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
    if exists.returncode == 0:
        return tag
    log(f"building sandbox image {tag} (first run only, about a minute)...")
    with tempfile.TemporaryDirectory() as ctx:
        shutil.copy(DOCKER_DIR / "Dockerfile", Path(ctx) / "Dockerfile")
        (Path(ctx) / "git-readonly").write_text(_git_shim_text())
        build = subprocess.run(
            ["docker", "build", "-q", "-t", tag, ctx], capture_output=True, text=True
        )
    if build.returncode != 0:
        raise SandboxError(f"docker build failed:\n{build.stdout}{build.stderr}")
    return tag


class DockerSandbox:
    """A running container with `workspace` mounted at /work. Use as a context manager."""

    def __init__(
        self,
        workspace: Path,
        *,
        network: bool = False,
        memory: str = "2g",
        cpus: str = "2",
        tmp_size: str = "1g",
        image: str | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.network = network
        self.memory, self.cpus, self.tmp_size = memory, cpus, tmp_size
        self.image = image
        self.name = f"miniagent-{uuid.uuid4().hex[:12]}"
        self._started = False

    def start(self) -> DockerSandbox:
        image = self.image or ensure_image()
        # Run as your own uid/gid: files the agent creates in /work belong to you on the
        # host, not to root, and the agent has no root rights inside the container.
        user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "1000:1000"
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", self.name,
            "--user", user,
            "--network", "bridge" if self.network else "none",
            "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", "512",
            "--read-only",
            "--tmpfs", f"/tmp:rw,exec,size={self.tmp_size}",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "-v", f"{self.workspace}:{WORKDIR}",
            "-w", WORKDIR,
            image,
            "sleep", "infinity",
        ]  # fmt: skip
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SandboxError(f"could not start the sandbox:\n{proc.stderr}")
        self._started = True
        return self

    def exec(self, command: str | list[str], timeout_s: int) -> ToolOutput:
        """Run a command in /work. A shell string goes through bash -c."""
        argv = ["bash", "-c", command] if isinstance(command, str) else command
        proc = subprocess.Popen(
            ["docker", "exec", "-i", "-w", WORKDIR, self.name, *argv],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            output, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._kill_commands()
            proc.kill()
            output, _ = proc.communicate()
            return ToolOutput(
                f"Timed out after {timeout_s}s and was killed.\n{truncate(output or '')}",
                is_error=True,
            )
        return ToolOutput(f"exit code: {proc.returncode}\n{truncate(output)}")

    def _kill_commands(self) -> None:
        # Killing the `docker exec` client does not stop what it started inside the
        # container. `kill -9 -1` signals every process we may signal except PID 1
        # (`sleep infinity`), so the container survives and is ready for the next command.
        subprocess.run(
            ["docker", "exec", self.name, "sh", "-c", "kill -9 -1"],
            capture_output=True,
            timeout=30,
        )

    def stop(self) -> None:
        if self._started:
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=60)
            self._started = False

    def __enter__(self) -> DockerSandbox:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
