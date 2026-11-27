# Docker sandbox: walkthrough

Before this PR, the agent's `bash` and `run_tests` ran directly on your machine, as
your user. The guard rails (command deny-list, read-only git shim, secrets stripped
from the environment) filter what the agent *asks* for. They can't stop a script it
writes from reading `~/.ssh`, leaving files in `/tmp`, or opening a network
connection. The C-MAPSS runs showed both gaps in practice: Sonnet and Haiku wrote
experiment scripts to `/tmp/exp.py`, and in principle either one could have read
the hidden test set from `AgenticLab/tasks/_shared/`.

Now those commands run in a Docker container that can't see those things at all.

```bash
uv run --env-file .env miniagent run cmapss-rul-target            # sandboxed (default)
uv run --env-file .env miniagent run cmapss-rul-target --sandbox none   # old behaviour
uv run --env-file .env miniagent run-repo ~/code/myproject "..."  # sandboxed, network on
```

The first run builds the image, which takes about a minute. After that, starting the
container adds about a second per run.

---

## 1. What the container can and can't reach

`DockerSandbox.start()` in `src/miniagent/sandbox.py` runs:

```
docker run -d --rm --name miniagent-<id>
  --user <your uid>:<your gid>
  --network none                 # benchmark runs; run-repo: bridge, see §4
  --memory 2g --cpus 2 --pids-limit 512
  --read-only --tmpfs /tmp:rw,exec,size=1g
  --cap-drop ALL --security-opt no-new-privileges
  -v <workspace>:/work -w /work
  miniagent-sandbox:<hash> sleep infinity
```

Line by line, and why:

- **`-v <workspace>:/work` is the only mount.** This is the most important line.
  AgenticLab, `tasks/_shared` (the held-out test sets), `hidden_tests/`,
  `solution/`, your home folder and your other repos don't exist inside the
  container. Before, protecting them relied on the agent behaving. Now the files
  just aren't there.
- **`--user <your uid>:<your gid>`.** Files the agent creates in `/work` belong to
  you on the host, so you never need `sudo rm`. It also means the agent isn't root
  inside the container.
- **`--read-only` plus a tmpfs `/tmp`.** The agent can write in exactly two places:
  `/work`, which is the copy you review as a patch, and a private `/tmp` held in
  memory that disappears with the container. `/tmp/exp.py` now lives there, not on
  your disk. `HOME=/tmp` in the image, so tools that write caches to `~` keep
  working.
- **`--network none` for benchmark runs.** A task contains everything it needs.
  With no network the agent has nowhere to send data and nothing to download
  mid-run, which also keeps runs reproducible.
- **`--cap-drop ALL`, `no-new-privileges`, and the limits.** No Linux capabilities,
  no setuid escalation. A runaway process (a fork bomb, or a 20 GB array by
  mistake) hits the container's limits instead of freezing your PC.
- **No `-e` flags, so no environment.** Your shell's variables, including
  `ANTHROPIC_API_KEY`, never reach the container. Before, this depended on
  `SECRET_ENV_PREFIXES` listing every secret name. Now nothing is passed in, so
  nothing can be missed.

**Rejected: a stronger isolation runtime such as gVisor or Firecracker.** It would
be safer against kernel exploits, but needs extra setup on WSL. Plain Docker
already stops a coding agent from causing harm by accident, which is the realistic
risk here.

## 2. One container per run, `docker exec` per command

The container starts once, runs `sleep infinity`, and each `bash` or `run_tests`
call becomes `docker exec -w /work <name> bash -c "<command>"`.

**Rejected: `docker run --rm` per command.** That adds about a second to every
command, and nothing carries over between commands: a venv built in one step
would be gone in the next, and so would a file in `/tmp`. One container per run
behaves like a terminal session, which is what the agent expects.

**The file tools stay on the host.** `read_file`, `edit_file` and the others work on
the workspace directory, which is the same directory mounted at `/work`, so both
sides see the same files. `safe_path` already resolves paths, symlinks included,
and refuses anything outside the workspace. Running them in the container as well
would add a `docker exec` per file read for no gain.

**The interface is one method.** `tools.py` defines a `CommandRunner` protocol with a
single `exec(command, timeout_s) -> ToolOutput`. `bash` and `run_tests` call it
when one is passed in, and `run_agent(..., sandbox=...)` passes it through. Tools
and the loop don't import Docker at all, which is why `test_agent_loop.py` can use
a 10-line `RecordingSandbox` instead of a real container. The orchestrator later
can hand each parallel agent its own sandbox the same way.

## 3. Timeouts inside a container

On the host, a timeout kills the command's whole process group (`os.killpg`).
Killing the `docker exec` client does not stop what it started inside the
container: the processes keep running. So on timeout, `_kill_commands()` runs
`kill -9 -1` inside the container, which signals every process the user may signal
except PID 1 and itself. PID 1 is `sleep infinity`, so the container survives, the
next command works, and the escape test checks both.

## 4. `run-repo`: network on, by default

Your real projects have dependencies. `uv run pytest` inside the container needs
to install them into `/work/.venv`, which is the copy, ignored by `changes.patch`.
That takes network. So `run-repo` uses `--network bridge` by default, and
`--no-network` turns it off when dependencies are vendored.

Network on is still much safer than no sandbox. The container has no secrets,
doesn't see your home folder, and only has the copy of your repo, so there is
little to leak, and nothing outside the copy can be changed. The hard rule still
holds: your repo and its git get nothing, and the copy has no `.git` at all.

The default test command inside the sandbox is `uv run pytest -q` when the repo has a
`pyproject.toml`, since the image always has uv. Otherwise it is `python -m pytest -q`.

## 5. The image

`docker/Dockerfile`: `python:3.12-slim`, plus git and ripgrep, numpy, pytest and uv,
with the read-only git shim at `/usr/local/bin/git` ahead of the real one.

- **The tag is a content hash** (`miniagent-sandbox:<sha256 of Dockerfile + shim>`).
  Editing either one makes a new tag, so the next run rebuilds automatically. There
  is never a stale image to remember to rebuild.
- **The shim text comes from `tools.py`** (`_GIT_SHIM`), the same one used outside
  the sandbox, written into the build context. One definition of "read-only git".
- **The read-only git rules apply in both places.** The command guard refuses
  `git commit` before anything reaches the container, and the shim inside catches
  scripts that call git. Inside the sandbox, the worst case is a write to the
  `/work` copy anyway.
- **Keep numpy in step with the `tasks` group in pyproject.toml.** Grading still runs
  on the host with the host's numpy. A large version gap could make a task pass in
  the container and fail at grading.

## 6. Grading stays outside

`miniagent run` stops the container and only then grades, on the host, reading
`tasks/_shared` for the held-out data. The data was never mounted, so a score like
Haiku's 13.4 is trustworthy by construction. There's no need to grep the
transcript for `test_FD001`.

## 7. The escape tests

`tests/test_sandbox.py` (skipped without Docker; CI runs them) tries what a
misbehaving agent would:

| attempt | expected |
|---|---|
| `cat` a file next to the workspace, `ls ~`, `ls /home` | not found |
| `ls tasks/_shared` | doesn't exist |
| `touch /usr/evil`, `touch /etc/evil` | read-only file system |
| write `/tmp/x`, then look on the host and in a fresh container | only in this run's container |
| `socket.create_connection(("1.1.1.1", 53))` | blocked |
| `env` with `ANTHROPIC_API_KEY` set on the host | not there |
| `id -u`, `CapEff` | your uid, zero capabilities |
| `sleep 60 & sleep 60` with a 2 s timeout | killed, container still usable |
| `git commit`, directly and from a script | refused |
| a full benchmark task (`cmapss-debug-failed-run`), tests + nightly job | passes, grades on host |

*Things to try:*
- Run `--sandbox none` and `--sandbox docker` on the same task and compare
  transcripts. The agent shouldn't notice any difference.
- Lower `--memory` to `256m` and run `cmapss-rul-target` to see how an agent copes
  with an out-of-memory kill.
- Remove `--read-only` and check which escape test fails.
