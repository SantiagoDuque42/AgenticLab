# miniagent

A coding agent written from scratch in Python, with its own eval harness. It
uses the raw Anthropic Messages API and no agent framework. The plan is to grow
it into a multi-agent orchestrator later, and the benchmark will show whether
that beats a single agent.

## Quick start

Runs on Linux, macOS or WSL. On Windows, use WSL (`wsl --install -d Ubuntu`) and reach
your repos at `/mnt/c/...`; the CLI refuses to start on native Windows.

The agent's shell commands and tests run in a Docker container that only sees the
run's workspace (no network for benchmark tasks, no access to your home folder or
the held-out test data). It needs Docker running; the image builds itself on first
use. `--sandbox none` runs without it.

```bash
uv sync
cp .env.example .env              # add your ANTHROPIC_API_KEY and model names
uv run pytest -q                  # harness self-tests
uv run miniagent tasks            # list benchmark tasks
uv run --env-file .env miniagent run fix-off-by-one   # live run, costs a few cents

# Your own repo: the agent works on a copy. Your repo and its git are never touched;
# you get a changes.patch to review and apply yourself.
uv run --env-file .env miniagent run-repo ~/code/myproject "Fix the failing test in tests/test_api.py"
```

How the loop works and why each piece is built the way it is:
[docs/milestone-1-walkthrough.md](docs/milestone-1-walkthrough.md), then
[docs/stage-2-walkthrough.md](docs/stage-2-walkthrough.md). The real-world ML tasks
(C-MAPSS turbofan RUL: hit a metric target, debug a failed nightly run) are explained
in [docs/mlops-tasks-walkthrough.md](docs/mlops-tasks-walkthrough.md), and the sandbox
in [docs/sandbox-walkthrough.md](docs/sandbox-walkthrough.md).

## Layout

```
src/miniagent/
  agent/
    types.py     # AgentConfig, RunResult: the seams the orchestrator reuses
    loop.py      # the agent loop: model call, tool calls, repeat
    tools.py     # tool schemas + implementations
  harness/
    tasks.py     # load tasks, copy a fresh repo per run
    grade.py     # add hidden tests to the final repo and run pytest
    metrics.py   # record_metric(): numbers like test RMSE, reported by hidden tests
  workspace.py   # run-repo workspaces: copy + baseline, changes.patch
  sandbox.py     # Docker sandbox: one container per run, only the workspace mounted
  cli.py         # `miniagent tasks | run | run-repo | grade`
docker/Dockerfile  # the sandbox image (python, numpy, pytest, uv, read-only git)
tasks/<id>/
  task.yaml      # prompt, limits, tags
  repo/          # starting code
  hidden_tests/  # never shown to the agent
  solution/      # reference fix, proves the task is passable
tasks/_shared/   # datasets several tasks use (C-MAPSS FD001), listed under `shared:`
runs/            # transcripts and results (gitignored)
```

## Milestones

1. Bare loop with `read_file`, `write_file`, `run_tests` that passes `fix-off-by-one`
2. Docker sandbox and better tools
3. OpenTelemetry tracing (Langfuse or Phoenix)
4. Harness v1: 10 tasks, runner, report
5. Context management and stopping
6. 20 tasks, A/B changes, smoke benchmark gate in CI
7. Code-search retrieval tool against plain grep
8. Portfolio write-up

Then: multi-agent orchestrator (planner, coders in parallel workspaces, reviewer).
