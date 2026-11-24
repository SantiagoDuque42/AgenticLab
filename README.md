# miniagent

A coding agent written from scratch in Python, with its own eval harness. It
uses the raw Anthropic Messages API and no agent framework. The plan is to grow
it into a multi-agent orchestrator later, and the benchmark will show whether
that beats a single agent.

## Quick start

```bash
uv sync
cp .env.example .env              # add your ANTHROPIC_API_KEY and model names
uv run pytest -q                  # harness self-tests
uv run miniagent tasks            # list benchmark tasks
uv run miniagent run fix-off-by-one   # needs the agent loop (milestone 1)
```

## Layout

```
src/miniagent/
  agent/
    types.py     # AgentConfig, RunResult: the seams the orchestrator reuses
    loop.py      # the agent loop (milestone 1, written by hand)
    tools.py     # tool schemas + implementations
  harness/
    tasks.py     # load tasks, copy a fresh repo per run
    grade.py     # add hidden tests to the final repo and run pytest
  cli.py         # `miniagent tasks | run | grade`
tasks/<id>/
  task.yaml      # prompt, limits, tags
  repo/          # starting code
  hidden_tests/  # never shown to the agent
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

Then: multi-agent orchestrator (planner, coders in parallel worktrees, reviewer).
