# Milestone 1 walkthrough: the agent loop

This explains every design decision in `src/miniagent/agent/`. Read it with the code
open. Each section names a file and a function, says what the code does, and then says
why, including the alternatives that were rejected.

The milestone goal: `uv run --env-file .env miniagent run fix-off-by-one` makes the agent
read the buggy `chunk()` function, fix it, run the tests, and stop. The grader then runs
hidden tests and prints PASS.

---

## 1. The big picture

An "agent" here is one loop:

```
messages = [task prompt]
repeat up to max_steps:
    response = model(system prompt, tool schemas, messages)
    append response to messages
    if the model asked for tools:
        run them, append their results to messages, continue
    else:
        stop
```

That's all there is to it. Everything else in this milestone is about doing those few lines
*safely* (tools can't escape the repo), *observably* (every step is logged with its
token counts), and *testably* (the loop runs against a fake model in CI).

Four files:

| File | Role |
| --- | --- |
| `agent/types.py` | `AgentConfig` (inputs that vary per run) and `RunResult` (what came out) |
| `agent/tools.py` | Tool schemas the model sees, plus the Python that runs each tool |
| `agent/loop.py` | `run_agent()`, the loop itself |
| `cli.py` | `miniagent run`: copies the task repo, runs the agent, prints steps, grades |

---

## 2. Why a hand-written loop and not a framework

The Anthropic SDK has a "tool runner" (`client.beta.messages.tool_runner`) that runs
this loop for you, and frameworks like LangGraph or the Claude Agent SDK do much more.
We skip them for now because the point of this milestone is to see the parts they
hide:

- **When the loop stops.** A framework decides that for you. Here you can see that it
  stops on `stop_reason == "end_turn"`, and what happens on every other stop reason.
- **What goes back into the context.** Every token you send costs money and attention.
  When you write the append calls yourself, you know exactly what the model sees.
- **Cost per step.** We read `response.usage` on every call. Frameworks often bury this.

Stage 7 of the plan compares this loop against a framework, and that comparison only
means something if you know what the hand-written version does.

---

## 3. `types.py`: config in, result out

```python
class AgentConfig(BaseModel):
    model: str
    role: str = "coder"
    system_prompt: str = ...
    tools: list[str] = ["read_file", "write_file", "run_tests"]
    max_steps: int = 30
    max_tokens: int = 16000
    effort: str | None = None
    timeout_s: int = 600
```

**Decision: everything that varies goes in one config object.** The orchestrator in
the multi-agent stage will run a planner, coders, and a reviewer. These are the *same
loop* with different system prompts, tool lists, and models. If those values were
hard-coded in `loop.py`, each role would need its own copy of the loop.

**Decision: Pydantic, not a plain dict.** Typos like `max_step=5` fail immediately
instead of being silently ignored. `model_dump()` also gives a JSON-ready dict for the
transcript for free.

**Decision: `max_tokens = 16000`.** This caps the length of *one* model response.
`write_file` sends a whole file as a tool argument, so a low cap would cut a file off
mid-write. When that happens the API returns `stop_reason == "max_tokens"`, and the
loop treats it as an error (section 6). 16k is the usual safe default for
non-streaming requests: it is large enough for real files and still finishes within
the SDK's HTTP timeout.

**Decision: `effort` is optional.** Current Claude models accept
`output_config={"effort": "low" | "medium" | "high" | ...}`, which trades thinking
depth against tokens. Leaving it as `None` uses the model's default. It is a knob for
the benchmark stage, where you will measure whether `low` is good enough for easy
tasks, which would save money.

```python
class RunResult(BaseModel):
    stop_reason: StopReason   # finished | step_limit | timeout | error
    steps, input_tokens, cache_read_tokens, cache_write_tokens, output_tokens
    wall_time_s, summary, transcript_path
```

**Decision: three separate input token counters.** With prompt caching (section 7),
the API reports input in three buckets, and each one has a different price. Uncached
input costs full price, cache reads cost about 0.1x, and cache writes cost about
1.25x. If you added them together, the cost report later would be wrong.

**Decision: `StopReason` is an enum, not a free string.** The harness will group
failures by reason ("12 timeouts, 3 step limits"), and an enum makes that grouping
reliable.

---

## 4. `tools.py`: what the model can do

### 4.1 Schemas

Each tool is a JSON schema the model reads, for example:

```python
"read_file": {
    "name": "read_file",
    "description": "Read a UTF-8 text file ... `path` is relative to the repository root ...",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string", ...}},
        "required": ["path"],
        "additionalProperties": False,
    },
}
```

**Decision: descriptions are written for the model.** The model only knows a tool
through its name and description. "Path is relative to the repo root, e.g.
'src/app.py'" stops the model from guessing absolute paths. "Run it after every
change" nudges it toward the workflow we want. Tool descriptions are prompts, so
treat them like prompts.

**Decision: `strict: True`.** With strict mode on, the API guarantees that
`tool_use.input` matches the schema exactly. That means `read_file` can do
`input["path"]` without checking whether the key exists or is a string. Strict mode
requires `additionalProperties: false` and a `required` list, which is why both are
always present.

**Decision: `write_file` takes the whole file, not a diff.** Whole-file writes are the
simplest thing that works, and the model rarely gets them wrong. The cost is tokens,
because the model resends the full file even for a one-line change. Milestone 2 adds an
`edit_file` tool that replaces just a snippet, and the benchmark will show how much that
saves.

**Decision: `tool_schemas()` sorts by name.** Prompt caching matches the request
prefix byte for byte, and tools come first in that prefix. If the tool order changed
between runs, the cache would miss every time. Sorting makes the order deterministic.

### 4.2 Implementations

**Decision: every tool takes `repo_path` explicitly.** No tool calls `os.chdir()` or
relies on the working directory. This matters twice:

1. Several agents can run in parallel in one process during the orchestrator stage,
   and the working directory is shared global state.
2. In milestone 2 the tools move into Docker. With `repo_path` passed in, that change
   only touches the implementations, not the loop.

**Decision: `safe_path()` on every path.**

```python
target = (root / relative).resolve()
if not target.is_relative_to(root):
    raise ValueError("path escapes the repository")
```

`resolve()` collapses `..` and follows symlinks *before* the check, so
`../../etc/passwd` and a symlink pointing outside are both caught. This is a guard
against mistakes, not a security boundary. `run_tests` still runs arbitrary code,
because pytest executes the repo's code. The real boundary is the Docker sandbox in
milestone 2, so until then only run the agent on repos you trust.

**Decision: tools never raise; they return `ToolOutput(text, is_error)`.**

```python
def execute_tool(name, tool_input, repo_path) -> ToolOutput:
    ...
    except (ValueError, OSError, UnicodeDecodeError, TypeError) as exc:
        return ToolOutput(f"{type(exc).__name__}: {exc}", is_error=True)
```

If a tool raised, the whole run would crash on something as ordinary as a typo in a
filename. Returning the error as a `tool_result` with `is_error: true` lets the model
read "No such file: src/chunk.py", fix the typo, and try again. That recovery is a
large part of what makes agents work at all. We catch a specific list of exceptions,
not bare `Exception`, so a real bug in our own code still crashes loudly.

**Decision: a failing test run is not an error.** `run_tests` returns
`is_error=False` even when pytest exits 1. The tool did its job and is reporting a
result. `is_error` means "the tool itself broke", for example a timeout. Mixing the
two would teach the model that running tests is risky.

**Decision: `read_file` adds line numbers.** For example, `  12 | for start in range(...)`.
The model can then say "line 6 is wrong", and it reads the code more accurately. The
cost is a few extra tokens per line.

**Decision: `truncate()` keeps the head and the tail.** One test log with thousands of
lines could cost more than the rest of the run combined, so outputs are capped at
10,000 characters. We keep both ends because pytest prints the failure summary at the
bottom, which is the most useful part. The marker `[... N characters truncated ...]`
tells the model that something is missing, so it doesn't think it has the whole file.

**Decision: `run_tests` runs `sys.executable -m pytest` with a timeout.** Using
`sys.executable` runs the same Python interpreter as the agent, so it doesn't depend on
whatever `pytest` happens to be on `PATH`. The 120 s timeout stops one infinite loop in
the code under test from hanging the whole run. `-p no:cacheprovider` stops pytest from
writing `.pytest_cache` into the agent's repo.

---

## 5. `loop.py`: the loop, line by line

### 5.1 Signature

```python
def run_agent(task_id, task_prompt, repo_path, config, *,
              client=None, transcript_path=None, on_step=None) -> RunResult:
```

**Decision: a pure function of its inputs.** There are no globals, no singletons, and
no working-directory changes. Same inputs, same behaviour (apart from the model's
randomness). This is the most important seam for the orchestrator stage.

**Decision: `client` is injectable.** In production it defaults to
`anthropic.Anthropic()`, which reads `ANTHROPIC_API_KEY` from the environment. In tests
we pass a `FakeClient` that replays a script. This is plain dependency injection, and
it is why the loop has real tests that cost nothing and run in CI without a key
(section 8).

**Decision: `on_step` callback instead of `print()`.** The loop doesn't know how it
will be displayed. The CLI passes a function that prints with `rich`. The benchmark
runner will pass nothing and stay quiet. Later, the OpenTelemetry stage can open a span
in this callback. Keeping display out of the loop means none of those changes touch it.

### 5.2 One step

```python
response = client.messages.create(**_request(config, tools, messages))
result.steps = step
_add_usage(result, response.usage)
messages.append({"role": "assistant", "content": response.content})
```

**Decision: the whole history is sent on every call.** The Messages API is stateless.
It remembers nothing between calls, so each request carries the full conversation.
That is why the context, and the cost, grows with every step. It is also why prompt
caching matters so much (section 7), and why context management gets its own stage of
the plan.

**Decision: append `response.content`, not just the text.** The assistant turn can hold
several blocks: text, one or more `tool_use` blocks, and on current models `thinking`
blocks. The API expects them back exactly as it sent them. If you dropped the thinking
blocks, or kept only the text, the next request would fail or the model would lose its
reasoning. Rule of thumb: never rewrite what the model said, only append to it.

### 5.3 Running tools

```python
for block in content:
    if block.type == "tool_use":
        output = execute_tool(block.name, block.input, repo_path)
        results.append({"type": "tool_result", "tool_use_id": block.id,
                        "content": output.text, "is_error": output.is_error})
...
messages.append({"role": "user", "content": tool_results})
```

**Decision: `tool_use_id` links each result to its call.** The model can ask for
several tools in one turn (parallel tool use), and the id is how it knows which result
belongs to which call.

**Decision: all results go back in one user message.** If the model asked for three
reads, all three results return together. Splitting them across messages is a known
anti-pattern. It confuses the turn structure and trains the model to stop making
parallel calls.

**Decision: tools run one after another, not concurrently.** It's simpler, and our
tools are fast. When tools get slow (Docker exec, long test runs), running them on a
thread pool is an easy upgrade inside `_run_tool_calls`.

### 5.4 When to stop

```python
if response.stop_reason == "tool_use":   continue
if response.stop_reason == "end_turn":   FINISHED, keep the final text as summary
else:                                     ERROR (max_tokens, refusal, anything new)
```

Plus two limits checked by the loop itself: `max_steps` (STEP_LIMIT) and `timeout_s`
(TIMEOUT).

**Decision: the model decides when it's done (`end_turn`).** The system prompt tells
it to "reply with a short summary and stop calling tools" when the tests pass. Its
final text becomes `RunResult.summary`. Milestone 5 replaces this with an explicit
`finish` tool, which is more reliable than hoping the model stops.

**Decision: an unknown stop reason is an error, not a crash and not a success.**
`max_tokens` means the response was cut off. `refusal` means the model declined. Any
stop reason added to the API in the future lands here too. Recording it as `ERROR`
with the reason in `summary` means the benchmark can count each kind.

**Decision: hard limits on steps and time.** Without them, a confused model can loop
forever, and every step costs money. `max_steps` is the main brake. The toy task sets
it to 15 in `task.yaml`, while a healthy run takes 4 to 6 steps. The timeout is
checked at the start of each step. It can't interrupt a step that is already running,
but the per-test timeout and the SDK's HTTP timeout bound each step.

**Decision: API errors end the run cleanly.** The `except anthropic.APIError` turns a
network failure or an invalid model name into `RunResult(stop_reason=ERROR)`. It
doesn't need to retry: the SDK already retries 429s, 5xx errors and connection errors
twice by default. The `finally` block always records the wall time and writes the
`end` transcript line, even after an exception.

---

## 6. The request: `_request()`

```python
{
    "model": config.model,
    "max_tokens": config.max_tokens,
    "system": config.system_prompt,
    "tools": tools,
    "messages": messages,
    "cache_control": {"type": "ephemeral"},
}
```

**Decision: the system prompt describes a workflow, not a personality.** It tells the
model to read before editing, make the smallest change, test after each change, and
summarise and stop when done. Each instruction maps to a failure you would otherwise
see: blind edits, rewriting whole modules, never testing, and never stopping.

**Decision: no `thinking` parameter.** Current Claude models think adaptively by
default. They decide per step how much reasoning a step needs. The `effort` knob is
how we will tune that, not a thinking budget.

**Decision: no `temperature`.** Recent API versions reject sampling parameters on
current models. Runs vary anyway, which is why the benchmark repeats each task three
times.

---

## 7. Prompt caching: the biggest cost lever

`"cache_control": {"type": "ephemeral"}` at the top level turns on automatic caching.
The API caches the longest stable prefix of the request (tools, then system, then
messages) and reuses it on the next request if every byte matches.

Here is why that matters for an agent. Step 5 resends steps 1 to 4 word for word. Without
caching you pay full price for that history on every step, so the total cost grows
roughly with the square of the step count. With caching, the repeated prefix is billed
at about a tenth of the price. For a 10-step run, most input tokens become cache reads.

You can see whether it works in the step output: from step 2 on, `cache read` should be
much larger than uncached `in`. If `cache_read_tokens` stays at 0, something in the
prefix is changing between calls, such as a timestamp in the system prompt or an
unstable tool order. That's why `tool_schemas()` sorts its output.

Caveats: a cache entry lives about 5 minutes, and a prefix has to reach a minimum size
before it is cached at all. Very short prompts, like the toy task's first step, may not
be cached. That's fine.

---

## 8. The transcript: `runs/<task>/<timestamp>.jsonl`

Every run writes one JSON object per line:

```
{"event": "start", "task_id": ..., "config": {...}}
{"event": "step", "step": 1, "stop_reason": "tool_use", "content": [...], "tool_results": [...], "usage": {...}}
...
{"event": "end", "result": {...}}
```

**Decision: JSONL, written and flushed per event.** If the run crashes at step 7, steps
1 to 6 are already on disk. One line per event is easy to `grep`, to load with
`pandas.read_json(lines=True)`, and to replay later. It is also the raw material for the
observability stage: each `step` line maps to one span.

**Decision: `runs/` is gitignored.** Transcripts contain full file contents and can be
large. They are data, not source code.

---

## 9. Tests without an API key: `tests/test_agent_loop.py`

```python
class FakeClient:
    def __init__(self, script): self.messages = FakeMessages(script)
```

`FakeMessages.create()` returns the next scripted `anthropic.types.Message`, built with
`Message.model_validate(...)` so it has the *same type* the real API returns. The main
test scripts a perfect run: read the file, write the fix, run the tests, end turn. It
then checks that:

- the run finishes in 4 steps with the right summary and summed token counts;
- the hidden tests pass on the repo the fake agent edited, which exercises the real
  tools end to end;
- the conversation sent on the last call alternates `user` and `assistant` correctly;
- each `tool_result` carries the id of its `tool_use`;
- the transcript has `start`, four `step` lines, and `end`.

Other tests cover the step limit, `max_tokens` handled as an error, tools returning
errors instead of raising, and a failing test run not counting as a tool error.

**Why this matters:** these tests check *your* code, meaning the loop mechanics, and
not the model's intelligence. They run in about a second, cost nothing, and run on
every PR. The benchmark (stage 5) is where the model's intelligence gets measured,
and that costs money, so it runs rarely.

---

## 10. Running it for real

```bash
cp .env.example .env            # put your ANTHROPIC_API_KEY in it
uv run --env-file .env miniagent run fix-off-by-one
uv run --env-file .env miniagent run fix-off-by-one --model claude-haiku-5-5   # cheaper
```

You will see each step with the model's text, the tool calls (long arguments shortened
to `<N chars>`), the first line of each result, and then token totals, the transcript
path, and PASS or FAIL.

Rough cost: a healthy run is 4 to 6 steps of a few thousand input tokens each, which is
a few US cents on Sonnet and well under a cent on Haiku. This is an estimate, so check
the real totals that the run prints. Set a monthly spend limit in the Anthropic console
before your first run.

### Things to try (each one is a small experiment)

1. Run the task 3 times. Do the step counts differ? Why?
2. Delete the "run the tests after each change" sentence from the system prompt. Does
   the agent still test?
3. Run with `--effort low`. Does it still pass? Compare tokens.
4. Make `read_file` return an error for every call. Watch how the model reacts.
5. Open a transcript and find the step where `cache_read_input_tokens` first becomes
   non-zero.
