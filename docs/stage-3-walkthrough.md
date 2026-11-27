# Stage 3: context management, walkthrough

Every step of the agent loop resends the whole conversation: the system prompt, the
tool definitions, the task, and every tool call and result so far. So the size of
that conversation drives three things at once:

- **Cost.** Step 30 pays for steps 1 to 29 again (mostly at the cache-read price,
  but still).
- **Quality.** A model reading 150K tokens of old `cat` output finds the one line
  that matters less reliably than one reading 20K.
- **Whether the run can finish at all.** Past the context window the request is
  rejected.

This stage adds `ContextManager` (`src/miniagent/agent/context.py`), which owns the
message list and runs a small set of strategies before every request. It also adds
cost tracking, a cost budget, an optional notes tool, and loading the target repo's
own `AGENTS.md`/`CLAUDE.md`.

```bash
# defaults: clear old tool outputs at 60K tokens, compact at 120K
uv run --env-file .env miniagent run cmapss-rul-target

# force both to happen on a normal task, cheaply, to watch them work
uv run --env-file .env miniagent run cmapss-rul-target --model claude-haiku-5-5 \
    --clear-at 10000 --compact-at 20000 --notes --max-cost 0.10
```

The run summary now prints a line like:

```
cost: $0.0412   cache hit rate: 91%   peak context: 19872 tokens   clears: 2   compactions: 1
```

---

## 1. The constraint that shaped everything: preserved thinking

Newer models (Sonnet 5.5, Haiku 5.5, Opus 5.5) bind each thinking block to the exact
conversation that came before it. If you change an earlier turn and resend, every
thinking block after the change no longer matches, and on newer accounts the API
rejects the request with a 400. Older guides on context management (and my first
draft) assume you can freely edit old turns. You can't anymore.

What is still allowed:

| edit | allowed? |
|---|---|
| append new turns (the normal loop) | yes |
| remove **all** thinking blocks from the history | yes |
| let the **server** edit its copy (context editing beta) | yes |
| replace the **whole** history with a summary ("simple compaction") | yes |
| edit an old tool result and keep the old thinking blocks | **no** |
| summarize the old part, keep the last few turns verbatim ("keep-tail") | **no** |

Every strategy below is one of the "yes" rows. `test_history_is_append_only_until_a_strategy_fires`
checks that between strategy firings, each request is the previous one plus new
turns, byte for byte, which is also what keeps the prompt cache working.

## 2. `ContextManager`: who owns the messages

Before this stage, `loop.py` kept `messages` as a local list. Now the manager owns
it, and the loop calls four methods:

```python
events = ctx.prepare(client)                 # before each request: measure, run strategies
response = client.messages.create(**_request(config, ctx))
ctx.record_response(response.content, response.usage)
ctx.add_tool_results(results)
```

**Why a class and not functions in the loop:** the strategies need shared state (the
messages, the current size estimate, the notes, the summarizer's usage), and the
loop shouldn't know which strategies exist. Adding a strategy means adding a class
with `apply()` and `request_params()`, the same pluggable shape as the
`CommandRunner` protocol in the sandbox stage.

**Strategy order is fixed: server clearing, client clearing, then compaction.**
Clearing is cheap and loses little, so it goes first. Compaction only fires if the
history is still too big after clearing.

**`wire()` strips bookkeeping.** `ClearToolResults` marks cleared blocks with
`_cleared: True` so it doesn't clear them twice. Keys starting with `_` are removed
before sending, so the API never sees them.

## 3. Measuring: how big is the next request?

A strategy needs to know the size *before* sending. Two options:

- **Estimate (default, free).** Every response reports exactly how big its request
  was (`input_tokens + cache_read + cache_write`), plus `output_tokens` for the
  reply we just appended. That is the true size up to the last assistant turn. The
  only unknown is the tool results we append afterwards, which we estimate at 3.5
  characters per token. The error is limited to one step's tool output, and the next
  response corrects it.
- **`--count-tokens api` (exact).** Calls the token-counting endpoint before each
  request. It is free but rate-limited and adds a round trip per step. Use it when
  you want to check how good the estimate is.

The transcript logs `context_estimate` and `context_actual` on every step, so you
can compare them. *Rejected:* a local tokenizer. There is no official one for these
models, and an unofficial one is a guess with extra dependencies.

## 4. Strategy 1: clearing old tool results (`ClearToolResults`)

Most of a coding agent's context is tool output: file reads, test logs, `grep`
results. Twenty steps later the agent rarely needs the raw text, and if it does, it
can run the tool again. Its own decisions (the tool calls, its text) are small and
are kept.

When the context passes `--clear-at` (60K), every tool result except the most recent
`keep` (3) is replaced with:

```
[output cleared to save context: 14302 characters. Run the tool again if you need it.]
```

Decisions:

- **Clear in one batch, not one result per step.** Changing an early turn
  invalidates the prompt cache from that point on, so the next request pays the
  full write price for everything after it. One big clear means one cache miss. A
  rolling clear of one result per step would mean a miss on every step, which costs
  more than it saves. `test_clear_fires_in_batches_not_every_step` checks this.
- **Strip all thinking blocks at the same time.** This is the price of editing old
  turns under preserved thinking (§1). The model loses its old private reasoning,
  not what it did or saw. Since we already take a cache miss for the clear, the
  thinking removal costs nothing extra.
- **The placeholder says how to recover.** "Run the tool again" is an instruction the
  model can follow, which "[removed]" isn't.

## 5. Strategy 1b: the server does it (`--server-clear-at`)

The API has the same feature built in (context editing, beta header
`context-management-2025-06-27`). We send our full history, and the server clears old
tool results in its copy before the model reads it. Our list stays append-only, so
thinking blocks and the cache prefix stay valid with no work from us.

`clear_at_least` is set so the server only clears when it frees at least 10K tokens
(or a quarter of the trigger), for the same batching reason as above. The response
reports what was cleared (`applied_edits`), and the transcript logs it as
`server_edits`.

**Why both?** The client-side one is the one you learn from: you can read exactly
what changes. The server one is what you'd use in production. Running the same task
with each and comparing cost and pass rate is a good experiment. Server clearing is
off by default because it is a beta and depends on the SDK's beta endpoint.

## 6. Strategy 2: compaction (`Compact`)

When clearing isn't enough (a long run, or many big results within the last 3), at
`--compact-at` (120K) the manager:

1. Sends the history plus a summary request (`SUMMARY_PROMPT`: task, what was done,
   current state, failed approaches, next step, with exact paths and numbers). The
   request uses the same model, system prompt and tools as the run, so it is almost
   entirely a cache read. `tool_choice: none` stops the model from calling a tool
   instead of writing.
2. Replaces the **whole** history with one user message: the original task, the
   summary, the notes, and "Continue the task from where you left off."

Decisions:

- **Simple compaction, not keep-tail.** Keeping the last few turns verbatim feels
  safer, but it is the "no" row in §1: the kept turns' thinking blocks were bound to
  the deleted history. Starting clean is the version that works with preserved
  thinking.
- **The original task is copied in verbatim,** not left to the summary. The task is
  the one thing that must not be paraphrased.
- **A failed summary doesn't end the run.** If the summarizer errors or returns
  nothing, the event is logged as `compact_failed` and the run carries on with its
  history uncompacted. The next limit (steps, cost, or the context window) still
  stops it if needed.
- **The summarizer's tokens are billed to the run.** `extra_usage` is added to the
  run's totals and cost, so a run that compacts doesn't look cheaper than it was.

*Rejected: server-side compaction (beta `compact-2026-01-12`).* It does this on the
server and works with preserved thinking, but its trigger can't go below 50K and
defaults to 150K, so you can't watch it happen on our tasks without a very long run.
Writing it by hand also shows what a good summary prompt needs. It's a good
follow-up to compare against.

## 7. Prompt caching: a second breakpoint on the system prompt

Since Stage 1 the request has one top-level `cache_control`, which caches everything
up to the last message, and that point moves forward each step. That's still there.
Now the system prompt block also gets its own `cache_control`. Tools render before
the system prompt, so this breakpoint covers tools and system.

**Why it matters now:** after a compaction the conversation is new, so the moving
breakpoint has nothing to hit. The fixed one still does, so tools and system (a few
thousand tokens, sent on every step) stay at the cache-read price even right after a
compaction or clear.

**Cache hit rate** is `cache_read / (input + cache_read + cache_write)`, printed in
the summary. A normal run should be well above 80%. A rate that drops after each
clear is the batching cost from §4, and you can see it step by step in the
transcript.

## 8. The notes tool (`--notes`)

A scratchpad tool with `read`, `append` and `write`. Its contents live in the manager,
not in the conversation, and are copied into the restart message after a
compaction. Think of it as the agent's lab notebook: the C-MAPSS task already asks
for "a note of what you tried", and this gives that note somewhere to survive.

It is off by default because it adds a tool definition to every request, and on
short tasks it is unused. The final notes are written to the transcript as a `notes`
event. *Rejected:* a `NOTES.md` file in the workspace. It would show up in
`changes.patch` for `run-repo`, and the agent could already write that with
`write_file`.

## 9. Project instructions

If the workspace has `AGENTS.md`, `CLAUDE.md` or `MINIAGENT.md` (first one found
wins), up to 8,000 characters of it are added to the system prompt inside
`<project_instructions>`, with "follow them unless the task says otherwise". This is
how Claude Code and other agents pick up repo conventions ("use `uv run`", "tests are
in `tests/unit`").

Loaded **once, into the system prompt**, so it is part of the cached prefix and
survives compaction. The 8K cap stops a huge file from eating the budget. Only the
workspace copy is read, so this changes nothing in your repo.

## 10. Cost, and a cost budget

`src/miniagent/agent/pricing.py` holds per-model prices (USD per million tokens,
from the pricing page on 2026-11-03):

| model | input | output | cache read | cache write |
|---|---|---|---|---|
| Opus 5.5 | 4.00 | 20.00 | 0.20 | 5.00 |
| Sonnet 5.5 | 2.00 | 10.00 | 0.10 | 2.50 |
| Haiku 5.5 | 0.10 | 0.50 | 0.01 | 0.125 |

(Haiku 5.5 costs 5x more for prompts above 100K tokens. Our tasks stay well below
that, which is another reason to keep context small.)

`RunResult.cost_usd` is recomputed after every response, summarizer calls included.
`--max-cost 0.25` stops the run with stop reason `budget` before the next request
once it has spent that much. This is the guard for your 20 to 30 EUR a month: a run
that loops can't quietly burn through it. *Rejected:* estimating the next request's
cost and stopping before it. A run stops at most one step over the budget, which is
a few cents.

A model not in the table gets `cost: n/a`, so a new model never shows a wrong price.
Update the table when prices change.

## 11. What the console shows

Each step's header shows the size of the request it sent
(`step 7 · context 10,432 tokens`), so you can watch the context grow toward the
thresholds. Each clear or compaction prints its own line:

```
context: cleared 4 old tool outputs, 10,432 -> 6,120 tokens
context: compacted the history into a 1,830-character summary, 20,511 -> 2,890 tokens
```

Server-side clears print `context (server): ...` on the step where the API applied
them. Ctrl+C ends the run with stop reason `interrupted` and still prints the result
block (cost, cache hit rate, peak context). A benchmark run stopped this way is not
graded, and the workspace path is printed so you can look at what the agent did.

## 12. What the transcript shows

New fields per step: `context_estimate`, `context_actual`, `server_edits`. New
events: `context` (each clear or compaction, with tokens before and after, and the
full summary text for compactions), and `notes` at the end. The final `result` has
`cost_usd`, `cache_hit_rate`, `peak_context_tokens`, `clears`, `compactions`.

## 13. Tests

`tests/test_context.py` uses the fake client, so no API calls:

| test | checks |
|---|---|
| clear keeps recent + strips thinking | the 3 oldest of 5 results cleared, last 2 intact, no `_` keys or thinking sent |
| clear fires in batches | a second `prepare` with nothing new to clear does nothing |
| below threshold | nothing changes, byte for byte |
| compaction restarts from summary | one message: task verbatim + summary + notes; summary request uses `tool_choice: none` and is billed |
| failed summary | `compact_failed`, history unchanged |
| server clearing | beta endpoint used, `context_management` sent, edits logged |
| append-only history | each request extends the previous one |
| long run compacts and finishes | a scripted run jumps past the trigger, compacts once, then finishes |
| cost and budget stop | cost computed, `--max-cost` stops with `budget` |
| notes tool | offered only with `--notes`; read/append/write |
| project instructions | `AGENTS.md` lands in the system prompt |
| exact token count | `count_tokens` result replaces the estimate |
| pricing | known model priced, unknown model `None` |
| context events reach the callback | the CLI's hook sees each compaction |
| Ctrl+C | stop reason `interrupted`, with the steps and cost so far |

## 14. Things to try

These are cheap with Haiku (a few cents per run, from your earlier Haiku runs):

- **Clearing vs compaction on the same task.** Run `cmapss-rul-target` three times:
  defaults, `--clear-at 10000 --compact-at 0`, and `--clear-at 0 --compact-at 20000`.
  Compare pass rate, steps, cost and cache hit rate. My guess is clearing alone is
  enough for this task and compaction costs a few steps of re-orientation.
- **Notes or not.** Add `--notes` to the compaction run. Does the agent use the notes,
  and does it repeat fewer experiments after the restart?
- **Client vs server clearing.** `--clear-at 0 --server-clear-at 10000`. Same pass
  rate, better cache hit rate?
- **How good is the estimate?** Run once with `--count-tokens api` and compare
  `context_estimate` with `context_actual` in the transcript.
- **Read a summary.** The `context` event has the full summary text. Is it what you'd
  want to continue from? Editing `SUMMARY_PROMPT` and rerunning is a small prompt
  engineering exercise with a measurable result.
