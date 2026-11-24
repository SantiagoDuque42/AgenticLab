# MLOps benchmark tasks: walkthrough

This PR adds two benchmark tasks that look like real ML engineering work, built on
the NASA C-MAPSS turbofan dataset (FD001), plus the harness features they need.

| task | what the agent has to do | graded by |
|------|--------------------------|-----------|
| `cmapss-rul-target` | improve a remaining-useful-life model from RMSE 32 to 19 or better, by experimenting | RMSE on the official test set, which the agent never sees |
| `cmapss-debug-failed-run` | a nightly training job crashed; find the root cause from the log, then review last week's changes for a second, silent bug | four hidden tests that tell root-cause fixes from symptom patches |

The old tasks (off-by-one and the rest) check whether the loop and the tools work.
These two check whether the agent can do what you'd hire someone for: run an
experiment loop toward a number, and debug from evidence instead of guessing.

---

## 1. The data: C-MAPSS FD001

C-MAPSS is NASA's simulated turbofan degradation dataset (public, from the NASA
Prognostics Center of Excellence). In FD001, each row is one engine at one cycle:
unit id, cycle, 3 operational settings and 21 sensors. The 100 training engines run
until they fail. The 100 test engines are cut off at some point before failure, and
`RUL_FD001.txt` gives the true cycles left at each cut. The standard score is the
RMSE of the RUL predicted at each test engine's last cycle.

**Decision: commit all of FD001, gzipped (1.3 MB), in `tasks/_shared/cmapss/`.** I
had suggested a small sample plus a download script. Once I measured it, the full
set compresses to 1.3 MB, which is small enough, and it is better in three ways: the
scores match the published FD001 numbers, CI needs no network, and nobody's run
depends on NASA's URL staying up. `numpy.loadtxt` reads `.gz` directly, so nothing
needs to be unpacked.

## 2. Harness changes

### Shared files: `shared:` in task.yaml

Both tasks need the same training file. Instead of two copies, the file lives once
in `tasks/_shared/` and each task lists what to copy into its workspace:

```yaml
shared:
  data/train_FD001.txt.gz: cmapss/train_FD001.txt.gz   # workspace path: _shared path
```

`prepare_workspace` copies these after `repo/`. `_shared/` has no `task.yaml`, so
`load_tasks` doesn't treat it as a task.

**The test set never enters the workspace.** Only the training file is listed.
The hidden tests read `test_FD001` and `RUL_FD001` from `MINIAGENT_SHARED_DIR`, an
environment variable the grader sets. The agent can't tune on the test set because
it doesn't have it, which is how a real held-out set should work. A test in
`test_harness.py` checks this.

*Honest caveat:* the agent's `bash` could still `cat` an absolute path into
`AgenticLab/tasks/_shared/`, just as it could read `hidden_tests/` or `solution/`
today. The planned Docker sandbox closes that: the container will only mount the
workspace.

### Metrics: `record_metric`

Pass/fail says whether the agent cleared the bar. The metric says by how much, and
that is what you compare across models, prompts and agents (and against Claude Code
on the same task). Hidden tests run in a separate pytest process, so they hand
numbers back through a file:

```python
from miniagent.harness.metrics import record_metric
record_metric("test_rmse", score)        # appends {"name":..., "value":...} as JSON
```

`grade()` points `MINIAGENT_METRICS_FILE` at a temp file, reads it after pytest
exits, and returns `GradeResult.metrics`. `miniagent run` and `miniagent grade`
print them. *Rejected:* parsing numbers out of pytest's stdout, which breaks as soon
as someone changes the output.

**The metric grader is just a hidden test.** The threshold is an `assert` inside it,
so no new grader type was needed. The pytest-based harness already covers the
"metric threshold" case: the test computes the number, records it, and asserts the
bar.

### Benchmark tasks share miniagent's Python env

The task repos import numpy. Before this PR, the agent's `bash` deliberately
removed miniagent's venv from the environment (right for `run-repo`: your project's
`python` must be its own). For a benchmark task, that left a bare `python` with no
numpy, so the agent couldn't even run `python -m rul.evaluate`.

So `AgentConfig.share_python_env` (default False) is passed through `execute_tool` to
`bash`. When it is True, `_clean_env` keeps `VIRTUAL_ENV` and puts miniagent's
venv `bin/` first on PATH. Only `miniagent run` (benchmark mode) sets it. `run-repo`
behaves exactly as before, and a test covers both cases. Secrets are still stripped,
and the read-only git shim still comes first on PATH.

numpy is added in a new dependency group `tasks`, installed by default
(`[tool.uv] default-groups`). It is not a dependency of miniagent itself: the agent
doesn't need it, only the task repos do.

## 3. Task `cmapss-rul-target`: hit a metric target

**Starting repo** (`turbofan-rul`): load data, all 24 columns as features,
standardize, ridge regression in closed form, and an uncapped RUL target. That gives
test RMSE 32.0.

**The prompt** gives the bar (19), says the test set is hidden, and points at
`python -m rul.evaluate` as the proxy. It asks the agent to keep a note of what it
tried, which is good practice and makes the transcript easy to read afterwards.

**How the validation mimics the test set** (`rul/evaluate.py`):
- It holds out whole engines, never rows. Neighbouring cycles of one engine are
  nearly identical, so a row split would validate on engines the model trained on.
- It cuts each held-out engine at a random cycle, keeping at least 31 observed
  cycles and at most 150 to go, like the FD001 test set. My first version cut
  anywhere, and its scores pointed the wrong way: a better model scored worse,
  because uniform cuts include RULs of 300 that the test set never has. A proxy
  metric that doesn't track the real one sends the agent the wrong way, so this
  was the most important thing to get right.
- It averages over 5 seeds, because 20 engines per split is noisy.

**Setting the bar.** I measured what each idea gets on the real test set:

| change | test RMSE |
|--------|-----------|
| starting model | 32.0 |
| cap target at 125 cycles (one line) | 21.9 |
| + only the 14 informative sensors | 22.5 |
| + quadratic feature crosses (still linear) | ~20.1 |
| 14 sensors + 30-cycle trailing means + small MLP (reference solution) | **17.5** |

A bar of 19 means the obvious one-line trick (capping) isn't enough, and neither is
squeezing the linear model. The agent has to find the main ideas (capped target,
smoothing over time, a non-linear model) and implement an MLP with Adam in plain
numpy. That's real work, and the reference solution clears the bar by 1.5, leaving
room for BLAS differences between machines.

*Why numpy only, no scikit-learn or torch:* it keeps the install small on your
machine, and it means the agent has to write the model rather than call `.fit()`.
*Things to try:* allow scikit-learn and see how much easier it gets, or lower the
bar to 16, where published LSTMs land, and see whether any model gets there.

**The grading test** fits the model itself from the pristine training file in
`_shared`, so editing `data/` can't help. It then predicts on the hidden test set,
records `test_rmse`, and asserts it is 19.0 or below. It also checks that the shape
is right and every prediction is finite.

## 4. Task `cmapss-debug-failed-run`: debug from a log

**The story.** A nightly job (`rul/nightly.py`) validates, trains and saves the
model. `logs/nightly-2026-11-16.log` is the last good run (val RMSE 22.90).
`logs/nightly-2026-11-23.log` is the failure: a `RuntimeWarning: invalid value
encountered in divide` in `Scaler.transform`, then `validation RMSE nan`, then
`RuntimeError`. I generated that log by running the broken code, so it is the real
output, with paths changed to `/srv/turbofan`. `CHANGES.md` lists last week's change
set, as a commit message would.

**Two planted bugs, one loud and one silent:**

1. **Loud: division by zero in the scaler.** The change set swapped the hand-picked
   sensors for "all columns, let the model decide". FD001 has columns that never
   change (`op3`, `s18`, `s19`), so their std is 0, and the same change "simplified"
   `Scaler.fit` by removing the guard. 0/0 = NaN poisons the fit.
2. **Silent: a leaky validation split.** The "vectorized" `split_units` now picks
   random *rows*, not engines. Nothing crashes, and with a linear model the number
   barely moves (22.9 vs 22.9). That is exactly why it's dangerous, and why the
   prompt asks the agent to review every change in `CHANGES.md`. The test that
   would have caught it was deleted in the same change, and the module's docstring
   still says "never rows". A careful reviewer notices the mismatch.

**The hidden tests are built to reject symptom patches.** I checked these variants
against the grader:

| fix | nightly runs | split test | drift test | grade |
|-----|:---:|:---:|:---:|:---:|
| untouched | ✗ | ✗ | ✗ | FAIL |
| `np.nan_to_num` on the scaler output + split fix | ✓ | ✓ | ✗ | FAIL |
| scaler guard only | ✓ | ✗ | ✓ | FAIL |
| scaler guard + split fix | ✓ | ✓ | ✓ | PASS |

The drift test is the interesting one. It fits the model, then adds +5 to `s18` and
-2 to `s19` in the test data, the way a recalibrated sensor would in service.
Predictions must stay finite and within 1 cycle of the undrifted ones. With a
proper guard, a constant column scales by 1 and gets weight 0, so drift does
nothing. With `nan_to_num`, the drifted column becomes `5/0 = inf`, then the largest
float, and the predictions are garbage. So the test separates fixing the cause from
hiding the symptom, which is the skill this task measures.

**A subtlety I hit while building it.** `s1` is constant in the file, but
`x.std(axis=0)` on the stacked matrix returns 6.5e-11, not 0, because of float
rounding. A guard of `std > 0` lets it through, and the scaler then divides rounding
noise by 6.5e-11, turning it into a feature with weight 3e-9 that a drift of 2.0
would amplify to about 100 cycles. The reference solution uses a relative tolerance
(`std <= 1e-8 * max(|mean|, 1)` counts as constant). The hidden test only drifts the
exactly-constant sensors, so the plain `std > 0` fix also passes, which is fair.
*Things to try:* add `s1` to the drift test and see which agents use a tolerance.

The last hidden test is a sanity floor: official test RMSE ≤ 24 (the fixed pipeline
gets 22.55), recorded as a metric.

## 5. Running them

```bash
uv sync                                                    # installs numpy too
uv run miniagent tasks                                     # the two new tasks are listed
uv run --env-file .env miniagent run cmapss-debug-failed-run --keep
uv run --env-file .env miniagent run cmapss-rul-target --keep
```

Cost: these runs are longer than the toy tasks (up to 40 and 50 steps). With Sonnet
and prompt caching, my rough guess is 0.30 to 1.50 USD per run (not measured yet), so try
the debugging task first. To compare with Claude Code, copy a task's `repo/`, add the
training file under `data/`, give Claude Code the prompt from `task.yaml`, then run
`uv run miniagent grade <task> <that folder>` to score it the same way.
