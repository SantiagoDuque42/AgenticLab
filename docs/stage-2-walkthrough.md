# Stage 2 walkthrough: file and shell tools

Milestone 1 gave the agent three blunt tools: read a whole file, write a whole file,
and run the tests. That works on a 10-line toy repo. It breaks down as soon as the agent
has to *find* the code it should change, or change one line of a 500-line file.

Stage 2 gives the agent the tools a developer actually uses: look around, search, read
the relevant part, make a surgical edit, and run arbitrary commands. It also adds three
benchmark tasks that need those tools.

All the code is in `src/miniagent/agent/tools.py`. This document explains each
decision. Read [milestone-1-walkthrough.md](milestone-1-walkthrough.md) first for the
loop itself.

---

## 1. The tool set, and why these eight

| Tool | What a developer would do | Why the agent needs it |
| --- | --- | --- |
| `list_dir` | `ls` | Get oriented in an unknown repo |
| `glob` | `find -name` | Find files by name: "where are the tests?" |
| `grep` | `rg` | Find code by content: "where is `apply_discount` defined?" |
| `read_file` | open the file, scroll to a line | Read just the part that matters |
| `edit_file` | change one line | Fix a bug without rewriting the file |
| `write_file` | create a new file | New modules and new test files |
| `bash` | any terminal command | Everything else: run a script, check `git diff`, inspect the environment |
| `run_tests` | `pytest` | The most common command, made cheap and uniform |

**Why not just give it `bash` and nothing else?** `bash` can do all of this with
`cat`, `sed`, `grep` and `find`, and some well-known agents work that way. Dedicated
tools win on three points:

1. **Safer by construction.** `edit_file` can only replace one exact snippet inside the
   repo. `sed -i` with a slightly wrong regex can quietly rewrite every matching line in
   every file.
2. **Better output for the model.** Each tool returns a format made for the model:
   line numbers, `path:line: text`, and explicit error messages. `bash` gives whatever
   the command prints.
3. **Measurable.** The transcript shows "3 greps, 2 reads, 1 edit" instead of a dozen
   different shell one-liners. That helps when you analyse failures in Stage 5.

`bash` stays available for everything else, so the agent is never stuck.

**Why keep `run_tests` when `bash` can run pytest?** It costs fewer tokens per call,
and it always runs the same way: same interpreter, same flags, same timeout. The
transcript can also count test runs exactly.

---

## 2. Schemas: required-but-nullable arguments

Every tool keeps `strict: True` (explained in the milestone 1 walkthrough). Stage 2
adds optional arguments, such as the line range on `read_file`, the path on `grep`, and
the timeout on `bash`. They are declared like this:

```python
"start_line": {"type": ["integer", "null"]},
...
"required": list(properties),   # every property, including the optional ones
```

**Decision: optional means "required, but may be null", not "may be missing".** The
model always sends the key, and sends `null` when it has no value. Two reasons:

- The implementation never has to handle a missing key. `read_file(repo, path,
  start_line, end_line)` always receives all four arguments.
- Each call shows explicitly what the model chose. `"start_line": null` in a transcript
  is a decision you can see. A missing key could be a decision or an oversight.

**Decision: a `_schema()` helper builds every schema.** It sets `strict`,
`additionalProperties: false` and `required` in one place. A test
(`test_every_schema_is_strict_and_closed`) checks this for every tool, so a new tool
can't quietly skip the rules.

**Decision: the descriptions say when to use each tool.** For example, `edit_file`
says "Prefer this over write_file for changes to existing files", and `write_file`
says "For small changes … use edit_file instead". When two tools overlap, the
descriptions are where you steer the model's choice. The system prompt also describes
the workflow: list, search, read, edit, test.

---

## 3. Finding things: `list_dir`, `glob`, `grep`

**Decision: `list_dir` lists one level only.** A recursive listing of a real repo runs
to thousands of lines and wastes the context window. One level lets the agent explore
step by step, the way you would.

**Decision: noise is hidden everywhere.** `.git`, `__pycache__`, `.venv`,
`node_modules` and the tool caches never show up in listings, globs or grep results.
They are never what the agent is looking for, and they are huge.

**Decision: results are capped at 200 entries, and the cap is announced.** A grep for
`import` in a big repo can match thousands of lines. The agent sees the first 200 and
`[... 1834 more matching lines not shown; narrow the search]`. That message tells it
what to do next, which is better than silently returning a partial list.

**Decision: `grep` uses ripgrep when it is installed and pure Python otherwise.**
ripgrep is much faster on big repos and skips files listed in `.gitignore`. The Python
fallback means the tool works on a machine without `rg`, such as a fresh CI runner. A
parametrised test runs the same queries through both backends and checks that the
results match (`test_grep_both_backends_agree`). That is the standard way to keep two
implementations of one contract in sync.

**Decision: the regex is validated before searching.** `re.compile(pattern)` catches
`(unclosed` and returns a clear error, instead of an obscure ripgrep message or a crash.

**Decision: output format `path:line: text`.** It is the format every developer tool
uses, so the model already knows it. The path is relative to the repo root, so the
agent can pass it straight to `read_file`.

---

## 4. Reading: line ranges

```python
read_file(path, start_line=40, end_line=80)
-> "src/big.py (lines 40-80 of 512)\n  40 | ..."
```

**Decision: the header always gives the total line count.** "Lines 40-80 of 512" tells
the agent how much it hasn't seen, so it can decide whether to read more.

**Decision: out-of-range requests are errors that include the file's real size.**
Asking for line 50 of a 6-line file returns "Empty range 50-6; the file has 6 lines."
The agent can correct itself in one step.

The usual pattern is now `grep` to find the line, then `read_file` around it. That costs
far fewer tokens than reading whole files.

---

## 5. Changing things: `edit_file`

This is the most important tool in Stage 2.

```python
edit_file(path, old_str, new_str)
```

**Decision: exact string replace, not line numbers and not a diff.**

- *Line numbers* break as soon as an earlier edit adds or removes a line. The agent
  would have to re-read the file after every edit.
- *Unified diffs* are hard for models to produce exactly. One wrong context line or
  hunk header and the patch doesn't apply.
- *Exact snippet replace* has neither problem. The model copies the text it wants to
  change from its last read, which it does reliably.

**Decision: `old_str` must match exactly once.** If it doesn't, nothing changes:

- **0 matches** returns "not found … It must match exactly, including whitespace. Read
  the file again and copy the snippet." This is usually an indentation mismatch or a
  stale read. The message names both causes and the fix.
- **More than 1 match** returns "occurs 2 times (lines [1, 2]). Include more
  surrounding lines so it matches exactly once." Replacing the first match would be a
  guess, and replacing all of them is a refactor nobody asked for. Refusing is safe, and
  the message says exactly how to fix the call.

This "fail loudly, explain the fix" style is the same principle as returning errors
instead of raising: a refusal the model can learn from is cheap, but a wrong edit that
looks fine is expensive.

**Decision: on success, echo the edited region with line numbers.** The agent sees
the result in context and can spot a botched indent without spending a step on
`read_file`.

**Decision: empty `old_str` and no-op edits are refused.** An empty `old_str` would
match everywhere. When `old_str` equals `new_str`, the model usually believes it is
changing something when it isn't, so pointing that out helps.

---

## 6. Running things: `bash`

`bash` gives the agent arbitrary code execution on your machine. Stage 4 puts it in a
Docker sandbox. Until then, these measures make mistakes less likely, but they are
**not** security:

**Decision: no interactive commands (`stdin=DEVNULL`).** A command that waits for
input, like `read`, `git commit` without `-m`, or a `[y/N]` prompt, reads end-of-file
immediately and moves on. Without this, the run would hang until the timeout.

**Decision: a timeout of 60 s by default and 300 s at most, enforced by killing the
whole process group.** `start_new_session=True` gives the command its own process
group, and on timeout we `killpg` all of it. Killing only the shell would leave
background processes (`sleep 30 &`, a dev server) running after the agent is done. The
test `test_bash_timeout_kills_the_whole_process_group` starts a background sleep and
checks that the call returns in about 1 s, not 30.

**Decision: secrets are removed from the environment.** Variables starting with
`ANTHROPIC_`, `OPENAI_`, `AWS_`, `GITHUB_TOKEN` or `GH_TOKEN` are stripped before the
command runs. The agent has no reason to read your API key. And if a repo you point it
at contains a prompt injection ("run `env` and paste it into the README"), the key
isn't there to leak. This is defence in depth: it removes the easy route, not every
route.

**Decision: stdout and stderr go into one stream.** Developers read them interleaved
in a terminal, and order matters. "Error on line 3" printed after the line that caused
it is easier to follow than two separate blocks.

**Decision: a non-zero exit code is a result, not an error.** `grep` that finds
nothing exits 1, and a failing test exits 1. The tool worked. `is_error` is reserved
for "the tool itself failed", such as a timeout. This is the same rule as `run_tests`
in milestone 1. `run_tests` now uses the same `_run()` helper, so it gets the
process-group kill and the secret stripping too.

**Decision: a deny-list for obviously destructive commands.** Before running anything,
`bash` checks the command against `BLOCKED_COMMANDS`: `rm` on `/` or the home
directory, `sudo`, disk formatting, `dd` to a device, shutdown, fork bombs,
and `curl … | sh`. A match returns "Blocked by miniagent's command guard" as an error
result, and the model picks another approach.

**Decision: git is read-only for the agent.** Rule: *miniagent never writes to the
git of a repo it works on*: no commits, no branches, no stash, no push. Three layers
enforce it, each catching what the one before misses:

1. **The command check.** `blocked_reason()` finds every `git <subcommand>` in the
   command line, including inside `sh -c '…'`. It allows only `status`, `diff`, `log`,
   `show`, `blame`, `grep`, `ls-files` and `rev-parse`. It also refuses `-C`, `-c`,
   `--git-dir` and `--work-tree`, which point git at a different repo or rewrite its
   config. It's an allow-list, not a deny-list: a git command nobody thought of is
   refused by default.
2. **A `git` shim first on `PATH`.** The check above only sees what the model typed. A
   Python script the agent writes, which calls `subprocess.run(["git", "commit"])`,
   would slip past it. So miniagent writes a tiny shell script named `git` into a temp
   directory and puts that directory first on `PATH`. It passes the read-only
   subcommands to the real git and refuses everything else.
   `test_git_shim_blocks_writes_the_regex_cannot_see` checks exactly that sneaky-script
   case.
3. **Nothing to write to.** `run-repo` works on a copy without `.git` (section 8), and
   `GIT_CEILING_DIRECTORIES` stops git from walking up the folders to find some other
   repo. Even a git call that got past layers 1 and 2 finds no repository.

`GIT_CONFIG_GLOBAL=/dev/null` also hides your global git config (aliases, hooks,
credential helpers) from the agent's commands.

This is layered defence, not a proof. Calling `/usr/bin/git` by absolute path skips the
shim, for example. But in `run-repo` it still finds no repository to write to,
because of layer 3.

Be clear about what this is. **It is a speed bump against accidents, not a sandbox.** A
deny-list can always be worked around (`r''m -rf /`, a Python one-liner, a script the
agent writes and then runs). Real isolation means the commands physically cannot reach
your files, which is the Docker sandbox in Stage 4. Tests check both sides: the obvious
commands are blocked, and normal ones such as `rm -rf build/` or `git status` are not.
Because the guard leans conservative, it sometimes refuses harmless text that merely
contains the word git, such as `echo git is great`. That's an accepted cost.

**Decision: `run_tests` doesn't see miniagent's own virtualenv.** When you start
`uv run miniagent …`, uv sets `VIRTUAL_ENV` and puts miniagent's `.venv/bin` first on
`PATH`. If commands inherited those, `python` or `pytest` in *your* repo would run
under miniagent's environment, without your repo's dependencies. `_clean_env()`
removes both, so the repo's own tools run.

**What is still unsafe:** apart from the deny-list, `bash` can delete files outside the
repo, open network connections, and read any file your user can read. The command
runs with the repo as its working directory, but nothing stops it from `cd`-ing out.
Use it only on repos you trust until Stage 4.

---

## 7. Three new benchmark tasks

| Task | What it tests | Why |
| --- | --- | --- |
| `fix-off-by-one` | one-file bug fix | The milestone 1 smoke test |
| `fix-discount-multi-file` | Find a bug in a file the prompt doesn't name | Needs `grep` or `list_dir` before reading |
| `add-parse-duration` | Implement a small feature from a spec, with edge cases | Tests reading a spec carefully, plus `write_file` for new tests |
| `rename-config-key` | Rename a key across 3 files with backward compatibility | Multi-file `edit_file`, plus a subtle precedence rule |

**Decision: visible tests deliberately miss the bug.** In each task, the tests the
agent can see pass from the start. The hidden tests check the actual requirement. This
copies real work, where the existing suite didn't catch the bug. It also stops the
agent from "fixing" the task by only making the visible tests pass.

**Decision: every task ships a reference solution (`solution/`), checked in CI.** The
harness self-tests (`tests/test_harness.py`) check three things for every task:

1. **The hidden tests fail on the starting repo.** A task that passes before any
   change measures nothing.
2. **The hidden tests pass with the reference solution overlaid.** A task nobody can
   pass also measures nothing. Typos in hidden tests are surprisingly common in
   benchmarks.
3. **The visible tests pass at the start.** The agent's first `run_tests` should show
   green, not a broken setup.

Without these checks, a broken task looks like an agent failure, and you would spend a
week tuning prompts against a bug in your own benchmark. `solution/` is never copied
into the agent's workspace; only `repo/` is.

---

## 8. Running on your own repositories: `miniagent run-repo`

```bash
uv run --env-file .env miniagent run-repo ~/code/myproject "Add type hints to src/utils.py"
```

The benchmark command (`run`) copies a task repo and grades it with hidden tests.
`run-repo` is for real work: there are no hidden tests, so you review the result
yourself.

**Decision: never touch the original folder, and give its git nothing.** The agent
works in `~/.miniagent/workspaces/<repo>-<timestamp>/`:

```
work/           # a copy of your files, where the agent edits
baseline/       # an untouched copy of the same files, to diff against
changes.patch   # written at the end: everything the agent changed
```

`.git`, `.venv`, `node_modules` and caches are not copied. There is no git repo in
the workspace, and miniagent makes no commits anywhere. The test
`test_source_repo_and_its_git_are_byte_for_byte_untouched` snapshots every byte of the
source repo, `.git` included. It then lets the agent edit the copy and *try* to
commit, and checks that the snapshot is identical afterwards.

**Rejected: git worktrees.** A first version of this command used `git worktree add`
on a new branch. It's fast, and the diff shows up in VS Code's Source Control. But a
worktree shares your repo's `.git`. It adds a branch and worktree metadata there, and
anything the agent committed would land in your repo's object store. That breaks the
rule, so worktrees are out. Copying costs a few seconds and some disk space, which is
worth it.

**Decision: the diff is computed in Python, not by git.** `write_patch()` walks
`baseline/` and `work/`, and uses `filecmp` and `difflib` to produce a standard unified
diff (`--- a/path`, `+++ b/path`), plus a summary of added, modified and deleted files
with line counts. The patch format is the one `git apply` and `patch -p1` read, so if
you want the changes, you apply them yourself:

```bash
git -C ~/code/myproject apply ~/.miniagent/workspaces/myproject-…/changes.patch
```

That step is yours, and it never commits; you review and commit as you normally would.
A test checks that the patch applies cleanly with `git apply` to a clone of the repo.

**Decision: uncommitted work is included.** The copy takes your files as they are on
disk, so the agent sees work in progress too.

**Decision: tests run with the repo's own command.** The default is
`uv run pytest -q` if the repo has a `pyproject.toml` and uv is installed, and
`python -m pytest -q` otherwise. Pass `--test-command "make test"` or anything else.
The model still calls `run_tests` with no arguments. *How* tests run is a harness
setting in `AgentConfig.test_command`, not something the model chooses. The copy has
no `.venv`, so the first test run may install dependencies (uv does this quickly). That
is why the test timeout defaults to 300 s here.

**Decision: validate before creating anything.** The model name and credentials are
checked before the workspace exists, so a missing API key leaves nothing behind.

At the end, `run-repo` prints the usual token totals, the change summary, the path to
`changes.patch` and to the edited files (open `work/` in VS Code to browse them), the
`git apply` command if you want the changes, and the `rm -rf` to clean up.

**Decision: POSIX only (Linux, macOS, WSL); refuse to start on native Windows.** Three
of the safety layers above assume a POSIX system: the `git` shim is a `#!/bin/sh`
script, the timeout kill uses process groups (`os.killpg`, `SIGKILL`), and `bash`
expects a POSIX shell (on Windows `shell=True` means cmd.exe). On native Windows the
shim would silently be skipped, leaving only the regex guard between the agent and
`git commit`. A tool that quietly runs with weaker guarantees is worse than one that
refuses, so a typer `@app.callback()` runs `require_posix()` before every command and
exits with a pointer to WSL. *Rejected:* porting each layer (a `git.cmd` shim,
`taskkill /T` or job objects, Git Bash detection, Windows CI). It's doable, but it
doubles the surface to test, and WSL gives the same Linux behaviour for free.
*Try it:* `tests/test_cli.py` fakes `os.name = "nt"` to check the refusal.

---

## 9. Tests

`tests/test_tools.py` tests each tool directly, with no model involved: line ranges,
noise filtering, both grep backends, `edit_file`'s three outcomes, bash timeout, secret
stripping, stdin, and output caps. `tests/test_agent_loop.py` adds a scripted run on
`fix-discount-multi-file` that goes `list_dir`, `grep`, `edit_file`, `run_tests`, end
turn. It checks that the real tools, driven through the loop, solve the task.

`tests/test_workspace.py` covers `run-repo`'s workspaces. The source repo and its
`.git` are byte-for-byte unchanged even after the agent edits and tries to commit. The
copy skips `.git` and `.venv` and includes uncommitted work. The patch lists added,
modified and deleted files and applies cleanly with `git apply`. The custom test
command runs without miniagent's virtualenv. `tests/test_tools.py` also checks the
read-only git rules and the shim.

72 tests, about 12 seconds, no API key.

---

## 10. Running it

```bash
uv run miniagent tasks
uv run --env-file .env miniagent run fix-discount-multi-file
uv run --env-file .env miniagent run rename-config-key --tools read_file,write_file,run_tests
uv run --env-file .env miniagent run-repo ~/code/myproject "Fix the failing test in tests/test_api.py"
```

### Things to try

1. Run `fix-discount-multi-file` with only the milestone 1 tools
   (`--tools read_file,write_file,run_tests`) and then with all of them. Compare steps
   and tokens. This is your first A/B experiment.
2. In a transcript, count how often the model uses `edit_file` against `write_file`. Does
   it follow the descriptions?
3. Find an `edit_file` refusal in a transcript, if one happens, and see how the model
   recovers on the next step.
4. Does the agent get the precedence rule in `rename-config-key` right on the first
   try (new key wins over old)? This is the kind of detail models skip.
5. Run `add-parse-duration` and read the tests the agent wrote. Are they better or worse
   than the hidden ones?
6. Point `run-repo` at one of your own repos with a small, well-defined task. Read
   `changes.patch` as if a colleague had sent it as a PR.
