# miniagent's sandbox: where the agent's shell commands and tests run.
# Built automatically on first use (see src/miniagent/sandbox.py). The tag is a hash
# of this file and the git shim, so editing either triggers a rebuild.
ARG BASE=python:3.12-slim
FROM ${BASE}

# git (read-only, see below) and ripgrep, for the agent's own searches.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ripgrep \
 && rm -rf /var/lib/apt/lists/*

# What benchmark tasks import, plus uv so `run-repo` can run a project's own
# `uv run pytest`. Keep in step with the `tasks` group in pyproject.toml.
RUN pip install --no-cache-dir "numpy>=2.0" "pytest>=8.2" uv

# Same read-only git shim as outside the sandbox, ahead of the real git on PATH.
COPY git-readonly /usr/local/bin/git
RUN chmod 755 /usr/local/bin/git

# The root filesystem is mounted read-only at run time; /tmp is the only writable
# place besides /work, so caches and HOME point there.
ENV HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_TERMINAL_PROMPT=0
WORKDIR /work
