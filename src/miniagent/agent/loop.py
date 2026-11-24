"""The agent loop. Milestone 1: write this yourself.

Suggested shape (raw Anthropic Messages API with tool use, no framework):

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": task_prompt}]
    for step in range(config.max_steps):
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=config.system_prompt,
            tools=tool_schemas(config.tools),
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        tool_results = [run_tool(block, repo_path) for block in response.content
                        if block.type == "tool_use"]
        messages.append({"role": "user", "content": tool_results})

Things worth getting right early: count tokens from `response.usage`, print
each step with `rich`, and write every message to a JSONL transcript.
"""

from __future__ import annotations

from pathlib import Path

from miniagent.agent.types import AgentConfig, RunResult


def run_agent(task_id: str, task_prompt: str, repo_path: Path, config: AgentConfig) -> RunResult:
    """Run the agent on one task inside `repo_path` and return what happened."""
    raise NotImplementedError("Milestone 1: implement the agent loop here.")
