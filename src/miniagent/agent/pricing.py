"""What a run cost, in USD, from its token counts.

Prices are per million tokens, from Anthropic's price list as of 2026-11. Check
https://platform.claude.com/docs/en/about-claude/pricing before trusting them for
budgeting; a model missing from the table simply reports no cost.

Four kinds of input token, four prices:
- input: uncached input, full price;
- cache write: input written to the prompt cache, 1.25x input (5-minute cache);
- cache read: input served from the cache, about 0.05x to 0.1x input;
- output: what the model generates, including thinking.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float

    @property
    def cache_write(self) -> float:
        return self.input * 1.25


PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_read=0.10),
    "claude-sonnet-5": Price(input=2.00, output=10.00, cache_read=0.20),
    # Haiku 5.5: these are the rates for prompts up to 100K tokens; above that the
    # whole request is billed at $0.50 / $2.50, which a context limit keeps us under.
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_read=0.01),
}


def cost_usd(
    model: str, input_tokens: int, cache_read: int, cache_write: int, output_tokens: int
) -> float | None:
    price = PRICES.get(model)
    if price is None:
        return None
    total = (
        input_tokens * price.input
        + cache_read * price.cache_read
        + cache_write * price.cache_write
        + output_tokens * price.output
    )
    return total / 1_000_000
