"""
===============================================================================
MODEL TIERING AND COST TRACKING                                      [CORE]
===============================================================================

WHAT THIS DOES
    Routes each task to an appropriately sized model and records what every
    call cost, in tokens and dollars.

WHY TIER AT ALL
    Zomato's published lesson: use big models where judgement matters and
    small ones where it does not. They ran 70B for intent detection and 8B for
    chat completion. The same split maps onto Claude:

        classification / judging  -> Opus 5     (accuracy matters most)
        chat completion, JSON->text -> Haiku 4.5 (latency and cost matter most)

    Getting the classification wrong sends the whole turn down the wrong path,
    so that call is worth paying for. Rewriting a JSON blob as a sentence is
    not, and it sits directly in the customer's latency budget.

THE LATENCY BUDGET
    The target is a sub-10-second reply. Three things keep it there, and only
    one of them is model size:
      1. Tiering: the user-visible generation runs on the fast model.
      2. Token reduction: the function classifier fetches only the data the
         question needs, so prompts stay small (see function_classifier.py).
      3. Short outputs: response time scales roughly linearly with output
         tokens, so the prompt asks for brevity and max_tokens is capped.

COST FIGURES
    Per-million-token prices as of the model table this was written against.
    They are used only for the dashboard's cost estimate - update them if
    pricing moves.
===============================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import anthropic

from . import config


class Task(str, Enum):
    """What a call is for. Determines which model handles it."""

    CLASSIFY = "classify"        # function + action classification: accuracy
    GENERATE = "generate"        # customer-facing reply: latency
    NARRATE = "narrate"          # JSON -> natural language: latency
    JUDGE = "judge"              # LLM-as-judge evaluation: accuracy
    VISION = "vision"            # image authenticity: accuracy


# Task -> model. The whole tiering policy, in one dict.
MODEL_FOR: dict[Task, str] = {
    Task.CLASSIFY: config.CLASSIFIER_MODEL,
    Task.GENERATE: config.CHAT_MODEL,
    Task.NARRATE: config.CHAT_MODEL,
    Task.JUDGE: config.JUDGE_MODEL,
    Task.VISION: config.VISION_MODEL,
}

# USD per 1M tokens: (input, output).
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICING.get(model, (0.0, 0.0))
    return (input_tokens * inp + output_tokens * out) / 1_000_000


@dataclass
class CallRecord:
    task: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost: float


@dataclass
class Usage:
    """Accumulates every model call made while handling one turn."""

    calls: list[CallRecord] = field(default_factory=list)

    def add(self, record: CallRecord) -> None:
        self.calls.append(record)

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_cost(self) -> float:
        return sum(c.cost for c in self.calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.total_cost, 6),
            "calls": [
                {
                    "task": c.task, "model": c.model, "latency_ms": c.latency_ms,
                    "in": c.input_tokens, "out": c.output_tokens,
                    "cost": round(c.cost, 6),
                }
                for c in self.calls
            ],
        }


class LLM:
    """Thin wrapper that picks the model, times the call and records the cost."""

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client
        self.usage = Usage()

    @property
    def client(self) -> anthropic.Anthropic:
        # Constructed lazily so importing this module never needs credentials.
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def complete(
        self,
        task: Task,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Any:
        """One plain text completion, billed and timed."""
        model = MODEL_FOR[task]
        started = time.perf_counter()
        response = self.client.messages.create(
            model=model, max_tokens=max_tokens, system=system, messages=messages, **kwargs
        )
        self._record(task, model, response, started)
        return response

    def parse(
        self,
        task: Task,
        system: str,
        messages: list[dict[str, Any]],
        output_format: Any,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Any:
        """Structured output via messages.parse, billed and timed.

        Every classifier in the pipeline uses this: a guaranteed-shape result
        means the downstream code never has to parse free text, which is where
        classification pipelines usually break.
        """
        model = MODEL_FOR[task]
        started = time.perf_counter()
        response = self.client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            output_format=output_format,
            **kwargs,
        )
        self._record(task, model, response, started)
        return response

    def _record(self, task: Task, model: str, response: Any, started: float) -> None:
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = getattr(response, "usage", None)
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        self.usage.add(
            CallRecord(
                task=task.value, model=model, input_tokens=inp, output_tokens=out,
                latency_ms=latency_ms, cost=cost_usd(model, inp, out),
            )
        )


def text_of(response: Any) -> str:
    """Concatenate the text blocks of a response, ignoring the rest."""
    return "".join(b.text for b in response.content if b.type == "text").strip()
