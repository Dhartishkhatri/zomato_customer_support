"""
===============================================================================
EVALUATION - LLM-as-judge plus human-in-the-loop grading             [CORE]
===============================================================================

FOUR METRICS, each scored 0..1:

    FACTUAL CONSISTENCY    Does the reply agree with the backend data it was
                           given? Catches the failure that matters most - an
                           invented ETA or amount.
    INFORMATION RELEVANCE  Does it answer what was actually asked, rather than
                           something adjacent?
    GUIDELINE COMPLIANCE   Tone, length, no promises, no internal vocabulary,
                           no claiming to be an AI.
    PAIN POINT RESOLUTION  Is the customer's problem actually further along,
                           or were they just politely acknowledged?

WHY THE JUDGE RUNS ON THE BIG MODEL
    It grades the small model's work. A judge no stronger than the thing it
    judges cannot reliably catch its mistakes, so this is routed to Task.JUDGE
    (Opus 5) while chat runs on Haiku.

WHY SAMPLING
    Grading every turn would roughly double cost and add latency to live
    traffic. EVAL_SAMPLE_RATE controls the share of live turns graded
    asynchronously. Every deliberate system CHANGE should still be graded in
    full via `evaluate_transcript` on a fixed set before it ships.

HUMAN IN THE LOOP
    `record_human_grade` writes the same four scores with grader='human'.
    Keeping both in one table means you can measure the judge against humans -
    if they diverge, trust the humans and re-tune the judge prompt.
===============================================================================
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from . import config
from .db import get_db
from .llm import LLM, Task

_SYSTEM = """You grade a customer-support reply from a food-delivery chatbot.

You get: the customer's message, the factual CONTEXT the bot was given, and the
bot's REPLY. Score each dimension from 0.0 to 1.0.

FACTUAL_CONSISTENCY - does every factual claim in the reply follow from the
context? Invented ETAs, amounts, order ids or statuses score 0. A reply that
makes no factual claims scores 1.0.

INFORMATION_RELEVANCE - does it address what the customer actually asked?
Generic acknowledgement of a specific question scores low.

GUIDELINE_COMPLIANCE - short and warm; apologises once at most; promises
nothing (no "your refund has been processed" unless the context says so);
never mentions AI, bots, models, tools, internal checks or scores; never names
another customer or a courier's surname.

PAIN_POINT_RESOLUTION - is the customer's problem meaningfully advanced -
answered, actioned, or properly handed over? Politely saying nothing scores
low even when it reads nicely.

Be strict. A reply that sounds good but invents a delivery time is a failure,
not a near-miss. Cite the specific phrase behind any score below 0.7."""


class JudgeScores(BaseModel):
    factual_consistency: float = Field(ge=0.0, le=1.0)
    information_relevance: float = Field(ge=0.0, le=1.0)
    guideline_compliance: float = Field(ge=0.0, le=1.0)
    pain_point_resolution: float = Field(ge=0.0, le=1.0)
    notes: str = Field(description="One or two sentences on the weakest dimension.")


@dataclass
class EvalResult:
    scores: JudgeScores
    session_id: str | None = None

    @property
    def lowest(self) -> float:
        return min(
            self.scores.factual_consistency,
            self.scores.information_relevance,
            self.scores.guideline_compliance,
            self.scores.pain_point_resolution,
        )

    @property
    def needs_review(self) -> bool:
        return self.lowest < config.EVAL_ALERT_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scores.model_dump(),
            "lowest": round(self.lowest, 3),
            "needs_review": self.needs_review,
        }


def should_sample(rate: float | None = None) -> bool:
    """Whether this turn is picked for automated grading."""
    return random.random() < (config.EVAL_SAMPLE_RATE if rate is None else rate)


def judge_turn(
    customer_message: str,
    context: str,
    reply: str,
    llm: LLM | None = None,
    session_id: str | None = None,
    persist: bool = True,
) -> EvalResult | None:
    """Grade one turn. Returns None if the judge could not be reached."""
    llm = llm or LLM()
    try:
        response = llm.parse(
            Task.JUDGE,
            system=_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"CUSTOMER MESSAGE:\n{customer_message}\n\n"
                        f"CONTEXT GIVEN TO THE BOT:\n{context or '(none)'}\n\n"
                        f"BOT REPLY:\n{reply}"
                    ),
                }
            ],
            output_format=JudgeScores,
            max_tokens=1024,
        )
    except Exception:
        return None

    result = EvalResult(scores=response.parsed_output, session_id=session_id)
    if persist:
        record_grade(session_id, "llm_judge", result.scores.model_dump())
    return result


def record_grade(session_id: str | None, grader: str, scores: dict[str, Any]) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO eval_results (session_id, grader, factual_consistency, "
            "information_relevance, guideline_compliance, pain_point_resolution, notes) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                session_id, grader,
                scores.get("factual_consistency"), scores.get("information_relevance"),
                scores.get("guideline_compliance"), scores.get("pain_point_resolution"),
                scores.get("notes", ""),
            ),
        )


def record_human_grade(session_id: str, scores: dict[str, float], notes: str = "") -> None:
    """Human-in-the-loop grading, stored beside the judge's for comparison."""
    record_grade(session_id, "human", {**scores, "notes": notes})


def judge_vs_human() -> dict[str, Any]:
    """Mean absolute gap between the judge and humans, per metric.

    If this drifts, trust the humans and re-tune the judge prompt - an
    uncalibrated judge silently corrupts every quality number on the dashboard.
    """
    metrics = (
        "factual_consistency", "information_relevance",
        "guideline_compliance", "pain_point_resolution",
    )
    with get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM eval_results WHERE session_id IS NOT NULL"
        ).fetchall()]

    by_session: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        by_session.setdefault(row["session_id"], {}).setdefault(row["grader"], []).append(row)

    gaps: dict[str, list[float]] = {m: [] for m in metrics}
    for graders in by_session.values():
        if "human" not in graders or "llm_judge" not in graders:
            continue
        human, judge = graders["human"][-1], graders["llm_judge"][-1]
        for metric in metrics:
            if human.get(metric) is not None and judge.get(metric) is not None:
                gaps[metric].append(abs(human[metric] - judge[metric]))

    return {
        "compared_sessions": sum(
            1 for g in by_session.values() if "human" in g and "llm_judge" in g
        ),
        "mean_absolute_gap": {
            m: round(sum(v) / len(v), 3) if v else None for m, v in gaps.items()
        },
    }


def weekly_digest() -> dict[str, Any]:
    """The summary Zomato post to Slack each week."""
    from .metrics import dashboard

    data = dashboard()
    with get_db() as conn:
        weak = [dict(r) for r in conn.execute(
            "SELECT session_id, factual_consistency, information_relevance, "
            "guideline_compliance, pain_point_resolution, notes FROM eval_results "
            "WHERE grader = 'llm_judge' AND (factual_consistency < ? OR "
            "information_relevance < ? OR guideline_compliance < ? OR "
            "pain_point_resolution < ?) ORDER BY id DESC LIMIT 10",
            (config.EVAL_ALERT_THRESHOLD,) * 4,
        ).fetchall()]
        low_csat = [dict(r) for r in conn.execute(
            "SELECT session_id, rating, comment FROM chat_ratings WHERE rating <= 2 "
            "ORDER BY id DESC LIMIT 10"
        ).fetchall()]

    return {
        "headline": {
            "csat": data["csat"]["average"],
            "containment_pct": data["containment"]["pct"],
            "p95_response_ms": data["response_time"]["p95_ms"],
            "cost_per_session_usd": data["cost"]["per_session_usd"],
        },
        "quality": data["quality"],
        "turns_needing_review": weak,
        "low_csat_chats": low_csat,
        "judge_calibration": judge_vs_human(),
    }
