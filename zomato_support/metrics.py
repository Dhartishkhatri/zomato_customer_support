"""
===============================================================================
METRICS - the four numbers the system is judged on                   [CORE]
===============================================================================

    1. CSAT               average chat rating, 1-5
    2. CONTAINMENT %      share of chats the bot handled without a human
    3. COST               dollars per conversation, and per contained chat
    4. RESPONSE TIME      p50 / p95 latency against the 10-second target

WHY CONTAINMENT NEEDS CARE
    Containment is trivially gameable: refuse to escalate and it hits 100%
    while customers suffer. It is only meaningful read ALONGSIDE CSAT. A rise
    in containment with flat-or-rising CSAT is a real improvement; a rise with
    falling CSAT means the bot is trapping people.

    The dashboard therefore always shows the two together, and separately
    reports how often the policy layer REJECTED an escalation the model asked
    for. That rejection rate is the health check on the stabiliser itself: a
    little is the layer doing its job, a lot means the model has learned to
    escalate reflexively and needs prompt work.
===============================================================================
"""

from __future__ import annotations

import json
from typing import Any

from . import config
from .db import get_db
from .llm import Usage


def record_turn(
    session_id: str,
    latency_ms: int,
    classify_ms: int,
    generate_ms: int,
    action_ms: int,
    usage: Usage,
    escalated: bool,
    escalation_reason: str | None,
    escalation_upheld: bool | None,
    functions: list[str],
    action_proposed: str,
    guardrail_blocked: bool,
) -> None:
    """One row per assistant turn. Everything the dashboard reads comes from here."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO turn_metrics (session_id, latency_ms, classify_ms, generate_ms, "
            "action_ms, input_tokens, output_tokens, cost_usd, escalated, "
            "escalation_reason, escalation_upheld, functions_called, action_proposed, "
            "guardrail_blocked) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                session_id, latency_ms, classify_ms, generate_ms, action_ms,
                usage.input_tokens, usage.output_tokens, round(usage.total_cost, 6),
                1 if escalated else 0, escalation_reason,
                None if escalation_upheld is None else (1 if escalation_upheld else 0),
                json.dumps(functions), action_proposed,
                1 if guardrail_blocked else 0,
            ),
        )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def dashboard() -> dict[str, Any]:
    """Every figure the ops dashboard shows, in one query pass."""
    with get_db() as conn:
        turns = [dict(r) for r in conn.execute("SELECT * FROM turn_metrics").fetchall()]
        sessions = [dict(r) for r in conn.execute("SELECT * FROM chat_sessions").fetchall()]
        ratings = [r["rating"] for r in conn.execute("SELECT rating FROM chat_ratings").fetchall()]
        actions = [dict(r) for r in conn.execute("SELECT * FROM action_log").fetchall()]
        evals = [dict(r) for r in conn.execute(
            "SELECT * FROM eval_results WHERE grader = 'llm_judge'"
        ).fetchall()]

    latencies = [t["latency_ms"] for t in turns if t["latency_ms"]]
    total_cost = sum(t["cost_usd"] or 0 for t in turns)

    # Containment is a property of a CONVERSATION, not of a turn.
    finished = [s for s in sessions if s["contained"] is not None]
    contained = [s for s in finished if s["contained"] == 1]
    containment_pct = (len(contained) / len(finished) * 100) if finished else 0.0

    requested = [t for t in turns if t["escalation_upheld"] is not None]
    rejected = [t for t in requested if t["escalation_upheld"] == 0]

    proposed_actions = [a for a in actions if a["proposed"]]
    verified_actions = [a for a in proposed_actions if a["verified"]]
    executed_actions = [a for a in proposed_actions if a["executed"]]

    def avg(rows: list[dict[str, Any]], key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    return {
        "csat": {
            "average": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "count": len(ratings),
            "distribution": {str(i): ratings.count(i) for i in range(1, 6)},
        },
        "containment": {
            "pct": round(containment_pct, 1),
            "contained": len(contained),
            "total_sessions": len(finished),
            "escalations_requested": len(requested),
            "escalations_rejected_by_policy": len(rejected),
            # High values here mean the model is escalating reflexively.
            "policy_rejection_pct": (
                round(len(rejected) / len(requested) * 100, 1) if requested else 0.0
            ),
        },
        "response_time": {
            "p50_ms": int(_percentile(latencies, 50)),
            "p95_ms": int(_percentile(latencies, 95)),
            "target_ms": config.TARGET_RESPONSE_MS,
            "within_target_pct": (
                round(
                    sum(1 for l in latencies if l <= config.TARGET_RESPONSE_MS)
                    / len(latencies) * 100, 1
                )
                if latencies else 0.0
            ),
            "turns": len(turns),
        },
        "cost": {
            "total_usd": round(total_cost, 4),
            "per_turn_usd": round(total_cost / len(turns), 5) if turns else 0.0,
            "per_session_usd": (
                round(total_cost / len(sessions), 5) if sessions else 0.0
            ),
            "input_tokens": sum(t["input_tokens"] or 0 for t in turns),
            "output_tokens": sum(t["output_tokens"] or 0 for t in turns),
        },
        "actions": {
            "proposed": len(proposed_actions),
            "passed_verification": len(verified_actions),
            "executed": len(executed_actions),
            # Proposals the eligibility check caught - the value of step 2.
            "blocked_by_verification": len(proposed_actions) - len(verified_actions),
            "by_type": _count_by(proposed_actions, "action_type"),
        },
        "quality": {
            "factual_consistency": avg(evals, "factual_consistency"),
            "information_relevance": avg(evals, "information_relevance"),
            "guideline_compliance": avg(evals, "guideline_compliance"),
            "pain_point_resolution": avg(evals, "pain_point_resolution"),
            "graded_turns": len(evals),
        },
        "guardrail": {
            "blocked_turns": sum(1 for t in turns if t["guardrail_blocked"]),
        },
        "escalation_reasons": _count_by(
            [t for t in turns if t["escalation_reason"]], "escalation_reason"
        ),
    }


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = row.get(key) or "NONE"
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
