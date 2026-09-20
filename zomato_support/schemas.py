"""
===============================================================================
SHARED DATA TYPES                                                    [CORE]
===============================================================================

WHAT THIS DOES
    The records that move between the orchestrator, the policy gate, the
    forensics pipeline and the backends. No logic lives here.

WHY IT EXISTS
    ConversationContext in particular is the backbone of the whole design:
    it is the set of facts the policy gate is allowed to trust. Read its
    docstring before changing anything in policy.py.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Decision(str, Enum):
    """What the permission gate decided about one write-tool call."""

    ALLOW = "allow"               # Proceed.
    REQUIRE_APPROVAL = "require_approval"  # A human must sign off first.
    ESCALATE = "escalate"         # Hand to a specialist queue; stop acting.
    DENY = "deny"                 # Never allowed. Do not retry.


class ImageVerdict(str, Enum):
    """Outcome of the AI-morph / authenticity check.

    INCONCLUSIVE is a first-class, *correct* answer - not a failure. It is what
    the pipeline returns whenever it cannot separate ordinary compression from
    editing, and it routes to a human rather than to an accusation.
    """

    AUTHENTIC = "authentic"
    INCONCLUSIVE = "inconclusive"
    MANIPULATED = "manipulated"
    AI_GENERATED = "ai_generated"


@dataclass(frozen=True)
class PolicyResult:
    """One gate decision, with enough context for a human to second-guess it."""

    decision: Decision
    rule_id: str      # e.g. "R11-image-manipulated" - traceable in policy.py
    reason: str       # Plain English, safe to show an internal reviewer.
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass
class ForensicSignal:
    """One measurement from the forensics pipeline.

    score  - 0..1, where 1 means "strongly indicates manipulation"
    weight - how much the aggregate should trust this signal
    detail - human-readable, shown in the CLI and sent to the vision judge
    raw    - the underlying numbers, for debugging and for the audit log
    """

    name: str
    score: float
    weight: float
    detail: str
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImageAnalysis:
    """The full result of checking one image. Stored on the context; only a
    redacted subset (see pipeline.analysis_for_agent) ever reaches the model."""

    verdict: ImageVerdict
    confidence: float
    manipulation_score: float
    signals: list[ForensicSignal]
    summary: str
    vision_used: bool = False
    vision_notes: str = ""
    vision_verdict: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 3),
            "manipulation_score": round(self.manipulation_score, 3),
            "summary": self.summary,
            "vision_used": self.vision_used,
            "vision_verdict": self.vision_verdict,
            "vision_notes": self.vision_notes,
            "signals": [s.to_dict() for s in self.signals],
        }


@dataclass
class ConversationContext:
    """Facts established during one conversation - THE TRUST BOUNDARY.

    This is the most important type in the codebase. The policy gate makes its
    decisions from this object and from the backends, never from what the model
    asserts in its tool arguments.

    Concretely: `orders_seen` is populated only when get_order actually
    returned an order belonging to this customer. That is why a refund on an
    order id the model simply invented gets denied - the id is not in here.
    """

    customer_id: str
    ticket_id: str
    channel: str = "chat"

    orders_seen: dict[str, dict[str, Any]] = field(default_factory=dict)
    customer_record: dict[str, Any] | None = None
    image_analyses: dict[str, ImageAnalysis] = field(default_factory=dict)
    refunds_issued: list[dict[str, Any]] = field(default_factory=list)
    escalated: bool = False
    pending_approvals: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentResult:
    """What SupportAgent.handle() gives back to the caller."""

    reply: str                          # The text to send the customer.
    context: ConversationContext
    audit: list[dict[str, Any]]
    turns: int
    escalated: bool
    guardrail_report: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
