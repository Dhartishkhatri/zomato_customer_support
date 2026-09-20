"""
===============================================================================
PRE-SEND GUARDRAIL CHECK                                             [CORE]
===============================================================================

WHAT THIS DOES
    Inspects the drafted reply after the agent loop finishes and BEFORE a
    single character reaches the customer. Returns a report; the orchestrator
    decides what to do with it.

WHY IT IS DETERMINISTIC (no model call)
    A checker that is itself an LLM can be talked out of its job by the very
    same context that produced the bad draft. Plain string and regex checks
    cannot be persuaded. This is the last line, so it does not negotiate.

TWO SEVERITIES
    BLOCKING - the reply must not be sent as drafted. The orchestrator asks
               for one rewrite, then falls back to a neutral holding reply.
    ADVISORY - logged for review, reply still goes out.

WHAT IT CATCHES, AND WHY EACH ONE MATTERS
    1. reply_too_long         - advisory; support replies should be short
    2. banned_phrase          - promises policy cannot honour; "I am an AI"
    3. internal_vocabulary    - THE BIG ONE. Blocks "forensic", "manipulated",
                                "AI generated", "fraud", "trust score", "EXIF"
                                and similar ever reaching a customer. The
                                forensics result is internal; a customer is
                                never told their photo was doubted.
    4. third_party_pii        - another customer's name or email
    5. partner_pii            - a courier's surname (first names only)
    6. ungrounded_amount      - any money figure that appears in NO tool result
                                or policy citation. This is what stops a
                                hallucinated refund amount being quoted.
    7. unbacked_refund_claim  - "your refund is on its way" when no refund
                                tool call actually succeeded
    8. approval_misrepresented- a pending approval described as settled
    9. escalation_not_communicated - advisory; escalated but the reply does not
                                say so

IS IT CRUCIAL?
    Yes. The gate stops bad ACTIONS; this stops bad STATEMENTS. They fail in
    different ways and you need both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import config
from .backends import Backends
from .schemas import ConversationContext

BLOCKING = "blocking"
ADVISORY = "advisory"

# Internal forensic and risk vocabulary. None of this may reach a customer,
# whatever the verdict was - KB-PHOTO-002 and KB-TONE-004.
_INTERNAL_VOCAB = (
    "forensic", "error level", "ela ", "manipulation score", "confidence score",
    "trust score", "ai-generated", "ai generated", "manipulated", "tampered",
    "doctored", "morphed", "photoshopped", "edited photo", "fake photo",
    "fraud", "fraudulent", "scam", "suspicious", "trust and safety",
    "trust & safety", "flagged your account", "velocity", "spectral",
    "metadata", "exif",
)

# Language that asserts money has moved.
_REFUND_ASSERTIONS = (
    "refund has been", "have refunded", "i've refunded", "i have refunded",
    "refund is on its way", "refund of", "has been credited", "have credited",
    "processed your refund", "processing your refund", "initiated your refund",
    "refund initiated", "refunded to", "money back to",
)

_AMOUNT = re.compile(r"(?:₹|INR\s*|Rs\.?\s*)(\d[\d,]*)", re.IGNORECASE)

# Money is quoted bare at least as often as it is marked: "a refund of 2500",
# "we'll credit 500 back". Catch a number only when a money word introduces it,
# and never when a time unit follows it, so "resolved within 24 hours" and
# "3-5 business days" stay clear.
_BARE_AMOUNT = re.compile(
    r"(?:refund(?:ed|ing)?|credit(?:ed|ing)?|reimburs\w+|amount|money back|pay(?:ing)? back)"
    r"[^.\n]{0,24}?(?<![\d.])(\d[\d,]*)(?>(?:\s*[-\u2013]\s*\d[\d,]*)?)"
    r"(?!\s*(?:hours?|hrs?|days?|minutes?|mins?|weeks?|months?|business|working|%|st|nd|rd|th))",
    re.IGNORECASE,
)


@dataclass
class Violation:
    code: str
    severity: str
    detail: str


@dataclass
class GuardrailReport:
    violations: list[Violation] = field(default_factory=list)
    checked_chars: int = 0

    @property
    def blocked(self) -> bool:
        return any(v.severity == BLOCKING for v in self.violations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "checked_chars": self.checked_chars,
            "violations": [
                {"code": v.code, "severity": v.severity, "detail": v.detail}
                for v in self.violations
            ],
        }

    def feedback(self) -> str:
        """The correction note handed back to the model for one retry."""
        lines = [
            "Your drafted reply failed the pre-send guardrail check and was not sent.",
            "Fix every point below and rewrite the reply. Do not call any more tools.",
        ]
        lines += [f"- [{v.code}] {v.detail}" for v in self.violations]
        return "\n".join(lines)


def _grounded_amounts(grounding_text: str) -> set[str]:
    """Amounts that appeared in a tool result or a retrieved policy chunk."""
    found = set()
    for match in re.finditer(r"\d[\d,]*", grounding_text):
        found.add(match.group(0).replace(",", ""))
    return found


def check_reply(
    reply: str,
    ctx: ConversationContext,
    backends: Backends,
    grounding_text: str,
) -> GuardrailReport:
    report = GuardrailReport(checked_chars=len(reply))
    low = reply.lower()

    # --- 1. Length ----------------------------------------------------------
    if len(reply) > config.MAX_REPLY_CHARS:
        report.violations.append(
            Violation(
                "reply_too_long",
                ADVISORY,
                f"Reply is {len(reply)} characters, over the {config.MAX_REPLY_CHARS} "
                "limit. Support replies should be short.",
            )
        )

    # --- 2. Commitments the policy cannot back ------------------------------
    for phrase in config.BANNED_PHRASES:
        if phrase in low:
            report.violations.append(
                Violation(
                    "banned_phrase",
                    BLOCKING,
                    f"Reply contains the prohibited phrase '{phrase}'.",
                )
            )

    # --- 3. Internal vocabulary leak ----------------------------------------
    for term in _INTERNAL_VOCAB:
        if term in low:
            report.violations.append(
                Violation(
                    "internal_vocabulary",
                    BLOCKING,
                    f"Reply exposes internal review vocabulary ('{term.strip()}'). "
                    "Describe the case as 'under review' without naming any check, "
                    "score or suspicion.",
                )
            )

    # --- 4. Third-party privacy ---------------------------------------------
    for record in backends.crm.all_customers():
        if record["customer_id"] == ctx.customer_id:
            continue
        if record["name"].lower() in low or record["email"].lower() in low:
            report.violations.append(
                Violation(
                    "third_party_pii",
                    BLOCKING,
                    f"Reply names another customer ({record['customer_id']}).",
                )
            )

    for order_id in ctx.orders_seen:
        tracking = backends.tracking.get(order_id)
        if not tracking or not tracking.get("partner_name"):
            continue
        parts = tracking["partner_name"].split()
        surname = parts[-1] if len(parts) > 1 else ""
        if surname and len(surname) > 2 and surname.lower() in low:
            report.violations.append(
                Violation(
                    "partner_pii",
                    BLOCKING,
                    "Reply includes a delivery partner's surname; first name only "
                    "(KB-TONE-004).",
                )
            )

    # --- 5. Ungrounded money figures ----------------------------------------
    allowed = _grounded_amounts(grounding_text)
    flagged: set[str] = set()
    for pattern in (_AMOUNT, _BARE_AMOUNT):
        for match in pattern.finditer(reply):
            value = match.group(1).replace(",", "")
            if value in allowed or value in flagged:
                continue
            flagged.add(value)
            report.violations.append(
                Violation(
                    "ungrounded_amount",
                    BLOCKING,
                    f"Reply states {config.CURRENCY_SYMBOL}{match.group(1)}, which "
                    "appears in no tool result or policy citation. Only quote amounts "
                    "a tool actually returned.",
                )
            )

    # --- 6. Refund claimed but never issued ---------------------------------
    asserts_refund = any(p in low for p in _REFUND_ASSERTIONS)
    if asserts_refund and not ctx.refunds_issued:
        report.violations.append(
            Violation(
                "unbacked_refund_claim",
                BLOCKING,
                "Reply tells the customer a refund has been made, but no refund tool "
                "call succeeded in this conversation.",
            )
        )

    # --- 7. Escalation consistency ------------------------------------------
    if ctx.escalated:
        signals = ("review", "specialist", "team", "get back", "24 hours", "look into")
        if not any(s in low for s in signals):
            report.violations.append(
                Violation(
                    "escalation_not_communicated",
                    ADVISORY,
                    "Case was escalated but the reply does not tell the customer it is "
                    "under review or when to expect a response.",
                )
            )

    if ctx.pending_approvals and not ctx.refunds_issued:
        if any(p in low for p in ("has been credited", "refund has been", "have refunded")):
            report.violations.append(
                Violation(
                    "approval_misrepresented",
                    BLOCKING,
                    "A supervisor approval is still pending; the reply presents the "
                    "refund as settled.",
                )
            )

    return report


SAFE_FALLBACK = (
    "Thanks for your patience. I've passed your case to a specialist on our team who "
    "will review it and get back to you within 24 hours with an update. You'll get an "
    "email at the address on your account as soon as there's news."
)
