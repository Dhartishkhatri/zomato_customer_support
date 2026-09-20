"""
===============================================================================
ESCALATION - hand to a human and stop acting                         [CORE]
===============================================================================

NEVER GATED, and that is the point. An agent must always be able to escalate,
including - especially - when it is unsure whether it is allowed to do
anything else. If escalation could be blocked, an uncertain agent would be
left with only bad options.

Escalating is not a failure mode. A wrong refund and a wrong accusation both
cost far more than a handover.
===============================================================================
"""

from __future__ import annotations

from typing import Any

from ..backends import Backends
from ..schemas import ConversationContext

# Queues a case can be routed to. The gate picks one of these automatically
# when a rule escalates (see the "queue" key in each rule's evidence dict).
QUEUES = (
    "trust-and-safety",        # edited/AI photos, refund-velocity fraud review
    "food-safety",             # illness, injury, foreign objects
    "grievance-officer",       # claims past the refund window
    "legal-escalations",       # legal, regulatory or press threats
    "delivery-investigations", # marked delivered but not received
    "supervisor",              # catch-all
)


def escalate_to_human(
    backends: Backends,
    ctx: ConversationContext,
    queue: str,
    reason: str,
    customer_summary: str,
) -> dict[str, Any]:
    if queue not in QUEUES:
        queue = "supervisor"

    escalation = backends.crm.escalate(
        ctx.ticket_id,
        queue,
        reason,
        # Everything the human picking this up needs to reconstruct the case.
        context={
            "customer_id": ctx.customer_id,
            "orders_discussed": list(ctx.orders_seen),
            "image_verdicts": {n: a.verdict.value for n, a in ctx.image_analyses.items()},
            "customer_summary": customer_summary,
        },
    )
    ctx.escalated = True
    return {
        "case_id": escalation["case_id"],
        "queue": escalation["queue"],
        "status": "escalated",
        # Instructions travel back to the model in the tool result, so correct
        # behaviour is stated at the moment it is needed rather than relying on
        # the system prompt being remembered twelve turns later.
        "note": (
            "A human owns this case now. Tell the customer it is with a specialist "
            "and give the expected response time. Do not attempt further actions."
        ),
    }
