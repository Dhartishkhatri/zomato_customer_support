"""
===============================================================================
WRITE TOOLS                                                          [CORE]
===============================================================================

Thin wrappers over the backends. They are thin on purpose: all the judgement
lives in policy.py, and these are only ever reached after the gate said yes.

Do not add validation here - add it to the gate, where it is auditable and
testable in isolation. The one exception is PaymentsBackend, which guards its
own invariants as a last line of defence.
===============================================================================
"""

from __future__ import annotations

from typing import Any

from ..backends import Backends
from ..schemas import ConversationContext


def issue_refund(
    backends: Backends,
    ctx: ConversationContext,
    order_id: str,
    amount: int,
    reason: str,
) -> dict[str, Any]:
    """Move money. The only such path in the system."""
    refund = backends.payments.issue_refund(order_id, int(amount), reason)
    # Recording this is what lets the guardrail later verify that a reply
    # claiming "your refund is on its way" is actually backed by a real refund.
    ctx.refunds_issued.append(refund)
    return {
        "refund_id": refund["refund_id"],
        "order_id": refund["order_id"],
        "amount": refund["amount"],
        "currency": refund["currency"],
        "status": refund["status"],
        "destination": refund["destination"],
        "settles_in": refund["settles_in"],
    }


def update_ticket(
    backends: Backends,
    ctx: ConversationContext,
    ticket_id: str,
    status: str,
    summary: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    ticket = backends.crm.update_ticket(ticket_id, status, summary, tags)
    return {
        "ticket_id": ticket["ticket_id"],
        "status": ticket["status"],
        "tags": ticket["tags"],
        "updated_at": ticket["updated_at"],
    }


def send_email(
    backends: Backends,
    ctx: ConversationContext,
    customer_id: str,
    subject: str,
    body: str,
) -> dict[str, Any]:
    message = backends.crm.send_email(customer_id, subject, body)
    return {
        "message_id": message["message_id"],
        "to": message["to"],
        "subject": message["subject"],
        "sent_at": message["sent_at"],
    }
