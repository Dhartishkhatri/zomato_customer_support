"""
===============================================================================
READ TOOLS - safe by construction, but they shape what the model knows [CORE]
===============================================================================

No side effects, so no permission gate. But they are NOT merely pass-throughs:
each one decides what the model is allowed to see, and that filtering is a
security control in its own right.

Two examples worth understanding:
  * get_customer strips trust scores and fraud notes. The gate reads them; the
    model never does, so it cannot leak or argue about them.
  * get_order refuses to confirm another customer's order even exists.
===============================================================================
"""

from __future__ import annotations

from typing import Any

from ..backends import Backends
from ..backends.crm import INTERNAL_ONLY_FIELDS
from ..schemas import ConversationContext


def kb_search(backends: Backends, ctx: ConversationContext, query: str) -> dict[str, Any]:
    """Search support policy. Results carry a citation id the agent must quote."""
    return {
        "query": query,
        "results": [
            {
                "citation": hit["chunk_id"],
                "title": hit["title"],
                "text": hit["text"],
                "relevance": hit["score"],
            }
            for hit in backends.kb.search(query, k=3)
        ],
    }


def get_order(backends: Backends, ctx: ConversationContext, order_id: str) -> dict[str, Any]:
    """Fetch one order belonging to THIS conversation's customer.

    Two things happen here that matter:

    1. An order belonging to someone else returns a plain "not found". We do
       not confirm or deny that another customer's order exists - that would
       turn this tool into an order-enumeration oracle.

    2. On success the order is recorded in ctx.orders_seen. That record is
       what rule R02 in the policy gate later requires before any refund. This
       line is the reason an invented order id cannot be refunded.
    """
    order = backends.orders.get(order_id)

    if order["customer_id"] != ctx.customer_id:
        return {"error": "not_found", "message": f"No order {order_id} on this customer's account."}

    ctx.orders_seen[order_id] = order
    already = backends.orders.refunded_total(order_id)
    # The delivery partner lives on the tracking record, not the order.
    tracking = backends.tracking.get(order_id)
    return {
        "order_id": order["order_id"],
        "restaurant": order["restaurant"],
        "status": order["status"],
        "delivered_at": order["delivered_at"],
        "hours_since_delivery": order["hours_since_delivery"],
        "total": order["total"],
        "currency": order["currency"],
        "payment_method": order["payment_method"],
        "items": order["items"],
        # First name only - policy forbids giving out a courier's surname.
        "delivery_partner_first_name": (
            tracking["partner_name"].split()[0] if tracking else None
        ),
        "eta_minutes": tracking["eta_minutes"] if tracking else None,
        "already_refunded": already,
        "refundable_balance": order["total"] - already,
    }


def get_customer(backends: Backends, ctx: ConversationContext) -> dict[str, Any]:
    """Profile of THIS conversation's customer.

    Takes no customer_id argument by design - the model cannot look anyone
    else up, so there is no lookup to abuse. Internal risk fields are stripped
    on the way out (see INTERNAL_ONLY_FIELDS).
    """
    customer = backends.crm.get_customer(ctx.customer_id)
    ctx.customer_record = customer  # The gate uses the FULL record.
    return {k: v for k, v in customer.items() if k not in INTERNAL_ONLY_FIELDS}
