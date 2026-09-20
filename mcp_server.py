"""
===============================================================================
MCP SERVER - Zomato support backends as Model Context Protocol tools
===============================================================================

WHAT THIS DOES
    Exposes the orders, tracking, CRM, policy KB and the two verification
    layers as MCP tools, so any MCP client (Claude Desktop, an IDE, another
    agent, or this project's own Zia pipeline) can use them over stdio.

WHY MCP IS A GOOD FIT HERE
    The backends were already written behind a narrow adapter interface
    precisely so they could be relocated. MCP is that relocation: the same
    operations, now reachable from outside this process, with schemas the
    client discovers at runtime.

WHAT IS DELIBERATELY NOT EXPOSED
    Every tool here is READ-ONLY or a DRY-RUN VERIFIER. Nothing issues a
    refund, cancels an order or sends an email.

    That is a security decision, not an oversight. MCP tools are callable by
    whatever client connects, which puts them outside this project's
    permission gate, its approval transports and its audit trail. Money-moving
    operations stay behind the gate in policy.py where those controls live.
    The verifier tools below let a client ASK whether an action would be
    permitted - which is the useful half - without being able to perform it.

RUN IT
    python mcp_server.py

REGISTER IT WITH CLAUDE DESKTOP
    Add to claude_desktop_config.json:

    {
      "mcpServers": {
        "zomato-support": {
          "command": "python",
          "args": ["C:/Users/Test/Desktop/customer_support_agent/mcp_server.py"]
        }
      }
    }
===============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.mcpserver import MCPServer

from zomato_support.backends import Backends
from zomato_support.backends.crm import INTERNAL_ONLY_FIELDS
from zomato_support.db import ensure_db
from zomato_support.pipeline.action_decisioning import verify_action
from zomato_support.pipeline.escalation_policy import (
    ALL_CODES,
    verify_escalation,
)

ensure_db()
backends = Backends.mock()

mcp = MCPServer(
    name="zomato-support",
    instructions=(
        "Read-only access to Zomato's support backends: orders, live delivery "
        "tracking, customer profiles and the support policy knowledge base, plus "
        "dry-run verifiers for actions and escalations. No tool here moves money "
        "or changes an order."
    ),
)


@mcp.tool(description="Fetch one order: status, items, totals, payment and refunds.")
def get_order(order_id: str) -> dict[str, Any]:
    try:
        order = backends.orders.get(order_id)
    except KeyError:
        return {"error": "not_found", "order_id": order_id}
    order.pop("payment_ref", None)  # Internal reference, not useful to a client.
    return order


@mcp.tool(description="List a customer's orders, most recent first.")
def list_orders(customer_id: str) -> list[dict[str, Any]]:
    return [
        {
            "order_id": o["order_id"],
            "restaurant": o["restaurant"],
            "status": o["status"],
            "total": o["total"],
            "placed_minutes_ago": o.get("placed_minutes_ago"),
            "hours_since_delivery": o.get("hours_since_delivery"),
        }
        for o in backends.orders.for_customer(customer_id)
    ]


@mcp.tool(
    description=(
        "Live delivery tracking for an in-flight order: rider first name, area, "
        "distance, ETA, whether it is late and whether the rider has stopped moving."
    )
)
def get_delivery_tracking(order_id: str) -> dict[str, Any]:
    tracking = backends.tracking.get(order_id)
    if tracking is None:
        return {"error": "no_tracking", "message": "Order is not out for delivery."}
    # Surname and phone number are withheld, exactly as in the chat pipeline.
    tracking = dict(tracking)
    tracking["partner_first_name"] = (tracking.pop("partner_name") or "").split()[0]
    tracking.pop("partner_phone", None)
    return tracking


@mcp.tool(
    description=(
        "Customer profile. Internal risk fields (trust score, refund history, "
        "agent notes) are never returned."
    )
)
def get_customer(customer_id: str) -> dict[str, Any]:
    try:
        customer = backends.crm.get_customer(customer_id)
    except KeyError:
        return {"error": "not_found", "customer_id": customer_id}
    return {k: v for k, v in customer.items() if k not in INTERNAL_ONLY_FIELDS}


@mcp.tool(
    description=(
        "Search Zomato's support policy knowledge base. Returns passages with "
        "citation ids for grounding an answer."
    )
)
def search_policy(query: str, limit: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "citation": hit["chunk_id"],
            "title": hit["title"],
            "text": hit["text"],
            "relevance": hit["score"],
        }
        for hit in backends.kb.search(query, k=max(1, min(limit, 5)))
    ]


@mcp.tool(
    description=(
        "DRY RUN. Check whether an action would be permitted on an order without "
        "performing it. Actions: CANCEL_ORDER, ADD_DELIVERY_INSTRUCTIONS, "
        "CHANGE_DELIVERY_ADDRESS, CONTACT_DELIVERY_PARTNER, REQUEST_REFUND. "
        "Returns eligible=true/false with the reason."
    )
)
def check_action_eligibility(
    action: str,
    order_id: str,
    customer_id: str,
    parameter: str | None = None,
) -> dict[str, Any]:
    decision = verify_action(backends, action.upper(), order_id, parameter, customer_id)
    return {
        "action": decision.action,
        "order_id": decision.order_id,
        "eligible": decision.eligible,
        "reason": decision.reject_reason or "Action is permitted.",
        "confirmation_prompt": decision.prompt if decision.eligible else None,
    }


@mcp.tool(
    description=(
        "DRY RUN. Check whether an escalation reason code is supported by real "
        "order data. This is the containment stabiliser: codes like "
        "DP_MOVEMENT_ISSUE are only upheld if the rider has genuinely stopped "
        "moving. Safety codes are always upheld."
    )
)
def check_escalation_reason(
    reason_code: str, order_id: str | None = None, customer_id: str = ""
) -> dict[str, Any]:
    order = None
    if order_id:
        try:
            order = backends.orders.get(order_id)
        except KeyError:
            pass
    return verify_escalation(backends, reason_code, order, customer_id).to_dict()


@mcp.tool(description="List every valid escalation reason code and its queue.")
def list_escalation_codes() -> dict[str, str]:
    return ALL_CODES


if __name__ == "__main__":
    mcp.run(transport="stdio")
