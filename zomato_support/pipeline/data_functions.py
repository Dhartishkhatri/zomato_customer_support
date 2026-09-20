"""
===============================================================================
STEP 2: TARGETED FETCH + JSON -> NATURAL LANGUAGE                    [CORE]
===============================================================================

WHAT THIS DOES
    Runs only the lookups the classifier asked for, then rewrites each JSON
    result as a short paragraph of plain English for the prompt.

WHY CONVERT JSON TO PROSE AT ALL
    Raw API JSON is a poor prompt ingredient. It is verbose, full of fields
    nobody asked for, and models drift into echoing its keys and quoting nulls
    at customers. Prose costs fewer tokens and reads as context rather than as
    something to repeat.

WHY THE CONVERSION IS TEMPLATED, NOT MODEL-GENERATED
    This is the layer where you CONTROL THE NARRATIVE. Templates decide what
    the model is even able to say:

      * The courier's surname and phone number never enter the prompt, so they
        cannot be leaked - the same "never tell it" principle used for the
        forensic scores.
      * Lateness is pre-computed into words ("running about 12 minutes late")
        so the model never does arithmetic on timestamps, which it does badly.
      * Internal risk fields are absent by construction.

    It is also free and instant, which matters inside a 10-second budget: an
    extra model call per lookup would be the slowest thing in the pipeline.
    `narrate_with_llm` exists for genuinely unstructured payloads, but nothing
    in the default path uses it.
===============================================================================
"""

from __future__ import annotations

from typing import Any

from .. import config
from ..backends import Backends
from ..llm import LLM, Task, text_of

RUPEE = config.CURRENCY_SYMBOL


def _money(amount: Any) -> str:
    return f"{RUPEE}{int(amount)}"


# ---------------------------------------------------------------------------
# One narrator per lookup. Each returns prose, or None when it has nothing.
# ---------------------------------------------------------------------------
def _order_status(order: dict[str, Any]) -> str:
    status = order["status"].replace("_", " ")
    line = f"Order {order['order_id']} from {order['restaurant']} is currently '{status}'."
    if order.get("hours_since_delivery") is not None:
        hours = order["hours_since_delivery"]
        when = f"{int(hours)} hours ago" if hours >= 1 else "less than an hour ago"
        line += f" It was delivered {when}."
        # Pre-computed so the model never has to reason about the window.
        if hours > config.REFUND_WINDOW_HOURS:
            line += (
                f" This is past the {config.REFUND_WINDOW_HOURS}-hour support window, "
                "so it cannot be settled in chat."
            )
        else:
            line += f" It is still inside the {config.REFUND_WINDOW_HOURS}-hour support window."
    elif order.get("placed_minutes_ago") is not None:
        line += f" It was placed {order['placed_minutes_ago']} minutes ago."
    return line


def _delivery_tracking(order: dict[str, Any], tracking: dict[str, Any] | None) -> str:
    if tracking is None:
        return (
            f"Order {order['order_id']} has no live tracking - it is not currently "
            "out for delivery."
        )
    # First name only: the surname and phone never enter the prompt.
    first_name = (tracking["partner_name"] or "Your rider").split()[0]
    parts = [
        f"{first_name} is delivering order {order['order_id']} by "
        f"{(tracking['vehicle'] or 'vehicle').lower()}, currently around "
        f"{tracking['current_area']}, about {tracking['distance_km']} km away.",
        f"The estimated arrival is {tracking['eta_minutes']} minutes.",
    ]
    if tracking["is_late"]:
        parts.append(
            f"This is running about {tracking['minutes_late']} minutes later than the "
            f"{tracking['promised_eta_min']}-minute estimate given at checkout."
        )
    else:
        parts.append("This is within the estimate given at checkout.")
    if tracking["is_stuck"]:
        parts.append(
            f"The rider has not moved for {tracking['last_moved_min_ago']} minutes, "
            "which suggests they are held up."
        )
    return " ".join(parts)


def _order_items(order: dict[str, Any]) -> str:
    lines = [f"{i['qty']}x {i['name']} ({_money(i['price'])})" for i in order["items"]]
    return (
        f"Order {order['order_id']} contains: {', '.join(lines)}. "
        f"The order total is {_money(order['total'])}."
    )


def _order_payment(order: dict[str, Any], refunded: int) -> str:
    line = (
        f"Order {order['order_id']} was paid by {order['payment_method']} for "
        f"{_money(order['total'])}."
    )
    if refunded:
        line += (
            f" {_money(refunded)} has already been refunded against it, so no further "
            "refund can be issued for this order."
        )
    else:
        line += f" No refund has been issued yet; up to {_money(order['total'])} is refundable."
    return line


def _customer_profile(customer: dict[str, Any]) -> str:
    tier = "a Zomato Gold member" if customer["gold_member"] else "a registered customer"
    return (
        f"{customer['name']} is {tier} in {customer['city']}, "
        f"with the account since {customer['member_since']}."
    )


def _delivery_address(order: dict[str, Any]) -> str:
    line = f"Order {order['order_id']} is being delivered to {order['delivery_address']}."
    instructions = order.get("delivery_instructions")
    line += (
        f" Current delivery instructions: \"{instructions}\"."
        if instructions
        else " There are no delivery instructions on this order."
    )
    return line


def _restaurant_info(order: dict[str, Any]) -> str:
    return (
        f"Order {order['order_id']} is from {order['restaurant']}"
        + (f", reachable on {order['restaurant_phone']}." if order.get("restaurant_phone") else ".")
    )


def _policy(backends: Backends, query: str) -> str:
    hits = backends.kb.search(query or "refund policy", k=2)
    if not hits:
        return ""
    return "Relevant support policy:\n" + "\n".join(
        f"- [{h['chunk_id']}] {h['text'].strip()}" for h in hits
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def fetch_and_narrate(
    backends: Backends,
    functions: list[str],
    order_id: str | None,
    customer_id: str,
    query: str = "",
) -> tuple[str, dict[str, Any]]:
    """Run the selected lookups and return (prose_for_prompt, raw_payloads).

    The prose goes to the model. The raw payloads go to the action pipeline,
    the escalation policy layer and the evaluation harness, all of which need
    exact values rather than sentences.
    """
    sections: list[str] = []
    raw: dict[str, Any] = {}

    order: dict[str, Any] | None = None
    if order_id:
        try:
            candidate = backends.orders.get(order_id)
            # Never surface another customer's order, whatever was requested.
            if candidate["customer_id"] == customer_id:
                order = candidate
                raw["order"] = order
        except KeyError:
            sections.append(f"No order with id {order_id} exists on this account.")

    needs_order = {
        "ORDER_STATUS", "DELIVERY_TRACKING", "ORDER_ITEMS",
        "ORDER_PAYMENT", "DELIVERY_ADDRESS", "RESTAURANT_INFO",
    }
    if order is None and any(f in needs_order for f in functions):
        sections.append(
            "The customer has not identified which order they mean, and it could not "
            "be inferred. Ask them which order this is about."
        )

    for name in functions:
        if name in needs_order and order is None:
            continue
        if name == "ORDER_STATUS":
            sections.append(_order_status(order))
        elif name == "DELIVERY_TRACKING":
            tracking = backends.tracking.get(order["order_id"])
            raw["tracking"] = tracking
            sections.append(_delivery_tracking(order, tracking))
        elif name == "ORDER_ITEMS":
            sections.append(_order_items(order))
        elif name == "ORDER_PAYMENT":
            refunded = backends.orders.refunded_total(order["order_id"])
            raw["refunded_total"] = refunded
            sections.append(_order_payment(order, refunded))
        elif name == "CUSTOMER_PROFILE":
            customer = backends.crm.get_customer(customer_id)
            raw["customer"] = customer
            sections.append(_customer_profile(customer))
        elif name == "DELIVERY_ADDRESS":
            sections.append(_delivery_address(order))
        elif name == "RESTAURANT_INFO":
            sections.append(_restaurant_info(order))
        elif name == "POLICY":
            text = _policy(backends, query)
            if text:
                raw["policy_query"] = query
                sections.append(text)

    return "\n\n".join(s for s in sections if s), raw


def narrate_with_llm(llm: LLM, label: str, payload: dict[str, Any]) -> str:
    """Escape hatch for payloads with no template.

    Deliberately unused by the default path: it adds a model call inside the
    latency budget and hands narrative control back to the model. Reach for it
    only when a new data source is too irregular to template.
    """
    return text_of(
        llm.complete(
            Task.NARRATE,
            system=(
                "Rewrite this API payload as two or three short factual sentences for "
                "a support agent's context. No preamble, no speculation, no field "
                "names, and never invent values that are not present."
            ),
            messages=[{"role": "user", "content": f"{label}:\n{payload}"}],
            max_tokens=250,
        )
    )
