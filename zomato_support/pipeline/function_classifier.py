"""
===============================================================================
STEP 1: FUNCTION CLASSIFICATION - decide what data this turn needs   [CORE]
===============================================================================

WHAT THIS DOES
    Reads the conversation and decides WHICH backend lookups are needed, before
    any of them run. "Where's my order?" needs live tracking, not the itemised
    bill. "Refund me for the missing coffee" needs the bill, not the courier's
    GPS position.

WHY NOT JUST FETCH EVERYTHING
    Three reasons, all of which matter at scale:

    1. TOKENS. Every field you fetch ends up in the prompt. Fetching the full
       order graph for a one-line ETA question can triple the prompt for no
       gain. Fewer tokens is directly less money and less latency.
    2. LATENCY. Response time scales with prompt size, and backend calls you
       skip are backend calls you do not wait for.
    3. ACCURACY. A model handed six sections of context it does not need is
       measurably more likely to answer from the wrong one.

WHY IT RUNS ON THE BIG MODEL
    A misclassification here derails the whole turn: the reply gets generated
    against the wrong data and no later stage can recover it. That makes this
    the single call most worth spending on, which is why it is routed to
    Task.CLASSIFY (Opus 5) while the reply itself runs on Haiku.

A NOTE ON NATIVE TOOL CALLING
    Zomato built this before native function calling was widely available.
    It is still the right shape here, because classifying first means you
    decide what to fetch in ONE round trip instead of an agentic loop's
    several - which is what makes a sub-10-second budget achievable.
===============================================================================
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..llm import LLM, Task

# The catalogue. Each entry maps to exactly one fetcher in data_functions.py.
FUNCTIONS: dict[str, str] = {
    "ORDER_STATUS": "Current status of an order (placed, preparing, out for delivery, delivered, cancelled) and when it was placed or delivered.",
    "DELIVERY_TRACKING": "Live courier position, distance, ETA, how late the order is, and whether the courier has stopped moving. Use for 'where is my order', 'how long', 'it's late'.",
    "ORDER_ITEMS": "The itemised contents of an order with per-item prices. Use for missing items, wrong items, and refund amounts.",
    "ORDER_PAYMENT": "Payment method, amount paid, and any refunds already issued against the order.",
    "CUSTOMER_PROFILE": "The customer's name, city, membership tier and tenure.",
    "DELIVERY_ADDRESS": "The delivery address and any standing delivery instructions on the order.",
    "RESTAURANT_INFO": "Restaurant name and contact number.",
    "POLICY": "Zomato support policy: refund eligibility, time windows, photo requirements, escalation rules.",
}

_SYSTEM = """You decide which backend data lookups a customer-support turn needs.

You are given the conversation so far. Return ONLY the lookups required to \
answer the latest message well. Fetching extra data costs money and latency and \
makes the reply worse, so do not select a lookup "just in case".

Available lookups:
{catalogue}

Guidance:
- "Where is my order / how long / it's late"  -> DELIVERY_TRACKING (+ ORDER_STATUS)
- "An item is missing / wrong / damaged"      -> ORDER_ITEMS (+ ORDER_PAYMENT if money is involved)
- "Cancel my order"                           -> ORDER_STATUS (+ ORDER_PAYMENT)
- "Change the address / add instructions"     -> DELIVERY_ADDRESS (+ ORDER_STATUS)
- "Can I get a refund / what's your policy"   -> POLICY (+ ORDER_PAYMENT)
- Greetings, thanks, or chit-chat             -> no lookups at all

Also extract the order the customer is referring to. If they name an order id, \
use it exactly. If they say "my order" or "my last order" without an id, set \
order_id to null and set refers_to_latest_order to true."""


class FunctionPlan(BaseModel):
    """What the classifier decided this turn needs."""

    functions: list[str] = Field(
        description="Lookup names from the catalogue. Empty list if none are needed."
    )
    order_id: str | None = Field(
        default=None, description="Order id the customer named, or null."
    )
    refers_to_latest_order: bool = Field(
        default=False,
        description="True if they mean 'my order' without naming one.",
    )
    intent: Literal[
        "track_order", "missing_or_wrong_item", "food_quality", "cancel_order",
        "change_delivery", "refund_status", "payment_issue", "policy_question",
        "delivery_partner_issue", "greeting", "other",
    ] = Field(description="Primary intent of the latest customer message.")
    reasoning: str = Field(description="One short sentence on why these lookups.")


def classify_functions(
    llm: LLM,
    history: list[dict[str, Any]],
    latest_message: str,
) -> FunctionPlan:
    """Decide which lookups this turn needs. Falls back safely on error."""
    catalogue = "\n".join(f"  {name}: {desc}" for name, desc in FUNCTIONS.items())
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-8:])

    try:
        response = llm.parse(
            Task.CLASSIFY,
            system=_SYSTEM.format(catalogue=catalogue),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{transcript or '(none)'}\n\n"
                        f"Latest customer message:\n{latest_message}"
                    ),
                }
            ],
            output_format=FunctionPlan,
            max_tokens=1024,
        )
        plan = response.parsed_output
    except Exception:
        # A classifier failure must not take the turn down. Fetch the safe
        # general set and let the reply be a little more generic than ideal.
        return FunctionPlan(
            functions=["ORDER_STATUS", "POLICY"],
            refers_to_latest_order=True,
            intent="other",
            reasoning="Classifier unavailable; fell back to a general lookup set.",
        )

    # Drop anything hallucinated outside the catalogue.
    plan.functions = [f for f in plan.functions if f in FUNCTIONS]
    return plan
