"""
===============================================================================
ACTION DECISIONING - two-step, and the second step is the real one   [CORE]
===============================================================================

WHAT THIS DOES
    Decides whether this turn should offer the customer a concrete action -
    cancel the order, add delivery instructions, change the address, call the
    rider - and whether that action is actually permitted right now.

THE TWO STEPS, AND WHY BOTH EXIST
    STEP 1  CLASSIFY (LLM, big model)
            Multi-class: what action, if any, does this conversation call for?
            The model is good at reading "I'll be in a meeting, leave it at the
            door" as an add-instructions request. That is genuine intent
            understanding and it is what makes the bot feel conversational.

    STEP 2  VERIFY (code, no model)
            Check the proposal against real order state. An LLM asked whether
            an order can be cancelled will say yes to be helpful. The order
            being already out for delivery is a fact, and facts are checked
            here, in Python, against the database.

    Step 1 without step 2 is a bot that cheerfully promises to cancel orders
    that left the restaurant twenty minutes ago.

NOTHING HERE EXECUTES ANYTHING
    This module only ever PROPOSES. Execution happens after the customer
    confirms the popup in the UI, and then still passes through the permission
    gate in policy.py. Three layers: intent -> eligibility -> authorisation.

IT RUNS IN PARALLEL WITH THE REPLY
    zia.py fires this concurrently with response generation, so the
    verification costs wall-clock time only if it is slower than the reply.
    That is what keeps the two-step check affordable inside a 10s budget.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..backends import Backends
from ..llm import LLM, Task

ActionType = Literal[
    "CANCEL_ORDER",
    "ADD_DELIVERY_INSTRUCTIONS",
    "CHANGE_DELIVERY_ADDRESS",
    "CONTACT_DELIVERY_PARTNER",
    "REQUEST_REFUND",
    "NONE",
]

# Customer-facing copy for the confirmation popup, per action.
PROMPTS: dict[str, tuple[str, str]] = {
    "CANCEL_ORDER": ("Cancel this order?", "Cancel order"),
    "ADD_DELIVERY_INSTRUCTIONS": ("Add these delivery instructions?", "Add instructions"),
    "CHANGE_DELIVERY_ADDRESS": ("Update the delivery address?", "Update address"),
    "CONTACT_DELIVERY_PARTNER": ("Ask your rider to call you?", "Request a call"),
    "REQUEST_REFUND": ("Raise a refund request for this order?", "Request refund"),
}

_SYSTEM = """You identify which concrete ACTION a food-delivery support \
conversation is asking for, if any.

Actions:
- CANCEL_ORDER: the customer wants the order stopped or cancelled.
- ADD_DELIVERY_INSTRUCTIONS: they want the rider told something ("leave at the
  door", "call when you arrive", "gate code is 1234", "I'm in a meeting").
- CHANGE_DELIVERY_ADDRESS: they want it delivered somewhere else.
- CONTACT_DELIVERY_PARTNER: they want the rider to call or be called.
- REQUEST_REFUND: they want money back.
- NONE: they are asking a question, venting, or chatting. Most turns are NONE.

Rules:
- Only propose an action the customer actually wants. Asking "where is my
  order?" is NOT a cancellation request, even if the order is late.
- Complaining about lateness alone is NONE. Saying "forget it, cancel it" is
  CANCEL_ORDER.
- Extract the parameter where one applies: the instruction text, or the new
  address, verbatim from what the customer wrote.
- If you are unsure, choose NONE. A missed action costs one extra turn; a
  wrong one costs a cancelled dinner."""


class ActionProposal(BaseModel):
    action: ActionType
    order_id: str | None = None
    parameter: str | None = Field(
        default=None,
        description="Instruction text or new address, verbatim. Null otherwise.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(description="One short sentence.")


@dataclass
class ActionDecision:
    """The proposal plus the verdict of the eligibility check."""

    action: str
    order_id: str | None = None
    parameter: str | None = None
    confidence: float = 0.0
    rationale: str = ""
    eligible: bool = False
    reject_reason: str = ""
    prompt: str = ""
    button: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def should_offer(self) -> bool:
        return self.action != "NONE" and self.eligible

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "order_id": self.order_id,
            "parameter": self.parameter,
            "confidence": round(self.confidence, 2),
            "eligible": self.eligible,
            "reject_reason": self.reject_reason,
            "prompt": self.prompt,
            "button": self.button,
            "details": self.details,
        }


# Below this the proposal is treated as noise rather than intent.
MIN_CONFIDENCE = 0.6


def decide_action(
    llm: LLM,
    backends: Backends,
    history: list[dict[str, Any]],
    latest_message: str,
    order_id: str | None,
    customer_id: str,
) -> ActionDecision:
    """Step 1 (classify) then step 2 (verify). Never executes anything."""
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])

    try:
        response = llm.parse(
            Task.CLASSIFY,
            system=_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Conversation:\n{transcript or '(none)'}\n\n"
                        f"Latest customer message:\n{latest_message}\n\n"
                        f"Order under discussion: {order_id or 'unknown'}"
                    ),
                }
            ],
            output_format=ActionProposal,
            max_tokens=800,
        )
        proposal = response.parsed_output
    except Exception:
        # No action is always a safe default: the customer simply gets a reply.
        return ActionDecision(action="NONE", rationale="Action classifier unavailable.")

    if proposal.action == "NONE" or proposal.confidence < MIN_CONFIDENCE:
        return ActionDecision(
            action="NONE",
            confidence=proposal.confidence,
            rationale=proposal.rationale,
        )

    return verify_action(
        backends,
        proposal.action,
        proposal.order_id or order_id,
        proposal.parameter,
        customer_id,
        confidence=proposal.confidence,
        rationale=proposal.rationale,
    )


def verify_action(
    backends: Backends,
    action: str,
    order_id: str | None,
    parameter: str | None,
    customer_id: str,
    confidence: float = 1.0,
    rationale: str = "",
) -> ActionDecision:
    """STEP 2. Check a proposed action against real order state.

    Pure Python against the database - no model involved, because this is
    exactly the judgement an eager model gets wrong.
    """
    prompt, button = PROMPTS.get(action, ("Proceed?", "Confirm"))
    decision = ActionDecision(
        action=action, order_id=order_id, parameter=parameter,
        confidence=confidence, rationale=rationale, prompt=prompt, button=button,
    )

    if not order_id:
        decision.reject_reason = "No order identified for this action."
        return decision

    try:
        order = backends.orders.get(order_id)
    except KeyError:
        decision.reject_reason = f"Order {order_id} does not exist."
        return decision

    if order["customer_id"] != customer_id:
        decision.reject_reason = "Order belongs to a different customer."
        return decision

    status = order["status"]
    decision.details = {
        "order_id": order_id,
        "restaurant": order["restaurant"],
        "total": order["total"],
        "status": status,
    }

    # --- per-action eligibility ------------------------------------------
    if action == "CANCEL_ORDER":
        if status in ("delivered", "cancelled"):
            decision.reject_reason = f"Order is already {status}; it cannot be cancelled."
            return decision
        placed = order.get("placed_minutes_ago") or 0
        window = order.get("cancellable_until_min") or 15
        if status == "out_for_delivery":
            # The restaurant has cooked it and a rider is carrying it. Free
            # cancellation here is exactly what an eager LLM would promise.
            decision.reject_reason = (
                "Order is already with the rider, so it cannot be cancelled free of "
                "charge. Offer to raise the issue with support instead."
            )
            return decision
        if placed > window:
            decision.reject_reason = (
                f"The {window}-minute free-cancellation window has passed "
                f"({placed} minutes since it was placed)."
            )
            return decision
        decision.details["refund_on_cancel"] = order["total"]
        decision.prompt = (
            f"Cancel order {order_id} from {order['restaurant']}? "
            f"You'll be refunded the full {order['currency']} {order['total']}."
        )

    elif action in ("ADD_DELIVERY_INSTRUCTIONS", "CHANGE_DELIVERY_ADDRESS"):
        if status in ("delivered", "cancelled"):
            decision.reject_reason = f"Order is already {status}."
            return decision
        if not parameter:
            decision.reject_reason = "No instruction or address text was provided."
            return decision
        if action == "CHANGE_DELIVERY_ADDRESS" and status == "out_for_delivery":
            # Re-routing a rider mid-journey is a dispatch decision, not a
            # self-service one.
            decision.reject_reason = (
                "The rider is already en route, so the address cannot be changed "
                "automatically."
            )
            return decision
        decision.prompt = (
            f'Add "{parameter}" as the delivery instruction for this order?'
            if action == "ADD_DELIVERY_INSTRUCTIONS"
            else f'Change the delivery address to "{parameter}"?'
        )

    elif action == "CONTACT_DELIVERY_PARTNER":
        if status != "out_for_delivery":
            decision.reject_reason = "No rider is currently assigned to this order."
            return decision
        tracking = backends.tracking.get(order_id)
        if not tracking:
            decision.reject_reason = "No live tracking available for this order."
            return decision
        first_name = (tracking["partner_name"] or "your rider").split()[0]
        decision.prompt = f"Ask {first_name} to call you about this order?"

    elif action == "REQUEST_REFUND":
        if status != "delivered":
            decision.reject_reason = "Refunds apply to delivered orders only."
            return decision
        if backends.orders.refunded_total(order_id) > 0:
            decision.reject_reason = "This order has already been refunded."
            return decision
        decision.prompt = f"Raise a refund request for order {order_id}?"

    else:
        decision.reject_reason = f"Unknown action {action}."
        return decision

    decision.eligible = True
    return decision
